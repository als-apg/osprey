"""Two genuinely different connector-host targets on one machine, with no EPICS.

A deployment's ``live`` target resolves to whatever non-simulated connector its
config names, so pointing ``control_system.type`` at the mock connector's
dotted path gives a real, servable ``live``. ``va`` always resolves to the
``virtual_accelerator`` type, which is a registry name rather than a path — so
the children are launched with a scratch directory on their ``PYTHONPATH``
holding a ``sitecustomize`` that registers a mock variant under that name
before the child's own ``register_builtin_connectors()`` runs (which never
replaces an existing registration). The result is two genuinely different
targets, each with its own connector block and probe channel, neither of which
touches Channel Access.

That variant serves two channels with behaviour the tests need and a mock
cannot give them: :data:`REFUSE_CHANNEL` raises, and :data:`SLOW_CHANNEL`
blocks for far longer than any drain deadline. It also runs the EPICS gateway
selection — the same rule, reading the same per-type write posture, installing
the same environment variables — so a target whose block carries a ``gateways``
table exercises the real role-selection path without any Channel Access. The
same class is reached by dotted path for ``live``, which is how one config can
give the two targets different postures and have both children act on them.

The fixtures that build a manager over those targets live here rather than in
one of the modules that use them: a fixture a sibling module needs has to be
where pytest looks for it. ``tests/mcp_server/conftest.py`` re-exports them for
the directory.
"""

import asyncio
import contextlib
import os
from pathlib import Path

import pytest
import yaml

from osprey.mcp_server.control_system import target_state
from osprey.mcp_server.control_system.connector_host_manager import ConnectorHostManager
from osprey.mcp_server.control_system.server_context import MCPServerConfig
from osprey_connectors import control_context, posture_store
from osprey_connectors.types import EPICS
from tests._control_context_fixtures import write_control_context

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_PATHS = (str(REPO_ROOT / "src"), str(REPO_ROOT / "packages" / "osprey-connectors" / "src"))

#: The mock connector by dotted path: what lets a test *serve* ``live`` from a
#: real child on a machine with no Channel Access, which is what nearly every
#: test here does — hence this module's default.
SERVED_LIVE_TYPE = "osprey_connectors.control_system.mock_connector.MockConnector"
#: A Channel Access type, for the tests that need ``live`` to read as a real
#: machine rather than be served by one. Nothing is spawned from it.
CA_LIVE_TYPE = EPICS
LIVE_PROBE = "SR:BEAM:CURRENT"
VA_PROBE = "VA:BEAM:CURRENT"
REFUSE_CHANNEL = "FIXTURE:REFUSE"
SLOW_CHANNEL = "FIXTURE:SLOW"

#: Tight enough that a hang fails the test rather than the run.
SPAWN_TIMEOUT_S = 30.0
SETTLE_TIMEOUT_S = 15.0

#: The fixture connector by dotted path, so ``live`` selects and installs a CA
#: gateway exactly the way the EPICS connector does — without any EPICS.
GATEWAY_TYPE = "switch_fixture_connectors.FixtureConnector"
GATEWAY_HOST = "127.0.0.1"
READ_GATEWAY_PORT = 5064
#: Configured on the ``write_access`` row and served by nothing: the fixture
#: refuses every read while this port is the one installed in the environment.
DEAD_WRITE_PORT = 5555
#: The simulator's own gateway pair, for a deployment that arms writes on 'va'
#: alone. Nothing serves these either, but no probe treats them as dead — a
#: target armed on its own block has to be reachable through the write-capable
#: gateway it selects, or there would be nothing to verify.
VA_READ_GATEWAY_PORT = 5065
VA_WRITE_GATEWAY_PORT = 5066

#: The audit session id the narrowing tests stamp this process with.
POSTURE_SESSION = "switch-lifecycle-session"

FIXTURE_MODULE = '''\
"""A mock variant with the channels and the gateway selection the tests need.

``connect()`` selects a gateway role by exactly the rule EPICSConnector
applies — this connector's own per-type write posture, and a configured
``write_access`` row — and installs the same environment variables, so the
child's post-connect report and the parent's verification of it exercise the
real role-selection path.

Reads answer only while the installed port is not the dead write-gateway port,
which is what a configured-but-unserved ``write_access`` endpoint does to a
probe; :data:`REFUSE_CHANNEL` raises and :data:`SLOW_CHANNEL` blocks for far
longer than any drain deadline. Writes are refused unless the write-capable
gateway is the one this connector installed, which is what a real read-only CA
gateway does to a put.
"""

import asyncio
import os

from osprey_connectors.control_system.base import ChannelWriteResult, WriteOutcome
from osprey_connectors.control_system.mock_connector import MockConnector

REFUSE_CHANNEL = "FIXTURE:REFUSE"
SLOW_CHANNEL = "FIXTURE:SLOW"
SLOW_SECONDS = 120.0
DEAD_WRITE_PORT = "5555"
WRITE_ROLE = "write_access"


class FixtureConnector(MockConnector):
    #: The role connect() actually installed, or None when it configured no
    #: gateway at all.
    _gateway_role = None

    async def connect(self, config):
        await super().connect(config)
        gateways = config.get("gateways") or {}
        write_gateway = gateways.get(WRITE_ROLE) or {}
        try:
            armed = self._writes_enabled
        except Exception:  # a child with no project config is simply unarmed
            armed = False
        if armed and write_gateway:
            self._gateway_role = WRITE_ROLE
            selected = write_gateway
        else:
            selected = gateways.get("read_only") or {}
            self._gateway_role = "read_only" if selected else None
        if selected:
            os.environ["EPICS_CA_ADDR_LIST"] = str(selected.get("address", ""))
            os.environ["EPICS_CA_SERVER_PORT"] = str(selected.get("port", 5064))
            os.environ.pop("EPICS_CA_NAME_SERVERS", None)
            self._epics_configured = True

    async def read_channel(self, channel_address, timeout=None):
        if os.environ.get("EPICS_CA_SERVER_PORT") == DEAD_WRITE_PORT:
            raise TimeoutError(
                f"probe read of {channel_address!r} timed out: nothing serves this gateway"
            )
        if channel_address == REFUSE_CHANNEL:
            raise ConnectionError(f"the fixture connector refuses {channel_address}")
        if channel_address == SLOW_CHANNEL:
            await asyncio.sleep(SLOW_SECONDS)
        return await super().read_channel(channel_address, timeout=timeout)

    async def write_channel(self, channel_address, value, timeout=None, **kwargs):
        if self._gateway_role != WRITE_ROLE:
            return ChannelWriteResult(
                channel_address=channel_address,
                value_written=value,
                outcome=WriteOutcome.REFUSED,
                refusal_reason="CONTROL_SYSTEM_REFUSED",
                error_message="the read-only gateway refused the write",
            )
        return await super().write_channel(channel_address, value, timeout=timeout, **kwargs)
'''

SITECUSTOMIZE = '''\
"""Register the fixture connector as this deployment's virtual accelerator.

``register_builtin_connectors()`` never replaces an existing registration, so a
child started with this directory on its PYTHONPATH builds the fixture
connector for target 'va' and runs the whole real path — resolver, factory,
connect() — with no EPICS anywhere.
"""

try:
    from switch_fixture_connectors import FixtureConnector

    from osprey_connectors.factory import ConnectorFactory

    ConnectorFactory.register_control_system("virtual_accelerator", FixtureConnector)
except Exception:  # a child that cannot register it fails loudly in the test
    pass
'''


def raw_config(
    *,
    live_probe=LIVE_PROBE,
    va_probe=VA_PROBE,
    drain_timeout_s=None,
    live_type=SERVED_LIVE_TYPE,
):
    """A config with a servable block for each target."""
    live_block = {"response_delay_ms": 1, "noise_level": 0.0}
    va_block = {"response_delay_ms": 1, "noise_level": 0.0}
    if live_probe:
        live_block["probe_channel"] = live_probe
    if va_probe:
        va_block["probe_channel"] = va_probe
    control_system = {
        "type": live_type,
        "writes_enabled": False,
        "connector": {live_type: live_block, "virtual_accelerator": va_block},
    }
    if drain_timeout_s is not None:
        control_system["target_switch"] = {"drain_timeout_s": drain_timeout_s}
    return {"control_system": control_system, "archiver": {"type": "mongodb_archiver"}}


def gateway_config(
    *,
    writes_enabled=True,
    read_gateway=True,
    read_port=READ_GATEWAY_PORT,
    live_writes_enabled=None,
    va_writes_enabled=None,
    va_gateways=False,
):
    """A config whose ``live`` target routes through configured CA gateways.

    The ``write_access`` row always points at :data:`DEAD_WRITE_PORT`, which
    nothing serves — the posture issue #718 is about: a write-capable gateway
    that is configured (so the role is selected) but not actually running.

    Write posture is per connector type, so each target's own block can carry
    it: *live_writes_enabled* and *va_writes_enabled* write ``writes_enabled``
    into that block, and ``None`` leaves the key out — which is what makes the
    target inherit the deployment-wide *writes_enabled*. *va_gateways* gives
    the simulator a gateway pair of its own, so a ``va`` armed on its own block
    has a write-capable gateway to select.
    """
    gateways = {"write_access": {"address": GATEWAY_HOST, "port": DEAD_WRITE_PORT}}
    if read_gateway:
        gateways["read_only"] = {"address": GATEWAY_HOST, "port": read_port}
    live_block = {
        "response_delay_ms": 1,
        "noise_level": 0.0,
        "probe_channel": LIVE_PROBE,
        "gateways": gateways,
    }
    va_block = {"response_delay_ms": 1, "noise_level": 0.0, "probe_channel": VA_PROBE}
    if va_gateways:
        va_block["gateways"] = {
            "read_only": {"address": GATEWAY_HOST, "port": VA_READ_GATEWAY_PORT},
            "write_access": {"address": GATEWAY_HOST, "port": VA_WRITE_GATEWAY_PORT},
        }
    if live_writes_enabled is not None:
        live_block["writes_enabled"] = live_writes_enabled
    if va_writes_enabled is not None:
        va_block["writes_enabled"] = va_writes_enabled
    return {
        "control_system": {
            "type": GATEWAY_TYPE,
            "writes_enabled": writes_enabled,
            "connector": {GATEWAY_TYPE: live_block, "virtual_accelerator": va_block},
        },
        "archiver": {"type": "mongodb_archiver"},
    }


def project_config(tmp_path, control_system):
    """Write the ``config.yml`` a child reads its own write posture from.

    The child reads posture from the ``CONFIG_FILE`` the parent hands it, not
    from the init payload — so a write-armed child needs a real file saying so,
    saying it the same way the parent's raw config does.
    """
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump({"control_system": control_system}), encoding="utf-8")
    return path


# ------------------------------------------------------------------ fixtures


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory):
    """A scratch directory the children import their VA connector from."""
    directory = tmp_path_factory.mktemp("switch_fixture")
    (directory / "switch_fixture_connectors.py").write_text(FIXTURE_MODULE, encoding="utf-8")
    (directory / "sitecustomize.py").write_text(SITECUSTOMIZE, encoding="utf-8")
    return directory


@pytest.fixture
def child_environment(fixture_dir, monkeypatch):
    """Children see the repo, the fixture connector, and no project config."""
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(fixture_dir), *REPO_PATHS]))
    monkeypatch.delenv("CONFIG_FILE", raising=False)


@pytest.fixture
def state_root(tmp_path, monkeypatch):
    """Anchor the deployment's agent data in tmp_path instead of a real one.

    Both anchors are set. The per-server reports resolve through
    ``target_state``'s own root helper; the control-context record a served
    context claims resolves through the ``OSPREY_AGENT_DATA_ROOT`` stamp.
    Anchoring only one of them writes the record into whatever checkout the
    tests are running from.
    """
    monkeypatch.setattr(target_state, "resolve_shared_data_root", lambda: tmp_path)
    monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(tmp_path))
    control_context.invalidate_cache()
    yield tmp_path
    control_context.invalidate_cache()


@pytest.fixture
def live_type(request):
    """The connector type the deployment under test names as its ``live`` target.

    Defaults to :data:`SERVED_LIVE_TYPE`, because nearly every test here wants a
    ``live`` target a real child can actually serve. Eligibility reads the
    connector type, so a test that needs ``live`` to read as a real machine —
    an away-switch, a Channel Access rung — must name a Channel Access type
    instead. Ask for one by parametrising this fixture::

        @pytest.mark.parametrize("live_type", [CA_LIVE_TYPE], indirect=True)

    rather than by adding a second config helper beside ``raw_config``.
    """
    return getattr(request, "param", SERVED_LIVE_TYPE)


@pytest.fixture
async def make_manager(state_root, live_type):
    """Managers whose children are all reaped when the test ends."""
    created = []

    def factory(raw=None, config_path=None, **overrides):
        options = {
            "drain_timeout_s": 1.0,
            "probe_timeout_s": 10.0,
            "spawn_timeout_s": SPAWN_TIMEOUT_S,
            "terminate_grace_s": 2.0,
        }
        options.update(overrides)
        manager = ConnectorHostManager(
            MCPServerConfig(
                raw=raw if raw is not None else raw_config(live_type=live_type),
                config_path=config_path,
            ),
            **options,
        )
        manager.spawned = []
        original_spawn = manager._spawn

        async def recording_spawn(target):
            process = await original_spawn(target)
            manager.spawned.append(process)
            return process

        manager._spawn = recording_spawn
        manager.reset_state()
        created.append(manager)
        return manager

    yield factory

    for manager in created:
        with contextlib.suppress(Exception):
            await manager.shutdown()
        for process in manager.spawned:
            if process.returncode is None:  # pragma: no cover - teardown safety net
                with contextlib.suppress(ProcessLookupError, OSError):
                    process.kill()
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(process.wait(), SETTLE_TIMEOUT_S)


@pytest.fixture
def posture_root(tmp_path, monkeypatch):
    """A scratch agent-data root this process and its children are stamped for.

    The stamp goes into the real environment rather than a patched lookup
    because the connector-host children have to read the same record the
    parent does.
    """
    root = tmp_path / "agent_data"
    monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(root))
    monkeypatch.setenv("OSPREY_POSTURE_SESSION", POSTURE_SESSION)
    monkeypatch.delenv("OSPREY_EXECUTION_MODE", raising=False)
    posture_store.invalidate_cache()
    yield root
    posture_store.invalidate_cache()


# ------------------------------------------------------------------- helpers


async def started_on(factory, target, **overrides):
    """A manager with a live child on *target*."""
    manager = factory(**overrides)
    await manager.start(target)
    assert manager.has_child()
    return manager


def narrow(root, *targets):
    """Record the operator's narrowing of *targets* on this deployment."""
    write_control_context(root, posture=dict.fromkeys(targets, posture_store.POSTURE_SANDBOX))
    posture_store.invalidate_cache()
