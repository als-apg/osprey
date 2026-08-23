"""The runtime target switch, end to end, on a pair of mock control systems.

The unit suites each pin one layer: the supervisor's process lifecycle
(``tests/mcp_server/test_switch_lifecycle.py``), the typed error frame
(``tests/mcp_server/test_error_envelope_ipc.py``), the executor's stamp
(``tests/runtime/test_executor_target_stamp.py``). What none of them can claim
is that the layers *compose* — that a tool call made through the real server
context lands on the process the switch just adopted, that a batch stays a
batch on the way there, and that a child dying underneath a call is refused
rather than papered over.

That is what this file is for, and every test in it drives the full parent side:
the real :class:`ControlSystemContext` on a switch-capable config, the real
``ConnectorHostManager``, real ``python -m osprey_connectors.ipc.host``
children, and the real MCP tools where a tool is the thing under test.

Two targets, no containers and no EPICS
---------------------------------------
The "two endpoints" are two mock connectors that answer the same channel with
different numbers (:mod:`tests.integration._mock_pair_connectors`). ``live``
names one of them by dotted path — ``resolve_target`` returns
``control_system.type`` verbatim — and ``va`` gets the other through a scratch
``sitecustomize`` on the children's ``PYTHONPATH`` that registers it under the
``virtual_accelerator`` name. Routing is therefore proven by *value*: the same
read returns 101.0 before the switch and 202.0 after it, and the retired
target's probe channel stops answering at all. Nothing here opens a socket to a
control system, so the suite runs in CI unchanged.

What is deliberately not re-tested here
---------------------------------------
The switch's own failure modes — a failed probe, a failed verification, the
drain deadline — belong to the supervisor and are pinned in
``test_switch_lifecycle.py``. The exception-fidelity test below overlaps
``test_error_envelope_ipc.py`` on purpose and only where the composition is
new: that suite drives a hand-built proxy, this one reaches the child through
``registry.control_system()``, which is the seam every tool actually uses.

Write-visibility isolation — that a value written on one target is not visible
on the other — is also absent here, and cannot be tested with this pair: these
connectors are stateless by construction, answering constants so that a read
identifies the process that served it. Proving isolation needs endpoints that
remember what was written to them, so it is carried by the scripted
two-endpoint demo and the virtual-accelerator e2e scenarios instead.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.mcp_server.control_system import error_handling, server_context, target_state
from osprey.mcp_server.control_system.connector_host_manager import (
    ConnectorHostManager,
    NoConnectorHostError,
)
from osprey.mcp_server.control_system.server_context import (
    ConnectorEntry,
    ControlSystemContext,
    MCPServerConfig,
)
from osprey.mcp_server.control_system.target_eligibility import derive_endpoints
from osprey.mcp_server.control_system.tools.channel_read import channel_read
from osprey.mcp_server.control_system.tools.control_target import control_target
from osprey.mcp_server.python_executor import executor as host_executor
from osprey_connectors.control_system.base import ChannelValue
from osprey_connectors.control_system.va_connector import fill_gateway_ports
from osprey_connectors.errors import ChannelLimitsViolationError
from osprey_connectors.factory import ConnectorFactory, isolated_connector_registries
from osprey_connectors.ipc import frames
from osprey_connectors.ipc.proxy import ConnectorHostProxy
from tests.integration._mock_pair_connectors import (
    BATCH_CHANNELS,
    LIMITS_CHANNEL,
    LIMITS_FIELDS,
    LIVE_PROBE,
    LIVE_SEED,
    LIVE_TYPE,
    SHARED_CHANNEL,
    SLOW_CHANNEL,
    VA_PROBE,
    VA_SEED,
    make_limits_violation,
)

# The two tool helpers are shared MCP-tool test plumbing rather than fixtures,
# so they are imported rather than restated: `get_tool_fn` unwraps FastMCP's
# FunctionTool, and `assert_raises_error` parses the envelope a ToolError
# carries. Importing the module applies none of its autouse fixtures.
from tests.mcp_server.conftest import assert_raises_error, get_tool_fn

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The repo root rides along so a child can import the pair connectors by their
#: dotted ``tests.`` path, exactly as the error-envelope suite does.
REPO_PATHS = (
    str(REPO_ROOT),
    str(REPO_ROOT / "src"),
    str(REPO_ROOT / "packages" / "osprey-connectors" / "src"),
)

#: Tight enough that a hang fails one test rather than the run.
SPAWN_TIMEOUT_S = 30.0
SETTLE_TIMEOUT_S = 15.0
CALL_TIMEOUT_S = 10.0

SITECUSTOMIZE = '''\
"""Register the pair's VA connector as this deployment's virtual accelerator.

``register_builtin_connectors()`` never replaces an existing registration, so a
child started with this directory on its PYTHONPATH builds the fixture
connector for target 'va' and runs the whole real path — resolver, factory,
connect() — with no EPICS anywhere.
"""

try:
    from osprey_connectors.factory import ConnectorFactory

    from tests.integration._mock_pair_connectors import VaConnector

    ConnectorFactory.register_control_system("virtual_accelerator", VaConnector)
except Exception:  # a child that cannot register it fails loudly in the test
    pass
'''


def raw_config() -> dict[str, Any]:
    """A switch-capable config with a servable connector block per target."""
    return {
        "control_system": {
            "type": LIVE_TYPE,
            # The limits violation below is raised by the connector, one layer
            # under the base class's writes_enabled guard: with writes off the
            # guard returns a blocked result and the raise is never reached.
            "writes_enabled": True,
            "connector": {
                LIVE_TYPE: {"probe_channel": LIVE_PROBE},
                "virtual_accelerator": {"probe_channel": VA_PROBE},
            },
        },
        "archiver": {"type": "mongodb_archiver"},
    }


# ------------------------------------------------------------------ fixtures


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory):
    """A scratch directory the children import their VA registration from."""
    directory = tmp_path_factory.mktemp("mock_pair_fixture")
    (directory / "sitecustomize.py").write_text(SITECUSTOMIZE, encoding="utf-8")
    return directory


@pytest.fixture(autouse=True)
def child_environment(fixture_dir, monkeypatch):
    """Children see the repo, the fixture registration, and no project config."""
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(fixture_dir), *REPO_PATHS]))
    monkeypatch.delenv("CONFIG_FILE", raising=False)


@pytest.fixture(autouse=True)
def state_root(tmp_path, monkeypatch):
    """Anchor the state file in tmp_path instead of a real deployment."""
    monkeypatch.setattr(target_state, "resolve_shared_data_root", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def config_file(tmp_path) -> Path:
    """The project config the children are pointed at.

    The child reads its write posture from this file, not from the init frame,
    so parent and child agree about the deployment only if both see it.
    """
    path = tmp_path / "config.yml"
    path.write_text(yaml.dump(raw_config()), encoding="utf-8")
    return path


@dataclass
class Pair:
    """The parent half of a running deployment: supervisor plus server context."""

    manager: ConnectorHostManager
    context: ControlSystemContext


@pytest.fixture
async def pair(config_file, monkeypatch):
    """A switch-capable server context, wired to a real supervisor.

    ``initialize()`` reads a deployment's config.yml off disk and builds the
    workspace singletons; everything this file exercises is downstream of that,
    so the fields it would have filled are filled here directly — the same
    shortcut ``test_switch_lifecycle.py`` takes.
    """
    manager = ConnectorHostManager(
        MCPServerConfig(raw=raw_config(), config_path=config_file),
        drain_timeout_s=1.0,
        probe_timeout_s=CALL_TIMEOUT_S,
        spawn_timeout_s=SPAWN_TIMEOUT_S,
        terminate_grace_s=2.0,
    )
    manager.reset_state()

    context = ControlSystemContext()
    context._config = manager._config
    context._connector_hosts = manager
    context._connectors["control_system"] = ConnectorEntry(
        config=manager._config.control_system, connector_type="control_system"
    )
    context._connectors["archiver"] = ConnectorEntry(
        config=manager._config.archiver, connector_type="archiver"
    )
    # The tools reach the context through the module singleton, so this is what
    # makes channel_read and control_target talk to *this* deployment.
    monkeypatch.setattr(server_context, "_registry", context)

    yield Pair(manager=manager, context=context)

    with contextlib.suppress(Exception):
        await manager.shutdown()


# ------------------------------------------------------------------- helpers


async def wait_for(predicate, timeout: float = SETTLE_TIMEOUT_S) -> bool:
    """Poll *predicate* on the event loop until it holds, or give up."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.05)
    return False


async def read_through_the_tool(*channels: str) -> dict[str, Any]:
    """Call the real ``channel_read`` tool and return its parsed response."""
    return json.loads(
        await get_tool_fn(channel_read)(channels=list(channels), include_metadata=False)
    )


def value_read(payload: dict[str, Any], channel: str) -> Any:
    """The value ``channel_read`` reported for one channel."""
    return payload["summary"]["readings"][channel]["value"]


async def envelope_for(exc: BaseException, tool_name: str = "channel_write") -> dict[str, Any]:
    """The agent-facing envelope ``connector_error_handler`` renders for *exc*."""
    with assert_raises_error() as captured:
        async with error_handling.connector_error_handler(tool_name):
            raise exc
    return captured["envelope"]


class FrameLog:
    """What crossed the pipe while it was being watched."""

    def __init__(self) -> None:
        self.requests: list[str] = []
        self.replies: list[Any] = []


@contextlib.contextmanager
def counted_frames():
    """Count the frames the parent encodes and decodes, through the codec itself.

    Both halves are counted at the one place every request and every reply must
    pass: ``frames.encode_request`` is what the proxy calls to put a request on
    the wire, and ``frames.decode_frame`` is what ``FrameReader.feed`` calls for
    each whole frame that comes back. Nothing about the proxy is stubbed, so a
    batch that was quietly split into N calls is visible as N request frames
    rather than as a slower test that still passes.
    """
    log = FrameLog()
    real_encode, real_decode = frames.encode_request, frames.decode_frame

    def encode_request(request_id, method, kwargs=None):
        log.requests.append(method)
        return real_encode(request_id, method, kwargs)

    def decode_frame(raw):
        frame = real_decode(raw)
        log.replies.append(frame)
        return frame

    frames.encode_request = encode_request
    frames.decode_frame = decode_frame
    try:
        yield log
    finally:
        frames.encode_request = real_encode
        frames.decode_frame = real_decode


# ------------------------------------------------------------------- routing


class TestRoutingFollowsTheActiveTarget:
    """A switch moves where every later tool call lands, and says so."""

    async def test_the_same_read_answers_with_the_target_that_is_active(self, pair):
        """The claim the whole feature rests on, made through the real tool.

        The value is the evidence: 101.0 is the live connector's constant and
        202.0 is the VA connector's, so a read that still answered 101.0 after
        the switch would be the server serving from a process the session has
        already left.
        """
        # Without this the reads below would be answered by an in-process
        # connector and the switch would move nothing they can see.
        assert pair.context.switch_capable is True

        before = await read_through_the_tool(SHARED_CHANNEL)
        assert value_read(before, SHARED_CHANNEL) == LIVE_SEED
        assert pair.manager.active_target() == "live"

        result = await pair.manager.switch("va")

        assert (result["target"], result["generation"], result["target_changed"]) == ("va", 1, True)
        after = await read_through_the_tool(SHARED_CHANNEL)
        assert value_read(after, SHARED_CHANNEL) == VA_SEED

    async def test_the_retired_targets_probe_channel_stops_answering(self, pair):
        """Stated negatively: the old target's channel goes with its process.

        A single shared channel could in principle be answered by a stale cache
        on this side. A channel only the retired connector ever served cannot
        be, so its refusal is what rules that out.
        """
        connector = await pair.context.control_system()
        assert isinstance(connector, ConnectorHostProxy)
        assert isinstance(
            await connector.read_channel(LIVE_PROBE, timeout=CALL_TIMEOUT_S), ChannelValue
        )

        await pair.manager.switch("va")

        connector = await pair.context.control_system()
        assert isinstance(
            await connector.read_channel(VA_PROBE, timeout=CALL_TIMEOUT_S), ChannelValue
        )
        with pytest.raises(ConnectionError):
            await connector.read_channel(LIVE_PROBE, timeout=CALL_TIMEOUT_S)

    async def test_the_roster_and_the_state_file_report_the_new_target(self, pair):
        """What the operator and every out-of-process reader are told.

        The roster is the agent's answer to "where am I", and the state file is
        what the prompt hook and the executor read. Both are asserted here
        because a switch that moved the reads but not the reporting would be a
        session that lies about which machine it is on.
        """
        await read_through_the_tool(SHARED_CHANNEL)

        await pair.manager.switch("va")

        roster = json.loads(await get_tool_fn(control_target)())
        assert roster["summary"]["target"] == "va"
        assert roster["summary"]["generation"] == 1
        assert roster["summary"]["baseline_target"] == "live"
        assert roster["summary"]["connector_host_alive"] is True
        assert roster["access_details"]["targets"]["va"]["active"] is True
        assert roster["access_details"]["targets"]["live"]["active"] is False

        child_pid = pair.manager.status()["child_pid"]
        assert isinstance(child_pid, int)
        record = target_state.read()
        assert record["target"] == "va"
        assert record["generation"] == 1
        assert record["children"] == [child_pid]


# ------------------------------------------------------------ error fidelity


class TestRefusalsCrossTheServingSeam:
    """A refusal raised in the child is the refusal the operator reads."""

    async def test_a_limits_violation_keeps_every_field_and_its_allowed_range(self, pair):
        """Field fidelity, from the child through the seam to the envelope.

        ``test_error_envelope_ipc.py`` pins the same fields against a
        hand-built proxy over a child it spawned itself. What is new here is the
        composition: the connector is obtained the way a tool obtains it, from
        the server context on a switch-capable deployment, and the exception
        travels the supervisor's own proxy — so a manager that handed back a
        connector wired differently would be caught by this and not by that.
        """
        connector = await pair.context.control_system()

        with pytest.raises(ChannelLimitsViolationError) as raised:
            await connector.write_channel(
                LIMITS_CHANNEL, LIMITS_FIELDS["attempted_value"], timeout=CALL_TIMEOUT_S
            )

        exc = raised.value
        # Identity, not isinstance: a look-alike would take a different branch
        # in the envelope's except clauses.
        assert type(exc) is ChannelLimitsViolationError
        assert {name: getattr(exc, name) for name in LIMITS_FIELDS} == LIMITS_FIELDS

        proxied = await envelope_for(exc)

        assert proxied == await envelope_for(make_limits_violation())
        assert proxied["error_type"] == "limits_violation"
        # The allowed range is what the operator is told to report, so it is
        # asserted as text rather than as the fields it was rendered from.
        assert proxied["error_message"] == (
            f"Channel limits violated during channel_write: {LIMITS_FIELDS['violation_reason']} "
            "(allowed range: [-25.0, 100.0])"
        )
        assert proxied["details"]["max_step"] == LIMITS_FIELDS["max_step"]
        assert proxied["details"]["current_value"] == LIMITS_FIELDS["current_value"]

    async def test_a_refusal_leaves_the_child_serving(self, pair):
        """A refused write is an answer, not a casualty."""
        connector = await pair.context.control_system()
        pid = pair.manager.status()["child_pid"]

        with pytest.raises(ChannelLimitsViolationError):
            await connector.write_channel(
                LIMITS_CHANNEL, LIMITS_FIELDS["attempted_value"], timeout=CALL_TIMEOUT_S
            )

        assert value_read(await read_through_the_tool(SHARED_CHANNEL), SHARED_CHANNEL) == LIVE_SEED
        assert pair.manager.status()["child_pid"] == pid


# ------------------------------------------------------------------ batching


class TestBatchedReadsCrossOnce:
    """N channels cost one round trip, not N.

    The claim is about the batched method, which is the one the boundary could
    have broken. The ``channel_read`` *tool* deliberately does something else:
    it issues N ``read_channel`` frames concurrently — one burst pipelined over
    the single pipe, matched back by request id, not N sequential round trips —
    because it needs each address's exception first-hand to report per-channel
    failures, and ``read_multiple_channels`` drops failures silently by
    contract. Both shapes are correct; only the batched one is asserted here.
    """

    async def test_a_batched_read_is_a_single_request_frame(self, pair):
        """The batch stays a batch across the process boundary.

        The child fans the addresses out internally through the connector's own
        ``read_multiple_channels``, exactly as an in-process connector would.
        What must not happen is the *parent* splitting the batch, which would
        turn one pipe round trip into six and make the boundary a cost the
        in-process path never had.
        """
        connector = await pair.context.control_system()

        with counted_frames() as log:
            readings = await connector.read_multiple_channels(
                list(BATCH_CHANNELS), timeout=CALL_TIMEOUT_S
            )

        assert log.requests == ["read_multiple_channels"]
        assert len(log.replies) == 1
        assert sorted(readings) == sorted(BATCH_CHANNELS)
        assert [readings[address].value for address in BATCH_CHANNELS] == [
            LIVE_SEED + index + 1 for index in range(len(BATCH_CHANNELS))
        ]

    async def test_the_same_channels_read_one_by_one_cost_one_frame_each(self, pair):
        """The instrument can count past one, so the assertion above means something."""
        connector = await pair.context.control_system()

        with counted_frames() as log:
            for address in BATCH_CHANNELS:
                await connector.read_channel(address, timeout=CALL_TIMEOUT_S)

        assert log.requests == ["read_channel"] * len(BATCH_CHANNELS)
        assert len(log.replies) == len(BATCH_CHANNELS)


# --------------------------------------------------------------- child death


class TestChildDeathIsRefusedThenRecovered:
    async def test_a_killed_child_fails_the_call_refuses_the_next_and_comes_back(
        self, pair, monkeypatch
    ):
        """The whole recovery loop, composed once, through the real tool.

        The invalidate seam is recorded rather than executed for the first two
        calls: it is what turns a lost child into a respawn, and letting it run
        immediately would hide the state this test exists to pin — a session
        that still knows its target, has no process serving it, and refuses
        every control-system call with that as the reason (FR-1). The real seam
        is then called explicitly, which is the recovery the error handler
        performs for a deployment nobody is watching.
        """
        await pair.context.control_system()
        invalidated: list[str] = []

        async def record(connector_name: str) -> None:
            invalidated.append(connector_name)

        monkeypatch.setattr(error_handling, "invalidate_active_connector", record)

        # A call already on the wire when the process is killed outright: no
        # error frame, no goodbye — the failure a fail-closed refusal is for.
        hung = asyncio.ensure_future(read_through_the_tool(SLOW_CHANNEL))
        await asyncio.sleep(0.3)
        assert not hung.done()
        os.kill(pair.manager.status()["child_pid"], signal.SIGKILL)

        with assert_raises_error(error_type="connection_error") as captured:
            await asyncio.wait_for(hung, SETTLE_TIMEOUT_S)
        assert "connector-host child" in captured["envelope"]["error_message"]

        # The session is still pointed where it was — a dead child is not a
        # switch — and the seam has no child to hand out.
        assert await wait_for(lambda: not pair.manager.has_child())
        with pytest.raises(NoConnectorHostError) as refusal:
            await pair.context.control_system()
        assert refusal.value.as_dict()["reason"] == "no_connector_host"
        assert refusal.value.target == "live"
        assert refusal.value.generation == pair.manager.active_generation()

        with assert_raises_error(error_type="connection_error") as captured:
            await read_through_the_tool(SHARED_CHANNEL)
        refused = captured["envelope"]["error_message"]
        assert "No connector-host child is serving target 'live'" in refused
        assert invalidated == ["control_system", "control_system"]

        await pair.context.invalidate_connector("control_system")

        assert pair.manager.has_child() is True
        # A respawn is not a switch: same target, same generation, new process.
        assert pair.manager.active_target() == "live"
        assert pair.manager.active_generation() == 0
        assert value_read(await read_through_the_tool(SHARED_CHANNEL), SHARED_CHANNEL) == LIVE_SEED


# ----------------------------------------------- CF-5: the sandbox's endpoint

#: A deployment whose two targets point at visibly different gateways, so a
#: sandbox that read the wrong connector block names the wrong host.
SANDBOX_LIVE_HOST = "live-gateway.example.org"
SANDBOX_VA_HOST = "va-gateway.example.org"

#: The VA gateway leaves its port unset, which is the shipped posture: the port
#: follows the service the project deploys.
SANDBOX_VA_PORT = 5099

SANDBOX_SECTION: dict[str, Any] = {
    "type": "mock",
    "connector": {
        "mock": {"response_delay_ms": 0},
        "epics": {
            "probe_channel": LIVE_PROBE,
            "gateways": {"read_only": {"address": SANDBOX_LIVE_HOST, "port": 5064}},
        },
        "virtual_accelerator": {
            "probe_channel": VA_PROBE,
            "gateways": {"read_only": {"address": SANDBOX_VA_HOST, "use_name_server": True}},
        },
    },
}

SANDBOX_CONFIG: dict[str, Any] = {
    "control_system": SANDBOX_SECTION,
    "services": {"virtual_accelerator": {"port": SANDBOX_VA_PORT}},
}


def _config_reader(section: dict[str, Any], va_port: int):
    """A ``get_config_value`` stand-in serving the two keys this half reads."""

    def get_config_value(path, default=None, config_path=None):
        if path == "control_system":
            return section
        if path == "services.virtual_accelerator.port":
            return va_port
        return default

    return get_config_value


def _write_state_record(target: str = "va", generation: int = 0) -> Path:
    """One state record describing a live server owned by this session."""
    pid = os.getpid()
    path = target_state.state_file_path(pid)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "target": target,
                "generation": generation,
                "server_pid": pid,
                "owner_ppid": os.getppid(),
                "targets": {},
                "children": [],
            }
        ),
        encoding="utf-8",
    )
    return path


class _CapturingConnector:
    """Records the type-specific config block the factory handed ``connect()``."""

    last_config: dict[str, Any] | None = None

    async def connect(self, config: dict[str, Any]) -> None:
        type(self).last_config = config

    async def disconnect(self) -> None:  # pragma: no cover - cleanup path
        return None


def _endpoint_of(gateway: dict[str, Any]) -> tuple[str, Any, str]:
    """``(host, port, mode)`` for one gateway block, as the child will install it."""
    return (
        str(gateway.get("address", "")),
        gateway.get("port"),
        "name_server" if gateway.get("use_name_server", False) else "addr_list",
    )


class TestSandboxEndpointMatchesTheDerivation:
    """CF-5's integration half: both halves of the stamp name one endpoint."""

    @pytest.fixture
    def sandbox(self, state_root, monkeypatch):
        """A stamped sandbox with the fake registry the factory will build from."""
        import osprey.runtime as runtime

        monkeypatch.setattr(
            "osprey_connectors.config.get_config_value",
            _config_reader(SANDBOX_SECTION, SANDBOX_VA_PORT),
        )
        runtime._runtime_connector = None
        runtime._limits_validator = None
        with isolated_connector_registries():
            for name in ("mock", "epics", "virtual_accelerator"):
                ConnectorFactory.register_control_system(name, _CapturingConnector)
            _CapturingConnector.last_config = None
            yield runtime
        runtime._runtime_connector = None
        runtime._limits_validator = None
        _CapturingConnector.last_config = None

    async def test_the_stamped_sandbox_builds_the_endpoint_the_va_derivation_names(
        self, sandbox, monkeypatch
    ):
        """The host stamps, the sandbox routes, and the two land on one gateway.

        The unit suite proves the sandbox reads the *virtual accelerator block*
        rather than the config's own type. The claim that only shows up in
        composition is the stronger one: the endpoint that block resolves to is
        the endpoint ``derive_endpoints`` — the function the roster reports and
        the switch verifies its child against — says the target has. Two
        answers to "where is the virtual accelerator", from two processes'
        worth of code, and they have to be the same answer.
        """
        _write_state_record(target="va", generation=2)
        env: dict[str, str] = {}

        assert host_executor._apply_target_stamp(env) == "va"
        assert env == {
            host_executor.ENV_CONTROL_TARGET: "va",
            host_executor.ENV_CONTROL_TARGET_GENERATION: "2",
            host_executor.ENV_CONTROL_TARGET_STATE_PID: str(os.getpid()),
        }
        for name, value in env.items():
            monkeypatch.setenv(name, value)

        derivation = derive_endpoints(SANDBOX_CONFIG, "va")
        expected = derivation.selected_endpoint()
        assert expected is not None

        await sandbox._get_connector()

        block = _CapturingConnector.last_config
        assert block is not None
        # The port is unset in config and filled from the deployed service, so
        # the block is put through the connector's own filler — the same helper
        # the derivation used — before the two endpoints are compared.
        gateway = fill_gateway_ports(block)["gateways"][derivation.selected_role]
        assert _endpoint_of(gateway) == (expected.host, expected.port, expected.mode)
        assert (expected.host, expected.port) == (SANDBOX_VA_HOST, SANDBOX_VA_PORT)
        # The live gateway is what a sandbox reading the wrong block would have
        # reached, so its absence is part of the claim.
        assert gateway["address"] != SANDBOX_LIVE_HOST

        # The stamp, the record and the pin agree, which is what lets this
        # sandbox write at all.
        sandbox._assert_target_pin()
