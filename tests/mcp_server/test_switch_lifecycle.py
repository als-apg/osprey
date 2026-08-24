"""The connector-host supervisor and the spawn-then-swap target switch.

Every test that is about the *mechanics* of a switch spawns real
``python -m osprey_connectors.ipc.host`` children and swaps between them, for
the same reason the child's own tests do: the things worth pinning here — that
the old process actually dies, that a hung read on it really does end, that a
failed candidate leaves nothing behind — only exist between processes.

Two targets on one machine, with no EPICS
-----------------------------------------
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
blocks for far longer than any drain deadline.
"""

import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from osprey.mcp_server.control_system import connector_host_manager, target_state
from osprey.mcp_server.control_system.connector_host_manager import (
    DEFAULT_DRAIN_TIMEOUT_S,
    ConnectorHostManager,
    NoConnectorHostError,
    SwitchError,
    baseline_target,
    kill_orphans,
    looks_like_a_connector_host,
    switch_capable,
    target_display_metadata,
)
from osprey.mcp_server.control_system.server_context import (
    ConnectorEntry,
    ControlSystemContext,
    MCPServerConfig,
)
from osprey.mcp_server.control_system.target_eligibility import (
    REASON_PROBE_CHANNEL_MISSING,
    REASON_TARGET_UNRESOLVABLE,
    Endpoint,
    TargetDerivation,
)
from osprey_connectors.control_system.base import ChannelValue
from osprey_connectors.factory import ConnectorFactory, isolated_connector_registries
from osprey_connectors.ipc.proxy import ConnectorHostProxy
from tests.fixtures.control_context import context_for

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_PATHS = (str(REPO_ROOT / "src"), str(REPO_ROOT / "packages" / "osprey-connectors" / "src"))

#: The mock connector by dotted path, so ``live`` resolves to something real.
LIVE_TYPE = "osprey_connectors.control_system.mock_connector.MockConnector"
LIVE_PROBE = "SR:BEAM:CURRENT"
VA_PROBE = "VA:BEAM:CURRENT"
REFUSE_CHANNEL = "FIXTURE:REFUSE"
SLOW_CHANNEL = "FIXTURE:SLOW"

#: Tight enough that a hang fails the test rather than the run.
SPAWN_TIMEOUT_S = 30.0
SETTLE_TIMEOUT_S = 15.0

FIXTURE_MODULE = '''\
"""A mock variant with two channels the switch tests need."""

import asyncio

from osprey_connectors.control_system.mock_connector import MockConnector

REFUSE_CHANNEL = "FIXTURE:REFUSE"
SLOW_CHANNEL = "FIXTURE:SLOW"
SLOW_SECONDS = 120.0


class FixtureConnector(MockConnector):
    async def read_channel(self, channel_address, timeout=None):
        if channel_address == REFUSE_CHANNEL:
            raise ConnectionError(f"the fixture connector refuses {channel_address}")
        if channel_address == SLOW_CHANNEL:
            await asyncio.sleep(SLOW_SECONDS)
        return await super().read_channel(channel_address, timeout=timeout)
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
    live_type=LIVE_TYPE,
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


# ------------------------------------------------------------------ fixtures


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory):
    """A scratch directory the children import their VA connector from."""
    directory = tmp_path_factory.mktemp("switch_fixture")
    (directory / "switch_fixture_connectors.py").write_text(FIXTURE_MODULE, encoding="utf-8")
    (directory / "sitecustomize.py").write_text(SITECUSTOMIZE, encoding="utf-8")
    return directory


@pytest.fixture(autouse=True)
def child_environment(fixture_dir, monkeypatch):
    """Children see the repo, the fixture connector, and no project config."""
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(fixture_dir), *REPO_PATHS]))
    monkeypatch.delenv("CONFIG_FILE", raising=False)


@pytest.fixture(autouse=True)
def state_root(tmp_path, monkeypatch):
    """Anchor the state file in tmp_path instead of a real deployment."""
    monkeypatch.setattr(target_state, "resolve_shared_data_root", lambda: tmp_path)
    return tmp_path


@pytest.fixture
async def make_manager(state_root):
    """Managers whose children are all reaped when the test ends."""
    created = []

    def factory(raw=None, **overrides):
        options = {
            "drain_timeout_s": 1.0,
            "probe_timeout_s": 10.0,
            "spawn_timeout_s": SPAWN_TIMEOUT_S,
            "terminate_grace_s": 2.0,
        }
        options.update(overrides)
        manager = ConnectorHostManager(
            MCPServerConfig(raw=raw if raw is not None else raw_config(), config_path=None),
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


async def started_on(factory, target, **overrides):
    """A manager with a live child on *target*."""
    manager = factory(**overrides)
    await manager.start(target)
    assert manager.has_child()
    return manager


class _NeverBuilt:
    """Registered under the deployment's own connector type as a tripwire.

    On a switch-capable deployment the server must never construct an
    in-process control-system connector: doing so would load a control-system
    client into the process that is supposed to hold none, pinned to whatever
    target the config described at start. Constructing this is that bug.
    """

    def __init__(self):
        raise AssertionError("the in-process connector was built on a switch-capable deployment")


class _FakeArchiver:
    """An archiver that connects without touching a database."""

    def __init__(self):
        self.config = None

    async def connect(self, config):
        self.config = config

    async def disconnect(self):
        return None


async def wait_for(predicate, timeout=SETTLE_TIMEOUT_S):
    """Poll *predicate* on the event loop until it holds, or fail the test."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.05)
    return False


# ------------------------------------------------------------------- success


class TestSuccessfulSwitch:
    async def test_switching_targets_bumps_the_generation_exactly_once(self, make_manager):
        manager = await started_on(make_manager, "live")
        assert manager.active_generation() == 0

        result = await manager.switch("va")

        assert result["target"] == "va"
        assert result["previous_target"] == "live"
        assert result["target_changed"] is True
        assert result["generation"] == 1
        assert manager.active_target() == "va"
        assert manager.active_generation() == 1

    async def test_the_switch_publishes_the_new_target_and_child_to_the_state_file(
        self, make_manager
    ):
        manager = await started_on(make_manager, "live")

        result = await manager.switch("va")

        record = target_state.read()
        assert record["target"] == "va"
        assert record["generation"] == 1
        assert record["children"] == [result["child_pid"]]
        # Display metadata written at start survives the switch.
        assert record["targets"]["va"]["label"] == "virtual accelerator (simulation)"

    async def test_the_previous_child_process_actually_exits(self, make_manager):
        manager = await started_on(make_manager, "live")
        previous = manager.spawned[0]
        assert previous.returncode is None

        await manager.switch("va")

        assert await wait_for(lambda: previous.returncode is not None), (
            "the child for the previous target was still running after the switch"
        )
        assert manager.status()["child_pid"] == manager.spawned[1].pid

    async def test_the_new_child_serves_reads_on_its_own_target(self, make_manager):
        manager = await started_on(make_manager, "live")

        await manager.switch("va")

        value = await manager.active_proxy().read_channel(VA_PROBE, timeout=10.0)
        assert isinstance(value, ChannelValue)
        # The refusing channel is the fixture connector's, so this really is the
        # VA child answering and not the mock the live target left behind.
        with pytest.raises(ConnectionError):
            await manager.active_proxy().read_channel(REFUSE_CHANNEL, timeout=10.0)

    async def test_the_result_names_the_type_and_the_probe_channel_it_proved(self, make_manager):
        manager = await started_on(make_manager, "live")

        result = await manager.switch("va")

        assert result["connector_type"] == "virtual_accelerator"
        assert result["probe_channel"] == VA_PROBE
        assert result["previous_drained"] is True
        assert result["drain_timeout_s"] == 1.0

    async def test_a_switch_whose_state_file_cannot_be_written_still_succeeded(
        self, make_manager, monkeypatch
    ):
        """The swap is irreversible by the time the file is written.

        Child A is dead and every tool call now lands on the new target, so an
        unwritable state directory must not be reported to the caller as a
        switch that did not happen — it is a copy that could not be updated.
        """
        manager = await started_on(make_manager, "live")

        def unwritable(*args, **kwargs):
            raise OSError("read-only file system")

        monkeypatch.setattr(target_state, "publish_switch", unwritable)

        result = await manager.switch("va")

        assert result["target"] == "va"
        assert result["generation"] == 1
        assert manager.active_target() == "va"
        assert manager.active_generation() == 1
        assert isinstance(
            await manager.active_proxy().read_channel(VA_PROBE, timeout=10.0), ChannelValue
        )

    async def test_a_round_trip_bumps_the_generation_each_way(self, make_manager):
        manager = await started_on(make_manager, "live")

        await manager.switch("va")
        result = await manager.switch("live")

        assert result["generation"] == 2
        assert manager.active_target() == "live"
        assert target_state.read()["generation"] == 2


# ------------------------------------------------------------ failed switches


class TestFailedSwitchLeavesThePreviousTargetActive:
    async def test_a_failed_probe_kills_the_candidate_and_keeps_the_previous_child(
        self, make_manager
    ):
        manager = await started_on(make_manager, "live", raw=raw_config(va_probe=REFUSE_CHANNEL))

        with pytest.raises(SwitchError) as raised:
            await manager.switch("va")

        assert raised.value.stage == "probe"
        assert raised.value.reason == connector_host_manager.REASON_PROBE_FAILED
        assert REFUSE_CHANNEL in raised.value.detail

        # The candidate is gone, and nothing about the session moved.
        candidate = manager.spawned[1]
        assert await wait_for(lambda: candidate.returncode is not None), (
            "the candidate child outlived the switch that failed to adopt it"
        )
        assert manager.active_target() == "live"
        assert manager.active_generation() == 0
        assert manager.status()["child_pid"] == manager.spawned[0].pid

        # And the previous child is still serving, not merely still running.
        value = await manager.active_proxy().read_channel(LIVE_PROBE, timeout=10.0)
        assert isinstance(value, ChannelValue)

    async def test_a_probe_that_times_out_names_the_channel_in_the_refusal(self, make_manager):
        """A refusal an operator can act on names what would not answer.

        The child bounds the probe with ``asyncio.wait_for``, which raises a
        bare TimeoutError; without a reason attached at that bound the switch
        refusal renders as a sentence ending in ": ." — technically a refusal,
        practically useless.
        """
        manager = await started_on(
            make_manager,
            "live",
            raw=raw_config(va_probe=SLOW_CHANNEL),
            probe_timeout_s=0.5,
        )

        with pytest.raises(SwitchError) as raised:
            await manager.switch("va")

        detail = raised.value.detail
        assert raised.value.stage == "probe"
        assert SLOW_CHANNEL in detail
        assert "timed out" in detail
        assert not detail.rstrip().endswith(": .")
        assert manager.active_target() == "live"
        assert manager.active_generation() == 0

    async def test_a_failed_verification_aborts_with_the_field_that_disagreed(
        self, make_manager, monkeypatch
    ):
        manager = await started_on(make_manager, "live")
        real_derive = connector_host_manager.derive_endpoints

        def derive(config, target, **kwargs):
            if target != "va":
                return real_derive(config, target, **kwargs)
            # A derivation that says the child will configure a gateway. The
            # child configures none, which is exactly the failure mode
            # verification exists for: a child pointed somewhere else.
            return TargetDerivation(
                target="va",
                connector_type="virtual_accelerator",
                endpoints={
                    "read_only": Endpoint(host="gw.example.org", port=5064, mode="addr_list")
                },
                selected_role="read_only",
            )

        monkeypatch.setattr(connector_host_manager, "derive_endpoints", derive)

        with pytest.raises(SwitchError) as raised:
            await manager.switch("va")

        assert raised.value.stage == "verify"
        assert raised.value.reason == connector_host_manager.REASON_VERIFICATION_FAILED
        assert raised.value.verification.field == "_epics_configured"

        candidate = manager.spawned[1]
        assert await wait_for(lambda: candidate.returncode is not None)
        assert manager.active_target() == "live"
        assert manager.active_generation() == 0
        assert isinstance(
            await manager.active_proxy().read_channel(LIVE_PROBE, timeout=10.0), ChannelValue
        )

    async def test_a_target_without_a_probe_channel_is_refused_before_any_spawn(self, make_manager):
        manager = await started_on(make_manager, "live", raw=raw_config(va_probe=None))
        spawned_before = len(manager.spawned)

        with pytest.raises(SwitchError) as raised:
            await manager.switch("va")

        assert raised.value.stage == "probe_channel"
        assert raised.value.reason == REASON_PROBE_CHANNEL_MISSING
        assert "probe_channel" in raised.value.detail
        assert len(manager.spawned) == spawned_before
        assert manager.active_target() == "live"

    async def test_an_underivable_target_is_refused_before_any_spawn(self, make_manager):
        # A virtual-accelerator baseline that never named a real machine: there
        # is no 'live' to resolve, and nothing is guessed.
        config = {
            "control_system": {
                "type": "virtual_accelerator",
                "connector": {"virtual_accelerator": {"probe_channel": VA_PROBE}},
            }
        }
        manager = make_manager(raw=config)

        with pytest.raises(SwitchError) as raised:
            await manager.switch("live")

        assert raised.value.stage == "target"
        assert raised.value.reason == REASON_TARGET_UNRESOLVABLE
        assert manager.spawned == []

    async def test_an_unknown_target_name_is_refused(self, make_manager):
        manager = make_manager()

        with pytest.raises(SwitchError) as raised:
            await manager.switch("elsewhere")

        assert raised.value.reason == REASON_TARGET_UNRESOLVABLE
        assert "elsewhere" in raised.value.detail
        assert manager.spawned == []


# ----------------------------------------------------------------- draining


class TestDraining:
    async def test_the_drain_deadline_kills_the_child_and_names_the_switch(self, make_manager):
        drain = 0.5
        manager = await started_on(make_manager, "va", drain_timeout_s=drain)
        previous = manager.spawned[0]

        hung = asyncio.create_task(manager.active_proxy().read_channel(SLOW_CHANNEL))
        # Let the request reach the child before the switch starts draining.
        await asyncio.sleep(0.3)
        assert not hung.done()

        started = time.monotonic()
        result = await manager.switch("live")
        elapsed = time.monotonic() - started

        assert result["previous_drained"] is False
        assert result["target"] == "live"
        assert manager.active_generation() == 2  # start on 'va' moved it once
        assert elapsed < 30.0, "the switch waited for the hung read instead of killing it"
        assert previous.returncode is not None

        with pytest.raises(ConnectionError) as raised:
            await asyncio.wait_for(hung, SETTLE_TIMEOUT_S)
        message = str(raised.value)
        assert "target switch" in message
        assert "'va'" in message and "'live'" in message
        assert "drain deadline" in message

    async def test_a_request_arriving_during_the_switch_is_refused_by_name(self, make_manager):
        manager = await started_on(make_manager, "va", drain_timeout_s=0.5)
        retired = manager.active_proxy()

        hung = asyncio.create_task(retired.read_channel(SLOW_CHANNEL))
        await asyncio.sleep(0.3)
        await manager.switch("live")

        with pytest.raises(ConnectionError) as raised:
            await retired.read_channel(LIVE_PROBE, timeout=5.0)
        assert "target switch" in str(raised.value)
        assert "'live'" in str(raised.value)

        hung.cancel()
        with contextlib.suppress(asyncio.CancelledError, ConnectionError):
            await hung

    async def test_an_idle_child_drains_cleanly(self, make_manager):
        manager = await started_on(make_manager, "live", drain_timeout_s=5.0)

        result = await manager.switch("va")

        assert result["previous_drained"] is True


# ----------------------------------------------------------------- respawns


class TestRespawn:
    async def test_a_same_target_respawn_replaces_the_process_without_a_generation_bump(
        self, make_manager
    ):
        manager = await started_on(make_manager, "va")
        generation = manager.active_generation()
        first_pid = manager.status()["child_pid"]

        result = await manager.respawn_same_target()

        assert result["target"] == "va"
        assert result["target_changed"] is False
        assert result["generation"] == generation
        assert manager.active_generation() == generation
        assert manager.status()["child_pid"] != first_pid
        assert isinstance(
            await manager.active_proxy().read_channel(VA_PROBE, timeout=10.0), ChannelValue
        )
        assert target_state.read()["children"] == [manager.status()["child_pid"]]

    async def test_invalidating_the_control_system_respawns_the_running_child(self, make_manager):
        manager = await started_on(make_manager, "live")
        context = ControlSystemContext()
        context._config = manager._config
        context._connector_hosts = manager
        first_pid = manager.status()["child_pid"]

        await context.invalidate_connector("control_system")

        assert manager.status()["child_pid"] != first_pid
        assert manager.active_generation() == 0
        assert manager.active_target() == "live"

    async def test_invalidating_without_a_connector_host_is_unchanged(self):
        class Recording:
            def __init__(self):
                self.disconnected = False

            async def disconnect(self):
                self.disconnected = True

        context = ControlSystemContext()
        instance = Recording()
        context._connectors["control_system"] = ConnectorEntry(
            config={}, instance=instance, connector_type="control_system"
        )

        await context.invalidate_connector("control_system")

        assert instance.disconnected is True
        assert context._connectors["control_system"].instance is None


# ------------------------------------------------------------ startup sweep


class TestStartupSweep:
    @staticmethod
    def _dead_pid():
        """A PID whose process has been reaped, so nothing answers to it."""
        finished = subprocess.Popen([sys.executable, "-c", ""])
        finished.wait(timeout=SETTLE_TIMEOUT_S)
        return finished.pid

    @staticmethod
    def _orphan_host():
        """A real connector-host child with nobody talking to it."""
        return subprocess.Popen(
            [sys.executable, "-m", "osprey_connectors.ipc.host"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    async def test_orphans_from_a_dead_server_are_killed_and_its_file_swept(
        self, make_manager, state_root
    ):
        orphan = self._orphan_host()
        dead_server = self._dead_pid()
        stale = state_root / target_state.STATE_DIR_NAME
        stale.mkdir(parents=True, exist_ok=True)
        stale_file = stale / f"target_state_{dead_server}.json"
        stale_file.write_text(
            json.dumps(
                {
                    "target": "va",
                    "generation": 4,
                    "server_pid": dead_server,
                    "owner_ppid": 1,
                    "targets": {},
                    "children": [orphan.pid],
                }
            ),
            encoding="utf-8",
        )

        try:
            manager = make_manager()  # reset_state() runs in the factory

            assert orphan.wait(timeout=SETTLE_TIMEOUT_S) is not None
            assert not stale_file.exists()
            record = target_state.read()
            assert record["target"] == manager.baseline == "live"
            assert record["generation"] == 0
            assert record["children"] == []
        finally:
            if orphan.poll() is None:  # pragma: no cover - teardown safety net
                orphan.kill()
                orphan.wait(timeout=SETTLE_TIMEOUT_S)

    def test_a_pid_that_is_already_gone_is_not_reported_as_killed(self):
        assert kill_orphans([self._dead_pid()], grace_s=0.5) == []

    def test_a_reused_pid_belonging_to_something_else_is_left_alone(self):
        """A recorded PID the OS has since handed to another process.

        Killing whatever now holds that number would be a far worse failure
        than leaving one orphan behind, so the sweep checks the command line
        before it signals anything.
        """
        bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        try:
            assert looks_like_a_connector_host(bystander.pid) is False
            assert kill_orphans([bystander.pid], grace_s=0.5) == []
            assert bystander.poll() is None
        finally:
            bystander.kill()
            bystander.wait(timeout=SETTLE_TIMEOUT_S)

    def test_a_live_connector_host_is_recognised_as_one(self):
        orphan = self._orphan_host()
        try:
            assert looks_like_a_connector_host(orphan.pid) is True
        finally:
            orphan.kill()
            orphan.wait(timeout=SETTLE_TIMEOUT_S)


# --------------------------------------------------------- the no-child state


class TestNoChildState:
    async def test_a_child_that_dies_outside_a_switch_leaves_the_session_without_one(
        self, make_manager
    ):
        manager = await started_on(make_manager, "va")
        pid = manager.status()["child_pid"]

        os.kill(pid, signal.SIGKILL)

        assert await wait_for(lambda: not manager.has_child()), (
            "the manager kept reporting a child that had been killed"
        )
        assert manager.active_proxy() is None
        assert manager.status()["child_alive"] is False
        # The session is still pointed where it was: a dead child is not a
        # switch, and claiming otherwise would invent one.
        assert manager.active_target() == "va"
        assert manager.active_generation() == 1

    async def test_a_failed_switch_never_produces_the_no_child_state(self, make_manager):
        manager = await started_on(make_manager, "live", raw=raw_config(va_probe=REFUSE_CHANNEL))

        with pytest.raises(SwitchError):
            await manager.switch("va")

        assert manager.has_child() is True


# ---------------------------------------------------------------- concurrency


class TestConcurrentSwitches:
    async def test_two_switches_serialize_and_the_second_sees_the_first(self, make_manager):
        manager = await started_on(make_manager, "live")

        first, second = await asyncio.gather(manager.switch("va"), manager.switch("live"))

        # Whichever ran first, the other started from its outcome.
        ordered = sorted([first, second], key=lambda result: result["generation"])
        assert [result["generation"] for result in ordered] == [1, 2]
        assert ordered[1]["previous_target"] == ordered[0]["target"]
        assert manager.active_target() == ordered[1]["target"]
        assert manager.active_generation() == 2

        def alive():
            return [process for process in manager.spawned if process.returncode is None]

        assert await wait_for(lambda: len(alive()) == 1), (
            f"{len(alive())} connector-host children were left alive by two switches"
        )


# --------------------------------------------------- verification rule (unit)


class TestVerificationRule:
    NOTHING_CONFIGURED = {
        "selected_role": None,
        "mode": None,
        "host": None,
        "port": None,
        "_epics_configured": False,
    }

    def _derivation(self, endpoints=None):
        return TargetDerivation(
            target="va",
            connector_type="virtual_accelerator",
            endpoints=endpoints or {},
            selected_role="read_only",
        )

    def test_a_gatewayless_target_passes_when_the_child_configured_nothing(self):
        verification = connector_host_manager._verify(self._derivation(), self.NOTHING_CONFIGURED)

        assert verification.ok is True
        assert "derives no gateway" in verification.detail

    def test_a_gatewayless_target_fails_when_the_child_configured_a_gateway(self):
        # The ambient-inheritance failure mode: nothing in config named an
        # endpoint, yet the child came up on one.
        report = {
            "selected_role": "read_only",
            "mode": "addr_list",
            "host": "ambient.example.org",
            "port": 5064,
            "_epics_configured": True,
        }

        verification = connector_host_manager._verify(self._derivation(), report)

        assert verification.ok is False
        assert verification.field == "_epics_configured"
        assert "ambient.example.org" in verification.detail

    def test_a_missing_row_for_the_selected_role_is_not_a_gatewayless_target(self):
        """The unpinned-CA case: gateways configured, none for the role selected.

        A child launched against that config connects to whatever default
        broadcast address it finds. Treating it as "nothing derived, nothing
        configured" would wave it through, so it is verified like any other
        gateway deployment and refused.
        """
        derivation = TargetDerivation(
            target="live",
            connector_type="epics",
            endpoints={
                "write_access": Endpoint(host="gw.example.org", port=5064, mode="addr_list")
            },
            selected_role="read_only",
        )

        verification = connector_host_manager._verify(derivation, self.NOTHING_CONFIGURED)

        assert verification.ok is False
        assert verification.field == "_epics_configured"

    def test_a_missing_row_for_the_selected_role_refuses_a_child_that_found_a_gateway(self):
        derivation = TargetDerivation(
            target="live",
            connector_type="epics",
            endpoints={
                "write_access": Endpoint(host="gw.example.org", port=5064, mode="addr_list")
            },
            selected_role="read_only",
        )
        report = {
            "selected_role": "read_only",
            "mode": "addr_list",
            "host": "255.255.255.255",
            "port": 5064,
            "_epics_configured": True,
        }

        verification = connector_host_manager._verify(derivation, report)

        assert verification.ok is False
        assert verification.field == "endpoints"

    def test_a_configured_target_is_checked_field_by_field(self):
        derivation = self._derivation(
            {"read_only": Endpoint(host="gw.example.org", port=5064, mode="addr_list")}
        )
        report = {
            "selected_role": "read_only",
            "mode": "addr_list",
            "host": "elsewhere.example.org",
            "port": 5064,
            "_epics_configured": True,
        }

        verification = connector_host_manager._verify(derivation, report)

        assert verification.ok is False
        assert verification.field == "host"
        assert verification.expected == "gw.example.org"
        assert verification.got == "elsewhere.example.org"


# ------------------------------------------------- config-derived facts (unit)


class TestConfigDerivedFacts:
    def test_baseline_is_va_only_for_a_virtual_accelerator_deployment(self):
        assert baseline_target({"control_system": {"type": "virtual_accelerator"}}) == "va"
        assert baseline_target({"control_system": {"type": "epics"}}) == "live"
        assert baseline_target({}) == "live"

    def test_display_metadata_carries_the_probe_channel_and_the_real_machine_flag(self):
        metadata = target_display_metadata(raw_config())

        assert metadata["va"]["probe_channel"] == VA_PROBE
        assert metadata["va"]["real_machine"] is False
        assert metadata["va"]["label"] == "virtual accelerator (simulation)"
        assert metadata["live"]["probe_channel"] == LIVE_PROBE
        # A deployment whose control_system.type names a connector class is a
        # deployment that has said what its real machine is, whatever that
        # class turns out to do.
        assert metadata["live"]["real_machine"] is True
        assert metadata["live"]["label"] == "LIVE MACHINE"

    def test_display_metadata_still_describes_a_target_it_cannot_derive(self):
        metadata = target_display_metadata({"control_system": {"type": "virtual_accelerator"}})

        assert metadata["live"] == {
            "label": "live machine (not configured)",
            "endpoint": "",
            "real_machine": False,
            "probe_channel": "",
        }

    async def test_the_drain_timeout_comes_from_config_and_falls_back(self, make_manager):
        assert make_manager(drain_timeout_s=None)._drain_timeout() == DEFAULT_DRAIN_TIMEOUT_S
        assert (
            make_manager(raw=raw_config(drain_timeout_s=2.5), drain_timeout_s=None)._drain_timeout()
            == 2.5
        )
        assert (
            make_manager(
                raw=raw_config(drain_timeout_s="soon"), drain_timeout_s=None
            )._drain_timeout()
            == DEFAULT_DRAIN_TIMEOUT_S
        )

    async def test_the_child_environment_drops_every_epics_variable(
        self, make_manager, monkeypatch
    ):
        monkeypatch.setenv("EPICS_CA_ADDR_LIST", "ambient.example.org")
        monkeypatch.setenv("EPICS_PVA_NAME_SERVERS", "ambient.example.org:5075")
        monkeypatch.setenv("PYEPICS_LIBCA", "/opt/libca.dylib")

        env = make_manager().child_env()

        assert "EPICS_CA_ADDR_LIST" not in env
        assert "EPICS_PVA_NAME_SERVERS" not in env
        assert env["PYEPICS_LIBCA"] == "/opt/libca.dylib"


# ------------------------------------------------------ the serving seam


class TestSwitchCapability:
    """Which deployments serve their tools from a child, and which do not."""

    def test_a_deployment_with_both_targets_configured_is_capable(self):
        assert switch_capable(raw_config()) is True

    def test_a_virtual_accelerator_deployment_with_a_live_block_is_capable(self):
        config = {
            "control_system": {
                "type": "virtual_accelerator",
                "connector": {"virtual_accelerator": {"timeout": 5.0}, "epics": {"timeout": 5.0}},
            }
        }

        assert switch_capable(config) is True

    def test_a_mock_deployment_with_an_epics_block_is_not_capable(self):
        """The case that makes the naive predicate dangerous.

        ``resolve_target`` answers 'live' for a mock deployment by looking in
        the connector table and finding the epics block — so a predicate that
        only asked "do both targets resolve" would serve this deployment from a
        child pointed at a real machine its own config never selected.
        """
        config = {
            "control_system": {
                "type": "mock",
                "connector": {"mock": {}, "epics": {"gateways": {"read_only": {}}}},
            }
        }

        assert switch_capable(config) is False

    def test_a_single_target_deployment_is_not_capable(self):
        assert (
            switch_capable({"control_system": {"type": "epics", "connector": {"epics": {}}}})
            is False
        )

    def test_a_config_with_no_control_system_section_is_not_capable(self):
        assert switch_capable({}) is False


class TestServingFromTheChild:
    async def test_the_first_tool_call_brings_the_baseline_child_up(self, make_manager):
        manager = make_manager()
        context = context_for(manager)
        assert context.switch_capable is True
        assert manager.is_started() is False

        connector = await context.control_system()

        assert isinstance(connector, ConnectorHostProxy)
        assert connector is manager.active_proxy()
        assert manager.is_started() is True
        assert manager.active_target() == manager.baseline == "live"
        # Bringing the first child up is not a switch: nothing moved.
        assert manager.active_generation() == 0
        assert len(manager.spawned) == 1

    async def test_concurrent_first_calls_start_exactly_one_child(self, make_manager):
        manager = make_manager()
        context = context_for(manager)

        connectors = await asyncio.gather(*(context.control_system() for _ in range(4)))

        assert all(connector is connectors[0] for connector in connectors)
        assert len(manager.spawned) == 1

    async def test_tool_calls_reach_the_child_and_never_build_a_local_connector(self, make_manager):
        """The seam this whole task exists for: a switch moves where reads land.

        The tripwire is registered under both targets' connector types, so any
        path that built an in-process connector — before or after the switch —
        fails the test instead of silently answering from the baseline
        connector while the prompt says the session is somewhere else.
        """
        manager = await started_on(make_manager, "live")
        context = context_for(manager)

        with isolated_connector_registries():
            ConnectorFactory.register_control_system(LIVE_TYPE, _NeverBuilt)
            ConnectorFactory.register_control_system("virtual_accelerator", _NeverBuilt)

            before = await context.control_system()
            assert isinstance(before, ConnectorHostProxy)
            assert isinstance(await before.read_channel(LIVE_PROBE, timeout=10.0), ChannelValue)

            result = await manager.switch("va")

            after = await context.control_system()
            assert after is manager.active_proxy()
            assert after is not before
            # The VA child is the fixture connector, which refuses this channel:
            # the read is being served by the process the switch just adopted.
            with pytest.raises(ConnectionError):
                await after.read_channel(REFUSE_CHANNEL, timeout=10.0)
            assert isinstance(await after.read_channel(VA_PROBE, timeout=10.0), ChannelValue)
            assert result["child_pid"] == manager.status()["child_pid"]

        # Nothing was ever cached in the in-process entry.
        assert context._connectors["control_system"].instance is None

    async def test_a_write_is_answered_by_the_child_that_serves_the_target(self, make_manager):
        """A write crosses the boundary and comes back with the child's verdict.

        These children run without a project config, so writes are disabled and
        the honest answer is a refusal — one produced by the connector in the
        child, carried back over the wire with its reason intact.
        """
        manager = await started_on(make_manager, "va")
        context = context_for(manager)

        connector = await context.control_system()
        result = await connector.write_channel("SR:CORR:1:SP", 0.5, timeout=10.0)

        assert result.blocked is True
        assert result.refusal_reason == "WRITES_DISABLED"
        assert result.channel_address == "SR:CORR:1:SP"

    async def test_the_archiver_is_never_routed_through_the_connector_host(self, make_manager):
        manager = await started_on(make_manager, "live")
        context = context_for(manager)

        with isolated_connector_registries():
            ConnectorFactory.register_control_system(LIVE_TYPE, _NeverBuilt)
            ConnectorFactory.register_archiver("mongodb_archiver", _FakeArchiver)

            archiver = await context.archiver()

            assert isinstance(archiver, _FakeArchiver)
            assert await context.archiver() is archiver

    async def test_the_baseline_starts_unprobed_but_a_switch_to_it_still_needs_a_probe(
        self, make_manager
    ):
        """The shipped posture: the live block carries no probe channel.

        Starting the deployment's own baseline is not a swap, so it does not
        need a channel to prove itself with — refusing to start would leave the
        server with no control system at all. Switching *to* that target later
        is a swap, and is refused for exactly the missing channel.
        """
        manager = make_manager(raw=raw_config(live_probe=None))
        context = context_for(manager)

        connector = await context.control_system()
        assert isinstance(connector, ConnectorHostProxy)
        assert manager.active_target() == "live"

        await manager.switch("va")
        with pytest.raises(SwitchError) as raised:
            await manager.switch("live")

        assert raised.value.reason == REASON_PROBE_CHANNEL_MISSING
        assert manager.active_target() == "va"


class TestNoChildRefusal:
    async def test_the_accessor_refuses_with_the_reason_when_the_child_is_dead(self, make_manager):
        manager = await started_on(make_manager, "va")
        context = context_for(manager)
        await context.control_system()

        os.kill(manager.status()["child_pid"], signal.SIGKILL)
        assert await wait_for(lambda: not manager.has_child())

        with pytest.raises(NoConnectorHostError) as raised:
            await context.control_system()

        refusal = raised.value
        assert isinstance(refusal, ConnectionError)  # the tools' handler knows this one
        assert refusal.target == "va"
        assert refusal.generation == manager.active_generation()
        assert refusal.as_dict()["reason"] == "no_connector_host"
        assert "va" in str(refusal)
        # A dead child is never quietly restarted on the way to a read.
        assert manager.has_child() is False
        assert len(manager.spawned) == 1

    async def test_invalidating_the_connector_is_what_brings_it_back(self, make_manager):
        """The recovery route: refuse, invalidate, respawn, serve.

        This is the path the tools' shared error handler already takes on a
        ConnectionError, which is why the refusal is one.
        """
        manager = await started_on(make_manager, "va")
        context = context_for(manager)
        await context.control_system()
        generation = manager.active_generation()

        os.kill(manager.status()["child_pid"], signal.SIGKILL)
        assert await wait_for(lambda: not manager.has_child())
        with pytest.raises(NoConnectorHostError):
            await context.control_system()

        await context.invalidate_connector("control_system")

        assert manager.has_child() is True
        assert manager.active_target() == "va"
        assert manager.active_generation() == generation
        connector = await context.control_system()
        assert isinstance(await connector.read_channel(VA_PROBE, timeout=10.0), ChannelValue)


class TestNonCapableDeploymentIsUntouched:
    """A deployment that cannot switch keeps the path it always had."""

    @staticmethod
    def _mock_context():
        raw = {
            "control_system": {
                "type": "mock",
                "connector": {"mock": {"response_delay_ms": 1, "noise_level": 0.0}},
            },
            "archiver": {"type": "mongodb_archiver"},
        }
        context = ControlSystemContext()
        context._config = MCPServerConfig(raw=raw, config_path=None)
        # What initialize() does, minus reading the deployment's config.yml.
        context._register_connector_types()
        context._connectors["control_system"] = ConnectorEntry(
            config=raw["control_system"], connector_type="control_system"
        )
        return context

    async def test_the_in_process_connector_is_built_and_cached(self):
        context = self._mock_context()
        assert context.switch_capable is False

        connector = await context.control_system()

        from osprey_connectors.control_system.mock_connector import MockConnector

        assert isinstance(connector, MockConnector)
        assert await context.control_system() is connector
        assert context._connectors["control_system"].instance is connector
        # No supervisor is created, so nothing can spawn a child.
        assert context._connector_hosts is None

    async def test_invalidating_drops_the_instance_and_the_next_call_rebuilds(self):
        context = self._mock_context()
        first = await context.control_system()

        await context.invalidate_connector("control_system")

        assert context._connectors["control_system"].instance is None
        assert context._connector_hosts is None
        assert await context.control_system() is not first

    async def test_an_unknown_connector_name_still_raises_value_error(self):
        context = self._mock_context()

        with pytest.raises(ValueError, match="Unknown connector"):
            await context._get_connector("nonexistent")
