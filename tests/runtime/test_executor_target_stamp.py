"""The control-target stamp: host stamps it, sandbox routes and pins on it.

Two processes are involved and neither can see the other's state directly. The
host (``python_executor.executor``) resolves which controls server belongs to
this session and writes the target into the sandbox environment; the sandbox
(``osprey.runtime``) builds its connector from that stamp and refuses writes
once the generation it was stamped at has moved on.

Every test here drives one of those two halves against a real state file in
``tmp_path``. Nothing touches EPICS: the sandbox half registers a fake connector
class inside ``isolated_connector_registries`` and asserts on the config that
class was handed, which is the only evidence that routing actually happened.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest

from osprey.mcp_server.control_system import target_state
from osprey.mcp_server.python_executor import executor as host_executor
from osprey.runtime import ControlTargetChangedError

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def state_root(tmp_path, monkeypatch):
    """Point the target-state module at a throwaway shared-data root.

    ``state_dir()`` resolves through the name ``resolve_shared_data_root`` bound
    in ``target_state``'s namespace, so patching it there redirects every reader
    in this process — the host helper and the sandbox helper alike.
    """
    root = tmp_path / "var" / "agent_data"
    monkeypatch.setattr(target_state, "resolve_shared_data_root", lambda: root)
    (root / target_state.STATE_DIR_NAME).mkdir(parents=True)
    return root


def write_record(
    *,
    target: str = "va",
    generation: int = 0,
    server_pid: int | None = None,
    owner_ppid: int | None = None,
) -> Path:
    """Write one state record. Defaults describe a live, this-session server."""
    pid = os.getpid() if server_pid is None else server_pid
    record = {
        "target": target,
        "generation": generation,
        "server_pid": pid,
        "owner_ppid": os.getppid() if owner_ppid is None else owner_ppid,
        "targets": {
            name: {"label": "", "endpoint": "", "real_machine": False} for name in ("live", "va")
        },
        "children": [],
    }
    path = target_state.state_file_path(pid)
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def stamp_env(monkeypatch, *, target: str, generation: str, state_pid: int | None = None) -> None:
    """Put a stamp in this process's environment, as the host would."""
    monkeypatch.setenv(host_executor.ENV_CONTROL_TARGET, target)
    monkeypatch.setenv(host_executor.ENV_CONTROL_TARGET_GENERATION, generation)
    pid = os.getpid() if state_pid is None else state_pid
    monkeypatch.setenv(host_executor.ENV_CONTROL_TARGET_STATE_PID, str(pid))


@pytest.fixture
def clear_stamp(monkeypatch):
    """Start every test from an unstamped environment."""
    for name in host_executor._STAMP_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


#: A deployment that has both a simulated baseline and one real machine, so
#: 'va' and 'live' each resolve to exactly one connector block.
CONTROL_SYSTEM_SECTION = {
    "type": "mock",
    "connector": {
        "mock": {"response_delay_ms": 0},
        "epics": {"timeout": 1.0},
        "virtual_accelerator": {"timeout": 9.0},
    },
}

#: A development checkout that has never named a real machine: 'live' has no
#: answer here, which is what resolve_target refuses on.
MOCK_ONLY_SECTION = {"type": "mock", "connector": {"mock": {}}}


def _section_reader(section):
    """A ``get_config_value`` stand-in serving *section* as ``control_system``."""

    def get_config_value(path, default=None, config_path=None):
        return section if path == "control_system" else default

    return get_config_value


@pytest.fixture
def deployment_config(monkeypatch):
    """Serve one control_system section to both halves of the stamp.

    Host and sandbox deliberately read the same function, so a target the host
    stamps is a target the sandbox can build.
    """
    monkeypatch.setattr(
        "osprey_connectors.config.get_config_value", _section_reader(CONTROL_SYSTEM_SECTION)
    )


@pytest.fixture
def clear_runtime_state():
    """Drop the runtime's cached connector around each test."""
    import osprey.runtime as runtime

    runtime._runtime_connector = None
    runtime._limits_validator = None
    yield
    runtime._runtime_connector = None
    runtime._limits_validator = None


def test_env_names_agree_across_the_process_boundary():
    """The stamp is a contract between two modules; the literals must match.

    They are spelled twice on purpose — the sandbox reader must not import the
    host executor — so the only thing keeping them equal is this assertion.
    """
    import osprey.runtime as runtime

    assert host_executor.ENV_CONTROL_TARGET == runtime.ENV_CONTROL_TARGET
    assert host_executor.ENV_CONTROL_TARGET_GENERATION == runtime.ENV_CONTROL_TARGET_GENERATION
    assert host_executor.ENV_CONTROL_TARGET_STATE_PID == runtime.ENV_CONTROL_TARGET_STATE_PID


# ---------------------------------------------------------------------------
# Host side: resolving the session record and stamping the sandbox env
# ---------------------------------------------------------------------------


class TestSessionRecordLookup:
    """Which state file describes *this* session, and when is the answer none."""

    def test_matching_owner_ppid_is_the_session_record(self, state_root):
        write_record(target="va", generation=3)

        record = host_executor._session_target_record()

        assert record is not None
        assert record["target"] == "va"
        assert record["generation"] == 3

    def test_other_sessions_record_is_not_ours(self, state_root):
        write_record(target="va", generation=3, owner_ppid=os.getppid() + 100000)

        assert host_executor._session_target_record() is None

    def test_dead_server_record_is_residue(self, state_root, monkeypatch):
        write_record(target="va", generation=3)
        monkeypatch.setattr(target_state, "is_process_alive", lambda pid: False)

        assert host_executor._session_target_record() is None

    def test_two_records_sharing_our_ppid_are_ambiguous(self, state_root):
        write_record(target="va", generation=3, server_pid=os.getpid())
        write_record(target="live", generation=4, server_pid=os.getppid())

        assert host_executor._session_target_record() is None

    def test_missing_state_directory_is_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            target_state, "resolve_shared_data_root", lambda: tmp_path / "never-created"
        )

        assert host_executor._session_target_record() is None

    def test_corrupt_record_is_ignored(self, state_root):
        target_state.state_file_path(os.getpid()).write_text("{not json", encoding="utf-8")

        assert host_executor._session_target_record() is None

    def test_unknown_target_name_is_not_stamped(self, state_root):
        write_record(target="production", generation=1)

        assert host_executor._session_target_record() is None


class TestStampApplication:
    """What ``_apply_target_stamp`` puts in — and takes out of — the sandbox env."""

    def test_every_name_is_stamped(self, state_root, deployment_config):
        write_record(target="va", generation=7)
        env: dict[str, str] = {}

        assert host_executor._apply_target_stamp(env) == "va"
        assert env[host_executor.ENV_CONTROL_TARGET] == "va"
        assert env[host_executor.ENV_CONTROL_TARGET_GENERATION] == "7"
        # The record's identity, so the sandbox pins against this file and does
        # not have to search for one.
        assert env[host_executor.ENV_CONTROL_TARGET_STATE_PID] == str(os.getpid())

    def test_no_record_omits_every_name(self, state_root, deployment_config):
        env: dict[str, str] = {}

        assert host_executor._apply_target_stamp(env) == host_executor.CONTROL_TARGET_BASELINE
        for name in host_executor._STAMP_ENV_NAMES:
            assert name not in env

    def test_inherited_stamp_is_stripped_when_unresolvable(self, state_root, deployment_config):
        """An ancestor's stamp must not be passed through as if it were ours.

        The host inherits its own environment from Claude Code, so a stale
        ``OSPREY_CONTROL_TARGET`` can be sitting there. Leaving it in place would
        route agent code at a target this run never resolved.
        """
        env = {
            host_executor.ENV_CONTROL_TARGET: "live",
            host_executor.ENV_CONTROL_TARGET_GENERATION: "2",
            host_executor.ENV_CONTROL_TARGET_STATE_PID: "4321",
        }

        assert host_executor._apply_target_stamp(env) == host_executor.CONTROL_TARGET_BASELINE
        for name in host_executor._STAMP_ENV_NAMES:
            assert name not in env

    def test_live_on_a_deployment_without_a_real_machine_is_not_stamped(
        self, state_root, monkeypatch
    ):
        """A target the sandbox could not build is declined here, not there.

        ``resolve_target(section, 'live')`` refuses on a mock-only checkout by
        design. Stamping it anyway would turn every execute() on such a
        deployment into a ValueError raised inside the sandbox; declining leaves
        the run on the baseline, which is what it was on before any of this.
        """
        monkeypatch.setattr(
            "osprey_connectors.config.get_config_value", _section_reader(MOCK_ONLY_SECTION)
        )
        write_record(target="live", generation=1)
        env: dict[str, str] = {}

        assert host_executor._apply_target_stamp(env) == host_executor.CONTROL_TARGET_BASELINE
        for name in host_executor._STAMP_ENV_NAMES:
            assert name not in env

    def test_va_is_stamped_on_that_same_deployment(self, state_root, monkeypatch):
        """Only the unresolvable half is declined: 'va' resolves everywhere."""
        monkeypatch.setattr(
            "osprey_connectors.config.get_config_value", _section_reader(MOCK_ONLY_SECTION)
        )
        write_record(target="va", generation=1)
        env: dict[str, str] = {}

        assert host_executor._apply_target_stamp(env) == "va"
        assert env[host_executor.ENV_CONTROL_TARGET] == "va"


class TestExecuteViaLocalStamping:
    """End-to-end through ``_execute_via_local``, with the subprocess faked out."""

    @staticmethod
    def _run(tmp_path, monkeypatch) -> tuple[dict[str, str], Any]:
        """Run the adapter against a fake subprocess; return (env, result)."""
        captured: dict[str, dict[str, str]] = {}

        class _FakeProc:
            returncode = 0

            async def communicate(self):
                return b"", b""

        async def fake_exec(*args, **kwargs):
            captured["env"] = kwargs["env"]
            return _FakeProc()

        folder = tmp_path / "execution"
        (folder / "figures").mkdir(parents=True)

        monkeypatch.setattr(host_executor, "_resolve_project_root", lambda: tmp_path)
        monkeypatch.setattr(
            host_executor, "resolve_agent_interpreter", lambda root=None: "/bin/true"
        )
        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

        result = asyncio.run(
            host_executor._execute_via_local(
                "print('hello')",
                "readonly",
                {"timeout": 5},
                folder,
                None,
            )
        )
        return captured["env"], result

    def test_sandbox_env_carries_the_stamp(
        self, state_root, deployment_config, tmp_path, monkeypatch
    ):
        write_record(target="va", generation=5)

        env, result = self._run(tmp_path, monkeypatch)

        assert env[host_executor.ENV_CONTROL_TARGET] == "va"
        assert env[host_executor.ENV_CONTROL_TARGET_GENERATION] == "5"
        assert env[host_executor.ENV_CONTROL_TARGET_STATE_PID] == str(os.getpid())
        # The mode injection this stamp sits beside must survive untouched.
        assert env["OSPREY_EXECUTION_MODE"] == "readonly"
        assert result.control_target == "va"

    def test_unstamped_run_records_the_baseline(
        self, state_root, deployment_config, tmp_path, monkeypatch
    ):
        env, result = self._run(tmp_path, monkeypatch)

        for name in host_executor._STAMP_ENV_NAMES:
            assert name not in env
        assert result.control_target == host_executor.CONTROL_TARGET_BASELINE

    def test_result_default_is_the_baseline(self):
        """A result built without a target — a setup failure — claims nothing."""
        result = host_executor.ExecutionResult(success=False, stdout="", stderr="")

        assert result.control_target == host_executor.CONTROL_TARGET_BASELINE


# ---------------------------------------------------------------------------
# Sandbox side: routing the connector from the stamp
# ---------------------------------------------------------------------------


class _FakeConnector:
    """Records the type-specific config block the factory handed ``connect()``."""

    last_config: dict[str, Any] | None = None

    async def connect(self, config: dict[str, Any]) -> None:
        type(self).last_config = config

    async def disconnect(self) -> None:  # pragma: no cover - cleanup path
        pass


@pytest.fixture
def fake_registry(deployment_config):
    """Register the fake connector under every type this deployment can select."""
    from osprey_connectors.factory import ConnectorFactory, isolated_connector_registries

    with isolated_connector_registries():
        for name in ("mock", "epics", "virtual_accelerator"):
            ConnectorFactory.register_control_system(name, _FakeConnector)
        _FakeConnector.last_config = None
        yield
        _FakeConnector.last_config = None


class TestSandboxRouting:
    """The stamp, not ``control_system.type``, selects the connector block."""

    def test_va_stamp_builds_the_virtual_accelerator_block(
        self, monkeypatch, fake_registry, clear_runtime_state
    ):
        monkeypatch.setenv("OSPREY_CONTROL_TARGET", "va")

        import osprey.runtime as runtime

        asyncio.run(runtime._get_connector())

        # 9.0 is the VA block's timeout: reaching it proves the factory read
        # control_system.connector.virtual_accelerator and not the mock block.
        assert _FakeConnector.last_config == {"timeout": 9.0}

    def test_live_stamp_builds_the_deployments_real_machine_block(
        self, monkeypatch, fake_registry, clear_runtime_state
    ):
        monkeypatch.setenv("OSPREY_CONTROL_TARGET", "live")

        import osprey.runtime as runtime

        asyncio.run(runtime._get_connector())

        assert _FakeConnector.last_config == {"timeout": 1.0}

    def test_unstamped_resolution_is_unchanged(
        self, monkeypatch, clear_stamp, fake_registry, clear_runtime_state
    ):
        """No stamp means the factory loads the section itself, as it always did."""
        import osprey.runtime as runtime

        assert runtime._target_connector_config() is None

        asyncio.run(runtime._get_connector())

        assert _FakeConnector.last_config == {"response_delay_ms": 0}

    def test_blank_stamp_counts_as_absent(self, monkeypatch, clear_runtime_state):
        monkeypatch.setenv("OSPREY_CONTROL_TARGET", "   ")

        import osprey.runtime as runtime

        assert runtime._target_connector_config() is None

    def test_unresolvable_live_target_refuses_rather_than_falling_back(
        self, monkeypatch, clear_runtime_state
    ):
        """A deployment that never named its real machine gets an error, not the mock.

        The host declines to stamp this combination in the first place (see
        ``TestStampApplication``); this is the second line of that defence, for a
        stamp that arrives from anywhere else.
        """
        monkeypatch.setenv("OSPREY_CONTROL_TARGET", "live")
        monkeypatch.setattr(
            "osprey_connectors.config.get_config_value", _section_reader(MOCK_ONLY_SECTION)
        )

        import osprey.runtime as runtime

        with pytest.raises(ValueError, match="no control system on this deployment"):
            runtime._target_connector_config()


# ---------------------------------------------------------------------------
# Sandbox side: the write pin
# ---------------------------------------------------------------------------


class TestWritePin:
    """Writes refuse once the session's target or generation moves."""

    def test_matching_generation_lets_the_write_through(
        self, state_root, monkeypatch, clear_runtime_state
    ):
        write_record(target="va", generation=3)
        stamp_env(monkeypatch, target="va", generation="3")

        import osprey.runtime as runtime

        runtime._assert_target_pin()  # does not raise

    def test_moved_generation_refuses_and_names_both(
        self, state_root, monkeypatch, clear_runtime_state
    ):
        write_record(target="va", generation=4)
        stamp_env(monkeypatch, target="va", generation="3")

        import osprey.runtime as runtime

        with pytest.raises(ControlTargetChangedError) as excinfo:
            runtime._assert_target_pin()

        message = str(excinfo.value)
        assert "generation 3" in message
        assert "generation 4" in message
        assert "never reconnect" in message

    def test_moved_target_refuses_at_the_same_generation(
        self, state_root, monkeypatch, clear_runtime_state
    ):
        write_record(target="live", generation=3)
        stamp_env(monkeypatch, target="va", generation="3")

        import osprey.runtime as runtime

        with pytest.raises(ControlTargetChangedError) as excinfo:
            runtime._assert_target_pin()

        assert "'va'" in str(excinfo.value)
        assert "'live'" in str(excinfo.value)

    def test_stamped_but_state_missing_refuses(self, state_root, monkeypatch, clear_runtime_state):
        """The stamped server's file is gone: current generation is unknowable."""
        stamp_env(monkeypatch, target="va", generation="3")

        import osprey.runtime as runtime

        with pytest.raises(ControlTargetChangedError, match="is gone or its state file"):
            runtime._assert_target_pin()

    def test_stamped_server_no_longer_running_refuses(
        self, state_root, monkeypatch, clear_runtime_state
    ):
        """A record left behind by a dead server is residue, not current state."""
        write_record(target="va", generation=3)
        stamp_env(monkeypatch, target="va", generation="3")
        monkeypatch.setattr(target_state, "is_process_alive", lambda pid: False)

        import osprey.runtime as runtime

        with pytest.raises(ControlTargetChangedError, match="is gone or its state file"):
            runtime._assert_target_pin()

    def test_stamp_without_a_state_pid_refuses(self, state_root, monkeypatch, clear_runtime_state):
        """A target claim with no record identity cannot be checked, so it fails closed."""
        write_record(target="va", generation=3)
        monkeypatch.setenv("OSPREY_CONTROL_TARGET", "va")
        monkeypatch.setenv("OSPREY_CONTROL_TARGET_GENERATION", "3")
        monkeypatch.delenv("OSPREY_CONTROL_TARGET_STATE_PID", raising=False)

        import osprey.runtime as runtime

        with pytest.raises(ControlTargetChangedError, match="no state-file identity"):
            runtime._assert_target_pin()

    def test_stamped_but_generation_unparseable_refuses(
        self, state_root, monkeypatch, clear_runtime_state
    ):
        write_record(target="va", generation=3)
        stamp_env(monkeypatch, target="va", generation="not-a-number")

        import osprey.runtime as runtime

        with pytest.raises(ControlTargetChangedError, match="generation unknown"):
            runtime._assert_target_pin()

    def test_two_sessions_pin_against_their_own_record(
        self, state_root, monkeypatch, clear_runtime_state
    ):
        """Two sessions sharing a checkout is supported, and both must keep writing.

        The stamp carries the identity of the record it was taken from, so this
        session's pin reads only its own file. Without that identity the
        neighbour's record would either make every write ambiguous or — worse —
        answer for a server that is not this session's.
        """
        write_record(target="va", generation=3, server_pid=os.getpid())
        write_record(target="live", generation=9, server_pid=os.getppid())
        stamp_env(monkeypatch, target="va", generation="3", state_pid=os.getpid())

        import osprey.runtime as runtime

        runtime._assert_target_pin()  # the neighbour's record is irrelevant

    def test_a_strangers_record_can_never_satisfy_the_pin(
        self, state_root, monkeypatch, clear_runtime_state
    ):
        """This session's server died; only a foreign record is left. Refuse.

        The foreign record says exactly what the stamp says, so a pin that
        searched the directory for "the live record" would pass here — against a
        server this execution was never talking to.
        """
        write_record(target="va", generation=3, server_pid=os.getppid())
        stamp_env(monkeypatch, target="va", generation="3", state_pid=os.getpid())

        import osprey.runtime as runtime

        with pytest.raises(ControlTargetChangedError, match="is gone or its state file"):
            runtime._assert_target_pin()

    def test_unstamped_process_is_not_pinned(self, state_root, clear_stamp, clear_runtime_state):
        """Baseline routing claimed no target, so there is nothing to drift from."""
        write_record(target="va", generation=99)

        import osprey.runtime as runtime

        runtime._assert_target_pin()  # does not raise

    def test_write_channel_refuses_before_touching_the_connector(
        self, state_root, monkeypatch, fake_registry, clear_runtime_state
    ):
        """The refusal happens on the write path itself, not only in the helper."""
        write_record(target="va", generation=4)
        stamp_env(monkeypatch, target="va", generation="3")

        import osprey.runtime as runtime

        with pytest.raises(ControlTargetChangedError):
            runtime.write_channel("TEST:PV", 1.0)
        with pytest.raises(ControlTargetChangedError):
            runtime.write_channels({"TEST:PV1": 1.0, "TEST:PV2": 2.0})

        # Nothing was built, so nothing could have been written.
        assert _FakeConnector.last_config is None

    def test_reads_are_not_pinned(self, state_root, monkeypatch, clear_runtime_state):
        """FR-7 pins writes only: a run may keep reading the machine it started on."""
        write_record(target="va", generation=4)
        stamp_env(monkeypatch, target="va", generation="3")

        import osprey.runtime as runtime

        class _Reader:
            async def read_channel(self, channel_address, **kwargs):
                class _Value:
                    value = 42.0

                return _Value()

        runtime._runtime_connector = _Reader()

        assert runtime.read_channel("TEST:PV") == 42.0
