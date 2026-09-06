"""The control-target stamp: host stamps it, sandbox routes and pins on it.

Two processes are involved and neither can see the other's state directly. The
host (``python_executor.executor``) reads the deployment's control-context
record, admits the run only if the fleet has settled on it, and writes the
target into the sandbox environment; the sandbox (``osprey.runtime``) builds
its connector from that stamp and refuses writes once the generation it was
stamped at has moved on.

Every test here drives one of those halves against a real record in
``tmp_path``. Nothing touches EPICS: the sandbox half registers a fake
connector class inside ``isolated_connector_registries`` and asserts on the
config that class was handed, which is the only evidence that routing actually
happened.

The convergence half is the executor's *use* of
:func:`osprey_connectors.control_context.blocking_pids` — that predicate's own
rules are exercised exhaustively in ``tests/connectors/test_control_context.py``
and are not restated here.
"""

import asyncio
import contextlib
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from osprey.mcp_server.python_executor import executor as host_executor
from osprey.runtime import ControlTargetChangedError
from osprey_connectors import control_context, posture_store

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def state_root(tmp_path, monkeypatch):
    """Point every reader in this process at a throwaway agent-data root.

    The record, the server reports and the in-flight markers all resolve
    through the ``OSPREY_AGENT_DATA_ROOT`` stamp, so one environment variable
    redirects the host half and the sandbox half together — which is the point
    of there being one record per deployment rather than one file per process.
    """
    root = tmp_path / "var" / "agent_data"
    (root / control_context.STATE_DIR_NAME).mkdir(parents=True)
    monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(root))
    monkeypatch.delenv(posture_store.LAUNCH_POSTURE_ENV_VAR, raising=False)
    monkeypatch.delenv("OSPREY_POSTURE_SESSION", raising=False)
    control_context.invalidate_cache()
    yield root
    control_context.invalidate_cache()


def write_record(
    root: Path,
    *,
    target: str = "va",
    generation: int = 0,
    posture: dict[str, str] | None = None,
) -> Path:
    """Write the deployment's control-context record. Defaults describe ``va``/0."""
    path = control_context.record_path_under(root)
    record = control_context.ControlContext(
        target=target,
        generation=generation,
        posture=dict(posture or {}),
    )
    control_context.write_record(record, path=path)
    control_context.invalidate_cache()
    return path


def write_report(
    root: Path,
    pid: int,
    *,
    session: str | None = None,
    applied_target: str | None = None,
    applied_generation: int | None = None,
    last_switch: dict[str, Any] | None = None,
) -> Path:
    """Write one controls server's report file.

    ``pid`` has to name a live process for the fleet readers to keep the
    report, so the tests use this process and its parent — two PIDs that are
    certainly alive and certainly different.
    """
    payload = {
        "server_pid": pid,
        "session": session,
        "applied_target": applied_target,
        "applied_generation": applied_generation,
        "children": [],
        "reachability": {},
        "last_switch": last_switch,
        "targets": {},
        "updated_at": datetime.now(UTC).isoformat(),
    }
    path = control_context.report_path_under(root, pid)
    path.write_text(json.dumps(payload), encoding="utf-8")
    control_context.invalidate_cache()
    return path


def applying(generation: int, *, seconds_left: float = 60.0) -> dict[str, Any]:
    """A switch block saying this server is mid-swap and still within its bound."""
    return {
        "status": control_context.REPORT_APPLYING,
        "generation": generation,
        "expires_at": (datetime.now(UTC) + timedelta(seconds=seconds_left)).isoformat(),
    }


def failed(generation: int) -> dict[str, Any]:
    """A switch block saying this server could not follow the fleet."""
    return {"status": control_context.REPORT_FAILED, "generation": generation}


def stamp_env(monkeypatch, *, target: str, generation: str) -> None:
    """Put a stamp in this process's environment, as the host would."""
    monkeypatch.setenv(host_executor.ENV_CONTROL_TARGET, target)
    monkeypatch.setenv(host_executor.ENV_CONTROL_TARGET_GENERATION, generation)


@pytest.fixture
def clear_stamp(monkeypatch):
    """Start every test from an unstamped environment."""
    for name in host_executor._STAMP_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def markers_in(root: Path) -> list[dict[str, Any]]:
    """Every in-flight execution marker under *root*, read straight off disk."""
    directory = root / control_context.STATE_DIR_NAME
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob(f"{host_executor.INFLIGHT_FILE_PREFIX}*.json"))
    ]


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
    """Drop the runtime's cached connector, and the stamp it was built for."""
    import osprey.runtime as runtime

    def reset():
        runtime._runtime_connector = None
        runtime._connector_stamp = None
        runtime._cell_marker = None
        runtime._limits_validator = None

    reset()
    yield
    reset()


def test_env_names_agree_across_the_process_boundary():
    """The stamp is a contract between two modules; the literals must match.

    They are spelled twice on purpose — the sandbox reader must not import the
    host executor — so the only thing keeping them equal is this assertion.
    """
    import osprey.runtime as runtime

    assert host_executor.ENV_CONTROL_TARGET == runtime.ENV_CONTROL_TARGET
    assert host_executor.ENV_CONTROL_TARGET_GENERATION == runtime.ENV_CONTROL_TARGET_GENERATION


def test_the_sandbox_carries_no_state_file_identity():
    """The record is the deployment's, so there is no per-server file to name.

    The sandbox used to be handed the PID of the controls server its stamp came
    from, and pinned against that one file. With one record per deployment
    there is nothing to identify, and the reader must not have kept a way to
    ask.
    """
    import osprey.runtime as runtime

    assert not hasattr(runtime, "ENV_CONTROL_TARGET_STATE_PID")
    assert not hasattr(runtime, "_stamped_state_pid")


# ---------------------------------------------------------------------------
# Host side: reading the record and stamping the sandbox env
# ---------------------------------------------------------------------------


class TestDeploymentRecordLookup:
    """What the executor reads, and when the honest answer is ``None``."""

    def test_the_record_is_the_deployments_own(self, state_root):
        write_record(state_root, target="va", generation=3)

        record = host_executor._deployment_record()

        assert record is not None
        assert record.target == "va"
        assert record.generation == 3

    def test_no_record_is_not_an_error(self, state_root):
        assert host_executor._deployment_record() is None

    def test_corrupt_record_is_ignored(self, state_root):
        control_context.record_path_under(state_root).write_text("{not json", encoding="utf-8")
        control_context.invalidate_cache()

        assert host_executor._deployment_record() is None

    def test_a_record_without_a_generation_cannot_pin_a_run(self, state_root):
        """Generation is half the pin, so a record missing it describes nothing."""
        control_context.record_path_under(state_root).write_text(
            json.dumps({"schema": 1, "target": "va"}), encoding="utf-8"
        )
        control_context.invalidate_cache()

        assert host_executor._deployment_record() is None

    def test_unknown_target_name_is_not_a_record(self, state_root):
        control_context.record_path_under(state_root).write_text(
            json.dumps({"schema": 1, "target": "production", "generation": 1}), encoding="utf-8"
        )
        control_context.invalidate_cache()

        assert host_executor._deployment_record() is None

    def test_an_unresolvable_root_is_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(tmp_path / "never-created"))
        control_context.invalidate_cache()

        assert host_executor._deployment_record() is None


class TestStampApplication:
    """What ``_apply_target_stamp`` puts in — and takes out of — the sandbox env."""

    def test_the_target_and_generation_are_stamped(self, state_root, deployment_config):
        write_record(state_root, target="va", generation=7)
        env: dict[str, str] = {}

        assert host_executor._apply_target_stamp(env) == "va"
        assert env[host_executor.ENV_CONTROL_TARGET] == "va"
        assert env[host_executor.ENV_CONTROL_TARGET_GENERATION] == "7"

    def test_the_retired_state_pid_stamp_is_cleared_not_written(
        self, state_root, deployment_config
    ):
        """A stamped run carries no state-file identity.

        The sandbox pins against the record now. The name the stamp used to
        occupy is gone from the writer as well as from the reader, so there is
        nothing left to hand a sandbox an identity that means nothing here.
        """
        write_record(state_root, target="va", generation=7)
        env: dict[str, str] = {}

        assert host_executor._apply_target_stamp(env) == "va"
        assert not hasattr(host_executor, "ENV_CONTROL_TARGET_STATE_PID")
        assert "OSPREY_CONTROL_TARGET_STATE_PID" not in env

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
        write_record(state_root, target="live", generation=1)
        env: dict[str, str] = {}

        assert host_executor._apply_target_stamp(env) == host_executor.CONTROL_TARGET_BASELINE
        for name in host_executor._STAMP_ENV_NAMES:
            assert name not in env

    def test_va_is_stamped_on_that_same_deployment(self, state_root, monkeypatch):
        """Only the unresolvable half is declined: 'va' resolves everywhere."""
        monkeypatch.setattr(
            "osprey_connectors.config.get_config_value", _section_reader(MOCK_ONLY_SECTION)
        )
        write_record(state_root, target="va", generation=1)
        env: dict[str, str] = {}

        assert host_executor._apply_target_stamp(env) == "va"
        assert env[host_executor.ENV_CONTROL_TARGET] == "va"


class TestLaunchPostureStamp:
    """The second thing a launch pins: the posture, beside the target.

    The routing stamp says WHICH machine the sandbox talks to; this one says
    what the deployment allowed doing to it at the moment the run started. It
    is stamped on both paths — including the one that removes every routing
    name — because an unstamped run is the one whose target is unknowable, and
    that is the case the pin must cover most restrictively rather than least.
    The rule it feeds is exercised in
    ``tests/services/python_executor/test_launch_posture_pin.py``; here it is
    only that the executor stamps it, and stamps it every time.
    """

    def test_the_posture_is_stamped_beside_the_target(self, state_root, deployment_config):
        write_record(state_root, target="va", generation=7)
        env: dict[str, str] = {}

        assert host_executor._apply_target_stamp(env) == "va"
        assert env[host_executor.ENV_LAUNCH_POSTURE] == "va=writes"

    def test_a_narrowed_target_is_stamped_sandboxed(self, state_root, deployment_config):
        write_record(state_root, target="va", generation=7, posture={"va": "sandbox"})
        env: dict[str, str] = {}

        assert host_executor._apply_target_stamp(env) == "va"
        assert env[host_executor.ENV_LAUNCH_POSTURE] == "va=sandbox"

    def test_an_unstamped_run_is_still_pinned(self, state_root, monkeypatch):
        """A target the run cannot be placed on: routing names go, the pin stays.

        It names every target, because a run that cannot say which machine it is
        about must not be the one run a narrowing fails to reach.
        """
        monkeypatch.setattr(
            "osprey_connectors.config.get_config_value", _section_reader(MOCK_ONLY_SECTION)
        )
        write_record(state_root, target="live", generation=1, posture={"live": "sandbox"})
        env: dict[str, str] = {}

        assert host_executor._apply_target_stamp(env) == host_executor.CONTROL_TARGET_BASELINE
        for name in host_executor._STAMP_ENV_NAMES:
            assert name not in env
        assert env[host_executor.ENV_LAUNCH_POSTURE] == "*=sandbox"

    def test_an_inherited_posture_pin_is_overwritten_not_trusted(
        self, state_root, deployment_config
    ):
        """A stale value in the parent's environment must not survive the launch.

        The routing names are POPPED for the same reason; this one is always
        assigned instead, so a ``writes`` inherited from anywhere cannot outlive
        the record's actual answer for this run.
        """
        write_record(state_root, target="va", generation=1, posture={"va": "sandbox"})
        env = {host_executor.ENV_LAUNCH_POSTURE: "va=writes"}

        host_executor._apply_target_stamp(env)

        assert env[host_executor.ENV_LAUNCH_POSTURE] == "va=sandbox"


class TestConvergenceGate:
    """A run is admitted only once the fleet has settled on the record.

    The two PIDs are this process and its parent, because a report is kept only
    while its server is alive and these are the two PIDs a test can be sure
    of. ``SESSION_A``/``SESSION_B`` are the sessions those two servers belong
    to; which one this executor is in is set through ``OSPREY_POSTURE_SESSION``,
    exactly as a real deployment sets it.
    """

    SESSION_A = "session-a"
    SESSION_B = "session-b"

    @staticmethod
    def _pids() -> tuple[int, int]:
        return os.getpid(), os.getppid()

    def test_a_server_applying_this_generation_refuses_every_session(
        self, state_root, deployment_config, monkeypatch
    ):
        """A swap in flight stops the deployment, not just the session running it."""
        server_a, _ = self._pids()
        write_record(state_root, target="va", generation=4)
        write_report(state_root, server_a, session=self.SESSION_A, last_switch=applying(4))
        monkeypatch.setenv("OSPREY_POSTURE_SESSION", self.SESSION_B)
        env: dict[str, str] = {}

        with pytest.raises(host_executor._SwitchInProgress) as excinfo:
            host_executor._apply_target_stamp(env)

        assert excinfo.value.pids == (server_a,)
        assert f"switch_in_progress:{server_a}" in str(excinfo.value)
        # Nothing was stamped: the run is refused, not routed somewhere else.
        assert host_executor.ENV_CONTROL_TARGET not in env

    def test_the_refusal_names_every_blocking_server(
        self, state_root, deployment_config, monkeypatch
    ):
        """An operator has to be told which processes to wait for, not one of them."""
        server_a, server_b = self._pids()
        write_record(state_root, target="va", generation=4)
        write_report(state_root, server_a, session=self.SESSION_A, last_switch=applying(4))
        write_report(state_root, server_b, session=self.SESSION_B, last_switch=applying(4))
        monkeypatch.setenv("OSPREY_POSTURE_SESSION", self.SESSION_B)

        with pytest.raises(host_executor._SwitchInProgress) as excinfo:
            host_executor._apply_target_stamp({})

        assert excinfo.value.pids == tuple(sorted((server_a, server_b)))

    def test_a_null_bound_fresh_report_does_not_block(
        self, state_root, deployment_config, monkeypatch
    ):
        """A server that has not launched a child yet is not "somewhere else".

        Its binding is null because it has nothing to bind, and a first launch
        is exactly what it is waiting for. Reading null as "not on the record"
        would make a fresh server refuse its own session's first run.
        """
        server_a, _ = self._pids()
        write_record(state_root, target="va", generation=4)
        write_report(state_root, server_a, session=self.SESSION_A)
        monkeypatch.setenv("OSPREY_POSTURE_SESSION", self.SESSION_A)
        env: dict[str, str] = {}

        assert host_executor._apply_target_stamp(env) == "va"
        assert env[host_executor.ENV_CONTROL_TARGET_GENERATION] == "4"

    def test_a_failed_report_refuses_only_its_own_session(
        self, state_root, deployment_config, monkeypatch
    ):
        """SC-81: A failed at this generation, B applied. Only A's runs stop.

        That is the whole cost of a failed swap — the session whose server could
        not follow is refused, and every other session on the deployment carries
        on.
        """
        server_a, server_b = self._pids()
        write_record(state_root, target="va", generation=4)
        write_report(state_root, server_a, session=self.SESSION_A, last_switch=failed(4))
        write_report(
            state_root,
            server_b,
            session=self.SESSION_B,
            applied_target="va",
            applied_generation=4,
        )

        monkeypatch.setenv("OSPREY_POSTURE_SESSION", self.SESSION_B)
        assert host_executor._apply_target_stamp({}) == "va"

        monkeypatch.setenv("OSPREY_POSTURE_SESSION", self.SESSION_A)
        with pytest.raises(host_executor._SwitchInProgress) as excinfo:
            host_executor._apply_target_stamp({})
        assert excinfo.value.pids == (server_a,)

    def test_a_session_less_client_is_not_blocked_by_another_sessions_failure(
        self, state_root, deployment_config, monkeypatch
    ):
        """SC-81: a bare ``claude`` owns no report, so only a live swap stops it."""
        server_a, _ = self._pids()
        write_record(state_root, target="va", generation=4)
        write_report(state_root, server_a, session=self.SESSION_A, last_switch=failed(4))
        monkeypatch.delenv("OSPREY_POSTURE_SESSION", raising=False)

        assert host_executor._apply_target_stamp({}) == "va"

    def test_a_session_less_client_is_still_blocked_by_a_live_swap(
        self, state_root, deployment_config, monkeypatch
    ):
        """The other half of the same rule: owning no report is not an exemption."""
        server_a, _ = self._pids()
        write_record(state_root, target="va", generation=4)
        write_report(state_root, server_a, session=self.SESSION_A, last_switch=applying(4))
        monkeypatch.delenv("OSPREY_POSTURE_SESSION", raising=False)

        with pytest.raises(host_executor._SwitchInProgress):
            host_executor._apply_target_stamp({})

    def test_a_swap_at_another_generation_does_not_block(
        self, state_root, deployment_config, monkeypatch
    ):
        """The generation is the only thing the fleet coordinates on."""
        server_a, _ = self._pids()
        write_record(state_root, target="va", generation=4)
        write_report(
            state_root,
            server_a,
            session=self.SESSION_A,
            applied_target="va",
            applied_generation=4,
            last_switch=applying(3),
        )
        monkeypatch.setenv("OSPREY_POSTURE_SESSION", self.SESSION_A)

        assert host_executor._apply_target_stamp({}) == "va"

    def test_no_record_is_not_gated(self, state_root, deployment_config, monkeypatch):
        """With nothing to converge on there is nothing to wait for.

        A deployment that has never written a record runs unstamped on the
        baseline, and a report left over from one that has must not turn that
        into a refusal nobody can clear.
        """
        server_a, _ = self._pids()
        write_report(state_root, server_a, session=self.SESSION_A, last_switch=applying(4))
        monkeypatch.setenv("OSPREY_POSTURE_SESSION", self.SESSION_A)

        assert host_executor._apply_target_stamp({}) == host_executor.CONTROL_TARGET_BASELINE

    def test_an_unreadable_fleet_admits_the_run(self, state_root, deployment_config, monkeypatch):
        """Not being able to ask is not the same as being told to stop.

        Refusing here would make one unreadable directory refuse every execution
        on the deployment, and the guarantee that a run cannot write to a machine
        nobody selected is the generation pin inside the sandbox, not this gate.
        """

        def explode(*args, **kwargs):
            raise OSError("state directory is unreadable")

        write_record(state_root, target="va", generation=4)
        monkeypatch.setattr(control_context, "live_reports", explode)

        assert host_executor._apply_target_stamp({}) == "va"


class TestExecuteViaLocalStamping:
    """End-to-end through ``_execute_via_local``, with the subprocess faked out."""

    @staticmethod
    def _run(tmp_path, monkeypatch, *, spawn=True) -> tuple[dict[str, str], Any]:
        """Run the adapter against a fake subprocess; return (env, result)."""
        captured: dict[str, dict[str, str]] = {}

        class _FakeProc:
            returncode = 0

            async def communicate(self):
                return b"", b""

        async def fake_exec(*args, **kwargs):
            captured["env"] = kwargs["env"]
            return _FakeProc()

        async def refuse_exec(*args, **kwargs):
            raise AssertionError("the sandbox must not be spawned")

        folder = tmp_path / "execution"
        (folder / "figures").mkdir(parents=True)

        monkeypatch.setattr(host_executor, "_resolve_project_root", lambda: tmp_path)
        monkeypatch.setattr(
            host_executor, "resolve_agent_interpreter", lambda root=None: "/bin/true"
        )
        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec if spawn else refuse_exec)

        result = asyncio.run(
            host_executor._execute_via_local(
                "print('hello')",
                "readonly",
                {"timeout": 5},
                folder,
            )
        )
        return captured.get("env", {}), result

    def test_sandbox_env_carries_the_stamp(
        self, state_root, deployment_config, tmp_path, monkeypatch
    ):
        write_record(state_root, target="va", generation=5)

        env, result = self._run(tmp_path, monkeypatch)

        assert env[host_executor.ENV_CONTROL_TARGET] == "va"
        assert env[host_executor.ENV_CONTROL_TARGET_GENERATION] == "5"
        assert "OSPREY_CONTROL_TARGET_STATE_PID" not in env
        # The mode injection this stamp sits beside must survive untouched.
        assert env["OSPREY_EXECUTION_MODE"] == "readonly"
        assert result.control_target == "va"

    def test_sandbox_env_carries_the_launch_posture(
        self, state_root, deployment_config, tmp_path, monkeypatch
    ):
        """The pin reaches the child, and the marker states the same thing.

        Both halves of FR15 through the real launch path: the sandbox reads the
        stamp back through ``posture_store``, and the in-flight marker is what
        the posture route consults before it agrees to widen anything.
        """
        # Arrange
        write_record(state_root, target="va", generation=5, posture={"va": "sandbox"})
        seen: list[list[dict[str, Any]]] = []
        real_marker = host_executor._in_flight_marker

        @contextlib.contextmanager
        def watching_marker(control_target, launch_posture=None):
            with real_marker(control_target, launch_posture):
                seen.append(markers_in(state_root))
                yield

        monkeypatch.setattr(host_executor, "_in_flight_marker", watching_marker)

        # Act
        env, _ = self._run(tmp_path, monkeypatch)

        # Assert
        assert env[host_executor.ENV_LAUNCH_POSTURE] == "va=sandbox"
        assert [marker["launch_posture"] for marker in seen[0]] == ["va=sandbox"]

    def test_unstamped_run_records_the_baseline(
        self, state_root, deployment_config, tmp_path, monkeypatch
    ):
        env, result = self._run(tmp_path, monkeypatch)

        for name in host_executor._STAMP_ENV_NAMES:
            assert name not in env
        assert result.control_target == host_executor.CONTROL_TARGET_BASELINE

    def test_a_switch_in_flight_fails_the_run_without_spawning(
        self, state_root, deployment_config, tmp_path, monkeypatch
    ):
        """The refusal is a result, not a traceback, and nothing was executed.

        The submitted code is not at fault and did not run, so the failure is
        classed on its own kind and names the server to wait for.
        """
        write_record(state_root, target="va", generation=4)
        write_report(state_root, os.getpid(), session="session-a", last_switch=applying(4))
        monkeypatch.setenv("OSPREY_POSTURE_SESSION", "session-a")

        _, result = self._run(tmp_path, monkeypatch, spawn=False)

        assert result.success is False
        assert result.failure_kind == host_executor.FAILURE_KIND_SWITCH_IN_PROGRESS
        assert f"switch_in_progress:{os.getpid()}" in result.stderr
        assert result.error_message == result.stderr
        assert result.control_target == host_executor.CONTROL_TARGET_BASELINE
        # No marker was written, because no run was admitted to write one for.
        assert markers_in(state_root) == []

    def test_result_default_is_the_baseline(self):
        """A result built without a target — a setup failure — claims nothing."""
        result = host_executor.ExecutionResult(success=False, stdout="", stderr="")

        assert result.control_target == host_executor.CONTROL_TARGET_BASELINE


# ---------------------------------------------------------------------------
# Sandbox side: routing the connector from the stamp
# ---------------------------------------------------------------------------


class _FakeConnector:
    """Records the config block the factory handed ``connect()``, and its end."""

    last_config: dict[str, Any] | None = None
    #: Every instance that was disconnected, in the order it happened.
    disconnected: list["_FakeConnector"] = []

    async def connect(self, config: dict[str, Any]) -> None:
        type(self).last_config = config

    async def disconnect(self) -> None:
        type(self).disconnected.append(self)


@pytest.fixture
def fake_registry(deployment_config):
    """Register the fake connector under every type this deployment can select."""
    from osprey_connectors.factory import ConnectorFactory, isolated_connector_registries

    with isolated_connector_registries():
        for name in ("mock", "epics", "virtual_accelerator"):
            ConnectorFactory.register_control_system(name, _FakeConnector)
        _FakeConnector.last_config = None
        _FakeConnector.disconnected = []
        yield
        _FakeConnector.last_config = None
        _FakeConnector.disconnected = []


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


class TestConnectorRebuild:
    """A kernel is re-stamped every cell, so the connector follows the stamp.

    A sandbox never reaches these: it is stamped once and dies with the run.
    The kernel outlives every switch made under it, and a connector built for
    the old target would keep talking to the old machine's gateways.
    """

    def test_the_same_stamp_reuses_the_connector(
        self, monkeypatch, fake_registry, clear_runtime_state
    ):
        """Building once is what makes the write pin, and not a reconnect, the rule."""
        stamp_env(monkeypatch, target="va", generation="3")

        import osprey.runtime as runtime

        first = asyncio.run(runtime._get_connector())
        second = asyncio.run(runtime._get_connector())

        assert first is second
        assert _FakeConnector.disconnected == []

    def test_a_moved_target_rebuilds_on_the_new_one(
        self, monkeypatch, fake_registry, clear_runtime_state
    ):
        """One disconnect of the old connector, and the new block is read."""
        stamp_env(monkeypatch, target="va", generation="3")

        import osprey.runtime as runtime

        first = asyncio.run(runtime._get_connector())
        stamp_env(monkeypatch, target="live", generation="4")
        second = asyncio.run(runtime._get_connector())

        assert second is not first
        assert _FakeConnector.disconnected == [first]
        # 1.0 is the real machine's block; 9.0 would be the VA's.
        assert _FakeConnector.last_config == {"timeout": 1.0}

    def test_a_moved_generation_alone_rebuilds_too(
        self, monkeypatch, fake_registry, clear_runtime_state
    ):
        """The same machine at a new generation is still a switch that landed."""
        stamp_env(monkeypatch, target="va", generation="3")

        import osprey.runtime as runtime

        first = asyncio.run(runtime._get_connector())
        stamp_env(monkeypatch, target="va", generation="4")
        second = asyncio.run(runtime._get_connector())

        assert second is not first
        assert _FakeConnector.disconnected == [first]

    def test_the_rebuild_does_not_wait_on_its_own_lock(
        self, monkeypatch, fake_registry, clear_runtime_state
    ):
        """``_connector_lock`` is not reentrant, so the rebuild disconnects locked.

        A rebuild routed through ``cleanup_runtime`` would take the lock it is
        already holding and never return, which is a hang rather than a failure
        — hence the bound.
        """
        stamp_env(monkeypatch, target="va", generation="3")

        import osprey.runtime as runtime

        asyncio.run(runtime._get_connector())
        stamp_env(monkeypatch, target="live", generation="4")

        connector = asyncio.run(asyncio.wait_for(runtime._get_connector(), timeout=10))

        assert connector is runtime._runtime_connector


# ---------------------------------------------------------------------------
# Sandbox side: the write pin
# ---------------------------------------------------------------------------


class TestWritePin:
    """Writes refuse once the deployment's target or generation moves."""

    def test_matching_generation_lets_the_write_through(
        self, state_root, monkeypatch, clear_runtime_state
    ):
        write_record(state_root, target="va", generation=3)
        stamp_env(monkeypatch, target="va", generation="3")

        import osprey.runtime as runtime

        runtime._assert_target_pin()  # does not raise

    def test_moved_generation_refuses_and_names_both(
        self, state_root, monkeypatch, clear_runtime_state
    ):
        write_record(state_root, target="va", generation=4)
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
        write_record(state_root, target="live", generation=3)
        stamp_env(monkeypatch, target="va", generation="3")

        import osprey.runtime as runtime

        with pytest.raises(ControlTargetChangedError) as excinfo:
            runtime._assert_target_pin()

        assert "'va'" in str(excinfo.value)
        assert "'live'" in str(excinfo.value)

    def test_stamped_but_no_record_refuses(self, state_root, monkeypatch, clear_runtime_state):
        """No record: the current generation is unknowable, so the write fails closed."""
        stamp_env(monkeypatch, target="va", generation="3")

        import osprey.runtime as runtime

        with pytest.raises(ControlTargetChangedError, match="missing or unreadable"):
            runtime._assert_target_pin()

    def test_stamped_but_generation_unparseable_refuses(
        self, state_root, monkeypatch, clear_runtime_state
    ):
        write_record(state_root, target="va", generation=3)
        stamp_env(monkeypatch, target="va", generation="not-a-number")

        import osprey.runtime as runtime

        with pytest.raises(ControlTargetChangedError, match="generation unknown"):
            runtime._assert_target_pin()

    def test_the_pin_does_not_evaluate_convergence(
        self, state_root, monkeypatch, clear_runtime_state
    ):
        """A swap in flight does not retro-refuse a process already holding a connector.

        Convergence gates ADMISSION — the executor's stamp, the kernel's cell
        gate — and this process was admitted. Its connector is still bound to
        the gateways of the target it started on, and the record still names
        that target at that generation, so the write goes where the stamp says.
        """
        write_record(state_root, target="va", generation=3)
        write_report(state_root, os.getpid(), session="session-a", last_switch=applying(3))
        monkeypatch.setenv("OSPREY_POSTURE_SESSION", "session-a")
        stamp_env(monkeypatch, target="va", generation="3")

        import osprey.runtime as runtime

        runtime._assert_target_pin()  # does not raise

    def test_unstamped_process_is_not_pinned(self, state_root, clear_stamp, clear_runtime_state):
        """Baseline routing claimed no target, so there is nothing to drift from."""
        write_record(state_root, target="va", generation=99)

        import osprey.runtime as runtime

        runtime._assert_target_pin()  # does not raise

    def test_write_channel_refuses_before_touching_the_connector(
        self, state_root, monkeypatch, fake_registry, clear_runtime_state
    ):
        """The refusal happens on the write path itself, not only in the helper."""
        write_record(state_root, target="va", generation=4)
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
        write_record(state_root, target="va", generation=4)
        stamp_env(monkeypatch, target="va", generation="3")

        import osprey.runtime as runtime

        class _Reader:
            async def read_channel(self, channel_address, **kwargs):
                class _Value:
                    value = 42.0

                return _Value()

        runtime._runtime_connector = _Reader()
        # The stamp this stand-in stands for: a connector whose stamp does not
        # match the environment is one the ground moved under, and is rebuilt.
        runtime._connector_stamp = ("va", 3)

        assert runtime.read_channel("TEST:PV") == 42.0
