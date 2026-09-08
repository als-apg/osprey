"""One target read feeds both the sandbox stamp and the embedded limits policy.

The executor used to answer the same question twice. ``execute_code`` built the
limits validator up front, before anything about the deployment's control target
was known, and ``_execute_via_local`` resolved the stamp that routes the sandbox
later, from its own read of the deployment's control-context record. That was
harmless while the limits posture was deployment-wide and the same for every
machine. Once it is per target it is not: a switch landing between the two reads
would embed one machine's policy into a sandbox stamped for another, and the run
would enforce the simulator's relaxed posture against the live machine or the
reverse.

So the record is read once, and the target it answers feeds both. These tests
make a second read *visible*: the patched record answers ``va`` the first time
and ``live`` every time after, so a run that reads twice disagrees with itself
instead of quietly passing.
"""

import asyncio

import pytest

from osprey.mcp_server.python_executor import executor as host_executor
from osprey_connectors import control_context
from osprey_connectors.control_system import limits_validator
from osprey_connectors.control_system.limits_validator import LimitsValidator

pytestmark = pytest.mark.unit


def _record(target: str) -> control_context.ControlContext:
    """The deployment's control-context record, naming *target*."""
    return control_context.ControlContext(target=target, generation=3)


def _validator_for(target: str | None) -> LimitsValidator:
    """A validator whose policy says out loud which target it was built for."""
    return LimitsValidator(
        {},
        {"allow_unlisted_channels": True, "allow_unlisted_key": f"resolved-for:{target}"},
    )


class _LocalRun:
    """What one faked ``_execute_via_local`` run left behind."""

    def __init__(self, env: dict[str, str], script: str, reads: int, targets: list[str | None]):
        self.env = env
        self.script = script
        self.reads = reads
        self.targets = targets


def _run_local(tmp_path, monkeypatch) -> _LocalRun:
    """Drive one local execution with the subprocess and the config faked out.

    The record reader is the seam under test, so it is the one thing that
    answers differently on a second call.
    """
    captured: dict[str, dict[str, str]] = {}
    reads = {"n": 0}
    targets: list[str | None] = []

    class _FakeProc:
        returncode = 0

        async def communicate(self):
            return b"", b""

    async def fake_exec(*args, **kwargs):
        captured["env"] = kwargs["env"]
        return _FakeProc()

    def fake_record() -> control_context.ControlContext:
        reads["n"] += 1
        return _record("va" if reads["n"] == 1 else "live")

    def fake_from_config(*, connector_type=None, target=None):
        targets.append(target)
        return _validator_for(target)

    # _execute_via_local writes its in-flight marker under target_state.state_dir(),
    # which resolves through the stamped agent-data root — without this the marker
    # directory lands in the repository and trips the agent-data guard.
    monkeypatch.setenv("OSPREY_AGENT_DATA_ROOT", str(tmp_path / "agent_data"))
    monkeypatch.setattr(host_executor, "_deployment_record", fake_record)
    # Resolvability is a config question, answered elsewhere and pinned in
    # tests/runtime/test_executor_target_stamp.py; here it only has to not
    # send the run to the baseline.
    monkeypatch.setattr(host_executor, "_target_is_resolvable", lambda target: True)
    monkeypatch.setattr(LimitsValidator, "from_config", fake_from_config)
    monkeypatch.setattr(host_executor, "_resolve_project_root", lambda: tmp_path)
    monkeypatch.setattr(host_executor, "resolve_agent_interpreter", lambda root=None: "/bin/true")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    folder = tmp_path / "execution"
    (folder / "figures").mkdir(parents=True)

    asyncio.run(
        host_executor._execute_via_local("print('hi')", "readwrite", {"timeout": 5}, folder)
    )
    return _LocalRun(
        captured["env"],
        (folder / "wrapped_script.py").read_text(encoding="utf-8"),
        reads["n"],
        targets,
    )


class TestOneRecordReadPerRun:
    """The stamp and the embedded policy come from the same answer."""

    def test_policy_and_stamp_name_the_same_target(self, tmp_path, monkeypatch):
        run = _run_local(tmp_path, monkeypatch)

        assert run.env[host_executor.ENV_CONTROL_TARGET] == "va"
        assert "resolved-for:va" in run.script
        # The record's second answer must not be anywhere in the run: if it is,
        # the two halves were resolved from two different reads.
        assert "resolved-for:live" not in run.script

    def test_the_record_is_read_once(self, tmp_path, monkeypatch):
        run = _run_local(tmp_path, monkeypatch)

        assert run.reads == 1

    def test_the_validator_is_built_for_the_stamped_target(self, tmp_path, monkeypatch):
        run = _run_local(tmp_path, monkeypatch)

        assert run.targets == ["va"]


class TestLoadLimitsValidator:
    """The helper threads a target through and stays honest about caller bugs."""

    def test_target_is_passed_through(self, monkeypatch):
        seen: list[dict] = []

        def fake_from_config(*, connector_type=None, target=None):
            seen.append({"connector_type": connector_type, "target": target})
            return None

        monkeypatch.setattr(LimitsValidator, "from_config", fake_from_config)

        assert host_executor._load_limits_validator(target="live") is None
        assert seen == [{"connector_type": None, "target": "live"}]

    def test_type_error_is_not_swallowed(self, monkeypatch):
        """A caller bug must surface as one, not as "limits checking is off".

        ``from_config`` raises ``TypeError`` before reading any config when it is
        handed both a connector type and a target. Swallowing that here would
        turn a mis-wired call site into a silently unvalidated sandbox.
        """

        def boom(*, connector_type=None, target=None):
            raise TypeError("takes connector_type or target, not both")

        monkeypatch.setattr(LimitsValidator, "from_config", boom)

        with pytest.raises(TypeError):
            host_executor._load_limits_validator(target="va")

    @pytest.mark.parametrize("exc", [FileNotFoundError, KeyError, RuntimeError, ImportError])
    def test_config_unavailable_is_none(self, monkeypatch, exc):
        """The errors ``from_config`` documents as "config unavailable" disable checking."""

        def boom(*, connector_type=None, target=None):
            raise exc("nope")

        monkeypatch.setattr(LimitsValidator, "from_config", boom)

        assert host_executor._load_limits_validator(target=None) is None


class TestStepReadTimeout:
    """The ``max_step`` fresh-read budget follows the same target as the policy.

    The budget bounds a read the embedded policy makes before a write. It is
    authored per connector block, beside that connector's own ``timeout``, so
    resolving it from the deployment's baseline type would hand a run that
    switched targets one connector's budget while its connector reads with
    another's — the disagreement between the script's check and the connector's
    that the key exists to prevent.
    """

    @staticmethod
    def _config(section):
        """Patch ``get_config_value`` to serve one ``control_system`` section."""

        def get_config_value(key, default=None):
            return section if key == "control_system" else default

        return get_config_value

    #: A deployment baselined on its simulator that also describes a real
    #: machine — the shape a target switch exists for, and the one where a
    #: baseline read and a target read disagree.
    SWITCHABLE = {
        "type": "virtual_accelerator",
        "connector": {
            "virtual_accelerator": {"step_read_timeout_s": 9.0},
            "epics": {"step_read_timeout_s": 0.5},
        },
    }

    def test_the_budget_comes_from_the_stamped_targets_block(self, monkeypatch):
        """A run switched onto the live machine reads with the live block's value."""
        monkeypatch.setattr(
            "osprey_connectors.config.get_config_value", self._config(self.SWITCHABLE)
        )

        assert host_executor._step_read_timeout_seconds("live") == 0.5
        assert host_executor._step_read_timeout_seconds("va") == 9.0

    def test_the_baseline_reads_the_deployments_own_block(self, monkeypatch):
        """A target that names no machine here gets the deployment-wide reading."""
        monkeypatch.setattr(
            "osprey_connectors.config.get_config_value", self._config(self.SWITCHABLE)
        )

        assert (
            host_executor._step_read_timeout_seconds(host_executor.CONTROL_TARGET_BASELINE) == 9.0
        )

    def test_a_block_declaring_none_takes_the_packages_default(self, monkeypatch):
        """A deployment that never authored the key still bounds the read."""
        monkeypatch.setattr(
            "osprey_connectors.config.get_config_value",
            self._config({"type": "epics", "connector": {"epics": {}}}),
        )

        assert host_executor._step_read_timeout_seconds("live") == (
            limits_validator.DEFAULT_STEP_READ_TIMEOUT_SECONDS
        )

    def test_a_config_that_cannot_be_read_still_bounds_the_read(self, monkeypatch):
        """Config that will not load must not be what takes the bound off the read."""

        def explode(key, default=None):
            raise RuntimeError("no config here")

        monkeypatch.setattr("osprey_connectors.config.get_config_value", explode)

        assert host_executor._step_read_timeout_seconds("live") == (
            limits_validator.DEFAULT_STEP_READ_TIMEOUT_SECONDS
        )

    def test_the_stamped_target_is_what_the_sandbox_is_built_with(self, tmp_path, monkeypatch):
        """The run resolves the budget once, for the target it stamped.

        The wiring, not the resolution: the generated source carries the number
        the helper answered for the stamped target, so a sandbox cannot end up
        holding a budget resolved for a machine it was not pointed at.
        """
        asked: list[str] = []

        def fake_budget(target: str) -> float:
            asked.append(target)
            return 0.125

        monkeypatch.setattr(host_executor, "_step_read_timeout_seconds", fake_budget)

        run = _run_local(tmp_path, monkeypatch)

        assert asked == ["va"]
        assert "_step_read_timeout = 0.125" in run.script
