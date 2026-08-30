"""One target read feeds both the sandbox stamp and the embedded limits policy.

The executor used to answer the same question twice. ``execute_code`` built the
limits validator up front, before anything about the session's control target
was known, and ``_execute_via_local`` resolved the stamp that routes the sandbox
later, from its own read of the controls server's target record. That was
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
import os

import pytest

from osprey.mcp_server.python_executor import executor as host_executor
from osprey_connectors.control_system.limits_validator import LimitsValidator

pytestmark = pytest.mark.unit


def _record(target: str) -> dict:
    """A target-state record for this session, naming *target*."""
    return {
        "target": target,
        "generation": 3,
        "server_pid": os.getpid(),
        "owner_ppid": os.getppid(),
    }


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

    def fake_record() -> dict:
        reads["n"] += 1
        return _record("va" if reads["n"] == 1 else "live")

    def fake_from_config(*, connector_type=None, target=None):
        targets.append(target)
        return _validator_for(target)

    monkeypatch.setattr(host_executor, "_session_target_record", fake_record)
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
