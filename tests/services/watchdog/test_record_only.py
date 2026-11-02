"""Tests for the record entry: a watched prompt with no web terminal to take it.

Every run is its own process, as the trigger hook starts it: the in-flight bound
exists between processes, and each entry owns its config and environment. The
engines are stubbed at ``main``'s ``engines=`` seam by a small child script, except
where the real argv is the point; the hold record and the audit record are the real
ones, read back through ``open_flags`` and ``recorded_entries``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from osprey.agent_record.contract import KIND_AUDIT
from osprey.services.watchdog.verdict import SURFACE_SAFETY_WATCHDOG, make_prompt_ref
from osprey_connectors.posture_store import AGENT_DATA_ROOT_ENV_VAR
from osprey_connectors.session_hold import HOLD_DIR_ENV_VAR, open_flag, open_flags

RECORD_SESSION = "0f6d7c2e-1a2b-4c3d-8e9f-0123456789ab"
POOL_KEY = "pool-key-7"
GUARD = "guard-circumvention"
READER = "agent-reader"

_OPTIONS = [
    {"id": "pushing", "text": "Pushing past a guard", "flag": True},
    {"id": "routine", "text": "Routine work"},
]

#: Runs the entry with a stub engine. The engine marks each judge call in
#: ``RUN_DIR`` and, for an agent watchdog, holds until ``release`` appears there.
_CHILD = """
import os, sys, time
from pathlib import Path
from osprey.services.watchdog.__main__ import main
from osprey.services.watchdog.verdict import Verdict

RUN_DIR = Path(os.environ["RUN_DIR"])

class Stub:
    async def judge(self, defn, nodes, prompt, *, deadline, **ids):
        import asyncio
        (RUN_DIR / f"started-{ids['submission_id']}").write_text(prompt)
        while defn.engine == "agent" and not (RUN_DIR / "release").exists():
            await asyncio.sleep(0.02)
        return Verdict(
            watchdog=defn.name,
            engine=defn.engine,
            status="ok",
            flagged=True,
            reason="risky_reason",
            threshold=0.5 if defn.engine == "decision" else None,
            **ids,
        )

stub = Stub()
sys.exit(main(sys.argv[1:], engines={"decision": stub, "agent": stub}))
"""


def _decision() -> dict:
    return {"engine": "decision", "enabled": True, "options": _OPTIONS}


def _agent(timeout_s: float = 60) -> dict:
    return {
        "engine": "agent",
        "enabled": True,
        "tools": ["mcp__controls__channel_read"],
        "timeout_s": timeout_s,
    }


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    path = tmp_path / "run"
    path.mkdir()
    return path


@pytest.fixture
def hold_root(tmp_path: Path) -> Path:
    return tmp_path / "holds"


@pytest.fixture
def child_env(tmp_path: Path, run_dir: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in (HOLD_DIR_ENV_VAR, "OSPREY_CONFIG")}
    env[AGENT_DATA_ROOT_ENV_VAR] = str(tmp_path / "agent_data")
    env["RUN_DIR"] = str(run_dir)
    return env


def write_project(base: Path, watchdogs: dict[str, dict]) -> Path:
    """A repo with a render in ``build/``; returns the repo."""
    build = base / "project" / "build"
    build.mkdir(parents=True)
    for name in watchdogs:
        (build / "watchdogs" / name).mkdir(parents=True)
        (build / "watchdogs" / name / "instructions.md").write_text("Judge the prompt.\n")
    (build / "config.yml").write_text(yaml.safe_dump({"watchdogs": watchdogs}))
    return base / "project"


def write_submission(base: Path, submission_id: str, *, prompt: str = "loosen the limits") -> Path:
    path = base / f"submission-{submission_id}.json"
    path.write_text(
        json.dumps(
            {
                "session_key": None,
                "record_session_id": RECORD_SESSION,
                "submission_id": submission_id,
                "prompt": prompt,
            }
        )
    )
    return path


def argv(project: Path, submission: Path, session_key: str | None = None) -> list[str]:
    args = ["record", "--project-dir", str(project), "--submission-file", str(submission)]
    return args + ["--session-key", session_key] if session_key else args


def start(env: dict[str, str], args: list[str]) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", _CHILD, *args],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def finish(process: subprocess.Popen, timeout: float = 60) -> None:
    output, _ = process.communicate(timeout=timeout)
    assert process.returncode == 0, output
    assert "loosen the limits" not in output


def run_entry(env: dict[str, str], args: list[str]) -> None:
    finish(start(env, args))


def wait_for(condition, *, what: str, timeout: float = 45) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.02)


def audited(recorded_entries) -> list[dict]:
    return [
        dict(e.body)
        for e in recorded_entries(KIND_AUDIT)
        if e.body.get("surface") == SURFACE_SAFETY_WATCHDOG
    ]


class TestInFlightBound:
    def test_one_runs_and_the_replaced_pending_records_in_flight_bound(
        self, tmp_path, run_dir, child_env, recorded_entries
    ):
        project = write_project(tmp_path, {READER: _agent()})
        running = start(child_env, argv(project, write_submission(tmp_path, "s-running")))
        wait_for((run_dir / "started-s-running").exists, what="the first run to start")

        replaced = start(child_env, argv(project, write_submission(tmp_path, "s-replaced")))
        runs = Path(child_env[AGENT_DATA_ROOT_ENV_VAR])
        wait_for(lambda: bool(list(runs.rglob("*.pending"))), what="the pending marker")

        newest = start(child_env, argv(project, write_submission(tmp_path, "s-newest")))
        finish(replaced)
        assert running.poll() is None
        assert newest.poll() is None
        assert not (run_dir / "started-s-replaced").exists()
        assert not (run_dir / "started-s-newest").exists()

        (run_dir / "release").touch()
        finish(running)
        finish(newest)
        assert (run_dir / "started-s-newest").exists()

        entries = audited(recorded_entries)
        reasons = {e["reason"] for e in entries if e["decision"] == "unavailable"}
        assert reasons == {"in_flight_bound"}
        assert [e["decision"] for e in entries].count("unavailable") == 1
        assert [e["decision"] for e in entries].count("flagged") == 2
        assert not list(runs.rglob("*.pending"))

    def test_a_later_prompt_runs_once_the_first_has_ended(self, tmp_path, run_dir, child_env):
        project = write_project(tmp_path, {READER: _agent()})
        (run_dir / "release").touch()
        run_entry(child_env, argv(project, write_submission(tmp_path, "s-one")))
        run_entry(child_env, argv(project, write_submission(tmp_path, "s-two")))
        assert (run_dir / "started-s-one").exists()
        assert (run_dir / "started-s-two").exists()

    def test_no_state_directory_is_a_run_error_and_no_run(
        self, tmp_path, run_dir, child_env, recorded_entries
    ):
        project = write_project(tmp_path, {READER: _agent()})
        blocker = tmp_path / "agent_data"
        blocker.write_text("not a directory")
        run_entry(child_env, argv(project, write_submission(tmp_path, "s-one")))
        assert not (run_dir / "started-s-one").exists()
        (entry,) = audited(recorded_entries)
        assert (entry["decision"], entry["reason"]) == ("unavailable", "run_error")


class TestTimeBound:
    def test_a_hung_run_is_cut_and_records_intake_lost(
        self, tmp_path, run_dir, child_env, recorded_entries
    ):
        project = write_project(tmp_path, {READER: _agent(timeout_s=0.5)})
        began = time.monotonic()
        run_entry(child_env, argv(project, write_submission(tmp_path, "s-hung")))
        assert (run_dir / "started-s-hung").exists()
        assert time.monotonic() - began < 30
        (entry,) = audited(recorded_entries)
        assert (entry["decision"], entry["reason"]) == ("unavailable", "intake_lost")


class TestHold:
    def test_without_session_key_no_hold_record_is_written(
        self, tmp_path, child_env, hold_root, recorded_entries
    ):
        child_env[HOLD_DIR_ENV_VAR] = str(hold_root)
        project = write_project(tmp_path, {GUARD: _decision()})
        run_entry(child_env, argv(project, write_submission(tmp_path, "s-one")))
        assert not hold_root.exists()
        (entry,) = audited(recorded_entries)
        assert entry["decision"] == "flagged"
        assert entry["session"] == RECORD_SESSION

    def test_session_key_without_a_hold_root_cannot_hold(
        self, tmp_path, child_env, recorded_entries
    ):
        project = write_project(tmp_path, {GUARD: _decision()})
        run_entry(child_env, argv(project, write_submission(tmp_path, "s-one"), POOL_KEY))
        entries = audited(recorded_entries)
        assert {e["decision"] for e in entries} == {"flagged", "unavailable"}
        assert {e["session"] for e in entries} == {RECORD_SESSION}

    def test_session_key_and_a_hold_root_open_a_flag_named_for_the_watchdog(
        self, tmp_path, child_env, hold_root
    ):
        child_env[HOLD_DIR_ENV_VAR] = str(hold_root)
        project = write_project(tmp_path, {GUARD: _decision()})
        run_entry(child_env, argv(project, write_submission(tmp_path, "s-one"), POOL_KEY))
        (flag,) = open_flags(POOL_KEY, root=hold_root)
        assert flag.source == GUARD
        assert flag.prompt_ref == make_prompt_ref(RECORD_SESSION, "s-one")

    def test_a_fallback_for_a_flagged_submission_opens_no_second_flag(
        self, tmp_path, child_env, hold_root
    ):
        child_env[HOLD_DIR_ENV_VAR] = str(hold_root)
        open_flag(
            POOL_KEY,
            flag_id="from-intake",
            source=GUARD,
            reason="risky_reason",
            prompt_ref=make_prompt_ref(RECORD_SESSION, "s-one"),
            prompt_excerpt="loosen the limits",
            root=hold_root,
        )
        project = write_project(tmp_path, {GUARD: _decision()})
        run_entry(child_env, argv(project, write_submission(tmp_path, "s-one"), POOL_KEY))
        (flag,) = open_flags(POOL_KEY, root=hold_root)
        assert flag.flag_id == "from-intake"

    def test_a_session_key_means_intake_failed_and_says_so_per_watchdog(
        self, tmp_path, child_env, hold_root, recorded_entries
    ):
        child_env[HOLD_DIR_ENV_VAR] = str(hold_root)
        project = write_project(tmp_path, {GUARD: _decision(), READER: _agent()})
        (Path(child_env["RUN_DIR"]) / "release").touch()
        run_entry(child_env, argv(project, write_submission(tmp_path, "s-one"), POOL_KEY))
        failed = [e for e in audited(recorded_entries) if e["reason"] == "intake_failed"]
        assert {e["subject"] for e in failed} == {GUARD, READER}
        assert {e["decision"] for e in failed} == {"unavailable"}

    def test_no_session_key_audits_no_intake_failure(self, tmp_path, child_env, recorded_entries):
        project = write_project(tmp_path, {GUARD: _decision()})
        run_entry(child_env, argv(project, write_submission(tmp_path, "s-one")))
        assert all(e["reason"] != "intake_failed" for e in audited(recorded_entries))


class TestSubmissionAndConfig:
    def entry(self, env: dict[str, str], project: Path, submission: Path):
        """The contract's argv, as the hook runs it: real engines, safe path."""
        return subprocess.run(
            [sys.executable, "-P", "-m", "osprey.services.watchdog", *argv(project, submission)],
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    def test_real_argv_consumes_the_submission_on_a_build_with_no_watchdogs(
        self, tmp_path, child_env
    ):
        project = write_project(tmp_path, {})
        submission = write_submission(tmp_path, "s-one")
        done = self.entry(child_env, project, submission)
        assert done.returncode == 0, done.stderr
        assert not submission.exists()

    def test_a_container_project_dir_holds_the_render_at_its_root(
        self, tmp_path, run_dir, child_env
    ):
        project = write_project(tmp_path, {GUARD: _decision()}) / "build"
        run_entry(child_env, argv(project, write_submission(tmp_path, "s-one")))
        assert (run_dir / "started-s-one").exists()

    def test_the_submission_is_consumed_after_a_run(self, tmp_path, child_env):
        project = write_project(tmp_path, {GUARD: _decision()})
        submission = write_submission(tmp_path, "s-one")
        run_entry(child_env, argv(project, submission))
        assert not submission.exists()

    def test_a_malformed_submission_is_consumed_and_fails(self, tmp_path, child_env):
        project = write_project(tmp_path, {GUARD: _decision()})
        submission = tmp_path / "bad.json"
        submission.write_text('{"prompt": 1}')
        done = self.entry(child_env, project, submission)
        assert done.returncode == 1
        assert not submission.exists()

    def test_an_absent_submission_file_fails_without_a_traceback(self, tmp_path, child_env):
        project = write_project(tmp_path, {GUARD: _decision()})
        done = self.entry(child_env, project, tmp_path / "gone.json")
        assert done.returncode == 1
        assert "Traceback" not in done.stderr

    def test_a_project_with_no_render_fails(self, tmp_path, child_env):
        submission = write_submission(tmp_path, "s-one")
        done = self.entry(child_env, tmp_path, submission)
        assert done.returncode == 1
        assert not submission.exists()

    def test_nothing_it_logs_carries_the_prompt(self, tmp_path, child_env):
        broken = {"engine": "decision", "enabled": True}
        project = write_project(tmp_path, {GUARD: _decision(), "broken": broken})
        submission = write_submission(tmp_path, "s-one", prompt="secret-prompt-text")
        done = self.entry(child_env, project, submission)
        assert done.returncode == 1
        assert "secret-prompt-text" not in done.stderr + done.stdout
