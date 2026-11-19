"""Tests for ``tests/_coverage_handoff.py``: a child given its environment starts without coverage."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests._coverage_handoff import COVERAGE_HANDOFF_PREFIX, without_coverage_handoff

#: What a child prints: whether coverage started in it before its first line ran.
_REPORTS_COVERAGE = "import coverage; print(coverage.Coverage.current() is not None)"


def _child_measures_itself(env: dict[str, str]) -> bool:
    done = subprocess.run(
        [sys.executable, "-c", _REPORTS_COVERAGE],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return done.stdout.strip() == "True"


def test_both_hand_off_variables_go_and_nothing_else_does() -> None:
    env = {
        "COVERAGE_PROCESS_CONFIG": "{}",
        "COVERAGE_PROCESS_START": "/x/.coveragerc",
        "COVERAGE_CORE": "sysmon",
        "PATH": "/bin",
        "PYTHONPATH": "/src",
    }

    assert without_coverage_handoff(env) == {
        "COVERAGE_CORE": "sysmon",
        "PATH": "/bin",
        "PYTHONPATH": "/src",
    }
    assert "COVERAGE_PROCESS_START" in env, "the caller's mapping is left as it was"


def test_the_default_is_this_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COVERAGE_PROCESS_START", "/x/.coveragerc")
    monkeypatch.setenv("OSPREY_HANDOFF_PROBE", "kept")

    env = without_coverage_handoff()

    assert env["OSPREY_HANDOFF_PROBE"] == "kept"
    assert not [key for key in env if key.startswith(COVERAGE_HANDOFF_PREFIX)]


def test_a_child_given_the_filtered_environment_does_not_start_coverage(tmp_path: Path) -> None:
    """The mechanism itself, with its control: the same child, handed the switch
    coverage's ``.pth`` start-up reads, does measure itself."""
    rc = tmp_path / ".coveragerc"
    rc.write_text(f"[run]\ndata_file = {tmp_path / '.coverage'}\n", encoding="utf-8")
    handed_off = {**without_coverage_handoff(), "COVERAGE_PROCESS_START": str(rc)}

    assert _child_measures_itself(handed_off), "control: the hand-off starts coverage"
    assert not _child_measures_itself(without_coverage_handoff(handed_off))
