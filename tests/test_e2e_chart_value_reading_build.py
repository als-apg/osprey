"""The chart-value-reading benchmark's build and host-sharing rules, pinned without a deploy.

A subagent runs the model id the build wrote into its definition, and the plot
in ``tests/e2e/tier2/test_chart_value_reading.py`` is read inside the
``logbook-search`` subagent. A build that names no model takes the provider
catalog's default, and the benchmark then scores that model whatever the lane
or the matrix cell asked for.

The benchmark board runs its cells in parallel on one host, while this lane's
stack has one fixed project name and port block so the cells share its images.
Two rules make that safe and affordable: one stack per host at a time, and one
image build per checked-out revision.
"""

from __future__ import annotations

import fcntl
from pathlib import Path

import pytest

from tests.e2e.provider import E2E_MODEL, FORCE_MODEL_ENV
from tests.e2e.tier2 import test_chart_value_reading as benchmark


def _model_edits(args: list[str]) -> list[str]:
    return [a for a in args if a.startswith("model=")]


def test_init_names_the_suite_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OSPREY_E2E_PROVIDER", "als-apg")
    monkeypatch.delenv(FORCE_MODEL_ENV, raising=False)
    assert _model_edits(benchmark._init_args(tmp_path / "repo")) == [f"model={E2E_MODEL}"]


def test_init_names_the_forced_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OSPREY_E2E_PROVIDER", "als-apg")
    monkeypatch.setenv(FORCE_MODEL_ENV, "claude-opus-5-5")
    assert _model_edits(benchmark._init_args(tmp_path / "repo")) == ["model=claude-opus-5-5"]


def _locked_elsewhere(path: Path) -> bool:
    with path.open("a+") as other:
        try:
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(other, fcntl.LOCK_UN)
        return False


def test_the_host_stack_lock_excludes_a_second_holder(tmp_path: Path) -> None:
    lock = tmp_path / "stack.lock"
    with benchmark.host_stack_lock(lock):
        assert _locked_elsewhere(lock)
    assert not _locked_elsewhere(lock)


def test_the_host_stack_lock_is_released_when_the_holder_raises(tmp_path: Path) -> None:
    lock = tmp_path / "stack.lock"
    with pytest.raises(RuntimeError), benchmark.host_stack_lock(lock):
        raise RuntimeError("deploy failed")
    assert not _locked_elsewhere(lock)


def test_images_are_rebuilt_once_per_revision(tmp_path: Path) -> None:
    stamp = tmp_path / "images.revision"
    assert not benchmark.images_built_for(stamp, "abc123")
    benchmark.record_images_built(stamp, "abc123")
    assert benchmark.images_built_for(stamp, "abc123")
    assert not benchmark.images_built_for(stamp, "def456")


def test_an_unknown_revision_always_rebuilds(tmp_path: Path) -> None:
    stamp = tmp_path / "images.revision"
    benchmark.record_images_built(stamp, None)
    assert not stamp.exists()
    assert not benchmark.images_built_for(stamp, None)
