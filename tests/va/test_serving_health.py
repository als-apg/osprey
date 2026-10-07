"""The served process's health record: its states, its counters and its file."""

from __future__ import annotations

import ast
import json
import os
from collections.abc import Iterable
from pathlib import Path

import pytest

from osprey.services.virtual_accelerator.serving import health
from osprey.services.virtual_accelerator.serving.health import ServingHealth


def _record(tolerance: int = 2, clock: Iterable[float] = (11.0, 12.0, 13.0, 14.0, 15.0)):
    return ServingHealth(tolerance, iter(clock).__next__, 10.0)


def test_a_record_before_any_pass_is_serving_with_no_last_pass() -> None:
    document = _record().document()

    assert document == {
        "state": "serving",
        "last_pass": None,
        "passes_ok": 0,
        "passes_failed": 0,
        "consecutive_failed": 0,
        "failed_pass_tolerance": 2,
        "last_failed_pass": None,
    }


def test_failures_within_the_tolerance_are_degraded() -> None:
    record = _record(tolerance=2)

    record.record_pass("the deck has no stable orbit")
    assert record.state == "degraded"
    record.record_pass("the deck has no stable orbit")

    document = record.document()
    assert document["state"] == "degraded"
    assert document["consecutive_failed"] == 2
    assert document["passes_failed"] == 2
    assert document["last_pass"]["outcome"] == "failed"
    assert document["last_failed_pass"] == {
        "error": "the deck has no stable orbit",
        "uptime_s": 2.0,
    }


def test_failures_beyond_the_tolerance_are_failed() -> None:
    record = _record(tolerance=2)

    for _ in range(3):
        record.record_pass("the pass failed")

    assert record.state == "failed"
    assert record.document()["consecutive_failed"] == 3


def test_a_tolerance_of_zero_fails_on_the_first_failed_pass() -> None:
    record = _record(tolerance=0)

    record.record_pass("the pass failed")

    assert record.state == "failed"


def test_a_success_after_failures_returns_to_serving() -> None:
    record = _record(tolerance=1)
    record.record_pass("first")
    record.record_pass("second")
    assert record.state == "failed"

    record.record_pass(None)

    document = record.document()
    assert document["state"] == "serving"
    assert document["consecutive_failed"] == 0
    assert (document["passes_ok"], document["passes_failed"]) == (1, 2)
    assert document["last_failed_pass"] == {"error": "second", "uptime_s": 2.0}
    assert document["last_pass"]["outcome"] == "ok"
    assert document["last_pass"]["uptime_s"] == 3.0


def test_the_last_pass_is_stamped_with_the_wall_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(health.time, "time", lambda: 1_700_000_000.5)
    record = _record()

    record.record_pass(None)

    assert record.document()["last_pass"] == {
        "outcome": "ok",
        "uptime_s": 1.0,
        "at": 1_700_000_000.5,
    }


def test_the_file_is_replaced_atomically_and_parses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "run" / "osprey-va" / "health.json"
    replaced: list[tuple[str, str]] = []
    real_replace = os.replace

    def spy(src: str, dst: str) -> None:
        replaced.append((str(src), str(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(health.os, "replace", spy)
    record = _record()
    record.record_pass(None)
    health.write_document(path, record.document())
    record.record_pass("the pass failed")
    health.write_document(path, record.document())

    assert json.loads(path.read_text(encoding="utf-8")) == record.document()
    assert len(replaced) == 2
    for src, dst in replaced:
        assert Path(src).parent == path.parent
        assert dst == str(path)
    assert sorted(p.name for p in path.parent.iterdir()) == ["health.json"]


@pytest.mark.parametrize("tolerance", [True, False, -1, 1.5, "3"])
def test_a_bad_tolerance_is_refused(tolerance: object) -> None:
    with pytest.raises(ValueError, match="failed_pass_tolerance"):
        ServingHealth(tolerance, lambda: 0.0, 0.0)  # type: ignore[arg-type]


def test_the_health_module_imports_no_server_library() -> None:
    """The record is decided in process, with no server library behind it."""
    tree = ast.parse(Path(health.__file__).read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    roots = {name.split(".")[0] for name in imported}
    assert roots.isdisjoint({"pcaspy", "p4p", "lume_pva_apg"})
    assert not any(name.endswith("serving.runner") for name in imported)
