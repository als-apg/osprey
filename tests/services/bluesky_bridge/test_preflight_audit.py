"""Every pre-flight verdict is filed in the queueserver's audit ledger.

The generator is driven by hand, the way a RunEngine would: ``next()`` yields
the ``wait_for`` message, and ``.send([finished])`` hands back a done future
holding the :class:`ProbeOutcome`.
"""

from __future__ import annotations

import concurrent.futures
import json
from types import SimpleNamespace

import pytest

from osprey.audit import writer
from osprey.services.bluesky_bridge import preflight
from osprey.services.bluesky_bridge.preflight import ProbeOutcome, probe_before_motion


class _Connector:
    async def validate_channel(self, _address):  # pragma: no cover - never awaited here
        return True


def _declared(*addresses: str) -> dict:
    connector = _Connector()
    return {
        f"d{i}": SimpleNamespace(_osprey_connector=connector, _read_pv=address)
        for i, address in enumerate(addresses)
    }


def _finished(outcome: ProbeOutcome) -> list:
    future: concurrent.futures.Future = concurrent.futures.Future()
    future.set_result(outcome)
    return [future]


def _drive(declared: dict, answer) -> BaseException | None:
    """Run the pre-flight to its end; the exception it raised, if any."""
    gen = probe_before_motion("grid_scan", declared)
    try:
        next(gen)
    except StopIteration:
        return None
    try:
        gen.send(answer)
    except StopIteration:
        return None
    except Exception as exc:  # the refusal is the result
        return exc
    return None


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    """The preflight ledger this process writes, under a pinned identity."""
    zone = tmp_path / "var" / "audit"
    monkeypatch.setattr(writer, "audit_dir", lambda: zone)
    monkeypatch.setenv("OSPREY_AUDIT_IDENTITY", "queueserver")
    monkeypatch.delenv("OSPREY_TERMINAL_USER", raising=False)
    monkeypatch.delenv(writer.AUDIT_WRITER_ENV, raising=False)
    monkeypatch.setattr(
        "osprey.services.bluesky_bridge.queue_backend.resolve_lane_identity",
        lambda: ("bluesky_va", "va"),
    )
    path = zone / "queueserver" / "preflight.jsonl"

    def records() -> list[dict]:
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    return records


def test_a_passing_probe_files_allowed(ledger):
    outcome = ProbeOutcome(unresponsive=(), timeout_s=1.0, budget_s=5.0)
    assert _drive(_declared("A:1", "A:2"), _finished(outcome)) is None

    (record,) = ledger()
    assert record["surface"] == preflight.SURFACE_PREFLIGHT
    assert record["decision"] == "allowed"
    assert record["reason"] == "all_responded"
    assert record["subject"] == "grid_scan"
    assert record["session"] is None
    assert record["actor"] == "queueserver"
    assert record["detail"] == ("lane=bluesky_va target=va addresses=2 unresponsive=0 unchecked=0")


def test_a_failing_probe_files_refused_with_counts(ledger):
    outcome = ProbeOutcome(unresponsive=("A:1",), unchecked=("A:3",), timeout_s=1.0, budget_s=5.0)
    error = _drive(_declared("A:1", "A:2", "A:3"), _finished(outcome))
    assert isinstance(error, ConnectionError)

    (record,) = ledger()
    assert record["decision"] == "refused"
    assert record["reason"] == "unresponsive"
    assert record["detail"] == ("lane=bluesky_va target=va addresses=3 unresponsive=1 unchecked=1")
    # The addresses stay in the run's error text, never in the ledger.
    assert "A:1" not in json.dumps(record)


def test_a_skipped_probe_files_skipped(ledger):
    declared = {"d0": SimpleNamespace(_read_pv="A:1")}
    assert _drive(declared, None) is None

    (record,) = ledger()
    assert record["decision"] == "allowed"
    assert record["reason"] == "skipped"


def test_a_walk_files_nothing(ledger):
    """A caller that walks the plan sends ``None`` back: nothing ran, nothing is filed."""
    assert _drive(_declared("A:1"), None) is None
    assert ledger() == []


@pytest.mark.usefixtures("ledger")
def test_a_ledger_failure_never_costs_the_run(monkeypatch):
    def broken(**_fields):
        raise OSError("disk gone")

    monkeypatch.setattr(writer, "record", broken)
    outcome = ProbeOutcome(unresponsive=(), timeout_s=1.0, budget_s=5.0)
    assert _drive(_declared("A:1"), _finished(outcome)) is None

    refused = ProbeOutcome(unresponsive=("A:1",), timeout_s=1.0, budget_s=5.0)
    assert isinstance(_drive(_declared("A:1"), _finished(refused)), ConnectionError)
