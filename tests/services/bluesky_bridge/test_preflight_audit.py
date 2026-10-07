"""Every pre-flight verdict is filed in the queueserver's audit ledger.

The generator is driven by hand, the way a RunEngine would: ``next()`` yields
the ``wait_for`` message, and ``.send([finished])`` hands back a done future
holding the :class:`ProbeOutcome`.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
from types import SimpleNamespace

import pytest

from osprey.audit import writer
from osprey.audit.envelope import MAX_DETAIL_CHARS
from osprey.services.bluesky_bridge import preflight
from osprey.services.bluesky_bridge.preflight import ProbeOutcome, probe_before_motion
from osprey_connectors.posture_store import (
    NO_OWNER,
    RESERVED_OWNER_KWARG,
    bind_owner,
    bound_owner,
)


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


def _digest(*addresses: str) -> str:
    return hashlib.sha256("\n".join(sorted(set(addresses))).encode()).hexdigest()[:12]


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
    assert record["detail"] == (
        "lane=bluesky_va target=va addresses=2 unresponsive=0 unchecked=0 "
        f"digest={_digest('A:1', 'A:2')} truncated=false channels=A:1,A:2"
    )


def test_a_failing_probe_files_refused_with_counts(ledger):
    outcome = ProbeOutcome(unresponsive=("A:1",), unchecked=("A:3",), timeout_s=1.0, budget_s=5.0)
    error = _drive(_declared("A:1", "A:2", "A:3"), _finished(outcome))
    assert isinstance(error, ConnectionError)

    (record,) = ledger()
    assert record["decision"] == "refused"
    assert record["reason"] == "unresponsive"
    assert record["detail"] == (
        "lane=bluesky_va target=va addresses=3 unresponsive=1 unchecked=1 "
        f"digest={_digest('A:1', 'A:2', 'A:3')} truncated=false channels=A:1,A:2,A:3"
    )


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


def test_a_person_owned_plan_carries_its_owner_and_channels(ledger):
    outcome = ProbeOutcome(unresponsive=(), timeout_s=1.0, budget_s=5.0)
    with bind_owner({RESERVED_OWNER_KWARG: "alice"}):
        assert _drive(_declared("B:2", "A:1", "B:2"), _finished(outcome)) is None
    assert bound_owner() is NO_OWNER

    (record,) = ledger()
    detail = record["detail"]
    assert detail.startswith("lane=bluesky_va target=va addresses=3 owner=alice ")
    assert f"digest={_digest('A:1', 'B:2')}" in detail
    # Sorted and de-duplicated, whatever order the plan declared them in.
    assert detail.endswith(" truncated=false channels=A:1,B:2")


def test_no_owner_gives_no_owner_token(ledger):
    outcome = ProbeOutcome(unresponsive=(), timeout_s=1.0, budget_s=5.0)
    with bind_owner({}):
        assert _drive(_declared("A:1"), _finished(outcome)) is None

    (record,) = ledger()
    assert "owner=" not in record["detail"]


def test_the_digest_ignores_declaration_order(ledger):
    outcome = ProbeOutcome(unresponsive=(), timeout_s=1.0, budget_s=5.0)
    _drive(_declared("A:1", "A:2"), _finished(outcome))
    _drive(_declared("A:2", "A:1", "A:1"), _finished(outcome))

    first, second = ledger()
    digest = lambda detail: detail.split("digest=")[1].split()[0]  # noqa: E731
    assert digest(first["detail"]) == digest(second["detail"]) == _digest("A:1", "A:2")


def test_five_thousand_channels_fit_the_cap_with_count_and_digest_intact(ledger):
    addresses = [f"SR:C{i:04d}:BPM:X" for i in range(5000)]
    outcome = ProbeOutcome(unresponsive=(), timeout_s=1.0, budget_s=5.0)
    with bind_owner({RESERVED_OWNER_KWARG: "alice"}):
        assert _drive(_declared(*addresses), _finished(outcome)) is None

    (record,) = ledger()
    detail = record["detail"]
    assert len(detail) <= MAX_DETAIL_CHARS
    assert " addresses=5000 " in detail
    assert " owner=alice " in detail
    assert f" digest={_digest(*addresses)} " in detail
    assert " truncated=true " in detail
    # The tail holds whole addresses only, the first ones in sorted order.
    shown = detail.split(" channels=")[1].split(",")
    assert shown == sorted(addresses)[: len(shown)]
    assert 0 < len(shown) < 5000
    # The fill is greedy: one more address would have crossed the cap.
    assert len(detail) + 1 + len(sorted(addresses)[len(shown)]) > MAX_DETAIL_CHARS
