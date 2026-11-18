"""The server fleet's state model, driven from the one table in ``fixtures/states.yaml``.

The docs show the same table, so a row changed here changes what they say.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.services.channel_directory.fleet.states import (
    MIN_COUNTER_INTERVAL_S,
    Finding,
    HealthReading,
    HealthStatus,
    LivenessKind,
    Reading,
    ReadOutcome,
    ReadPath,
    ServerInputs,
    State,
    classify,
    classify_fleet,
    classify_read,
    fleet_went_quiet,
    order_key,
)
from osprey_connectors.errors import ChannelUnreachableError, ConnectorUnavailableError

TABLE = yaml.safe_load((Path(__file__).parent / "fixtures" / "states.yaml").read_text())

_EXCEPTIONS = {
    "ChannelUnreachableError": lambda: ChannelUnreachableError("IOC1:HEARTBEAT"),
    "TimeoutError": lambda: TimeoutError("timed out"),
    "RuntimeError": lambda: RuntimeError("bad DBR type"),
    "ConnectorUnavailableError": lambda: ConnectorUnavailableError("context lost"),
}


def _reading(pair: list[Any] | None) -> Reading | None:
    return None if pair is None else Reading(value=pair[0], read_at=float(pair[1]))


def _inputs(raw: dict[str, Any], name: str = "IOC1") -> ServerInputs:
    raw = dict(raw)
    ioc = ServerInputs(
        name=raw.pop("name", name),
        outcome=ReadOutcome(raw.pop("outcome")),
        kind=LivenessKind(raw.pop("kind", "counter")),
        answer=_reading(raw.pop("answer", None)),
        previous=_reading(raw.pop("previous", None)),
        previous_state=State(raw.pop("previous_state")) if "previous_state" in raw else None,
        seen=raw.pop("seen", False),
        answered_last_cycle=raw.pop("answered_last_cycle", False),
        health=tuple(
            HealthReading(channel=h["channel"], value=h["value"], status=HealthStatus(h["status"]))
            for h in raw.pop("health", ())
        ),
        expected=raw.pop("expected", None),
        roster_configured=raw.pop("roster_configured", False),
        detail=raw.pop("detail", ""),
    )
    assert not raw, f"unknown fixture inputs: {sorted(raw)}"
    return ioc


# --- C7: read outcomes ----------------------------------------------------------


@pytest.mark.parametrize("row", TABLE["read_outcomes"], ids=lambda r: str(r["raises"]))
def test_read_outcome_class(row: dict[str, Any]) -> None:
    exc = None if row["raises"] is None else _EXCEPTIONS[row["raises"]]()
    assert classify_read(exc) is ReadOutcome(row["outcome"])


@pytest.mark.parametrize("row", TABLE["read_outcomes"], ids=lambda r: str(r["raises"]))
@pytest.mark.parametrize("seen", [True, False], ids=["seen", "never-seen"])
def test_read_outcome_feeds_state(row: dict[str, Any], seen: bool) -> None:
    outcome = ReadOutcome(row["outcome"])
    answer = Reading(value=1, read_at=100.0) if outcome is ReadOutcome.ANSWERED else None
    verdict = classify(
        ServerInputs(
            name="IOC1", outcome=outcome, kind=LivenessKind.VALUE, answer=answer, seen=seen
        ),
        read_path=ReadPath.OK,
    )
    assert verdict.state is State(row["if_seen" if seen else "if_never_seen"])


def test_table_covers_every_read_outcome_raised() -> None:
    raised = {ReadOutcome(r["outcome"]) for r in TABLE["read_outcomes"]}
    assert raised == set(ReadOutcome) - {ReadOutcome.NO_CHANNEL}


def test_no_answer_and_read_failed_never_meet() -> None:
    """'No answer' feeds silent/unreached; 'read failed' feeds unknown, whatever else holds."""
    rows = {r["outcome"]: r for r in TABLE["read_outcomes"]}
    assert {rows["no_answer"]["if_seen"], rows["no_answer"]["if_never_seen"]} == {
        "silent",
        "unreached",
    }
    assert rows["read_failed"]["if_seen"] == rows["read_failed"]["if_never_seen"] == "unknown"


def test_cancelled_read_is_not_an_outcome() -> None:
    import asyncio

    # A cancelled read is not a failure the model can judge; the caller must not cancel.
    with pytest.raises(TypeError):
        classify_read(asyncio.CancelledError())


# --- one IOC --------------------------------------------------------------------


@pytest.mark.parametrize("case", TABLE["cases"], ids=lambda c: c["name"])
def test_case(case: dict[str, Any]) -> None:
    verdict = classify(_inputs(case["inputs"]), read_path=ReadPath(case.get("read_path", "ok")))
    expect = case["expect"]
    assert verdict.state is State(expect["state"])
    assert verdict.findings == tuple(Finding(f) for f in expect["findings"])
    if "reason" in expect:
        assert expect["reason"].lower() in verdict.reason.lower()
    if "advance_reading" in expect:
        assert verdict.advance_reading is expect["advance_reading"]
    assert verdict.reason


def test_minimum_interval_matches_table() -> None:
    """The table's boundary row sits exactly at the model's minimum interval."""
    assert MIN_COUNTER_INTERVAL_S == 2.5


def test_minimum_interval_is_a_parameter() -> None:
    ioc = _inputs(
        {
            "outcome": "answered",
            "answer": [10, 100.0],
            "previous": [10, 90.0],
            "previous_state": "alive",
            "seen": True,
        }
    )
    assert classify(ioc, read_path=ReadPath.OK).state is State.STALLED
    kept = classify(ioc, read_path=ReadPath.OK, min_interval_s=20.0)
    assert kept.state is State.ALIVE
    assert kept.advance_reading is False


def test_every_state_and_finding_reachable_from_the_table() -> None:
    states = {c["expect"]["state"] for c in TABLE["cases"]}
    findings = {f for c in TABLE["cases"] for f in c["expect"]["findings"]}
    assert states == {s.value for s in State}
    assert findings == {f.value for f in Finding}


def test_findings_follow_c4_rules_over_the_whole_table() -> None:
    for case in TABLE["cases"]:
        verdict = classify(_inputs(case["inputs"]), read_path=ReadPath(case.get("read_path", "ok")))
        if verdict.state is State.UNKNOWN:
            assert verdict.findings == ()
        if verdict.state in (State.UNREACHED, State.SILENT, State.STALLED):
            assert set(verdict.findings) <= {Finding.EXPECTED_RUNNING_NOT_ALIVE}
        if Finding.DEGRADED in verdict.findings:
            assert verdict.state is State.ALIVE


def test_answered_without_a_reading_is_rejected() -> None:
    with pytest.raises(ValueError, match="answer"):
        ServerInputs(name="IOC1", outcome=ReadOutcome.ANSWERED)


def test_expected_must_be_a_roster_word() -> None:
    with pytest.raises(ValueError, match="expected"):
        ServerInputs(name="IOC1", outcome=ReadOutcome.NO_ANSWER, expected="maybe")


# --- one cycle ------------------------------------------------------------------


@pytest.mark.parametrize("case", TABLE["fleet_cases"], ids=lambda c: c["name"])
def test_fleet_case(case: dict[str, Any]) -> None:
    iocs = [_inputs(raw) for raw in case["servers"]]
    read_path = ReadPath(case["read_path"])
    verdicts = classify_fleet(iocs, read_path=read_path)
    assert {ioc.name: v.state.value for ioc, v in zip(iocs, verdicts, strict=True)} == case[
        "expect"
    ]
    quiet = fleet_went_quiet(iocs, read_path=read_path)
    assert quiet is any(v.state is State.UNKNOWN for v in verdicts)


def test_fleet_quiet_reason_names_the_unverified_read_path() -> None:
    iocs = [
        _inputs({"name": "a", "outcome": "no_answer", "seen": True, "answered_last_cycle": True})
    ]
    (verdict,) = classify_fleet(iocs, read_path=ReadPath.UNVERIFIED)
    assert verdict.state is State.UNKNOWN
    assert "unverified" in verdict.reason


def test_empty_fleet_is_not_quiet() -> None:
    assert fleet_went_quiet([], read_path=ReadPath.UNVERIFIED) is False
    assert classify_fleet([], read_path=ReadPath.UNVERIFIED) == []


# --- order ----------------------------------------------------------------------


def test_order() -> None:
    rows = TABLE["order"]["rows"]
    ordered = sorted(
        rows,
        key=lambda r: order_key(
            State(r["state"]), tuple(Finding(f) for f in r["findings"]), r["name"]
        ),
    )
    assert [r["name"] for r in ordered] == TABLE["order"]["expect"]


def test_order_key_accepts_plain_words() -> None:
    assert order_key("stalled", (), "x") == order_key(State.STALLED, (), "x")
    assert order_key("alive", ("degraded",), "x") == order_key(
        State.ALIVE, (Finding.DEGRADED,), "x"
    )
