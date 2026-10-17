"""The trajectory state contract: the snapshot, the result models, the SSE events.

``panel/state.schema.json`` is committed; the first test regenerates it from
``state.py`` and compares the bytes, so a model edit without a regeneration
fails. The acceptance table below walks the epic's panel state table row by row
and names the snapshot fields each row is drawn from.
"""

from __future__ import annotations

import json
import math
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from osprey.services.trajectory import state
from osprey.services.trajectory.state import (
    SCHEMA_PATH,
    ApplyEvent,
    ContinuousEndedEvent,
    ContinuousStatus,
    HolderView,
    LastAction,
    Moved,
    ReferenceLoad,
    RunLockHolder,
    RunningAction,
    ShotEvent,
    ShotFailure,
    ShotResult,
    ShotSummary,
    Snapshot,
    SnapshotEvent,
    Step,
    StepEvent,
    StoppedEvent,
    TakenOver,
    TakenOverEvent,
    parse_event,
    render_schema,
)

# --- fixtures -----------------------------------------------------------------


def _step(**overrides: Any) -> Step:
    fields: dict[str, Any] = {
        "digest": "d" * 64,
        "deltas": {"HCM1": -1.5e-5, "VCM1": 0.0},
        "factor": 0.5,
        "reason": "OK",
        "status": "ok",
        "all_zero": False,
        "zeroed_h": [],
        "zeroed_v": ["VCM1"],
        "no_response_names": [],
        "unsolved": [],
        "rank_discarded": 0,
        "singular_values": [2.0, 1.0],
        "inverse_weights": [0.5, 1.0],
        "spectrum_kind": ["LIVE", "LIVE"],
        "n_discard": 0,
        "shot_id": "shot-1",
        "reference": "golden",
    }
    fields.update(overrides)
    return Step(**fields)


def _shot(**overrides: Any) -> ShotResult:
    fields: dict[str, Any] = {
        "shot_id": "shot-1",
        "model": "sr",
        "x": {"BPM1:x": 1e-4, "BPM2:x": math.nan},
        "y": {"BPM1:y": -2e-4, "BPM2:y": math.nan},
        "sum": {"BPM1:sum": 1.0, "BPM2:sum": 0.0},
        "bpm_order": ["BPM1", "BPM2"],
        "lost": ["BPM2"],
        "last_bpm": "BPM1",
        "i0": 10,
        "window": (0, 30),
        "per_shot_lost_at": ["BPM2", None],
        "shots_included": 2,
        "shots_failed": [{"index": 2, "reason": "armed never cleared", "stragglers": ["BPM7"]}],
        "commands_fired": ["SR:BPM1:ARM", "SR:INJ:REQ"],
    }
    fields.update(overrides)
    return ShotResult(**fields)


def _reference() -> ReferenceLoad:
    return ReferenceLoad(
        name="golden",
        sha256="a" * 64,
        matched=["BPM1"],
        missing_in_reference=["BPM2"],
        unused_in_reference=["BPM9"],
        nan_count=0,
    )


def _holder(owner: str = "alice", kind: str = "panel") -> HolderView:
    return HolderView(owner=owner, kind=kind)


def _snapshot(**overrides: Any) -> Snapshot:
    fields: dict[str, Any] = {
        "target": "va",
        "caller": "alice",
        "writes_enabled": True,
        "posture": "writes enabled",
        "limits": "optional",
        "writing_available": True,
        "model": "sr",
        "model_kind": "single_pass",
        "run_lock_holder": None,
        "plan_running": False,
        "holder": _holder(),
        "taken_over": None,
        "mode": "idle",
        "action": None,
        "queue_wait_s": 0.0,
        "action_timeout_s": 60.0,
        "action_budget_s": 75.0,
        "stop_budget_s": 12.0,
        "last_shot": None,
        "step": None,
        "reference": None,
        "last_action": None,
        "continuous": None,
    }
    fields.update(overrides)
    return Snapshot(**fields)


def _every_event() -> list[BaseModel]:
    busy = _snapshot(
        mode="continuous",
        action=RunningAction(name="shot", stopping=True, armed_stragglers=["BPM3"]),
        run_lock_holder=RunLockHolder(
            pid=4242, started="2026-10-16T10:00:00Z", owner="bob", target="va", identity="sandbox"
        ),
        plan_running=None,
        taken_over=TakenOver(previous=_holder("bob", "agent"), by=_holder()),
        last_shot=ShotSummary.of(_shot()),
        step=_step(),
        reference=_reference(),
        last_action=LastAction(
            name="apply", stopped=True, moved=[Moved(name="HCM1", before=0.1, outcome="written")]
        ),
        continuous=ContinuousStatus(shots=3, max_shots=50, lease_expires_in_s=7.5),
    )
    return [
        SnapshotEvent(snapshot=_snapshot()),
        SnapshotEvent(snapshot=busy),
        ShotEvent(shot=_shot()),
        StepEvent(step=_step()),
        StepEvent(step=None),
        ApplyEvent(
            step_digest="d" * 64,
            shot_id="shot-1",
            moved=[Moved(name="HCM1", before=0.1, outcome="written")],
        ),
        ApplyEvent(step_digest=None, shot_id=None, moved=[]),
        StoppedEvent(
            action="apply",
            stopper="bob",
            moved=[Moved(name="HCM1", before=0.1, outcome="written")],
        ),
        TakenOverEvent(previous=_holder("bob", "agent"), by=_holder()),
        ContinuousEndedEvent(
            end="zero_step", shots=4, step_reason="NO_BEAM", step_status="no BPM with beam"
        ),
        ContinuousEndedEvent(end="stopped", shots=1, step_reason=None, step_status=None),
    ]


# --- the committed schema -------------------------------------------------------


def test_the_committed_schema_is_the_regenerated_schema_of_the_models() -> None:
    committed = SCHEMA_PATH.read_text(encoding="utf-8")
    assert committed == render_schema(), (
        "panel/state.schema.json is stale: run "
        "`uv run python -m osprey.services.trajectory.state` and commit the file"
    )


def test_the_schema_sits_beside_the_panel_bundle() -> None:
    assert SCHEMA_PATH.name == "state.schema.json"
    assert SCHEMA_PATH.parent.name == "panel"
    assert (SCHEMA_PATH.parent / "index.html").is_file()


def test_the_schema_reads_as_plain_json_with_every_event_and_the_snapshot() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    defs = schema["$defs"]
    for name in (
        "Snapshot",
        "SnapshotEvent",
        "ShotEvent",
        "StepEvent",
        "ApplyEvent",
        "StoppedEvent",
        "TakenOverEvent",
        "ContinuousEndedEvent",
        "ShotResult",
        "Step",
        "Moved",
        "ReferenceLoad",
    ):
        assert name in defs, name
    assert schema["discriminator"]["propertyName"] == "type"


# --- every event validates against the committed schema -------------------------


@pytest.mark.parametrize("event", _every_event(), ids=lambda e: type(e).__name__)
def test_every_event_validates_against_the_committed_schema(event: BaseModel) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(json.loads(event.model_dump_json()), schema)


@pytest.mark.parametrize("event", _every_event(), ids=lambda e: type(e).__name__)
def test_every_event_round_trips_through_its_wire_form(event: BaseModel) -> None:
    parsed = parse_event(event.model_dump_json())
    assert type(parsed) is type(event)
    assert parsed.model_dump_json() == event.model_dump_json()


def test_a_snapshot_alone_validates_against_the_snapshot_definition() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    snapshot_schema = {"$defs": schema["$defs"], "$ref": "#/$defs/Snapshot"}
    jsonschema.validate(json.loads(_snapshot().model_dump_json()), snapshot_schema)


def test_an_event_with_an_unknown_type_is_refused() -> None:
    with pytest.raises(ValidationError):
        parse_event(json.dumps({"type": "restore"}))


def test_an_unknown_field_is_refused() -> None:
    with pytest.raises(ValidationError):
        Moved(name="HCM1", before=0.1, outcome="written", after=0.2)  # type: ignore[call-arg]


# --- the result models ------------------------------------------------------------


def test_a_lost_bpm_reads_nan_in_memory_and_null_on_the_wire() -> None:
    shot = _shot()
    assert math.isnan(shot.x["BPM2:x"])
    wire = json.loads(shot.model_dump_json())
    assert wire["x"]["BPM2:x"] is None
    assert wire["y"]["BPM2:y"] is None
    assert wire["x"]["BPM1:x"] == pytest.approx(1e-4)


def test_a_shot_whose_every_attempt_failed_carries_no_extraction() -> None:
    shot = _shot(
        x={},
        y={},
        sum={},
        lost=[],
        last_bpm=None,
        i0=None,
        window=None,
        per_shot_lost_at=[],
        shots_included=0,
        shots_failed=[ShotFailure(index=0, reason="no BPM rose")],
    )
    assert shot.shots_included == 0
    assert shot.shots_failed[0].stragglers == []


def test_a_shot_summary_names_its_stragglers_and_counts() -> None:
    summary = ShotSummary.of(_shot())
    assert summary.shot_id == "shot-1"
    assert summary.shots_included == 2
    assert summary.armed_stragglers == ["BPM7"]
    assert summary.lost == ["BPM2"]
    assert summary.last_bpm == "BPM1"


def test_a_step_reason_is_one_of_the_tool_reasons() -> None:
    for reason in (
        "NO_BEAM",
        "NO_INVERSE",
        "NO_UPSTREAM",
        "NO_UPSTREAM_H",
        "NO_UPSTREAM_V",
        "ON_REFERENCE",
        "OK",
    ):
        assert _step(reason=reason).reason == reason
    with pytest.raises(ValidationError):
        _step(reason="FINE")


def test_a_step_spectrum_kind_is_one_of_the_tool_kinds() -> None:
    step = _step(spectrum_kind=["STRUCTURAL", "RANK_DISCARDED", "DISCARDED", "LIVE"])
    assert step.spectrum_kind[0] == "STRUCTURAL"
    with pytest.raises(ValidationError):
        _step(spectrum_kind=["DEAD"])


def test_moved_is_the_put_back_shape() -> None:
    moved = Moved(name="HCM1", before=0.25, outcome="written")
    assert set(moved.model_dump()) == {"name", "before", "outcome"}


def test_a_continuous_end_names_one_end_condition() -> None:
    for end in (
        "stopped",
        "lease_expired",
        "max_shots",
        "zero_step",
        "taken_over",
        "released",
        "shutdown",
        "failed",
    ):
        assert ContinuousEndedEvent(end=end, shots=0).end == end
    with pytest.raises(ValidationError):
        ContinuousEndedEvent(end="bored", shots=0)


def test_the_snapshot_mode_is_idle_action_or_continuous() -> None:
    for mode in ("idle", "action", "continuous"):
        assert _snapshot(mode=mode).mode == mode
    with pytest.raises(ValidationError):
        _snapshot(mode="paused")


def test_the_model_kind_is_none_or_single_pass() -> None:
    assert _snapshot(model_kind="none", model=None).model_kind == "none"
    with pytest.raises(ValidationError):
        _snapshot(model_kind="ring")


def test_the_holder_kind_is_a_door() -> None:
    with pytest.raises(ValidationError):
        HolderView(owner="alice", kind="cli")


# --- acceptance: the epic's panel state table ------------------------------------

#: Each row of the epic's panel state table, with the snapshot fields it is
#: drawn from. Omitted on purpose: "service unreachable" and "owner's terminal
#: on another target" (both client-side: the first is a failed fetch, the second
#: compares this snapshot's ``target`` with the terminal's own), and "pending
#: dead run" (voided by e-0003: no journal, no restore-as-action).
STATE_TABLE: dict[str, tuple[str, ...]] = {
    "no model carries the trajectory kind": ("writing_available", "model_kind", "model"),
    "single-pass model": ("model_kind", "model"),
    "reached without an operator identity": ("caller",),
    "posture read-only": ("writes_enabled", "posture"),
    "run lock held by another process": ("run_lock_holder",),
    "a Bluesky plan is running": ("plan_running",),
    "target": ("target",),
    "another client holds the session": ("holder",),
    "shot running": ("mode", "action"),
    "continuous running": ("mode", "continuous", "step"),
    "stopping": ("action", "stop_budget_s"),
    "taken over": ("taken_over",),
    "all-zero step": ("step",),
}

#: Fields the task names on top of the table.
TASK_FIELDS: tuple[str, ...] = (
    "holder",
    "taken_over",
    "target",
    "limits",
    "mode",
    "queue_wait_s",
    "action_timeout_s",
    "action_budget_s",
    "stop_budget_s",
    "last_shot",
    "step",
    "reference",
    "plan_running",
    "last_action",
    "continuous",
)


@pytest.mark.parametrize("row", sorted(STATE_TABLE))
def test_every_row_of_the_epic_state_table_has_its_snapshot_fields(row: str) -> None:
    for field in STATE_TABLE[row]:
        assert field in Snapshot.model_fields, f"{row!r} needs Snapshot.{field}"


def test_every_field_the_task_names_is_on_the_snapshot() -> None:
    missing = [name for name in TASK_FIELDS if name not in Snapshot.model_fields]
    assert missing == []


def test_the_rows_read_their_answers_from_nested_fields() -> None:
    snap = _snapshot(
        holder=_holder("bob", "agent"),
        taken_over=TakenOver(previous=_holder(), by=_holder("bob", "agent")),
        action=RunningAction(name="shot", stopping=True, armed_stragglers=["BPM3"]),
        step=_step(all_zero=True, reason="NO_BEAM", status="no BPM with beam"),
        reference=_reference(),
        run_lock_holder=RunLockHolder(
            pid=1, started=None, owner=None, target="va", identity="sandbox"
        ),
    )
    # another client holds the session: names the holder's kind and owner
    assert (snap.holder.kind, snap.holder.owner) == ("agent", "bob")
    # taken over: "taken over by <kind> of <owner>"
    assert (snap.taken_over.by.kind, snap.taken_over.by.owner) == ("agent", "bob")
    # shot running: armed stragglers shown; stopping: the in-flight bound
    assert snap.action.armed_stragglers == ["BPM3"]
    assert snap.action.stopping is True
    # all-zero step: Apply disabled with the step's status
    assert snap.step.all_zero is True
    # reference with match counts
    assert len(snap.reference.matched) == 1
    # run lock held by another process: the banner names the holder
    assert snap.run_lock_holder.identity == "sandbox"


def test_the_snapshot_never_carries_a_client_id() -> None:
    def walk(model: type[BaseModel]) -> set[str]:
        names = set(model.model_fields)
        for info in model.model_fields.values():
            for arg in _model_args(info.annotation):
                names |= walk(arg)
        return names

    assert "client_id" not in walk(Snapshot)


def _model_args(annotation: Any) -> list[type[BaseModel]]:
    import typing

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    found: list[type[BaseModel]] = []
    for arg in typing.get_args(annotation):
        found.extend(_model_args(arg))
    return found


def test_the_module_lists_its_public_names() -> None:
    for name in state.__all__:
        assert hasattr(state, name), name
