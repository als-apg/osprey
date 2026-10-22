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
    ApplyResult,
    Catalog,
    CatalogEntry,
    CatalogGroup,
    CatalogPlace,
    ChainBody,
    ChainResult,
    ChainStep,
    CommandWrite,
    ContinuousEndedEvent,
    ContinuousStatus,
    HolderView,
    LastAction,
    Machine,
    Moved,
    ReferenceImport,
    ReferenceLoad,
    ReferenceReading,
    RmSummary,
    RunningAction,
    SettingReadback,
    SetupBody,
    SetupReadbacks,
    SetupResult,
    SetupState,
    ShotEvent,
    ShotFailure,
    ShotResult,
    ShotSummary,
    Snapshot,
    SnapshotEvent,
    Solve,
    SolveConfig,
    SolveConfigPartial,
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
        taken_over=TakenOver(previous=_holder("bob", "agent"), by=_holder()),
        last_shot=ShotSummary.of(_shot()),
        step=_step(),
        reference=_reference(),
        last_action=LastAction(
            name="apply", stopped=True, moved=[Moved(name="HCM1", before=0.1, outcome="written")]
        ),
        continuous=ContinuousStatus(shots=3, lease_expires_in_s=7.5),
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
            end="failed", shots=4, step_reason="NO_BEAM", step_status="no BPM with beam"
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
        Moved(name="HCM1", before=0.1, outcome="written", undo=0.2)  # type: ignore[call-arg]


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


def test_a_step_carries_no_spectrum() -> None:
    for name, value in (
        ("rank_discarded", 0),
        ("singular_values", [2.0, 1.0]),
        ("inverse_weights", [0.5, 1.0]),
        ("spectrum_kind", ["LIVE", "LIVE"]),
        ("n_discard", 0),
    ):
        assert name not in Step.model_fields, name
        with pytest.raises(ValidationError):
            _step(**{name: value})


def test_moved_is_the_put_back_shape() -> None:
    moved = Moved(name="HCM1", before=0.25, outcome="written")
    assert set(moved.model_dump()) == {"name", "before", "after", "outcome"}
    assert moved.after is None


def test_a_continuous_end_names_one_end_condition() -> None:
    for end in (
        "stopped",
        "lease_expired",
        "taken_over",
        "released",
        "shutdown",
        "failed",
    ):
        assert ContinuousEndedEvent(end=end, shots=0).end == end
    for end in ("bored", "max_shots", "zero_step"):
        with pytest.raises(ValidationError):
            ContinuousEndedEvent(end=end, shots=0)


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
#: compares this snapshot's ``target`` with the terminal's own), "pending dead
#: run" (voided by e-0003: no journal, no restore-as-action), and "run lock held
#: by another process" and "a Bluesky plan is running" (both refusal-time: the
#: write refused with ``run_busy`` names the holder, the one refused with
#: ``plan_running`` names the lane).
STATE_TABLE: dict[str, tuple[str, ...]] = {
    "no model carries the trajectory kind": ("writing_available", "model_kind", "model"),
    "single-pass model": ("model_kind", "model"),
    "reached without an operator identity": ("caller",),
    "posture read-only": ("writes_enabled", "posture"),
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


# --- the end-state shape, published additively ------------------------------------


def _chain() -> list[ChainStep]:
    return [
        ChainStep(
            id="arm",
            label="Arm",
            description="arm the BPMs",
            command=None,
            default=True,
            position=0,
            timeout_s=None,
        ),
        ChainStep(
            id="clean",
            label="Bunch cleaning",
            description="clear the stored charge",
            command=CommandWrite(address="SR:CLEAN:REQ", value=1),
            default=True,
            position=1,
            timeout_s=5.0,
            done=CommandWrite(address="SR:CLEAN:BUSY", value=0),
        ),
    ]


def _solve() -> Solve:
    return Solve(
        sha="s" * 64,
        singular_values=[2.0, 1.0, 1e-9],
        inverse_weights=[0.5, 1.0, 0.0],
        kind=["LIVE", "LIVE", "STRUCTURAL"],
        L=3,
        live_rank=2,
        rm=RmSummary(source="model", rows=4, h_cols=2, v_cols=2, computed_at=1.0),
        config=SolveConfig(
            alpha=0.0,
            keep_live=2,
            n_discard=0,
            gain=1.0,
            excluded_bpms=["BPM3"],
            excluded_cms=[],
            loss_cutoff=0.5,
            overridden=["excluded_bpms"],
        ),
    )


def _entry(name: str, s: float | None, place: str | None = "SR/S1", plane=None) -> CatalogEntry:
    places = (
        [CatalogPlace(id="SR", level="ring"), CatalogPlace(id=place, level="sector")]
        if place
        else []
    )
    return CatalogEntry(
        name=name,
        device=name.lower(),
        label=f"label {name}",
        places=places,
        s=s,
        plane=plane,
        groups=["g1"],
    )


def _catalog() -> Catalog:
    return Catalog(
        bpms=[_entry("BPM1", 1.0), _entry("BPM2", None, place=None)],
        hcms=[_entry("HCM1", 0.5, plane="x")],
        vcms=[_entry("VCM1", 0.7, plane="y")],
        groups=[CatalogGroup(id="g1", label="Group 1", members=["BPM1", "BPM2", "HCM1", "VCM1"])],
    )


def _filled_snapshot() -> Snapshot:
    reference = _reference().model_copy(
        update={
            "readout": "tbt",
            "readings": {"BPM1": ReferenceReading(x=1e-4, y=None)},
        }
    )
    return _snapshot(
        chain=_chain(),
        catalog=_catalog(),
        machine=Machine(rf_on=1, stored_current=0.25),
        setup=SetupState(applied_at=1.0, by="alice", settings_sha="e" * 64, turns_tracked=250),
        solve=_solve(),
        reference=reference,
        retention=100,
        n_shots_max=8,
        notices=["setup voided: pilot_tone moved"],
    )


def _filled_shot() -> ShotResult:
    return _shot(
        readout="pseudo_orbit",
        average="nanmean",
        n_turns=10,
        loss_cutoff=0.5,
        excluded_bpms=["BPM3"],
        setpoints={"HCM1": 0.1, "VCM1": -0.2},
        setup_sha="e" * 64,
        beam_current=0.25,
        raw_samples=260,
        parent_id="shot-0",
        fresh=False,
    )


#: Each typed model of the end-state shape, filled, for the schema checks.
def _typed_models() -> list[BaseModel]:
    return [
        _filled_snapshot(),
        _filled_shot(),
        ChainResult(steps_run=["arm", "inject"], commands_fired=["SR:INJ:REQ"], shot=None),
        ChainResult(steps_run=["readout"], commands_fired=[], shot=_filled_shot()),
        ChainBody(steps=["arm", "inject", "readout"]),
        ChainBody(steps=["readout"], readout="pseudo_orbit", n_turns=50, n_shots=3),
        SetupBody(settings={"pilot_tone": "on", "turns_tracked": 250}, excluded_bpms=["BPM3"]),
        SetupBody(),
        SetupReadbacks(
            settings=[
                SettingReadback(id="turns_tracked", scope="global", value=250),
                SettingReadback(id="pilot_tone", scope="device", values={"BPM1": "on"}),
            ],
            read_at=2.0,
        ),
        SetupResult(
            moved=[Moved(name="SR:BPM1:PT", before="off", after="on", outcome="written")],
            unsettled=["SR:BPM2:PT"],
            setup=None,
        ),
        SolveConfigPartial(alpha=0.1, keep_live=2),
        SolveConfigPartial(),
        ReferenceImport(
            name="gold",
            readings={"BPM1": ReferenceReading(x=0.0, y=None)},
            readout="tbt",
            select=True,
            source_target="va",
            source_model="sr",
        ),
        ApplyResult(
            step_digest="d" * 64,
            shot_id="shot-1",
            moved=[Moved(name="HCM1", before=0.1, after=0.2, outcome="written")],
        ),
    ]


def test_a_snapshot_without_any_new_field_still_validates() -> None:
    snap = _snapshot()
    assert snap.chain == []
    assert snap.catalog is None
    assert snap.machine is None
    assert snap.setup is None
    assert snap.solve is None
    assert snap.retention is None
    assert snap.n_shots_max is None
    assert snap.notices == []


def test_a_shot_without_any_new_field_is_a_fresh_tbt_shot() -> None:
    shot = _shot()
    assert shot.fresh is True
    assert shot.readout == "tbt"
    assert shot.average == "mean"
    assert shot.excluded_bpms == []
    for name in (
        "n_turns",
        "loss_cutoff",
        "setpoints",
        "setup_sha",
        "beam_current",
        "raw_samples",
        "parent_id",
    ):
        assert getattr(shot, name) is None, name


def test_the_snapshot_carries_every_end_state_field() -> None:
    for name in (
        "chain",
        "catalog",
        "machine",
        "setup",
        "solve",
        "reference",
        "retention",
        "n_shots_max",
    ):
        assert name in Snapshot.model_fields, name


def test_the_shot_result_carries_every_end_state_field() -> None:
    for name in (
        "readout",
        "average",
        "n_turns",
        "loss_cutoff",
        "excluded_bpms",
        "setpoints",
        "setup_sha",
        "beam_current",
        "raw_samples",
        "parent_id",
        "fresh",
    ):
        assert name in ShotResult.model_fields, name


def test_the_solve_and_its_config_name_every_published_knob() -> None:
    assert set(Solve.model_fields) == {
        "sha",
        "singular_values",
        "inverse_weights",
        "kind",
        "L",
        "live_rank",
        "rm",
        "config",
    }
    assert set(SolveConfig.model_fields) == {
        "alpha",
        "keep_live",
        "n_discard",
        "gain",
        "excluded_bpms",
        "excluded_cms",
        "loss_cutoff",
        "overridden",
    }
    assert set(SetupState.model_fields) >= {"applied_at", "by", "settings_sha", "turns_tracked"}
    assert set(Machine.model_fields) == {"rf_on", "stored_current"}
    assert set(Catalog.model_fields) == {"bpms", "hcms", "vcms", "groups"}
    assert set(CatalogEntry.model_fields) == {
        "name",
        "device",
        "label",
        "places",
        "s",
        "plane",
        "groups",
    }
    assert "families" not in Snapshot.model_fields
    assert set(ChainResult.model_fields) == {"steps_run", "commands_fired", "shot"}


def test_the_retired_fields_are_gone_from_the_contract() -> None:
    retired = {"rank_discarded", "singular_values", "inverse_weights", "spectrum_kind", "n_discard"}
    assert not retired & set(Step.model_fields)
    assert "max_shots" not in ContinuousStatus.model_fields
    with pytest.raises(ValidationError):
        ContinuousStatus(shots=1, lease_expires_in_s=None, max_shots=50)  # type: ignore[call-arg]
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert not retired & set(schema["$defs"]["Step"]["properties"])
    assert "max_shots" not in schema["$defs"]["ContinuousStatus"]["properties"]
    ends = json.dumps(schema["$defs"]["ContinuousEndedEvent"]["properties"]["end"])
    assert "max_shots" not in ends and "zero_step" not in ends


def test_a_moved_setting_reports_the_value_written() -> None:
    moved = Moved(name="HCM1", before=0.1, after=0.2, outcome="written")
    assert json.loads(moved.model_dump_json())["after"] == pytest.approx(0.2)


def test_a_chain_body_runs_at_least_one_step_in_a_known_readout() -> None:
    body = ChainBody(steps=["arm", "inject", "readout"])
    assert (body.readout, body.n_shots, body.n_turns, body.average) == ("tbt", 1, 1, "mean")
    with pytest.raises(ValidationError):
        ChainBody(steps=[])
    with pytest.raises(ValidationError):
        ChainBody(steps=["readout"], readout="closed_orbit")
    with pytest.raises(ValidationError):
        ChainBody(steps=["readout"], n_shots=0)
    with pytest.raises(ValidationError):
        ChainBody(steps=["readout"], n_turns=0)
    with pytest.raises(ValidationError):
        ChainBody(steps=["readout"], max_shots=5)  # type: ignore[call-arg]


def test_a_setup_body_is_a_partial_update() -> None:
    body = SetupBody(settings={"turns_tracked": 250})
    assert body.excluded_bpms is None
    assert SetupBody().settings == {}


def test_a_solve_config_partial_sends_only_its_own_keys() -> None:
    partial = SolveConfigPartial(gain=0.5)
    assert partial.model_dump(exclude_unset=True) == {"gain": 0.5}
    assert SolveConfigPartial().model_dump(exclude_unset=True) == {}
    with pytest.raises(ValidationError):
        SolveConfigPartial(n_discard=2)  # type: ignore[call-arg]


def test_a_reference_import_round_trips_through_its_wire_form() -> None:
    body = ReferenceImport(
        name="gold",
        readings={"BPM1": ReferenceReading(x=math.nan, y=2e-4)},
        readout="tbt",
    )
    wire = json.loads(body.model_dump_json())
    assert wire["readings"]["BPM1"] == {"x": None, "y": pytest.approx(2e-4)}
    assert wire["select"] is False
    again = ReferenceImport.model_validate_json(body.model_dump_json())
    assert again.model_dump_json() == body.model_dump_json()


def test_the_snapshot_reference_carries_its_mode_and_readings() -> None:
    assert {"readout", "readings"} <= set(ReferenceLoad.model_fields)
    assert _reference().readout is None
    assert _reference().readings is None


@pytest.mark.parametrize("model", _typed_models(), ids=lambda m: type(m).__name__)
def test_every_typed_model_validates_against_its_committed_definition(model: BaseModel) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    definition = {"$defs": schema["$defs"], "$ref": f"#/$defs/{type(model).__name__}"}
    jsonschema.validate(json.loads(model.model_dump_json()), definition)


@pytest.mark.parametrize("model", _typed_models(), ids=lambda m: type(m).__name__)
def test_every_typed_model_round_trips_through_its_wire_form(model: BaseModel) -> None:
    again = type(model).model_validate_json(model.model_dump_json())
    assert again.model_dump_json() == model.model_dump_json()


def test_the_schema_defines_every_typed_model() -> None:
    defs = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))["$defs"]
    for name in (
        "ChainResult",
        "ChainBody",
        "ChainStep",
        "Catalog",
        "CatalogEntry",
        "CatalogGroup",
        "CatalogPlace",
        "Machine",
        "SetupState",
        "SetupBody",
        "SetupReadbacks",
        "SetupResult",
        "Solve",
        "SolveConfig",
        "SolveConfigPartial",
        "RmSummary",
        "ReferenceImport",
        "ApplyResult",
    ):
        assert name in defs, name


def test_the_catalog_reads_the_build_catalog_into_ring_order() -> None:
    def raw(name: str, s: float | None) -> dict:
        return {
            "name": name,
            "device": name,
            "label": name,
            "places": [],
            "s": s,
            "plane": None,
            "groups": [],
        }

    built = {
        "bpms": [raw("B1", 5.0), raw("B2", None), raw("B3", 1.0), raw("B4", 1.0)],
        "hcms": [],
        "vcms": [],
        "groups": [{"id": "g", "label": "G", "members": ["B1"]}],
    }
    catalog = Catalog.in_position_order(built)
    # s first, ties and the unplaced in matrix order, the unplaced last
    assert catalog.names("bpms") == ["B3", "B4", "B1", "B2"]
    assert catalog.groups[0].members == ["B1"]
    with pytest.raises(ValidationError):
        Catalog.in_position_order({**built, "bpms": [{"name": "B1"}]})


def test_the_snapshot_catalog_matches_the_committed_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    definition = {"$defs": schema["$defs"], "$ref": "#/$defs/Snapshot"}
    wire = json.loads(_filled_snapshot().model_dump_json())
    jsonschema.validate(wire, definition)
    assert "families" not in schema["$defs"]["Snapshot"]["properties"]
    assert "Families" not in schema["$defs"]
    bad = {**wire, "catalog": {"bpms": ["BPM1"], "hcms": [], "vcms": []}}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, definition)


def test_a_filled_snapshot_event_validates_against_the_committed_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    event = SnapshotEvent(snapshot=_filled_snapshot())
    jsonschema.validate(json.loads(event.model_dump_json()), schema)
    assert parse_event(event.model_dump_json()).model_dump_json() == event.model_dump_json()
