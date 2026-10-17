"""The trajectory service's state contract: results, the snapshot, the SSE events.

Pure Pydantic models, import-clean of the connector and the action layer, so the
service, the MCP server and the tests share one spelling of every wire shape:

- **Results** — :class:`ShotResult` (one shot action, averaged),
  :class:`Step` (the pending correction, fields as ``Trajectory.correct``
  reports them), :class:`Moved` (one non-refused setting write, in put-back
  shape) and :class:`ReferenceLoad` (a reference matched by BPM name).
- **The snapshot** — :class:`Snapshot`, what ``GET /state`` answers and the
  ``snapshot`` event carries: every fact the panel's states are drawn from. It
  never carries a ``client_id``.
- **The events** — one model per SSE event, told apart by ``type`` (which is
  also the SSE ``event:`` name); :data:`Event` is their union and
  :func:`parse_event` reads one back.

A BPM without beam reads NaN in memory and ``null`` on the wire.

``panel/state.schema.json`` is :func:`render_schema`'s output, committed beside
the panel bundle and served with it; ``python -m
osprey.services.trajectory.state`` rewrites it, and a test compares the bytes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

__all__ = [
    "SCHEMA_PATH",
    "ApplyEvent",
    "ContinuousEnd",
    "ContinuousEndedEvent",
    "ContinuousStatus",
    "Event",
    "HolderView",
    "LastAction",
    "LimitsPosture",
    "Mode",
    "ModelKind",
    "Moved",
    "ReferenceLoad",
    "RunLockHolder",
    "RunningAction",
    "SessionKind",
    "ShotEvent",
    "ShotFailure",
    "ShotResult",
    "ShotSummary",
    "Snapshot",
    "SnapshotEvent",
    "SpectrumKind",
    "Step",
    "StepEvent",
    "StepReason",
    "StoppedEvent",
    "TakenOver",
    "TakenOverEvent",
    "parse_event",
    "render_schema",
    "state_schema",
]

#: The committed schema, beside the panel bundle that reads it.
SCHEMA_PATH = Path(__file__).resolve().parent / "panel" / "state.schema.json"

#: The door a session was opened or taken over through.
SessionKind = Literal["panel", "agent"]

#: What the service is doing: nothing, one action, or a continuous run.
Mode = Literal["idle", "action", "continuous"]

#: Whether a model carries the trajectory kind, and of which shape.
ModelKind = Literal["none", "single_pass"]

#: The limits posture the service's actions run under; ``unavailable`` when the
#: configuration could not be read or the limits database failed to load.
LimitsPosture = Literal["off", "optional", "exclusive", "unavailable"]

#: ``Trajectory.correct``'s reason, the first match in this order.
StepReason = Literal[
    "NO_BEAM", "NO_INVERSE", "NO_UPSTREAM", "NO_UPSTREAM_H", "NO_UPSTREAM_V", "ON_REFERENCE", "OK"
]

#: One entry of the tool's guarded, damped singular spectrum.
SpectrumKind = Literal["LIVE", "RANK_DISCARDED", "DISCARDED", "STRUCTURAL"]

#: Why a continuous run ended.
ContinuousEnd = Literal[
    "stopped",
    "lease_expired",
    "max_shots",
    "zero_step",
    "taken_over",
    "released",
    "shutdown",
    "failed",
]

#: A value a setting holds before a write.
SettingValue = float | int | str | bool

#: One BPM reading; ``None`` (NaN in memory) where the BPM saw no beam.
Reading = float | None


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- results -------------------------------------------------------------------------


class Moved(_Model):
    """One non-refused setting write, in put-back shape.

    Attributes:
        name: The RM column or declared module-slot address written.
        before: The value read just before the write; putting it back undoes it.
        outcome: What the write did, as the connector reported it.
    """

    name: str
    before: SettingValue
    outcome: str


class ShotFailure(_Model):
    """One shot of a shot action that was excluded, never retried.

    Attributes:
        index: The shot's position in the action, from 0.
        reason: Why it failed (armed never set or cleared, stale buffer, no BPM
            rose, first-turn sum below the threshold, a blocked command).
        stragglers: The BPMs whose ``armed`` flag missed the deadline.
    """

    index: int
    reason: str
    stragglers: list[str] = Field(default_factory=list)


class ShotResult(_Model):
    """One shot action: its included shots averaged, the failures listed.

    ``x``, ``y`` and ``sum`` are keyed by the sidecar's RM row names; x and y
    are NaN at every BPM in the lost set. A shot's writes are commands, listed
    in ``commands_fired``; it moves no setting.

    Attributes:
        shot_id: The shot's id; a step carries it.
        model: The resolved model the shot was read for.
        x: Horizontal first-turn position by x-row name.
        y: Vertical first-turn position by y-row name.
        sum: First-turn sum by sum-row name.
        bpm_order: The BPMs in rotated s-order.
        lost: The lost set: the first non-rising BPM and every BPM after it.
        last_bpm: The last BPM with beam, ``None`` when no shot was included.
        i0: The first sample with beam, ``None`` when no shot was included.
        window: The raw window ``[start, stop)`` in samples.
        per_shot_lost_at: Per included shot, the first lost BPM (``None``: a
            full turn).
        shots_included: How many shots were averaged.
        shots_failed: The excluded shots and why.
        commands_fired: The command addresses written.
        timestamp: When the shot was read, seconds since the epoch.
    """

    shot_id: str
    model: str
    x: dict[str, Reading]
    y: dict[str, Reading]
    sum: dict[str, Reading]
    bpm_order: list[str]
    lost: list[str]
    last_bpm: str | None
    i0: int | None
    window: tuple[int, int] | None
    per_shot_lost_at: list[str | None]
    shots_included: int
    shots_failed: list[ShotFailure]
    commands_fired: list[str]
    timestamp: float | None = None


class Step(_Model):
    """The pending correction, as ``GET /api/step`` serves it.

    The increment written is ``before + deltas[name]``, with
    ``δ = −factor·step``. Apply refuses on ``all_zero``, never on ``reason``.

    Attributes:
        digest: The service-minted digest an apply must carry.
        deltas: The increment by corrector name.
        factor: The uniform ``max_step`` scale, at most 1.
        reason: The tool's reason.
        status: The tool's sentence for the reason.
        all_zero: Whether every delta is zero.
        zeroed_h: Horizontal correctors zeroed, in column order.
        zeroed_v: Vertical correctors zeroed, in column order.
        no_response_names: Correctors with a zero RM column.
        unsolved: The sidecar's correctors with no response.
        rank_discarded: Live singular values the rank guard dropped.
        singular_values: The spectrum, index-aligned with the two below.
        inverse_weights: The damped inverse weight of each value.
        spectrum_kind: The kind of each value.
        n_discard: The ``n_discard`` the step was solved with.
        shot_id: The shot the step was computed from.
        reference: The reference's name, ``None`` when solved to zero.
    """

    digest: str
    deltas: dict[str, float]
    factor: float
    reason: StepReason
    status: str
    all_zero: bool
    zeroed_h: list[str]
    zeroed_v: list[str]
    no_response_names: list[str]
    unsolved: list[str]
    rank_discarded: int
    singular_values: list[float]
    inverse_weights: list[float]
    spectrum_kind: list[SpectrumKind]
    n_discard: int
    shot_id: str
    reference: str | None


class ReferenceLoad(_Model):
    """A reference matched by BPM name over the current BPM selection.

    Attributes:
        name: The reference's name.
        sha256: The sha256 of the reference file's content.
        matched: BPMs in both the selection and the reference.
        missing_in_reference: Selected BPMs the reference lacks (they read NaN).
        unused_in_reference: Reference BPMs outside the selection.
        nan_count: Readings in the reference that are NaN.
    """

    name: str
    sha256: str
    matched: list[str]
    missing_in_reference: list[str]
    unused_in_reference: list[str]
    nan_count: int


# --- snapshot parts --------------------------------------------------------------------


class HolderView(_Model):
    """A session holder as everyone may see it: the owner and the door."""

    owner: str
    kind: SessionKind


class TakenOver(_Model):
    """The latest take-over: "taken over by <kind> of <owner>"."""

    previous: HolderView
    by: HolderView


class RunLockHolder(_Model):
    """Another process holding this target's run lock, as its holder record says."""

    pid: int | None
    started: str | None
    owner: str | None
    target: str | None
    identity: str | None


class RunningAction(_Model):
    """The action in flight.

    Attributes:
        name: The action (``shot``, ``setup``, ``apply``, ``put_back``,
            ``reference``).
        stopping: Whether Stop was set; it ends within ``stop_budget_s``.
        armed_stragglers: BPMs whose ``armed`` flag has not yet answered.
    """

    name: str
    stopping: bool
    armed_stragglers: list[str] = Field(default_factory=list)


class ShotSummary(_Model):
    """The last shot action, in brief."""

    shot_id: str
    model: str
    shots_included: int
    shots_failed: list[ShotFailure]
    armed_stragglers: list[str]
    lost: list[str]
    last_bpm: str | None
    commands_fired: list[str]

    @classmethod
    def of(cls, shot: ShotResult) -> Self:
        """The summary of *shot*; its stragglers are every failed shot's."""
        stragglers = sorted({name for failure in shot.shots_failed for name in failure.stragglers})
        return cls(
            shot_id=shot.shot_id,
            model=shot.model,
            shots_included=shot.shots_included,
            shots_failed=shot.shots_failed,
            armed_stragglers=stragglers,
            lost=shot.lost,
            last_bpm=shot.last_bpm,
            commands_fired=shot.commands_fired,
        )


class LastAction(_Model):
    """The last finished writing action and the settings it moved."""

    name: str
    stopped: bool
    moved: list[Moved]


class ContinuousStatus(_Model):
    """A continuous run in progress.

    Attributes:
        shots: Shot actions submitted so far.
        max_shots: ``continuous_max_shots``.
        lease_expires_in_s: Seconds until the lease lapses without a heartbeat.
    """

    shots: int
    max_shots: int
    lease_expires_in_s: float | None


class Snapshot(_Model):
    """Every fact the panel's states are drawn from; ``GET /state`` answers it.

    Attributes:
        target: The service's bound target (the panel compares it with the
            terminal's own).
        caller: The operator identity the door admitted, ``None`` without one.
        writes_enabled: Whether the posture lets the service write.
        posture: The posture as an operator reads it.
        limits: The limits posture actions run under.
        writing_available: Whether the build gave the service anything to
            command.
        model: The model carrying the trajectory kind, ``None`` when none does.
        model_kind: ``none`` or ``single_pass``.
        run_lock_holder: Another process holding the run lock, else ``None``.
        plan_running: Whether a Bluesky plan runs; ``None`` when the bridge did
            not answer.
        holder: The session holder, ``None`` when nobody holds it.
        taken_over: The latest take-over, ``None`` when there was none.
        mode: Idle, one action, or a continuous run.
        action: The action in flight, ``None`` when idle.
        queue_wait_s: How long a call submitted now would wait to start.
        action_timeout_s: The configured action timeout.
        action_budget_s: The longest an admitted action may take.
        stop_budget_s: The longest an action takes to end after Stop.
        last_shot: The last shot action, in brief.
        step: The pending step, ``None`` when there is none.
        reference: The current reference and its match counts.
        last_action: The last finished writing action and what it moved.
        continuous: The continuous run in progress, else ``None``.
    """

    target: str
    caller: str | None
    writes_enabled: bool
    posture: str
    limits: LimitsPosture
    writing_available: bool
    model: str | None
    model_kind: ModelKind
    run_lock_holder: RunLockHolder | None
    plan_running: bool | None
    holder: HolderView | None
    taken_over: TakenOver | None
    mode: Mode
    action: RunningAction | None
    queue_wait_s: float
    action_timeout_s: float
    action_budget_s: float
    stop_budget_s: float
    last_shot: ShotSummary | None
    step: Step | None
    reference: ReferenceLoad | None
    last_action: LastAction | None
    continuous: ContinuousStatus | None


# --- SSE events ----------------------------------------------------------------------


class SnapshotEvent(_Model):
    """The whole snapshot, sent on connect and on every change."""

    type: Literal["snapshot"] = "snapshot"
    snapshot: Snapshot


class ShotEvent(_Model):
    """A shot action finished."""

    type: Literal["shot"] = "shot"
    shot: ShotResult


class StepEvent(_Model):
    """The pending step was computed (``step``) or voided (``None``)."""

    type: Literal["step"] = "step"
    step: Step | None


class ApplyEvent(_Model):
    """An apply finished: a step (``step_digest``) or a put-back (``None``)."""

    type: Literal["apply"] = "apply"
    step_digest: str | None
    shot_id: str | None
    moved: list[Moved]


class StoppedEvent(_Model):
    """An action ended on Stop; ``moved`` puts back what it wrote."""

    type: Literal["stopped"] = "stopped"
    action: str
    stopper: str | None
    moved: list[Moved]


class TakenOverEvent(_Model):
    """The session changed hands."""

    type: Literal["taken_over"] = "taken_over"
    previous: HolderView
    by: HolderView


class ContinuousEndedEvent(_Model):
    """A continuous run ended, with the last step's reason and status."""

    type: Literal["continuous_ended"] = "continuous_ended"
    end: ContinuousEnd
    shots: int
    step_reason: StepReason | None = None
    step_status: str | None = None


#: Every SSE event, told apart by ``type``.
Event = Annotated[
    SnapshotEvent
    | ShotEvent
    | StepEvent
    | ApplyEvent
    | StoppedEvent
    | TakenOverEvent
    | ContinuousEndedEvent,
    Field(discriminator="type"),
]

_EVENTS: TypeAdapter[Any] = TypeAdapter(Event)


def parse_event(data: str | bytes) -> BaseModel:
    """The event the JSON *data* spells.

    Raises:
        pydantic.ValidationError: *data* is no event of this contract.
    """
    return _EVENTS.validate_json(data)


def state_schema() -> dict[str, Any]:
    """The JSON Schema of every SSE event; ``$defs`` holds every model."""
    schema = _EVENTS.json_schema(mode="serialization")
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "TrajectoryEvent",
        **schema,
    }


def render_schema() -> str:
    """The bytes ``panel/state.schema.json`` holds."""
    return json.dumps(state_schema(), indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    SCHEMA_PATH.write_text(render_schema(), encoding="utf-8")
