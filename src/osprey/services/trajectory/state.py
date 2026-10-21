"""The trajectory service's state contract: results, the snapshot, the SSE events.

Pure Pydantic models, import-clean of the connector and the action layer, so the
service, the MCP server and the tests share one spelling of every wire shape:

- **Results** — :class:`ShotResult` (one action, its injections averaged),
  :class:`ChainResult` (one chain run), :class:`Step` (the pending correction,
  fields as ``Trajectory.correct`` reports them), :class:`Moved` (one
  non-refused setting write, in put-back shape), :class:`ReferenceLoad` (a
  reference matched by BPM name), :class:`SetupResult`,
  :class:`SetupReadbacks` and :class:`ApplyResult`.
- **The request bodies** — :class:`ChainBody` (a chain run, also the
  continuous start), :class:`SetupBody`, :class:`SolveConfigPartial` and
  :class:`ReferenceImport`.
- **The snapshot** — :class:`Snapshot`, what ``GET /state`` answers and the
  ``snapshot`` event carries: every fact the panel's states are drawn from,
  among them the declared :class:`ChainStep` list, the build's device
  :class:`Catalog`, the :class:`Machine` readings, the :class:`SetupState`
  and the published :class:`Solve`. It never carries a ``client_id``.
- **The events** — one model per SSE event, told apart by ``type`` (which is
  also the SSE ``event:`` name); :data:`Event` is their union and
  :func:`parse_event` reads one back.

An *action* is one chain run of ``n_shots`` *injections*. A BPM without beam
reads NaN in memory and ``null`` on the wire.

``panel/state.schema.json`` is :func:`render_schema`'s output (every event and,
in ``$defs``, every model here), committed beside
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
    "ApplyResult",
    "Average",
    "ChainBody",
    "ChainResult",
    "ChainStep",
    "CommandWrite",
    "ContinuousEnd",
    "ContinuousEndedEvent",
    "ContinuousStatus",
    "Catalog",
    "CatalogEntry",
    "CatalogGroup",
    "CatalogPlace",
    "Event",
    "HolderView",
    "LastAction",
    "LimitsPosture",
    "Machine",
    "Mode",
    "ModelKind",
    "Moved",
    "Readout",
    "ReferenceImport",
    "ReferenceLoad",
    "ReferenceReading",
    "RmSummary",
    "RunningAction",
    "SessionKind",
    "SettingReadback",
    "SettingScope",
    "SettingValue",
    "SetupBody",
    "SetupReadbacks",
    "SetupResult",
    "SetupState",
    "ShotEvent",
    "ShotFailure",
    "ShotResult",
    "ShotSummary",
    "Snapshot",
    "SnapshotEvent",
    "Solve",
    "SolveConfig",
    "SolveConfigPartial",
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
    "taken_over",
    "released",
    "shutdown",
    "failed",
]

#: A value a setting or command holds.
SettingValue = float | int | str | bool

#: How an action's injections are read: the first turn (``tbt``, which makes a
#: step) or the mean over ``n_turns`` turns (``pseudo_orbit``, which makes none).
Readout = Literal["tbt", "pseudo_orbit"]

#: How an action's included injections are combined.
Average = Literal["mean", "nanmean"]

#: Whether a declared setting is held per BPM device or once for the model.
SettingScope = Literal["device", "global"]

#: One BPM reading; ``None`` (NaN in memory) where the BPM saw no beam.
Reading = float | None


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- results -------------------------------------------------------------------------


class Moved(_Model):
    """One non-refused setting write, in put-back shape.

    Attributes:
        name: The RM column or declared setting address written.
        before: The value read just before the write; putting it back undoes it.
        after: The value written, ``None`` where the writer did not report it.
        outcome: What the write did, as the connector reported it.
    """

    name: str
    before: SettingValue
    after: SettingValue | None = None
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
    """One action's readout: its included injections averaged, the failures listed.

    ``x``, ``y`` and ``sum`` are keyed by the sidecar's RM row names; x and y
    are NaN at every BPM in the lost set. A shot's writes are commands, listed
    in ``commands_fired``; it moves no setting. A shot is ``fresh`` when its
    action armed and injected and the machine held still across it; only a
    fresh TbT shot makes a step or becomes a reference.

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
        readout: How the injections were read.
        average: How the included injections were combined.
        n_turns: The turns a pseudo-orbit readout averaged over, ``None`` for
            TbT.
        loss_cutoff: The loss cutoff the shot was extracted with, as a
            fraction of the injected-charge scale.
        excluded_bpms: The BPMs the action left out; they read NaN.
        setpoints: The RM-column corrector setpoints read after readout, by
            corrector name.
        setup_sha: The setup identity the action ran under, ``None`` when no
            current setup held.
        beam_current: The stored current read just before inject.
        raw_samples: The samples per BPM row of the action's raw buffer.
        parent_id: The shot this one was re-extracted from, ``None`` for a
            shot read from the machine.
        fresh: Whether the shot may make a step and become a reference.
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
    readout: Readout = "tbt"
    average: Average = "mean"
    n_turns: int | None = None
    loss_cutoff: float | None = None
    excluded_bpms: list[str] = Field(default_factory=list)
    setpoints: dict[str, float] | None = None
    setup_sha: str | None = None
    beam_current: float | None = None
    raw_samples: int | None = None
    parent_id: str | None = None
    fresh: bool = True


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
    shot_id: str
    reference: str | None


class ReferenceReading(_Model):
    """One BPM's reference reading; ``None`` (NaN in memory) where it saw no beam."""

    x: Reading
    y: Reading


class ReferenceLoad(_Model):
    """A reference matched by BPM name over the current BPM selection.

    Attributes:
        name: The reference's name.
        sha256: The sha256 of the reference file's content.
        matched: BPMs in both the selection and the reference.
        missing_in_reference: Selected BPMs the reference lacks (they read NaN).
        unused_in_reference: Reference BPMs outside the selection.
        nan_count: Readings in the reference that are NaN.
        readout: The readout mode the reference was taken in; a reference is
            never used across modes.
        readings: The reference's reading by selected BPM, which the panel
            subtracts for dx and dy.
    """

    name: str
    sha256: str
    matched: list[str]
    missing_in_reference: list[str]
    unused_in_reference: list[str]
    nan_count: int
    readout: Readout | None = None
    readings: dict[str, ReferenceReading] | None = None


class ChainResult(_Model):
    """One chain run: the steps it ran, the commands it wrote, the shot it read.

    Attributes:
        steps_run: The ids of the steps run, in chain order.
        commands_fired: The command addresses written.
        shot: The shot the readout step stored, ``None`` for a chain without
            readout.
    """

    steps_run: list[str]
    commands_fired: list[str]
    shot: ShotResult | None


class SettingReadback(_Model):
    """One declared setting as read back.

    Attributes:
        id: The setting's declared id.
        scope: ``device`` (read per BPM) or ``global`` (read once).
        value: A global setting's reading, ``None`` when unread.
        values: A device setting's reading by BPM device, ``None`` where unread.
    """

    id: str
    scope: SettingScope
    value: SettingValue | None = None
    values: dict[str, SettingValue | None] = Field(default_factory=dict)


class SetupState(_Model):
    """The setup the service last applied and still holds as current.

    Attributes:
        applied_at: When the setup was applied, seconds since the epoch.
        by: The operator who applied it, ``None`` without an identity.
        settings_sha: The identity of the settings applied; a shot carries it.
        turns_tracked: The turns tracked the setup applied.
        void: Whether a re-check found the machine diverged from it; a void
            setup admits no panel chain run.
    """

    applied_at: float
    by: str | None
    settings_sha: str
    turns_tracked: int
    void: bool = False


class SetupReadbacks(_Model):
    """Every declared setting as read back, which ``GET`` on the setup route answers.

    Attributes:
        settings: One readback per declared setting, in declared order.
        read_at: When they were read, seconds since the epoch.
        setup: The current setup, ``None`` when none holds.
    """

    settings: list[SettingReadback]
    read_at: float | None = None
    setup: SetupState | None = None


class SetupResult(_Model):
    """A setup action: what moved, what did not settle, the setup it left.

    Attributes:
        moved: Each setting write, in put-back shape.
        unsettled: The setting addresses whose readback missed the tolerance in
            time.
        setup: The setup now current, ``None`` when it did not take.
    """

    moved: list[Moved]
    unsettled: list[str] = Field(default_factory=list)
    setup: SetupState | None = None


class ApplyResult(_Model):
    """An apply: a step (``step_digest``) or a put-back (``None``) and what it moved.

    Attributes:
        step_digest: The digest of the step applied, ``None`` for a put-back.
        shot_id: The shot the step was computed from.
        moved: Each corrector or setting written, before and after.
        stopped: Whether the apply ended on Stop.
    """

    step_digest: str | None
    shot_id: str | None
    moved: list[Moved]
    stopped: bool = False


# --- snapshot parts --------------------------------------------------------------------


class HolderView(_Model):
    """A session holder as everyone may see it: the owner and the door."""

    owner: str
    kind: SessionKind


class TakenOver(_Model):
    """The latest take-over: "taken over by <kind> of <owner>"."""

    previous: HolderView
    by: HolderView


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
        lease_expires_in_s: Seconds until the lease lapses without a heartbeat.
    """

    shots: int
    lease_expires_in_s: float | None


class CommandWrite(_Model):
    """A declared command: the address written and the value written to it."""

    address: str
    value: SettingValue


class ChainStep(_Model):
    """One step of the declared chain.

    Attributes:
        id: The step's id; a chain body names steps by it.
        label: The step's label.
        description: What the step does, as declared.
        command: The command a command step writes, ``None`` for a step the
            service runs itself.
        default: Whether the step is ticked by default.
        position: The step's place in the chain, from 0.
        timeout_s: The command's bound, ``None`` for a service-run step.
        done: The readback a command step waits on up to ``timeout_s``,
            ``None`` when it writes and continues.
    """

    id: str
    label: str | None
    description: str | None
    command: CommandWrite | None
    default: bool
    position: int
    timeout_s: float | None
    done: CommandWrite | None = None


class CatalogPlace(_Model):
    """One link of a device's place chain.

    Attributes:
        id: The place's id.
        level: The facility's own word for the place's level, ``None`` when it
            states none.
    """

    id: str
    level: str | None


class CatalogEntry(_Model):
    """One device the response matrix covers, as the facility names it.

    Attributes:
        name: The matrix name the service addresses the device by: a BPM's
            x-row address, a corrector's setpoint address.
        device: The facility device id.
        label: The name a person reads for the device.
        places: The places holding the device, root to leaf.
        s: Its position along the model in metres, ``None`` when unknown.
        plane: ``x`` or ``y`` for a corrector, ``None`` for a BPM.
        groups: The ids of the facility groups listing the device.
    """

    name: str
    device: str
    label: str
    places: list[CatalogPlace] = Field(default_factory=list)
    s: float | None = None
    plane: Literal["x", "y"] | None = None
    groups: list[str] = Field(default_factory=list)


class CatalogGroup(_Model):
    """A facility group listing matrix devices.

    Attributes:
        id: The group's id.
        label: The name a person reads for the group.
        members: The matrix names of its devices.
    """

    id: str
    label: str
    members: list[str]


class Catalog(_Model):
    """The devices the response matrix covers, each family in ring order.

    Ring order is ``s``, then matrix order; a device without ``s`` follows the
    placed ones, in matrix order. Selection addresses a device by its
    ``name``; an excluded device keeps its entry.

    Attributes:
        bpms: The BPMs, one per matrix row pair.
        hcms: The horizontal correctors.
        vcms: The vertical correctors.
        groups: The facility groups listing matrix devices.
    """

    bpms: list[CatalogEntry]
    hcms: list[CatalogEntry]
    vcms: list[CatalogEntry]
    groups: list[CatalogGroup] = Field(default_factory=list)

    @classmethod
    def in_ring_order(cls, catalog: Any) -> Self:
        """The catalog the build wrote (matrix order), each family put in ring order.

        Raises:
            pydantic.ValidationError: *catalog* is not a catalog.
        """
        read = cls.model_validate(catalog)

        def ring(entries: list[CatalogEntry]) -> list[CatalogEntry]:
            order = sorted(
                range(len(entries)),
                key=lambda i: (entries[i].s is None, entries[i].s or 0.0, i),
            )
            return [entries[i] for i in order]

        return read.model_copy(
            update={"bpms": ring(read.bpms), "hcms": ring(read.hcms), "vcms": ring(read.vcms)}
        )

    def names(self, family: Literal["bpms", "hcms", "vcms"]) -> list[str]:
        """The matrix names of one family, in ring order."""
        return [entry.name for entry in getattr(self, family)]


class Machine(_Model):
    """The machine readings the header shows; ``None`` until read or undeclared.

    Attributes:
        rf_on: The RF-on readback, one of its declared options.
        stored_current: The stored-current readback.
    """

    rf_on: SettingValue | None = None
    stored_current: float | None = None


class RmSummary(_Model):
    """The response matrix the solve used.

    Attributes:
        source: Where the matrix came from; the build-time model matrix.
        rows: BPM rows after the exclusions.
        h_cols: Horizontal corrector columns after the exclusions.
        v_cols: Vertical corrector columns after the exclusions.
        computed_at: When it was restricted, seconds since the epoch.
    """

    source: Literal["model"]
    rows: int
    h_cols: int
    v_cols: int
    computed_at: float | None = None


class SolveConfig(_Model):
    """The SolveConfig in force for this target and model.

    Attributes:
        alpha: The Tikhonov damping.
        keep_live: How many live singular values the solve keeps.
        n_discard: The discard count ``keep_live`` stands for over the current
            exclusions.
        gain: The fraction of the step applied.
        excluded_bpms: The excluded BPMs, by name.
        excluded_cms: The excluded correctors, by name.
        loss_cutoff: The loss cutoff, a fraction of the injected-charge scale.
        overridden: The keys an operator override sets; the rest are the file
            defaults.
    """

    alpha: float
    keep_live: int
    n_discard: int
    gain: float
    excluded_bpms: list[str]
    excluded_cms: list[str]
    loss_cutoff: float
    overridden: list[str]


class Solve(_Model):
    """The published solve: its spectrum, dimensions, matrix and config.

    Attributes:
        sha: The solve's identity; a step's digest binds it.
        singular_values: The spectrum, index-aligned with the two below.
        inverse_weights: The damped inverse weight of each value.
        kind: The kind of each value.
        L: How many singular values the restricted matrix has.
        live_rank: How many of them are live.
        rm: The matrix the solve used.
        config: The SolveConfig it was solved with.
    """

    sha: str
    singular_values: list[float]
    inverse_weights: list[float]
    kind: list[SpectrumKind]
    L: int
    live_rank: int
    rm: RmSummary
    config: SolveConfig


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
        chain: The declared chain, in order.
        catalog: The devices the build's response matrix covers, ``None``
            when the build wrote no catalog for the model.
        machine: The machine readings, ``None`` until read.
        setup: The current setup, ``None`` when none was applied.
        solve: The published solve, ``None`` before one is computed.
        retention: How many shots' raw buffers are kept; the panel's record
            cap.
        n_shots_max: The largest ``n_shots`` the action budget admits.
        notices: What the service voided or dropped since, for the operator.
    """

    target: str
    caller: str | None
    writes_enabled: bool
    posture: str
    limits: LimitsPosture
    writing_available: bool
    model: str | None
    model_kind: ModelKind
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
    chain: list[ChainStep] = Field(default_factory=list)
    catalog: Catalog | None = None
    machine: Machine | None = None
    setup: SetupState | None = None
    solve: Solve | None = None
    retention: int | None = None
    n_shots_max: int | None = None
    notices: list[str] = Field(default_factory=list)


# --- request bodies ------------------------------------------------------------------


class ChainBody(_Model):
    """A chain run on the panel door; also the body a continuous run starts with.

    Attributes:
        steps: The ticked step ids; the service runs them in chain order.
        readout: How a readout step reads the injections.
        n_turns: The turns a pseudo-orbit readout averages over.
        n_shots: The injections in the action.
        average: How the included injections are combined.
    """

    steps: list[str] = Field(min_length=1)
    readout: Readout = "tbt"
    n_turns: int = Field(default=1, ge=1)
    n_shots: int = Field(default=1, ge=1)
    average: Average = "mean"


class SetupBody(_Model):
    """A setup: a partial update, every absent key left as it is.

    Attributes:
        settings: The value per declared setting id to apply.
        excluded_bpms: The BPM exclusions to hand the SolveConfig writer,
            ``None`` to leave them.
    """

    settings: dict[str, SettingValue] = Field(default_factory=dict)
    excluded_bpms: list[str] | None = None


class SolveConfigPartial(_Model):
    """A SolveConfig update: each caller sends only its own keys; empty recomputes."""

    alpha: float | None = None
    keep_live: int | None = None
    gain: float | None = None
    excluded_bpms: list[str] | None = None
    excluded_cms: list[str] | None = None
    loss_cutoff: float | None = None


class ReferenceImport(_Model):
    """A reference file's content; the import body and the readings route's answer.

    Attributes:
        name: The name to store it under.
        readings: One reading per BPM.
        readout: The readout mode it was taken in.
        select: Whether to make it the current reference.
        source_target: The target it was taken on, ``None`` when unknown.
        source_model: The model it was taken for, ``None`` when unknown.
    """

    name: str
    readings: dict[str, ReferenceReading]
    readout: Readout
    select: bool = False
    source_target: str | None = None
    source_model: str | None = None


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
    event: BaseModel = _EVENTS.validate_json(data)
    return event


#: Models the panel reads or sends outside any event, defined beside them.
_STANDALONE: tuple[type[BaseModel], ...] = (
    ChainResult,
    SetupReadbacks,
    SetupResult,
    ApplyResult,
    ChainBody,
    SetupBody,
    SolveConfigPartial,
    ReferenceImport,
)


def state_schema() -> dict[str, Any]:
    """The JSON Schema of every SSE event; ``$defs`` holds every model.

    Raises:
        RuntimeError: Two models spell one definition two ways.
    """
    schema = _EVENTS.json_schema(mode="serialization")
    defs: dict[str, Any] = schema.setdefault("$defs", {})
    for model in _STANDALONE:
        own = model.model_json_schema(mode="serialization")
        for name, definition in {**own.pop("$defs", {}), model.__name__: own}.items():
            if defs.setdefault(name, definition) != definition:
                raise RuntimeError(f"the schema defines {name} two ways")
    schema["$defs"] = dict(sorted(defs.items()))
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
