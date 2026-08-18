"""Shipped plan: ``orbit_bump_sweep``, a closed local orbit-bump diagnostic.

Discovered via the layered directory catalog's ``shipped`` tier (a folder scan
of this package's ``plans_core/`` dir — see ``plan_loader.py``): drive three or
four correctors together in a combination whose kicks cancel outside the bump
region, so the stored beam is displaced locally and the orbit beyond the bump
comes back to where it was.

**Why several correctors and not one.** A single corrector kick does not stay
local: it launches a betatron oscillation that rings all the way around the
ring, so every BPM sees it. Three (or four) kicks placed across a span can be
chosen so their contributions cancel downstream — the orbit is displaced across
the span and unchanged outside it. That closure is the whole point of the
diagnostic: it is what makes a deliberate local orbit change safe on a machine
that is delivering beam.

**Parameterized in orbit space, not corrector space.** An operator asks for a
displacement *at BPMs* — "this much at these ``targets``, nothing at these
``closure_readbacks``" — and the plan is what finds the kicks that produce it.
That is the only parameterization that means the same thing at two facilities,
or at one facility after a lattice change: a set of corrector currents is a
statement about one machine on one day, whereas "300 microns at BPM 7 and
closure at BPMs 12 and 13" is a statement about the beam.

Solving for ``len(correctors)`` kicks needs at least that many independent
orbit constraints, which is why ``len(targets) + len(closure_readbacks)`` must be at
least ``len(correctors)``: fewer rows than unknowns is an underdetermined solve
whose answer is a direction in kick space, not a bump, and this schema refuses
it rather than letting the runtime pick one of infinitely many.

**No magnitude ceilings live here.** ``probe_amplitude`` is an excursion in
corrector units and the target values are displacements in BPM units; how large
either may be is a property of the deployment, not of this schema. The
connector's reference monitor checks every setpoint against the project's own
``channel_limits.json`` at write time and refuses — aborting the run — what the
machine will not take. A literal bound here would only ever be one facility's
number.

Device-agnostic: every channel is resolved by string name against whatever
``devices`` dict the bridge passes in; nothing here names a facility PV or a
fixed device set. Every channel field declares its role, so every consumer —
the load gate, the enqueue pre-check, the dry run, the pre-flight trajectory —
reads what this plan drives and what it reads rather than guessing it from a
field name.
"""

from __future__ import annotations

import logging
import math
import statistics
from collections.abc import Callable, Iterable, Mapping
from typing import Annotated, Any, Literal, NamedTuple

from bluesky import plan_stubs as bps
from bluesky import preprocessors as bpp
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

# Absolute imports, not relative: the plan loader execs this file by path under
# a synthetic `sys.modules` name with no parent package (see
# `plan_loader._load_plan_module_from_path`), so a relative import raises at
# load time and quarantines the plan.
from osprey.services.bluesky_bridge.bump_analysis import (
    band_converged,
    band_violations,
    demand_is_negligible,
    fit_probe_response,
    noise_floor_violations,
    solve_offsets,
)
from osprey.services.bluesky_bridge.figure import (
    Figure,
    LinesMark,
    Panel,
    Point,
    RowWindow,
    Series,
    decimate,
)
from osprey.services.bluesky_bridge.plan_fields import (
    MovableChannels,
    OptionalReadableChannel,
    ReadableChannel,
    ReadableChannels,
    resolve_column,
    scan_metadata,
)

logger = logging.getLogger("osprey.services.bluesky_bridge.plans_core.orbit_bump_sweep")

PLAN_METADATA = {
    "name": "orbit_bump_sweep",
    "description": (
        "Sweep a closed local orbit bump: three or four correctors driven "
        "together so their kicks cancel outside the bump region, displacing "
        "the beam at the target BPMs while the closure BPMs stay put. Every "
        "corrector is returned to its pre-scan working point."
    ),
    "writes": True,
}


class TargetPoint(BaseModel):
    """One BPM and the orbit displacement asked for there.

    ``value`` carries no magnitude bound of its own — see the module docstring
    on where bounds actually live — but it must be a real number: a non-finite
    demand cannot be refused by a limits check (``nan < low`` and ``nan >
    high`` are both false), so it would reach the machine and fail only as a
    settle timeout.
    """

    readback: ReadableChannel = Field(
        ..., description="BPM channel name this displacement is asked for."
    )
    value: float = Field(
        ...,
        allow_inf_nan=False,
        description="Orbit displacement asked for at this BPM, in BPM units.",
    )


class PARAMS(BaseModel):
    """Parameters for ``orbit_bump_sweep``: the bump's correctors and its orbit goal.

    The bump is defined by what it should do to the orbit — ``targets`` say
    where the beam should move and by how much, ``closure_readbacks`` say
    where it must not move at all — and by which ``correctors`` are allowed to
    do it. ``readbacks`` and ``monitors`` are recorded at every point without
    taking part in the solve, so a run carries the surrounding orbit and any
    other instrument the operator wants alongside it.

    With ``leakage_tolerance`` set, the monitor BPMs are additionally *judged*:
    a step that moves any of them further than that band from the reference
    orbit is a leaking bump, surfaced exactly like a convergence miss. They
    still take no part in the solve — they are watched, not asked for — so a
    leak is a finding about the bump's constraint set, not something a trim
    pass could remove.

    ``mode`` reads the target values either as a displacement *from the orbit
    the machine is already on* (``relative``) or as an absolute orbit position
    (``absolute``). ``sweep`` then decides which side of the working orbit the
    sweep visits: ``monodirectional`` runs from zero out to the requested
    bump, ``bidirectional`` runs symmetrically through it in both directions,
    which is what lets a linear fit reject a BPM offset.

    ``absolute`` with ``bidirectional`` is refused, because the two do not
    compose: negating an absolute orbit position is not "the other side" of
    anything — the reflection is only meaningful about the working orbit, which
    is exactly what an absolute target declines to be measured from.

    Zero-valued targets are legal and deliberately so. Whether a bump is
    actually asking the machine for anything is decided at runtime by the
    demand gate, which alone sees the reference orbit, the measured response,
    and the BPM noise floor; a schema that rejected zeros here would also
    reject the all-zero dry run that proves the plan safe on mock devices.

    The ``x-widget`` schema hints steer the plan panel's parameter GUI —
    device lists render as scrollable channel columns, ``mode`` and ``sweep``
    as two-way segmented controls — without changing what this model validates.
    """

    correctors: MovableChannels = Field(
        ...,
        min_length=3,
        max_length=4,
        title="Correctors",
        description=(
            "The three or four corrector channel names forming the bump. Fewer "
            "cannot close a bump; more is a different diagnostic."
        ),
        json_schema_extra={"x-widget": "channel-list"},
    )
    targets: list[TargetPoint] = Field(
        ...,
        min_length=1,
        title="Bump targets",
        description="BPMs the bump should displace, and by how much.",
    )
    closure_readbacks: ReadableChannels = Field(
        ...,
        title="Closure BPMs",
        description=(
            "BPMs the bump must leave undisturbed. These are what make the "
            "bump closed rather than merely local."
        ),
        json_schema_extra={"x-widget": "channel-list"},
    )
    readbacks: ReadableChannels = Field(
        ...,
        title="Monitor BPMs",
        description=(
            "BPMs recorded at every point but not used in the solve — the "
            "surrounding orbit, for seeing whether the bump stayed put."
        ),
        json_schema_extra={"x-widget": "channel-list"},
    )
    mode: Literal["relative", "absolute"] = Field(
        default="relative",
        title="Target mode",
        description=(
            "relative reads target values as a displacement from the current "
            "orbit; absolute reads them as orbit positions."
        ),
        json_schema_extra={"x-widget": "segmented"},
    )
    sweep: Literal["monodirectional", "bidirectional"] = Field(
        default="monodirectional",
        title="Sweep direction",
        description=(
            "monodirectional sweeps from zero to the requested bump; "
            "bidirectional sweeps symmetrically through it both ways."
        ),
        json_schema_extra={"x-widget": "segmented"},
    )
    num: int = Field(
        ...,
        ge=2,
        title="Number of steps",
        description=(
            "Number of amplitude increments from zero to the requested bump; "
            "the sweep visits 2*num steps (4*num bidirectional)."
        ),
    )
    baseline_reads: int = Field(
        default=5,
        ge=3,
        title="Baseline reads",
        description=(
            "Reads taken before the bump moves anything. These are the "
            "reference orbit and the BPM noise estimate the demand gate needs, "
            "so a handful is a floor, not a preference."
        ),
    )
    probe_amplitude: float = Field(
        ...,
        gt=0,
        allow_inf_nan=False,
        title="Probe amplitude",
        description=(
            "Kick size, in corrector units, used to measure each corrector's "
            "own response before the bump is solved for."
        ),
    )
    tolerance: float = Field(
        ...,
        gt=0,
        allow_inf_nan=False,
        title="Tolerance",
        description=(
            "How close, in BPM units, a reached orbit must be to the requested "
            "one before the bump counts as achieved."
        ),
    )
    max_trim_iterations: int = Field(
        ...,
        ge=1,
        title="Max trim iterations",
        description=(
            "Most correction passes spent bringing the orbit inside tolerance "
            "at one amplitude before the plan gives up on that point."
        ),
    )
    leakage_tolerance: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = Field(
        default=None,
        title="Leakage tolerance",
        description=(
            "Band, in BPM units, the monitor BPMs must stay inside relative "
            "to the reference orbit at every step. Unset, they are recorded "
            "but never judged."
        ),
    )
    best_effort: bool = Field(
        default=False,
        title="Best effort",
        description=(
            "Continue the sweep when a point cannot be trimmed inside "
            "tolerance, recording what was reached, instead of stopping."
        ),
    )
    settle_s: float = Field(
        ...,
        ge=0,
        allow_inf_nan=False,
        title="Settle time (s)",
        description="Wait after each move before reading, in seconds.",
    )
    beam_current_readback: OptionalReadableChannel = Field(
        default=None,
        title="Beam current",
        description=(
            "Channel whose reading guards the sweep against running on lost "
            "beam. Paired with min_beam_current."
        ),
    )
    min_beam_current: Annotated[float, Field(allow_inf_nan=False)] | None = Field(
        default=None,
        title="Minimum beam current",
        description="Reading below which the sweep must not proceed.",
    )
    monitors: ReadableChannels = Field(
        default_factory=list,
        title="Monitors",
        description="Any other channels to record at every point.",
        json_schema_extra={"x-widget": "channel-list"},
    )

    @field_validator("correctors", "closure_readbacks", "readbacks", "monitors")
    @classmethod
    def _names_unique(cls, value: list[str], info: ValidationInfo) -> list[str]:
        """Reject a device named twice in one list.

        A repeat is never what the operator meant, and it is not harmless: a
        repeated corrector is a column of the solve duplicated, which makes the
        system singular in exactly the way the constraint count below exists to
        prevent.
        """
        duplicates = sorted({name for name in value if value.count(name) > 1})
        if duplicates:
            raise ValueError(f"{info.field_name} must be unique (repeated: {duplicates})")
        return value

    @field_validator("targets")
    @classmethod
    def _target_readbacks_unique(cls, value: list[TargetPoint]) -> list[TargetPoint]:
        """Reject two demands on the same BPM — they are two answers to one question."""
        names = [target.readback for target in value]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"targets must name each BPM at most once (repeated: {duplicates})")
        return value

    @model_validator(mode="after")
    def _closure_disjoint_from_targets(self) -> PARAMS:
        """Reject a BPM asked to move and to stay still at the same time."""
        overlap = {target.readback for target in self.targets} & set(self.closure_readbacks)
        if overlap:
            raise ValueError(
                "closure_readbacks and target readbacks must be disjoint "
                f"(overlap: {sorted(overlap)}) — a BPM cannot both be bumped "
                "and held at the reference orbit"
            )
        return self

    @model_validator(mode="after")
    def _constraints_cover_correctors(self) -> PARAMS:
        """Reject a bump with fewer orbit constraints than corrector kicks.

        Each target and each closure BPM is one row of the solve; each
        corrector is one unknown. With fewer rows than unknowns the solve is
        underdetermined — its solution set is a whole direction in kick space,
        and any point of it is as good an answer as any other. Refusing here
        means the operator picks, rather than the numerics.
        """
        rows = len(self.targets) + len(self.closure_readbacks)
        if rows < len(self.correctors):
            raise ValueError(
                f"underdetermined bump: {rows} orbit constraint(s) "
                f"(targets + closure_readbacks) for {len(self.correctors)} correctors; "
                "add closure BPMs or drop a corrector"
            )
        return self

    @model_validator(mode="after")
    def _leakage_check_has_monitors_to_judge(self) -> PARAMS:
        """Reject a leakage band with no BPM it could apply to.

        The leakage check judges the monitor BPMs (``readbacks``) that are not
        already constraint rows — a target or closure BPM is asserted more
        strictly by the solve's own band. A ``leakage_tolerance`` whose every
        candidate is already constrained (or that has no ``readbacks`` at all)
        would be a guard that is on while judging nothing, the same inert-guard
        state the beam-current pairing below refuses.
        """
        if self.leakage_tolerance is None:
            return self
        constrained = {target.readback for target in self.targets} | set(self.closure_readbacks)
        if not (set(self.readbacks) - constrained):
            raise ValueError(
                "leakage_tolerance needs at least one readbacks BPM that is "
                "not already a target or closure BPM — there is nothing else "
                "for it to judge"
            )
        return self

    @model_validator(mode="after")
    def _beam_current_guard_is_paired(self) -> PARAMS:
        """Reject half a beam-current guard.

        A channel with no threshold has nothing to compare against, and a
        threshold with no channel has nothing to read — either alone reads as a
        guard that is on while being inert, which is the worst of the three
        states. Both or neither.
        """
        has_readback = self.beam_current_readback is not None
        has_minimum = self.min_beam_current is not None
        if has_readback != has_minimum:
            raise ValueError(
                "beam_current_readback and min_beam_current must be given together or not at all"
            )
        return self

    @model_validator(mode="after")
    def _absolute_targets_are_single_sided(self) -> PARAMS:
        """Reject ``absolute`` targets swept ``bidirectional``.

        A bidirectional sweep visits both signs of the requested bump, which is
        a reflection *about the working orbit*. An absolute target names an
        orbit position instead, so there is no working orbit to reflect about
        and the negated amplitude names no orbit at all.
        """
        if self.mode == "absolute" and self.sweep == "bidirectional":
            raise ValueError(
                "mode='absolute' cannot be swept bidirectionally: negating an "
                "absolute orbit position is not a bump on the other side"
            )
        return self


def _read_channel_names(params: PARAMS) -> list[str]:
    """Every channel read at each point, in a stable order and named once.

    Targets, closure BPMs, and monitor BPMs are separate roles a single channel
    may hold more than one of (a monitor list is often just "all of them"), and
    ``trigger_and_read`` must be handed each device once — a repeat triggers it
    twice for one event.
    """
    names = [
        *(target.readback for target in params.targets),
        *params.closure_readbacks,
        *params.readbacks,
        *params.monitors,
    ]
    if params.beam_current_readback is not None:
        names.append(params.beam_current_readback)

    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def _constrained_names(params: PARAMS) -> list[str]:
    """The constrained BPMs — targets, then closure — in the one row order.

    Every desired vector, sigma vector, and constraint-row slice of the fitted
    response shares this order, and the render half re-derives the same rows to
    judge the band, so it is spelled once here rather than at each site.
    ``PARAMS`` already rejects a BPM appearing in both lists, so this needs no
    deduplication of its own.
    """
    return [target.readback for target in params.targets] + list(params.closure_readbacks)


def _profile_scales(num: int, sweep: str) -> list[float]:
    """The amplitude profile: every scale factor visited, in visiting order.

    Adjacent scales always differ by exactly one increment of ``1/num``, so
    the machine is never asked to jump more than one amplitude step at a time
    — including across leg boundaries, which is why each descending leg starts
    one increment below the peak the previous leg ended on.

    ``monodirectional``: up ``1/num .. 1``, back down ``(num-1)/num .. 0`` —
    ``2*num`` steps. ``bidirectional``: the same out-and-back, then its mirror
    through the working orbit, ``-1/num .. -1`` and back up to ``0`` —
    ``4*num`` steps. Both profiles end at exactly ``+0.0``: the last step is
    the plan's verification that the machine really came back. The mirrored
    legs are spelled ``0.0 - scale`` rather than ``-scale`` so that end really
    is ``+0.0`` — negating the descending leg's final ``0.0`` yields ``-0.0``,
    which every step and series label formats as "scale -0".
    """
    up = [k / num for k in range(1, num + 1)]
    down = [k / num for k in range(num - 1, -1, -1)]
    if sweep == "monodirectional":
        return up + down
    return up + down + [0.0 - scale for scale in up] + [0.0 - scale for scale in down]


def _row_value(row: Mapping[str, Any], name: str) -> Any:
    """The entry in *row* keyed for channel *name*, or ``None``.

    The exact-then-prefix rule is `plan_fields.resolve_column`'s; this only
    adds the lookup, so every caller holding a row can ask for a value.
    """
    column = resolve_column(name, row)
    return None if column is None else row[column]


def _finite_values(
    readings: Mapping[str, Any], names: Iterable[str], context: str
) -> dict[str, float]:
    """Extract one finite float per channel name from a readings mapping.

    *readings* is what ``bps.read``/``bps.trigger_and_read`` hand back — event
    keys mapped to ``{"value": ..., "timestamp": ...}`` readings, where a key
    is the bare channel name or ``f"{name}-{signal}"`` depending on how many
    readable children the device exposes (the same fuzziness
    `plan_fields.resolve_column` exists for).

    Raises:
        ValueError: A named device is absent from *readings* or read back a
            non-finite value. A non-finite orbit reading poisons every mean,
            fit, and band check downstream while satisfying none of their
            comparisons, so the plan aborts (restoring the correctors) the
            moment one appears rather than trimming toward a NaN.
    """
    values: dict[str, float] = {}
    for name in names:
        entry = _row_value(readings, name)
        if entry is None:
            raise ValueError(
                f"orbit_bump_sweep plan: device {name!r} reported no reading during {context}"
            )
        value = float(entry["value"])
        if not math.isfinite(value):
            raise ValueError(
                f"orbit_bump_sweep plan: device {name!r} read back a non-finite value "
                f"({value}) during {context}; aborting and restoring the correctors"
            )
        values[name] = value
    return values


_SNAPSHOT_GROUP = "orbit_bump_sweep-snapshot"


def _snapshot(devices_to_read: list[Any]) -> Any:
    """Read devices off the record: no event, no data row, just the values.

    The plan's probe and convergence reads must not appear in the run's data
    — the pinned row layout is ``baseline_reads`` baseline rows then exactly
    one row per amplitude step — so this reads outside any event bundle:
    trigger whatever is triggerable (mirroring ``bps.trigger_and_read``'s
    treatment, so a triggered device here reads the same way it does in a
    data row), then plain ``read`` messages with no ``create``/``save`` pair
    around them, which the RunEngine answers without emitting a document.
    """
    readings: dict[str, Any] = {}
    for device in devices_to_read:
        if hasattr(device, "trigger"):
            yield from bps.trigger(device, group=_SNAPSHOT_GROUP, wait=True)
        reading = yield from bps.read(device)
        if reading is not None:
            readings.update(reading)
    return readings


def _snapshot_values(devices_to_read: list[Any], names: list[str], context: str) -> Any:
    """One finite float per name, read off the record: `_snapshot` then `_finite_values`.

    Every off-the-record read in the plan wants both halves, and spelling them
    separately means a ``yield from`` nested inside a call argument at each
    site.
    """
    readings = yield from _snapshot(devices_to_read)
    return _finite_values(readings, names, context)


def build_plan(devices: dict[str, Any], params: PARAMS) -> Any:
    """Build the orbit-bump sweep generator.

    Reads where every corrector is already sitting, takes ``baseline_reads``
    reference points with nothing moved, then measures its way to the bump:

    1. **Baseline.** ``baseline_reads`` full data rows with nothing moved. The
       per-BPM mean over them is the reference orbit every later offset is
       measured against (``mode='absolute'`` targets are converted to
       reference-relative offsets here, once); the per-BPM sample standard
       deviation is the noise estimate. A ``tolerance`` narrower than twice
       that noise at any constrained BPM fails the run *before any sweep
       write* — see ``noise_floor_violations`` for why that band is
       unverifiable.
    2. **Probe.** Each corrector in turn is dithered ``±probe_amplitude``
       about its working point (settling and reading off the record at each
       side, restored before the next corrector is touched) and the
       corrector→BPM response is fitted from the pairs
       (``fit_probe_response``). Whether the run is asking for anything at
       all is then decided once, at run level, on the unscaled targets
       (``demand_is_negligible``): a negligible demand takes the
       trivially-converged path — no solve anywhere, every step's solution
       identically zero — but the profile below is still walked in full,
       probes, verification, data rows, and restore included, which is what
       keeps a physics-free mock dry run structurally identical to a real
       one.
    3. **Profile walk.** The bump amplitude is walked through the scales
       ``_profile_scales`` lays out (never jumping more than ``1/num`` at a
       time). Per step: solve the constraint rows for the corrector offsets
       producing ``scale * desired`` (targets at their scaled values, closure
       BPMs at zero), apply them on top of the working points, settle, and
       verify against the ``±tolerance`` band using the mean of two
       off-the-record reads. A step outside the band is trimmed — re-solve on
       the residual, re-apply, re-check — at most ``max_trim_iterations``
       times; a step that still misses raises (the ``finally`` below
       restores) unless ``best_effort`` is set, in which case the achieved
       orbit is recorded and the sweep continues. With ``leakage_tolerance``
       set, each settled step is then also judged at the monitor BPMs — the
       ``readbacks`` not already constrained — against ``±leakage_tolerance``
       around the reference orbit, with the same miss handling; no trim, since
       the monitors are not in the solve. Every step then emits
       exactly one data row, so a run's rows are ``baseline_reads`` baseline
       rows followed by one row per scale, in profile order.
    4. **Terminal step.** The last step of the profile is scale ``0``, and it
       is commanded as the recorded working points *verbatim* — no solve, no
       trim. It is the run's verification that the machine actually came
       back: an out-of-band reading there is a finding to surface (recorded
       under ``best_effort``, raised otherwise), never something to trim
       away, because trimming would mean ending the run with the correctors
       somewhere other than where they started. Mid-profile scale-0 visits
       (the bidirectional crossing) keep normal solve-and-trim behavior.

    When a beam-current guard is configured, the device is read before every
    write batch — each corrector's probe, each step, each trim pass — and a
    reading below ``min_beam_current`` stops the run (the ``finally`` below
    restores). A non-finite BPM or beam-current reading aborts the same way,
    wherever it appears.

    **Relative, because a machine with beam in it is not at zero.** A ring
    running corrected orbit holds every corrector at a nonzero
    orbit-correction working point, and a bump is a change *on top of* that.
    "Restoring" to a literal 0 A would drop the entire correction the ring was
    holding — on a stored beam, that is the orbit gone. Reading each
    corrector's working point costs one read per corrector and makes the plan
    mean the same thing on a real ring as on a virtual accelerator whose
    correctors happen to idle at zero.

    The read is `bps.rd`, so the working point is the corrector's own readback
    — the right value to restore under this bridge's device contract, where a
    settable's ``set()`` does not complete until the readback agrees with the
    demand (see ``devices/connector.py``'s ``ConnectorSettable``). Every read
    happens BEFORE the ``try``, so a corrector whose read fails is never
    entered at all and the restore below can never run without a target.

    Correctors are restored on every exit path, including a RunEngine abort,
    which is delivered into the plan as an exception, and one corrector's
    restore failure never stops the rest from being put back. What a restore
    failure then *becomes* depends on whether an error is already in flight:
    mid-sweep it is caught and logged rather than allowed to replace the
    exception that actually explains the run, but on a clean completion it is
    raised — a run that ends with a corrector stuck off its working point must
    never read as a success.

    Raises:
        ValueError: A corrector read back a non-finite working point (every
            setpoint derived from it would be non-finite too, and a NaN demand
            is not something a limits check can refuse — ``nan < low`` and
            ``nan > high`` are both false — so it would reach the IOC and only
            then fail as a readback-settle timeout; refusing here means no
            write is attempted at all). Also: a BPM or the beam-current device
            went missing or read back non-finite anywhere in the run, or
            ``tolerance`` is below the measured noise floor at a constrained
            BPM.
        RuntimeError: The beam current fell below ``min_beam_current`` before
            a write batch; a step could not be trimmed inside ``tolerance``
            (with ``best_effort`` unset); a settled step moved a monitor BPM
            outside ``±leakage_tolerance`` (with ``best_effort`` unset); or,
            on an otherwise clean run, a corrector could not be restored to
            its working point.
        DegenerateBumpError: The measured machine cannot answer the bump asked
            of it — see ``bump_analysis``.
    """
    correctors = [(name, devices[name]) for name in params.correctors]
    corrector_devices = [corrector for _, corrector in correctors]
    read_names = _read_channel_names(params)
    read_devices = [devices[name] for name in read_names]
    all_devices = corrector_devices + read_devices

    # This plan opens its own run, so it stamps its own run metadata: the
    # channels it moves and reads, and the total data-row count — the baseline
    # rows plus one row per amplitude step. Probe and convergence reads are
    # off the record and add no rows.
    scales = _profile_scales(params.num, params.sweep)
    run_md = scan_metadata(
        movable=params.correctors,
        readable=read_names,
        points=params.baseline_reads + len(scales),
    )

    # The fit's own row axis is the constrained rows plus whatever monitor BPMs
    # aren't already in them, so the constraint rows are always the leading
    # `len(constrained_names)`.
    constrained_names = _constrained_names(params)
    fit_names = constrained_names + [
        name for name in params.readbacks if name not in constrained_names
    ]
    fit_devices = [devices[name] for name in fit_names]
    constrained_devices = [devices[name] for name in constrained_names]
    # The leakage-judged BPMs: the monitor readbacks that are not already
    # constraint rows (a constrained BPM is asserted more strictly by the
    # solve's own band; the PARAMS validator guarantees this set is non-empty
    # whenever the band is set). `fit_names` is constrained + the rest of
    # `readbacks`, so this is exactly its tail.
    leak_names = fit_names[len(constrained_names) :] if params.leakage_tolerance is not None else []
    leak_devices = [devices[name] for name in leak_names]
    beam_device = (
        devices[params.beam_current_readback] if params.beam_current_readback is not None else None
    )

    def _guard(context: str) -> Any:
        """The beam-current gate, run before every write batch."""
        if beam_device is None:
            return
        assert params.beam_current_readback is not None and params.min_beam_current is not None
        values = yield from _snapshot_values([beam_device], [params.beam_current_readback], context)
        current = values[params.beam_current_readback]
        if current < params.min_beam_current:
            raise RuntimeError(
                f"orbit_bump_sweep plan: beam current {current} is below "
                f"min_beam_current {params.min_beam_current} before {context}; "
                "stopping and restoring the correctors"
            )

    def _apply(working_points: list[float], solution: list[float] | None) -> Any:
        """Move every corrector at once: working point plus its offset.

        ``solution=None`` is the terminal step's verbatim restore-shaped
        write — the recorded working points themselves, not ``wp + 0.0``,
        so the commanded values are bit-identical to what was read.
        """
        pairs: list[Any] = []
        for (_, corrector), working_point, offset in zip(
            correctors, working_points, solution or [0.0] * len(correctors), strict=True
        ):
            pairs.extend((corrector, working_point if solution is None else working_point + offset))
        yield from bps.mv(*pairs)

    def _measure_offsets(
        devices_to_read: list[Any], names: list[str], reference: dict[str, float], context: str
    ) -> Any:
        """*names*' orbit relative to *reference*: the mean of two reads.

        Two reads because both band checks compare a mean against their band
        with no noise term of their own — see ``band_converged`` on why σ/√2
        is part of that band's honesty. The convergence check runs this over
        the constrained BPMs, the leakage check over the monitor BPMs.
        """
        first = yield from _snapshot_values(devices_to_read, names, context)
        second = yield from _snapshot_values(devices_to_read, names, context)
        return [(first[name] + second[name]) / 2.0 - reference[name] for name in names]

    @bpp.stage_decorator(all_devices)
    @bpp.run_decorator(md=run_md)
    def _bump():
        working_points: list[float] = []
        for name, corrector in correctors:
            working_point = float((yield from bps.rd(corrector)))
            if not math.isfinite(working_point):
                raise ValueError(
                    f"orbit_bump_sweep plan: corrector {name!r} read back a non-finite "
                    f"working point ({working_point}); refusing to bump relative to it"
                )
            working_points.append(working_point)

        error_in_flight = False
        try:
            # --- Baseline: the reference orbit and its noise, before any write.
            baseline_rows: list[dict[str, float]] = []
            for _ in range(params.baseline_reads):
                readings = yield from bps.trigger_and_read(all_devices)
                baseline_rows.append(_finite_values(readings, fit_names, "the baseline"))

            # The reference covers the leakage-judged BPMs too (`fit_names` is
            # the constrained rows plus the remaining monitors), so both band
            # checks measure against the same baseline.
            reference = {
                name: statistics.fmean(row[name] for row in baseline_rows) for name in fit_names
            }
            sigma = [
                statistics.stdev([row[name] for row in baseline_rows]) for name in constrained_names
            ]
            violations = noise_floor_violations(sigma, params.tolerance)
            if violations:
                offenders = [constrained_names[index] for index in violations]
                raise ValueError(
                    f"orbit_bump_sweep plan: tolerance {params.tolerance} is narrower than "
                    f"twice the measured baseline noise at {offenders} — a band the orbit "
                    "falls outside of by chance alone cannot be converged into; widen the "
                    "tolerance or fix the BPM(s)"
                )
            if params.leakage_tolerance is not None:
                # The leakage band is held to the same standard as the
                # convergence band, for the same reason: a monitor whose noise
                # exceeds it would read as leaking by chance alone.
                leak_sigma = [
                    statistics.stdev([row[name] for row in baseline_rows]) for name in leak_names
                ]
                leak_floor = noise_floor_violations(leak_sigma, params.leakage_tolerance)
                if leak_floor:
                    offenders = [leak_names[index] for index in leak_floor]
                    raise ValueError(
                        f"orbit_bump_sweep plan: leakage_tolerance {params.leakage_tolerance} "
                        f"is narrower than twice the measured baseline noise at {offenders} — "
                        "a leakage band the orbit falls outside of by chance alone cannot be "
                        "certified; widen it or fix the BPM(s)"
                    )

            # The run-level demand, as reference-relative offsets over the
            # constraint rows. `absolute` targets are converted exactly once,
            # here — every later use is `scale * desired`.
            desired = [
                target.value - reference[target.readback]
                if params.mode == "absolute"
                else target.value
                for target in params.targets
            ] + [0.0] * len(params.closure_readbacks)

            # --- Probe: measure each corrector's own response, one at a time.
            probe_rows: list[dict[str, float]] = []
            for (name, corrector), working_point in zip(correctors, working_points, strict=True):
                yield from _guard(f"corrector {name!r}'s probe")
                # Both sides of the dither, +probe first: `fit_probe_response`
                # reads the pair positionally, as row `2j` and row `2j + 1`.
                for sign, label in ((1.0, "+probe"), (-1.0, "-probe")):
                    yield from bps.mv(corrector, working_point + sign * params.probe_amplitude)
                    yield from bps.sleep(params.settle_s)
                    row = yield from _snapshot_values(
                        fit_devices, fit_names, f"corrector {name!r}'s {label}"
                    )
                    probe_rows.append(row)
                yield from bps.mv(corrector, working_point)

            response = fit_probe_response(
                probe_rows, params.correctors, fit_names, params.probe_amplitude
            )
            constraint_response = response[: len(constrained_names), :]

            # Decided once on the run-level (unscaled) demand, never per step
            # — a scale-0 step would trivially pass it and short out the
            # verification the profile exists to do.
            trivial = demand_is_negligible(desired, constraint_response, sigma, params.tolerance)

            # --- Profile walk: one data row per amplitude step, over the
            # `scales` the run metadata already counted.
            for index, scale in enumerate(scales):
                terminal = index == len(scales) - 1
                step_label = f"amplitude step {index + 1}/{len(scales)} (scale {scale:+g})"
                desired_step = [scale * value for value in desired]
                if trivial or terminal:
                    solution = [0.0] * len(correctors)
                else:
                    solution = [
                        float(offset) for offset in solve_offsets(constraint_response, desired_step)
                    ]

                yield from _guard(step_label)
                yield from _apply(working_points, None if terminal else solution)
                yield from bps.sleep(params.settle_s)
                measured = yield from _measure_offsets(
                    constrained_devices, constrained_names, reference, "a convergence check"
                )
                converged = band_converged(measured, desired_step, params.tolerance)

                # Trim only where a solve is allowed: never on the terminal
                # step (out-of-band there is the finding, not a miss to
                # correct) and never on the trivially-converged path.
                if not converged and not terminal and not trivial:
                    for _ in range(params.max_trim_iterations):
                        yield from _guard(f"a trim pass at {step_label}")
                        residual = [
                            want - got for want, got in zip(desired_step, measured, strict=True)
                        ]
                        delta = solve_offsets(constraint_response, residual)
                        solution = [
                            offset + float(trim)
                            for offset, trim in zip(solution, delta, strict=True)
                        ]
                        yield from _apply(working_points, solution)
                        yield from bps.sleep(params.settle_s)
                        measured = yield from _measure_offsets(
                            constrained_devices, constrained_names, reference, "a convergence check"
                        )
                        converged = band_converged(measured, desired_step, params.tolerance)
                        if converged:
                            break

                if not converged:
                    if params.best_effort:
                        logger.warning(
                            "orbit_bump_sweep plan: %s did not converge within tolerance %r "
                            "(measured offsets %r against desired %r); best_effort is set, "
                            "recording the achieved orbit and continuing",
                            step_label,
                            params.tolerance,
                            measured,
                            desired_step,
                        )
                    else:
                        raise RuntimeError(
                            f"orbit_bump_sweep plan: {step_label} did not converge within "
                            f"tolerance {params.tolerance} (measured offsets {measured} "
                            f"against desired {desired_step})"
                            + (
                                " at the terminal working-point step — the machine did not "
                                "come back to its reference orbit"
                                if terminal
                                else f" after {params.max_trim_iterations} trim pass(es)"
                            )
                        )

                # The leakage check: after the constrained rows have settled
                # (post-trim), the monitor BPMs must still be on the reference
                # orbit. Measured the same way the convergence band is (the
                # mean of two reads), judged after the trim loop because a
                # trim moves the correctors and so moves the leakage too.
                if params.leakage_tolerance is not None:
                    leak_measured = yield from _measure_offsets(
                        leak_devices, leak_names, reference, "a leakage check"
                    )
                    leaks = band_violations(
                        leak_measured, [0.0] * len(leak_names), params.leakage_tolerance
                    )
                    if leaks:
                        offenders = ", ".join(
                            f"{leak_names[index]} ({leak_measured[index]:+.4g})" for index in leaks
                        )
                        if params.best_effort:
                            logger.warning(
                                "orbit_bump_sweep plan: %s leaked outside ±%g BPM units at "
                                "monitor BPM(s) %s relative to the reference orbit; "
                                "best_effort is set, recording the step and continuing",
                                step_label,
                                params.leakage_tolerance,
                                offenders,
                            )
                        else:
                            raise RuntimeError(
                                f"orbit_bump_sweep plan: {step_label} leaked outside "
                                f"±{params.leakage_tolerance:g} BPM units at monitor BPM(s) "
                                f"{offenders} relative to the reference orbit"
                                + (
                                    " at the terminal working-point step — the machine did "
                                    "not come back to its reference orbit"
                                    if terminal
                                    else " — the monitors are watched, not solved for, so a "
                                    "trim cannot remove this; add them as closure BPMs to "
                                    "constrain them"
                                )
                            )

                yield from bps.trigger_and_read(all_devices)
        except BaseException:
            error_in_flight = True
            raise
        finally:
            restore_failures: list[str] = []
            for (name, corrector), working_point in zip(correctors, working_points, strict=True):
                try:
                    yield from bps.mv(corrector, working_point)
                except Exception:
                    logger.warning(
                        "orbit_bump_sweep plan: failed to restore corrector %s to its "
                        "pre-scan working point (%r) during cleanup",
                        name,
                        working_point,
                        exc_info=True,
                    )
                    restore_failures.append(name)
            # In-flight error: the log above is all a failure here may become,
            # so the exception that explains the run is the one that surfaces.
            # Clean path: a run that ends with correctors off their working
            # points must never read as a success.
            if restore_failures and not error_in_flight:
                raise RuntimeError(
                    f"orbit_bump_sweep plan: the sweep completed but corrector(s) "
                    f"{restore_failures} could not be restored to their pre-scan "
                    "working points"
                )

    return _bump()


# --- Live view ---------------------------------------------------------------
#
# `render()` runs in the bridge API process on every poll tick of every client
# watching an `orbit_bump_sweep` run, over the run's FULL stored row set.
# Everything below is sized for that: plain-Python arithmetic over the rows
# once, and hard caps on how many series reach the wire.
#
# Rows carry no scale or step number of their own, so every step here is
# attributed by position — `baseline_reads` baseline rows, then one row per
# scale of `_profile_scales`, in profile order — exactly as `orm`'s render
# attributes its rows to correctors. A short row set is a run that has not got
# there yet (or was aborted): the leading rows still mean what they mean, so
# the figure draws them and says how much of the profile is missing.

_MAX_STEP_SERIES = 12
"""Most amplitude steps drawn as orbit traces. The drawn steps are chosen from
the profile *length* — which the parameters fix before the run starts — and not
from the rows in hand, so a series never swaps identity between poll ticks."""

_MAX_BAND_SERIES = 12
"""Most constrained BPMs traced in the residual panel. The first N in the
requested order (targets, then closure), not the N worst: a "worst" selection
would swap lines in and out as the sweep fills in."""

_MAX_MONITOR_SERIES = 12
"""Most monitor traces in the response panel, counted across every leg — each
monitor is drawn once per leg, so this is what the per-leg monitor count is
divided out of."""

_MAX_MISS_NOTES = 6
"""Most unconverged steps called out one by one in an annotation. Past this the
annotation says how many more there were rather than listing them."""


def _finite(value: Any) -> float | None:
    """*value* as a finite float, or ``None`` when it is nothing a plot can use.

    ``None`` here means "no reading at this row": a key the event never carried,
    a string or boolean, a ``NaN`` that survived the row adapter, or an ``inf``
    — which is a number the run produced but not one an axis has a place for.
    """
    if value is None or isinstance(value, bool | str | bytes):
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return as_float if math.isfinite(as_float) else None


def _column_mean(rows: list[dict[str, Any]], name: str) -> float | None:
    """The mean of *name*'s finite readings over *rows*, or ``None`` if it has none.

    Over the baseline rows this is the two references every panel is drawn
    against: a BPM's reference orbit, and a corrector's pre-scan working point.
    """
    values = [value for row in rows if (value := _finite(_row_value(row, name))) is not None]
    return statistics.fmean(values) if values else None


class _Sweep(NamedTuple):
    """The run's rows, resolved into what the panels actually plot."""

    baseline: list[dict[str, Any]]
    steps: list[dict[str, Any]]
    """The amplitude-step rows, positionally aligned with the leading
    ``len(steps)`` entries of `scales`; shorter than `scales` for a run that has
    not finished the profile."""

    scales: list[float]
    """The whole profile the parameters call for, finished or not."""

    reference: dict[str, float | None]
    """Per-BPM reference orbit: the baseline mean, or ``None`` when the baseline
    rows never carried that BPM."""

    working: dict[str, float | None]
    """Per-corrector pre-scan working point, on the same terms."""

    desired: dict[str, float | None]
    """Per constrained BPM, the *unscaled* orbit change asked for, relative to
    the reference — step ``i`` asks for ``scales[i]`` times this. ``absolute``
    targets are converted here, once, exactly as `build_plan` converts them."""

    bpm_order: list[str]
    """Every requested BPM, named once, in request order: targets, then closure,
    then monitors."""

    extra_rows: int
    """Rows past the end of the profile — never expected, annotated if seen."""


def _prepare(rows: list[dict[str, Any]], params: PARAMS) -> _Sweep | None:
    """Resolve *rows* into a `_Sweep`, or ``None`` when there is nothing to draw.

    ``None`` means the run has not yet emitted a single amplitude step — it is
    still taking baseline reads, or it produced no rows at all. Every panel here
    is a statement about a step, so there is no honest partial view before then.
    """
    scales = _profile_scales(params.num, params.sweep)
    baseline = list(rows[: params.baseline_reads])
    if len(baseline) < params.baseline_reads:
        return None

    tail = list(rows[params.baseline_reads :])
    steps = tail[: len(scales)]
    if not steps:
        return None

    bpm_order: list[str] = []
    for name in [target.readback for target in params.targets] + [
        *params.closure_readbacks,
        *params.readbacks,
    ]:
        if name not in bpm_order:
            bpm_order.append(name)

    reference = {name: _column_mean(baseline, name) for name in bpm_order}

    desired: dict[str, float | None] = {}
    for target in params.targets:
        if params.mode == "absolute":
            against = reference.get(target.readback)
            desired[target.readback] = None if against is None else target.value - against
        else:
            desired[target.readback] = target.value
    for name in params.closure_readbacks:
        desired[name] = 0.0

    return _Sweep(
        baseline=baseline,
        steps=steps,
        scales=scales,
        reference=reference,
        working={name: _column_mean(baseline, name) for name in params.correctors},
        desired=desired,
        bpm_order=bpm_order,
        extra_rows=len(tail) - len(steps),
    )


def _legs(params: PARAMS) -> list[tuple[str, int, int]]:
    """Each leg of the profile as ``(name, first_index, stop_index)``.

    The legs are what make a hysteresis loop legible: the beam does not follow
    the same orbit going up as coming back down, and a figure that drew one line
    through both would average that difference away.
    """
    num = params.num
    legs = [("ascent", 0, num), ("descent", num, 2 * num)]
    if params.sweep != "monodirectional":
        legs.extend(
            [
                ("ascent, negative side", 2 * num, 3 * num),
                ("descent, negative side", 3 * num, 4 * num),
            ]
        )
    return legs


def _leg_of(index: int, legs: list[tuple[str, int, int]]) -> str:
    """The leg step *index* belongs to; ``"sweep"`` if it falls outside them all."""
    for name, start, stop in legs:
        if start <= index < stop:
            return name
    return "sweep"


def _selected_steps(total: int, cap: int) -> list[int]:
    """Which of *total* steps to draw: evenly spaced, both ends kept.

    Chosen from the profile length alone — a number the parameters fix before
    the run starts — so the drawn set is the same on every poll tick and a live
    figure fills in rather than rearranging. Spanning the whole profile is what
    keeps both legs represented; taking the first N would draw a bidirectional
    run's ascent and nothing else.
    """
    if total <= cap:
        return list(range(total))
    return sorted({(i * (total - 1)) // (cap - 1) for i in range(cap)})


def _index_list(positions: list[int], limit: int = 8) -> str:
    """Positions as a readable list, trailing off past *limit* rather than running on."""
    shown = ", ".join(str(position) for position in positions[:limit])
    hidden = len(positions) - limit
    return f"{shown} and {hidden} more" if hidden > 0 else shown


def _lines(series: list[Series]) -> LinesMark:
    """A `LinesMark` carrying *series* with its decimation truth summed up."""
    return LinesMark(
        series=series,
        decimated=any(item.decimated for item in series),
        source_points=sum(item.source_points for item in series),
    )


def _drawn(series: list[Series]) -> bool:
    """Whether any series has a point to draw. An all-gaps panel says less than none."""
    return any(point.y is not None for item in series for point in item.points)


def _orbit_shift_panel(sweep: _Sweep, params: PARAMS) -> Panel | None:
    """The bump itself: each drawn step's orbit shift across the requested BPMs.

    This is the panel that shows whether the bump is a *bump* — a displacement
    over the target BPMs that comes back to the reference orbit at the closure
    BPMs and stays there. Reading it needs the roles, so the annotations say
    which x positions are targets and which are closure.

    The BPM axis is the requested order, which is not `s` order: this plan is
    device-agnostic and carries no lattice positions, so the panel says so
    rather than implying a geometry it cannot know.
    """
    legs = _legs(params)
    selected = [
        index
        for index in _selected_steps(len(sweep.scales), _MAX_STEP_SERIES)
        if index < len(sweep.steps)
    ]
    if not selected or not sweep.bpm_order:
        return None

    series: list[Series] = []
    for index in selected:
        row = sweep.steps[index]
        points: list[Point] = []
        for position, name in enumerate(sweep.bpm_order):
            against = sweep.reference.get(name)
            value = _finite(_row_value(row, name))
            points.append(
                Point(
                    x=float(position + 1),
                    y=None if against is None or value is None else value - against,
                )
            )
        thinned = decimate(points)
        series.append(
            Series(
                label=(
                    f"step {index + 1}, scale {sweep.scales[index]:+g} ({_leg_of(index, legs)})"
                ),
                **thinned._asdict(),
            )
        )

    if not _drawn(series):
        return None

    targets = [sweep.bpm_order.index(target.readback) + 1 for target in params.targets]
    closure = [sweep.bpm_order.index(name) + 1 for name in params.closure_readbacks]
    beyond = (
        "anything beyond them is recorded, not constrained."
        if params.leakage_tolerance is None
        else (
            f"the remaining monitor BPMs are held inside ±{params.leakage_tolerance:g} "
            "BPM units of the reference — the leakage band — without joining the solve."
        )
    )
    annotations = [
        "Each line is one amplitude step's orbit, measured against the "
        "baseline reference orbit. A closed bump is displaced across the "
        f"target BPMs (index {_index_list(targets)}) and back on the reference "
        f"orbit at the closure BPMs (index {_index_list(closure)}); {beyond}",
        "BPMs are in the order they were requested, which is not their order "
        "around the ring — this plan carries no lattice positions.",
    ]
    if len(selected) < len(sweep.scales):
        annotations.append(
            f"Showing {len(selected)} of {len(sweep.scales)} amplitude steps, "
            "spaced evenly across the profile so every leg is represented."
        )

    return Panel(
        title="Orbit shift across BPMs",
        x_label="BPM index",
        y_label="Orbit shift",
        y_units="BPM units",
        annotations=annotations,
        series_picker=True,
        mark=_lines(series),
    )


def _step_offsets(sweep: _Sweep, name: str) -> list[float | None]:
    """*name*'s orbit offset from the reference, one entry per recorded step."""
    against = sweep.reference.get(name)
    if against is None:
        return [None] * len(sweep.steps)
    values = [_finite(_row_value(row, name)) for row in sweep.steps]
    return [None if value is None else value - against for value in values]


def _misses(sweep: _Sweep, params: PARAMS) -> list[tuple[int, str, float, float]]:
    """Steps that landed outside the band, as ``(index, bpm, achieved, requested)``.

    One entry per offending step, naming the constrained BPM that missed by the
    most. Judged over *every* constrained BPM, including any the residual panel
    caps away, so a capped panel never hides a miss.
    """
    constrained = _constrained_names(params)
    offsets = {name: _step_offsets(sweep, name) for name in constrained}

    misses: list[tuple[int, str, float, float]] = []
    for index, scale in enumerate(sweep.scales[: len(sweep.steps)]):
        worst: tuple[float, str, float, float] | None = None
        for name in constrained:
            want = sweep.desired.get(name)
            got = offsets[name][index]
            if want is None or got is None:
                continue
            requested = scale * want
            residual = abs(got - requested)
            if residual > params.tolerance and (worst is None or residual > worst[0]):
                worst = (residual, name, got, requested)
        if worst is not None:
            misses.append((index, worst[1], worst[2], worst[3]))
    return misses


def _band_residual_panel(sweep: _Sweep, params: PARAMS) -> Panel | None:
    """How far each constrained BPM sat from what its step asked for.

    The residual, not the orbit: a target asked for ``scale * value`` and a
    closure BPM for nothing, so subtracting the demand puts both on one axis
    where the only thing to look for is whether a line leaves the band.
    """
    constrained = _constrained_names(params)
    shown = constrained[:_MAX_BAND_SERIES]
    if not shown:
        return None

    scales = sweep.scales[: len(sweep.steps)]
    series: list[Series] = []
    for name in shown:
        want = sweep.desired.get(name)
        offsets = _step_offsets(sweep, name)
        points = [
            Point(
                x=float(index + 1),
                y=None if want is None or got is None else got - scale * want,
            )
            for index, (scale, got) in enumerate(zip(scales, offsets, strict=True))
        ]
        thinned = decimate(points)
        series.append(Series(label=name, **thinned._asdict()))

    if not _drawn(series):
        return None

    # The band drawn as two flat lines: the panel's whole question is whether a
    # residual left it, and an annotation alone leaves the reader estimating it
    # off the axis.
    for sign, label in ((1.0, "+tolerance"), (-1.0, "-tolerance")):
        edge = decimate(
            [Point(x=float(index + 1), y=sign * params.tolerance) for index in range(len(scales))]
        )
        series.append(Series(label=label, **edge._asdict()))

    legs = [
        f"steps {start + 1}-{min(stop, len(scales))} {name}"
        for name, start, stop in _legs(params)
        if start < len(scales)
    ]
    annotations = [
        f"The band is the requested tolerance, ±{params.tolerance:g} BPM units, "
        "drawn as the two flat lines. A step is converged when every "
        "constrained BPM sits inside it.",
        f"Sweep legs: {'; '.join(legs)}." if legs else "",
    ]
    if len(shown) < len(constrained):
        annotations.append(
            f"Showing the first {len(shown)} of {len(constrained)} constrained "
            "BPMs; every one of them is still checked for the notes below."
        )
    if len(sweep.steps) >= len(sweep.scales):
        annotations.append(
            "The last step is the terminal working-point verification: the "
            "correctors are commanded back to the values they were read at, so "
            "a residual there is the machine not coming back, never a miss to "
            "trim away."
        )

    misses = _misses(sweep, params)
    if misses:
        annotations.append(
            "Band membership here is judged from each step's recorded row, "
            "while the plan judged it on the mean of two separate reads, so a "
            "step may sit marginally outside the band drawn here and still have "
            "been recorded as converged."
        )
        for index, name, achieved, requested in misses[:_MAX_MISS_NOTES]:
            annotations.append(
                f"Step {index + 1} (scale {sweep.scales[index]:+g}) is outside the "
                f"band at {name}: reached {achieved:+.4g} against {requested:+.4g} "
                "BPM units requested."
            )
        hidden = len(misses) - _MAX_MISS_NOTES
        if hidden > 0:
            annotations.append(f"{hidden} further step(s) are outside the band.")

    return Panel(
        title="Band residual at constrained BPMs",
        x_label="Amplitude step",
        y_label="Residual (reached minus requested)",
        y_units="BPM units",
        annotations=[note for note in annotations if note],
        series_picker=True,
        mark=_lines(series),
    )


def _corrector_offset_panel(sweep: _Sweep, params: PARAMS) -> Panel | None:
    """Where each corrector sat, relative to the working point it started on.

    Not the raw current: a ring running a corrected orbit idles every corrector
    somewhere nonzero, and the bump is the change on top of that. Zero here
    means "back where the run found it", which is what the terminal step is
    supposed to read.
    """
    # No cap: PARAMS admits three or four correctors and nothing else.
    series: list[Series] = []
    for name in params.correctors:
        start = sweep.working.get(name)
        values = [_finite(_row_value(row, name)) for row in sweep.steps]
        points = [
            Point(
                x=float(index + 1),
                y=None if start is None or value is None else value - start,
            )
            for index, value in enumerate(values)
        ]
        thinned = decimate(points)
        series.append(Series(label=name, **thinned._asdict()))

    if not _drawn(series):
        return None

    return Panel(
        title="Corrector offsets",
        x_label="Amplitude step",
        y_label="Offset from working point",
        y_units="corrector units",
        annotations=[
            "Each corrector's readback measured against its own pre-scan "
            "working point — the mean of the baseline rows, before anything "
            "moved.",
            "The last step commands those working points back verbatim, so "
            "every line should return to zero there; one that does not is a "
            "corrector the run did not restore.",
        ],
        mark=_lines(series),
    )


def _monitor_panel(sweep: _Sweep, params: PARAMS) -> Panel | None:
    """Monitor readings against bump amplitude, one line per leg.

    Amplitude rather than step number, because what a monitor has to say about
    a bump is how it responds to the excursion's size. Each leg is its own line:
    plotted against amplitude the profile doubles back on itself, and a single
    line through both legs would close a loop that is real — the hysteresis —
    into a scribble.
    """
    if not params.monitors:
        return None

    legs = _legs(params)
    per_leg = max(1, _MAX_MONITOR_SERIES // max(1, len(legs)))
    shown = params.monitors[:per_leg]

    series: list[Series] = []
    for name in shown:
        for leg, start, stop in legs:
            indices = list(range(start, min(stop, len(sweep.steps))))
            if not indices:
                continue
            points = [
                Point(
                    x=sweep.scales[index],
                    y=_finite(_row_value(sweep.steps[index], name)),
                )
                for index in indices
            ]
            thinned = decimate(points)
            series.append(Series(label=f"{name} ({leg})", **thinned._asdict()))

    if not _drawn(series):
        return None

    annotations = [
        "Amplitude is the fraction of the requested bump the step asked for; "
        "each leg is drawn separately, so an ascent and a descent that do not "
        "lie on top of each other are a hysteresis the sweep measured.",
    ]
    if len(shown) < len(params.monitors):
        annotations.append(f"Showing the first {len(shown)} of {len(params.monitors)} monitors.")

    return Panel(
        title="Monitor response",
        x_label="Bump amplitude (fraction of requested)",
        y_label="Monitor reading",
        annotations=annotations,
        series_picker=True,
        mark=_lines(series),
    )


def _panel_or_none(build: Callable[[], Panel | None], what: str) -> Panel | None:
    """Build one panel, or drop it. One panel's failure is not the figure's."""
    try:
        return build()
    except Exception:
        logger.warning(
            "orbit_bump_sweep render: the %s panel could not be built; omitting it",
            what,
            exc_info=True,
        )
        return None


def _build(window: RowWindow, params: PARAMS) -> Figure:
    """Assemble the figure. Individual panels degrade; `render` catches the rest."""
    sweep = _prepare(window.rows, params)
    if sweep is None:
        return Figure(panels=[], partial=True, source="live")

    panels = [
        panel
        for panel in (
            _panel_or_none(lambda: _orbit_shift_panel(sweep, params), "orbit shift"),
            _panel_or_none(lambda: _band_residual_panel(sweep, params), "band residual"),
            _panel_or_none(lambda: _corrector_offset_panel(sweep, params), "corrector offset"),
            _panel_or_none(lambda: _monitor_panel(sweep, params), "monitor response"),
        )
        if panel is not None
    ]

    lead: list[str] = []
    if not window.rows_complete:
        # A capped window is always a PREFIX of the run (no figure source ever
        # hands a plan a middle or a tail), so the positional step attribution
        # below stays sound — the drawn rows keep meaning what they mean, and
        # only the tail is missing.
        lead.append(
            f"This run has produced more rows than are being drawn "
            f"({len(window.rows):,} of {window.total_seen:,}); the drawn steps keep "
            "their attribution, and the run's tail is not in this figure."
        )
    if len(sweep.steps) < len(sweep.scales):
        lead.append(
            f"{len(sweep.steps)} of {len(sweep.scales)} amplitude steps recorded so "
            "far; the rest of the profile is not in this figure."
        )
    if sweep.extra_rows:
        lead.append(
            f"This run produced {sweep.extra_rows} row(s) past the end of its "
            "amplitude profile, which are not drawn."
        )
    if lead and panels:
        panels[0] = panels[0].model_copy(update={"annotations": [*lead, *panels[0].annotations]})

    # Placeholders: the figure route stamps the truth onto both on every path.
    # A render sees rows, never their provenance, so `partial=True` is the
    # honest direction for a forgotten overwrite — extra polling, never a false
    # "settled".
    return Figure(panels=panels, partial=True, source="live")


def render(window: RowWindow, params: PARAMS) -> Figure:
    """The `orbit_bump_sweep` plan's own view of a run: the bump, and what it cost.

    Panel order is stable, so a live figure grows rather than rearranging:

    1. **Orbit shift across BPMs** — each drawn amplitude step's orbit measured
       against the baseline reference, over the requested BPMs. This is where a
       closed bump looks like one: displaced at the targets, back on the
       reference at the closure BPMs.
    2. **Band residual at constrained BPMs** — reached minus requested at every
       target and closure BPM, against step number, with the ``±tolerance``
       band drawn as two flat lines. Steps that sat outside it are called out
       by name, with what they reached against what they asked for, which is
       what a ``best_effort`` run records instead of stopping.
    3. **Corrector offsets** — each corrector's readback against its own
       pre-scan working point, so the terminal step's return to zero is
       visible.
    4. **Monitor response** — only when monitors were requested: their
       readings against bump amplitude, one line per leg of the sweep.

    Rows carry no step number, so steps are attributed by position:
    ``baseline_reads`` baseline rows, then one row per scale of the amplitude
    profile in profile order. A row set that stops short is a run still walking
    the profile (or one that aborted): the leading rows still mean what they
    mean, so they are drawn and the figure says how much of the profile is
    missing. A capped window (``window.rows_complete`` false) is the same
    story told by the source instead of the run — the rows are a prefix, the
    attribution holds, and an annotation quotes ``window.total_seen`` for how
    much went undrawn.

    **Never raises.** A figure is a view, not a result: a missing device key, a
    row set with no steps in it yet, an unreadable baseline — each degrades to
    fewer panels, and a failure below all of them to a figure with none, so the
    route always has something to serve.

    **`partial` and `source` are placeholders the route overwrites.** This
    function sees rows, not their provenance, so it returns ``partial=True,
    source="live"`` unconditionally and the figure route stamps the truth onto
    both on every path. ``True`` is the honest direction for the placeholder: a
    forgotten overwrite costs a client extra polling, never a false "settled".

    Args:
        window: The run's event `data` dicts in emission order, and whether
            they are all of them (`RowWindow`).
        params: The parameters the run was launched with.

    Returns:
        A `Figure` whose panels are as complete as the rows allow.
    """
    try:
        return _build(window, params)
    except Exception:
        logger.warning("orbit_bump_sweep render failed; serving an empty figure", exc_info=True)
        return Figure(panels=[], partial=True, source="live")


_RENDER_CONFORMANCE: Callable[[RowWindow, PARAMS], Figure] = render
"""Static proof that `render` matches `PlanSpec.render`'s signature.

The loader holds `PlanSpec[Any]`, so a module-level `render` is not type-checked
by merely existing (see `plan_loader._resolve_render`); this binding is what
makes mypy check it here, where the mistake would be made.
"""
