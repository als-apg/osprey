"""Acquisition: the action that runs the trajectory chain.

**The chain** is the measurement file's declared list of steps
(:class:`ChainDeclaration`): the service-run steps ``arm``, ``inject`` and
``readout``, and the declared command steps (cleaning), each with whether it
is ticked by default. A run (:class:`ChainRun`) names the ticked steps, the
readout mode, the turns a pseudo-orbit readout averages over, the injections
and how they are combined; the ticked steps run once per injection, in chain
order, inside one holder action:

- with ``readout`` ticked, the **pre-inject batch** reads every BPM record's
  timestamp and, when the declaration states one, the turns-tracked readback
  in one call; the readback is the source of the shot's ``raw_samples`` and
  pseudo-orbit's bound;
- ``arm`` fires every device's arm command in one command batch and waits
  until every ``armed`` flag reads 1;
- ``inject`` reads the stored current (when declared) just before it runs the
  module's ``inject``, then, when ``arm`` ran, waits until every flag reads 0;
- a command step writes its declared command and, when it declares a
  ``done`` readback, waits up to its ``timeout_s`` for that value;
- ``readout`` reads the records by the device table's row names, keeps their
  first ``raw_samples`` samples — the trigger sample plus the turns tracked,
  every sample when no readback is declared or the records hold fewer — and
  extracts the first turn (``tbt``) or the mean over ``n_turns`` turns from
  those samples alone (``pseudo_orbit``, bounded by the turns tracked).

A readout is **fresh** when ``arm`` and ``inject`` ran in the same action;
only then is each record required to be stamped later than the pre-inject
batch read it. A wait runs under one deadline across the flags,
``armed_timeout_s``, and polls the stop event; a flag that missed it is named.
An injection that fails is listed with its reason, excluded and never
retried. Only a chain with ``readout`` makes a shot: the included injections
are averaged, the result is written as the shot file beside the raw buffer
of every included injection (:mod:`.raw`; none when no injection was
included), and a fresh TbT shot
with at least one included injection is handed to the post-shot hook
(:func:`makes_step`). A chain without readout stores nothing; one of its steps
failing ends the action with :class:`ChainFailed`, naming what it fired.

**Exclusions** reach the turn: the BPMs the SolveConfig excludes, read when
the action is built, are neither armed, waited on nor read, so a dead excluded
BPM never fails a shot. An arm command or ``armed`` flag several devices share
stays while any included device uses it. The shot keeps every BPM's row, an
excluded one reading NaN, and records its ``excluded_bpms`` and the
``loss_cutoff`` it was extracted with; its raw buffer holds the included BPMs
only.

**A derived shot** (:func:`derive_shot`) re-extracts every included injection
of a stored shot's raw buffer at another loss cutoff, averaged as the parent
was. It names the parent in ``parent_id``, inherits its freshness, readout,
setup and setpoints, fires nothing, and replaces an earlier derived shot of
the same parent; it keeps no buffer of its own, and the raw route reads its
parent's. It is refused when the buffer was pruned (410 ``raw_evicted``) or
when the exclusions changed since the parent was read (409
``exclusions_changed``).

A run whose worst case does not fit the action timeout,
``n_shots × per-shot sum`` (:func:`chain_budget_s`) plus the setpoint
snapshot's batch reads as the last terms (:func:`snapshot_budget_s`), is
refused before the turn (400 ``over_budget``). A stop ends the action at its
next checkpoint with :class:`ShotStopped`, which names the commands fired; no
shot file is written and no hook runs. A chain moves no setting: its writes are
commands.

**The setpoint snapshot**: a chain that reads out, on a service whose tool
names RM columns, reads the full RM-column corrector setpoint vector
(:func:`setpoint_names`) in one batch after the last readout and stores it on
the shot as ``setpoints``. With ``n_shots > 1`` it reads the vector once more
before the first injection; a setpoint that differs between the two reads by
more than :data:`SNAPSHOT_TOLERANCE` × its ``max_step`` (:func:`moved_setpoints`)
makes the shot not fresh, so it makes no step. A snapshot read that fails or
reads a setpoint that is no finite number leaves ``setpoints`` ``None`` and the
shot not fresh. The snapshot is what an apply re-reads to tell whether the
machine moved since the shot.

**The setup re-check** (:class:`SetupCheck`): when the chain is given one,
every injection that injects or reads out first re-checks the current setup
against its readbacks — folded into the pre-inject batch read when the chain
reads out, one batch read of its own otherwise. The shot carries the setup's
``setup_sha`` only when every injection's re-check held it; a re-check that
found the machine diverged voids the setup (the check's own business) and the
shot carries ``None``. The re-check never fails a shot and never refuses a
door.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import time
import uuid
import warnings
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

import numpy as np

from osprey.services.trajectory.actions import Action
from osprey.services.trajectory.admit import Admission
from osprey.services.trajectory.calls import guarded_call
from osprey.services.trajectory.constants import CHECKPOINT_POLL_S
from osprey.services.trajectory.extraction import (
    ExtractionFailure,
    ShotExtraction,
    average_shots,
    extract_shot,
)
from osprey.services.trajectory.raw import RAW_EVICTED, RawBuffer, RawEvicted, RawUnits, units_of
from osprey.services.trajectory.reach import TRAJECTORY_UNAVAILABLE
from osprey.services.trajectory.refusal import Refused
from osprey.services.trajectory.sequence import SequenceHandle, SequenceStopped
from osprey.services.trajectory.startup import TrajectorySettings, TrajectoryStartup
from osprey.services.trajectory.state import (
    ChainBody,
    ChainResult,
    ChainStep,
    ShotFailure,
    ShotResult,
)
from osprey.services.trajectory.step import TrajectoryTool, max_steps
from osprey.services.trajectory.store import TrajectoryStore
from osprey_connectors.control_system.call_timeout import DEFAULT_TIMEOUT_S

__all__ = [
    "ARM",
    "ARMED_NEVER_CLEARED",
    "ARMED_NEVER_SET",
    "CHAIN_FAILED",
    "CONTINUOUS_RUNNING",
    "DERIVED_SUFFIX",
    "DEFAULT_DECLARATION",
    "EXCLUSIONS_CHANGED",
    "INJECT",
    "INVALID_CHAIN",
    "NO_SEQUENCE_MODULE",
    "OVER_BUDGET",
    "READOUT",
    "SETUP_REQUIRED",
    "SHOT",
    "SNAPSHOT_TOLERANCE",
    "TURNS_TRACKED",
    "UNKNOWN_STEP",
    "AcquisitionRefused",
    "ChainDeclaration",
    "ChainFailed",
    "ChainRun",
    "PostShotHook",
    "ShotRows",
    "SetupCheck",
    "ShotStopped",
    "chain_action",
    "chain_budget_s",
    "chain_declaration",
    "chain_run",
    "default_run",
    "derive_shot",
    "makes_step",
    "moved_setpoints",
    "n_shots_max",
    "setpoint_names",
    "setpoints_of",
    "shot_action",
    "snapshot_budget_s",
]

NO_SEQUENCE_MODULE = "no_sequence_module"
OVER_BUDGET = "over_budget"
CONTINUOUS_RUNNING = "continuous_running"
UNKNOWN_STEP = "unknown_step"
INVALID_CHAIN = "invalid_chain"
SETUP_REQUIRED = "setup_required"
CHAIN_FAILED = "chain_failed"
EXCLUSIONS_CHANGED = "exclusions_changed"

#: Appended to a parent's id to name the shot derived from it; one per parent.
DERIVED_SUFFIX = "-cut"

#: The name a chain action carries in the service state.
SHOT = "shot"

#: The service-run steps: arming the BPMs, the module's injection, the readout.
ARM = "arm"
INJECT = "inject"
READOUT = "readout"
SERVICE_STEPS = (ARM, INJECT, READOUT)

#: The declared setting whose readback is the turns every BPM records.
TURNS_TRACKED = "turns_tracked"

#: A shot's reason when a device's ``armed`` flag did not read 1 after the arm.
ARMED_NEVER_SET = "armed never set"
#: A shot's reason when a device's ``armed`` flag did not read 0 after the injection.
ARMED_NEVER_CLEARED = "armed never cleared"

#: The fraction of a corrector's ``max_step`` its setpoint may drift and still
#: match a snapshot; a corrector stating no ``max_step`` must match exactly.
SNAPSHOT_TOLERANCE = 1e-6


class AcquisitionRefused(Refused):
    """A chain the service cannot run as asked; nothing was written."""

    statuses = {
        NO_SEQUENCE_MODULE: 503,
        TRAJECTORY_UNAVAILABLE: 503,
        OVER_BUDGET: 400,
        UNKNOWN_STEP: 400,
        INVALID_CHAIN: 400,
        CONTINUOUS_RUNNING: 409,
        SETUP_REQUIRED: 409,
        EXCLUSIONS_CHANGED: 409,
        RAW_EVICTED: 410,
    }


class SetupCheck(Protocol):
    """The current setup's re-check, as a chain turn calls it."""

    def addresses(self) -> list[str]:
        """The readbacks the re-check reads; none while no current setup holds."""
        ...

    async def check(self, values: Mapping[str, Any]) -> str | None:
        """The setup's ``settings_sha`` when *values* still hold it, else ``None``."""
        ...


# -- the declaration ---------------------------------------------------------------


def _service_step(step_id: str, label: str, description: str, position: int) -> ChainStep:
    return ChainStep(
        id=step_id,
        label=label,
        description=description,
        command=None,
        default=True,
        position=position,
        timeout_s=None,
    )


@dataclass(frozen=True)
class ChainDeclaration:
    """A model's declared chain and the readbacks the chain and the header read.

    Attributes:
        steps: The chain's steps, in chain order.
        turns_tracked: The turns-tracked readback, ``None`` when undeclared.
        rf_on: The RF-on readback, ``None`` when undeclared.
        stored_current: The stored-current readback, ``None`` when undeclared.
        units: Each plane's unit, as the device table's channel facts state
            it; the raw buffer's header carries them.
    """

    steps: tuple[ChainStep, ...]
    turns_tracked: str | None = None
    rf_on: str | None = None
    stored_current: str | None = None
    units: RawUnits = RawUnits()

    @property
    def ids(self) -> tuple[str, ...]:
        """Every step id, in chain order."""
        return tuple(step.id for step in self.steps)

    @property
    def defaults(self) -> tuple[str, ...]:
        """The steps ticked by default, in chain order: the agent door's chain."""
        return tuple(step.id for step in self.steps if step.default)

    @property
    def commands(self) -> frozenset[str]:
        """Every command step's address: a command, never a setting."""
        return frozenset(step.command.address for step in self.steps if step.command is not None)

    @property
    def machine(self) -> dict[str, str]:
        """The header's readbacks that are declared, by reading name."""
        named = {"rf_on": self.rf_on, "stored_current": self.stored_current}
        return {name: address for name, address in named.items() if address}


#: The chain of a model whose measurement file declares no steps: arm, inject, readout.
DEFAULT_DECLARATION = ChainDeclaration(
    steps=(
        _service_step(ARM, "Arm", "Arms every BPM's turn-by-turn acquisition.", 0),
        _service_step(INJECT, "Inject", "Fires one injected shot.", 1),
        _service_step(READOUT, "Readout", "Reads every BPM's turn-by-turn buffers.", 2),
    )
)


def _address(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def chain_declaration(slots: Mapping[str, Any] | None) -> ChainDeclaration:
    """The chain a model's ``trajectory_slots`` entry declares.

    A model whose entry states no steps runs :data:`DEFAULT_DECLARATION`. A
    stated chain carries ``arm``, ``inject`` and ``readout`` as steps the
    service runs (no command) and every other step as a command step with
    its ``timeout_s``.

    Args:
        slots: The model's ``trajectory_slots`` entry, ``None`` without one.

    Returns:
        The declaration.

    Raises:
        ValueError: A step is malformed, an id repeats, a service step carries
            a command or a command step none, or a service step is missing.
    """
    slots = slots if isinstance(slots, Mapping) else {}
    turns = next(
        (
            setting
            for setting in slots.get("settings") or ()
            if isinstance(setting, Mapping) and setting.get("id") == TURNS_TRACKED
        ),
        None,
    )
    turns_tracked = None
    if turns is not None:
        turns_tracked = _address(turns.get("readback")) or _address(turns.get("setting"))
    declared_machine = slots.get("machine")
    machine: Mapping[str, Any] = declared_machine if isinstance(declared_machine, Mapping) else {}
    readings = {
        "turns_tracked": turns_tracked,
        "rf_on": _address(machine.get("rf_on")),
        "stored_current": _address(machine.get("stored_current")),
    }
    units = units_of(slots)
    declared = slots.get("steps") or ()
    if not declared:
        return ChainDeclaration(steps=DEFAULT_DECLARATION.steps, units=units, **readings)
    steps: list[ChainStep] = []
    for position, raw in enumerate(declared):
        try:
            step = ChainStep.model_validate({**dict(raw), "position": position})
        except (TypeError, ValueError) as exc:
            raise ValueError(f"chain step {position} is malformed: {exc}") from exc
        if step.id in SERVICE_STEPS and step.command is not None:
            raise ValueError(f"chain step {step.id!r} is run by the service and takes no command")
        if step.id not in SERVICE_STEPS and (step.command is None or step.timeout_s is None):
            raise ValueError(
                f"chain step {step.id!r} is not one the service runs "
                f"({', '.join(SERVICE_STEPS)}), so it declares a command and its timeout_s"
            )
        steps.append(step)
    ids = [step.id for step in steps]
    repeated = sorted({step_id for step_id in ids if ids.count(step_id) > 1})
    if repeated:
        raise ValueError(f"the chain declares {', '.join(repeated)} more than once")
    missing = [step_id for step_id in SERVICE_STEPS if step_id not in ids]
    if missing:
        raise ValueError(f"the chain declares no {', '.join(missing)} step")
    return ChainDeclaration(steps=tuple(steps), units=units, **readings)


# -- the run -----------------------------------------------------------------------


@dataclass(frozen=True)
class ChainRun:
    """One chain run as asked: the ticked steps and how the injections are read.

    Attributes:
        steps: The ticked step ids, in chain order.
        readout: ``tbt`` or ``pseudo_orbit``.
        n_turns: The turns a pseudo-orbit readout averages over.
        n_shots: The injections in the action.
        average: How the included injections are combined.
    """

    steps: tuple[str, ...]
    readout: str = "tbt"
    n_turns: int = 1
    n_shots: int = 1
    average: str = "mean"

    @property
    def reads(self) -> bool:
        """Whether the run reads the BPMs, and so stores a shot."""
        return READOUT in self.steps

    @property
    def fresh(self) -> bool:
        """Whether the run's readout follows its own arm and inject."""
        return ARM in self.steps and INJECT in self.steps


def chain_run(declaration: ChainDeclaration, body: ChainBody) -> ChainRun:
    """The run a panel chain body asks for, refused when it cannot be run.

    The ticked steps run in chain order, whatever order the body lists them in.

    Raises:
        AcquisitionRefused: ``unknown_step`` for a step the chain does not
            declare; ``invalid_chain`` for ``n_turns`` above 1 without a
            pseudo-orbit readout, a pseudo-orbit readout without the readout
            step, or several injections without the inject step.
    """
    unknown = [step for step in body.steps if step not in declaration.ids]
    if unknown:
        raise AcquisitionRefused(
            UNKNOWN_STEP,
            f"the chain declares no step {', '.join(map(repr, unknown))}; its steps are "
            f"{', '.join(declaration.ids)}",
        )
    ticked = set(body.steps)
    steps = tuple(step for step in declaration.ids if step in ticked)
    if body.readout == "pseudo_orbit" and READOUT not in ticked:
        raise AcquisitionRefused(INVALID_CHAIN, "a pseudo-orbit readout needs the readout step")
    if body.n_turns > 1 and body.readout != "pseudo_orbit":
        raise AcquisitionRefused(
            INVALID_CHAIN, f"n_turns {body.n_turns} is read only by a pseudo-orbit readout"
        )
    if body.n_shots > 1 and INJECT not in ticked:
        raise AcquisitionRefused(
            INVALID_CHAIN,
            f"{body.n_shots} shots need the inject step; without it every repeat reads the same",
        )
    return ChainRun(
        steps=steps,
        readout=body.readout,
        n_turns=body.n_turns,
        n_shots=body.n_shots,
        average=body.average,
    )


def default_run(declaration: ChainDeclaration, *, n_shots: int, average: str) -> ChainRun:
    """The agent door's run: the declared default chain, read turn by turn.

    Raises:
        AcquisitionRefused: ``invalid_chain`` when the default chain does not
            arm, inject and read out, so it could make no step.
    """
    steps = declaration.defaults
    missing = [step for step in SERVICE_STEPS if step not in steps]
    if missing:
        raise AcquisitionRefused(
            INVALID_CHAIN,
            f"the declared default chain leaves out {', '.join(missing)}, so it makes no step",
        )
    return ChainRun(steps=steps, n_shots=n_shots, average=average)


def chain_budget_s(
    declaration: ChainDeclaration,
    steps: Collection[str],
    *,
    armed_timeout_s: float,
    max_pause_s: float,
    call_timeout_s: float,
) -> float:
    """The longest one injection of *steps* may take, from declared bounds only.

    Readout: the pre-inject batch and the readout read, one call each. Arm:
    the arm batch and the armed wait. Inject: the request write, the module's
    longest pause, the stored-current read when declared, and the armed-clear
    wait when the arm ran. A command step: its write and its ``timeout_s``.
    """
    ticked = set(steps)
    total = 0.0
    if READOUT in ticked:
        total += 2 * call_timeout_s
    if ARM in ticked:
        total += call_timeout_s + armed_timeout_s
    if INJECT in ticked:
        total += call_timeout_s + max_pause_s
        if declaration.stored_current:
            total += call_timeout_s
        if ARM in ticked:
            total += armed_timeout_s
    for step in declaration.steps:
        if step.command is not None and step.id in ticked:
            total += call_timeout_s + (step.timeout_s or 0.0)
    return total


def snapshot_budget_s(n_shots: int, *, call_timeout_s: float) -> float:
    """The longest the setpoint snapshot of a run of *n_shots* that reads out may take.

    One batch read after the last readout; with ``n_shots > 1`` one more before
    the first injection.
    """
    return (1 if n_shots == 1 else 2) * call_timeout_s


def n_shots_max(
    declaration: ChainDeclaration,
    settings: TrajectorySettings,
    *,
    max_pause_s: float,
    call_timeout_s: float,
) -> int:
    """The most injections of the default chain the action timeout admits.

    The largest ``n`` with ``n × per-shot sum`` plus the setpoint snapshot's
    reads (:func:`snapshot_budget_s`, when the default chain reads out) within
    ``action_timeout_s``.
    """
    per_shot = chain_budget_s(
        declaration,
        declaration.defaults,
        armed_timeout_s=settings.armed_timeout_s,
        max_pause_s=max_pause_s,
        call_timeout_s=call_timeout_s,
    )
    if per_shot <= 0:
        return 0
    reads = READOUT in declaration.defaults

    def snapshot(n: int) -> float:
        return snapshot_budget_s(n, call_timeout_s=call_timeout_s) if reads else 0.0

    timeout = settings.action_timeout_s
    many = int(math.floor((timeout - snapshot(2)) / per_shot))
    if many >= 2:
        return many
    return 1 if per_shot + snapshot(1) <= timeout else 0


def makes_step(shot: ShotResult) -> bool:
    """Whether *shot* goes to the post-shot hook: fresh, TbT and with an included injection."""
    return shot.fresh and shot.readout == "tbt" and shot.shots_included > 0


# -- the setpoint snapshot -----------------------------------------------------------


def setpoint_names(startup: TrajectoryStartup) -> tuple[str, ...]:
    """The RM-column correctors a shot's setpoint snapshot reads, in RM column order.

    Empty when the service has no tool to name its RM columns.
    """
    tool = startup.tool
    if not isinstance(tool, TrajectoryTool):
        return ()
    return tuple(dict.fromkeys([*tool.layout.hcorr_names, *tool.layout.vcorr_names]))


def setpoints_of(values: Mapping[str, Any], names: Sequence[str]) -> dict[str, float] | None:
    """The setpoint of each of *names* in a batch read, ``None`` unless each is a finite number."""
    setpoints: dict[str, float] = {}
    for name in names:
        reading = values.get(name)
        number = None if reading is None else _number(getattr(reading, "value", None))
        if number is None:
            return None
        setpoints[name] = number
    return setpoints


def moved_setpoints(
    snapshot: Mapping[str, float],
    current: Mapping[str, Any],
    bounds: Mapping[str, float],
) -> list[str]:
    """The names of *snapshot* whose *current* value no longer matches it.

    A value matches while it differs from the snapshot by at most
    :data:`SNAPSHOT_TOLERANCE` × the name's ``max_step`` in *bounds*, exactly
    when the name states none. A name *current* lacks, or reads as no finite
    number, moved.
    """
    moved: list[str] = []
    for name, then in snapshot.items():
        now = _number(current.get(name))
        tolerance = SNAPSHOT_TOLERANCE * abs(bounds.get(name, 0.0))
        if now is None or abs(now - then) > tolerance:
            moved.append(name)
    return moved


# -- the action --------------------------------------------------------------------

#: Called once per chain action whose shot makes a step (:func:`makes_step`).
PostShotHook = Callable[[ShotResult], Any]


@dataclass(frozen=True)
class ShotRows:
    """The row layout one readout reads, per BPM, from the declaration's device table.

    Every tuple is in the BPMs' rotated s-order, one entry per BPM.

    Attributes:
        bpms: The BPM names.
        x: The x-row name (the horizontal turn series' address) of each BPM.
        y: The y-row name of each BPM.
        sum: The sum-row name of each BPM.

    Raises:
        ValueError: The layout names no BPM, or its tuples differ in length.
    """

    bpms: tuple[str, ...]
    x: tuple[str, ...]
    y: tuple[str, ...]
    sum: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.bpms:
            raise ValueError("a shot's row layout names no BPM")
        if not len(self.bpms) == len(self.x) == len(self.y) == len(self.sum):
            raise ValueError(
                f"a shot's row layout gives {len(self.bpms)} BPMs, {len(self.x)} x rows, "
                f"{len(self.y)} y rows and {len(self.sum)} sum rows; it needs one of each per BPM"
            )

    @property
    def records(self) -> list[str]:
        """Every record a readout reads: the x rows, the y rows, then the sum rows."""
        return [*self.x, *self.y, *self.sum]

    def without(self, excluded: Collection[str]) -> ShotRows:
        """The layout of the BPMs not in *excluded*, in the same order.

        Raises:
            ValueError: Every BPM is excluded.
        """
        kept = [index for index, bpm in enumerate(self.bpms) if bpm not in excluded]
        if not kept:
            raise ValueError("every BPM of the shot's row layout is excluded")
        return ShotRows(
            bpms=tuple(self.bpms[index] for index in kept),
            x=tuple(self.x[index] for index in kept),
            y=tuple(self.y[index] for index in kept),
            sum=tuple(self.sum[index] for index in kept),
        )


class ShotStopped(Exception):
    """The stop event ended a chain action; no shot file was written.

    Attributes:
        commands_fired: Every command address that went out, in order.
    """

    def __init__(self, commands_fired: list[str]) -> None:
        super().__init__("the shot was stopped")
        self.commands_fired = commands_fired


class ChainFailed(Exception):
    """A step of a chain without readout failed; the action ended there.

    Attributes:
        reason: Why the step failed.
        steps_run: The steps that finished, in the order they ran.
        commands_fired: Every command address that went out, in order.
    """

    def __init__(self, reason: str, steps_run: list[str], commands_fired: list[str]) -> None:
        super().__init__(reason)
        self.reason = reason
        self.steps_run = steps_run
        self.commands_fired = commands_fired


class _ShotFailed(Exception):
    """One injection of the action fails; it is listed, excluded and never retried."""

    def __init__(self, reason: str, stragglers: Sequence[str] = ()) -> None:
        super().__init__(reason)
        self.reason = reason
        self.stragglers = list(stragglers)


@dataclass(frozen=True)
class _Read:
    """One included injection's readout and what was read around it."""

    extraction: ShotExtraction
    raw_samples: int
    beam_current: float | None
    planes: tuple[np.ndarray, np.ndarray, np.ndarray]


def _shot_id() -> str:
    """A fresh shot id: the wall time it was made, then a random tail."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _reading(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _plain(value: Any) -> Any:
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def _number(value: Any) -> float | None:
    try:
        number = float(_plain(value))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _whole(value: Any) -> int | None:
    number = _number(value)
    if number is None or number < 0 or number != int(number):
        return None
    return int(number)


def _pseudo_orbit(
    extracted: ShotExtraction, planes: list[np.ndarray], n_turns: int
) -> ShotExtraction:
    """*extracted* with each BPM's mean over ``n_turns`` turns from ``i0`` in place of its first turn."""
    i0 = extracted.i0
    with warnings.catch_warnings():
        # A BPM whose turns are all NaN averages to NaN; it is masked as lost or reads NaN.
        warnings.simplefilter("ignore", RuntimeWarning)
        x, y, total = (np.nanmean(plane[:, i0 : i0 + n_turns], axis=1) for plane in planes)
    x, y, total = (np.array(a, dtype=float) for a in (x, y, total))
    if extracted.lost_at is not None:
        x[extracted.lost_at :] = np.nan
        y[extracted.lost_at :] = np.nan
        total[extracted.lost_at :] = 0.0
    return ShotExtraction(
        x=x, y=y, sum=total, lost_at=extracted.lost_at, i0=i0, window=extracted.window
    )


def _extracted(
    planes: Sequence[np.ndarray],
    *,
    loss_cutoff: float,
    settings: TrajectorySettings,
    trigger_sample: int,
    readout: str,
    n_turns: int,
    turns: int | None,
) -> ShotExtraction:
    """One injection's extraction from its ``(x, y, sum)`` planes, read as the run reads.

    Raises:
        _ShotFailed: The planes are unreadable, the shot fails extraction, or a
            pseudo-orbit readout asks for more turns than were kept.
    """
    try:
        extracted = extract_shot(
            *planes,
            loss_cutoff=loss_cutoff,
            shot_threshold=settings.shot_threshold_fraction,
            raw_window_turns=settings.raw_window_turns,
            trigger_sample=trigger_sample,
        )
    except ValueError as exc:
        raise _ShotFailed(f"unreadable turn series: {exc}") from exc
    if isinstance(extracted, ExtractionFailure):
        raise _ShotFailed(extracted.reason)
    if readout == "pseudo_orbit":
        bound = turns if turns is not None else planes[2].shape[1] - extracted.i0
        if n_turns > bound:
            held = "turns tracked" if turns is not None else "turns in the buffer"
            raise _ShotFailed(f"n_turns {n_turns} is over the {bound} {held}")
        extracted = _pseudo_orbit(extracted, list(planes), n_turns)
    return extracted


def chain_action(
    startup: TrajectoryStartup,
    store: TrajectoryStore,
    *,
    rows: ShotRows,
    run: ChainRun,
    declaration: ChainDeclaration = DEFAULT_DECLARATION,
    post_shot: PostShotHook | None = None,
    call_timeout_s: float = DEFAULT_TIMEOUT_S,
    excluded_bpms: Collection[str] = (),
    loss_cutoff: float | None = None,
    setup_check: SetupCheck | None = None,
    shot_id: Callable[[], str] = _shot_id,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
    wall: Callable[[], float] = time.time,
    poll_s: float = CHECKPOINT_POLL_S,
) -> Action[ChainResult]:
    """The chain action for :meth:`~.actions.ActionLayer.submit`.

    Args:
        startup: The started service: connector, command set, module and
            settings.
        store: Where the shot file is written.
        rows: The row layout the readout reads.
        run: The ticked steps and how the injections are read.
        declaration: The declared chain the run's steps come from.
        post_shot: Called once, inside the turn, with a shot that makes a
            step (:func:`makes_step`); awaited when it answers an awaitable.
        call_timeout_s: The connector's configured call timeout, ``T_read``.
        excluded_bpms: The BPMs the SolveConfig excludes, by name; they are
            neither armed nor read.
        loss_cutoff: The SolveConfig's loss cutoff, a fraction of the
            injected-charge scale; the configured ``loss_cutoff_fraction``
            when ``None``.
        setup_check: Re-checks the current setup before every injection that
            injects or reads out; the shot carries no ``setup_sha`` without one.
        shot_id: Makes the shot's id.
        sleep: Awaited to pause.
        clock: Seconds, monotonic; bounds the waits.
        wall: Seconds since the epoch; stamps the shot.
        poll_s: Seconds between two reads of a wait.

    Returns:
        An action answering the :class:`~.state.ChainResult`.

    Raises:
        AcquisitionRefused: ``trajectory_unavailable`` when the build gave the
            service nothing to command; ``no_sequence_module`` when the run
            injects and the service has no sequence module; ``over_budget``
            when ``n_shots`` worst-case injections do not fit the action
            timeout; ``invalid_chain`` when every BPM is excluded.
        ValueError: The run's ``n_shots`` or ``n_turns`` is below 1, or it
            names no step.
    """
    commands = startup.commands
    if commands is None:
        raise AcquisitionRefused(
            TRAJECTORY_UNAVAILABLE, "the build has no trajectory model, so there is nothing to arm"
        )
    for name, count in (("n_shots", run.n_shots), ("n_turns", run.n_turns)):
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"{name} must be a whole number of at least 1, not {count!r}")
    if not run.steps:
        raise ValueError("a chain run names at least one step")
    module = startup.sequence
    injects = INJECT in run.steps
    if injects and module is None:
        raise AcquisitionRefused(
            NO_SEQUENCE_MODULE,
            "the service has no sequence module, so nothing requests an injection",
        )
    settings = startup.settings
    armed_timeout_s = settings.armed_timeout_s
    max_pause_s = module.metadata.max_pause_s if module is not None else 0.0
    per_shot_s = chain_budget_s(
        declaration,
        run.steps,
        armed_timeout_s=armed_timeout_s,
        max_pause_s=max_pause_s,
        call_timeout_s=call_timeout_s,
    )
    snapshot_s = snapshot_budget_s(run.n_shots, call_timeout_s=call_timeout_s) if run.reads else 0.0
    worst_s = run.n_shots * per_shot_s + snapshot_s
    if worst_s > settings.action_timeout_s:
        raise AcquisitionRefused(
            OVER_BUDGET,
            f"{run.n_shots} shots of {', '.join(run.steps)} may take {run.n_shots} × "
            f"{per_shot_s:g} s + {snapshot_s:g} s of setpoint reads = {worst_s:g} s, "
            f"over action_timeout_s {settings.action_timeout_s:g} s",
        )
    connector = startup.connector
    excluded = frozenset(excluded_bpms)
    layout = rows
    try:
        rows = layout.without(excluded)
    except ValueError as exc:
        raise AcquisitionRefused(INVALID_CHAIN, f"{exc}; nothing is left to read") from exc
    cutoff = settings.loss_cutoff_fraction if loss_cutoff is None else float(loss_cutoff)
    # Devices may share one arm command and one armed flag; each is written or
    # read once, and stays while any included device uses it.
    devices = [device for device in commands.devices if device.x_row not in excluded]
    arms = list(dict.fromkeys(device.arm for device in devices))
    flags = list(dict.fromkeys(device.armed for device in devices))
    allowed = commands.commands | declaration.commands
    by_id = {step.id: step for step in declaration.steps}
    bpm_of = {
        address: bpm
        for half in (rows.x, rows.y, rows.sum)
        for bpm, address in zip(rows.bpms, half, strict=True)
    }
    fresh = run.fresh
    rechecks = setup_check is not None and (run.reads or INJECT in run.steps)
    snapshot_names = list(setpoint_names(startup)) if run.reads else []

    async def action(admission: Admission, stop: asyncio.Event) -> ChainResult:  # noqa: ARG001
        from osprey.runtime.guarded_run import command_write
        from osprey_connectors.errors import ChannelWriteBlockedError, ChannelWriteFailedError

        fired: list[str] = []
        steps_run: list[str] = []
        setup_shas: list[str | None] = []

        def checkpoint() -> None:
            if stop.is_set():
                raise ShotStopped(list(fired))

        async def pause(seconds: float) -> None:
            checkpoint()
            sleeping = asyncio.ensure_future(sleep(seconds))
            stopping = asyncio.ensure_future(stop.wait())
            try:
                await asyncio.wait({sleeping, stopping}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for task in (sleeping, stopping):
                    task.cancel()
                await asyncio.gather(sleeping, stopping, return_exceptions=True)
            checkpoint()

        async def read(addresses: list[str]) -> dict[str, Any]:
            checkpoint()
            values: dict[str, Any] = await guarded_call(
                lambda: connector.read_multiple_channels(addresses, timeout=call_timeout_s),
                record=lambda _: None,
            )
            checkpoint()
            return values

        async def wait_for(addresses: list[str], value: Any, timeout_s: float, reason: str) -> None:
            pending = list(addresses)
            deadline = clock() + timeout_s
            while pending:
                values = await read(pending)
                pending = [
                    address
                    for address in pending
                    if address not in values or values[address].value != value
                ]
                if not pending:
                    return
                remaining = deadline - clock()
                if remaining <= 0:
                    raise _ShotFailed(reason, pending)
                await pause(min(poll_s, remaining))

        async def fire(addresses: list[str], value: Any, what: str) -> None:
            def record(outcome: Any) -> None:
                if isinstance(outcome, list):
                    fired.extend(outcome)
                else:
                    fired.extend(getattr(outcome, "commands_fired", None) or ())

            checkpoint()
            try:
                await guarded_call(
                    lambda: command_write(connector, addresses, value, commands=allowed),
                    record=record,
                )
            except (ChannelWriteBlockedError, ChannelWriteFailedError) as exc:
                raise _ShotFailed(f"the {what} command did not go out: {exc}") from exc
            checkpoint()

        async def inject() -> None:
            assert module is not None  # refused above when the run injects without one
            handle = SequenceHandle(
                connector,
                commands=commands,
                metadata=module.metadata,
                stop=stop,
                sleep=sleep,
                clock=clock,
                poll_s=poll_s,
            )
            try:
                await module.inject(handle)
            except SequenceStopped:
                fired.extend(handle.commands_fired)
                raise ShotStopped(list(fired)) from None
            except (ChannelWriteBlockedError, ChannelWriteFailedError, TimeoutError) as exc:
                fired.extend(handle.commands_fired)
                raise _ShotFailed(f"inject failed: {exc}") from exc
            fired.extend(handle.commands_fired)

        async def command(step: ChainStep) -> None:
            assert step.command is not None
            await fire([step.command.address], step.command.value, f"{step.id} step's")
            if step.done is not None:
                await wait_for(
                    [step.done.address],
                    step.done.value,
                    step.timeout_s or 0.0,
                    f"{step.id} did not finish within {step.timeout_s:g} s",
                )

        def first_missing_or_stale(
            before: Mapping[str, Any], after: Mapping[str, Any]
        ) -> str | None:
            for index, bpm in enumerate(rows.bpms):
                for half in (rows.x, rows.y, rows.sum):
                    address = half[index]
                    if address not in after:
                        return f"no reading at {bpm}"
                    if fresh and not after[address].timestamp > before[address].timestamp:
                        return f"stale buffer at {bpm}"
            return None

        def extract(
            after: Mapping[str, Any], turns: int | None
        ) -> tuple[ShotExtraction, int, list[np.ndarray]]:
            try:
                read = [
                    np.array([np.asarray(after[address].value, dtype=float) for address in half])
                    for half in (rows.x, rows.y, rows.sum)
                ]
                if read[0].ndim != 2:
                    raise ValueError(f"the records read as shape {read[0].shape}, not BPM × sample")
                n_samples = read[0].shape[1]
                raw_samples = (
                    min(commands.trigger_sample + turns, n_samples)
                    if turns is not None
                    else n_samples
                )
                planes = [plane[:, :raw_samples] for plane in read]
            except ValueError as exc:
                raise _ShotFailed(f"unreadable turn series: {exc}") from exc
            extracted = _extracted(
                planes,
                loss_cutoff=cutoff,
                settings=settings,
                trigger_sample=commands.trigger_sample,
                readout=run.readout,
                n_turns=run.n_turns,
                turns=turns,
            )
            return extracted, raw_samples, planes

        async def one_injection() -> _Read | None:
            records = rows.records
            before: dict[str, Any] = {}
            turns: int | None = None
            beam_current: float | None = None
            armed = False
            setup_addresses = setup_check.addresses() if setup_check and rechecks else []
            if run.reads:
                batch = list(
                    dict.fromkeys(
                        [
                            *records,
                            *([declaration.turns_tracked] if declaration.turns_tracked else []),
                            *setup_addresses,
                        ]
                    )
                )
                before = await read(batch)
            if setup_check is not None and rechecks:
                setup_values = before if run.reads or not setup_addresses else None
                if setup_values is None:
                    try:
                        setup_values = await read(setup_addresses)
                    except ShotStopped:
                        raise
                    except Exception:  # an unread setup is no setup the shot can vouch for
                        setup_values = {}
                setup_shas.append(await setup_check.check(setup_values))
            if run.reads:
                missing = [address for address in records if address not in before]
                if missing:
                    raise _ShotFailed(f"no reading at {bpm_of[missing[0]]}")
                if declaration.turns_tracked and declaration.turns_tracked in before:
                    turns = _whole(before[declaration.turns_tracked].value)
            read_out: _Read | None = None
            for step_id in run.steps:
                if step_id == ARM:
                    await fire(arms, 1, "arm")
                    await wait_for(flags, 1, armed_timeout_s, ARMED_NEVER_SET)
                    armed = True
                elif step_id == INJECT:
                    if declaration.stored_current:
                        current = await read([declaration.stored_current])
                        reading = current.get(declaration.stored_current)
                        beam_current = None if reading is None else _number(reading.value)
                    await inject()
                    if armed:
                        await wait_for(flags, 0, armed_timeout_s, ARMED_NEVER_CLEARED)
                elif step_id == READOUT:
                    after = await read(records)
                    problem = first_missing_or_stale(before, after)
                    if problem is not None:
                        raise _ShotFailed(problem)
                    extracted, raw_samples, planes = extract(after, turns)
                    read_out = _Read(
                        extracted, raw_samples, beam_current, (planes[0], planes[1], planes[2])
                    )
                else:
                    await command(by_id[step_id])
                if step_id not in steps_run:
                    steps_run.append(step_id)
            return read_out

        async def snapshot() -> dict[str, float] | None:
            try:
                values = await read(snapshot_names)
            except ShotStopped:
                raise
            except Exception:  # an unread vector is no snapshot an apply can match
                return None
            return setpoints_of(values, snapshot_names)

        first: dict[str, float] | None = None
        if snapshot_names and run.n_shots > 1:
            first = await snapshot()
        included: list[_Read] = []
        failed: list[ShotFailure] = []
        for index in range(run.n_shots):
            try:
                read_out = await one_injection()
            except _ShotFailed as failure:
                if not run.reads:
                    raise ChainFailed(failure.reason, list(steps_run), list(fired)) from failure
                failed.append(
                    ShotFailure(index=index, reason=failure.reason, stragglers=failure.stragglers)
                )
                continue
            if read_out is not None:
                included.append(read_out)
        shot: ShotResult | None = None
        if run.reads:
            setpoints: dict[str, float] | None = None
            unmoved = True
            if snapshot_names:
                setpoints = await snapshot()
                bounds = max_steps(getattr(connector, "_limits_validator", None))
                unmoved = setpoints is not None and (
                    run.n_shots == 1
                    or (first is not None and not moved_setpoints(first, setpoints, bounds))
                )
            shot = _result(
                shot_id(),
                commands.model,
                layout,
                rows,
                included,
                failed,
                fired,
                wall(),
                readout=run.readout,
                average=run.average,
                n_turns=run.n_turns,
                fresh=run.fresh and unmoved,
                loss_cutoff=cutoff,
                setup_sha=_held_sha(setup_shas),
                setpoints=setpoints,
            )
            # Every injection keeps the samples the shortest one read, so they stack.
            kept = min((read.raw_samples for read in included), default=0)
            raw = (
                RawBuffer.of(
                    [tuple(plane[:, :kept] for plane in read.planes) for read in included],
                    trigger_sample=commands.trigger_sample,
                    units=declaration.units,
                )
                if included
                else None
            )
            await store.save_shot(shot, raw)
            if post_shot is not None and makes_step(shot):
                answer = post_shot(shot)
                if inspect.isawaitable(answer):
                    await answer
        return ChainResult(steps_run=list(steps_run), commands_fired=list(fired), shot=shot)

    return action


def shot_action(
    startup: TrajectoryStartup,
    store: TrajectoryStore,
    *,
    rows: ShotRows,
    n_shots: int = 1,
    post_shot: PostShotHook | None = None,
    call_timeout_s: float = DEFAULT_TIMEOUT_S,
    average: str = "mean",
    declaration: ChainDeclaration = DEFAULT_DECLARATION,
    shot_id: Callable[[], str] = _shot_id,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
    wall: Callable[[], float] = time.time,
    poll_s: float = CHECKPOINT_POLL_S,
) -> Action[ShotResult]:
    """The agent door's action: the declared default chain, answering its shot.

    As :func:`chain_action` with :func:`default_run`'s run.

    Raises:
        AcquisitionRefused: As :func:`chain_action` and :func:`default_run`.
        ValueError: *n_shots* is below 1.
    """
    chained = chain_action(
        startup,
        store,
        rows=rows,
        run=default_run(declaration, n_shots=n_shots, average=average),
        declaration=declaration,
        post_shot=post_shot,
        call_timeout_s=call_timeout_s,
        shot_id=shot_id,
        sleep=sleep,
        clock=clock,
        wall=wall,
        poll_s=poll_s,
    )

    async def action(admission: Admission, stop: asyncio.Event) -> ShotResult:
        result = await chained(admission, stop)
        assert result.shot is not None  # the default chain reads out
        return result.shot

    return action


def _held_sha(shas: Sequence[str | None]) -> str | None:
    """The setup every injection's re-check held, ``None`` unless all held the same one."""
    if not shas or shas[0] is None or any(sha != shas[0] for sha in shas):
        return None
    return shas[0]


def _result(
    shot_id: str,
    model: str,
    layout: ShotRows,
    rows: ShotRows,
    included: list[_Read],
    failed: list[ShotFailure],
    fired: list[str],
    timestamp: float | None,
    *,
    readout: str,
    average: str,
    n_turns: int,
    fresh: bool,
    loss_cutoff: float,
    setup_sha: str | None = None,
    setpoints: Mapping[str, float] | None = None,
) -> ShotResult:
    """The chain's shot: the included injections averaged, keyed by row name.

    *layout* is every BPM's row layout and *rows* the included BPMs'; an
    excluded BPM's rows read NaN.
    """
    currents = [read.beam_current for read in included if read.beam_current is not None]
    excluded = [bpm for bpm in layout.bpms if bpm not in rows.bpms]
    common: dict[str, Any] = {
        "shot_id": shot_id,
        "model": model,
        "bpm_order": list(layout.bpms),
        "shots_included": len(included),
        "shots_failed": failed,
        "commands_fired": list(fired),
        "timestamp": timestamp,
        "readout": readout,
        "average": average,
        "n_turns": n_turns if readout == "pseudo_orbit" else None,
        "loss_cutoff": loss_cutoff,
        "excluded_bpms": excluded,
        "beam_current": sum(currents) / len(currents) if currents else None,
        "raw_samples": min(read.raw_samples for read in included) if included else None,
        "fresh": fresh,
        "setup_sha": setup_sha,
        "setpoints": dict(setpoints) if setpoints is not None else None,
    }
    if not included:
        return ShotResult(
            x=dict.fromkeys(layout.x),
            y=dict.fromkeys(layout.y),
            sum=dict.fromkeys(layout.sum),
            lost=[],
            last_bpm=None,
            i0=None,
            window=None,
            per_shot_lost_at=[],
            **common,
        )
    averaged = average_shots([read.extraction for read in included], average=average)
    lost_at = averaged.lost_at

    def name(at: int | None) -> str | None:
        return None if at is None else rows.bpms[at]

    def plane(names: tuple[str, ...], kept: tuple[str, ...], values: np.ndarray) -> dict:
        read = {row: _reading(v) for row, v in zip(kept, values, strict=True)}
        return {row: read.get(row) for row in names}

    return ShotResult(
        x=plane(layout.x, rows.x, averaged.x),
        y=plane(layout.y, rows.y, averaged.y),
        sum=plane(layout.sum, rows.sum, averaged.sum),
        lost=[] if lost_at is None else list(rows.bpms[lost_at:]),
        last_bpm=name(averaged.last_bpm) if averaged.last_bpm >= 0 else None,
        i0=averaged.i0,
        window=averaged.window,
        per_shot_lost_at=[name(at) for at in averaged.per_shot_lost_at],
        **common,
    )


# -- derived shots -----------------------------------------------------------------


def _layout_of(shot: ShotResult) -> ShotRows:
    """The row layout *shot* was read with: its BPM order and its row names."""
    return ShotRows(
        bpms=tuple(shot.bpm_order), x=tuple(shot.x), y=tuple(shot.y), sum=tuple(shot.sum)
    )


async def derive_shot(
    store: TrajectoryStore,
    model: str,
    parent_id: str,
    *,
    loss_cutoff: float,
    excluded_bpms: Collection[str],
    settings: TrajectorySettings,
) -> ShotResult:
    """Re-extract the stored shot *parent_id* at *loss_cutoff*, stored as its derived shot.

    Every included injection of the parent's raw buffer is extracted again and
    the included ones combined with the parent's ``average``. The derived shot
    is ``<parent_id>-cut``: it names the parent in ``parent_id``, inherits its
    freshness, readout, timestamp, stored current, setup and setpoints, fires
    nothing, keeps no buffer, and replaces an earlier one of the same parent.

    Args:
        store: Where the parent is read and the derived shot written.
        model: The model the parent was read for.
        parent_id: The shot to re-extract.
        loss_cutoff: The new loss cutoff, a fraction of the injected-charge scale.
        excluded_bpms: The BPMs the SolveConfig excludes now.
        settings: The service's tunable keys: the shot threshold and the raw
            window.

    Returns:
        The derived shot, already stored.

    Raises:
        AcquisitionRefused: ``raw_evicted`` when the parent's buffer was
            pruned; ``exclusions_changed`` when the exclusions differ from the
            parent's.
        LookupError: No such shot, or it included no injection and so kept no
            buffer.
    """
    try:
        _root, raw = await store.load_raw(model, parent_id)
    except RawEvicted as exc:
        raise AcquisitionRefused(
            RAW_EVICTED, f"shot {parent_id!r} cannot be re-extracted: {exc}"
        ) from exc
    parent = await store.load_shot(model, parent_id)
    if sorted(set(parent.excluded_bpms)) != sorted(set(excluded_bpms)):
        raise AcquisitionRefused(
            EXCLUSIONS_CHANGED,
            f"shot {parent_id!r} was read with the BPMs "
            f"{', '.join(sorted(parent.excluded_bpms)) or 'none'} excluded, the SolveConfig now "
            f"excludes {', '.join(sorted(excluded_bpms)) or 'none'}; the loss cutoff is saved "
            "and applies from the next shot",
        )
    layout = _layout_of(parent)
    rows = layout.without(parent.excluded_bpms)
    n_included, _planes, n_bpm, _samples = raw.header.shape
    if n_bpm != len(rows.bpms):
        raise LookupError(
            f"the raw buffer of shot {parent_id!r} holds {n_bpm} BPMs, the shot reads "
            f"{len(rows.bpms)}"
        )
    included: list[_Read] = []
    failed: list[ShotFailure] = []
    for index in range(n_included):
        planes = [np.asarray(raw.samples[index, plane], dtype=float) for plane in range(3)]
        try:
            extracted = _extracted(
                planes,
                loss_cutoff=loss_cutoff,
                settings=settings,
                trigger_sample=raw.header.trigger_sample,
                readout=parent.readout,
                n_turns=parent.n_turns or 1,
                turns=None,
            )
        except _ShotFailed as failure:
            failed.append(ShotFailure(index=index, reason=failure.reason))
            continue
        included.append(
            _Read(extracted, planes[0].shape[1], None, (planes[0], planes[1], planes[2]))
        )
    derived = _result(
        f"{parent.shot_id}{DERIVED_SUFFIX}",
        model,
        layout,
        rows,
        included,
        failed,
        [],
        parent.timestamp,
        readout=parent.readout,
        average=parent.average,
        n_turns=parent.n_turns or 1,
        fresh=parent.fresh,
        loss_cutoff=float(loss_cutoff),
    ).model_copy(
        update={
            "parent_id": parent.shot_id,
            "beam_current": parent.beam_current,
            "raw_samples": parent.raw_samples,
            "setpoints": parent.setpoints,
            "setup_sha": parent.setup_sha,
        }
    )
    await store.save_shot(derived, None)
    return derived
