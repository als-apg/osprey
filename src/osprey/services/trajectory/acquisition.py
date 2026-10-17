"""Acquisition: the actions that ready the machine for a shot and take one.

**Setup** runs the sequence module's ``bpm_mode`` and, when asked, its
``bunch_clean`` — in that order, each with its own
:class:`~.sequence.SequenceHandle` — inside one turn of the
:class:`~.actions.ActionLayer`, so under the run lock and the bound owner, and
outside any shot: it never arms and never requests an injection. It is
refused while a continuous run is in progress (409 ``continuous_running``),
before the turn and again as the action's first statement. An entry
point the module does not define is not run for ``bpm_mode``, which a facility
whose BPMs need no mode change leaves out; a ``bunch_clean`` asked of a module
that defines none is refused before the turn, as is any setup when the service
has no module (503 ``no_sequence_module``).

Setup answers a :class:`~.state.LastAction` named ``setup`` whose ``moved``
lists every module-slot write that was not refused, in order, in put-back
shape. A stop mid-setup ends at the next handle checkpoint and still answers,
``stopped`` and listing what moved before it; any other failure raises
:class:`SetupFailed`, which carries what moved before it, so nothing written
goes unlisted.

**A shot** runs ``n_shots`` times, inside one turn: read the BPM records'
timestamps; arm every device in one command batch; wait until every
``armed`` flag reads 1; run the module's ``inject``; wait until every flag
reads 0 again; read the records by the sidecar's row names and require each
to be stamped later than before the arm; extract the first turn. Each wait
runs under one deadline across the flags, ``armed_timeout_s``, and polls the
stop event; a flag that missed it is named. A shot that fails is listed with
its reason, excluded and never retried. The included shots are averaged, the
result is written as the shot file and, when at least one shot was included,
handed to the post-shot hook. A shot whose worst case does not fit the
action timeout, ``n_shots × (3·armed_timeout_s + max_pause_s + T_read)``, is
refused before the turn (400 ``over_budget``). A stop ends the shot at its
next checkpoint with :class:`ShotStopped`, which names the commands fired;
no shot file is written and no hook runs. A shot moves no setting: its
writes are commands.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

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
from osprey.services.trajectory.reach import TRAJECTORY_UNAVAILABLE
from osprey.services.trajectory.refusal import Refused
from osprey.services.trajectory.sequence import SequenceHandle, SequenceStopped
from osprey.services.trajectory.startup import TrajectoryStartup
from osprey.services.trajectory.state import LastAction, Moved, ShotFailure, ShotResult
from osprey.services.trajectory.store import TrajectoryStore
from osprey_connectors.control_system.call_timeout import DEFAULT_TIMEOUT_S

__all__ = [
    "ARMED_NEVER_CLEARED",
    "ARMED_NEVER_SET",
    "CONTINUOUS_RUNNING",
    "NO_BUNCH_CLEAN",
    "NO_SEQUENCE_MODULE",
    "OVER_BUDGET",
    "SETUP",
    "SHOT",
    "AcquisitionRefused",
    "PostShotHook",
    "SetupFailed",
    "ShotRows",
    "ShotStopped",
    "setup_action",
    "shot_action",
]

NO_SEQUENCE_MODULE = "no_sequence_module"
NO_BUNCH_CLEAN = "no_bunch_clean"
OVER_BUDGET = "over_budget"
CONTINUOUS_RUNNING = "continuous_running"

#: The name a shot action carries in the service state.
SHOT = "shot"

#: A shot's reason when a device's ``armed`` flag did not read 1 after the arm.
ARMED_NEVER_SET = "armed never set"
#: A shot's reason when a device's ``armed`` flag did not read 0 after the injection.
ARMED_NEVER_CLEARED = "armed never cleared"

#: The name a setup action carries in the service state.
SETUP = "setup"

#: The entry points setup runs, in order; ``bunch_clean`` only when asked.
_BPM_MODE = "bpm_mode"
_BUNCH_CLEAN = "bunch_clean"


class AcquisitionRefused(Refused):
    """An acquisition the service cannot run as asked; nothing was written."""

    statuses = {
        NO_SEQUENCE_MODULE: 503,
        NO_BUNCH_CLEAN: 503,
        TRAJECTORY_UNAVAILABLE: 503,
        OVER_BUDGET: 400,
        CONTINUOUS_RUNNING: 409,
    }


class SetupFailed(Exception):
    """A setup entry point raised; what moved before it is listed for a put-back.

    Attributes:
        moved: Every module-slot write that was not refused, in order.
    """

    def __init__(self, entry_point: str, moved: list[Moved], cause: BaseException) -> None:
        super().__init__(f"setup failed in {entry_point}: {cause!r}")
        self.entry_point = entry_point
        self.moved = moved


def setup_action(
    startup: TrajectoryStartup,
    *,
    bunch_clean: bool = False,
    tag_addresses: Mapping[str, Sequence[str]] | None = None,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
    poll_s: float = CHECKPOINT_POLL_S,
    continuous_running: Callable[[], bool] | None = None,
) -> Action[LastAction]:
    """The setup action for :meth:`~.actions.ActionLayer.submit`.

    Args:
        startup: The started service: connector, command set and module.
        bunch_clean: Run the module's ``bunch_clean`` after ``bpm_mode``.
        tag_addresses: Facility signal to its addresses, one per BPM in BPM
            order, for the module's per-BPM slots.
        sleep: Awaited by the handles to pause.
        clock: Seconds, monotonic; bounds the handles' waits.
        poll_s: Seconds between two reads of a handle's wait.
        continuous_running: Answers whether a continuous run is in progress.

    Returns:
        An action answering a :class:`~.state.LastAction` named ``setup``.

    Raises:
        AcquisitionRefused: ``no_sequence_module`` when the service has no
            sequence module; ``no_bunch_clean`` when *bunch_clean* is asked of
            a module that defines none; ``continuous_running`` while a
            continuous run is in progress, also from the action.
    """

    def refuse_while_continuous() -> None:
        if continuous_running is not None and continuous_running():
            raise AcquisitionRefused(
                CONTINUOUS_RUNNING, "a continuous run is in progress; stop it before a setup"
            )

    refuse_while_continuous()
    module = startup.sequence
    if module is None:
        raise AcquisitionRefused(
            NO_SEQUENCE_MODULE,
            "the service has no sequence module, so there is no setup to run",
        )
    if bunch_clean and module.entry_point(_BUNCH_CLEAN) is None:
        raise AcquisitionRefused(
            NO_BUNCH_CLEAN, f"the sequence module {module.path} defines no bunch_clean"
        )
    names = (_BPM_MODE, _BUNCH_CLEAN) if bunch_clean else (_BPM_MODE,)

    async def action(admission: Admission, stop: asyncio.Event) -> LastAction:  # noqa: ARG001
        refuse_while_continuous()
        moved: list[Moved] = []
        for name in names:
            entry = module.entry_point(name)
            if entry is None:
                continue
            handle = SequenceHandle(
                startup.connector,
                commands=startup.commands,
                metadata=module.metadata,
                entry_point=name,
                stop=stop,
                tag_addresses=tag_addresses,
                sleep=sleep,
                clock=clock,
                poll_s=poll_s,
            )
            try:
                await entry(handle)
            except SequenceStopped:
                moved.extend(handle.moved)
                return LastAction(name=SETUP, stopped=True, moved=moved)
            except Exception as exc:
                moved.extend(handle.moved)
                raise SetupFailed(name, moved, exc) from exc
            moved.extend(handle.moved)
        return LastAction(name=SETUP, stopped=False, moved=moved)

    return action


# -- the shot ----------------------------------------------------------------------

#: Called once per shot action with at least one included shot, with its result.
PostShotHook = Callable[[ShotResult], Any]


@dataclass(frozen=True)
class ShotRows:
    """The sidecar's row layout: which records one shot reads, per BPM.

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
        """Every record a shot reads: the x rows, the y rows, then the sum rows."""
        return [*self.x, *self.y, *self.sum]


class ShotStopped(Exception):
    """The stop event ended a shot action; no shot file was written.

    Attributes:
        commands_fired: Every command address that went out, in order.
    """

    def __init__(self, commands_fired: list[str]) -> None:
        super().__init__("the shot was stopped")
        self.commands_fired = commands_fired


class _ShotFailed(Exception):
    """One shot of the action fails; it is listed, excluded and never retried."""

    def __init__(self, reason: str, stragglers: Sequence[str] = ()) -> None:
        super().__init__(reason)
        self.reason = reason
        self.stragglers = list(stragglers)


def _shot_id() -> str:
    """A fresh shot id: the wall time it was made, then a random tail."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _reading(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def shot_action(
    startup: TrajectoryStartup,
    store: TrajectoryStore,
    *,
    rows: ShotRows,
    n_shots: int = 1,
    post_shot: PostShotHook | None = None,
    call_timeout_s: float = DEFAULT_TIMEOUT_S,
    average: str = "mean",
    shot_id: Callable[[], str] = _shot_id,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
    wall: Callable[[], float] = time.time,
    poll_s: float = CHECKPOINT_POLL_S,
) -> Action[ShotResult]:
    """The shot action for :meth:`~.actions.ActionLayer.submit`.

    Args:
        startup: The started service: connector, command set, module and
            settings.
        store: Where the shot file is written.
        rows: The sidecar's row layout the shot reads.
        n_shots: How many shots are taken and averaged, at least 1.
        post_shot: Called once, inside the turn, with the result when at
            least one shot was included; awaited when it answers an awaitable.
        call_timeout_s: The connector's configured call timeout, ``T_read``.
        average: How included shots are combined, ``"mean"`` or ``"nanmean"``.
        shot_id: Makes the result's shot id.
        sleep: Awaited to pause.
        clock: Seconds, monotonic; bounds the waits.
        wall: Seconds since the epoch; stamps the result.
        poll_s: Seconds between two reads of a wait.

    Returns:
        An action answering the averaged :class:`~.state.ShotResult`.

    Raises:
        AcquisitionRefused: ``trajectory_unavailable`` when the build gave the
            service nothing to command; ``no_sequence_module`` when it has no
            sequence module; ``over_budget`` when ``n_shots`` worst-case
            shots do not fit the action timeout.
        ValueError: *n_shots* is below 1.
    """
    commands = startup.commands
    if commands is None:
        raise AcquisitionRefused(
            TRAJECTORY_UNAVAILABLE, "the build has no trajectory model, so there is nothing to arm"
        )
    module = startup.sequence
    if module is None:
        raise AcquisitionRefused(
            NO_SEQUENCE_MODULE,
            "the service has no sequence module, so nothing requests an injection",
        )
    if isinstance(n_shots, bool) or not isinstance(n_shots, int) or n_shots < 1:
        raise ValueError(f"n_shots must be a whole number of at least 1, not {n_shots!r}")
    settings = startup.settings
    armed_timeout_s = settings.armed_timeout_s
    per_shot_s = 3 * armed_timeout_s + module.metadata.max_pause_s + call_timeout_s
    if n_shots * per_shot_s > settings.action_timeout_s:
        raise AcquisitionRefused(
            OVER_BUDGET,
            f"{n_shots} shots may take {n_shots} × (3 × {armed_timeout_s:g} + "
            f"{module.metadata.max_pause_s:g} + {call_timeout_s:g}) = {n_shots * per_shot_s:g} s, "
            f"over action_timeout_s {settings.action_timeout_s:g} s",
        )
    connector = startup.connector
    bpm_of = {
        address: bpm
        for half in (rows.x, rows.y, rows.sum)
        for bpm, address in zip(rows.bpms, half, strict=True)
    }

    async def action(admission: Admission, stop: asyncio.Event) -> ShotResult:  # noqa: ARG001
        from osprey.runtime.guarded_run import command_write
        from osprey_connectors.errors import ChannelWriteBlockedError, ChannelWriteFailedError

        fired: list[str] = []

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

        async def wait_armed(value: int, reason: str) -> None:
            pending = list(commands.armed)
            deadline = clock() + armed_timeout_s
            while pending:
                values = await read(pending)
                pending = [
                    flag for flag in pending if flag not in values or values[flag].value != value
                ]
                if not pending:
                    return
                remaining = deadline - clock()
                if remaining <= 0:
                    raise _ShotFailed(reason, pending)
                await pause(min(poll_s, remaining))

        async def fire_arm() -> None:
            def record(outcome: Any) -> None:
                if isinstance(outcome, list):
                    fired.extend(outcome)
                else:
                    fired.extend(getattr(outcome, "commands_fired", None) or ())

            checkpoint()
            try:
                await guarded_call(
                    lambda: command_write(
                        connector, list(commands.arm), 1, commands=commands.commands
                    ),
                    record=record,
                )
            except (ChannelWriteBlockedError, ChannelWriteFailedError) as exc:
                raise _ShotFailed(f"the arm command did not go out: {exc}") from exc
            checkpoint()

        async def inject() -> None:
            handle = SequenceHandle(
                connector,
                commands=commands,
                metadata=module.metadata,
                entry_point="inject",
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

        def first_missing_or_stale(
            before: Mapping[str, Any], after: Mapping[str, Any]
        ) -> str | None:
            for index, bpm in enumerate(rows.bpms):
                for half in (rows.x, rows.y, rows.sum):
                    address = half[index]
                    if address not in after:
                        return f"no reading at {bpm}"
                    if not after[address].timestamp > before[address].timestamp:
                        return f"stale buffer at {bpm}"
            return None

        async def one_shot() -> ShotExtraction:
            records = rows.records
            before = await read(records)
            missing = [address for address in records if address not in before]
            if missing:
                raise _ShotFailed(f"no reading at {bpm_of[missing[0]]}")
            await fire_arm()
            await wait_armed(1, ARMED_NEVER_SET)
            await inject()
            await wait_armed(0, ARMED_NEVER_CLEARED)
            after = await read(records)
            problem = first_missing_or_stale(before, after)
            if problem is not None:
                raise _ShotFailed(problem)
            try:
                planes = [
                    np.array([np.asarray(after[address].value, dtype=float) for address in half])
                    for half in (rows.x, rows.y, rows.sum)
                ]
                extracted = extract_shot(
                    *planes,
                    loss_sum_cutoff=settings.loss_sum_cutoff,
                    shot_sum_threshold=settings.shot_sum_threshold,
                    raw_window_turns=settings.raw_window_turns,
                )
            except ValueError as exc:
                raise _ShotFailed(f"unreadable turn series: {exc}") from exc
            if isinstance(extracted, ExtractionFailure):
                raise _ShotFailed(extracted.reason)
            return extracted

        included: list[ShotExtraction] = []
        failed: list[ShotFailure] = []
        for index in range(n_shots):
            try:
                included.append(await one_shot())
            except _ShotFailed as failure:
                failed.append(
                    ShotFailure(index=index, reason=failure.reason, stragglers=failure.stragglers)
                )
        result = _result(
            shot_id(), commands.model, rows, included, failed, fired, wall(), average=average
        )
        await store.save_shot(result)
        if included and post_shot is not None:
            answer = post_shot(result)
            if inspect.isawaitable(answer):
                await answer
        return result

    return action


def _result(
    shot_id: str,
    model: str,
    rows: ShotRows,
    included: list[ShotExtraction],
    failed: list[ShotFailure],
    fired: list[str],
    timestamp: float,
    *,
    average: str,
) -> ShotResult:
    """The shot action's result: the included shots averaged, keyed by row name."""
    common: dict[str, Any] = {
        "shot_id": shot_id,
        "model": model,
        "bpm_order": list(rows.bpms),
        "shots_included": len(included),
        "shots_failed": failed,
        "commands_fired": list(fired),
        "timestamp": timestamp,
    }
    if not included:
        return ShotResult(
            x=dict.fromkeys(rows.x),
            y=dict.fromkeys(rows.y),
            sum=dict.fromkeys(rows.sum),
            lost=[],
            last_bpm=None,
            i0=None,
            window=None,
            per_shot_lost_at=[],
            **common,
        )
    averaged = average_shots(included, average=average)
    lost_at = averaged.lost_at

    def name(at: int | None) -> str | None:
        return None if at is None else rows.bpms[at]

    return ShotResult(
        x={row: _reading(v) for row, v in zip(rows.x, averaged.x, strict=True)},
        y={row: _reading(v) for row, v in zip(rows.y, averaged.y, strict=True)},
        sum={row: _reading(v) for row, v in zip(rows.sum, averaged.sum, strict=True)},
        lost=[] if lost_at is None else list(rows.bpms[lost_at:]),
        last_bpm=name(averaged.last_bpm) if averaged.last_bpm >= 0 else None,
        i0=averaged.i0,
        window=averaged.window,
        per_shot_lost_at=[name(at) for at in averaged.per_shot_lost_at],
        **common,
    )
