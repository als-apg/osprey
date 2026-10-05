"""One pass of a picture module: the only code that walks entries for one.

:func:`drive_image_module` is shared by the catch-up (with a budget) and the
manual ``enhance`` drain (without one). A pass runs, in order:

1. skip when a cancelled call of the module is still running (``offload_busy``);
2. :func:`~osprey.services.ariel_search.enhancement.availability.preflight`,
   skipping the module with statuses and attempts untouched when unhealthy;
3. the set-based batch mark of entries with nothing left to do, in keyset
   batches until one returns fewer than ``IMAGE_MARK_BATCH_SIZE`` ids;
4. the walk of entries holding a viewable picture not yet done, through
   ``run_entry`` and the outcome table of
   :class:`~osprey.services.ariel_search.enhancement.base.ImageEntryOutcome`.

The pass breakers live here, once, for every picture module: a second
consecutive transient failure on a different entry ends the pass uncharged,
and three consecutive deterministic failures with one signature on distinct
entries end it too. Every way a module turns out unavailable goes to the
availability tracker, never an ERROR per poll.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from osprey.imaging.formats import is_viewable
from osprey.services.ariel_search.attachments import attachment_id_for
from osprey.services.ariel_search.enhancement import availability
from osprey.services.ariel_search.enhancement._offload import offload_busy
from osprey.services.ariel_search.enhancement.base import ImageEntryOutcome
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from osprey.services.ariel_search.database.repository import ARIELRepository
    from osprey.services.ariel_search.enhancement.base import BaseEnhancementModule
    from osprey.services.ariel_search.models import EnhancedLogbookEntry

logger = get_logger("ariel")

T = TypeVar("T")

#: A picture is not started with less than this many seconds of budget left.
MIN_PICTURE_SECONDS = 5.0

#: Entries one to-do read returns beyond those already visited in the pass.
WALK_BATCH = 1000

#: Consecutive same-signature deterministic failures that end a pass.
DETERMINISTIC_TRIP = 3


async def viewable_in_list_order(
    entry: EnhancedLogbookEntry, repository: ARIELRepository
) -> list[str] | None:
    """The entry's viewable pictures in attachment list order; rows the list does not name come last.

    Read with no lock.

    Args:
        entry: The entry.
        repository: Repository of the database the rows are read from.

    Returns:
        The attachment ids, or None when the store has no copy state.
    """
    entry_id = entry["entry_id"]
    mapping = await repository.get_attachment_rows([entry_id])
    if mapping is None:
        return None
    rows = {row["attachment_id"]: row for row in mapping.get(entry_id, [])}
    order: list[str] = []
    for item in entry.get("attachments") or []:
        attachment_id = attachment_id_for(entry_id, item)
        if attachment_id is not None and attachment_id in rows and attachment_id not in order:
            order.append(attachment_id)
    order.extend(a for a in rows if a not in order)
    return [a for a in order if is_viewable(rows[a])]


@dataclass
class DriveResult:
    """What one pass of a picture module did.

    Attributes:
        module: Module name.
        entries_walked: Entries handed to ``run_entry``.
        marked_complete: Entries completed by the set-based batch mark.
        charged: Entries charged one failed attempt.
        skipped: Why the pass did not run at all (``busy``, ``unavailable``),
            or None when it ran.
        ended: Why the walk stopped early (``budget``, ``stop``, ``limit``, an
            availability or breaker reason), or None when it ran out of entries.
    """

    module: str
    entries_walked: int = 0
    marked_complete: int = 0
    charged: int = 0
    skipped: str | None = None
    ended: str | None = None


class _PassGate:
    """The :class:`~osprey.services.ariel_search.enhancement.base.PictureGate` of one pass."""

    def __init__(
        self,
        module: str,
        marker: str,
        deadline: float | None,
        stop_event: asyncio.Event | None,
    ) -> None:
        self.module = module
        self.marker = marker
        self.deadline = deadline
        self.stop_event = stop_event
        self.entry_id: str | None = None
        self.tripped: str | None = None
        self.stopped: str | None = None
        self._signature: str | None = None
        self._count = 0
        self._last_entry: str | None = None

    def out_of_time(self) -> str | None:
        """Why no further picture may start (``stop`` or ``budget``), or None."""
        if self.stop_event is not None and self.stop_event.is_set():
            return "stop"
        if self.deadline is not None and self.deadline - time.monotonic() < MIN_PICTURE_SECONDS:
            return "budget"
        return None

    def may_start_picture(self) -> bool:
        if self.tripped is not None:
            return False
        why = self.out_of_time()
        if why is not None:
            self.stopped = why
            return False
        return True

    def succeeded(self) -> None:
        availability.note_success(self.module, self.marker)
        self._signature = None
        self._count = 0
        self._last_entry = None

    def deterministic(self, signature: str) -> bool:
        if signature != self._signature:
            self._signature = signature
            self._count = 1
            self._last_entry = self.entry_id
        elif self.entry_id != self._last_entry:
            self._count += 1
            self._last_entry = self.entry_id
        if self._count >= DETERMINISTIC_TRIP:
            self.tripped = signature
            return False
        return availability.has_success(self.module, self.marker)


class _PassEnded(Exception):
    """The pass ends with the module unavailable for ``reason``."""

    def __init__(self, reason: str, message: str, fix: str | None = None) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message
        self.fix = fix


async def _statement(statement: Awaitable[T], module: str) -> T:
    """Await one of the driver's own statements, ending the pass on an availability error."""
    try:
        return await statement
    except Exception as exc:
        reason = availability.unavailable_reason(exc)
        if reason is None:
            raise
        raise _PassEnded(
            reason, f"{type(exc).__name__}: {exc}", availability.fix_for(module, reason, exc)
        ) from exc


async def drive_image_module(
    module: BaseEnhancementModule,
    repository: ARIELRepository,
    *,
    budget: float | None,
    stop_event: asyncio.Event | None,
    progress: Callable[[str], None] | None = None,
    limit: int | None = None,
) -> DriveResult:
    """Run one pass of a ``runs_inline=False`` module.

    Args:
        module: The configured picture module.
        repository: Repository of the database the module works on.
        budget: Seconds the pass may start pictures in; None for no limit. The
            budget is checked only before a picture, so a pass may overrun it
            by at most one call.
        stop_event: Checked before each entry and each picture; set means stop.
        progress: Optional progress callback.
        limit: Most entries handed to ``run_entry``; None for no limit. The
            set-based batch mark is not counted.

    Returns:
        What the pass did.
    """
    from osprey.services.ariel_search.database.repository import IMAGE_MARK_BATCH_SIZE

    name = module.name
    result = DriveResult(module=name)
    started = time.monotonic()
    deadline = None if budget is None else started + budget

    if offload_busy(name):
        logger.info(f"{name}: a cancelled call is still running; skipping this pass")
        result.skipped = "busy"
        return result

    try:
        health = await availability.preflight(module, repository)
        if health.reachable is False:
            raise _PassEnded(health.reason or "unreachable", health.message)
        marker = module.completion_marker()
        if not marker:
            raise _PassEnded("config", "the module reports no completion marker")

        after = ""
        while True:
            ids = await _statement(
                repository.mark_image_module_complete_batch(name, marker, after=after), name
            )
            result.marked_complete += len(ids)
            if ids:
                after = max(ids)
            if len(ids) < IMAGE_MARK_BATCH_SIZE:
                break

        await _walk(module, repository, marker, deadline, stop_event, result, progress, limit)
    except _PassEnded as ended:
        if result.entries_walked == 0 and result.ended is None:
            result.skipped = "unavailable"
        result.ended = ended.reason
        availability.report_unavailable(name, ended.reason, ended.message, ended.fix)
        return result

    availability.report_available(name)
    return result


async def _walk(
    module: BaseEnhancementModule,
    repository: ARIELRepository,
    marker: str,
    deadline: float | None,
    stop_event: asyncio.Event | None,
    result: DriveResult,
    progress: Callable[[str], None] | None,
    limit: int | None = None,
) -> None:
    """Walk the module's to-do entries through ``run_entry`` and the outcome table."""
    from osprey.services.ariel_search.database.repository import MAX_ENHANCEMENT_ATTEMPTS

    name = module.name
    gate = _PassGate(name, marker, deadline, stop_event)
    seen: set[str] = set()
    last_transient: str | None = None

    while True:
        found = await _statement(
            repository.get_incomplete_entries(
                module_name=name, limit=len(seen) + WALK_BATCH, marker=marker
            ),
            name,
        )
        fresh = [e for e in found if e["entry_id"] not in seen]
        if not fresh:
            return
        for entry in fresh:
            entry_id = entry["entry_id"]
            if limit is not None and result.entries_walked >= limit:
                result.ended = "limit"
                return
            seen.add(entry_id)
            why = gate.out_of_time()
            if why is not None:
                result.ended = why
                return
            gate.entry_id = entry_id
            try:
                outcome = await module.run_entry(entry, repository, gate=gate)
            except Exception as exc:
                classified = availability.unavailable_reason(exc)
                outcome = (
                    ImageEntryOutcome.unavailable(classified)
                    if classified is not None
                    else ImageEntryOutcome.transient_error(f"{type(exc).__name__}: {exc}")
                )
            result.entries_walked += 1

            if outcome.kind == "unavailable":
                reason = outcome.reason or "unreachable"
                if reason != "unreachable":
                    raise _PassEnded(reason, f"{entry_id}: the model service answered {reason}")
                health = await availability.preflight(module, repository)
                if health.reachable is False:
                    raise _PassEnded(health.reason or "unreachable", health.message)
                outcome = ImageEntryOutcome.transient_error("unreachable")
            elif outcome.kind == "module_error":
                raise _PassEnded(outcome.reason or "module_error", f"{entry_id}: module refused")

            if outcome.kind == "transient_error":
                if last_transient is not None and last_transient != entry_id:
                    raise _PassEnded(
                        "transient",
                        f"two consecutive transient failures ({outcome.reason})",
                    )
                last_transient = entry_id
                attempts = await _statement(
                    repository.mark_enhancement_failed(
                        entry_id, name, outcome.reason or "transient", marker=marker
                    ),
                    name,
                )
                result.charged += 1
                if attempts >= MAX_ENHANCEMENT_ATTEMPTS:
                    logger.warning(
                        f"Entry {entry_id}: {name} failed {attempts} times; it is left "
                        f"out of later passes ({(outcome.reason or '')[:200]})"
                    )
            else:  # done or partial
                last_transient = None

            if gate.tripped is not None:
                if availability.has_success(name, marker):
                    raise _PassEnded(gate.tripped, "the same failure on three entries in a row")
                raise _PassEnded(
                    "model",
                    "every picture was rejected with the same failure: the model does "
                    "not accept pictures",
                )
            if gate.stopped is not None:
                result.ended = gate.stopped
                return
            if progress and result.entries_walked % 10 == 0:
                progress(f"  {name}: processed {result.entries_walked} entries...")
