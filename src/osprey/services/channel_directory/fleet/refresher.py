"""The refresher of one target's server fleet, runnable as its own process (D2).

One refresher reads one target through one read-only connector and writes that
target's snapshot (C5) and memory (D9) to the state directory. The app process
starts one per refreshed target, restarts it when it exits and serves the files;
nothing passes between the two but those files and the exit status.

Each cycle, in order:

1. Read the target's probe channel (D5). When it does not answer, every IOC is
   published ``unknown`` and nothing else is read. A target without one is
   ``unverified``.
2. Every :data:`LISTING_EVERY` cycles (and on the first cycle of a process, and
   on the cycle after a failed attempt) list the target's directory and run the
   liveness and health selectors unfiltered, one paged search per selector, under
   ``listing_deadline_s`` (D8). A good result is persisted; a failed one leaves
   the last good one in use and the directory reported unavailable with its age.
   ``cannot_list`` is published only when no listing has ever succeeded, or the
   target has no directory that can list (D7).
3. On ``live``, read the roster sources and apply the named classes; a class's
   PV named by a pattern on the IOC name is used for an IOC the lookup did not
   cover.
4. Read every liveness and health PV concurrently, each on the connector's own
   bound and never cancelled early (C7), including the PVs of remembered IOCs the
   directory no longer lists.
5. Classify, update the memory, write the snapshot with its cycle duration and
   write one audit record naming the target, the PV count and the cycle number.

The first cycle of a process runs one period after the stored last cycle time,
so a successor never compares two counter readings moments apart (D3). An
overrunning cycle skips the ticks it ran over.

The refresher watches ``connector.wait_stale()`` beside its cycle. On connector
loss it publishes the cycle from what it has read, persists the memory and exits
:data:`EXIT_CONNECTOR_LOST`. A connector that cannot be built writes
``not_refreshed`` with the reason and exits :data:`EXIT_CANNOT_BUILD`; a target
the fleet does not refresh (D12) writes ``not_refreshed`` and exits
:data:`EXIT_NOT_REFRESHED`, which a supervisor does not restart.

The process enters the read-only posture before it builds its connector:
sandbox mode and the raw-put block (:mod:`osprey.runtime.read_only_posture`).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import math
import os
import sys
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from osprey.services.channel_directory._connectors import (
    DirectoryQuery,
    Unavailable,
    service_targets,
)
from osprey.services.channel_directory.fleet.memory import FleetMemory, RememberedServer
from osprey.services.channel_directory.fleet.settings import (
    ChannelSelector,
    FleetClass,
    FleetSettings,
    HealthChannel,
    resolve_fleet,
)
from osprey.services.channel_directory.fleet.snapshot import (
    Liveness,
    Outcome,
    ReadPathStatus,
    ServerFleetRow,
    Snapshot,
    SourceStatus,
    read_snapshot,
    write_snapshot,
)
from osprey.services.channel_directory.fleet.states import (
    HealthReading,
    HealthStatus,
    LivenessKind,
    Reading,
    ReadOutcome,
    ReadPath,
    ServerInputs,
    State,
    classify_fleet,
    classify_read,
)
from osprey.services.channel_directory.health_selectors import (
    HealthSelector,
    choose,
    health_selectors,
    match_rows,
)
from osprey.services.channel_directory.roster import (
    MergedRosterRow,
    RosterSource,
    load_roster_sources,
    merge_rosters,
)

__all__ = [
    "DIRECTORY_SOURCE",
    "EXIT_CANNOT_BUILD",
    "EXIT_CONNECTOR_LOST",
    "EXIT_DONE",
    "EXIT_NOT_REFRESHED",
    "LISTING_EVERY",
    "Refresher",
    "main",
    "prepare_read_only_process",
    "run_target",
]

logger = logging.getLogger(__name__)

#: The process ran the cycles it was asked for (only when a cycle limit is given).
EXIT_DONE = 0
#: The connector was lost (``wait_stale`` raised): restart the refresher.
EXIT_CONNECTOR_LOST = 3
#: The connector could not be built: retry with exponential backoff.
EXIT_CANNOT_BUILD = 4
#: The fleet does not refresh this target (D12): do not restart.
EXIT_NOT_REFRESHED = 5

#: Cycles between two listings of the directory (D8).
LISTING_EVERY = 10

#: The name the directory carries in a snapshot's ``sources``.
DIRECTORY_SOURCE = "directory"

#: Rows per page of an unfiltered selector search.
_LOOKUP_PAGE_SIZE = 500
#: Pages one selector search may take before the lookup is called incomplete.
_MAX_PAGES = 10_000
#: How long past the listing deadline a directory call may run before it is dropped.
_LISTING_GRACE_S = 1.0
#: The audit surface and subject of a cycle's record.
_AUDIT_SURFACE = "server_fleet"
_AUDIT_SUBJECT = "fleet_cycle"

AuditCycle = Callable[[str, int, int], None]
"""``audit(target, channel_count, cycle)``: write the cycle's one audit record."""


class _ConnectorLost(Exception):
    """The connector's transport went stale while the refresher waited on it."""


# --- The cycle's working values ---------------------------------------------------


@dataclass(frozen=True)
class _Listing:
    """A listing and its lookup in use, as persisted in the memory."""

    servers: Mapping[str, Mapping[str, Any]]
    lookup: Mapping[str, Mapping[str, Any]]
    at: float


@dataclass
class _Plan:
    """One IOC's row inputs before its reads: where it is, how it is read."""

    name: str
    host: str | None
    fleet_class: FleetClass
    group: str | None
    listed: bool
    directory_status: str | None
    sources: tuple[str, ...]
    expected: str | None
    liveness_channel: str | None
    liveness_ambiguous: bool
    #: Each health PV with its class's spec; ``None`` for a remembered PV, which has no range.
    health_channels: tuple[tuple[str, HealthChannel | None], ...]


@dataclass
class _Previous:
    """What the last published cycle said of one IOC."""

    state: State
    since: float | None
    answered: bool


@dataclass
class _Reads:
    """The cycle's reads, by PV: the value, or the exception it raised, or neither (lost)."""

    values: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, BaseException] = field(default_factory=dict)
    lost: set[str] = field(default_factory=set)


# --- The refresher ------------------------------------------------------------------


class Refresher:
    """One target's refresher; :meth:`run` until the connector is lost.

    Every collaborator is handed in, so the process wrapper (:func:`run_target`)
    builds the real ones and a test builds fakes.

    Args:
        target: The target refreshed.
        settings: The resolved fleet settings.
        connector: A connected, read-only connector for *target*: ``read_channel``
            and ``wait_stale`` are all the refresher calls.
        state_dir: Where the snapshot and the memory are written.
        route: The read path's name in the memory (the connector type).
        probe_channel: The target's probe channel, ``None`` when it has none.
        directory: The target's channel directory backend, ``None`` when there is none
            to list (then *no_directory_reason* says why, or the listing fails
            with *directory_problem*).
        directory_kind: The directory's type (``simulator_view`` selects by
            served signal).
        no_directory_reason: Set when the target has no directory that can list:
            every cycle then publishes ``cannot_list`` with it.
        directory_problem: Why the directory could not be built; each listing
            attempt fails with it and the last good listing stays in use.
        roster_sources: The loaded roster sources; read on ``live`` only.
        audit: Writes the cycle's one audit record.
        clock: Seconds since the epoch.
        sleep: Waits the given seconds.
    """

    def __init__(
        self,
        *,
        target: str,
        settings: FleetSettings,
        connector: Any,
        state_dir: Path,
        route: str,
        probe_channel: str | None,
        directory: Any = None,
        directory_kind: str = "",
        no_directory_reason: str | None = None,
        directory_problem: str | None = None,
        roster_sources: Sequence[RosterSource] = (),
        audit: AuditCycle | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        self.target = target
        self.settings = settings
        self.connector = connector
        self.state_dir = Path(state_dir)
        self.route = route
        self.probe_channel = probe_channel
        self.directory = directory
        self.directory_kind = directory_kind
        self.no_directory_reason = no_directory_reason
        self.directory_problem = directory_problem
        self.roster_sources = tuple(roster_sources) if target == "live" else ()
        self._audit = audit or _audit_cycle
        self._clock = clock
        self._sleep = sleep
        self.memory = FleetMemory.load(self.state_dir, retention=settings.retention_days * 86_400.0)
        self._cycle = 0
        self._previous: dict[str, _Previous] = {}
        self._seed_from_snapshot()
        self._last_listing_cycle: int | None = None
        self._listing_failed: str | None = None
        self._roster: dict[str, MergedRosterRow] = {}
        self._roster_status: tuple[SourceStatus, ...] = ()
        self._stale: asyncio.Future[Any] | None = None

    # -- running -------------------------------------------------------------------

    async def run(self, *, max_cycles: int | None = None) -> int:
        """Cycle on the clock until the connector is lost; the exit status.

        The first cycle runs one period after the stored last cycle time (now,
        when there is none or it is further back). *max_cycles* stops after that
        many cycles with :data:`EXIT_DONE`.
        """
        self._stale = asyncio.ensure_future(self.connector.wait_stale())
        try:
            last = self.memory.last_cycle_time(self.target, self.route)
            now = self._clock()
            next_at = now if last is None else max(now, last + self.settings.period_s)
            ran = 0
            while max_cycles is None or ran < max_cycles:
                delay = next_at - self._clock()
                if delay > 0:
                    await self._race(self._sleep(delay))
                if self._stale.done():
                    raise _ConnectorLost
                await self.cycle()
                ran += 1
                if self._stale.done():
                    raise _ConnectorLost
                next_at = self._next_tick(next_at)
            return EXIT_DONE
        except _ConnectorLost:
            logger.warning("%s: the connector was lost; leaving for a fresh process", self.target)
            self._persist()
            return EXIT_CONNECTOR_LOST
        finally:
            stale, self._stale = self._stale, None
            if not stale.done():
                stale.cancel()
            with contextlib.suppress(BaseException):
                await stale

    def _next_tick(self, scheduled: float) -> float:
        """The next tick after a cycle scheduled at *scheduled*, skipping those it overran."""
        period = self.settings.period_s
        following = scheduled + period
        now = self._clock()
        if now > following:
            skipped = math.ceil((now - following) / period)
            logger.warning("%s: the cycle overran; skipping %d tick(s)", self.target, skipped)
            following += skipped * period
        return following

    async def _race(self, awaitable: Awaitable[Any]) -> Any:
        """Await *awaitable* unless the connector is lost first."""
        task = asyncio.ensure_future(awaitable)
        stale = self._stale
        if stale is None:
            return await task
        await asyncio.wait({task, stale}, return_when=asyncio.FIRST_COMPLETED)
        if task.done():
            return task.result()
        task.cancel()
        with contextlib.suppress(BaseException):
            await task
        raise _ConnectorLost

    def _lost(self) -> bool:
        return self._stale is not None and self._stale.done()

    # -- one cycle -------------------------------------------------------------------

    async def cycle(self) -> Snapshot:
        """Run one cycle and publish it; the snapshot written."""
        started = self._clock()
        self._cycle += 1
        if self.no_directory_reason is not None:
            snapshot = Snapshot.not_ok(
                Outcome.CANNOT_LIST, target=self.target, reason=self.no_directory_reason
            )
            self._publish(snapshot, started, channel_count=0)
            return snapshot

        probe = await self._read_probe()
        if probe is ReadPath.DOWN:
            return self._publish_down(started)

        directory_status = await self._refresh_listing(started)
        listing = self._listing()
        if listing is None:
            reason = self._listing_failed or "no listing of the directory has succeeded yet"
            snapshot = Snapshot.not_ok(
                Outcome.CANNOT_LIST,
                target=self.target,
                reason=f"no listing of the directory has ever succeeded: {reason}",
            )
            self._publish(snapshot, started, channel_count=1 if self.probe_channel else 0)
            return snapshot

        if self.roster_sources and not self._lost():
            await self._read_roster()
        plans = self._plans(listing)
        pvs = _channels_of(plans)
        reads = await self._read_all(pvs)
        snapshot = self._ok_snapshot(
            started,
            probe,
            (directory_status, *self._roster_status),
            self._classify(plans, reads, probe, started),
        )
        self._publish(snapshot, started, channel_count=len(pvs) + (1 if self.probe_channel else 0))
        return snapshot

    def _ok_snapshot(
        self,
        started: float,
        read_path: ReadPath,
        sources: tuple[SourceStatus, ...],
        rows: list[ServerFleetRow],
    ) -> Snapshot:
        """This cycle's ``ok`` snapshot of ``rows``, with the read path ``read_path``."""
        return Snapshot.ok(
            target=self.target,
            generated_at=started,
            cycle=self._cycle,
            period_s=self.settings.period_s,
            cycle_duration_s=max(0.0, self._clock() - started),
            read_path=ReadPathStatus(status=read_path, channel=self.probe_channel),
            sources=sources,
            group_by=self.settings.group_by,
            servers=rows,
        )

    def _publish(self, snapshot: Snapshot, started: float, *, channel_count: int) -> None:
        """Remember the cycle, persist the memory, write the snapshot and the audit record."""
        self.memory.record_cycle(self.target, self.route, started)
        self._persist()
        write_snapshot(self.state_dir, snapshot)
        self._audit(self.target, channel_count, self._cycle)

    def _persist(self) -> None:
        try:
            self.memory.save()
        except OSError:
            logger.exception("%s: the fleet memory could not be written", self.target)

    def _publish_down(self, started: float) -> Snapshot:
        """The probe did not answer: every known IOC ``unknown``, nothing else read."""
        listing = self._listing()
        if listing is None:
            snapshot = Snapshot.not_ok(
                Outcome.CANNOT_LIST,
                target=self.target,
                reason="no listing of the directory has ever succeeded; the read path is down",
            )
        else:
            rows = self._classify(self._plans(listing), _Reads(), ReadPath.DOWN, started)
            skipped = SourceStatus(
                name=DIRECTORY_SOURCE,
                outcome="skipped",
                reason="the read path is down; the directory was not asked",
            )
            snapshot = self._ok_snapshot(started, ReadPath.DOWN, (skipped,), rows)
        self._publish(snapshot, started, channel_count=1)
        return snapshot

    # -- the probe ------------------------------------------------------------------

    async def _read_probe(self) -> ReadPath:
        if self.probe_channel is None:
            return ReadPath.UNVERIFIED
        reads = await self._read_all([self.probe_channel])
        if self.probe_channel in reads.values:
            return ReadPath.OK
        return ReadPath.DOWN

    # -- the listing ------------------------------------------------------------------

    def _listing_due(self) -> bool:
        return (
            self._last_listing_cycle is None
            or self._listing_failed is not None
            or self._cycle - self._last_listing_cycle >= LISTING_EVERY
        )

    async def _refresh_listing(self, now: float) -> SourceStatus:
        """List and look up when due; the directory's status for the snapshot."""
        if self._listing_due() and not self._lost():
            self._last_listing_cycle = self._cycle
            try:
                listing = await self._race(self._list_and_look_up())
            except _ConnectorLost:
                listing = "the connector was lost during the listing"
            if isinstance(listing, str):
                self._listing_failed = listing
                logger.warning("%s: listing the directory failed: %s", self.target, listing)
            else:
                self._listing_failed = None
                servers, lookup = listing
                self.memory.record_good_listing(
                    self.target, self.route, listing=servers, lookup=lookup, at=now
                )
        if self._listing_failed is None:
            return SourceStatus(name=DIRECTORY_SOURCE, outcome="ok")
        good = self.memory.last_good_listing(self.target, self.route)
        age = "no listing has succeeded yet"
        if good is not None:
            age = f"reading from the last good listing, {max(0.0, now - good.at):.0f} s old"
        return SourceStatus(
            name=DIRECTORY_SOURCE,
            outcome="unavailable",
            reason=f"{self._listing_failed}; {age}",
        )

    def _listing(self) -> _Listing | None:
        good = self.memory.last_good_listing(self.target, self.route)
        if good is None:
            return None
        return _Listing(servers=good.listing or {}, lookup=good.lookup or {}, at=good.at)

    async def _list_and_look_up(self) -> tuple[dict[str, Any], dict[str, Any]] | str:
        """The directory's servers and every IOC's liveness and health PVs, or why not."""
        if self.directory is None:
            return self.directory_problem or "the target has no channel directory"
        deadline = self.settings.listing_deadline_s
        selectors = self._selectors()
        try:
            async with asyncio.timeout(deadline + _LISTING_GRACE_S):
                listed, *pages = await asyncio.gather(
                    self.directory.list_servers(deadline),
                    *(self._search_all(selector, deadline) for selector in selectors),
                )
        except TimeoutError:
            return f"the directory did not answer within {deadline:g} s"
        except Exception as exc:  # a backend's failure costs this listing only
            return f"the directory failed: {type(exc).__name__}: {exc}"
        if isinstance(listed, Unavailable):
            return listed.reason
        failed = next((page for page in pages if isinstance(page, str)), None)
        if failed is not None:
            return failed
        servers = {
            row.name: {
                "host": row.host,
                "channel_count": row.channel_count,
                "inactive_count": row.inactive_count,
            }
            for row in listed
        }
        matches = [
            match
            for selector, rows in zip(selectors, pages, strict=True)
            for match in match_rows(selector, rows)
        ]
        lookup: dict[str, dict[str, Any]] = {}
        for (ioc, role), choice in sorted(choose(matches).items()):
            lookup.setdefault(ioc, {})[role] = {
                "channel": choice.name,
                "ambiguous": choice.ambiguous,
            }
        return servers, lookup

    async def _search_all(self, selector: HealthSelector, deadline: float) -> list[dict] | str:
        """Every row one selector's search returns, paged; or why the search failed."""
        rows: list[dict[str, Any]] = []
        for page_number in range(_MAX_PAGES):
            query = DirectoryQuery(
                **selector.query(), page=page_number, page_size=_LOOKUP_PAGE_SIZE
            )
            page = await self.directory.search(query, deadline)
            outcome = getattr(page.outcome, "value", page.outcome)
            if outcome == "unavailable":
                return page.reason or "the directory search failed"
            if page.partial:
                return "the directory search was cut short"
            if outcome == "no_match":
                return rows
            rows.extend({"name": row.name, "server": row.server} for row in page.rows)
            if len(page.rows) < _LOOKUP_PAGE_SIZE or (
                page.total is not None and len(rows) >= page.total
            ):
                return rows
        return "the directory search did not end"

    def _selectors(self) -> tuple[HealthSelector, ...]:
        """One selector per directory-found role: the default class's, and on live each class's."""
        roles: dict[str, str] = {}
        for fleet_class in self._classes():
            for role, selector in _roles(fleet_class):
                if selector.suffix is not None:
                    roles[role] = selector.suffix
        return health_selectors(self.directory_kind, roles)

    def _classes(self) -> list[FleetClass]:
        classes = [self.settings.default_class]
        if self.target == "live":
            classes.extend(self.settings.classes.values())
        return classes

    # -- the roster ------------------------------------------------------------------

    async def _read_roster(self) -> None:
        deadline = self.settings.listing_deadline_s
        try:
            merged = await self._race(
                asyncio.to_thread(merge_rosters, self.roster_sources, deadline)
            )
        except _ConnectorLost:
            return
        self._roster = {row.name: row for row in merged.rows}
        self._roster_status = tuple(
            SourceStatus(name=s.name, outcome=s.outcome, reason=s.reason) for s in merged.sources
        )

    # -- planning --------------------------------------------------------------------

    def _plans(self, listing: _Listing) -> list[_Plan]:
        """Every IOC to read: listed, stated by a roster source, or remembered."""
        names = set(listing.servers) | set(self._roster)
        remembered = {
            ioc.name: ioc
            for ioc in self.memory.no_longer_listed(self.target, self.route, listed=names)
        }
        plans = []
        for name in sorted(names | set(remembered)):
            server = listing.servers.get(name)
            roster = self._roster.get(name)
            memory = remembered.get(name) or self.memory.server(self.target, self.route, name)
            fleet_class = self.settings.class_for(
                roster.class_name if roster else None, target=self.target
            )
            found = listing.lookup.get(name, {})
            liveness_channel, ambiguous = None, False
            if fleet_class.liveness is not None:
                liveness_channel, ambiguous = _address(
                    fleet_class.liveness.selector, _role_name(fleet_class, "liveness"), name, found
                )
            if liveness_channel is None and server is None and memory is not None:
                liveness_channel = memory.liveness_channel
            health: list[tuple[str, HealthChannel | None]] = []
            for index, health_channel in enumerate(fleet_class.health):
                pv, _ = _address(
                    health_channel.selector, _role_name(fleet_class, f"health.{index}"), name, found
                )
                if pv is not None:
                    health.append((pv, health_channel))
            if not health and server is None and memory is not None:
                health = [(pv, None) for pv in memory.health_channels]
            plans.append(
                _Plan(
                    name=name,
                    host=(server or {}).get("host")
                    or (roster.host if roster else None)
                    or (memory.host if memory else None),
                    fleet_class=fleet_class,
                    group=roster.group if roster else None,
                    listed=server is not None,
                    directory_status=_directory_status(server),
                    sources=roster.sources if roster else (),
                    expected=roster.expected if roster else None,
                    liveness_channel=liveness_channel,
                    liveness_ambiguous=ambiguous,
                    health_channels=tuple(health),
                )
            )
        return plans

    # -- reading ---------------------------------------------------------------------

    async def _read_all(self, pvs: Iterable[str]) -> _Reads:
        """Read every PV concurrently on the connector's bound; stop waiting on connector loss.

        A read is never cancelled early: the refresher waits for each to end on
        its own bound. Only when the connector is lost are the reads still open
        abandoned, and those are recorded as lost, not as failures.
        """
        reads = _Reads()
        bound = self.settings.read_bound_s or None
        tasks = {
            pv: asyncio.ensure_future(self.connector.read_channel(pv, timeout=bound))
            for pv in dict.fromkeys(pvs)
        }
        if not tasks:
            return reads
        pending: set[asyncio.Future[Any]] = set(tasks.values())
        while pending:
            waiting = pending if self._stale is None else pending | {self._stale}
            done, _ = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
            pending -= done
            if self._lost():
                break
        for pv, task in tasks.items():
            if not task.done():
                task.cancel()
                reads.lost.add(pv)
                continue
            if task.cancelled():
                reads.lost.add(pv)
            elif task.exception() is not None:
                reads.errors[pv] = task.exception()
            else:
                reads.values[pv] = getattr(task.result(), "value", task.result())
        for task in tasks.values():
            if not task.done():
                with contextlib.suppress(BaseException):
                    await task
        return reads

    # -- classifying -------------------------------------------------------------------

    def _classify(
        self, plans: list[_Plan], reads: _Reads, read_path: ReadPath, now: float
    ) -> list[ServerFleetRow]:
        inputs = [self._inputs(plan, reads, now) for plan in plans]
        verdicts = classify_fleet([i for i, _ in inputs], read_path=read_path)
        rows: list[ServerFleetRow] = []
        previous: dict[str, _Previous] = {}
        for plan, (ioc, kind), verdict in zip(plans, inputs, verdicts, strict=True):
            before = self._previous.get(plan.name)
            since = before.since if before is not None and before.state is verdict.state else now
            answered = ioc.outcome is ReadOutcome.ANSWERED
            previous[plan.name] = _Previous(state=verdict.state, since=since, answered=answered)
            if answered:
                self._remember(plan, ioc, verdict.advance_reading, now)
            rows.append(
                ServerFleetRow(
                    name=plan.name,
                    host=plan.host,
                    class_name=plan.fleet_class.name,
                    group=plan.group,
                    state=verdict.state,
                    since=since,
                    reason=verdict.reason,
                    findings=verdict.findings,
                    directory_status=plan.directory_status,
                    listed=plan.listed,
                    sources=plan.sources,
                    liveness=Liveness(
                        channel=plan.liveness_channel,
                        kind=kind,
                        value=ioc.answer.value if ioc.answer is not None else None,
                        read_at=ioc.answer.read_at if ioc.answer is not None else None,
                        ambiguous=plan.liveness_ambiguous,
                    ),
                    health=ioc.health,
                )
            )
        self._previous = previous
        listed = {plan.name for plan in plans if plan.listed} | set(self._roster)
        self.memory.prune(self.target, self.route, listed=listed, now=now)
        return rows

    def _inputs(self, plan: _Plan, reads: _Reads, now: float) -> tuple[ServerInputs, LivenessKind]:
        liveness = plan.fleet_class.liveness
        kind = (
            LivenessKind.COUNTER
            if liveness is None or liveness.kind == "counter"
            else LivenessKind.VALUE
        )
        pv = plan.liveness_channel
        answer: Reading | None = None
        detail = ""
        if pv is None:
            outcome = ReadOutcome.NO_CHANNEL
        elif pv in reads.values:
            outcome = ReadOutcome.ANSWERED
            answer = Reading(value=_comparable(reads.values[pv]), read_at=now)
        elif pv in reads.errors:
            error = reads.errors[pv]
            outcome = classify_read(error)
            detail = str(error) or type(error).__name__
        elif pv in reads.lost:
            outcome = ReadOutcome.CONNECTOR_LOSS
            detail = "the connector was lost before the read ended"
        else:
            outcome = ReadOutcome.NO_CHANNEL  # nothing was read: the read path is down
        stored = self.memory.previous_reading(self.target, self.route, plan.name)
        before = self._previous.get(plan.name)
        inputs = ServerInputs(
            name=plan.name,
            outcome=outcome,
            kind=kind,
            answer=answer,
            previous=None if stored is None else Reading(value=stored[0], read_at=stored[1]),
            previous_state=before.state if before is not None else None,
            seen=self.memory.seen_before(self.target, self.route, plan.name),
            answered_last_cycle=before.answered if before is not None else False,
            health=tuple(_health(pv, spec, reads) for pv, spec in plan.health_channels),
            expected=plan.expected,
            roster_configured=bool(self.roster_sources),
            detail=detail,
        )
        return inputs, kind

    def _remember(self, plan: _Plan, ioc: ServerInputs, advance: bool, now: float) -> None:
        stored = self.memory.server(self.target, self.route, plan.name)
        counter = stored.counter if stored is not None else None
        counter_read_at = stored.counter_read_at if stored is not None else None
        if advance and ioc.kind is LivenessKind.COUNTER and ioc.answer is not None:
            value = ioc.answer.value
            numeric = isinstance(value, int | float) and not isinstance(value, bool)
            if numeric and math.isfinite(value):
                counter, counter_read_at = float(value), ioc.answer.read_at
        self.memory.record_answer(
            self.target,
            self.route,
            RememberedServer(
                name=plan.name,
                last_answered=now,
                host=plan.host,
                server_class=plan.fleet_class.name,
                liveness_channel=plan.liveness_channel,
                health_channels=tuple(pv for pv, _ in plan.health_channels),
                counter=counter,
                counter_read_at=counter_read_at,
            ),
        )

    # -- resuming --------------------------------------------------------------------

    def _seed_from_snapshot(self) -> None:
        """Take the cycle number and each IOC's last state from the stored snapshot."""
        stored = read_snapshot(self.state_dir, self.target)
        if stored.outcome is not Outcome.OK:
            return
        self._cycle = stored.cycle or 0
        self._previous = {
            row.name: _Previous(
                state=row.state,
                since=row.since,
                answered=row.state in (State.ALIVE, State.STALLED),
            )
            for row in stored.servers
        }


# --- Small helpers -----------------------------------------------------------------


def _role_name(fleet_class: FleetClass, part: str) -> str:
    """The lookup role of a class's PV; the default class's liveness keeps its health role."""
    if part == "liveness" and fleet_class.liveness is not None:
        role = fleet_class.liveness.selector.role
        if role is not None:
            return role
    return f"{fleet_class.name}.{part}"


def _roles(fleet_class: FleetClass) -> list[tuple[str, ChannelSelector]]:
    roles = []
    if fleet_class.liveness is not None:
        roles.append((_role_name(fleet_class, "liveness"), fleet_class.liveness.selector))
    for index, health in enumerate(fleet_class.health):
        roles.append((_role_name(fleet_class, f"health.{index}"), health.selector))
    return roles


def _address(
    selector: ChannelSelector, role: str, ioc: str, found: Mapping[str, Any]
) -> tuple[str | None, bool]:
    """The PV for *ioc* in *role*: the lookup's choice, else the class's name pattern."""
    choice = found.get(role)
    if choice is not None:
        return choice["channel"], bool(choice.get("ambiguous"))
    return selector.address_for(ioc), False


def _directory_status(server: Mapping[str, Any] | None) -> str | None:
    if server is None:
        return None
    channel_count, inactive = server.get("channel_count") or 0, server.get("inactive_count") or 0
    return "Inactive" if channel_count and inactive >= channel_count else "Active"


def _channels_of(plans: Iterable[_Plan]) -> list[str]:
    pvs: list[str] = []
    for plan in plans:
        if plan.liveness_channel is not None:
            pvs.append(plan.liveness_channel)
        pvs.extend(pv for pv, _ in plan.health_channels)
    return list(dict.fromkeys(pvs))


def _comparable(value: Any) -> Any:
    """A reading in the form the memory stores it: numbers as floats."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return value


def _health(pv: str, spec: HealthChannel | None, reads: _Reads) -> HealthReading:
    if pv not in reads.values:
        return HealthReading(channel=pv, value=None, status=HealthStatus.UNREAD)
    value = reads.values[pv]
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return HealthReading(channel=pv, value=None, status=HealthStatus.UNREAD)
    inside = spec is None or spec.in_range(value)
    return HealthReading(
        channel=pv, value=value, status=HealthStatus.OK if inside else HealthStatus.OUT_OF_RANGE
    )


def _audit_cycle(target: str, channel_count: int, cycle: int) -> None:
    """The cycle's one audit record: the target, the PV count and the cycle number."""
    from osprey.audit import posture
    from osprey.audit.envelope import DECISION_ALLOWED
    from osprey.audit.writer import record

    record(
        surface=_AUDIT_SURFACE,
        posture=posture.posture(),
        posture_source=posture.posture_source(),
        session=f"{_AUDIT_SURFACE}:{target}:{os.getpid()}",
        subject=_AUDIT_SUBJECT,
        decision=DECISION_ALLOWED,
        reason="fleet_read",
        detail=f"target={target} channel_count={channel_count} cycle={cycle}",
    )


# --- The process ---------------------------------------------------------------------


def prepare_read_only_process() -> None:
    """Enter the read-only posture: sandbox mode, then the raw-put block."""
    from osprey.runtime.read_only_posture import arm_raw_put_block, enter_sandbox_mode

    enter_sandbox_mode()
    arm_raw_put_block()


async def _build_connector(control_system: Any, target: str) -> Any:
    from osprey_connectors.factory import ConnectorFactory, register_builtin_connectors

    register_builtin_connectors()  # idempotent; a fresh process has none registered
    return await ConnectorFactory.create_control_system_connector(
        control_system, control_target=target
    )


async def _build_directory(config: Mapping[str, Any], target: str, project_dir: Path | None):
    """``(directory, kind, no_directory_reason, problem)`` for *target*."""
    spec = service_targets(config)["directory"].get(target)
    if spec is None:
        return None, "", f"the target {target!r} has no channel directory", None
    from osprey_connectors.factory import ConnectorFactory

    try:
        directory = await ConnectorFactory.create_service(spec, project_dir=project_dir)
    except Exception as exc:
        return None, spec.type, None, f"cannot build the directory {spec.type!r}: {exc}"
    return directory, spec.type, None, None


def _probe_channel(control_system: Any, target: str) -> tuple[str, str | None]:
    """The target's connector type (the read path's name) and its probe channel."""
    from osprey_connectors.types import resolve_target

    connector_type = resolve_target(control_system, target)
    connector = control_system.get("connector") if isinstance(control_system, Mapping) else None
    block = connector.get(connector_type) if isinstance(connector, Mapping) else None
    probe = block.get("probe_channel") if isinstance(block, Mapping) else None
    return connector_type, probe if isinstance(probe, str) and probe else None


async def run_target(
    config: Mapping[str, Any],
    target: str,
    state_dir: Path,
    *,
    project_dir: Path | None = None,
    prepare: Callable[[], None] = prepare_read_only_process,
    build_connector: Callable[[Any, str], Awaitable[Any]] = _build_connector,
    build_directory: Callable[..., Awaitable[tuple[Any, str, str | None, str | None]]] = (
        _build_directory
    ),
    audit: AuditCycle | None = None,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    max_cycles: int | None = None,
) -> int:
    """Refresh *target* until the connector is lost; the process's exit status."""
    section = config.get("channel_directory") or {}
    settings = resolve_fleet(config, section.get("fleet") or {})
    named = settings.target(target)
    if named is None or not named.refreshable:
        reason = (
            named.reason
            if named is not None and named.reason
            else f"the fleet does not refresh {target!r}"
        )
        write_snapshot(
            state_dir, Snapshot.not_ok(Outcome.NOT_REFRESHED, target=target, reason=reason)
        )
        return EXIT_NOT_REFRESHED
    control_system = config.get("control_system")
    try:
        prepare()
        connector = await build_connector(control_system, target)
    except Exception as exc:
        reason = f"cannot build the read-only connector: {type(exc).__name__}: {exc}"
        logger.error("%s: %s", target, reason)
        write_snapshot(
            state_dir, Snapshot.not_ok(Outcome.NOT_REFRESHED, target=target, reason=reason)
        )
        return EXIT_CANNOT_BUILD
    route, probe = _probe_channel(control_system, target)
    directory, kind, no_directory, problem = await build_directory(config, target, project_dir)
    sources: list[RosterSource] = []
    if target == "live" and settings.roster_sources:
        try:
            sources = load_roster_sources(settings.roster_sources)
        except Exception as exc:
            logger.error("%s: the roster sources could not be loaded: %s", target, exc)
    try:
        refresher = Refresher(
            target=target,
            settings=settings,
            connector=connector,
            state_dir=state_dir,
            route=route,
            probe_channel=probe,
            directory=directory,
            directory_kind=kind,
            no_directory_reason=no_directory,
            directory_problem=problem,
            roster_sources=sources,
            audit=audit,
            clock=clock,
            sleep=sleep,
        )
        return await refresher.run(max_cycles=max_cycles)
    finally:
        with contextlib.suppress(Exception):
            await connector.disconnect()


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``python -m osprey.services.channel_directory.fleet.refresher``."""
    parser = argparse.ArgumentParser(description="Refresh one target's server fleet.")
    parser.add_argument("--config", required=True, help="the rendered config.yml")
    parser.add_argument("--target", required=True, help="the target to refresh")
    parser.add_argument("--state-dir", required=True, type=Path, help="where the files go")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    from osprey_connectors.config import load_config

    config = load_config(args.config)
    project_dir = Path(args.config).resolve().parent
    status = asyncio.run(run_target(config, args.target, args.state_dir, project_dir=project_dir))
    sys.exit(status)


if __name__ == "__main__":
    main()
