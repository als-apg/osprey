"""The directory service: the one place PV directory backends are built and asked.

:class:`DirectoryService` follows the session's control target: the target of the
control-context record, or the deployment's baseline when there is none. For that
target it builds each backend ``service_targets`` names, through the
connectors' factory, and answers searches, counts and records as plain payloads,
on every connectors release: without the PV directory modules every read answers
``unavailable`` with the upgrade reason, and a section with no source on the
current target answers the no-source form.

A search runs the page, the count and the facets concurrently under one deadline
(``settings.timeout_s``); a part that misses it is listed in ``partial``. Facets and
the origin and alarm filters are computed over the whole result set, fetched when
its count is within ``display.filter_cap``. Rows carry their origin mark and lose
the properties ``display.hidden_properties`` names. When the target has an alarm
feed, each row also carries ``alarm``: the feed's state for it, ``None`` when the
feed holds none; a feed that does not answer marks no row and is listed in
``partial`` instead. When the target has an archive, each row carries
``archived``: whether the store records the PV now, ``None`` when the archiver
cannot say (the page's ``archived_reason`` says why).

Backends, the archiver connector, the alarm snapshot and the cached sets belong
to one (target, generation): when either moves, they are dropped, the snapshot
closed, the connector disconnected, and the next call builds afresh.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Coroutine, Iterable, Mapping
from pathlib import Path
from typing import Any, TypeVar

from osprey.services.pv_directory._connectors import (
    DIRECTORY_CONNECTORS_AVAILABLE,
    FLOOR_REASON,
    NO_SOURCE_REASON,
    SERVED_FIELD,
    DirectoryPage,
    DirectoryQuery,
    Outcome,
    Source,
    Unavailable,
    as_payload,
    service_targets,
    unavailable_payload,
)
from osprey.services.pv_directory.alarm_snapshot import AlarmSnapshot
from osprey.services.pv_directory.archive import Extent, read_extent
from osprey.services.pv_directory.demo import resolve_replay_file
from osprey.services.pv_directory.origin import Origin, mark
from osprey_connectors import control_context
from osprey_connectors.archive_targets import archive_targets
from osprey_connectors.factory import ConnectorFactory
from osprey_connectors.types import baseline_target, resolve_archiver_type

__all__ = ["DirectoryService"]

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

_SECTION_KEY = "pv_directory"
_KINDS = ("pv_directory", "alarm_log", "alarm_state", "put_log")
#: The connectors' kind each section reads, as ``service_targets`` keys its table.
_TARGETS_ROW = {"pv_directory": "directory"}
_DEFAULT_TIMEOUT_S = 6.0
_DEFAULT_FILTER_CAP = 5000
#: How long a fetched count or result set answers repeat searches of one query.
_CACHE_TTL_S = 30.0
_CACHE_SIZE = 32
#: How far past its deadline an alarm-feed probe may run before it is cut off.
_PROBE_GRACE_S = 1.0

#: The alarm filter's values, worst first: the severities, then ``none`` (OK or no state).
_ALARM_VALUES = ("UNDEFINED", "INVALID", "MAJOR", "MINOR", "none")
#: The fields of a feed state a row carries.
_ALARM_FIELDS = ("severity", "status", "latched", "acknowledged", "time")
#: The archive filter's values: the store records the PV, does not, or cannot say.
_ARCHIVED_VALUES = ("yes", "no", "unknown")
#: The cache key of a failed archiver build's reason.
_ARCHIVE_FAILED = ("archive", "failed")
#: The cache key of the store's extent; the in-flight key of its read.
_ARCHIVE_EXTENT = ("archive",)
#: The in-flight key of the archiver connector's build.
_ARCHIVE_BUILD = ("archive", "build")
#: What an archive read answers when the target moved while it ran.
_MOVED_REASON = "the target moved"

#: The served names of the row fields the facets count, after ``origin``.
_ROW_FACETS = ("ioc", "record_type")
#: The contract field of each served name.
_CONTRACT_FIELD = {served: field for field, served in SERVED_FIELD.items()}


@dataclasses.dataclass(frozen=True)
class _FullSet:
    """Every row of a search, and the request that fetched them."""

    rows: list[Any]
    request: str


class _TooMany:
    """A result set above the filter cap: its size, and the words that say so."""

    def __init__(self, total: int) -> None:
        self.reason = f"too many results to filter ({total}); narrow the search"


class DirectoryService:
    """Every PV directory read of one process, for the session's current target.

    Args:
        config: The full config mapping, as loaded from the rendered ``config.yml``.
        project_dir: The rendered config's directory.
    """

    def __init__(self, config: Mapping[str, Any], *, project_dir: Path | None) -> None:
        self._config = config
        self._project_dir = project_dir
        section = config.get(_SECTION_KEY)
        self._section: Mapping[str, Any] = section if isinstance(section, Mapping) else {}
        display = self._section.get("display")
        self._display: Mapping[str, Any] = display if isinstance(display, Mapping) else {}
        self._at: tuple[str, int] | None = None
        self._built: dict[str, tuple[Any, Any, str | None]] = {}
        self._snapshot: AlarmSnapshot | None = None
        self._archive: tuple[Any, Any, None] | None = None
        #: The archive reads in flight, by (position, key); a caller that gives up leaves them running.
        self._pending: dict[tuple[Any, ...], asyncio.Task[Any]] = {}
        self._sets: OrderedDict[tuple[Any, ...], tuple[float, Any]] = OrderedDict()
        self._lock = asyncio.Lock()

    @property
    def filter_cap(self) -> int:
        """The largest set fetched whole to facet and export: ``display.filter_cap``, default 5000."""
        return _count_setting(self._display.get("filter_cap"), _DEFAULT_FILTER_CAP)

    # -- the API ----------------------------------------------------------

    async def search(self, **fields: Any) -> dict[str, Any]:
        """One page of the rows a search matches, as the ``DirectoryPage`` payload."""
        target = await self._sync()
        at = self._at
        backend, source, reason = await self._resolve("pv_directory")
        if backend is None:
            return _no_source_page(reason or NO_SOURCE_REASON, target)
        origins, alarms, archived, server = _split_filters(fields)
        cut = bool(origins or alarms or archived)
        deadline = self._timeout()
        started = time.monotonic()

        count = asyncio.create_task(self._count(at, backend, server, deadline))
        full = asyncio.create_task(self._full_set(at, backend, server, count, started, deadline))
        feed = asyncio.create_task(self._alarm_index(deadline))
        store = asyncio.create_task(self._archive_index(deadline))
        page = None if cut else asyncio.create_task(backend.search(server, deadline))
        tasks = [task for task in (count, full, feed, store, page) if task is not None]
        await asyncio.wait(tasks, timeout=deadline)
        late = f"no answer within {deadline:g} s"
        counted, fetched, index = _result(count, late), _result(full, late), _result(feed, late)
        held = _result(store, late)
        await _cancel(tasks)

        partial: list[Mapping[str, str]] = []
        total = counted if isinstance(counted, int) else None
        if isinstance(counted, Unavailable):
            partial.append({"part": "count", "reason": counted.reason})
        if isinstance(index, Unavailable):
            partial.append({"part": "alarm", "reason": index.reason})

        facets: dict[str, list[Mapping[str, Any]]] | None = None
        facets_reason: str | None = None
        marked: list[tuple[Any, Origin]] | None = None
        if isinstance(fetched, _FullSet):
            marked = self._marked(backend, fetched.rows)
            facets = _facets(marked, origins, alarms, archived, index, held)
        else:
            facets_reason = fetched.reason
            if not isinstance(fetched, _TooMany):
                partial.append({"part": "facets", "reason": fetched.reason})

        fields_out: dict[str, Any]
        if cut:
            fields_out = self._cut_page(
                server, (origins, alarms, archived), marked, fetched, index, held
            )
        else:
            fields_out = self._server_page(backend, _result(page, late), total, partial)
            if fields_out.get("total") is None:
                fields_out["total"] = total
        rows = fields_out.pop("rows", [])
        payload: dict[str, Any] = as_payload(
            DirectoryPage(
                searched=_searched(server, origins, alarms, archived),
                source=source,
                partial=partial,
                facets=facets,
                facets_reason=facets_reason,
                **fields_out,
            )
        )
        alarm_states = index if isinstance(index, dict) else None
        payload["rows"] = [_served_row(row, origin, alarm_states, held) for row, origin in rows]
        if isinstance(held, Unavailable):
            payload["archived_reason"] = held.reason
        return payload

    async def count(self, **fields: Any) -> dict[str, Any]:
        """How many rows a search matches: ``{outcome, total, source, reason?}``."""
        target = await self._sync()
        at = self._at
        backend, source, reason = await self._resolve("pv_directory")
        if backend is None:
            return _no_source_count(reason or NO_SOURCE_REASON, target)
        origins, alarms, archived, server = _split_filters(fields)
        deadline = self._timeout()
        started = time.monotonic()
        answer: Any
        if origins or alarms or archived:
            count = asyncio.create_task(self._count(at, backend, server, deadline))
            answer, index, held = await asyncio.gather(
                self._bounded(
                    self._full_set(at, backend, server, count, started, deadline), deadline
                ),
                self._bounded(self._alarm_index(deadline), deadline) if alarms else _none(),
                self._bounded(self._archive_index(deadline), deadline) if archived else _none(),
            )
            await _cancel([count])
            if isinstance(answer, _FullSet):
                answer = _unfiltered(alarms, index, archived, held) or sum(
                    _kept(origins, alarms, archived, index, held)(row, origin)
                    for row, origin in self._marked(backend, answer.rows)
                )
        else:
            answer = await self._bounded(self._count(at, backend, server, deadline), deadline)
        if not isinstance(answer, int):
            return {
                "outcome": "unavailable",
                "total": None,
                "source": as_payload(source),
                "reason": answer.reason,
            }
        return {
            "outcome": "found" if answer else "no_match",
            "total": answer,
            "source": as_payload(source),
        }

    async def describe(self, name: str) -> dict[str, Any]:
        """The record of the PV called exactly *name*: ``{outcome, source, origin, record, reason?}``.

        ``origin`` is set on every answer, ``not_listed`` included.
        """
        target = await self._sync()
        origin = mark([name], dict(self._config))[name]
        backend, source, reason = await self._resolve("pv_directory")
        if backend is None:
            payload = unavailable_payload(reason or NO_SOURCE_REASON, target=target)
            return {**payload, "origin": origin, "record": None}
        deadline = self._timeout()
        answer = await self._bounded(backend.describe(name, deadline), deadline)
        payload = {"source": as_payload(source), "origin": origin, "record": None}
        if isinstance(answer, Unavailable):
            return {**payload, "outcome": "unavailable", "reason": answer.reason}
        if answer == Outcome.NOT_LISTED:
            return {**payload, "outcome": "not_listed"}
        hidden = self._hidden()
        row = self._hide_row(backend, answer.row, hidden)
        record = as_payload(
            dataclasses.replace(
                answer,
                properties={k: v for k, v in answer.properties.items() if k not in hidden},
            )
        )
        record["row"] = _served_row(row, origin)
        inactive = row.status == "Inactive"
        return {
            **payload,
            "outcome": "inactive" if inactive else "found",
            "record": record,
        }

    async def backend(self, kind: str) -> tuple[Any, Any, str | None]:
        """``(backend, source, None)`` for the current target, ``(None, None, reason)`` when none."""
        await self._sync()
        return await self._resolve(kind)

    async def snapshot(self) -> AlarmSnapshot | None:
        """The alarm snapshot of the current target, or ``None`` when it has no alarm feed."""
        target = await self._sync()
        at = self._at
        if self._snapshot is None:
            backend, _source, _reason = await self._resolve("alarm_state")
            if self._at != at:
                return None  # the target moved, or the service closed, during the build
            if backend is not None and self._snapshot is None:
                self._snapshot = AlarmSnapshot(backend, target)
        return self._snapshot

    async def probes(self, kinds: Iterable[str] | None = None) -> dict[str, dict[str, Any]]:
        """Each configured service's probe on the current target, by kind.

        *kinds* limits the answer to those kinds. The ``alarm_state`` probe starts
        the feed through :meth:`snapshot` and waits for catch-up up to the
        deadline. Without the PV directory modules the answer is the one
        ``pv_directory`` floor entry, whatever *kinds* asks.
        """
        target = await self._sync()
        if not DIRECTORY_CONNECTORS_AVAILABLE:
            if self._config.get(_SECTION_KEY) is None:
                return {}
            return {"pv_directory": unavailable_payload(FLOOR_REASON, target=target)}
        wanted = _KINDS if kinds is None else tuple(kinds)
        table = service_targets(self._config)
        asked = [
            kind
            for kind in _KINDS
            if kind in wanted and any(table[_TARGETS_ROW.get(kind, kind)].values())
        ]
        answers = await asyncio.gather(*(self._probe(kind, target) for kind in asked))
        return dict(zip(asked, answers, strict=True))

    async def archive(self, *, fresh: bool = False) -> tuple[Any, Any, str | None]:
        """``(connector, source, None)`` for the current target's archive, else ``(None, None, reason)``.

        The archiver connector is built once per (target, generation), read
        only through ``get_metadata`` and ``store_extent``, and disconnected
        when the target moves. ``NO_SOURCE_REASON`` when the deployment's
        archive does not serve the target. A failed build answers its reason
        for as long as a fetched set is cached, so a store that is down costs
        one wait, not one per search; *fresh* tries again at once.
        """
        await self._sync()
        return await self._archive_connector(fresh=fresh)

    async def archive_extent(self, *, fresh: bool = False) -> Extent | None:
        """What the current target's archive holds, ``None`` when the target has none.

        The answer, known or not, is cached like a fetched result set; *fresh*
        reads the store again.
        """
        await self._sync()
        return await self._extent(self._timeout(), fresh=fresh)

    async def aclose(self) -> None:
        """Close the alarm snapshot and drop every backend and cached set; idempotent."""
        await self._retire()
        self._at = None

    # -- target and backends ---------------------------------------------

    def _position(self) -> tuple[str, int]:
        record = control_context.read_record()
        if record is not None:
            return record.target, record.generation
        return baseline_target(self._config.get("control_system")), 0

    async def _sync(self) -> str:
        """The current target, after dropping what belonged to a previous one."""
        position = self._position()
        if position != self._at:
            async with self._lock:
                if position != self._at:
                    await self._retire()
                    self._at = position
        return position[0]

    async def _retire(self) -> None:
        snapshot, self._snapshot = self._snapshot, None
        archive, self._archive = self._archive, None
        pending, self._pending = list(self._pending.values()), {}
        self._built.clear()
        self._sets.clear()
        await _cancel(pending)
        if snapshot is not None:
            await snapshot.aclose()
        if archive is not None:
            await _disconnect(archive[0])

    async def _resolve(self, kind: str) -> tuple[Any, Any, str | None]:
        """``(backend, source, None)``, or ``(None, None, reason)`` when there is none."""
        if not DIRECTORY_CONNECTORS_AVAILABLE:
            return None, None, FLOOR_REASON
        at = self._at
        if kind in self._built:
            return self._built[kind]
        async with self._lock:
            if kind in self._built and self._at == at:
                return self._built[kind]
            built = await self._build(kind, at[0] if at else self._position()[0])
            if self._at == at:
                self._built[kind] = built
        return built

    async def _build(self, kind: str, target: str) -> tuple[Any, Any, str | None]:
        table = service_targets(self._config)
        spec: Any = table[_TARGETS_ROW.get(kind, kind)][target]
        if spec is None:
            return None, None, NO_SOURCE_REASON
        if kind == "alarm_state" and spec.settings.get("replay_file"):
            spec = self._with_replay_path(spec)
        try:
            backend = await ConnectorFactory.create_service(spec, project_dir=self._project_dir)
        except Exception as failure:
            logger.warning("cannot build the %s backend %r: %s", kind, spec.type, failure)
            return None, None, f"cannot build the {kind} backend {spec.type!r}: {failure}"
        return backend, backend.source, None

    def _with_replay_path(self, spec: Any) -> Any:
        """*spec* with its ``replay_file`` resolved: ``demo:<name>`` to the packaged file."""
        value = spec.settings["replay_file"]
        try:
            path = resolve_replay_file(value, self._project_dir)
        except ValueError:
            return spec  # the replay backend answers an unresolved name as unavailable
        return dataclasses.replace(spec, settings={**spec.settings, "replay_file": str(path)})

    async def _probe(self, kind: str, target: str) -> dict[str, Any]:
        backend, source, reason = await self._resolve(kind)
        if backend is None:
            return unavailable_payload(reason or NO_SOURCE_REASON, target=target)
        deadline = _seconds(backend.spec.settings.get("timeout_s"), self._timeout())
        if kind == "alarm_state":
            snapshot = await self.snapshot()
            if snapshot is None:  # the target moved during the build
                return unavailable_payload(NO_SOURCE_REASON, target=target)
            # The snapshot spends the deadline waiting; the grace only bounds a
            # backend that overruns it, so "still loading" stays the backend's word.
            answer = await self._bounded(snapshot.probe(deadline), deadline + _PROBE_GRACE_S)
        else:
            answer = await self._bounded(backend.probe(deadline), deadline)
        if isinstance(answer, str):
            return {"outcome": "found", "source": as_payload(source), "evidence": answer}
        return {"outcome": "unavailable", "source": as_payload(source), "reason": answer.reason}

    # -- the archive -----------------------------------------------------

    def _has_archive(self) -> bool:
        """Whether the deployment's archive serves the current target."""
        if not DIRECTORY_CONNECTORS_AVAILABLE:
            return False
        target = self._at[0] if self._at else self._position()[0]
        return target in archive_targets(self._config)

    async def _archive_connector(self, *, fresh: bool = False) -> tuple[Any, Any, str | None]:
        if not DIRECTORY_CONNECTORS_AVAILABLE:
            return None, None, FLOOR_REASON
        if not self._has_archive():
            return None, None, NO_SOURCE_REASON
        at = self._at
        if self._archive is not None:
            return self._archive
        failed = None if fresh else self._cached(at, _ARCHIVE_FAILED)
        if isinstance(failed, str):
            return None, None, failed
        return await self._once(
            at, _ARCHIVE_BUILD, lambda: self._build_archive(at), (None, None, _MOVED_REASON)
        )

    async def _once(
        self,
        at: tuple[str, int] | None,
        key: tuple[Any, ...],
        read: Callable[[], Coroutine[Any, Any, _T]],
        moved: _T,
    ) -> _T:
        """The one in-flight *read* of *key* on *at*, started if none runs.

        The read runs to its own deadline whatever its callers do, so a caller
        that gives up first neither repeats it nor keeps it from recording its
        answer; a store that is down costs one wait, not one per search.
        *moved* answers when the target moves while the read runs.
        """
        full = (at, *key)
        task = self._pending.get(full)
        if task is None:
            task = asyncio.create_task(read())
            self._pending[full] = task
            task.add_done_callback(
                lambda done: self._pending.pop(full) if self._pending.get(full) is done else None
            )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if task.cancelled() and not (current is not None and current.cancelling()):
                return moved  # retired with its target, not this caller's cancellation
            raise

    async def _build_archive(self, at: tuple[str, int] | None) -> tuple[Any, Any, str | None]:
        section = self._config.get("archiver")
        target = at[0] if at else self._position()[0]
        deadline = self._timeout()
        try:
            connector = await asyncio.wait_for(
                ConnectorFactory.create_archiver_connector(dict(section or {})), deadline
            )
        except TimeoutError:
            reason = f"the archive did not answer within {deadline:g} s"
            self._store(at, _ARCHIVE_FAILED, reason)
            return None, None, reason
        except Exception as failure:
            logger.warning("cannot reach the archive", exc_info=True)
            reason = f"cannot reach the archive: {type(failure).__name__}"
            self._store(at, _ARCHIVE_FAILED, reason)
            return None, None, reason
        if self._at != at:
            await _disconnect(connector)
            return None, None, _MOVED_REASON
        source = Source(
            kind="archive",
            target=target,  # type: ignore[arg-type]
            service=resolve_archiver_type(section),
            rehearsal=target == "standin",
        )
        self._archive = (connector, source, None)
        return self._archive

    async def _extent(self, deadline: float, *, fresh: bool = False) -> Extent | None:
        if not self._has_archive():
            return None
        at = self._at
        if not fresh:
            cached = self._cached(at, _ARCHIVE_EXTENT)
            if isinstance(cached, Extent):
                return cached
        return await self._once(
            at,
            _ARCHIVE_EXTENT,
            lambda: self._read_extent(at, deadline),
            Extent(reason=_MOVED_REASON),
        )

    async def _read_extent(self, at: tuple[str, int] | None, deadline: float) -> Extent:
        """Read the store's extent and cache the answer, known or not, like a fetched set."""
        connector, _source, reason = await self._archive_connector()
        extent = (
            Extent(reason=reason) if connector is None else await read_extent(connector, deadline)
        )
        self._store(at, _ARCHIVE_EXTENT, extent)
        return extent

    async def _archive_index(self, deadline: float) -> Any:
        """The channels the store records: a set, ``Unavailable`` when unknown, ``None`` with no archive."""
        extent = await self._extent(deadline)
        if extent is None:
            return None
        if extent.channels is None:
            return Unavailable(extent.reason or "the archive does not say what it holds")
        return extent.channels

    # -- searching -------------------------------------------------------

    async def _count(self, at: Any, backend: Any, server: Any, deadline: float) -> Any:
        key = ("count", dataclasses.replace(server, page=0, page_size=0))
        cached = self._cached(at, key)
        if cached is not None:
            return cached
        answer = await backend.count(server, deadline)
        if isinstance(answer, int):
            self._store(at, key, answer)
        return answer

    async def _full_set(
        self,
        at: Any,
        backend: Any,
        server: Any,
        count: Awaitable[Any],
        started: float,
        deadline: float,
    ) -> Any:
        """Every row *server* matches, or why the set cannot be had."""
        key = ("set", dataclasses.replace(server, page=0, page_size=0))
        cached = self._cached(at, key)
        if cached is not None:
            return cached
        total = await count
        if not isinstance(total, int):
            return total
        cap = self.filter_cap
        if total > cap:
            return _TooMany(total)
        if total == 0:
            return _FullSet([], "")
        left = max(deadline - (time.monotonic() - started), 0.0)
        answer = await backend.search(dataclasses.replace(server, page=0, page_size=total), left)
        if answer.outcome == Outcome.UNAVAILABLE:
            return Unavailable(answer.reason or "no answer")
        fetched = _FullSet(list(answer.rows), answer.request)
        self._store(at, key, fetched)
        return fetched

    async def _alarm_index(self, deadline: float) -> Any:
        """The feed's state of every channel it holds, by channel.

        ``Unavailable`` when the feed does not answer, ``None`` when the target has
        no alarm-state source.
        """
        backend, _source, reason = await self._resolve("alarm_state")
        if backend is None:
            return None if reason is None or reason == NO_SOURCE_REASON else Unavailable(reason)
        snapshot = await self.snapshot()
        if snapshot is None:
            return Unavailable("the target moved")
        answer = await snapshot.states(deadline)
        if isinstance(answer, Unavailable):
            return answer
        return {row.channel: row.state for row in answer.rows}

    def _cut_page(
        self,
        query: Any,
        filters: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]],
        marked: list[tuple[Any, Origin]] | None,
        fetched: Any,
        index: Any,
        held: Any,
    ) -> dict[str, Any]:
        """The page of an origin-, alarm- or archive-filtered search, cut from the fetched set."""
        origins, alarms, archived = filters
        if marked is None:
            return {"outcome": Outcome.UNAVAILABLE, "request": "", "reason": fetched.reason}
        refused = _unfiltered(alarms, index, archived, held)
        if refused is not None:
            return {"outcome": Outcome.UNAVAILABLE, "request": "", "reason": refused.reason}
        keeps = _kept(origins, alarms, archived, index, held)
        kept = [pair for pair in marked if keeps(*pair)]
        size = max(query.page_size, 0)
        start = max(query.page, 0) * size
        rows = kept[start : start + size]
        return {
            "outcome": Outcome.FOUND if kept else Outcome.NO_MATCH,
            "request": fetched.request,
            "total": len(kept),
            "rows": rows,
            "truncated": start + len(rows) < len(kept),
        }

    def _server_page(
        self, backend: Any, page: Any, total: int | None, partial: list[Mapping[str, str]]
    ) -> dict[str, Any]:
        """The page the server answered, its rows marked."""
        if isinstance(page, Unavailable):
            return {"outcome": Outcome.UNAVAILABLE, "request": "", "reason": page.reason}
        known = {entry["part"] for entry in partial}
        partial.extend(
            dict(entry)
            for entry in page.partial
            if entry.get("part") not in known
            and not (entry.get("part") == "count" and total is not None)
        )
        return {
            "outcome": page.outcome,
            "request": page.request,
            "total": page.total if total is None else total,
            "rows": self._marked(backend, page.rows),
            "truncated": page.truncated,
            "reason": page.reason,
        }

    def _marked(self, backend: Any, rows: Iterable[Any]) -> list[tuple[Any, Origin]]:
        """*rows* without hidden properties, each paired with its origin mark."""
        rows = list(rows)
        marks = mark([row.name for row in rows], dict(self._config))
        hidden = self._hidden()
        return [(self._hide_row(backend, row, hidden), marks[row.name]) for row in rows]

    def _hidden(self) -> frozenset[str]:
        names = self._display.get("hidden_properties")
        if not isinstance(names, list):
            return frozenset()
        return frozenset(name for name in names if isinstance(name, str))

    def _hide_row(self, backend: Any, row: Any, hidden: frozenset[str]) -> Any:
        """*row* with the fields *backend* fills from a hidden property cleared."""
        if not hidden:
            return row
        by_property = backend.field_properties
        cleared = {field: None for field, name in by_property.items() if name in hidden}
        return dataclasses.replace(row, **cleared) if cleared else row

    # -- small helpers ---------------------------------------------------

    def _timeout(self) -> float:
        settings = self._section.get("settings")
        value = settings.get("timeout_s") if isinstance(settings, Mapping) else None
        return _seconds(value, _DEFAULT_TIMEOUT_S)

    async def _bounded(self, call: Awaitable[Any], deadline: float) -> Any:
        """*call*'s answer, or ``Unavailable`` when it misses *deadline* or raises."""
        try:
            return await asyncio.wait_for(call, deadline)
        except TimeoutError:
            return Unavailable(f"no answer within {deadline:g} s")
        except Exception as failure:
            logger.warning("directory backend failed", exc_info=True)
            return Unavailable(f"the service failed: {failure}")

    def _cached(self, at: Any, key: tuple[Any, ...]) -> Any:
        entry = self._sets.get((at, *key))
        if entry is None or time.monotonic() - entry[0] > _CACHE_TTL_S:
            return None
        return entry[1]

    def _store(self, at: Any, key: tuple[Any, ...], value: Any) -> None:
        """Cache *value* under the (target, generation) the call started on, if still current."""
        if at != self._at:
            return
        full = (at, *key)
        self._sets[full] = (time.monotonic(), value)
        self._sets.move_to_end(full)
        while len(self._sets) > _CACHE_SIZE:
            self._sets.popitem(last=False)


def _no_source_page(reason: str, target: str) -> dict[str, Any]:
    """The whole ``DirectoryPage`` a search answers with no source on *target*."""
    return {
        **unavailable_payload(reason, target=target),
        "searched": "",
        "request": "",
        "total": None,
        "rows": [],
        "truncated": False,
        "partial": [],
        "facets": None,
        "facets_reason": None,
    }


def _no_source_count(reason: str, target: str) -> dict[str, Any]:
    """The count a search answers with no source on *target*: ``total`` is ``None``."""
    return {**unavailable_payload(reason, target=target), "total": None}


async def _cancel(tasks: Iterable[asyncio.Task[Any]]) -> None:
    """Cancel the *tasks* still running and wait until each has finished."""
    tasks = list(tasks)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def _result(task: asyncio.Task[Any] | None, late: str) -> Any:
    """A finished task's answer; ``Unavailable`` when it is not done or raised."""
    if task is None:
        return Unavailable(late)
    if not task.done() or task.cancelled():
        return Unavailable(late)
    failure = task.exception()
    if failure is not None:
        logger.warning("directory backend failed", exc_info=failure)
        return Unavailable(f"the service failed: {failure}")
    return task.result()


async def _none() -> None:
    return None


async def _disconnect(connector: Any) -> None:
    """Disconnect an archiver connector; a failure is logged, never raised."""
    try:
        await connector.disconnect()
    except Exception:
        logger.warning("the archive connector did not disconnect cleanly", exc_info=True)


def _alarm_value(index: Mapping[str, Any], name: str) -> str:
    """The alarm filter's value of the PV *name*: its severity, or ``none`` for OK or no state."""
    state = index.get(name)
    severity = None if state is None else state.severity
    return "none" if severity in (None, "OK") else str(severity)


def _archived_value(held: Any, name: str) -> str:
    """The archive filter's value of the PV *name*: ``yes``, ``no``, or ``unknown``."""
    if not isinstance(held, frozenset):
        return "unknown"
    return "yes" if name in held else "no"


def _unfiltered(
    alarms: tuple[str, ...], index: Any, archived: tuple[str, ...] = (), held: Any = None
) -> Unavailable | None:
    """Why an alarm or archive filter cannot be applied: no answer, or no such service."""
    if alarms and not isinstance(index, dict):
        reason = index.reason if isinstance(index, Unavailable) else NO_SOURCE_REASON
        return Unavailable(f"cannot filter by alarm: {reason}")
    if archived and not isinstance(held, frozenset):
        reason = held.reason if isinstance(held, Unavailable) else NO_SOURCE_REASON
        return Unavailable(f"cannot filter by archive: {reason}")
    return None


def _kept(
    origins: tuple[str, ...],
    alarms: tuple[str, ...],
    archived: tuple[str, ...],
    index: Any,
    held: Any,
) -> Any:
    """Whether a marked row passes the origin, alarm and archive filters.

    A filter whose service did not answer passes every row; :func:`_unfiltered`
    refuses such a filter before any row is kept.
    """
    states = index if isinstance(index, dict) else None
    channels = held if isinstance(held, frozenset) else None

    def keeps(row: Any, origin: Origin) -> bool:
        if origins and origin not in origins:
            return False
        if (
            archived
            and channels is not None
            and _archived_value(channels, row.name) not in archived
        ):
            return False
        return not (alarms and states is not None) or _alarm_value(states, row.name) in alarms

    return keeps


def _facets(
    marked: list[tuple[Any, Origin]],
    origins: tuple[str, ...],
    alarms: tuple[str, ...],
    archived: tuple[str, ...],
    index: Any,
    held: Any,
) -> dict[str, list[Mapping[str, Any]]]:
    """Value counts over the fetched set, each group under the other groups' filters.

    The ``alarm`` group is present when the feed answered, its values worst first;
    the ``archived`` group when the target has an archive (``unknown`` for every
    row when the archive cannot say what it holds).
    """
    by_origin_group = _kept((), alarms, archived, index, held)
    by_alarm_group = _kept(origins, (), archived, index, held)
    by_archive_group = _kept(origins, alarms, (), index, held)
    every = _kept(origins, alarms, archived, index, held)
    kept = [row for row, origin in marked if every(row, origin)]
    facets: dict[str, list[Mapping[str, Any]]] = {
        "origin": _tally(origin for row, origin in marked if by_origin_group(row, origin)),
        **{
            name: _tally(getattr(row, _CONTRACT_FIELD[name]) for row in kept)
            for name in _ROW_FACETS
        },
    }
    if isinstance(index, dict):
        counted = _tally(
            _alarm_value(index, row.name) for row, origin in marked if by_alarm_group(row, origin)
        )
        facets["alarm"] = sorted(counted, key=_alarm_order)
    if held is not None:
        stored = _tally(
            _archived_value(held, row.name)
            for row, origin in marked
            if by_archive_group(row, origin)
        )
        facets["archived"] = sorted(
            stored, key=lambda entry: _ARCHIVED_VALUES.index(entry["value"])
        )
    return facets


def _alarm_order(entry: Mapping[str, Any]) -> int:
    """An alarm facet value's place: worst first, a severity outside the list before all."""
    value = entry["value"]
    return _ALARM_VALUES.index(value) if value in _ALARM_VALUES else -1


def _tally(values: Iterable[Any]) -> list[Mapping[str, Any]]:
    counts: dict[str, int] = {}
    for value in values:
        if value is not None:
            counts[value] = counts.get(value, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [{"value": value, "count": count} for value, count in ordered]


def _split_filters(
    fields: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], Any]:
    """The origin, alarm and archive filters of a search's served keywords, and the server's query."""
    rest = {_CONTRACT_FIELD.get(key, key): value for key, value in fields.items()}
    origins, alarms, archived = (
        _values(rest.pop(key, None)) for key in ("origin", "alarm", "archived")
    )
    return origins, alarms, archived, DirectoryQuery(**rest)


def _values(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    return (value,) if isinstance(value, str) else tuple(value)


def _served_row(
    row: Any, origin: Origin, alarm_states: Mapping[str, Any] | None = None, held: Any = None
) -> dict[str, Any]:
    """*row* as served: its name, its origin mark, its alarm when the feed answered, the rest.

    With an archive on the target (*held* not ``None``) the row carries
    ``archived``: ``True`` or ``False``, ``None`` when the archive cannot say.
    """
    payload: dict[str, Any] = as_payload(row)
    served: dict[str, Any] = {"name": payload.pop("name"), "origin": origin}
    if alarm_states is not None:
        state = alarm_states.get(served["name"])
        served["alarm"] = (
            None if state is None else {key: getattr(state, key) for key in _ALARM_FIELDS}
        )
    if held is not None:
        served["archived"] = served["name"] in held if isinstance(held, frozenset) else None
    return {**served, **payload}


def _searched(
    query: Any,
    origins: tuple[str, ...],
    alarms: tuple[str, ...] = (),
    archived: tuple[str, ...] = (),
) -> str:
    """The search in plain words."""
    parts = []
    if query.words.strip():
        parts.append(f"name or description matches {' '.join(query.words.split())}")
    if query.name:
        parts.append(f"name matches {query.name}")
    if query.status:
        parts.append(f"status {query.status}")
    if query.server:
        parts.append(f"IOC {' or '.join(query.server)}")
    if query.channel_type:
        parts.append(f"record type {' or '.join(query.channel_type)}")
    parts.extend(f"{key} {value}" for key, value in query.properties.items())
    if origins:
        parts.append(f"origin {' or '.join(origins)}")
    if alarms:
        parts.append(f"alarm {' or '.join(alarms)}")
    if archived:
        parts.append(f"archived {' or '.join(archived)}")
    return " · ".join(parts) or "every PV"


def _seconds(value: Any, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return default
    return float(value)


def _count_setting(value: Any, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return default
    return int(value)
