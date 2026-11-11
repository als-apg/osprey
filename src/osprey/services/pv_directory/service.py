"""The directory service: the one place PV directory backends are built and asked.

:class:`DirectoryService` follows the session's control target: the target of the
control-context record, or the deployment's baseline when there is none. For that
target it builds each backend ``pv_directory_targets`` names, through the
connectors' factory, and answers searches, counts and records as plain payloads,
on every connectors release: without the PV directory modules every read answers
``unavailable`` with the upgrade reason, and a section with no source on the
current target answers the no-source form.

A search runs the page, the count and the facets concurrently under one deadline
(``settings.timeout_s``); a part that misses it is listed in ``partial``. Facets and
the origin filter are computed over the whole result set, fetched when its count
is within ``display.filter_cap``. Rows carry their origin mark and lose the
properties ``display.hidden_properties`` names.

Backends, the alarm snapshot and the cached sets belong to one (target,
generation): when either moves, they are dropped, the snapshot closed, and the
next call builds afresh.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from collections import OrderedDict
from collections.abc import Awaitable, Iterable, Mapping
from pathlib import Path
from typing import Any

from osprey.services.pv_directory._connectors import (
    DEFAULT_DESCRIPTION_PROPERTY,
    DIRECTORY_CONNECTORS_AVAILABLE,
    FLOOR_REASON,
    NO_SOURCE_REASON,
    ROW_FIELD_PROPERTIES,
    DirectoryPage,
    Outcome,
    SearchQuery,
    Unavailable,
    as_payload,
    pv_directory_targets,
    unavailable_payload,
)
from osprey.services.pv_directory.alarm_snapshot import AlarmSnapshot
from osprey.services.pv_directory.demo import resolve_replay_file
from osprey.services.pv_directory.origin import mark
from osprey_connectors import control_context
from osprey_connectors.factory import ConnectorFactory
from osprey_connectors.types import baseline_target

__all__ = ["DirectoryService"]

logger = logging.getLogger(__name__)

_SECTION_KEY = "pv_directory"
_KINDS = ("pv_directory", "alarm_log", "alarm_state", "put_log")
_DEFAULT_TIMEOUT_S = 6.0
_DEFAULT_FILTER_CAP = 5000
#: How long a fetched count or result set answers repeat searches of one query.
_CACHE_TTL_S = 30.0
_CACHE_SIZE = 32
#: How far past its deadline an alarm-feed probe may run before it is cut off.
_PROBE_GRACE_S = 1.0

_FACETS = ("origin", "ioc", "record_type")


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
        query = SearchQuery(**fields)
        server = dataclasses.replace(query, origin=())
        deadline = self._timeout()
        started = time.monotonic()

        count = asyncio.create_task(self._count(at, backend, server, deadline))
        full = asyncio.create_task(self._full_set(at, backend, server, count, started, deadline))
        page = None if query.origin else asyncio.create_task(backend.search(server, deadline))
        tasks = [task for task in (count, full, page) if task is not None]
        await asyncio.wait(tasks, timeout=deadline)
        late = f"no answer within {deadline:g} s"
        counted, fetched = _result(count, late), _result(full, late)
        await _cancel(tasks)

        partial: list[Mapping[str, str]] = []
        total = counted if isinstance(counted, int) else None
        if isinstance(counted, Unavailable):
            partial.append({"part": "count", "reason": counted.reason})

        facets: dict[str, list[Mapping[str, Any]]] | None = None
        facets_reason: str | None = None
        marked: list[Any] | None = None
        if isinstance(fetched, _FullSet):
            marked = self._marked(backend, fetched.rows)
            facets = _facets(marked, query.origin)
        else:
            facets_reason = fetched.reason
            if not isinstance(fetched, _TooMany):
                partial.append({"part": "facets", "reason": fetched.reason})

        fields_out: dict[str, Any]
        if query.origin:
            fields_out = self._origin_page(query, marked, fetched)
        else:
            fields_out = self._server_page(backend, _result(page, late), total, partial)
            if fields_out.get("total") is None:
                fields_out["total"] = total
        payload: dict[str, Any] = as_payload(
            DirectoryPage(
                searched=_searched(query),
                source=source,
                partial=partial,
                facets=facets,
                facets_reason=facets_reason,
                **fields_out,
            )
        )
        return payload

    async def count(self, **fields: Any) -> dict[str, Any]:
        """How many rows a search matches: ``{outcome, total, source, reason?}``."""
        target = await self._sync()
        at = self._at
        backend, source, reason = await self._resolve("pv_directory")
        if backend is None:
            return _no_source_count(reason or NO_SOURCE_REASON, target)
        query = SearchQuery(**fields)
        server = dataclasses.replace(query, origin=())
        deadline = self._timeout()
        started = time.monotonic()
        answer: Any
        if query.origin:
            count = asyncio.create_task(self._count(at, backend, server, deadline))
            answer = await self._bounded(
                self._full_set(at, backend, server, count, started, deadline), deadline
            )
            await _cancel([count])
            if isinstance(answer, _FullSet):
                answer = sum(
                    row.origin in query.origin for row in self._marked(backend, answer.rows)
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
        record = dataclasses.replace(
            answer,
            row=self._hide_row(backend, dataclasses.replace(answer.row, origin=origin), hidden),
            properties={k: v for k, v in answer.properties.items() if k not in hidden},
        )
        inactive = record.row.status == "Inactive"
        return {
            **payload,
            "outcome": "inactive" if inactive else "found",
            "record": as_payload(record),
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
        table = pv_directory_targets(self._config, ariel_configured=False)
        asked = [kind for kind in _KINDS if kind in wanted and any(table[kind].values())]
        answers = await asyncio.gather(*(self._probe(kind, target) for kind in asked))
        return dict(zip(asked, answers, strict=True))

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
        self._built.clear()
        self._sets.clear()
        if snapshot is not None:
            await snapshot.aclose()

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
        spec: Any = pv_directory_targets(self._config, ariel_configured=False)[kind][target]
        if spec is None:
            return None, None, NO_SOURCE_REASON
        if kind == "alarm_state" and spec.settings.get("replay_file"):
            spec = self._with_replay_path(spec)
        create = getattr(ConnectorFactory, f"create_{kind}_connector")
        try:
            backend = await create(spec, project_dir=self._project_dir)
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

    def _origin_page(self, query: Any, marked: list[Any] | None, fetched: Any) -> dict[str, Any]:
        """The page of an origin-filtered search, cut from the whole fetched set."""
        if marked is None:
            return {"outcome": Outcome.UNAVAILABLE, "request": "", "reason": fetched.reason}
        kept = [row for row in marked if row.origin in query.origin]
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

    def _marked(self, backend: Any, rows: Iterable[Any]) -> list[Any]:
        """*rows* with their origin marks and without hidden properties."""
        rows = list(rows)
        marks = mark([row.name for row in rows], dict(self._config))
        hidden = self._hidden()
        return [
            self._hide_row(backend, dataclasses.replace(row, origin=marks[row.name]), hidden)
            for row in rows
        ]

    def _hidden(self) -> frozenset[str]:
        names = self._display.get("hidden_properties")
        if not isinstance(names, list):
            return frozenset()
        return frozenset(name for name in names if isinstance(name, str))

    def _hide_row(self, backend: Any, row: Any, hidden: frozenset[str]) -> Any:
        """*row* with the fields filled from a hidden property cleared.

        The description's property is the one *backend* was built with.
        """
        if not hidden:
            return row
        description = (
            backend.spec.settings.get("description_property") or DEFAULT_DESCRIPTION_PROPERTY
        )
        by_property = {**ROW_FIELD_PROPERTIES, "description": description}
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


def _facets(marked: list[Any], origins: tuple[str, ...]) -> dict[str, list[Mapping[str, Any]]]:
    """Value counts over the fetched set; IOC and record type over the origin-filtered part."""
    kept = [row for row in marked if not origins or row.origin in origins]
    return {
        name: _tally(getattr(row, name) for row in (marked if name == "origin" else kept))
        for name in _FACETS
    }


def _tally(values: Iterable[Any]) -> list[Mapping[str, Any]]:
    counts: dict[str, int] = {}
    for value in values:
        if value is not None:
            counts[value] = counts.get(value, 0) + 1
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [{"value": value, "count": count} for value, count in ordered]


def _searched(query: Any) -> str:
    """The search in plain words."""
    parts = []
    if query.words.strip():
        parts.append(f"name or description matches {' '.join(query.words.split())}")
    if query.name:
        parts.append(f"name matches {query.name}")
    if query.status:
        parts.append(f"status {query.status}")
    if query.ioc:
        parts.append(f"IOC {' or '.join(query.ioc)}")
    if query.record_type:
        parts.append(f"record type {' or '.join(query.record_type)}")
    parts.extend(f"{key} {value}" for key, value in query.properties.items())
    if query.origin:
        parts.append(f"origin {' or '.join(query.origin)}")
    return " · ".join(parts) or "every PV"


def _seconds(value: Any, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return default
    return float(value)


def _count_setting(value: Any, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return default
    return int(value)
