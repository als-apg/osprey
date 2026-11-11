"""A ChannelFinder service (filled by RecSync) as the control system's directory of PVs.

What the service makes of a query is measured, and shapes everything here:

- ``channels`` and ``channels/count`` take one grammar. The count is a bare JSON
  integer and paging is ``~size`` and ``~from``.
- Several ``~name`` parameters are ANDed, as are different property keys.
- A value is split on ``,``, ``|`` and ``;`` and the pieces ORed, and PV names can
  hold a comma. So those characters in a term become the one-character wildcard,
  and an OR is only ever built here, from a typed list.
- Exact or partial is the client's convention: the server matches globs literally.
  A term is wrapped as ``*term*`` unless it starts or ends with ``*``, a leading
  ``=`` means the term exactly, and the properties in ``exact_properties``, whose
  values are a known set, are never wrapped.
- Every property is optional and a string, and the schema differs per IOC.
- The root answers 404 on a deployed service and ``channels/{name}`` is not
  allowed here, so an exact name is a ``~name`` search checked on the answer.

Plain words search the description property first and names only when no
description matched, each of those requests holding all the words. Only ``~name``
is measured to AND when repeated, so the description search is one pattern and
matches the words in the order given. A page belongs to the first search with rows,
and the request it reports is every search that was sent, joined with `` · ``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from osprey_connectors.pv_directory._transport import ReadOnlyTransport
from osprey_connectors.pv_directory.contract import (
    DEFAULT_DESCRIPTION_PROPERTY,
    ROW_FIELD_PROPERTIES,
    BackendSpec,
    DirectoryPage,
    DirectoryRecord,
    DirectoryRow,
    Outcome,
    PVDirectory,
    SearchQuery,
    Unavailable,
)

_CHANNELS = "/resources/channels"
_COUNT = "/resources/channels/count"
_PROPERTIES = "/resources/properties"

_MALFORMED = Unavailable("malformed response")

#: The properties with a known set of values, for a deployment whose section names none.
_DEFAULT_EXACT_PROPERTIES = (ROW_FIELD_PROPERTIES["status"], ROW_FIELD_PROPERTIES["record_type"])

#: Rows asked for when only an exact name is wanted: a term with a wildcard in it
#: can match more than the one name it is checked against.
_DESCRIBE_SIZE = 100
#: Rows read to collect a property's values, since the service has no route for them.
_VALUES_SIZE = 1000
#: Rows read for a probe's newest announcement, since the service cannot sort by it.
_PROBE_SAMPLE_SIZE = 20

_SEPARATORS = str.maketrans(dict.fromkeys(",|;", "?"))

_Params = list[tuple[str, Any]]


def _one_value(text: str) -> str:
    """*text* with nothing the server would split into an OR."""
    return text.translate(_SEPARATORS)


def _is_name_term(word: str) -> bool:
    return word.startswith("=") or "*" in word or "?" in word


def _pattern(term: str) -> str:
    if term.startswith("="):
        return _one_value(term[1:])
    return _one_value(term if term.startswith("*") or term.endswith("*") else f"*{term}*")


def _any_of(values: Sequence[str]) -> str:
    return "|".join(_one_value(value) for value in values)


def _literal(path: str, params: _Params) -> str:
    """The request as a person reads it, wildcards unescaped."""
    query = "&".join(f"{key}={value}" for key, value in params)
    return f"GET {path}?{query}"


def _searched(query: SearchQuery) -> str:
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
    return " · ".join(parts) or "every PV"


def _channels(body: Any) -> list[Mapping[str, Any]] | Unavailable:
    if isinstance(body, Unavailable):
        return body
    if not isinstance(body, list) or not all(
        isinstance(item, Mapping) and isinstance(item.get("name"), str) for item in body
    ):
        return _MALFORMED
    return body


def _total(body: Any) -> int | Unavailable:
    if isinstance(body, Unavailable):
        return body
    return _MALFORMED if isinstance(body, bool) or not isinstance(body, int) or body < 0 else body


def _names_of(entries: Any) -> list[str]:
    if not isinstance(entries, list):
        return []
    return [
        entry["name"]
        for entry in entries
        if isinstance(entry, Mapping) and isinstance(entry.get("name"), str)
    ]


def _properties_of(channel: Mapping[str, Any]) -> dict[str, str]:
    entries = channel.get("properties")
    if not isinstance(entries, list):
        return {}
    return {
        entry["name"]: str(entry["value"])
        for entry in entries
        if isinstance(entry, Mapping)
        and isinstance(entry.get("name"), str)
        and entry.get("value") is not None
    }


class ChannelFinderDirectory(PVDirectory):
    """Search a ChannelFinder service over its read routes.

    Args:
        spec: The backend to build; ``settings`` holds the connection block and the
            folded ``description_property`` and ``exact_properties``.
        project_dir: Unused: the service is the whole answer.
        transport: Anything with :meth:`ReadOnlyTransport.request`; defaults to one
            built from ``spec.settings``.
    """

    ALLOWED_REQUESTS = (
        ("GET", _CHANNELS),
        ("GET", _COUNT),
        ("GET", _PROPERTIES),
        ("GET", "/resources/tags"),
    )

    def __init__(
        self,
        spec: BackendSpec,
        *,
        project_dir: Path | None = None,
        transport: Any = None,
        **seams: Any,
    ) -> None:
        super().__init__(spec, project_dir=project_dir, **seams)
        self._transport = transport or ReadOnlyTransport(
            spec.settings, self.ALLOWED_REQUESTS, where="pv_directory.settings"
        )
        description = spec.settings.get("description_property")
        self._description_property = (
            description
            if isinstance(description, str) and description.strip()
            else DEFAULT_DESCRIPTION_PROPERTY
        )
        exact = spec.settings.get("exact_properties")
        self._exact_properties = frozenset(
            (key for key in exact if isinstance(key, str))
            if isinstance(exact, list)
            else _DEFAULT_EXACT_PROPERTIES
        )

    def _filters(self, query: SearchQuery) -> _Params:
        params: _Params = []
        for key, value in query.properties.items():
            text = str(value)
            params.append(
                (key, _one_value(text.removeprefix("=")))
                if key in self._exact_properties
                else (key, _pattern(text))
            )
        if query.status:
            params.append((ROW_FIELD_PROPERTIES["status"], _one_value(query.status)))
        if query.ioc:
            params.append((ROW_FIELD_PROPERTIES["ioc"], _any_of(query.ioc)))
        if query.record_type:
            params.append((ROW_FIELD_PROPERTIES["record_type"], _any_of(query.record_type)))
        return params

    def _searches(self, query: SearchQuery) -> list[_Params]:
        """The searches to try in turn, each without its paging."""
        words = query.words.split()
        plain = [word for word in words if not _is_name_term(word)]
        terms = [*(word for word in words if _is_name_term(word)), *(query.name or "").split()]
        narrowing = [*(("~name", _pattern(term)) for term in terms), *self._filters(query)]
        if not plain:
            return [narrowing or [("~name", "*")]]
        joined = "*".join(_one_value(word) for word in plain)
        return [
            [(self._description_property, f"*{joined}*"), *narrowing],
            [*(("~name", _pattern(word)) for word in plain), *narrowing],
        ]

    def _record(self, channel: Mapping[str, Any]) -> DirectoryRecord:
        properties = _properties_of(channel)
        status = properties.pop(ROW_FIELD_PROPERTIES["status"], None)
        return DirectoryRecord(
            row=DirectoryRow(
                name=channel["name"],
                status=status if status in ("Active", "Inactive") else None,  # type: ignore[arg-type]
                description=properties.pop(self._description_property, None),
                record_type=properties.pop(ROW_FIELD_PROPERTIES["record_type"], None),
                ioc=properties.pop(ROW_FIELD_PROPERTIES["ioc"], None),
                host=properties.pop(ROW_FIELD_PROPERTIES["host"], None),
                last_announced=properties.pop(ROW_FIELD_PROPERTIES["last_announced"], None),
            ),
            properties=properties,
            tags=_names_of(channel.get("tags")),
        )

    async def _get(self, path: str, params: _Params | None, deadline: float) -> Any:
        return await self._transport.request("GET", path, params=params, deadline=deadline)

    async def search(self, query: SearchQuery, deadline: float) -> DirectoryPage:
        started = time.monotonic()
        searched = _searched(query)
        size = max(query.page_size, 0)
        start = max(query.page, 0) * size
        searches = self._searches(query)
        sent: list[str] = []

        def answer(outcome: Outcome, **fields: Any) -> DirectoryPage:
            return DirectoryPage(
                outcome=outcome,
                searched=searched,
                request=" · ".join(sent),
                source=self.source,
                **fields,
            )

        while True:
            narrowing = searches.pop(0)
            left = deadline - (time.monotonic() - started)
            params = [*narrowing, ("~size", size), ("~from", start)]
            sent.append(_literal(_CHANNELS, params))
            body, counted = await asyncio.gather(
                self._get(_CHANNELS, params, left), self._get(_COUNT, narrowing, left)
            )
            channels, total = _channels(body), _total(counted)
            known = None if isinstance(total, Unavailable) else total
            why_no_count = total.reason if isinstance(total, Unavailable) else ""
            if isinstance(channels, Unavailable):
                return answer(Outcome.UNAVAILABLE, reason=channels.reason, total=known)
            if not channels and searches:
                if known == 0 or (known is None and start == 0):
                    continue
                if known is None:
                    # Past the end of this search, or a page of the next: nothing here says.
                    return answer(
                        Outcome.UNAVAILABLE,
                        reason=f"cannot tell which search this page is of: {why_no_count}",
                    )
            return answer(
                Outcome.FOUND if channels or known else Outcome.NO_MATCH,
                total=known,
                rows=[self._record(channel).row for channel in channels],
                truncated=(
                    start + len(channels) < known
                    if known is not None
                    else bool(channels) and len(channels) == size
                ),
                partial=[{"part": "count", "reason": why_no_count}] if known is None else [],
            )

    async def count(self, query: SearchQuery, deadline: float) -> int | Unavailable:
        started = time.monotonic()
        total: int | Unavailable = 0
        for narrowing in self._searches(query):
            left = deadline - (time.monotonic() - started)
            total = _total(await self._get(_COUNT, narrowing, left))
            if total != 0:
                break
        return total

    async def describe(
        self, name: str, deadline: float
    ) -> DirectoryRecord | Literal[Outcome.NOT_LISTED] | Unavailable:
        params = [("~name", _one_value(name)), ("~size", _DESCRIBE_SIZE)]
        channels = _channels(await self._get(_CHANNELS, params, deadline))
        if isinstance(channels, Unavailable):
            return channels
        for channel in channels:
            if channel["name"] == name:
                return self._record(channel)
        return Outcome.NOT_LISTED

    async def property_names(self, deadline: float) -> list[str] | Unavailable:
        body = await self._get(_PROPERTIES, None, deadline)
        if isinstance(body, Unavailable):
            return body
        return sorted(_names_of(body)) if isinstance(body, list) else _MALFORMED

    async def property_values(
        self, name: str, prefix: str | None, deadline: float
    ) -> list[str] | Unavailable:
        wanted = f"{_one_value(prefix)}*" if prefix else "*"
        params = [(name, wanted), ("~size", _VALUES_SIZE)]
        channels = _channels(await self._get(_CHANNELS, params, deadline))
        if isinstance(channels, Unavailable):
            return channels
        values = {_properties_of(channel).get(name) for channel in channels}
        return sorted(
            value for value in values if value is not None and value.startswith(prefix or "")
        )

    async def probe(self, deadline: float) -> str | Unavailable:
        names, sample = await asyncio.gather(
            self.property_names(deadline),
            self._get(_CHANNELS, [("~name", "*"), ("~size", _PROBE_SAMPLE_SIZE)], deadline),
        )
        if isinstance(names, Unavailable):
            return names
        answered = f"properties route answered: {len(names)} property names"
        channels = _channels(sample)
        if isinstance(channels, Unavailable):
            return f"{answered}; no PVs read: {channels.reason}"
        times = [
            _properties_of(channel).get(ROW_FIELD_PROPERTIES["last_announced"])
            for channel in channels
        ]
        newest = max((stamp for stamp in times if stamp), default=None)
        if newest is None:
            return f"{answered}; {len(channels)} PVs read, none with an announcement time"
        return f"{answered}; newest announcement of {len(channels)} PVs read: {newest}"
