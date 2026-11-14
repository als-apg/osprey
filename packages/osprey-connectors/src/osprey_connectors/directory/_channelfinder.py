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
from types import MappingProxyType
from typing import Literal, NotRequired, TypedDict

from osprey_connectors._service_http import ReadOnlyTransport, Transport
from osprey_connectors.directory._contract import (
    ChannelDirectory,
    DirectoryPage,
    DirectoryQuery,
    DirectoryRecord,
    DirectoryRow,
)
from osprey_connectors.service_answers import BackendSpec, Outcome, Unavailable

_CHANNELS = "/resources/channels"
_COUNT = "/resources/channels/count"
_PROPERTIES = "/resources/properties"

_MALFORMED = Unavailable("malformed response")

#: The RecSync property that fills each ``DirectoryRow`` field, by field, in field order.
#: ``description`` is not here: its property is the section's ``description_property``.
_ROW_FIELD_PROPERTIES: Mapping[str, str] = MappingProxyType(
    {
        "status": "pvStatus",
        "channel_type": "recordType",
        "server": "iocName",
        "host": "hostName",
        "last_announced": "time",
    }
)

#: The ``pvStatus`` values a row reports; any other value reports no status.
_STATUSES: Mapping[str, Literal["Active", "Inactive"]] = MappingProxyType(
    {"Active": "Active", "Inactive": "Inactive"}
)

#: The description property RecSync records, for a section that names none.
_DEFAULT_DESCRIPTION_PROPERTY = "recordDesc"

#: The properties with a known set of values, for a deployment whose section names none.
_DEFAULT_EXACT_PROPERTIES = (_ROW_FIELD_PROPERTIES["status"], _ROW_FIELD_PROPERTIES["channel_type"])

#: Rows asked for when only an exact name is wanted: a term with a wildcard in it
#: can match more than the one name it is checked against.
_DESCRIBE_SIZE = 100
#: Rows read to collect a property's values, since the service has no route for them.
_VALUES_SIZE = 1000
#: Rows read for a probe's newest announcement, since the service cannot sort by it.
_PROBE_SAMPLE_SIZE = 20

_SEPARATORS = str.maketrans(dict.fromkeys(",|;", "?"))

_Params = list[tuple[str, str | int]]


class _Channel(TypedDict):
    """A channel as the service answers it, once its ``name`` is checked a string."""

    name: str
    properties: NotRequired[object]
    tags: NotRequired[object]


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


def _searched(query: DirectoryQuery) -> str:
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
    return " · ".join(parts) or "every PV"


def _channels(body: object) -> list[_Channel] | Unavailable:
    if isinstance(body, Unavailable):
        return body
    if not isinstance(body, list) or not all(
        isinstance(item, Mapping) and isinstance(item.get("name"), str) for item in body
    ):
        return _MALFORMED
    return body


def _total(body: object) -> int | Unavailable:
    if isinstance(body, Unavailable):
        return body
    return _MALFORMED if isinstance(body, bool) or not isinstance(body, int) or body < 0 else body


def _names_of(entries: object) -> list[str]:
    if not isinstance(entries, list):
        return []
    return [
        name
        for entry in entries
        if isinstance(entry, Mapping) and isinstance(name := entry.get("name"), str)
    ]


def _properties_of(channel: _Channel) -> dict[str, str]:
    entries = channel.get("properties")
    if not isinstance(entries, list):
        return {}
    return {
        name: str(value)
        for entry in entries
        if isinstance(entry, Mapping)
        and isinstance(name := entry.get("name"), str)
        and (value := entry.get("value")) is not None
    }


class ChannelFinderDirectory(ChannelDirectory):
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
        transport: Transport | None = None,
        **seams: object,
    ) -> None:
        super().__init__(spec, project_dir=project_dir, **seams)
        self._transport = transport or ReadOnlyTransport(
            spec.settings, self.ALLOWED_REQUESTS, where="pv_directory.settings"
        )
        description = spec.settings.get("description_property")
        self._description_property = (
            description
            if isinstance(description, str) and description.strip()
            else _DEFAULT_DESCRIPTION_PROPERTY
        )
        exact = spec.settings.get("exact_properties")
        self._exact_properties = frozenset(
            (key for key in exact if isinstance(key, str))
            if isinstance(exact, list)
            else _DEFAULT_EXACT_PROPERTIES
        )

    @property
    def field_properties(self) -> Mapping[str, str]:
        """RecSync's property of each row field, and the description property built with."""
        return MappingProxyType(
            {**_ROW_FIELD_PROPERTIES, "description": self._description_property}
        )

    def _filters(self, query: DirectoryQuery) -> _Params:
        params: _Params = []
        for key, value in query.properties.items():
            text = str(value)
            params.append(
                (key, _one_value(text.removeprefix("=")))
                if key in self._exact_properties
                else (key, _pattern(text))
            )
        if query.status:
            params.append((_ROW_FIELD_PROPERTIES["status"], _one_value(query.status)))
        if query.server:
            params.append((_ROW_FIELD_PROPERTIES["server"], _any_of(query.server)))
        if query.channel_type:
            params.append((_ROW_FIELD_PROPERTIES["channel_type"], _any_of(query.channel_type)))
        return params

    def _searches(self, query: DirectoryQuery) -> list[_Params]:
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

    def _record(self, channel: _Channel) -> DirectoryRecord:
        properties = _properties_of(channel)
        return DirectoryRecord(
            row=DirectoryRow(
                name=channel["name"],
                status=_STATUSES.get(properties.pop(_ROW_FIELD_PROPERTIES["status"], "")),
                description=properties.pop(self._description_property, None),
                channel_type=properties.pop(_ROW_FIELD_PROPERTIES["channel_type"], None),
                server=properties.pop(_ROW_FIELD_PROPERTIES["server"], None),
                host=properties.pop(_ROW_FIELD_PROPERTIES["host"], None),
                last_announced=properties.pop(_ROW_FIELD_PROPERTIES["last_announced"], None),
            ),
            properties=properties,
            tags=_names_of(channel.get("tags")),
        )

    async def _get(self, path: str, params: _Params | None, deadline: float) -> object:
        return await self._transport.request("GET", path, params=params, deadline=deadline)

    async def search(self, query: DirectoryQuery, deadline: float) -> DirectoryPage:
        started = time.monotonic()
        searched = _searched(query)
        size = max(query.page_size, 0)
        start = max(query.page, 0) * size
        searches = self._searches(query)
        sent: list[str] = []

        def answer(
            outcome: Outcome,
            *,
            total: int | None = None,
            rows: list[DirectoryRow] | None = None,
            truncated: bool = False,
            partial: list[Mapping[str, str]] | None = None,
            reason: str | None = None,
        ) -> DirectoryPage:
            return DirectoryPage(
                outcome=outcome,
                searched=searched,
                request=" · ".join(sent),
                source=self.source,
                total=total,
                rows=rows or [],
                truncated=truncated,
                partial=partial or [],
                reason=reason,
            )

        while True:
            narrowing = searches.pop(0)
            left = deadline - (time.monotonic() - started)
            params: _Params = [*narrowing, ("~size", size), ("~from", start)]
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

    async def count(self, query: DirectoryQuery, deadline: float) -> int | Unavailable:
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
        params: _Params = [("~name", _one_value(name)), ("~size", _DESCRIBE_SIZE)]
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
        params: _Params = [(name, wanted), ("~size", _VALUES_SIZE)]
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
            _properties_of(channel).get(_ROW_FIELD_PROPERTIES["last_announced"])
            for channel in channels
        ]
        newest = max((stamp for stamp in times if stamp), default=None)
        if newest is None:
            return f"{answered}; {len(channels)} PVs read, none with an announcement time"
        return f"{answered}; newest announcement of {len(channels)} PVs read: {newest}"
