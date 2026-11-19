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
from typing import Annotated, ClassVar

from pydantic import ConfigDict, Field, OnErrorOmit, TypeAdapter, ValidationError, model_validator
from typing_extensions import override

from osprey_connectors._service_http import (
    MALFORMED,
    HttpServiceSettings,
    ReadOnlyTransport,
    Transport,
)
from osprey_connectors.config_sections import ConnectorSettings
from osprey_connectors.directory._contract import (
    ChannelDirectory,
    DirectoryDetail,
    DirectoryEntry,
    DirectoryPage,
    DirectoryQuery,
    DirectoryStatus,
)
from osprey_connectors.model import FrozenModel, evolve
from osprey_connectors.service_answers import (
    BackendSpec,
    NotListed,
    Unavailable,
    _instant,
    _Recorded,
)

_CHANNELS = "/resources/channels"
_COUNT = "/resources/channels/count"
_PROPERTIES = "/resources/properties"


#: The RecSync property that fills each ``DirectoryEntry`` field, by field, in field order.
#: ``description`` is not here: its property is the section's ``description_property``.
#: ``time`` fills ``announced_at``, or ``recorded_stamp`` when it cannot be placed, so
#: hiding it clears both.
_ROW_FIELD_PROPERTIES: Mapping[str, str] = MappingProxyType(
    {
        "status": "pvStatus",
        "channel_type": "recordType",
        "server": "iocName",
        "host": "hostName",
        "announced_at": "time",
        "recorded_stamp": "time",
    }
)

#: The ``pvStatus`` values a row reports; any other value reports no status.
_STATUSES: Mapping[str, DirectoryStatus] = MappingProxyType(
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


class _Named(FrozenModel):
    """An entry the service lists by its string ``name``: a tag, a property, a property name."""

    model_config = ConfigDict(extra="ignore")

    name: str


class _Property(_Named):
    """A property of a channel; a ``None`` value is no value."""

    value: object = None


#: A list the service answers, an entry that is not one with a string name skipped.
_Entries = list[OnErrorOmit[_Named]]


class _Channel(FrozenModel):
    """A channel as the service answers it.

    ``name`` must be a string, or the whole answer is malformed. ``properties`` and
    ``tags`` are lists of named entries whose other entries are skipped, and either
    is ``None`` when it is not a list: every property is optional.
    """

    model_config = ConfigDict(extra="ignore")

    name: str
    properties: _Recorded[list[OnErrorOmit[_Property]]] = None
    tags: _Recorded[_Entries] = None


_CHANNELS_ANSWER = TypeAdapter(list[_Channel])
_NAMES_ANSWER = TypeAdapter(_Entries)
_COUNT_ANSWER: TypeAdapter[int] = TypeAdapter(Annotated[int, Field(ge=0)])


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
    if query.address_pattern:
        parts.append(f"name matches {query.address_pattern}")
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
    try:
        return _CHANNELS_ANSWER.validate_python(body, strict=True)
    except ValidationError:
        return MALFORMED


def _total(body: object) -> int | Unavailable:
    if isinstance(body, Unavailable):
        return body
    try:
        return _COUNT_ANSWER.validate_python(body, strict=True)
    except ValidationError:
        return MALFORMED


def _names(body: object) -> list[str] | Unavailable:
    if isinstance(body, Unavailable):
        return body
    try:
        return [entry.name for entry in _NAMES_ANSWER.validate_python(body, strict=True)]
    except ValidationError:
        return MALFORMED


def _properties_of(channel: _Channel) -> dict[str, str]:
    return {
        entry.name: str(entry.value)
        for entry in channel.properties or ()
        if entry.value is not None
    }


class ChannelFinderSettings(HttpServiceSettings):
    """ChannelFinder's block, ``pv_directory.settings``, read, with the section's folded keys.

    Attributes:
        description_property: The property a row's description is read from:
            the section's ``description_property`` when it is a non-blank
            string, else ``recordDesc``.
        exact_properties: The properties with a known set of values: the
            string entries of ``display.exact_properties`` when it is a list,
            else ``pvStatus`` and ``recordType``.
    """

    description_property: str = _DEFAULT_DESCRIPTION_PROPERTY
    exact_properties: frozenset[str] = frozenset(_DEFAULT_EXACT_PROPERTIES)

    @model_validator(mode="before")
    @classmethod
    def _folded(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data
        description = data.get("description_property")
        exact = data.get("exact_properties")
        return {
            **data,
            "description_property": (
                description
                if isinstance(description, str) and description.strip()
                else _DEFAULT_DESCRIPTION_PROPERTY
            ),
            "exact_properties": frozenset(
                (key for key in exact if isinstance(key, str))
                if isinstance(exact, list | tuple | frozenset | set)
                else _DEFAULT_EXACT_PROPERTIES
            ),
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

    settings_model: ClassVar[type[ConnectorSettings]] = ChannelFinderSettings
    SETTINGS_KEY: ClassVar[str] = "pv_directory.settings"

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
        settings = self._settings(ChannelFinderSettings)
        self._transport = transport or ReadOnlyTransport(settings.connection, self.ALLOWED_REQUESTS)
        self._description_property = settings.description_property
        self._exact_properties = settings.exact_properties

    @property
    @override
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
        terms = [
            *(word for word in words if _is_name_term(word)),
            *(query.address_pattern or "").split(),
        ]
        narrowing = [*(("~name", _pattern(term)) for term in terms), *self._filters(query)]
        if not plain:
            return [narrowing or [("~name", "*")]]
        joined = "*".join(_one_value(word) for word in plain)
        return [
            [(self._description_property, f"*{joined}*"), *narrowing],
            [*(("~name", _pattern(word)) for word in plain), *narrowing],
        ]

    def _record(self, channel: _Channel) -> DirectoryDetail:
        """*channel* as a record; an announcement time with no zone stays as text."""
        properties = _properties_of(channel)
        announced = properties.pop(_ROW_FIELD_PROPERTIES["announced_at"], None)
        announced_at = _instant(announced)
        return DirectoryDetail(
            entry=DirectoryEntry(
                address=channel.name,
                status=_STATUSES.get(properties.pop(_ROW_FIELD_PROPERTIES["status"], "")),
                description=properties.pop(self._description_property, None),
                channel_type=properties.pop(_ROW_FIELD_PROPERTIES["channel_type"], None),
                server=properties.pop(_ROW_FIELD_PROPERTIES["server"], None),
                host=properties.pop(_ROW_FIELD_PROPERTIES["host"], None),
                announced_at=announced_at,
                recorded_stamp=announced if announced_at is None else None,
            ),
            properties=properties,
            tags=tuple(tag.name for tag in channel.tags or ()),
        )

    async def _get(self, path: str, params: _Params | None, timeout_s: float) -> object:
        return await self._transport.request("GET", path, params=params, timeout_s=timeout_s)

    @override
    async def search(
        self, query: DirectoryQuery, *, timeout_s: float
    ) -> DirectoryPage | Unavailable:
        started = time.monotonic()
        size = max(query.page_size, 0)
        start = max(query.page, 0) * size
        searches = self._searches(query)
        sent: list[str] = []

        while True:
            narrowing = searches.pop(0)
            left = timeout_s - (time.monotonic() - started)
            params: _Params = [*narrowing, ("~size", size), ("~from", start)]
            sent.append(_literal(_CHANNELS, params))
            body, counted = await asyncio.gather(
                self._get(_CHANNELS, params, left), self._get(_COUNT, narrowing, left)
            )
            channels, total = _channels(body), _total(counted)
            known = None if isinstance(total, Unavailable) else total
            if isinstance(channels, Unavailable):
                return channels
            if not channels and searches:
                if known == 0 or (known is None and start == 0):
                    continue
                if isinstance(total, Unavailable):
                    # Past the end of this search, or a page of the next: nothing here says.
                    return evolve(
                        total, reason=f"cannot tell which search this page is of: {total.reason}"
                    )
            return DirectoryPage(
                outcome="found" if channels or known else "no_match",
                searched=_searched(query),
                request=" · ".join(sent),
                source=self.source,
                total=total,
                rows=tuple(self._record(channel).entry for channel in channels),
                truncated=(
                    start + len(channels) < known
                    if known is not None
                    else bool(channels) and len(channels) == size
                ),
            )

    @override
    async def count(self, query: DirectoryQuery, *, timeout_s: float) -> int | Unavailable:
        started = time.monotonic()
        total: int | Unavailable = 0
        for narrowing in self._searches(query):
            left = timeout_s - (time.monotonic() - started)
            total = _total(await self._get(_COUNT, narrowing, left))
            if total != 0:
                break
        return total

    @override
    async def describe(
        self, address: str, *, timeout_s: float
    ) -> DirectoryDetail | NotListed | Unavailable:
        params: _Params = [("~name", _one_value(address)), ("~size", _DESCRIBE_SIZE)]
        channels = _channels(await self._get(_CHANNELS, params, timeout_s))
        if isinstance(channels, Unavailable):
            return channels
        for channel in channels:
            if channel.name == address:
                return self._record(channel)
        return NotListed(address=address)

    @override
    async def property_names(self, *, timeout_s: float) -> list[str] | Unavailable:
        names = _names(await self._get(_PROPERTIES, None, timeout_s))
        return names if isinstance(names, Unavailable) else sorted(names)

    @override
    async def property_values(
        self, property: str, *, prefix: str | None = None, timeout_s: float
    ) -> list[str] | Unavailable:
        wanted = f"{_one_value(prefix)}*" if prefix else "*"
        params: _Params = [(property, wanted), ("~size", _VALUES_SIZE)]
        channels = _channels(await self._get(_CHANNELS, params, timeout_s))
        if isinstance(channels, Unavailable):
            return channels
        values = {_properties_of(channel).get(property) for channel in channels}
        return sorted(
            value for value in values if value is not None and value.startswith(prefix or "")
        )

    @override
    async def probe(self, *, timeout_s: float) -> str | Unavailable:
        names, sample = await asyncio.gather(
            self.property_names(timeout_s=timeout_s),
            self._get(_CHANNELS, [("~name", "*"), ("~size", _PROBE_SAMPLE_SIZE)], timeout_s),
        )
        if isinstance(names, Unavailable):
            return names
        answered = f"properties route answered: {len(names)} property names"
        channels = _channels(sample)
        if isinstance(channels, Unavailable):
            return f"{answered}; no PVs read: {channels.reason}"
        times = [
            _properties_of(channel).get(_ROW_FIELD_PROPERTIES["announced_at"])
            for channel in channels
        ]
        newest = max((stamp for stamp in times if stamp), default=None)
        if newest is None:
            return f"{answered}; {len(channels)} PVs read, none with an announcement time"
        return f"{answered}; newest announcement of {len(channels)} PVs read: {newest}"
