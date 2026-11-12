"""What a channel directory answers, and in what shape.

The shared answers (``Outcome``, ``Unavailable``, ``Source``, ``BackendSpec``)
and the rules every backend follows are in :mod:`osprey_connectors.service_answers`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from osprey_connectors.service_answers import (
    Outcome,
    Source,
    Unavailable,
    _Backend,
    _require_reason,
)


@dataclass(frozen=True, kw_only=True)
class DirectoryRow:
    """One channel, as the directory lists it.

    ``server`` is the process serving the channel and ``channel_type`` its type
    there: EPICS RecSync's ``iocName`` and ``recordType``, a Tango database's
    device-server instance and device class, a DOOCS ENS server name and device
    type. ``last_announced`` is RecSync's ``time`` or Tango's last export time.
    """

    name: str
    status: Literal["Active", "Inactive"] | None = None
    description: str | None = None
    channel_type: str | None = None
    server: str | None = None
    host: str | None = None
    last_announced: str | None = None


@dataclass(frozen=True, kw_only=True)
class DirectoryRecord:
    """One PV in full. Every property is optional and a string, as measured."""

    row: DirectoryRow
    properties: Mapping[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True, kw_only=True)
class DirectoryQuery:
    """One search, built only by the service; callers pass these fields as keywords.

    A query keys the service's cache, so it hashes: the repeatable filters become
    tuples and ``properties`` stays out of the hash.
    """

    words: str = ""
    name: str | None = None
    properties: Mapping[str, str] = field(default_factory=dict, hash=False)
    status: str | None = None
    server: tuple[str, ...] = ()
    channel_type: tuple[str, ...] = ()
    page: int = 0
    page_size: int = 50

    def __post_init__(self) -> None:
        for name in ("server", "channel_type"):
            object.__setattr__(self, name, _filter_values(getattr(self, name)))


def _filter_values(values: str | Iterable[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    return (values,) if isinstance(values, str) else tuple(values)


@dataclass(frozen=True, kw_only=True)
class DirectoryPage:
    """One page of a search.

    ``searched`` is the search in plain words and ``request`` the literal request
    sent. ``partial`` lists ``{part, reason}`` for what missed the deadline;
    ``facets`` maps a row field to ``{value, count}`` lists, and is ``None`` with
    ``facets_reason`` when it could not be computed.
    """

    outcome: Outcome
    searched: str
    request: str
    source: Source
    total: int | None = None
    rows: list[DirectoryRow] = field(default_factory=list)
    truncated: bool = False
    partial: list[Mapping[str, str]] = field(default_factory=list)
    facets: Mapping[str, list[Mapping[str, Any]]] | None = None
    facets_reason: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        _require_reason(self.outcome, self.reason)


class ChannelDirectory(_Backend, ABC):
    """Search the control system's own directory of PVs."""

    @property
    def field_properties(self) -> Mapping[str, str]:
        """The directory property that fills each row field.

        A field not listed comes from no single property.
        """
        return MappingProxyType({})

    @abstractmethod
    async def search(self, query: DirectoryQuery, deadline: float) -> DirectoryPage:
        """One page of the rows *query* matches; no rows is ``NO_MATCH``, not ``NOT_LISTED``."""

    @abstractmethod
    async def count(self, query: DirectoryQuery, deadline: float) -> int | Unavailable:
        """How many rows *query* matches over the whole directory."""

    @abstractmethod
    async def describe(
        self, name: str, deadline: float
    ) -> DirectoryRecord | Literal[Outcome.NOT_LISTED] | Unavailable:
        """The record of the PV called exactly *name*, or ``NOT_LISTED``."""

    @abstractmethod
    async def property_names(self, deadline: float) -> list[str] | Unavailable:
        """The property names the directory holds."""

    @abstractmethod
    async def property_values(
        self, name: str, prefix: str | None, deadline: float
    ) -> list[str] | Unavailable:
        """The values of property *name*, those starting with *prefix* if given."""

    @abstractmethod
    async def probe(self, deadline: float) -> str | Unavailable:
        """Whether the service answers: the evidence in plain words, or why not."""
