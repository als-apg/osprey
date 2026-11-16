"""What a channel directory answers, and in what shape.

The shared answers (``Outcome``, ``Unavailable``, ``Source``, ``BackendSpec``)
and the rules every backend follows are in :mod:`osprey_connectors.service_answers`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator
from typing_extensions import override

from osprey_connectors.model import FrozenModel, FrozenTuple, ReadOnlyMapping, empty_mapping
from osprey_connectors.service_answers import (
    Outcome,
    Source,
    Unavailable,
    _Backend,
    _require_reason,
)

#: Whether the directory lists a channel as announced now or no longer.
DirectoryStatus = Literal["Active", "Inactive"]


class DirectoryRow(FrozenModel):
    """One channel, as the directory lists it.

    ``server`` is the process serving the channel and ``channel_type`` its type
    there: EPICS RecSync's ``iocName`` and ``recordType``, a Tango database's
    device-server instance and device class, a DOOCS ENS server name and device
    type. ``last_announced`` is RecSync's ``time`` or Tango's last export time.
    """

    address: str
    status: DirectoryStatus | None = None
    description: str | None = None
    channel_type: str | None = None
    server: str | None = None
    host: str | None = None
    last_announced: str | None = None


class DirectoryRecord(FrozenModel):
    """One channel in full. Every property is optional and a string, as measured."""

    row: DirectoryRow
    properties: ReadOnlyMapping[str, str] = Field(default_factory=empty_mapping)
    tags: FrozenTuple[str] = ()


class DirectoryQuery(FrozenModel):
    """One search, built only by the service; callers pass these fields as keywords.

    A query keys the service's cache, so it hashes: the repeatable filters become
    tuples, from one string, any iterable of them or ``None``, and ``properties``
    stays out of the hash. ``address_pattern`` is a glob over channel addresses.
    """

    words: str = ""
    address_pattern: str | None = None
    properties: ReadOnlyMapping[str, str] = Field(default_factory=empty_mapping)
    status: DirectoryStatus | None = None
    server: tuple[str, ...] = ()
    channel_type: tuple[str, ...] = ()
    page: int = 0
    page_size: int = 50

    @field_validator("server", "channel_type", mode="before")
    @classmethod
    def _filter_values(cls, values: str | Iterable[str] | None) -> tuple[str, ...]:
        if values is None:
            return ()
        return (values,) if isinstance(values, str) else tuple(values)

    @override
    def __hash__(self) -> int:
        return hash(
            (
                self.words,
                self.address_pattern,
                self.status,
                self.server,
                self.channel_type,
                self.page,
                self.page_size,
            )
        )


class DirectoryPage(FrozenModel):
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
    rows: FrozenTuple[DirectoryRow] = ()
    truncated: bool = False
    partial: FrozenTuple[ReadOnlyMapping[str, str]] = ()
    facets: ReadOnlyMapping[str, FrozenTuple[ReadOnlyMapping[str, object]]] | None = None
    facets_reason: str | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def _unavailable_says_why(self) -> Self:
        _require_reason(self.outcome, self.reason)
        return self


class ChannelDirectory(_Backend, ABC):
    """Search the control system's own directory of channels."""

    @property
    def field_properties(self) -> Mapping[str, str]:
        """The directory property that fills each row field.

        A field not listed comes from no single property.
        """
        return MappingProxyType({})

    @abstractmethod
    async def search(self, query: DirectoryQuery, *, timeout_s: float) -> DirectoryPage:
        """One page of the rows *query* matches; no rows is ``NO_MATCH``, not ``NOT_LISTED``."""

    @abstractmethod
    async def count(self, query: DirectoryQuery, *, timeout_s: float) -> int | Unavailable:
        """How many rows *query* matches over the whole directory."""

    @abstractmethod
    async def describe(
        self, address: str, *, timeout_s: float
    ) -> DirectoryRecord | Literal[Outcome.NOT_LISTED] | Unavailable:
        """The record of the channel at exactly *address*, or ``NOT_LISTED``."""

    @abstractmethod
    async def property_names(self, *, timeout_s: float) -> list[str] | Unavailable:
        """The property names the directory holds."""

    @abstractmethod
    async def property_values(
        self, property: str, *, prefix: str | None = None, timeout_s: float
    ) -> list[str] | Unavailable:
        """The values of *property*, those starting with *prefix* if given."""

    @abstractmethod
    async def probe(self, *, timeout_s: float) -> str | Unavailable:
        """Whether the service answers: the evidence in plain words, or why not."""
