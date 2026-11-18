"""The virtual accelerator's directory of PVs, answered from the build's simulator view.

The view is read once, on the first call that answers, and held for the life of
the backend; a view it could not read is tried again on the next call. Each row
is a channel of the view, or a status or IOC health address the simulator serves
beside the channels. The IOC of a row is ``sim-<model>``, the served physics
model that owns the address, else ``sim-texture``, as the composite routes it.

A search means what it means on ChannelFinder. Words match descriptions first and
names only when no description matched, in any case; all of them must match. A
word holding ``*`` or ``?`` and each term of ``DirectoryQuery.address_pattern`` match names as
globs, wrapped as ``*term*`` unless the term starts or ends with ``*``; a leading
``=`` is the exact name. Properties compare whole values.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping

from typing_extensions import override

from osprey_connectors.directory._contract import (
    ChannelDirectory,
    DirectoryDetail,
    DirectoryEntry,
    DirectoryPage,
    DirectoryQuery,
)
from osprey_connectors.service_answers import NotListed, Unavailable
from osprey_connectors.simulation.records import ChannelRecord
from osprey_connectors.simulation.view import (
    ENGINEER,
    HEARTBEAT,
    SERVED_SIGNAL_PROPERTY,
    SETPOINT,
    STATUS_SIGNAL,
    UPTIME,
    NoSimulatorViewError,
    SimulatorView,
)

_HOST = "virtual-accelerator"
_REQUEST_PREFIX = "simulator data"
#: The IOC of every channel no served physics model owns.
_TEXTURE_IOC = "sim-texture"

_OUTPUT_RECORD_TYPES = {
    "float": "ao",
    "int": "longout",
    "bool": "bo",
    "enum": "mbbo",
    "string": "stringout",
    "waveform": "waveform",
}
_INPUT_RECORD_TYPES = {
    "float": "ai",
    "int": "longin",
    "bool": "bi",
    "enum": "mbbi",
    "string": "stringin",
    "waveform": "waveform",
}

#: What each simulator-served signal is: its record type and its description.
_SERVED_ADDRESSES = {
    STATUS_SIGNAL: ("stringin", "simulation status of IOC {server}"),
    HEARTBEAT: ("longin", "heartbeat counter of IOC {server}"),
    UPTIME: ("ai", "seconds since IOC {server} started"),
    ENGINEER: ("stringin", "responsible engineer of IOC {server}"),
}


def _first_sentence(text: str | None) -> str | None:
    head, sep, _rest = " ".join((text or "").split()).partition(". ")
    return (f"{head}." if sep else head) or None


def _channel_record(channel: ChannelRecord, server: str) -> DirectoryDetail:
    types = _OUTPUT_RECORD_TYPES if channel.role == SETPOINT else _INPUT_RECORD_TYPES
    properties: dict[str, str] = {"role": channel.role, "valueType": channel.value_type}
    if channel.unit is not None:
        properties["unit"] = str(channel.unit)
    return DirectoryDetail(
        entry=DirectoryEntry(
            address=channel.address,
            status="Active",
            description=_first_sentence(channel.description),
            channel_type=types.get(channel.value_type),
            server=server,
            host=_HOST,
        ),
        properties=properties,
    )


def _served_record(address: str, server: str, signal: str) -> DirectoryDetail:
    channel_type, description = _SERVED_ADDRESSES[signal]
    return DirectoryDetail(
        entry=DirectoryEntry(
            address=address,
            status="Active",
            description=description.format(server=server),
            channel_type=channel_type,
            server=server,
            host=_HOST,
        ),
        properties={SERVED_SIGNAL_PROPERTY: signal},
    )


def _read_records(view: SimulatorView) -> dict[str, DirectoryDetail]:
    """Every PV of ``view``, sorted by name."""
    physics = {model.name for model in view.physics_models()}
    records = {}
    for address in view.channels():
        channel = view.channel(address)
        server = f"sim-{channel.owner}" if channel.owner in physics else _TEXTURE_IOC
        records[address] = _channel_record(channel, server)
    for model, address in view.status_addresses().items():
        records[address] = _served_record(address, f"sim-{model}", STATUS_SIGNAL)
    for model, signals in view.ioc_health_by_model().items():
        for signal, address in signals.items():
            records[address] = _served_record(address, f"sim-{model}", signal)
    return dict(sorted(records.items()))


def _is_name_term(word: str) -> bool:
    return word.startswith("=") or "*" in word or "?" in word


def _name_matcher(term: str) -> Callable[[str], object]:
    if term.startswith("="):
        return term[1:].__eq__
    glob = term if term.startswith("*") or term.endswith("*") else f"*{term}*"
    pattern = "".join(
        ".*" if char == "*" else "." if char == "?" else re.escape(char) for char in glob
    )
    return re.compile(pattern, re.IGNORECASE).fullmatch


def _passes_filters(record: DirectoryDetail, query: DirectoryQuery) -> bool:
    row = record.entry
    return (
        (query.status is None or row.status == query.status)
        and (not query.server or row.server in query.server)
        and (not query.channel_type or row.channel_type in query.channel_type)
        and all(record.properties.get(key) == value for key, value in query.properties.items())
    )


def _matching(records: Iterable[DirectoryDetail], query: DirectoryQuery) -> list[DirectoryDetail]:
    """The records ``query`` matches, in the order given."""
    words = query.words.split()
    matchers = [
        _name_matcher(term)
        for term in (
            *(word for word in words if _is_name_term(word)),
            *(query.address_pattern or "").split(),
        )
    ]
    plain = [word.casefold() for word in words if not _is_name_term(word)]
    kept = [
        record
        for record in records
        if _passes_filters(record, query) and all(match(record.entry.address) for match in matchers)
    ]
    if not plain:
        return kept
    in_description = [
        record
        for record in kept
        if all(word in (record.entry.description or "").casefold() for word in plain)
    ]
    return in_description or [
        record for record in kept if all(word in record.entry.address.casefold() for word in plain)
    ]


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


class SimulatorDirectory(ChannelDirectory):
    """Every PV the virtual accelerator serves, from the simulator view under ``project_dir``."""

    _records_read: Mapping[str, DirectoryDetail] | None = None

    def _records(self) -> Mapping[str, DirectoryDetail] | Unavailable:
        if self._records_read is not None:
            return self._records_read
        if self.project_dir is None:
            return Unavailable(
                cause="misconfigured",
                reason="no project directory to find the simulator view under",
            )
        try:
            self._records_read = _read_records(SimulatorView.of_project(self.project_dir))
        except NoSimulatorViewError as absent:
            return Unavailable(cause="misconfigured", reason=str(absent))
        except (LookupError, OSError, TypeError, ValueError, AttributeError) as failure:
            return Unavailable(
                cause="failed", reason=f"the simulator view cannot be read: {failure}"
            )
        return self._records_read

    @override
    async def search(
        self, query: DirectoryQuery, *, timeout_s: float
    ) -> DirectoryPage | Unavailable:
        records = self._records()
        if isinstance(records, Unavailable):
            return records
        searched = _searched(query)
        hits = _matching(records.values(), query)
        size = max(query.page_size, 0)
        start = max(query.page, 0) * size
        rows = [record.entry for record in hits[start : start + size]]
        return DirectoryPage(
            outcome="found" if hits else "no_match",
            searched=searched,
            request=f"{_REQUEST_PREFIX} · {searched}",
            source=self.source,
            total=len(hits),
            rows=tuple(rows),
            truncated=start + len(rows) < len(hits),
        )

    @override
    async def count(self, query: DirectoryQuery, *, timeout_s: float) -> int | Unavailable:
        records = self._records()
        return (
            records if isinstance(records, Unavailable) else len(_matching(records.values(), query))
        )

    @override
    async def describe(
        self,
        address: str,
        *,
        timeout_s: float,
    ) -> DirectoryDetail | NotListed | Unavailable:
        records = self._records()
        if isinstance(records, Unavailable):
            return records
        record = records.get(address)
        return NotListed(address=address) if record is None else record

    @override
    async def property_names(self, *, timeout_s: float) -> list[str] | Unavailable:
        records = self._records()
        if isinstance(records, Unavailable):
            return records
        return sorted({key for record in records.values() for key in record.properties})

    @override
    async def property_values(
        self,
        property: str,
        *,
        prefix: str | None = None,
        timeout_s: float,
    ) -> list[str] | Unavailable:
        records = self._records()
        if isinstance(records, Unavailable):
            return records
        values = {
            record.properties[property]
            for record in records.values()
            if property in record.properties
        }
        return sorted(value for value in values if value.startswith(prefix or ""))

    @override
    async def probe(self, *, timeout_s: float) -> str | Unavailable:
        records = self._records()
        if isinstance(records, Unavailable):
            return records
        return f"simulator view read: {len(records)} PVs"
