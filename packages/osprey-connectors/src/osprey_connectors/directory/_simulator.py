"""The virtual accelerator's directory of channels, answered from the build's simulator view.

The view is read once, on the first call that answers, and held for the life of
the backend; a view it could not read is tried again on the next call. Backends
of one process share the rows of one view: a second backend reads the view the
first one opened and builds no row of its own. Each row
is a channel of the view, or a status or server health address the simulator serves
beside the channels. The server of a row is ``sim-<model>``, the served physics
model that owns the address, else ``sim-texture``, as the composite routes it.

Its servers are those: every served physics model's and the texture's, each
listed even when it serves no channel. The listing reads the view alone, so a server a
scenario stops is still listed: the directory says what exists, not what answers.

A search means what it means on ChannelFinder. Words match descriptions first and
names only when no description matched, in any case; all of them must match. A
word holding ``*`` or ``?`` and each term of ``DirectoryQuery.name`` match names as
globs, wrapped as ``*term*`` unless the term starts or ends with ``*``; a leading
``=`` is the exact name. Properties compare whole values.
"""

from __future__ import annotations

import re
import threading
import weakref
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from types import MappingProxyType
from typing import Literal

from typing_extensions import override

from osprey_connectors.directory._contract import (
    ChannelDirectory,
    DirectoryPage,
    DirectoryQuery,
    DirectoryRecord,
    DirectoryRow,
    ServerRow,
)
from osprey_connectors.service_answers import Outcome, Unavailable
from osprey_connectors.simulation.servers import channel_server, model_server, served_servers
from osprey_connectors.simulation.view import (
    ENGINEER,
    HEARTBEAT,
    SERVED_SIGNAL_PROPERTY,
    SETPOINT,
    STATUS_SIGNAL,
    UPTIME,
    Channel,
    NoSimulatorView,
    SimulatorView,
)

_HOST = "virtual-accelerator"
_REQUEST_PREFIX = "simulator data"

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


def _channel_record(channel: Channel, server: str) -> DirectoryRecord:
    types = _OUTPUT_RECORD_TYPES if channel.role == SETPOINT else _INPUT_RECORD_TYPES
    properties = {"role": channel.role, "valueType": channel.value_type}
    if channel.unit is not None:
        properties["unit"] = str(channel.unit)
    return DirectoryRecord(
        row=DirectoryRow(
            name=channel.address,
            status="Active",
            description=_first_sentence(channel.description),
            channel_type=types.get(channel.value_type),
            server=server,
            host=_HOST,
        ),
        properties=properties,
    )


def _served_record(address: str, server: str, signal: str) -> DirectoryRecord:
    channel_type, description = _SERVED_ADDRESSES[signal]
    return DirectoryRecord(
        row=DirectoryRow(
            name=address,
            status="Active",
            description=description.format(server=server),
            channel_type=channel_type,
            server=server,
            host=_HOST,
        ),
        properties={SERVED_SIGNAL_PROPERTY: signal},
    )


def _read_records(view: SimulatorView) -> dict[str, DirectoryRecord]:
    """Every channel of ``view``, sorted by name."""
    physics = {model.name for model in view.physics_models()}
    records = {}
    for address in view.channels():
        channel = view.channel(address)
        records[address] = _channel_record(channel, channel_server(channel.owner, physics))
    for model, address in view.status_addresses().items():
        records[address] = _served_record(address, model_server(model), STATUS_SIGNAL)
    for model, signals in view.server_health_by_model().items():
        for signal, address in signals.items():
            records[address] = _served_record(address, model_server(model), signal)
    return dict(sorted(records.items()))


#: Each open view's rows and servers, built once per view of the process.
_READ: weakref.WeakKeyDictionary[
    SimulatorView, tuple[Mapping[str, DirectoryRecord], frozenset[str]]
] = weakref.WeakKeyDictionary()
_READ_LOCK = threading.Lock()


def _rows_of(view: SimulatorView) -> tuple[Mapping[str, DirectoryRecord], frozenset[str]]:
    """``view``'s rows by name and its servers, built on the first call for that view."""
    with _READ_LOCK:
        read = _READ.get(view)
        if read is None:
            servers = served_servers(model.name for model in view.physics_models())
            read = _READ[view] = (MappingProxyType(_read_records(view)), servers)
        return read


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


def _passes_filters(record: DirectoryRecord, query: DirectoryQuery) -> bool:
    row = record.row
    return (
        (query.status is None or row.status == query.status)
        and (not query.server or row.server in query.server)
        and (not query.channel_type or row.channel_type in query.channel_type)
        and all(record.properties.get(key) == value for key, value in query.properties.items())
    )


def _matching(records: Iterable[DirectoryRecord], query: DirectoryQuery) -> list[DirectoryRecord]:
    """The records ``query`` matches, in the order given."""
    words = query.words.split()
    matchers = [
        _name_matcher(term)
        for term in (*(word for word in words if _is_name_term(word)), *(query.name or "").split())
    ]
    plain = [word.casefold() for word in words if not _is_name_term(word)]
    kept = [
        record
        for record in records
        if _passes_filters(record, query) and all(match(record.row.name) for match in matchers)
    ]
    if not plain:
        return kept
    in_description = [
        record
        for record in kept
        if all(word in (record.row.description or "").casefold() for word in plain)
    ]
    return in_description or [
        record for record in kept if all(word in record.row.name.casefold() for word in plain)
    ]


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


class SimulatorDirectory(ChannelDirectory):
    """Every channel the virtual accelerator serves, from the simulator view under ``project_dir``."""

    _records_read: Mapping[str, DirectoryRecord] | None = None
    _servers_read: frozenset[str] = frozenset()

    def _records(self) -> Mapping[str, DirectoryRecord] | Unavailable:
        if self._records_read is not None:
            return self._records_read
        if self.project_dir is None:
            return Unavailable("no project directory to find the simulator view under")
        try:
            records, servers = _rows_of(SimulatorView.of_project(self.project_dir))
        except NoSimulatorView as absent:
            return Unavailable(str(absent))
        except (LookupError, OSError, TypeError, ValueError, AttributeError) as failure:
            return Unavailable(f"the simulator view cannot be read: {failure}")
        self._records_read, self._servers_read = records, servers
        return records

    @override
    async def search(self, query: DirectoryQuery, deadline: float) -> DirectoryPage:
        searched = _searched(query)
        request = f"{_REQUEST_PREFIX} · {searched}"
        records = self._records()
        if isinstance(records, Unavailable):
            return DirectoryPage(
                outcome=Outcome.UNAVAILABLE,
                searched=searched,
                request=request,
                source=self.source,
                reason=records.reason,
            )
        hits = _matching(records.values(), query)
        size = max(query.page_size, 0)
        start = max(query.page, 0) * size
        rows = [record.row for record in hits[start : start + size]]
        return DirectoryPage(
            outcome=Outcome.FOUND if hits else Outcome.NO_MATCH,
            searched=searched,
            request=request,
            source=self.source,
            total=len(hits),
            rows=rows,
            truncated=start + len(rows) < len(hits),
        )

    @override
    async def count(self, query: DirectoryQuery, deadline: float) -> int | Unavailable:
        records = self._records()
        return (
            records if isinstance(records, Unavailable) else len(_matching(records.values(), query))
        )

    @override
    async def describe(
        self,
        name: str,
        deadline: float,
    ) -> DirectoryRecord | Literal[Outcome.NOT_LISTED] | Unavailable:
        records = self._records()
        if isinstance(records, Unavailable):
            return records
        record = records.get(name)
        return Outcome.NOT_LISTED if record is None else record

    @override
    async def property_names(self, deadline: float) -> list[str] | Unavailable:
        records = self._records()
        if isinstance(records, Unavailable):
            return records
        return sorted({key for record in records.values() for key in record.properties})

    @override
    async def property_values(
        self,
        name: str,
        prefix: str | None,
        deadline: float,
    ) -> list[str] | Unavailable:
        records = self._records()
        if isinstance(records, Unavailable):
            return records
        values = {
            record.properties[name] for record in records.values() if name in record.properties
        }
        return sorted(value for value in values if value.startswith(prefix or ""))

    @override
    async def list_servers(self, deadline: float) -> list[ServerRow] | Unavailable:
        records = self._records()
        if isinstance(records, Unavailable):
            return records
        counts = Counter(record.row.server for record in records.values())
        inactive = Counter(
            record.row.server for record in records.values() if record.row.status == "Inactive"
        )
        return [
            ServerRow(
                name=name, host=_HOST, channel_count=counts[name], inactive_count=inactive[name]
            )
            for name in sorted(self._servers_read | {name for name in counts if name})
        ]

    @override
    async def probe(self, deadline: float) -> str | Unavailable:
        records = self._records()
        if isinstance(records, Unavailable):
            return records
        return f"simulator view read: {len(records)} PVs"
