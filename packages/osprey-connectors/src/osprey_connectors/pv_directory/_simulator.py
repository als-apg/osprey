"""The virtual accelerator's directory of PVs, answered from the build's simulator view.

The view is read once, on the first call that answers, and held for the life of
the backend; a view it could not read is tried again on the next call. Each row
is a channel of the view, or a status or IOC health address the simulator serves
beside the channels. The IOC of a row is ``sim-<model>``, the served physics
model that owns the address, else ``sim-texture``, as the composite routes it.

A search means what it means on ChannelFinder. Words match descriptions first and
names only when no description matched, in any case; all of them must match. A
word holding ``*`` or ``?`` and each term of ``SearchQuery.name`` match names as
globs, wrapped as ``*term*`` unless the term starts or ends with ``*``; a leading
``=`` is the exact name. Properties compare whole values.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from typing import Literal

from osprey_connectors.pv_directory.contract import (
    SERVED_SIGNAL_PROPERTY,
    DirectoryPage,
    DirectoryRecord,
    DirectoryRow,
    Outcome,
    PVDirectory,
    SearchQuery,
    Unavailable,
)
from osprey_connectors.simulation.view import (
    ENGINEER,
    HEARTBEAT,
    SETPOINT,
    STATUS_SIGNAL,
    UPTIME,
    Channel,
    NoSimulatorView,
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
    STATUS_SIGNAL: ("stringin", "simulation status of IOC {ioc}"),
    HEARTBEAT: ("longin", "heartbeat counter of IOC {ioc}"),
    UPTIME: ("ai", "seconds since IOC {ioc} started"),
    ENGINEER: ("stringin", "responsible engineer of IOC {ioc}"),
}


def _first_sentence(text: str | None) -> str | None:
    head, sep, _rest = " ".join((text or "").split()).partition(". ")
    return (f"{head}." if sep else head) or None


def _channel_record(channel: Channel, ioc: str) -> DirectoryRecord:
    types = _OUTPUT_RECORD_TYPES if channel.role == SETPOINT else _INPUT_RECORD_TYPES
    properties = {"role": channel.role, "valueType": channel.value_type}
    if channel.unit is not None:
        properties["unit"] = str(channel.unit)
    return DirectoryRecord(
        row=DirectoryRow(
            name=channel.address,
            status="Active",
            description=_first_sentence(channel.description),
            record_type=types.get(channel.value_type),
            ioc=ioc,
            host=_HOST,
        ),
        properties=properties,
    )


def _served_record(address: str, ioc: str, signal: str) -> DirectoryRecord:
    record_type, description = _SERVED_ADDRESSES[signal]
    return DirectoryRecord(
        row=DirectoryRow(
            name=address,
            status="Active",
            description=description.format(ioc=ioc),
            record_type=record_type,
            ioc=ioc,
            host=_HOST,
        ),
        properties={SERVED_SIGNAL_PROPERTY: signal},
    )


def _read_records(view: SimulatorView) -> dict[str, DirectoryRecord]:
    """Every PV of ``view``, sorted by name."""
    physics = {model.name for model in view.physics_models()}
    records = {}
    for address in view.channels():
        channel = view.channel(address)
        ioc = f"sim-{channel.owner}" if channel.owner in physics else _TEXTURE_IOC
        records[address] = _channel_record(channel, ioc)
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


def _passes_filters(record: DirectoryRecord, query: SearchQuery) -> bool:
    row = record.row
    return (
        (query.status is None or row.status == query.status)
        and (not query.ioc or row.ioc in query.ioc)
        and (not query.record_type or row.record_type in query.record_type)
        and all(record.properties.get(key) == value for key, value in query.properties.items())
    )


def _matching(records: Iterable[DirectoryRecord], query: SearchQuery) -> list[DirectoryRecord]:
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


class SimulatorDirectory(PVDirectory):
    """Every PV the virtual accelerator serves, from the simulator view under ``project_dir``."""

    _records_read: Mapping[str, DirectoryRecord] | None = None

    def _records(self) -> Mapping[str, DirectoryRecord] | Unavailable:
        if self._records_read is not None:
            return self._records_read
        if self.project_dir is None:
            return Unavailable("no project directory to find the simulator view under")
        try:
            self._records_read = _read_records(SimulatorView.of_project(self.project_dir))
        except NoSimulatorView as absent:
            return Unavailable(str(absent))
        except (LookupError, OSError, TypeError, ValueError, AttributeError) as failure:
            return Unavailable(f"the simulator view cannot be read: {failure}")
        return self._records_read

    async def search(self, query: SearchQuery, deadline: float) -> DirectoryPage:  # noqa: ARG002
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

    async def count(self, query: SearchQuery, deadline: float) -> int | Unavailable:  # noqa: ARG002
        records = self._records()
        return (
            records if isinstance(records, Unavailable) else len(_matching(records.values(), query))
        )

    async def describe(
        self,
        name: str,
        deadline: float,  # noqa: ARG002
    ) -> DirectoryRecord | Literal[Outcome.NOT_LISTED] | Unavailable:
        records = self._records()
        if isinstance(records, Unavailable):
            return records
        record = records.get(name)
        return Outcome.NOT_LISTED if record is None else record

    async def property_names(self, deadline: float) -> list[str] | Unavailable:  # noqa: ARG002
        records = self._records()
        if isinstance(records, Unavailable):
            return records
        return sorted({key for record in records.values() for key in record.properties})

    async def property_values(
        self,
        name: str,
        prefix: str | None,
        deadline: float,  # noqa: ARG002
    ) -> list[str] | Unavailable:
        records = self._records()
        if isinstance(records, Unavailable):
            return records
        values = {
            record.properties[name] for record in records.values() if name in record.properties
        }
        return sorted(value for value in values if value.startswith(prefix or ""))

    async def probe(self, deadline: float) -> str | Unavailable:  # noqa: ARG002
        records = self._records()
        if isinstance(records, Unavailable):
            return records
        return f"simulator view read: {len(records)} PVs"
