"""The fleet's memory of what it has seen, per target and read path (D9).

A refresher keeps one :class:`FleetMemory` for the state directory it is given.
For each target and read path it holds every IOC that has answered (when it last
did, its host, its class, the liveness and health PV addresses last used, and its
last counter reading with the time of that reading), the time of the last cycle,
and the last good directory listing and lookup with their time. That is what a
restarted refresher needs to carry on where the previous one stopped, and what
lets the view keep reading an IOC the directory has stopped listing.

The memory is one JSON file, :data:`MEMORY_FILE`, replaced whole by
:meth:`FleetMemory.save`. :meth:`FleetMemory.load` never fails on the file: a
missing, unreadable or corrupt file gives an empty memory, and the reason is
logged and kept in :attr:`FleetMemory.load_problem` for the caller to report.

Times are seconds since the epoch. The listing and lookup are stored as the
caller hands them and must be JSON values.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from osprey_connectors.safety_file import atomic_replace

__all__ = ["MEMORY_FILE", "FleetMemory", "GoodListing", "RememberedServer"]

logger = logging.getLogger(__name__)

#: The memory's file name inside the state directory.
MEMORY_FILE = "fleet-memory.json"

#: The file format; a file of any other version is treated as corrupt.
_VERSION = 1

_FILE_MODE = 0o644


@dataclass(frozen=True, kw_only=True)
class RememberedServer:
    """One IOC as it was when it last answered."""

    name: str
    last_answered: float
    host: str | None
    server_class: str
    liveness_channel: str | None
    health_channels: tuple[str, ...]
    counter: float | None
    counter_read_at: float | None


@dataclass(frozen=True, kw_only=True)
class GoodListing:
    """The last directory listing and lookup that succeeded, and when."""

    listing: Any
    lookup: Any
    at: float


@dataclass
class _PathMemory:
    """What is remembered for one target and read path."""

    servers: dict[str, RememberedServer] = field(default_factory=dict)
    last_cycle: float | None = None
    listing: GoodListing | None = None


class _Corrupt(ValueError):
    """The memory file does not have the expected shape."""


class FleetMemory:
    """The persisted state of one state directory; build it with :meth:`load`."""

    def __init__(self, directory: Path, *, retention: float) -> None:
        if not retention > 0:
            raise ValueError(f"retention must be a positive number of seconds, got {retention!r}")
        self.directory = Path(directory)
        self.retention = float(retention)
        #: Why the memory started empty, or ``None`` when the file loaded.
        self.load_problem: str | None = None
        self._paths: dict[tuple[str, str], _PathMemory] = {}

    @property
    def path(self) -> Path:
        """The memory file."""
        return self.directory / MEMORY_FILE

    # --- Loading and saving ---------------------------------------------------

    @classmethod
    def load(cls, directory: Path, *, retention: float) -> FleetMemory:
        """The memory stored in ``directory``; empty, with a reported reason, when it cannot be read."""
        memory = cls(directory, retention=retention)
        try:
            raw = memory.path.read_bytes()
        except FileNotFoundError:
            memory._start_empty(f"no memory file at {memory.path}; starting empty")
            return memory
        except OSError as exc:
            memory._start_empty(
                f"memory file {memory.path} could not be read ({exc}); starting empty"
            )
            return memory
        try:
            memory._paths = _decode(raw)
        except (_Corrupt, ValueError, TypeError, KeyError) as exc:
            memory._start_empty(f"memory file {memory.path} is corrupt ({exc}); starting empty")
        return memory

    def _start_empty(self, problem: str) -> None:
        self._paths = {}
        self.load_problem = problem
        logger.warning("%s", problem)

    def save(self) -> None:
        """Replace the memory file with the current memory, whole or not at all.

        The directory is created when missing. A failure to write raises
        :class:`OSError` and leaves the previous file as it was.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        data = json.dumps(_encode(self._paths), indent=2, allow_nan=False).encode("utf-8")
        atomic_replace(self.path, data, mode=_FILE_MODE, sync="required")

    # --- Recording ------------------------------------------------------------

    def record_answer(self, target: str, read_path: str, ioc: RememberedServer) -> None:
        """Remember ``server`` as it was when it answered, replacing what was known of it."""
        self._at(target, read_path).servers[ioc.name] = ioc

    def record_cycle(self, target: str, read_path: str, at: float) -> None:
        """Remember that a cycle ran at ``at``."""
        self._at(target, read_path).last_cycle = float(at)

    def record_good_listing(
        self, target: str, read_path: str, *, listing: Any, lookup: Any, at: float
    ) -> None:
        """Remember a listing and lookup that succeeded at ``at``; both must be JSON values."""
        json.dumps([listing, lookup], allow_nan=False)
        self._at(target, read_path).listing = GoodListing(
            listing=listing, lookup=lookup, at=float(at)
        )

    # --- Questions ------------------------------------------------------------

    def seen_before(self, target: str, read_path: str, name: str) -> bool:
        """Whether the IOC ``name`` has answered on this target and read path."""
        return name in self._get(target, read_path).servers

    def server(self, target: str, read_path: str, name: str) -> RememberedServer | None:
        """What is remembered of the IOC ``name``, or ``None`` when it never answered."""
        return self._get(target, read_path).servers.get(name)

    def previous_reading(
        self, target: str, read_path: str, name: str
    ) -> tuple[float, float] | None:
        """The IOC's last counter reading and its read time, or ``None`` without one."""
        remembered = self.server(target, read_path, name)
        if remembered is None or remembered.counter is None or remembered.counter_read_at is None:
            return None
        return remembered.counter, remembered.counter_read_at

    def last_cycle_time(self, target: str, read_path: str) -> float | None:
        """When the last cycle ran, or ``None`` before the first."""
        return self._get(target, read_path).last_cycle

    def last_good_listing(self, target: str, read_path: str) -> GoodListing | None:
        """The last listing and lookup that succeeded, or ``None`` before the first."""
        return self._get(target, read_path).listing

    def no_longer_listed(
        self, target: str, read_path: str, *, listed: Iterable[str]
    ) -> list[RememberedServer]:
        """Remembered IOCs that are not in ``listed``, by name, with the addresses to read them by."""
        names = set(listed)
        iocs = self._get(target, read_path).servers
        return [iocs[name] for name in sorted(iocs) if name not in names]

    def prune(self, target: str, read_path: str, *, listed: Iterable[str], now: float) -> list[str]:
        """Forget unlisted IOCs quiet for longer than the retention period; return their names.

        An IOC that ``listed`` names is kept however long it has been quiet.
        """
        iocs = self._get(target, read_path).servers
        dropped = [
            ioc.name
            for ioc in self.no_longer_listed(target, read_path, listed=listed)
            if now - ioc.last_answered > self.retention
        ]
        for name in dropped:
            del iocs[name]
        return dropped

    # --- Internals ------------------------------------------------------------

    def _get(self, target: str, read_path: str) -> _PathMemory:
        return self._paths.get((target, read_path)) or _PathMemory()

    def _at(self, target: str, read_path: str) -> _PathMemory:
        return self._paths.setdefault((target, read_path), _PathMemory())


# --- File format ---------------------------------------------------------------


def _encode(paths: dict[tuple[str, str], _PathMemory]) -> dict[str, Any]:
    return {
        "version": _VERSION,
        "paths": [
            {
                "target": target,
                "read_path": read_path,
                "last_cycle": memory.last_cycle,
                "listing": None
                if memory.listing is None
                else {
                    "listing": memory.listing.listing,
                    "lookup": memory.listing.lookup,
                    "at": memory.listing.at,
                },
                "servers": {
                    name: {
                        "last_answered": ioc.last_answered,
                        "host": ioc.host,
                        "server_class": ioc.server_class,
                        "liveness_channel": ioc.liveness_channel,
                        "health_channels": list(ioc.health_channels),
                        "counter": ioc.counter,
                        "counter_read_at": ioc.counter_read_at,
                    }
                    for name, ioc in sorted(memory.servers.items())
                },
            }
            for (target, read_path), memory in sorted(paths.items())
        ],
    }


def _decode(raw: bytes) -> dict[tuple[str, str], _PathMemory]:
    document = json.loads(raw.decode("utf-8"))
    if not isinstance(document, dict):
        raise _Corrupt("the file is not a JSON object")
    if document.get("version") != _VERSION:
        raise _Corrupt(f"unknown version {document.get('version')!r}")
    entries = document["paths"]
    if not isinstance(entries, list):
        raise _Corrupt("'paths' is not a list")
    paths: dict[tuple[str, str], _PathMemory] = {}
    for entry in entries:
        key = (_text(entry["target"]), _text(entry["read_path"]))
        if key in paths:
            raise _Corrupt(f"target {key[0]!r} read path {key[1]!r} appears twice")
        listing = entry["listing"]
        iocs = entry["servers"]
        if not isinstance(iocs, dict):
            raise _Corrupt("'servers' is not an object")
        paths[key] = _PathMemory(
            servers={
                _text(name): _decode_server(_text(name), fields) for name, fields in iocs.items()
            },
            last_cycle=_optional_time(entry["last_cycle"]),
            listing=None
            if listing is None
            else GoodListing(
                listing=listing["listing"], lookup=listing["lookup"], at=_time(listing["at"])
            ),
        )
    return paths


def _decode_server(name: str, fields: Any) -> RememberedServer:
    if not isinstance(fields, dict):
        raise _Corrupt(f"IOC {name!r} is not an object")
    health = fields["health_channels"]
    if not isinstance(health, list):
        raise _Corrupt(f"IOC {name!r} health PVs are not a list")
    return RememberedServer(
        name=name,
        last_answered=_time(fields["last_answered"]),
        host=_optional_text(fields["host"]),
        server_class=_text(fields["server_class"]),
        liveness_channel=_optional_text(fields["liveness_channel"]),
        health_channels=tuple(_text(pv) for pv in health),
        counter=_optional_number(fields["counter"]),
        counter_read_at=_optional_time(fields["counter_read_at"]),
    )


def _text(value: Any) -> str:
    if not isinstance(value, str):
        raise _Corrupt(f"expected text, found {value!r}")
    return value


def _optional_text(value: Any) -> str | None:
    return None if value is None else _text(value)


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise _Corrupt(f"expected a number, found {value!r}")
    return float(value)


def _optional_number(value: Any) -> float | None:
    return None if value is None else _number(value)


_time = _number
_optional_time = _optional_number
