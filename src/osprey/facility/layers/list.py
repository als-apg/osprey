"""The list layer: a CSV of channel addresses imported as facility sources.

The file is UTF-8 CSV. With a header row naming ``address``, each further row
is one channel, and the columns, in any order, are::

    address      the channel's full address; required
    role         setpoint, readback or none; empty means readback
    pair         a setpoint's readback address
    device       the device the channel belongs to
    place        the place the channel belongs to
    unit         free text
    description  free text
    tags         ``;``-separated
    s            the device's entrance in metres along ``model``
    model        the model whose deck ``s`` is measured in

A file whose first row names no ``address`` column has no header and holds one
address per line. Blank lines are skipped and every cell is stripped.

The layer writes ``imported/list/channels.yaml``, one record per row sorted by
address, holding only the fields its row fills; the import step seeds the
knowledge stub, and the layer seeds no authored file of its own.
The list declares the devices it names: for each device a row names, the layer
writes ``{id}`` to ``imported/list/devices.yaml``, plus ``s``, and ``model``
when stated, where a row states the device's ``s``, and no other device field.
``s`` and ``model`` state the position of the row's device, never of the
channel. A device another source declares merges with the record field by
field; one declared nowhere else holds only its id and position. The layer
writes ``devices.yaml`` only when some row names a device. When another layer
or ``records/devices.yaml`` declares devices, one ``import list:`` line
names the devices this list alone declares; ``fixes.yaml`` is not read for it.

A row naming both a device and a place, a role outside the three, an unknown or
repeated column, a row with more cells than the header, a row with no address,
an address on two rows, ``s`` or ``model`` on a row with no device, an ``s``
that is not a finite number, and two rows stating different ``s`` or ``model``
for one device are ``source-invalid``, every one reported, and nothing is
written. A ``model`` for a device no row states ``s`` for is ignored, with one
warning line per device each time the file is read. The layer prints nothing:
it returns its lines, warnings first, and a refusal carries the warnings found
before it.
"""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from osprey.facility.errors import FacilityBuildError, LayerSourceInvalid, quoted_slots
from osprey.facility.sources import load_sources
from osprey_connectors import yaml_loader

__all__ = ["COLUMNS", "LAYER_DIR", "import_list"]

#: The layer's directory, relative to ``data/facility/``.
LAYER_DIR = "imported/list"

#: Every column a header may name, in the order the docs list them.
COLUMNS: tuple[str, ...] = (
    "address",
    "role",
    "pair",
    "device",
    "place",
    "unit",
    "description",
    "tags",
    "s",
    "model",
)

#: The device slots a row may state, in the order a device record holds them.
_POSITION = ("s", "model")

_ROLES = ("setpoint", "readback", "none")


@dataclass
class _Reading:
    """What one pass over the file found."""

    name: str
    channels: dict[str, dict[str, Any]] = field(default_factory=dict)
    rows: dict[str, int] = field(default_factory=dict)
    devices: set[str] = field(default_factory=set)
    positions: dict[str, dict[str, tuple[Any, int]]] = field(default_factory=dict)
    errors: list[FacilityBuildError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def refuse(self, record_kind: str, record_id: str, detail: str, remedy: str) -> None:
        self.errors.append(
            FacilityBuildError(
                "source-invalid",
                record_id,
                [self.name],
                remedy,
                record_kind=record_kind,
                detail=detail,
            )
        )


def import_list(path: Path, facility_dir: Path) -> tuple[list[Path], list[str]]:
    """Write a channel list as the list layer's sources.

    Args:
        path: The CSV file.
        facility_dir: The ``data/facility`` directory.

    Returns:
        Every file written, and the lines to show on stderr in order: the
        warnings, then the devices this list alone declares.

    Raises:
        LayerSourceInvalid: The file has rows the layer refuses; nothing is
            written. Its ``warnings`` hold the warning lines found first.
    """
    reading = _read(path)
    lines = list(reading.warnings)
    if reading.errors:
        raise LayerSourceInvalid(reading.errors, warnings=tuple(lines))
    alone = _declared_alone(facility_dir, reading.devices)
    if alone:
        lines.append(f"import list: declared by this list only: {', '.join(alone)}")
    layer = facility_dir / LAYER_DIR
    layer.mkdir(parents=True, exist_ok=True)
    written = [
        _dump(layer / "channels.yaml", [reading.channels[a] for a in sorted(reading.channels)])
    ]
    devices = layer / "devices.yaml"
    if reading.devices:
        written.append(_dump(devices, _devices(reading)))
    elif devices.exists():
        devices.unlink()
    return written, lines


def _devices(reading: _Reading) -> list[dict[str, Any]]:
    """One record per device a row names, with the position the rows state for it."""
    records = []
    for device in sorted(reading.devices):
        stated = reading.positions.get(device, {})
        records.append(
            {"id": device, **{slot: stated[slot][0] for slot in _POSITION if slot in stated}}
        )
    return records


def _declared_alone(facility_dir: Path, devices: set[str]) -> list[str]:
    """The listed devices no other source declares, when another source declares any.

    A list that is the facility's only source of device records declares all
    of them, and none is named.
    """
    elsewhere = {
        record.id
        for record in load_sources(facility_dir).sources.records
        if record.kind == "device" and record.file != f"{LAYER_DIR}/devices.yaml"
    }
    return sorted(devices - elsewhere) if elsewhere else []


def _read(path: Path) -> _Reading:
    reading = _Reading(path.name)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as error:
        reading.refuse("path", path.name, f"is not UTF-8: {error.reason}", "save it as UTF-8")
        return reading
    try:
        rows = [
            (number, [cell.strip() for cell in row])
            for number, row in enumerate(csv.reader(io.StringIO(text, newline="")), start=1)
            if any(cell.strip() for cell in row)
        ]
    except csv.Error as error:
        reading.refuse("path", path.name, f"does not parse as CSV: {error}", "fix the CSV")
        return reading
    if rows and "address" in rows[0][1]:
        _read_table(reading, rows[0][1], rows[1:])
    else:
        _read_addresses(reading, rows)
    _check_positions(reading)
    return reading


def _check_positions(reading: _Reading) -> None:
    """Drop, with a warning, a ``model`` stated for a device no row states ``s`` for."""
    for device, stated in sorted(reading.positions.items()):
        if "s" in stated:
            continue
        model, number = stated["model"]
        reading.warnings.append(
            f"import list: {reading.name} row {number} states model {model} for device "
            f"{device} and no row states its s; the model is ignored; "
            "fix: fill `s` on a row of the device, or empty `model`"
        )
    reading.positions = {
        device: stated for device, stated in reading.positions.items() if "s" in stated
    }


def _read_addresses(reading: _Reading, rows: Iterable[tuple[int, list[str]]]) -> None:
    for number, cells in rows:
        filled = [cell for cell in cells if cell]
        if len(cells) > 1:
            reading.refuse(
                "path",
                reading.name,
                f"row {number} holds {len(cells)} cells and the file has no header",
                "add a header row naming `address`, or write one address per line",
            )
            continue
        _add(reading, number, {"id": filled[0]})


def _read_table(reading: _Reading, header: list[str], rows: list[tuple[int, list[str]]]) -> None:
    unknown = [name for name in header if name not in COLUMNS]
    for name in unknown:
        reading.refuse(
            "path",
            reading.name,
            f"unknown column `{name}`",
            f"remove the column; the columns are {', '.join(COLUMNS)}",
        )
    repeated = sorted({name for name in header if header.count(name) > 1})
    for name in repeated:
        reading.refuse("path", reading.name, f"names column `{name}` twice", "keep one")
    if unknown or repeated:
        return
    for number, cells in rows:
        if len(cells) > len(header):
            reading.refuse(
                "path",
                reading.name,
                f"row {number} holds {len(cells)} cells and the header names {len(header)}",
                "fill one cell per header column",
            )
            continue
        row = dict(zip(header, cells, strict=False))
        address = row.get("address", "")
        if not address:
            reading.refuse(
                "path",
                reading.name,
                f"row {number} has no address",
                "fill `address` or remove the row",
            )
            continue
        record = _channel(reading, number, address, row)
        if record is not None:
            _add(reading, number, record)


def _channel(
    reading: _Reading, number: int, address: str, row: dict[str, str]
) -> dict[str, Any] | None:
    """The channel record one row states, or ``None`` after refusing it."""
    record: dict[str, Any] = {"id": address}
    ok = True
    role = row.get("role", "")
    if role and role not in _ROLES:
        reading.refuse(
            "channel",
            address,
            f"{reading.name} row {number} states role {role}",
            "write setpoint, readback or none, or leave it empty",
        )
        ok = False
    device, place = row.get("device", ""), row.get("place", "")
    if device and place:
        reading.refuse(
            "channel",
            address,
            f"{reading.name} row {number} names both a device and a place",
            f"keep one of `device` and `place` in {reading.name}",
        )
        ok = False
    position = _position(reading, number, address, row)
    if position is None:
        ok = False
    elif position and not device:
        slots = [slot for slot in _POSITION if slot in position]
        reading.refuse(
            "channel",
            address,
            f"{reading.name} row {number} states {quoted_slots(slots)} with no device; "
            "a position belongs to a device",
            "fill `device`, or empty " + " and ".join(f"`{slot}`" for slot in slots),
        )
        ok = False
    if not ok:
        return None
    if device:
        reading.devices.add(device)
    if position:
        _state_position(reading, number, device, position)
    if role:
        record["role"] = role
    if row.get("pair"):
        record["pair"] = row["pair"]
    if device:
        record["on"] = {"device": device}
    elif place:
        record["on"] = {"place": place}
    for name in ("unit", "description"):
        if row.get(name):
            record[name] = row[name]
    tags = sorted({tag.strip() for tag in row.get("tags", "").split(";") if tag.strip()})
    if tags:
        record["tags"] = tags
    return record


def _position(
    reading: _Reading, number: int, address: str, row: dict[str, str]
) -> dict[str, Any] | None:
    """The ``s`` and ``model`` one row fills, or ``None`` after refusing its ``s``."""
    position: dict[str, Any] = {}
    text = row.get("s", "")
    if text:
        try:
            s = float(text)
        except ValueError:
            s = math.nan
        if not math.isfinite(s):
            reading.refuse(
                "channel",
                address,
                f"{reading.name} row {number} states s {text}, not a finite number",
                "write `s` in metres as a number",
            )
            return None
        position["s"] = s
    if row.get("model"):
        position["model"] = row["model"]
    return position


def _state_position(reading: _Reading, number: int, device: str, position: dict[str, Any]) -> None:
    """Record a row's position of its device, refusing one that differs from an earlier row's."""
    stated = reading.positions.setdefault(device, {})
    for slot, value in position.items():
        if slot not in stated:
            stated[slot] = (value, number)
            continue
        earlier, first = stated[slot]
        if earlier != value:
            reading.refuse(
                "device",
                device,
                f"{reading.name} rows {first} and {number} state {slot} "
                f"{_shown(earlier)} and {_shown(value)}",
                f"state one `{slot}` for the device",
            )


def _shown(value: Any) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def _add(reading: _Reading, number: int, record: dict[str, Any]) -> None:
    address = record["id"]
    first = reading.rows.get(address)
    if first is not None:
        reading.refuse(
            "channel",
            address,
            f"{reading.name} rows {first} and {number} both state it",
            "keep one row per address",
        )
        return
    reading.rows[address] = number
    reading.channels[address] = record


def _dump(path: Path, rows: list[dict[str, Any]]) -> Path:
    text = yaml_loader.safe_dump(
        rows, sort_keys=False, default_flow_style=False, allow_unicode=True, width=100
    )
    path.write_text(text, encoding="utf-8")
    return path
