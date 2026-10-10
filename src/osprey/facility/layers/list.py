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

A file whose first row names no ``address`` column has no header and holds one
address per line. Blank lines are skipped and every cell is stripped.

The layer writes ``imported/list/channels.yaml``, one record per row sorted by
address, holding only the fields its row fills; it seeds no authored file. A
row naming both a device and a place, a role outside the three, an unknown or
repeated column, a row with more cells than the header, a row with no address
or an address on two rows is ``source-invalid``, every one reported, and
nothing is written.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click
import yaml

from osprey.facility.errors import FacilityBuildError

__all__ = ["COLUMNS", "LAYER_DIR", "ListSourceInvalid", "import_list"]

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
)

_ROLES = ("setpoint", "readback", "none")


class ListSourceInvalid(click.ClickException):
    """The list file holds rows the layer cannot write; nothing was written.

    Attributes:
        errors: One ``source-invalid`` error per problem, in file order.
    """

    exit_code = 1

    def __init__(self, errors: list[FacilityBuildError]) -> None:
        self.errors = errors
        super().__init__("\n".join(error.format_message() for error in errors))

    def show(self, file: Any = None) -> None:
        """Write each error's line alone, with no ``Error: `` prefix.

        Args:
            file: The stream to write to; stderr when omitted.
        """
        for error in self.errors:
            error.show(file)


@dataclass
class _Reading:
    """What one pass over the file found."""

    name: str
    channels: dict[str, dict[str, Any]] = field(default_factory=dict)
    rows: dict[str, int] = field(default_factory=dict)
    errors: list[FacilityBuildError] = field(default_factory=list)

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


def import_list(path: Path, facility_dir: Path) -> list[Path]:
    """Write a channel list as the list layer's sources.

    Args:
        path: The CSV file.
        facility_dir: The ``data/facility`` directory.

    Returns:
        Every file written.

    Raises:
        ListSourceInvalid: The file has rows the layer refuses; nothing is
            written.
    """
    reading = _read(path)
    if reading.errors:
        raise ListSourceInvalid(reading.errors)
    layer = facility_dir / LAYER_DIR
    layer.mkdir(parents=True, exist_ok=True)
    return [_dump(layer / "channels.yaml", [reading.channels[a] for a in sorted(reading.channels)])]


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
    return reading


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
    if not ok:
        return None
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
    text = yaml.safe_dump(
        rows, sort_keys=False, default_flow_style=False, allow_unicode=True, width=100
    )
    path.write_text(text, encoding="utf-8")
    return path
