"""A result as a file: CSV, or SDDS for the elegant toolchain.

Both carry one table and the result's scalars, so the file a physicist opens holds the
numbers the panel shows and the agent reads.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping
from typing import Any

#: The formats an export is offered in, each with its media type.
FORMATS = {"csv": "text/csv", "sdds": "text/plain"}


def _provenance(document: Mapping[str, Any]) -> dict[str, Any]:
    job = document.get("job") or {}
    return {
        "model": job.get("model"),
        "analysis": job.get("analysis"),
        "engine": job.get("engine"),
        "deck_sha256": job.get("deck_sha256"),
        "computed_at": document.get("computed_at"),
    }


def to_csv(document: Mapping[str, Any], table: str) -> str:
    """One table as CSV, under ``# name = value`` lines of provenance and scalars."""
    columns = document["tables"][table]["columns"]
    units = document["tables"][table].get("units") or {}
    out = io.StringIO()
    for name, value in {**_provenance(document), **document.get("scalars", {})}.items():
        if value is not None:
            out.write(f"# {name} = {value}\n")
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow([f"{name} [{units[name]}]" if name in units else name for name in columns])
    for row in zip(*columns.values(), strict=True):
        writer.writerow(["" if value is None else value for value in row])
    return out.getvalue()


def _sdds_text(value: Any) -> str:
    text = str(value).replace('"', "'")
    return f'"{text}"'


def _sdds_value(value: Any, kind: str) -> str:
    if kind == "string":
        return _sdds_text("" if value is None else value)
    return "nan" if value is None else repr(float(value))


def _sdds_field(what: str, name: str, kind: str, unit: str | None) -> str:
    units = f", units={_sdds_text(unit)}" if unit else ""
    return f"&{what} name={name}, type={kind}{units} &end\n"


def to_sdds(document: Mapping[str, Any], table: str) -> str:
    """One table as an ASCII SDDS file: scalars as parameters, columns as columns."""
    columns = document["tables"][table]["columns"]
    column_units = document["tables"][table].get("units") or {}
    parameters = {
        name: value
        for name, value in {**_provenance(document), **document.get("scalars", {})}.items()
        if value is not None
    }
    scalar_units = document.get("units") or {}

    def kind(values: Any) -> str:
        return "string" if any(isinstance(value, str) for value in values) else "double"

    out = io.StringIO()
    out.write("SDDS1\n")
    for name, value in parameters.items():
        out.write(_sdds_field("parameter", name, kind([value]), scalar_units.get(name)))
    for name, values in columns.items():
        out.write(_sdds_field("column", name, kind(values), column_units.get(name)))
    out.write("&data mode=ascii &end\n")
    for value in parameters.values():
        out.write(_sdds_value(value, kind([value])) + "\n")
    rows = list(zip(*columns.values(), strict=True))
    out.write(f"{len(rows)}\n")
    kinds = [kind(values) for values in columns.values()]
    for row in rows:
        fields = (_sdds_value(value, k) for value, k in zip(row, kinds, strict=True))
        out.write(" ".join(fields) + "\n")
    return out.getvalue()


def render(document: Mapping[str, Any], table: str, fmt: str) -> str:
    """*table* of *document* in *fmt*, one of :data:`FORMATS`."""
    return to_sdds(document, table) if fmt == "sdds" else to_csv(document, table)
