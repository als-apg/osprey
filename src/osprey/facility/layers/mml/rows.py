"""The mml layer's row table, ``imported/mml/rows.json``.

An MML export names each device of a family by its ``DeviceList`` row, and a
response export names its monitor and corrector rows the same way. The import
records which device each export row became, so the response check matches an
export row to a device through this table and never through the device record::

    {
      "schema": "osprey.facility.mml_rows/1",
      "rows": [
        {"model": <model name>, "family": <raw family>,
         "device_list": [<int>, ...], "device": <device id>},
        ...
      ]
    }

Rows are sorted by (model, family, device_list) and written with sorted keys,
so an unchanged import rewrites the same bytes. The file is the layer's own:
the build reads only the layer's ``*.yaml`` record files and never this one.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from osprey.facility.layers.mml.mapping import LAYER_DIR

__all__ = ["ROWS_FILE", "ROWS_SCHEMA", "read_rows", "write_rows"]

#: The row table's file name under the layer directory.
ROWS_FILE = "rows.json"

#: The header the row table carries.
ROWS_SCHEMA = "osprey.facility.mml_rows/1"


def _sort_key(row: Mapping[str, Any]) -> tuple[str, str, tuple[int, ...], str]:
    return (str(row["model"]), str(row["family"]), tuple(row["device_list"]), str(row["device"]))


def write_rows(layer: Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Write the row table into the layer directory.

    Args:
        layer: The layer directory, ``data/facility/imported/mml``.
        rows: One ``{model, family, device_list, device}`` mapping per export row.

    Returns:
        The file written.
    """
    table = [
        {
            "device": str(row["device"]),
            "device_list": [int(part) for part in row["device_list"]],
            "family": str(row["family"]),
            "model": str(row["model"]),
        }
        for row in rows
    ]
    table.sort(key=_sort_key)
    target = layer / ROWS_FILE
    text = json.dumps({"rows": table, "schema": ROWS_SCHEMA}, indent=2, sort_keys=True)
    target.write_text(text + "\n", encoding="utf-8")
    return target


def read_rows(facility_dir: Path) -> dict[tuple[str, str], dict[tuple[int, ...], str]]:
    """The row table as ``{(model, raw family): {row: device id}}``.

    A row two devices claim binds nothing. A missing table reads as empty.

    Args:
        facility_dir: The ``data/facility`` directory.

    Returns:
        Each (model, raw family)'s rows, keyed by their ``DeviceList`` row.
    """
    path = facility_dir / LAYER_DIR / ROWS_FILE
    if not path.is_file():
        return {}
    document = json.loads(path.read_text(encoding="utf-8"))
    claims: dict[tuple[str, str], dict[tuple[int, ...], set[str]]] = {}
    for row in document.get("rows", []):
        family = claims.setdefault((str(row["model"]), str(row["family"])), {})
        family.setdefault(tuple(int(part) for part in row["device_list"]), set()).add(
            str(row["device"])
        )
    return {
        key: {row: next(iter(devices)) for row, devices in family.items() if len(devices) == 1}
        for key, family in claims.items()
    }
