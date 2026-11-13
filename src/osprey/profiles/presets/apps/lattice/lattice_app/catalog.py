"""The analyses the panel shows, and the result document every engine answers with.

An engine is free in how it computes; what it hands back is fixed here, so the panel, the
agent's tools and the file exports read one shape whichever engine produced it::

    {"scalars": {"tune_x": 0.21, ...},
     "units":   {"circumference_m": "m", ...},
     "tables":  {"twiss": {"columns": {"s": [...], "beta_x": [...]},
                           "units":   {"s": "m", "beta_x": "m"}}}}

Each analysis names the tables and columns the panel plots. An engine may add scalars,
columns and tables of its own; they reach the agent and the exports unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

#: The result document's version, stored with every cached result.
RESULT_SCHEMA = "lattice.result/1"
#: The job file's version, which an engine may check.
JOB_SCHEMA = "lattice.job/1"


@dataclass(frozen=True)
class Analysis:
    """One analysis of a deck.

    Attributes:
        name: The analysis's key in config, routes and tools.
        label: What the panel calls it.
        settings: The default settings, which ``analyses.<name>.settings`` overrides key
            by key. Engines read the keys they know.
        tables: The tables the panel plots, each with the numeric columns it needs.
        scalars: The scalars the panel shows first.
    """

    name: str
    label: str
    settings: Mapping[str, Any] = field(default_factory=dict)
    tables: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    scalars: tuple[str, ...] = ()


ANALYSES: dict[str, Analysis] = {
    analysis.name: analysis
    for analysis in (
        Analysis(
            "optics",
            "Optics",
            tables={"twiss": ("s", "beta_x", "beta_y", "eta_x")},
            scalars=("tune_x", "tune_y", "chromaticity_x", "chromaticity_y"),
        ),
        Analysis(
            "chromaticity",
            "Chromaticity",
            settings={"dp_min_pct": -3.0, "dp_max_pct": 3.0, "n_steps": 25},
            tables={"tune_vs_dp": ("dp_pct", "tune_x", "tune_y")},
        ),
        Analysis(
            "da",
            "Dynamic aperture",
            settings={"nturns": 512, "n_angles": 19, "amp_max_mm": 30.0, "n_bisect": 15},
            tables={"aperture": ("x_mm", "y_mm")},
            scalars=("area_mm2",),
        ),
        Analysis(
            "footprint",
            "Tune footprint",
            settings={"n_amp": 10, "x_max_mm": 3.0, "y_max_mm": 1.0, "n_half": 256},
            tables={"tunes": ("x_mm", "y_mm", "tune_x", "tune_y")},
        ),
        Analysis(
            "lma",
            "Momentum aperture",
            settings={"nturns": 512, "n_refpts": 15, "dp_max_pct": 5.0, "n_bisect": 15},
            tables={"aperture": ("s", "dp_plus_pct", "dp_minus_pct")},
        ),
    )
}


def _number(value: Any) -> float | None:
    """*value* as a finite float; ``None`` for a missing, non-numeric or non-finite one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _column(table: str, name: str, values: Any) -> list[Any]:
    if not isinstance(values, list):
        raise ValueError(f"column {name} of table {table} is not a list")
    if all(isinstance(value, str) for value in values):
        return list(values)
    return [_number(value) for value in values]


def _table(name: str, raw: Any, required: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or not isinstance(raw.get("columns"), Mapping):
        raise ValueError(f"table {name} has no columns")
    columns = {str(key): _column(name, str(key), values) for key, values in raw["columns"].items()}
    if missing := [column for column in required if column not in columns]:
        raise ValueError(f"table {name} lacks column {', '.join(missing)}")
    if len({len(values) for values in columns.values()}) > 1:
        raise ValueError(f"the columns of table {name} differ in length")
    for column in required:
        if any(isinstance(value, str) for value in columns[column]):
            raise ValueError(f"column {column} of table {name} is not numeric")
    return {"columns": columns, "units": _units(raw.get("units"), columns)}


def _units(raw: Any, named: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): str(unit) for key, unit in raw.items() if key in named and unit}


def _mapping(raw: Any) -> Mapping[Any, Any]:
    return raw if isinstance(raw, Mapping) else {}


def normalize_result(analysis: Analysis, raw: Any) -> dict[str, Any]:
    """An engine's answer as a result document, checked against what the panel plots.

    A non-finite number becomes ``None``, so the document is strict JSON.

    Raises:
        ValueError: The answer is no result document, reports an error, or lacks a table
            or column the analysis needs.
    """
    if not isinstance(raw, Mapping):
        raise ValueError("the engine's output is not a JSON object")
    if raw.get("error"):
        raise ValueError(str(raw["error"]))
    tables_in = _mapping(raw.get("tables"))
    if missing := [name for name in analysis.tables if name not in tables_in]:
        raise ValueError(f"the engine's output lacks table {', '.join(missing)}")
    tables = {
        str(name): _table(str(name), table, analysis.tables.get(str(name), ()))
        for name, table in tables_in.items()
    }
    scalars_in = _mapping(raw.get("scalars"))
    scalars = {
        str(key): value if isinstance(value, str) else _number(value)
        for key, value in scalars_in.items()
    }
    return {"scalars": scalars, "units": _units(raw.get("units"), scalars), "tables": tables}
