"""Shared best-point / finite-objective selection over results frames.

Single source of truth for "which rows carry a usable objective" and "which row
is the best", so the runner, the cancellation partial-save, and the API summary
can't drift on direction handling or finite filtering.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

import polars as pl

from osprey.services.optimizer.models.config import ConstraintSpec
from osprey.services.optimizer.models.enums import ConstraintType
from osprey.services.optimizer.models.types import BestPoint, Direction


def feasible_filter(constraints: dict[str, ConstraintSpec]) -> pl.Expr:
    """Expression selecting rows that satisfy *all* outcome constraints.

    Each constraint applies a strict inequality against its observable column
    (``GREATER_THAN`` → ``col > value``, ``LESS_THAN`` → ``col < value``), matching
    xopt's VOCS feasibility. A null or non-finite (NaN/±inf) constraint reading is
    treated as INFEASIBLE: an unverifiable constraint cannot be shown satisfied, so
    the point is never eligible to be applied to hardware. With no constraints the
    filter is a constant ``True`` (every row eligible).
    """
    expr = pl.lit(value=True)
    for name, spec in constraints.items():
        col = pl.col(name)
        verifiable = finite_filter(name)
        satisfies = (
            col > spec.value if spec.type is ConstraintType.GREATER_THAN else col < spec.value
        )
        expr = expr & verifiable & satisfies
    return expr


def finite_filter(column: str) -> pl.Expr:
    """Expression selecting rows whose ``column`` is a real, finite value.

    Excludes null and non-finite (NaN/±inf) values — e.g. xopt hard-fail
    sentinels — so they can never be reported as a best point.
    """
    return pl.col(column).is_finite() & pl.col(column).is_not_null()


def drop_nonfinite_rows(data: pl.DataFrame, columns: Iterable[str]) -> pl.DataFrame:
    """Drop rows where any of ``columns`` is null or non-finite."""
    return data.filter(*[finite_filter(name) for name in columns])


def best_finite_row(
    data: pl.DataFrame, objective_name: str, *, direction: Direction
) -> dict[str, Any] | None:
    """Return the row with the best *finite* objective value, as a named dict.

    ``direction`` picks the largest finite value for ``MAXIMIZE``, otherwise the
    smallest. Returns ``None`` when no row has a finite objective.
    """
    finite = data.filter(finite_filter(objective_name))
    if finite.is_empty():
        return None
    return cast(
        dict[str, Any],
        finite.sort(objective_name, descending=(direction == "MAXIMIZE")).row(0, named=True),
    )


def best_feasible_row(
    data: pl.DataFrame,
    objective_name: str,
    constraints: dict[str, ConstraintSpec],
    *,
    direction: Direction,
) -> dict[str, Any] | None:
    """Best finite objective row among those satisfying all constraints.

    Applies the constraint feasibility filter first (a no-op when ``constraints``
    is empty, so behavior is unchanged for unconstrained runs), then selects the
    best finite objective for ``direction``. Returns ``None`` when no feasible row
    has a finite objective.
    """
    feasible = data.filter(feasible_filter(constraints)) if constraints else data
    return best_finite_row(feasible, objective_name, direction=direction)


def project_point(row: dict[str, Any], variables: Iterable[str]) -> BestPoint:
    """Project a selected row onto ``variables`` as ``{name: float(value)}``."""
    return {name: float(row[name]) for name in variables}


def build_best_point(
    data: pl.DataFrame,
    objective_name: str,
    variables: Iterable[str],
    *,
    direction: Direction,
) -> BestPoint:
    """Best finite point projected onto ``variables``, as ``{name: value}``.

    Rows where the objective *or any of* ``variables`` is non-finite are excluded
    before selection, so the returned setpoints are always real values safe to
    apply to hardware. ``direction`` picks the best objective. Returns ``{}`` when
    no row qualifies.
    """
    selected = list(variables)
    best_row = best_finite_row(
        drop_nonfinite_rows(data, selected), objective_name, direction=direction
    )
    return project_point(best_row, selected) if best_row is not None else {}
