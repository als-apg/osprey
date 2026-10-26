"""Computation helpers for optimization result summaries."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import polars as pl

from osprey_optimizer.models.api import AnnotatedValue, PointDelta, ResultSummary
from osprey_optimizer.models.types import DIRECTIONS, Direction
from osprey_optimizer.utils.json_safe import drop_non_finite

from ..optimization.best_point import finite_filter

logger = logging.getLogger(__name__)

# Number of best points included in a result summary.
TOP_N = 5


@dataclass
class SummaryContext:
    """Bundled context for computing a result summary."""

    objective_name: str
    variable_names: list[str]
    direction: Direction
    initial_point: dict[str, float] | None = None
    target_value: dict[str, float] | None = None
    variable_metadata: dict[str, dict[str, object]] | None = None
    observable_metadata: dict[str, dict[str, object]] | None = None
    model_file: str | None = None
    _all_metadata: dict[str, dict[str, object]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.direction not in DIRECTIONS:
            raise ValueError(
                f"Invalid direction {self.direction!r}; must be 'MAXIMIZE' or 'MINIMIZE'"
            )
        self.direction = Direction(self.direction)
        self._all_metadata = {**(self.variable_metadata or {}), **(self.observable_metadata or {})}


def _round_to_step(value: float, step_size: float | None) -> float:
    """Round *value* to the decimal precision implied by *step_size*."""
    if step_size is None or step_size <= 0:
        return value
    decimals = max(0, -math.floor(math.log10(step_size)))
    return round(value, decimals)


def _finite_map(values: dict[str, float] | None) -> dict[str, float] | None:
    """Drop non-finite entries so the summary stays JSON-serializable."""
    return None if values is None else drop_non_finite(values)


def _compute_delta(
    values: dict[str, float], reference: dict[str, float] | None
) -> dict[str, float] | None:
    """Compute per-key deltas between values and a reference dict."""
    if reference is None:
        return None
    return {k: values[k] - reference[k] for k in values if k in reference}


def _build_annotated(
    values: dict[str, float],
    initial: dict[str, float] | None,
    all_meta: dict[str, dict[str, object]],
) -> dict[str, AnnotatedValue]:
    """Build annotated value entries from raw values + metadata."""
    result: dict[str, AnnotatedValue] = {}
    for key, val in values.items():
        meta = all_meta.get(key, {})
        step = meta.get("step_size")
        step_f = float(step) if isinstance(step, (int, float)) else None
        rounded = _round_to_step(val, step_f)
        delta: float | None = None
        if initial and key in initial:
            delta = rounded - _round_to_step(initial[key], step_f)
        result[key] = AnnotatedValue(
            value=rounded,
            description=str(meta.get("description", "")),
            units=str(meta.get("units", "")),
            delta_vs_initial=delta,
        )
    return result


@dataclass
class _NarrativeInputs:
    context: SummaryContext
    obj_desc: str
    best_obj: float | None
    initial_obj: float | None
    total_iterations: int
    target_val: float | None
    target_units: str
    achievement_pct: float | None
    improved: bool | None
    best_point_annotated: dict[str, AnnotatedValue] | None


def _generate_text_summary(inputs: _NarrativeInputs) -> str:
    """Generate a multi-line plaintext narrative."""
    ctx = inputs.context
    u = inputs.target_units

    lines: list[str] = []

    header = f'Optimized "{inputs.obj_desc}" ({ctx.direction.value}'
    if inputs.target_val is not None:
        header += f", target: {inputs.target_val} {u}".rstrip()
    header += ")."
    lines.append(header)

    if inputs.best_obj is not None:
        best_line = f"Best: {inputs.best_obj} {u}".rstrip()
        if inputs.initial_obj is not None:
            best_line += f". Initial: {inputs.initial_obj} {u}".rstrip()
            delta = inputs.best_obj - inputs.initial_obj
            sign = "+" if delta >= 0 else ""
            best_line += f". Delta: {sign}{delta} {u}".rstrip()
        if inputs.achievement_pct is not None:
            best_line += f" ({inputs.achievement_pct:.1f}% of target)"
        best_line += "."
        lines.append(best_line)

    iter_line = f"{inputs.total_iterations} iterations evaluated."
    if inputs.improved is True:
        iter_line += " Improved over initial state."
    elif inputs.improved is False:
        iter_line += " No improvement over initial state."
    lines.append(iter_line)

    if inputs.best_point_annotated:
        var_deltas: list[tuple[str, float, str]] = []
        for key in ctx.variable_names:
            av = inputs.best_point_annotated.get(key)
            if av and av.delta_vs_initial is not None and av.delta_vs_initial != 0:
                label = av.description if av.description else key
                var_deltas.append((label, av.delta_vs_initial, av.units))
        var_deltas.sort(key=lambda t: abs(t[1]), reverse=True)
        if var_deltas:
            parts = [
                f"{name} {'+' if d >= 0 else ''}{d} {vu}".rstrip() for name, d, vu in var_deltas[:5]
            ]
            lines.append(f"Top variable changes: {', '.join(parts)}.")

    return "\n".join(lines)


def compute_result_summary(
    data: pl.DataFrame,
    context: SummaryContext,
) -> ResultSummary:
    """Compute a summary of the best optimization points with deltas.

    Args:
        data: Optimization data (rows are evaluation points).
        context: Summary context with objective, variables, and reference points.

    Returns:
        ResultSummary with best points, deltas, and reference values.
    """
    all_keys = [*context.variable_names, context.objective_name]
    total_iterations = len(data)

    # Reference points may carry NaN from failed reads; scrub before they reach
    # the response (see _finite_map). Non-finite objective rows are excluded too,
    # matching the runner's finite filter, so best points are always real values.
    safe_initial = _finite_map(context.initial_point)
    safe_target = _finite_map(context.target_value)

    filtered = data.filter(finite_filter(context.objective_name))

    if filtered.is_empty():
        return ResultSummary(
            best_points=[],
            initial_point=safe_initial,
            target_value=safe_target,
            objective_name=context.objective_name,
            direction=context.direction,
            total_iterations=total_iterations,
        )

    sorted_df = filtered.sort(
        context.objective_name, descending=(context.direction == "MAXIMIZE")
    ).head(TOP_N)

    if context.observable_metadata is None:
        logger.warning(
            "observable_metadata is None for objective %r; summary will lack "
            "description/units/target and tied-point detection is disabled",
            context.objective_name,
        )
    obj_meta = (context.observable_metadata or {}).get(context.objective_name, {})
    obj_step = obj_meta.get("step_size")
    obj_values = sorted_df[context.objective_name].to_list()

    tied_count: int | None = None
    tied_note: str | None = None
    if len(obj_values) > 1:
        first = obj_values[0]
        if obj_step and isinstance(obj_step, (int, float)) and obj_step > 0:
            tied = sum(1 for v in obj_values if abs(v - first) <= obj_step)
        else:
            tied = sum(1 for v in obj_values if v == first)
        if tied > 1:
            tied_count = tied
            tied_note = f"{tied} points tied at objective value {first}; showing first."
            sorted_df = sorted_df.head(1)

    best_points: list[PointDelta] = []
    all_metadata = context._all_metadata
    for row in sorted_df.iter_rows(named=True):
        values = {
            k: float(row[k]) for k in all_keys if row[k] is not None and math.isfinite(row[k])
        }
        annotated = _build_annotated(values, safe_initial, all_metadata) if all_metadata else None

        best_points.append(
            PointDelta(
                values=values,
                delta_vs_initial=_compute_delta(values, safe_initial),
                delta_vs_target=_compute_delta(values, safe_target),
                annotated_values=annotated,
            )
        )

    best_obj = best_points[0].values.get(context.objective_name) if best_points else None
    initial_obj = safe_initial.get(context.objective_name) if safe_initial else None

    obj_desc = str(obj_meta.get("description", ""))
    obj_units = str(obj_meta.get("units", ""))
    target_val_raw = obj_meta.get("target_value")
    target_val = float(target_val_raw) if isinstance(target_val_raw, (int, float)) else None
    if target_val is None and safe_target:
        target_val = safe_target.get(context.objective_name)

    achievement_pct: float | None = None
    if target_val is not None and target_val != 0 and best_obj is not None:
        achievement_pct = (best_obj / target_val) * 100

    improved: bool | None = None
    if best_obj is not None and initial_obj is not None:
        improved = (
            best_obj > initial_obj if context.direction == "MAXIMIZE" else best_obj < initial_obj
        )

    text_summary = _generate_text_summary(
        _NarrativeInputs(
            context=context,
            obj_desc=obj_desc or context.objective_name,
            best_obj=best_obj,
            initial_obj=initial_obj,
            total_iterations=total_iterations,
            target_val=target_val,
            target_units=obj_units,
            achievement_pct=achievement_pct,
            improved=improved,
            best_point_annotated=best_points[0].annotated_values if best_points else None,
        )
    )

    return ResultSummary(
        best_points=best_points,
        initial_point=safe_initial,
        target_value=safe_target,
        objective_name=context.objective_name,
        objective_description=obj_desc or None,
        direction=context.direction,
        total_iterations=total_iterations,
        best_objective_value=best_obj,
        initial_objective_value=initial_obj,
        target_value_scalar=target_val,
        target_units=obj_units or None,
        target_achievement_pct=achievement_pct,
        improved_over_initial=improved,
        tied_count=tied_count,
        tied_note=tied_note,
        text_summary=text_summary,
        model_file=context.model_file,
    )
