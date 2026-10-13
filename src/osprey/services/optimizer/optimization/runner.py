"""Backend-neutral optimization phase runner.

Provides ``run_optimization_phase`` and ``report_results``. The phase loop is
optimizer-agnostic: it drives any :class:`OptimizerProtocol` implementation
through a plain ask/evaluate/tell cycle (``suggest`` → ``evaluate`` →
``ingest``) and owns the accelerator-side concerns — initial-state seeding,
event emission, best-point selection, and the control-system error contract.
All xopt specifics live behind the optimizer (see ``xopt_backend.py``).
"""

import logging
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import polars as pl

from osprey.services.optimizer.events import (
    EventBus,
    EventType,
    IterationCompletedData,
    PhaseCompletedData,
    PhaseFailedData,
    PhaseStartedData,
    emit_event,
)
from osprey.services.optimizer.models.config import ConstraintSpec, OptimizationConfig
from osprey.services.optimizer.models.enums import AlgorithmType
from osprey.services.optimizer.models.exceptions import (
    ControlSystemUnreachableError,
    EvaluationError,
    WriteRefusedError,
)
from osprey.services.optimizer.models.protocols import OptimizerProtocol
from osprey.services.optimizer.models.types import BestPoint, Direction, VariableBounds

from ..utils.control import handle_control_callbacks
from .best_point import (
    best_feasible_row,
    drop_nonfinite_rows,
    feasible_filter,
    finite_filter,
    project_point,
)

logger = logging.getLogger(__name__)


@dataclass
class ControlContext:
    """Context object for control callbacks and event handling."""

    event_bus: EventBus | None = None
    check_cancellation: Callable[[], None] | None = None
    wait_if_paused: Callable[[], None] | None = None


@dataclass
class PhaseResult:
    """Result of a single optimization phase."""

    data: pl.DataFrame


def _emit_phase_failed(
    context: ControlContext, algorithm: AlgorithmType, exc: BaseException
) -> None:
    """Log and emit a PHASE_FAILED event for a control-system abort or a policy-refused write.

    ``str(exc)`` is the operator-facing message.
    """
    logger.error("Optimization phase aborted: %s", exc)
    emit_event(
        context.event_bus,
        EventType.PHASE_FAILED,
        PhaseFailedData(phase=algorithm, error_message=str(exc)),
    )


def _validate_current_env_values(
    bounds: VariableBounds, current_env_values: dict[str, float]
) -> dict[str, float]:
    """Return the current-state input covering *bounds*, raising if a variable is missing or out of range.

    Raises:
        ValueError: If current_env_values is missing a bounds variable or a
            value is outside its [min, max] range.
    """
    missing = [v for v in bounds if v not in current_env_values]
    if missing:
        raise ValueError(f"Current environment values missing variables: {missing}")

    current_input = {v: current_env_values[v] for v in bounds}

    out_of_bounds = [
        f"{var_name}={current_input[var_name]} not in [{lo}, {hi}]"
        for var_name, (lo, hi) in bounds.items()
        if current_input[var_name] < lo or current_input[var_name] > hi
    ]
    if out_of_bounds:
        raise ValueError("Current environment state is outside bounds: " + "; ".join(out_of_bounds))

    return current_input


# Suppressions below: linear orchestrator (setup → seed → iterate → finalize → emit);
# splitting it would invent abstractions without insight.
def run_optimization_phase(
    optimizer: OptimizerProtocol,
    evaluate: Callable[[dict[str, float]], dict[str, float]],
    bounds: VariableBounds,
    objective_name: str,
    direction: Direction,
    config: OptimizationConfig,
    context: ControlContext,
    current_env_values: dict[str, float],
    n_initial: int,
    static_columns: dict[str, float] | None = None,
    constraints: dict[str, ConstraintSpec] | None = None,
) -> PhaseResult:
    """Execute a single optimization phase over any :class:`OptimizerProtocol`.

    Args:
        optimizer: Backend optimizer, driven purely via suggest/ingest.
        evaluate: Evaluation function mapping proposed inputs to
            ``{**snapped_inputs, objective_name: value, **constraint_values}``.
            Receives only real setpoints (search space + fixed features); backend
            bookkeeping keys are filtered out before the call.
            Hard control-system faults propagate out of this call; an unreadable
            objective or constraint arrives as NaN.
        bounds: Resolved ``{name: [min, max]}`` search space; used to validate
            the current state and to draw uniform-random seed points.
        objective_name: Name of the objective column.
        direction: Objective direction; selects the best row.
        config: Optimization configuration (algorithm, n_iterations, ...).
        context: Control context (event bus, cancellation, pause).
        current_env_values: Current environment variable values. Must cover all
            bounds variables; evaluated first as the initial seed.
        n_initial: Number of initial samples (current state + random fill).
        static_columns: Fixed-feature columns (if any) appended to every
            recorded row so the persisted frame carries the full feature set.
        constraints: Optional outcome constraints. When present, best-so-far and
            the final best point are selected only from FEASIBLE rows (all
            constraints satisfied; an unverifiable reading is infeasible), and a
            phase that produces no feasible row fails — the applied setpoint must
            never violate an outcome constraint.

    Returns:
        PhaseResult containing the cumulative data DataFrame.

    Raises:
        EvaluationError: If no finite objective was measured, or (with
            constraints) if no evaluated point satisfied all constraints.
    """
    if not objective_name:
        raise ValueError("objective_name must be set in config")

    algorithm = config.algorithm
    n_iterations = config.n_iterations
    static = static_columns or {}
    constraint_specs = constraints or {}
    # Everything writable to hardware: the search space plus any fixed-feature
    # columns (the generator emits those in its candidates too). Anything else on a
    # candidate is backend bookkeeping — no key convention is hard-coded here.
    setpoint_names = set(bounds) | set(static)
    total_evaluations = n_initial + n_iterations

    current_input = _validate_current_env_values(bounds, current_env_values)

    logger.info("=" * 80)
    logger.info(
        "OPTIMIZATION: %s (%d initial + %d iterations = %d total evaluations)",
        algorithm,
        n_initial,
        n_iterations,
        total_evaluations,
    )
    logger.info("=" * 80)

    emit_event(
        context.event_bus,
        EventType.PHASE_STARTED,
        PhaseStartedData(phase=algorithm, n_samples=total_evaluations),
    )

    # Runner-owned cumulative record of every evaluated point (on-grid inputs +
    # objective + fixed-feature columns). Replaces xopt's internal data frame as
    # the source for published/persisted results and best-point selection.
    record: list[dict[str, Any]] = []

    def evaluate_ingest_record(proposed: dict[str, float]) -> None:
        """Evaluate *proposed*, tell the optimizer, and record the outcome.

        Only ``setpoint_names`` are evaluated; bookkeeping keys still ride the
        full *proposed* dict back to the optimizer and into the record, as the
        protocol requires. The outcome carries the snapped inputs and objective,
        so merging it over the proposed point lets the snapped values win.
        """
        outcome = evaluate({k: v for k, v in proposed.items() if k in setpoint_names})
        optimizer.ingest([{**proposed, **outcome}])
        record.append({**proposed, **outcome, **static})

    def publish(iteration: int) -> float:
        """Publish the cumulative record as an ITERATION_COMPLETED event.

        Called after every evaluation — seeding samples and optimization steps
        alike — so the interface updates incrementally. Returns the best finite
        objective seen so far.
        """
        # Infer the schema from every row, not polars' 100-row default: backend
        # bookkeeping columns first appear after the seed rows, so a run with
        # more than 100 seeds would silently lose them from the frame.
        current = pl.DataFrame(record, infer_schema_length=None)
        best_row = best_feasible_row(current, objective_name, constraint_specs, direction=direction)
        best_value = float(best_row[objective_name]) if best_row is not None else float("nan")
        emit_event(
            context.event_bus,
            EventType.ITERATION_COMPLETED,
            IterationCompletedData(
                phase=algorithm,
                iteration=iteration,
                total=total_evaluations,
                best_value=best_value,
                data=current,
            ),
        )
        return best_value

    try:
        if n_initial > 0:
            handle_control_callbacks(context)
            # Evaluate the current machine state first so the optimizer starts
            # from the actual operating point, then fill with uniform-random
            # points over the resolved bounds.
            logger.info("Evaluating current environment state as first initial sample")
            evaluate_ingest_record(current_input)
            publish(1)

            if n_initial > 1:
                logger.info("Seeding with %d additional random evaluation(s)...", n_initial - 1)
                for sample_idx in range(2, n_initial + 1):
                    # Uniform space-filling seed points; not security-sensitive (S311).
                    random_point = {
                        name: random.uniform(lo, hi) for name, (lo, hi) in bounds.items()
                    }
                    evaluate_ingest_record(random_point)
                    publish(sample_idx)

        logger.info("Running %d iterations...", n_iterations)
        for i in range(n_iterations):
            handle_control_callbacks(context)
            for suggestion in optimizer.suggest():
                evaluate_ingest_record(suggestion)
            iter_num = n_initial + i + 1
            best_value = publish(iter_num)
            logger.info(
                "  Iteration %d/%d - Best %s so far: %.4f",
                iter_num,
                total_evaluations,
                objective_name,
                best_value,
            )

        # Full-record schema inference, as in publish() above.
        data = pl.DataFrame(record, infer_schema_length=None)
        best_row = best_feasible_row(data, objective_name, constraint_specs, direction=direction)
        if best_row is None:
            if constraint_specs and not drop_nonfinite_rows(data, [objective_name]).is_empty():
                # The objective was measured, but no evaluated point satisfied the
                # constraints. Never apply an infeasible setpoint to hardware.
                described = ", ".join(
                    f"{name} {s.type.value} {s.value}" for name, s in constraint_specs.items()
                )
                raise EvaluationError(
                    f"Optimization produced no point satisfying the constraints ({described}) across "
                    f"{total_evaluations} evaluations; no feasible setpoint can be applied. "
                    "Relax the constraints or widen the search space."
                )
            raise EvaluationError(
                f"Optimization produced no finite '{objective_name}' values across "
                f"{total_evaluations} evaluations; the objective could not be measured. "
                "Check that its readback PV is reachable."
            )
        best_value = float(best_row[objective_name])
        logger.info("Optimization Phase Complete! Best %s: %.4f", objective_name, best_value)
        emit_event(
            context.event_bus,
            EventType.PHASE_COMPLETED,
            PhaseCompletedData(
                phase=algorithm,
                best_value=best_value,
            ),
        )
    except (ControlSystemUnreachableError, EvaluationError, WriteRefusedError) as exc:
        _emit_phase_failed(context, algorithm, exc)
        raise

    return PhaseResult(data=data)


def report_results(
    data: pl.DataFrame,
    env_variables: VariableBounds,
    objective_name: str,
    *,
    direction: Direction,
    constraints: dict[str, ConstraintSpec] | None = None,
) -> BestPoint:
    """Report and log final optimization results.

    Args:
        data: Optimization results DataFrame
        env_variables: Environment variable definitions
        objective_name: Name of the objective
        direction: Objective direction; selects the best row (the setpoint
            applied to hardware) accordingly.
        constraints: Optional outcome constraints; when present the applied best
            point is the best FEASIBLE point (all constraints satisfied), so an
            infeasible setpoint is never applied to hardware.

    Returns:
        Dictionary containing best point information
    """
    logger.info("=" * 80)
    logger.info("OPTIMIZATION COMPLETE!")
    logger.info("=" * 80)

    var_finite = drop_nonfinite_rows(data, env_variables)
    best_row = best_feasible_row(var_finite, objective_name, constraints or {}, direction=direction)
    if best_row is None:
        logger.warning("No finite feasible %s values to report", objective_name)
        return {}

    logger.info("Best result found:")
    logger.info("  %s: %.4f", objective_name, best_row[objective_name])
    logger.info("Best parameters:")
    for var_name in env_variables:
        logger.info("  %s: %.4f", var_name, best_row[var_name])

    logger.info("=" * 80)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 80)
    obj_finite = var_finite.filter(finite_filter(objective_name))
    summary_mean = float(obj_finite[objective_name].mean())  # type: ignore[arg-type]
    logger.info("  Best %s: %.4f", objective_name, best_row[objective_name])
    # The mean spans ALL finite evaluations (feasible or not) — with constraints
    # it is not comparable to "Best", which is feasible-only; label accordingly.
    logger.info("  Mean %s (all finite evaluations): %.4f", objective_name, summary_mean)
    if constraints:
        n_feasible = var_finite.filter(feasible_filter(constraints)).height
        logger.info("  Feasible evaluations: %d of %d finite", n_feasible, obj_finite.height)

    return project_point(best_row, env_variables)
