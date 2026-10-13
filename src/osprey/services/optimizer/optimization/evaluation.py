"""Evaluation function for the optimization loop over an EnvironmentProtocol."""

import logging
import math
from collections.abc import Callable
from typing import NamedTuple

from osprey.services.optimizer.models.protocols import EnvironmentProtocol
from osprey.services.optimizer.models.types import VariableInputs
from osprey.services.optimizer.optimization.coercion import reading_to_float
from osprey.services.optimizer.services.bounds_resolver import snap_value_to_grid

logger = logging.getLogger(__name__)


class VariableGrid(NamedTuple):
    """Step grid for one variable: values snap to ``min + k*step`` within ``[min, max]``."""

    min: float
    max: float
    step: float


class Evaluator:
    """Callable evaluation function for the optimization loop.

    Holds a reference to the environment and exposes a ``__call__`` interface
    the run loop invokes on each candidate point. Pure evaluation: it applies
    the (grid-snapped) inputs to hardware, reads the objective back, and returns
    the snapped inputs alongside the measured objective.
    """

    def __init__(
        self,
        environment: EnvironmentProtocol,
        objective_name: str,
        check_cancellation: Callable[[], None] | None = None,
        grid: dict[str, VariableGrid] | None = None,
        objective_retries: int = 2,
        constraint_names: list[str] | None = None,
    ):
        """Initialize the evaluator.

        Args:
            environment: Environment instance implementing EnvironmentProtocol.
            objective_name: Name of the observable to optimize.
            check_cancellation: Optional callable that raises CancellationError if cancelled.
            grid: Optional per-variable step grid. Each proposed value for a gridded
                variable is snapped to its grid before being applied to hardware, so the
                optimizer only ever evaluates grid points. Variables absent from the map
                stay continuous (unchanged behavior).
            objective_retries: Number of extra re-measurements attempted when any
                requested outcome (objective or constraint) reads back unreadable
                (NaN/None) before tolerating it as a non-finite observation. Default 2
                means up to 3 measurements total.
            constraint_names: Optional outcome-constraint observable names measured in the
                same ``evaluate`` transaction as the objective and returned alongside it.
                Empty (the default) is behaviorally identical to measuring the objective
                alone.
        """
        self.environment = environment
        self.objective_name = objective_name
        self.check_cancellation = check_cancellation
        self.grid = grid or {}
        self.objective_retries = objective_retries
        self.constraint_names = constraint_names or []

    def __call__(self, inputs: VariableInputs) -> dict[str, float]:
        """Evaluate the objective at the given variable settings.

        Delegates to the environment's ``evaluate`` method which handles
        setting variables, waiting for hardware to settle, and reading
        fresh measurements.

        Args:
            inputs: Dictionary with variable names and values

        Returns:
            Dictionary of the snapped (grid-quantized) variable inputs actually
            applied to hardware plus the measured objective and any constraint
            observables: ``{**snapped_inputs, objective_name: value, **constraints}``.
            Any outcome (objective or constraint) still unreadable after exhausting all
            retries is recorded as ``nan`` so the run loop records the point without
            aborting; the loop fails the phase only when NO finite objective is
            produced at all.

        Raises:
            ControlSystemUnreachableError: Propagated immediately from the
                environment; hard control-system errors are never retried.
            EvaluationError: Propagated immediately when raised by the
                environment itself (not when the objective is merely unreadable).
        """
        # Every key is a real setpoint — the run loop strips backend bookkeeping
        # before calling us — so an unknown name here is a genuine error and must
        # reach the environment to be rejected loudly, not be filtered away.
        variable_inputs = dict(inputs)
        # Quantize gridded variables to their step before applying to hardware: the
        # objective is then measured at a grid point, so the optimizer only ever
        # explores grid points. The snapped inputs are returned so the loop records
        # the on-grid point actually evaluated (not the proposed off-grid one).
        for name, g in self.grid.items():
            if name in variable_inputs:
                variable_inputs[name] = snap_value_to_grid(
                    variable_inputs[name], vmin=g.min, vmax=g.max, step=g.step
                )

        # Objective + constraint observables ride ONE evaluate transaction (constraints
        # are plain observables, no extra round-trip). Retry while any outcome reads
        # unreadable — a transient dropout must not poison the point, persistent loss
        # flows through as non-finite. Control-system errors propagate, never retried.
        requested = [self.objective_name, *self.constraint_names]
        attempts = self.objective_retries + 1
        values: dict[str, float | None] = {}
        for attempt in range(1, attempts + 1):
            obs_readings = self.environment.evaluate(
                variable_inputs,
                requested,
                check_cancellation=self.check_cancellation,
            )
            values = {name: reading_to_float(obs_readings[name]) for name in requested}
            unreadable = [name for name, v in values.items() if v is None]
            if not unreadable:
                break
            logger.warning(
                "Outcome(s) %s unreadable on attempt %d/%d", unreadable, attempt, attempts
            )
        else:
            logger.warning(
                "Outcome(s) %s still unreadable after %d attempt(s); recording NaN",
                unreadable,
                attempts,
            )

        return {
            **variable_inputs,
            **{name: (v if v is not None else math.nan) for name, v in values.items()},
        }
