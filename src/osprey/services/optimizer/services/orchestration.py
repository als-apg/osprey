"""Core optimization engine and result dataclasses."""

from __future__ import annotations

from dataclasses import dataclass

from osprey.services.optimizer.events import (
    EventBus,
    EventType,
    OptimizationCompletedData,
    OptimizationStartedData,
    emit_event,
)
from osprey.services.optimizer.models.config import OptimizationConfig, machine_config
from osprey.services.optimizer.models.protocols import EnvironmentFactory, EnvironmentProtocol
from osprey.services.optimizer.models.types import BestPoint

from ..optimization.evaluation import Evaluator, VariableGrid
from ..optimization.runner import (
    ControlContext,
    report_results,
    run_optimization_phase,
)
from ..optimization.types import OptimizationMetadata, OptimizationResult
from ..optimization.xopt_backend import XoptOptimizer
from ..utils.cancellation import CancellationToken
from .draft_resolution import read_required_variables

__all__ = [
    "OptimizationEngine",
    "OptimizationMetadata",
    "OptimizationResult",
    "RunnerServices",
    "apply_best_point",
    "read_required_variables",
    "require_objective",
    "resolve_environment",
]


def resolve_environment(
    config: OptimizationConfig,
    factory: EnvironmentFactory,
) -> tuple[str, EnvironmentProtocol]:
    """Validate ``environment_name`` and load the environment.

    Returns the validated name and the environment instance.

    Raises:
        ValueError: If *environment_name* is ``None``.
    """
    environment_name = config.environment_name
    if environment_name is None:
        raise ValueError("environment_name must be set in config")
    if config.control_target is None:
        # The API refuses such a start; this is the worker's own refusal, so a
        # config that reached it some other way still never picks a machine.
        raise ValueError("control_target must be set in config (va, standin or live)")
    env_config = {
        **(config.environment_config or {}),
        **machine_config(config.control_target, config.execution_mode),
    }
    env = factory(environment_name, env_config)
    return environment_name, env


def require_objective(
    env: EnvironmentProtocol,
    objective_name: str | None,
    environment_name: str,
) -> str:
    """Validate that the configured objective exists in the environment.

    Returns the validated objective name.

    Raises:
        ValueError: If *objective_name* is unset or not among the
            environment's observables.
    """
    if not objective_name:
        raise ValueError("objective_name must be set to run an optimization.")
    if objective_name not in env.observables:
        raise ValueError(
            f"Objective '{objective_name}' not found in environment '{environment_name}'. "
            f"Available objectives: {env.available_objectives}"
        )
    return objective_name


def apply_best_point(env: EnvironmentProtocol, result: OptimizationResult) -> BestPoint:
    """Set the environment's variables to the best point's values.

    Only variables the environment knows about are written; the env is driven
    inside its own context manager. Callers are responsible for loading *env*
    and handling load-time errors.
    """
    with env:
        variable_values = {
            name: result.best_point[name] for name in env.variables if name in result.best_point
        }
        env.set_variables(variable_values)
    return result.best_point


@dataclass
class RunnerServices:
    """Infrastructure services required by the runner."""

    cancellation_token: CancellationToken | None = None
    event_bus: EventBus | None = None


class OptimizationEngine:
    """High-level orchestrator for optimization runs."""

    def __init__(
        self,
        config: OptimizationConfig,
        environment_factory: EnvironmentFactory,
        services: RunnerServices | None = None,
        job_id: str | None = None,
        initial_point: dict[str, float] | None = None,
    ) -> None:
        self.config = config
        self.environment_factory = environment_factory
        self.initial_point = initial_point

        services = services or RunnerServices()
        self.event_bus = services.event_bus
        self.cancellation_token = services.cancellation_token
        self.job_id = job_id

    def run(self) -> OptimizationResult:
        """Execute optimization and return structured results."""
        environment_name, env = resolve_environment(self.config, self.environment_factory)

        objective_name = require_objective(env, self.config.objective_name, environment_name)

        # Validate constraint observables up front (parallel to the objective check)
        # so an unknown observable fails with a clear message before any hardware is
        # touched — never silently dropped, which would run an unconstrained search.
        constraints = self.config.constraints
        unknown_constraints = [name for name in constraints if name not in env.observables]
        if unknown_constraints:
            raise ValueError(
                f"Constraint observables {unknown_constraints} not found in environment "
                f"'{environment_name}'. Available observables: {list(env.observables.keys())}"
            )

        metadata = OptimizationMetadata(config=self.config, job_id=self.job_id)

        with env:
            all_vars = env.variables

            # Two-phase contract (issue #2): the config arriving here is a
            # fully resolved draft. Anything else is a lifecycle bug — fail
            # loudly, never re-resolve against the machine.
            if not self.config.variables:
                raise ValueError("config is not resolved: variables is unset")
            unknown = [name for name in self.config.variables if name not in all_vars]
            if unknown:
                raise ValueError(
                    f"Variables {unknown} not found in environment '{environment_name}'. "
                    f"Available variables: {list(all_vars.keys())}"
                )
            overrides = self.config.variable_overrides or {}
            active_variables: dict[str, list[float]] = {}
            grid: dict[str, VariableGrid] = {}
            for name in self.config.variables:
                override = overrides.get(name)
                if (
                    override is None
                    or override.min is None
                    or override.max is None
                    or override.step_size is None
                ):
                    raise ValueError(f"config is not resolved: incomplete override for '{name}'")
                active_variables[name] = [override.min, override.max]
                grid[name] = VariableGrid(override.min, override.max, override.step_size)

            current_env_values = self.initial_point
            if current_env_values is None:
                current_env_values = read_required_variables(env, list(all_vars.keys()))

            # Emit the start event now that the search space is resolved, so the
            # event carries the resolved [min, max] per variable (the VOCS bounds).
            # Consumers persist these to drive the live figure's normalization.
            emit_event(
                self.event_bus,
                EventType.OPTIMIZATION_STARTED,
                OptimizationStartedData(
                    objective=objective_name,
                    variable_bounds={name: [g.min, g.max] for name, g in grid.items()},
                ),
            )

            token = self.cancellation_token
            evaluator = Evaluator(
                env,
                objective_name,
                check_cancellation=token.check_cancellation if token else None,
                grid=grid,
                constraint_names=list(constraints),
            )

            direction = self.config.objective_direction or env.observables[objective_name].direction

            optimizer = XoptOptimizer(
                variables=active_variables,
                objective_name=objective_name,
                direction=direction,
                config=self.config,
                current_env_values=current_env_values,
                constraints=constraints,
            )
            n_initial = optimizer.resolve_n_initial()

            control_context = ControlContext(
                event_bus=self.event_bus,
                check_cancellation=token.check_cancellation if token else None,
                wait_if_paused=token.wait_if_paused if token else None,
            )

            phase_result = run_optimization_phase(
                optimizer=optimizer,
                evaluate=evaluator,
                bounds=active_variables,
                objective_name=objective_name,
                direction=direction,
                config=self.config,
                context=control_context,
                current_env_values=current_env_values,
                n_initial=n_initial,
                static_columns=optimizer.fixed_features,
                constraints=constraints,
            )
            data = phase_result.data

            best_row = report_results(
                data, active_variables, objective_name, direction=direction, constraints=constraints
            )

            emit_event(
                self.event_bus,
                EventType.OPTIMIZATION_COMPLETED,
                OptimizationCompletedData(objective=objective_name),
            )

            return OptimizationResult(
                data=data,
                best_point=best_row,
                metadata=metadata,
                initial_point=self.initial_point,
                model=optimizer.trained_model(),
                model_variables=optimizer.model_variables(),
            )
