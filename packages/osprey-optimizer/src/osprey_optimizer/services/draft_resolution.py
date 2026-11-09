"""Submit-time resolution of an optimization draft against the live setpoints.

The two-phase job contract (issue #2): a persisted draft is fully resolved —
explicit active-variable list and materialized min/max/step overrides — so the
worker consumes it verbatim and never re-resolves. See
docs/superpowers/specs/2026-07-24-machine-settings-precheck-design.md.
"""

from __future__ import annotations

import logging
from typing import cast

from osprey_optimizer.models.config import OptimizationConfig, VariableOverride
from osprey_optimizer.models.exceptions import ConnectionError as OptimizerConnectionError
from osprey_optimizer.models.protocols import EnvironmentProtocol
from osprey_optimizer.optimization.coercion import coerce_readings

from .bounds_resolver import effective_step_size, resolve_bounds

logger = logging.getLogger(__name__)


def read_current_setpoints(env: EnvironmentProtocol, names: list[str]) -> dict[str, float]:
    """Read the setpoints of *names* from *env*, raising if any is missing.

    The setpoint, not the readback, is a variable's current value as the
    optimizer sees it: the run starts from it, each search window is centred
    on it, and a restore writes it back.

    Returns the setpoint dict with all values present (no ``None``).

    Raises:
        OptimizerConnectionError: If any variable's setpoint PV could not be read.
    """
    coerced, failed = coerce_readings(env.get_setpoints(names), names)
    if failed:
        raise OptimizerConnectionError(
            f"Cannot read the setpoints of variables required for optimization: {failed}"
        )
    return cast("dict[str, float]", coerced)


def resolve_draft_config(config: OptimizationConfig, env: EnvironmentProtocol) -> None:
    """Resolve *config* in place against *env*'s current setpoints.

    Caller owns the environment context (pass an entered env). On return,
    ``config.variables`` is the explicit active list and
    ``config.variable_overrides`` carries materialized min/max/step for each —
    ``config.require_resolved()`` holds.

    Raises:
        ValueError: Unknown variable names, an override ``resolve_bounds``
            rejects for the variable, or a search window wider than the
            variable's ``max_step``.
        OptimizerConnectionError: An active variable's setpoint cannot be read.
    """
    all_vars = env.variables
    if config.variables:
        unknown = [n for n in config.variables if n not in all_vars]
        if unknown:
            raise ValueError(
                f"Variables {unknown} not found in environment. Available: {list(all_vars)}"
            )
    active_names = (
        [n for n in all_vars if n in config.variables] if config.variables else list(all_vars)
    )

    current_values = read_current_setpoints(env, active_names)

    overrides = config.variable_overrides or {}
    resolved: dict[str, VariableOverride] = {}
    for name in active_names:
        meta = all_vars[name]
        override = overrides.get(name)
        step = effective_step_size(override, meta.step_size)
        bounds = resolve_bounds(
            override,
            current_value=current_values[name],
            safety_min=meta.min_value,
            safety_max=meta.max_value,
            step_size=step,
        )
        if bounds.was_clamped:
            logger.info(
                "Variable %s: requested [%s, %s] clamped/snapped to [%s, %s]",
                name,
                bounds.requested_min,
                bounds.requested_max,
                bounds.min,
                bounds.max,
            )
        if meta.max_step is not None and bounds.max - bounds.min > meta.max_step:
            raise ValueError(
                f"Variable {name!r}: its search window [{bounds.min}, {bounds.max}] is wider "
                f"than the max_step {meta.max_step} its setpoint {meta.setpoint_pv} allows in "
                "one write, so moves inside it, and a restore across it, would be refused. "
                f"Narrow the window to at most {meta.max_step}."
            )
        resolved[name] = VariableOverride(min=bounds.min, max=bounds.max, step_size=step)

    config.variables = active_names
    config.variable_overrides = resolved
