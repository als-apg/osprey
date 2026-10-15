"""Run-configuration API router.

Endpoints that prepare an optimization run's configuration — the environment's
presets and the bounds-preview computation for the cockpit table. These are
run-config concerns rather than live machine I/O, so they live apart from
``routers/environments.py`` while sharing its ``/environments/{name}`` prefix.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status

from osprey_optimizer.api.dependencies import (
    Machine,
    get_environment_registry,
    require_machine,
)
from osprey_optimizer.api.environment_ops import (
    environment_presets,
    load_environment,
    lookup_preset,
    read_errors,
    validate_variable_names,
)
from osprey_optimizer.models.api import (
    InitialBoundsResponse,
    ResolveBoundsRequest,
    ResolveBoundsResponse,
    ResolvedBoundsModel,
)
from osprey_optimizer.models.config import Preset, VariableOverride
from osprey_optimizer.models.protocols import VariableMetadata
from osprey_optimizer.optimization.coercion import reading_to_float
from osprey_optimizer.services.bounds_resolver import effective_step_size, resolve_bounds
from osprey_optimizer.services.plugin_registry import EnvironmentRegistry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/environments", tags=["environments"])


def _resolve_one_bound(
    var_name: str,
    meta: VariableMetadata,
    override: VariableOverride | None,
    current_value: float,
) -> ResolvedBoundsModel:
    """Resolve one variable's bounds, mapping a ``ValueError`` to a 422."""
    step = effective_step_size(override, meta.step_size)
    try:
        return resolve_bounds(
            override,
            current_value=current_value,
            safety_min=meta.min_value,
            safety_max=meta.max_value,
            step_size=step,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Variable '{var_name}': {e}",
        ) from e


@router.get("/{name}/presets", response_model=dict[str, Preset])
def get_environment_presets(
    name: str,
    registry: EnvironmentRegistry = Depends(get_environment_registry),
) -> dict[str, Preset]:
    """The run setups the environment offers, by name; empty when it offers none."""
    return environment_presets(registry, name)


@router.get("/{name}/initial-bounds", response_model=InitialBoundsResponse)
def get_environment_initial_bounds(
    name: str,
    preset: str | None = Query(None, description="Preset whose variables and windows to apply."),
    machine: Machine = Depends(require_machine),
    registry: EnvironmentRegistry = Depends(get_environment_registry),
) -> InitialBoundsResponse:
    """Resolve initial bounds for the cockpit table.

    With a preset, only its variables are listed and its ``variable_overrides``
    apply to their current readbacks (e.g. ``window_fraction: 0.1`` → a window
    10% of the variable's safety range wide, centered on the current readback).
    Without one, every variable gets its safety range. Variables whose readback
    is unavailable (NaN) fall back to the safety range as well — the operator
    can still see the table while connectivity is being investigated.
    """
    chosen = lookup_preset(registry, name, preset) if preset is not None else None
    overrides = chosen.variable_overrides if chosen is not None else {}
    env = load_environment(name, registry, machine.env_config())

    with env:
        variables = env.variables
        if chosen is not None and chosen.variables is not None:
            validate_variable_names(chosen.variables, env, field="preset variable")
            variables = {n: variables[n] for n in chosen.variables}
        with read_errors(name, "read current values"):
            readings = env.get_variables(list(variables.keys()))

        bounds: dict[str, ResolvedBoundsModel] = {}
        for var_name, meta in variables.items():
            current = reading_to_float(readings[var_name]) if var_name in readings else None
            if current is None:
                bounds[var_name] = ResolvedBoundsModel(
                    min=meta.min_value,
                    max=meta.max_value,
                    requested_min=meta.min_value,
                    requested_max=meta.max_value,
                    was_clamped=False,
                )
                continue
            bounds[var_name] = _resolve_one_bound(var_name, meta, overrides.get(var_name), current)
        return InitialBoundsResponse(bounds=bounds, current_values=readings)


@router.post("/{name}/resolve-bounds", response_model=ResolveBoundsResponse)
def resolve_environment_bounds(
    name: str,
    body: ResolveBoundsRequest,
    registry: EnvironmentRegistry = Depends(get_environment_registry),
) -> ResolveBoundsResponse:
    """Resolve bounds for the variables named in current_values; every override needs a matching current_value."""
    env = load_environment(name, registry)
    with env:
        variables = env.variables
        validate_variable_names(list(body.overrides.keys()), env, field="override")
        validate_variable_names(list(body.current_values.keys()), env, field="current_value")
        missing = [n for n in body.overrides if n not in body.current_values]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Missing current_value(s) for overridden variable(s): {missing}",
            )

        bounds: dict[str, ResolvedBoundsModel] = {}
        for var_name, current_value in body.current_values.items():
            bounds[var_name] = _resolve_one_bound(
                var_name, variables[var_name], body.overrides.get(var_name), current_value
            )
        return ResolveBoundsResponse(bounds=bounds)
