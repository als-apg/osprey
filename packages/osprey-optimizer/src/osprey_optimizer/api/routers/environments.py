"""Environments API router.

Provides endpoints for discovering environments and reading/writing variables.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from osprey_optimizer.api.dependencies import (
    Machine,
    get_environment_registry,
    get_job_tracker,
    require_machine,
)
from osprey_optimizer.api.environment_ops import (
    apply_variables_to_environment,
    environment_details,
    guard_environment_write,
    load_environment,
    probe_environment_health,
    read_errors,
    validate_observable_names,
    validate_variable_names,
    write_errors,
)
from osprey_optimizer.api.job_tracker import JobTracker
from osprey_optimizer.models.api import (
    EnvironmentDetails,
    EnvironmentInfo,
    EnvironmentStatusResponse,
    EvaluateRequest,
    EvaluateResponse,
    SetVariablesRequest,
)
from osprey_optimizer.models.protocols import PVReading
from osprey_optimizer.services.plugin_registry import EnvironmentRegistry

router = APIRouter(prefix="/environments", tags=["environments"])


@router.get("", response_model=list[EnvironmentInfo])
def list_environments(
    registry: EnvironmentRegistry = Depends(get_environment_registry),
) -> list[EnvironmentInfo]:
    """List available environments."""
    return registry.list_environments()


@router.get("/{name}", response_model=EnvironmentDetails)
def get_environment_details(
    name: str,
    registry: EnvironmentRegistry = Depends(get_environment_registry),
) -> EnvironmentDetails:
    """Get environment metadata and variable definitions."""
    return environment_details(name, registry)


@router.get("/{name}/status", response_model=EnvironmentStatusResponse)
def check_environment_status(
    name: str,
    machine: Machine = Depends(require_machine),
    registry: EnvironmentRegistry = Depends(get_environment_registry),
) -> EnvironmentStatusResponse:
    """Check if the environment is reachable on the named target."""
    return probe_environment_health(name, registry, machine.target)


@router.put("/{name}/variables", status_code=status.HTTP_204_NO_CONTENT)
def set_environment_variables(
    name: str,
    body: SetVariablesRequest,
    machine: Machine = Depends(require_machine),
    registry: EnvironmentRegistry = Depends(get_environment_registry),
    tracker: JobTracker = Depends(get_job_tracker),
) -> None:
    """Apply variable settings to an environment on the named target."""
    guard_environment_write(
        name, machine.target, tracker, registry, override_unhealthy=body.override_unhealthy
    )
    env = load_environment(name, registry, machine.env_config())
    apply_variables_to_environment(env, name, body.variables)


@router.get("/{name}/variables", response_model=dict[str, PVReading])
def get_environment_variables(
    name: str,
    names: Annotated[list[str], Query()],
    machine: Machine = Depends(require_machine),
    registry: EnvironmentRegistry = Depends(get_environment_registry),
) -> dict[str, PVReading]:
    """Get current variable readings from an environment.

    Each reading carries the native value plus connected/timestamp metadata;
    an unreachable PV comes back as a reading with ``connected: false``.
    """
    env = load_environment(name, registry, machine.env_config())
    with env:
        validate_variable_names(names, env)
        with read_errors(name, "get variables"):
            return env.get_variables(names)


@router.get("/{name}/observables", response_model=dict[str, PVReading])
def get_environment_observables(
    name: str,
    names: Annotated[list[str], Query()],
    machine: Machine = Depends(require_machine),
    registry: EnvironmentRegistry = Depends(get_environment_registry),
) -> dict[str, PVReading]:
    """Read current observable readings from the environment.

    Observables are objective measurements; each crosses the wire as a native
    :class:`PVReading`.
    """
    env = load_environment(name, registry, machine.env_config())
    with env:
        validate_observable_names(names, env)
        with read_errors(name, "get observables"):
            return env.get_observables(names)


@router.get("/{name}/pv", response_model=dict[str, PVReading])
def get_environment_pvs(
    name: str,
    names: Annotated[list[str], Query()],
    machine: Machine = Depends(require_machine),
    registry: EnvironmentRegistry = Depends(get_environment_registry),
) -> dict[str, PVReading]:
    """Raw caget of arbitrary PV names through the env's CA context.

    The endpoint does not validate names against the env's metadata —
    callers can read any channel reachable from this env's interface
    (typical use: facility-wide readings such as beam current).

    Returns one :class:`PVReading` per requested name, preserving the native
    EPICS type: scalars as a number/string, waveforms as a JSON array, plus
    metadata (units, precision, alarm severity/status, timestamp, enum
    strings). A NaN value is reported as ``value: null``.

    Per-PV failures (bad name, IOC down, gateway timeout) come through as a
    reading with ``connected: false`` so callers see a stable response shape
    and can distinguish "the read failed" from "the value is null."

    The endpoint only returns 5xx for catastrophic / unexpected errors:
    - **404** — environment name unknown.
    - **503** — the env's CA worker thread didn't respond (catastrophic;
      typically means the worker is wedged or never started).
    - **500** — anything else unexpected.
    """
    env = load_environment(name, registry, machine.env_config())
    with read_errors(name, "read PVs"), env:
        return env.read_pvs(names)


@router.post("/{name}/evaluate", response_model=EvaluateResponse)
def evaluate_environment(
    name: str,
    body: EvaluateRequest,
    machine: Machine = Depends(require_machine),
    registry: EnvironmentRegistry = Depends(get_environment_registry),
    tracker: JobTracker = Depends(get_job_tracker),
) -> EvaluateResponse:
    """Set variables and read observables in one transaction, on the named target."""
    guard_environment_write(
        name, machine.target, tracker, registry, override_unhealthy=body.override_unhealthy
    )
    env = load_environment(name, registry, machine.env_config())
    with env:
        validate_observable_names(body.observables, env)
        validate_variable_names(list(body.variables.keys()), env)
        with write_errors(name, "evaluate"):
            obs_values = env.evaluate(body.variables, body.observables)
            readbacks = env.get_variables(list(body.variables.keys()))
        return EvaluateResponse(
            variable_readbacks=readbacks,
            observables=obs_values,
        )
