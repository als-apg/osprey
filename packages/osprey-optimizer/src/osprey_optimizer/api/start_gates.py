"""Start-time verification gates for optimization jobs (issue #2).

Run between the environment-selection guard and the worker spawn: the machine
must pass its settings table, and the active variables' setpoints must still
lie inside the resolved bounds reviewed at submit. Failures are 412 —
retryable after fixing the machine or re-submitting — and never mutate job
state.
"""

from __future__ import annotations

from fastapi import HTTPException, status

from osprey_optimizer.api.environment_ops import read_errors
from osprey_optimizer.models.config import OptimizationConfig
from osprey_optimizer.models.exceptions import MachineSettingsError
from osprey_optimizer.models.protocols import EnvironmentProtocol
from osprey_optimizer.optimization.coercion import reading_to_float


def run_start_gates(config: OptimizationConfig, env: EnvironmentProtocol, *, env_name: str) -> None:
    """Machine-settings check + drift guard; 412 aggregating every violation.

    409 when the stored config predates the resolved-draft contract (it cannot
    be verified — re-create the job).
    """
    try:
        config.require_resolved()
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Stored config is not a resolved draft ({e}); re-create the job.",
        ) from e

    active = config.variables or []
    overrides = config.variable_overrides or {}
    failures: list[str] = []
    with read_errors(env_name, "verify machine settings"), env:
        try:
            env.check_machine_settings()
        except MachineSettingsError as e:
            failures.extend(e.failures)
        readings = env.get_setpoints(active)
        for name in active:
            override = overrides[name]
            vmin, vmax = override.min, override.max
            if vmin is None or vmax is None:  # require_resolved holds; keep the types honest
                failures.append(f"{name}: stored override incomplete")
                continue
            value = reading_to_float(readings[name]) if name in readings else None
            if value is None:
                failures.append(f"{name}: setpoint unavailable")
            elif not vmin <= value <= vmax:
                failures.append(
                    f"{name}: current value {value} outside resolved bounds [{vmin}, {vmax}]"
                )
    if failures:
        raise HTTPException(
            status_code=status.HTTP_412_PRECONDITION_FAILED,
            detail="Machine not ready to start: " + "; ".join(failures),
        )
