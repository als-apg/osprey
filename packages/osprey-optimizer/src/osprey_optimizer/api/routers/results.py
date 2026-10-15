"""Results API router.

Provides endpoints for browsing historical runs and restoring their setpoints.
"""

import hashlib
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar, cast, get_args

from fastapi import APIRouter, Depends, HTTPException, Query, status

from osprey_optimizer.api.dependencies import get_environment_registry, get_job_tracker
from osprey_optimizer.api.environment_ops import (
    apply_variables_to_environment,
    guard_environment_write,
    is_sim_target,
    load_environment,
)
from osprey_optimizer.api.job_tracker import JobTracker
from osprey_optimizer.models.api import (
    HistoricalRunDetails,
    RestorePreviewResponse,
    RestoreResponse,
    ResultInfo,
)
from osprey_optimizer.models.config import ControlTarget, machine_config
from osprey_optimizer.results import (
    build_run_details,
    load_results,
    scan_result_infos,
)
from osprey_optimizer.services.plugin_registry import EnvironmentRegistry
from osprey_optimizer.utils.numeric import is_usable_setpoint_number

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/results", tags=["results"])


@router.get("", response_model=list[ResultInfo])
def list_results() -> list[ResultInfo]:
    """List available historical runs."""
    try:
        return scan_result_infos()
    except Exception as e:
        logger.exception("Failed to list results")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to list results: {e}",
        ) from e


_T = TypeVar("_T")


def _load_run_or_404(run_id: str, loader: Callable[[str], _T]) -> _T:
    """Traversal-guard ``run_id`` and run *loader* on it, mapping errors to HTTP codes.

    404 for identifiers that escape the results root (path traversal / absolute
    paths; legitimate identifiers are relative "<env>/<run>") or for missing
    runs; 500 with detail for any other load failure.
    """
    results_root = Path("results").resolve()
    if not (results_root / run_id).resolve().is_relative_to(results_root):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Run '{run_id}' not found"
        )
    try:
        return loader(run_id)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Run '{run_id}' not found"
        ) from e
    except Exception as e:
        logger.exception("Failed to load results for '%s'", run_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load results: {e}",
        ) from e


def _resolve_restore_values(run_id: str) -> tuple[str, ControlTarget, dict[str, float]]:
    """Load a run, validate its initial snapshot, return (environment, target, tuned values).

    The target is the one the run itself drove: its starting values belong to
    that machine and to no other. Raises the same 404/409 HTTPExceptions the
    restore endpoint surfaces. Performs no machine write — shared by restore
    (POST) and restore-preview (GET).
    """
    _data, _best, config = _load_run_or_404(run_id, load_results)

    cfg = dict(config)
    snapshots = cfg.get("snapshots")
    snapshot = snapshots.get("initial") if isinstance(snapshots, dict) else None
    if not isinstance(snapshot, dict):
        snapshot = None
    environment_name = cfg.get("environment_name")
    target = cfg.get("control_target")
    tuned = cfg.get("variables")
    if not snapshot:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Run '{run_id}' has no initial snapshot",
        )
    if not environment_name or not tuned:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Run '{run_id}' config is missing environment or variables",
        )
    if target not in get_args(ControlTarget):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Run '{run_id}' config does not record the control target it ran on",
        )

    missing = [name for name in tuned if name not in snapshot]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Run '{run_id}' snapshot is missing tuned variables: {', '.join(missing)}",
        )
    # Snapshot values come straight from json.load (Any); only a finite real
    # number may become a setpoint write — the annotation alone must not vouch.
    restore_values: dict[str, float] = {}
    unusable: list[str] = []
    for name in tuned:
        value = snapshot[name]
        if is_usable_setpoint_number(value):
            restore_values[name] = float(value)
        else:
            unusable.append(name)
    if unusable:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Run '{run_id}' snapshot has non-numeric or non-finite values for: {', '.join(unusable)}",
        )
    return environment_name, cast(ControlTarget, target), restore_values


def _restore_token(values: dict[str, float]) -> str:
    """Stable hash of the values a restore would write.

    Canonical JSON (sorted keys, fixed separators) so the same value-set always
    yields the same token regardless of dict ordering. Used as the preview->restore
    handshake token; it is a footgun guard, not an auth credential.
    """
    canonical = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Two non-greedy segments, not "{run_id:path}": a greedy param would swallow the
# three-segment /restore and /restore-preview URLs regardless of declaration order.


# sync def: blocking disk I/O (load_results) + blocking env write -> FastAPI threadpool
@router.post("/{env}/{timestamp}/restore", response_model=RestoreResponse)
def restore_run_initial(
    env: str,
    timestamp: str,
    confirm_token: str | None = Query(
        default=None,
        description="Hash from a prior restore-preview; required for non-sim environments.",
    ),
    override_unhealthy: bool = Query(default=False),
    registry: EnvironmentRegistry = Depends(get_environment_registry),
    tracker: JobTracker = Depends(get_job_tracker),
) -> RestoreResponse:
    """Restore a run's tuned variables to their initial (run-start) values, on the
    target the run drove.

    A run on a real target (standin, live) requires ``confirm_token`` to match the
    hash of the values a prior ``restore-preview`` returned; a run on the virtual
    accelerator proceeds with no token.
    """
    environment_name, target, restore_values = _resolve_restore_values(f"{env}/{timestamp}")
    guard_environment_write(
        environment_name, target, tracker, registry, override_unhealthy=override_unhealthy
    )
    if not is_sim_target(target):
        expected = _restore_token(restore_values)
        if confirm_token != expected:
            raise HTTPException(
                status_code=status.HTTP_428_PRECONDITION_REQUIRED,
                detail=(
                    "preview required before restoring a real environment; "
                    "GET restore-preview first and resubmit its preview_token as confirm_token"
                ),
            )
    environment = load_environment(environment_name, registry, machine_config(target))
    apply_variables_to_environment(environment, environment_name, restore_values)
    return RestoreResponse(
        environment=environment_name,
        control_target=target,
        restored=restore_values,
        count=len(restore_values),
    )


# sync def: blocking disk I/O (load_results); no machine write
@router.get("/{env}/{timestamp}/restore-preview", response_model=RestorePreviewResponse)
def preview_run_restore(env: str, timestamp: str) -> RestorePreviewResponse:
    """Return the values a Restore would write, without writing them."""
    environment_name, target, restore_values = _resolve_restore_values(f"{env}/{timestamp}")
    return RestorePreviewResponse(
        environment=environment_name,
        control_target=target,
        values=restore_values,
        count=len(restore_values),
        preview_token=_restore_token(restore_values),
    )


@router.get("/{env}/{timestamp}", response_model=HistoricalRunDetails)
def get_result_details(env: str, timestamp: str) -> HistoricalRunDetails:
    """Get detailed data for a specific run."""
    return _load_run_or_404(f"{env}/{timestamp}", build_run_details)
