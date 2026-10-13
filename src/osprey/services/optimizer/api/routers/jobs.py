"""Jobs API router.

Provides unified endpoints for monitoring and controlling any job, regardless
of type.
"""

import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status

from osprey.services.optimizer.api.dependencies import (
    get_environment_registry,
    get_job_tracker,
    require_job_meta,
)
from osprey.services.optimizer.api.environment_ops import guard_environment_start, load_environment
from osprey.services.optimizer.api.job_control import cancel_with_reread, raise_control_conflict_409
from osprey.services.optimizer.api.job_status import read_job_status
from osprey.services.optimizer.api.job_tracker import JobMeta, JobTracker, create_job_state
from osprey.services.optimizer.api.optimization_lifecycle import (
    read_optimization_config,
    start_optimization_job,
    update_optimization_config,
)
from osprey.services.optimizer.api.optimization_status import build_optimization_response
from osprey.services.optimizer.api.start_gates import run_start_gates
from osprey.services.optimizer.models.api import CurrentJobRequest, CurrentJobResponse, JobResponse
from osprey.services.optimizer.models.config import OptimizationConfig, machine_config
from osprey.services.optimizer.models.enums import JobStatus, JobType
from osprey.services.optimizer.optimization.state import get_optimization_writer
from osprey.services.optimizer.services.plugin_registry import EnvironmentRegistry
from osprey.services.optimizer.utils.cancellation import get_cancellation_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])


def _cancel_job(job_id: str, job_type: JobType, tracker: JobTracker) -> None:
    """Cancel a job by setting the cancellation token and updating state."""
    writer = create_job_state(job_type, job_id)
    token = get_cancellation_token(job_id)
    cancel_with_reread(job_id, writer, token, tracker)


def _require_optimization(meta: JobMeta, action: str) -> None:
    """409 if the job's type does not support ``action`` (404 already handled).

    start/pause/resume need the optimization lifecycle (two-phase start and
    ``wait_if_paused``). The job's existence is the ``require_job_meta``
    dependency's job, so this only checks the type.
    """
    if meta.job_type is not JobType.OPTIMIZATION:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"job type '{meta.job_type.value}' does not support {action}",
        )


def _control_action(
    job_id: str, action: Literal["pause", "resume"], tracker: JobTracker
) -> JobResponse:
    """Pause/resume an optimization job via a single status CAS.

    pause is only legal from RUNNING, resume only from PAUSED. The CAS from that
    required source is the sole gate — it rejects a terminal job (worker drove it
    done in the gap) and a wrong-source job (e.g. CREATED) alike. The cancellation
    token is toggled only after the swap commits, so a rejected action never
    poisons a job's token.
    """
    logger.info("%s request for job %s", action.capitalize(), job_id)
    try:
        token = get_cancellation_token(job_id)
        state_writer = get_optimization_writer(job_id)

        if action == "pause":
            required_source, new_status = JobStatus.RUNNING, JobStatus.PAUSED
        else:
            required_source, new_status = JobStatus.PAUSED, JobStatus.RUNNING

        if not state_writer.set_status_cas(required_source, new_status):
            raise_control_conflict_409(job_id, action, required_source, state_writer)

        if action == "pause":
            token.pause()
        else:
            token.resume()
        return read_job_status(tracker, job_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("%s action failed", action.capitalize())
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e


@router.get("", response_model=list[JobResponse])
def list_jobs(
    tracker: JobTracker = Depends(get_job_tracker),
) -> list[JobResponse]:
    """List every non-terminal job (created, submitted, running, paused).

    Created jobs do not hold the single active slot, so this scans all job
    metadata rather than reading the active pointer. Terminal jobs (which linger
    under their terminal TTL) are excluded.
    """
    responses: list[JobResponse] = []
    try:
        for meta in tracker.list_job_metas():
            job_id = meta.job_id
            if not job_id:
                continue
            try:
                response = read_job_status(tracker, job_id)
            except HTTPException:
                logger.warning("Skipping job %s in listing: metadata unreadable", job_id)
                continue
            if not response.status.is_terminal:
                responses.append(response)
        return responses
    except Exception as e:
        logger.exception("Failed to list jobs")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to read job state: {e}",
        ) from e


@router.get("/current", response_model=CurrentJobResponse)
def read_current_job(
    tracker: JobTracker = Depends(get_job_tracker),
) -> CurrentJobResponse:
    """Return the UI 'current job' pointer (id or null).

    Declared before /{job_id} so 'current' is not parsed as a job id.
    """
    return CurrentJobResponse(job_id=tracker.get_current_job())


@router.put("/current", response_model=CurrentJobResponse)
def put_current_job(
    body: CurrentJobRequest,
    tracker: JobTracker = Depends(get_job_tracker),
) -> CurrentJobResponse:
    """Set the UI 'current job' pointer to an existing job (404 if unknown)."""
    if not tracker.job_exists(body.job_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {body.job_id} not found"
        )
    tracker.set_current_job(body.job_id)
    return CurrentJobResponse(job_id=body.job_id)


@router.get("/{job_id}", response_model=JobResponse)
def get_job_status(
    job_id: str,
    detail: Literal["summary", "full"] | None = Query(None),
    tracker: JobTracker = Depends(get_job_tracker),
    meta: JobMeta = Depends(require_job_meta),
) -> JobResponse:
    """Get the status of any job; ``detail=summary|full`` enriches optimizations.

    Default (no ``detail``) returns the basic status for any job type.
    """
    try:
        if meta.job_type is JobType.OPTIMIZATION and detail is not None:
            return build_optimization_response(job_id, detail, tracker)
        response = read_job_status(tracker, job_id)
        if response.status.is_terminal:
            tracker.mark_completed(job_id)
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error reading job state for %s", job_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to read job state: {e}",
        ) from e


@router.post("/{job_id}/cancel", response_model=JobResponse)
def cancel_job(
    job_id: str,
    tracker: JobTracker = Depends(get_job_tracker),
    meta: JobMeta = Depends(require_job_meta),
) -> JobResponse:
    """Cancel any running job."""
    logger.info("Cancel request for job %s", job_id)

    current = create_job_state(meta.job_type, job_id).get_status()
    if current.is_terminal:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot cancel job {job_id}: already {current.value}",
        )

    try:
        _cancel_job(job_id, meta.job_type, tracker)
        return read_job_status(tracker, job_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Cancel job failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e


@router.post("/{job_id}/start", response_model=JobResponse)
def start_job(
    job_id: str,
    override_unhealthy: bool = Query(default=False),
    tracker: JobTracker = Depends(get_job_tracker),
    registry: EnvironmentRegistry = Depends(get_environment_registry),
    _meta: JobMeta = Depends(require_job_meta),  # 404 for an unknown job
) -> JobResponse:
    """Start a CREATED job's worker on the target its config names."""
    config = read_optimization_config(job_id)  # 404 when no config is stored
    env_name = config.environment_name
    target = config.control_target
    if env_name is None or target is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Job {job_id} cannot start: its config must name an environment_name and a "
                "control_target (va, standin or live)."
            ),
        )
    guard_environment_start(env_name, target, registry, override_unhealthy=override_unhealthy)
    env_config = {
        **(config.environment_config or {}),
        **machine_config(target, config.execution_mode),
    }
    run_start_gates(config, load_environment(env_name, registry, env_config), env_name=env_name)
    return start_optimization_job(job_id, tracker, registry)


@router.put("/{job_id}/config", response_model=JobResponse)
def update_job_config(
    job_id: str,
    request: OptimizationConfig,
    tracker: JobTracker = Depends(get_job_tracker),
    registry: EnvironmentRegistry = Depends(get_environment_registry),
    meta: JobMeta = Depends(require_job_meta),
) -> JobResponse:
    """Replace a CREATED optimization job's stored config (live-sync)."""
    _require_optimization(meta, "config update")
    return update_optimization_config(job_id, request, tracker, registry)


@router.get("/{job_id}/config", response_model=OptimizationConfig)
def get_job_config(
    job_id: str,
    meta: JobMeta = Depends(require_job_meta),
) -> OptimizationConfig:
    """Return an optimization job's stored config (read-side of live-sync)."""
    _require_optimization(meta, "config read")
    return read_optimization_config(job_id)


@router.post("/{job_id}/pause", response_model=JobResponse)
def pause_job(
    job_id: str,
    tracker: JobTracker = Depends(get_job_tracker),
    meta: JobMeta = Depends(require_job_meta),
) -> JobResponse:
    """Pause a running optimization job."""
    _require_optimization(meta, "pause")
    return _control_action(job_id, "pause", tracker)


@router.post("/{job_id}/resume", response_model=JobResponse)
def resume_job(
    job_id: str,
    tracker: JobTracker = Depends(get_job_tracker),
    meta: JobMeta = Depends(require_job_meta),
) -> JobResponse:
    """Resume a paused optimization job."""
    _require_optimization(meta, "resume")
    return _control_action(job_id, "resume", tracker)
