"""Optimization API router.

Provides the endpoint for creating optimization jobs. Reads go through the
unified ``GET /jobs/{id}`` endpoint.
"""

import logging

from fastapi import APIRouter, Depends

from osprey_optimizer.api.dependencies import get_environment_registry, get_job_tracker
from osprey_optimizer.api.job_status import read_job_status
from osprey_optimizer.api.job_tracker import JobTracker
from osprey_optimizer.api.optimization_lifecycle import create_optimization_job
from osprey_optimizer.models.api import JobResponse
from osprey_optimizer.models.config import OptimizationConfig
from osprey_optimizer.services.plugin_registry import EnvironmentRegistry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/optimizations", tags=["optimizations"])


@router.post("", response_model=JobResponse)
def create_optimization(
    request: OptimizationConfig,
    tracker: JobTracker = Depends(get_job_tracker),
    registry: EnvironmentRegistry = Depends(get_environment_registry),
) -> JobResponse:
    """Create an optimization job from a JSON config. Does not start it.

    Returns the new ``job_id`` with status ``created``; start it with
    ``POST /jobs/{job_id}/start``.
    """
    job_id = create_optimization_job(request, tracker, registry)
    # Point the server-side current-job at the new draft, atomically with
    # creation, so the web follow-poll can't read a stale pointer and revert the
    # selection before Start (which would start the wrong job).
    tracker.set_current_job(job_id)
    return read_job_status(tracker, job_id)
