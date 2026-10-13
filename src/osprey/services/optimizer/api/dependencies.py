"""FastAPI dependency callables for shared singletons.

Each function reads from ``request.app.state``, which is populated by the
lifespan handler in ``main.py``.  Use them with ``Depends()`` in endpoint
signatures so that routers stay decoupled from initialization logic.
"""

from fastapi import Depends, Request

from osprey.services.optimizer.api.job_status import load_job_meta
from osprey.services.optimizer.api.job_tracker import JobMeta, JobTracker
from osprey.services.optimizer.services.plugin_registry import EnvironmentRegistry


def get_job_tracker(request: Request) -> JobTracker:
    return request.app.state.job_tracker  # type: ignore[no-any-return]


def require_job_meta(job_id: str, tracker: JobTracker = Depends(get_job_tracker)) -> JobMeta:
    """Resolve the path job's metadata once (404 if unknown, 500 if corrupt).

    Endpoints under ``/jobs/{job_id}`` inject this instead of re-checking
    existence and re-reading the record: FastAPI caches it for the request, so
    the type/creation-time checks they used to make one call each now share a
    single Redis read.
    """
    return load_job_meta(tracker, job_id)


def get_environment_registry(request: Request) -> EnvironmentRegistry:
    return request.app.state.environment_registry  # type: ignore[no-any-return]
