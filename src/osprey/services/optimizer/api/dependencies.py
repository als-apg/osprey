"""FastAPI dependency callables for shared singletons.

Each function reads from ``request.app.state``, which is populated by the
lifespan handler in ``main.py``.  Use them with ``Depends()`` in endpoint
signatures so that routers stay decoupled from initialization logic.
"""

from dataclasses import dataclass
from typing import Any

from fastapi import Depends, Query, Request

from osprey.services.optimizer.api.job_status import load_job_meta
from osprey.services.optimizer.api.job_tracker import JobMeta, JobTracker
from osprey.services.optimizer.models.config import ControlTarget, ExecutionMode, machine_config
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


@dataclass(frozen=True)
class Machine:
    """The machine a request reaches, as the request named it."""

    target: ControlTarget
    execution_mode: ExecutionMode | None = None

    def env_config(self) -> dict[str, Any]:
        return machine_config(self.target, self.execution_mode)


def require_machine(
    target: ControlTarget = Query(
        description="The machine this call reaches: va, standin or live. No default."
    ),
    execution_mode: ExecutionMode | None = Query(
        default=None, description="readonly: reads only; every write is refused."
    ),
) -> Machine:
    """Every endpoint that reaches a machine names it; there is no ambient target."""
    return Machine(target=target, execution_mode=execution_mode)
