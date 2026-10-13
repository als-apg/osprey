"""Read and switch the shared current-environment pointer.

Reads return cached health to keep polling cheap; switching runs a fresh probe
and is rejected while a job is active (the optimizer owns the hardware). Selection
never blocks on health — an unreachable env is selectable and returns red status.
"""

from __future__ import annotations

import json
import logging

from fastapi import HTTPException, status

from osprey.services.optimizer.api.environment_ops import (
    environment_details,
    probe_environment_health,
)
from osprey.services.optimizer.api.job_tracker import JobTracker
from osprey.services.optimizer.models.api import (
    CurrentEnvironmentResponse,
    EnvironmentStatusResponse,
)
from osprey.services.optimizer.services.plugin_registry import EnvironmentRegistry
from osprey.services.optimizer.utils import redis_client as _redis_client
from osprey.services.optimizer.utils.job_keys import config_key

logger = logging.getLogger(__name__)

HEALTH_CACHE_KEY = "api:current_environment:health"
HEALTH_CACHE_TTL_SECONDS = 5


def build_current_environment_response(
    tracker: JobTracker, registry: EnvironmentRegistry
) -> CurrentEnvironmentResponse:
    name = tracker.get_current_environment()
    if name is None:
        return CurrentEnvironmentResponse()
    return CurrentEnvironmentResponse(
        name=name,
        details=environment_details(name, registry),
        status=_cached_or_probe_health(name, registry),
    )


def select_environment(
    name: str, tracker: JobTracker, registry: EnvironmentRegistry
) -> CurrentEnvironmentResponse:
    if not tracker.set_current_environment_if_idle(name):
        active = tracker.get_active_job()
        active_env = read_job_environment_name(active) if active else None
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot switch environment: a job is running on '{active_env}'.",
        )
    return CurrentEnvironmentResponse(
        name=name,
        details=environment_details(name, registry),
        status=_probe_and_cache_health(name, registry),
    )


def read_job_environment_name(job_id: str) -> str | None:
    """Environment name from a job's stored config (optimization or script)."""
    raw = _redis_client.get_redis_client(binary=False).get(config_key(job_id))
    if raw is None:
        return None
    value = json.loads(raw).get("environment_name")
    return value if isinstance(value, str) else None


def _probe_and_cache_health(name: str, registry: EnvironmentRegistry) -> EnvironmentStatusResponse:
    health = probe_environment_health(name, registry)
    redis = _redis_client.get_redis_client(binary=False)
    redis.set(HEALTH_CACHE_KEY, health.model_dump_json(), ex=HEALTH_CACHE_TTL_SECONDS)
    return health


def _cached_or_probe_health(name: str, registry: EnvironmentRegistry) -> EnvironmentStatusResponse:
    raw = _redis_client.get_redis_client(binary=False).get(HEALTH_CACHE_KEY)
    if raw is not None:
        cached = EnvironmentStatusResponse.model_validate_json(raw)
        if cached.name == name:
            return cached
    return _probe_and_cache_health(name, registry)
