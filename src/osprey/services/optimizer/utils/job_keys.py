"""Single source of truth for per-job Redis key naming.

Every key a job owns lives under the ``job:{job_id}:`` namespace, so one prefix
scan (:func:`job_scan_pattern`) reclaims a finished job's entire footprint —
metadata, state, and cancellation token — in a single sweep. Keeping the naming
here (rather than spread across the tracker, the state stores, and the
cancellation token) is what makes that single sweep possible: nothing can drift
outside the namespace.

The one deliberate exception is the global active-slot pointer
(``api:active_job`` in :mod:`osprey.services.optimizer.api.job_tracker`): it is not
per-job, is deleted explicitly, and must never be matched by a per-job scan.
"""

from __future__ import annotations

_JOB_NAMESPACE = "job:"


def job_scan_pattern(job_id: str) -> str:
    """``SCAN MATCH`` pattern for every key owned by one job (terminal sweep)."""
    return f"{_JOB_NAMESPACE}{job_id}:*"


def meta_key(job_id: str) -> str:
    """Tracker metadata key (JSON job record, str client)."""
    return f"{_JOB_NAMESPACE}{job_id}:meta"


def state_prefix(job_id: str) -> str:
    """Prefix for the job's state store; it appends ``:{suffix}`` per field.

    Optimization and script jobs share this prefix — a job is exactly one type
    (chosen by ``create_job_state(job_type, ...)``) and ``job_id`` is unique, so
    the two never collide.
    """
    return f"{_JOB_NAMESPACE}{job_id}:state"


def cancellation_key(job_id: str) -> str:
    """Cancellation-token hash key (cancelled/paused/waiting, str client)."""
    return f"{_JOB_NAMESPACE}{job_id}:cancel"


def config_key(job_id: str) -> str:
    """Stored run spec for a CREATED-but-not-started job (str client).

    Optimization jobs store the ``OptimizationConfig`` JSON; script jobs store a
    ``{script_id, environment_name, parameters}`` JSON blob.
    """
    return f"{_JOB_NAMESPACE}{job_id}:config"


def all_meta_scan_pattern() -> str:
    """``SCAN MATCH`` pattern for every job's metadata key (cross-job listing).

    Job ids are UUIDs (no colons), so ``job:*:meta`` matches exactly the meta
    keys — never a ``:state``/``:cancel``/``:config`` key.
    """
    return f"{_JOB_NAMESPACE}*:meta"
