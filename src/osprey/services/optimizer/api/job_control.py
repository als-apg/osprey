"""Shared control-action helpers for the job/optimization routers."""

from __future__ import annotations

import logging
import multiprocessing
from collections.abc import Callable
from typing import NoReturn

from fastapi import HTTPException, status

from osprey.services.optimizer.api.job_tracker import JobTracker
from osprey.services.optimizer.models.enums import JobStatus, JobType
from osprey.services.optimizer.models.protocols import JobStateWriter
from osprey.services.optimizer.utils import redis_client as _redis_client
from osprey.services.optimizer.utils.cancellation import CancellationToken
from osprey.services.optimizer.utils.job_keys import config_key
from osprey.services.optimizer.utils.process import reap_finished_children
from osprey.services.optimizer.utils.redis_state_store import CREATED_STATE_TTL_SECONDS

logger = logging.getLogger(__name__)


def require_created_or_409(job_id: str, action: str, current: JobStatus | None) -> None:
    """409 unless ``current`` is CREATED — the shared create/start two-phase gate.

    ``action`` names the attempted transition ("update"/"start") in the detail.
    Centralizes the wording so the optimization and script lifecycles can't drift.
    """
    if current is not JobStatus.CREATED:
        current_label = current.value if current else None
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot {action} job {job_id}: expected status 'created', got '{current_label}'.",
        )


def persist_created_job(
    *,
    tracker: JobTracker,
    job_id: str,
    job_type: JobType,
    payload_json: str,
    make_writer: Callable[[str], JobStateWriter],
) -> None:
    """Persist a job's stored config and register it as CREATED (non-reserving).

    The create tail shared by the optimization and script lifecycles: store the
    config so ``/jobs/{id}/start`` can spawn the worker later, set CREATED, and
    register with the tracker.
    """
    make_writer(job_id).set_status(JobStatus.CREATED)
    tracker.register_created_job(job_id, job_type=job_type)
    store_created_config(tracker, job_id, payload_json)


def store_created_config(tracker: JobTracker, job_id: str, payload_json: str) -> None:
    """Write a CREATED job's stored config and normalize its namespace TTL.

    The TTL is lifted across the whole ``job:{id}:*`` footprint (meta + state +
    config) so a created-but-never-started job self-expires as one unit. The
    state writer stamps the short RUNNING TTL on the status key inline; this
    lifts it to the created-job window to match meta and config.
    """
    client = _redis_client.get_redis_client(binary=False)
    client.set(config_key(job_id), payload_json, ex=CREATED_STATE_TTL_SECONDS)
    tracker.expire_job(job_id, CREATED_STATE_TTL_SECONDS)


def load_stored_config(job_id: str, *, missing_status: int, missing_detail: str) -> str:
    """Fetch a job's stored config payload, raising ``missing_status`` when absent.

    Callers pass their contract's status/wording (read: 404, start: 409) and
    parse the returned raw JSON themselves.
    """
    raw = _redis_client.get_redis_client(binary=False).get(config_key(job_id))
    if raw is None:
        raise HTTPException(status_code=missing_status, detail=missing_detail)
    return raw


def raise_control_conflict_409(
    job_id: str,
    action: str,
    required_source: JobStatus | None,
    writer: JobStateWriter,
) -> NoReturn:
    """Re-read the status and raise the shared control-conflict 409.

    Called when a control action's status CAS loses — because the worker drove
    the job terminal in the gap, or because the job was never in the state the
    action requires. The re-read reports the real current status; for pause/resume
    ``required_source`` names the state the action needs so the client knows what
    to do. Centralizes the wording so pause/resume and cancel can't drift.
    """
    latest = writer.get_status()
    if required_source is not None:
        detail = f"Cannot {action} job {job_id}: must be {required_source.value}, but job is {latest.value}"
    else:
        detail = f"Cannot {action} job {job_id}: job is {latest.value}"
    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)


def cancel_with_reread(
    job_id: str,
    writer: JobStateWriter,
    token: CancellationToken,
    tracker: JobTracker,
) -> None:
    """Signal cancellation, then flip the job to CANCELLED guarding against a lost update.

    Job status is a dual-writer field with no compare-and-set: the worker sets
    its own terminal status (COMPLETED/ERROR) when a run ends. Re-read the status
    immediately before writing CANCELLED — if the worker reached a terminal state
    in the gap, raise 409 rather than resurrecting a finished job (clients would
    poll it as live forever) and clobbering the terminal TTL it just applied.

    Shared by the optimization control path and the generic jobs cancel endpoint
    so the protocol has a single source of truth. Callers pass the already
    resolved ``writer``/``token`` to stay dependency-injection friendly.
    """
    token.cancel()

    current = writer.get_status()
    if current.is_terminal:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot cancel job {job_id}: already {current.value}",
        )

    # CAS from the status we just read: only flip to CANCELLED if the worker
    # hasn't driven it terminal in the gap. A lost CAS means the worker won —
    # 409 rather than clobber its terminal status/TTL.
    if not writer.set_status_cas(current, JobStatus.CANCELLED):
        raise_control_conflict_409(job_id, "cancel", None, writer)

    # mark_completed() sweeps the job's whole job:{id}:* footprint under the
    # terminal TTL — including a wedged/dead job's state keys, whose process
    # can't run its own lifecycle cleanup.
    tracker.mark_completed(job_id)


def launch_job(
    *,
    tracker: JobTracker,
    job_id: str,
    job_type: JobType,
    make_writer: Callable[[str], JobStateWriter],
    spawn_target: Callable[..., None],
    spawn_args: tuple[object, ...],
) -> int | None:
    """Claim the single active-job slot and spawn the worker subprocess.

    The concurrency-critical launch protocol shared by the optimization and
    script run endpoints, with rollback on each failure:

    1. Atomically ``SET NX`` claim the active slot (409 if a concurrent start
       won it) — the friendly ``get_active_job`` guard the caller runs first is
       not atomic across workers, so this is the real gate. Claiming also
       pre-registers the job at PID 0 so status polls don't 404 before spawn.
    2. Pre-initialize the state writer to RUNNING (503 + release the slot on a
       Redis failure) so polling doesn't 404.
    3. Reap finished children to prevent zombie accumulation.
    4. Spawn a daemon ``"spawn"``-context ``Process`` (500 + release on OSError;
       spawn avoids epics/libca fork issues).
    5. Register the real PID once the OS assigns one.

    Returns the spawned PID (or None if the OS did not assign one). Domain
    validation, the friendly active-job 409 label, and any pre-spawn resource
    setup (e.g. results-dir init) stay in the caller.
    """
    if not tracker.claim_active_job(job_id, 0, job_type=job_type):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A job is already running. Cancel it or wait for completion.",
        )

    try:
        writer = make_writer(job_id)
        writer.set_status(JobStatus.RUNNING)
    except Exception as e:
        logger.exception("Failed to pre-initialize job state")
        tracker.mark_completed(job_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to initialize job state (Redis unavailable): {e}",
        ) from e

    reap_finished_children()

    try:
        ctx = multiprocessing.get_context("spawn")
        process = ctx.Process(target=spawn_target, args=spawn_args, daemon=True)
        process.start()
    except OSError as e:
        logger.exception("Failed to start job process")
        tracker.mark_completed(job_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start job process: {e}",
        ) from e

    if process.pid:
        tracker.register_job(job_id, process.pid, job_type=job_type)

    return process.pid
