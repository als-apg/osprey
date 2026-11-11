"""Shared control-action helpers for the job/optimization routers."""

from __future__ import annotations

import logging
import multiprocessing
import time
from collections.abc import Callable
from multiprocessing.process import BaseProcess
from typing import NoReturn

from fastapi import HTTPException, status

from osprey_optimizer.api.job_tracker import JobTracker
from osprey_optimizer.models.enums import JobStatus
from osprey_optimizer.models.protocols import JobStateWriter
from osprey_optimizer.optimization.state import get_optimization_writer
from osprey_optimizer.utils import redis_client as _redis_client
from osprey_optimizer.utils.cancellation import CancellationToken, get_cancellation_token
from osprey_optimizer.utils.job_keys import config_key, meta_key, state_prefix
from osprey_optimizer.utils.process import reap_finished_children, shutdown_grace_seconds
from osprey_optimizer.utils.redis_state_store import CREATED_STATE_TTL_SECONDS
from osprey_optimizer.utils.serialization import serialize_obj

logger = logging.getLogger(__name__)

# The workers this API process launched, for stop_launched_jobs. Process-local
# on purpose: a worker is the child of the API process that spawned it, so under
# ``--workers N`` each API process stops only its own.
_launched: dict[str, BaseProcess] = {}

# After the shutdown grace: how long a terminated worker gets to exit before it
# is killed, and how long the kill gets to take effect. With the grace they
# bound a shutdown (see stop_launched_jobs).
TERMINATE_WAIT_SECONDS = 1.0
KILL_WAIT_SECONDS = 1.0

SHUTDOWN_KILLED_MESSAGE = (
    "Cancelled: the service shut down and the worker did not stop within the shutdown "
    "grace, so it was killed. No partial results were saved, and setpoints are as the "
    "worker left them, possibly part-way through a write; check the machine before the "
    "next run."
)

# Write a draft's config, with the draft's TTL, only while its status is still
# CREATED. KEYS[1]=the job's pickled status key, KEYS[2]=its config key,
# KEYS[3]=its meta key. ARGV[1]=pickled CREATED, ARGV[2]=config JSON, ARGV[3]=ttl.
# Byte equality on the pickle, as the status CAS in core.state does. Returns 1 on
# write, 0 when the job left CREATED, -1 when its status is gone (expired).
_STORE_IF_CREATED_LUA = """
local status = redis.call('GET', KEYS[1])
if not status then
    return -1
end
if status ~= ARGV[1] then
    return 0
end
redis.call('SET', KEYS[2], ARGV[2], 'EX', tonumber(ARGV[3]))
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[3]))
redis.call('EXPIRE', KEYS[3], tonumber(ARGV[3]))
return 1
"""


def require_created_or_409(job_id: str, action: str, current: JobStatus | None) -> None:
    """409 unless ``current`` is CREATED — the shared create/start two-phase gate.

    ``action`` names the attempted transition ("update"/"start") in the detail.
    Centralizes the wording so the job lifecycles can't drift.
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
    payload_json: str,
    make_writer: Callable[[str], JobStateWriter],
) -> None:
    """Persist a job's stored config and register it as CREATED (non-reserving).

    The create tail of a job lifecycle: store the
    config so ``/jobs/{id}/start`` can spawn the worker later, set CREATED, and
    register with the tracker.
    """
    make_writer(job_id).set_status(JobStatus.CREATED)
    tracker.register_created_job(job_id)
    store_created_config(job_id, payload_json)


def store_created_config(job_id: str, payload_json: str) -> None:
    """Write a CREATED job's stored config and normalize its TTL, in one step.

    Shared by create and edit. A draft owns its meta, status and config keys;
    all three get the created-job window together, so a draft that is never
    started self-expires as one unit (the state writer stamps the short RUNNING
    TTL on the status key; this lifts it). The write happens only while the
    status is still CREATED: 409 when a start or cancel landed while the draft
    was prepared, so a started job's config is never replaced; 404 when the
    draft expired.
    """
    stored = _redis_client.get_redis_client(binary=False).register_script(_STORE_IF_CREATED_LUA)(
        keys=[f"{state_prefix(job_id)}:status", config_key(job_id), meta_key(job_id)],
        args=[serialize_obj(JobStatus.CREATED), payload_json, CREATED_STATE_TTL_SECONDS],
    )
    if stored == -1:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found")
    if stored != 1:
        require_created_or_409(job_id, "update", get_optimization_writer(job_id).get_status())


def draft_changed_409(job_id: str) -> HTTPException:
    """The 409 for a start whose draft changed since its caller was shown it."""
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail=(
            f"Job {job_id}'s draft changed since it was shown; read it again "
            f"(GET /jobs/{job_id}/config) before starting it."
        ),
    )


def load_stored_config(job_id: str, *, missing_status: int, missing_detail: str) -> str:
    """Fetch a job's stored config payload, raising ``missing_status`` when absent.

    Callers pass their contract's status/wording (read: 404, start: 409) and
    parse the returned raw JSON themselves.
    """
    raw = _redis_client.as_str(_redis_client.get_redis_client(binary=False).get(config_key(job_id)))
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
    """Signal cancellation, then mark the job CANCELLING (or CANCELLED for a draft).

    Job status is a dual-writer field: the worker sets its own terminal status
    when a run ends. The status is re-read and swapped by compare-and-set — if
    the worker reached a terminal state in the gap, raise 409 rather than
    resurrecting a finished job. A repeated cancel of a CANCELLING job is a no-op.

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
    if current is JobStatus.CANCELLING:
        return

    # A CREATED draft has no worker: nothing else will ever end it, so the
    # cancel does, and releases its footprint under the terminal TTL.
    if current is JobStatus.CREATED:
        if not writer.set_status_cas(current, JobStatus.CANCELLED):
            raise_control_conflict_409(job_id, "cancel", None, writer)
        tracker.mark_completed(job_id)
        return

    # A launched job is ended by its worker, not here: the worker may be mid-write
    # when the cancel lands, so it keeps the active slot until it has stopped, and
    # its JobLifecycle sets CANCELLED and releases the slot. Until then the job is
    # CANCELLING, and no other job or manual write can start. A worker that dies
    # instead stops its heartbeat, and its keys and the slot expire. CAS from the
    # status just read: a lost CAS means the worker drove it terminal in the gap.
    if not writer.set_status_cas(current, JobStatus.CANCELLING):
        raise_control_conflict_409(job_id, "cancel", None, writer)


def launch_job(
    *,
    tracker: JobTracker,
    job_id: str,
    config_json: str,
    make_writer: Callable[[str], JobStateWriter],
    spawn_target: Callable[..., None],
    spawn_args: tuple[object, ...],
) -> int | None:
    """Claim the single active-job slot, start the draft, and spawn its worker.

    The concurrency-critical launch protocol, with rollback on each failure:

    1. Atomically ``SET NX`` claim the active slot (409 if a concurrent start
       won it) — the friendly ``get_active_job`` guard the caller runs first is
       not atomic across workers, so this is the real gate. Claiming also
       pre-registers the job at PID 0 so status polls don't 404 before spawn.
    2. Move the job CREATED -> RUNNING in one step, only while its stored config
       is still ``config_json``, the exact value the caller read and gated
       (:meth:`JobTracker.start_claimed_job`). A cancel or edit that landed
       first wins: the claim is undone and the start answers 409. Once the job
       is RUNNING an edit answers 409 and a cancel is the worker's to finish.
       503 + release the slot on a Redis failure.
    3. Reap finished children to prevent zombie accumulation.
    4. Spawn a daemon ``"spawn"``-context ``Process`` (500 + release on OSError;
       spawn avoids epics/libca fork issues).
    5. Register the real PID once the OS assigns one, and keep the process so
       a service shutdown can stop it (:func:`stop_launched_jobs`).

    Returns the spawned PID (or None if the OS did not assign one). Domain
    validation, the friendly active-job 409 label, and any pre-spawn resource
    setup (e.g. results-dir init) stay in the caller.
    """
    if not tracker.claim_active_job(job_id, 0):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A job is already running. Cancel it or wait for completion.",
        )

    try:
        started = tracker.start_claimed_job(job_id, config_json)
    except Exception as e:
        logger.exception("Failed to pre-initialize job state")
        tracker.mark_completed(job_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to initialize job state (Redis unavailable): {e}",
        ) from e
    if not started:
        current = make_writer(job_id).get_status()
        if current.is_terminal:
            # A cancel ended the draft before the claim, which overwrote the
            # meta the cancel left: put back what the cancel wrote.
            tracker.mark_completed(job_id)
        require_created_or_409(job_id, "start", current)
        raise draft_changed_409(job_id)

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

    for finished in [jid for jid, p in _launched.items() if not p.is_alive()]:
        del _launched[finished]
    _launched[job_id] = process

    if process.pid:
        tracker.register_job(job_id, process.pid)

    return process.pid


def stop_launched_jobs(tracker: JobTracker, grace_s: float | None = None) -> None:
    """Stop every job this process launched that is still running (service shutdown).

    1. Cancel each one through :func:`cancel_with_reread`, exactly as
       ``POST /jobs/{id}/cancel`` does: the token is set and the job goes
       CANCELLING.
    2. Wait, up to ``grace_s`` in all (``OPTIMIZER_SHUTDOWN_GRACE_S`` by
       default), for each worker to exit on its own. One that does has
       recorded CANCELLED and saved its partial results, as on any cancel.
    3. Terminate a worker still running (its signal handler turns that into
       the same cancel), wait ``TERMINATE_WAIT_SECONDS``, then kill one still
       alive and wait ``KILL_WAIT_SECONDS``.
    4. End each job whose worker is gone but left it non-terminal, and release
       its slot now rather than when the heartbeat TTL lapses.

    Nothing here writes a setpoint. A stopped run holds the machine where it
    is: restoring it is the operator's call, through the results restore
    endpoint. The bound is the grace plus the two waits, plus Redis round trips.
    """
    if grace_s is None:
        grace_s = shutdown_grace_seconds()
    running = {jid: process for jid, process in _launched.items() if process.is_alive()}
    if not running:
        return
    logger.warning(
        "Service shutting down: cancelling %d running job(s), %.1f s grace: %s",
        len(running),
        grace_s,
        ", ".join(running),
    )

    for job_id in running:
        try:
            cancel_with_reread(
                job_id, get_optimization_writer(job_id), get_cancellation_token(job_id), tracker
            )
        except HTTPException as e:
            # 409: the worker drove the job terminal in the gap; it exits on its own.
            logger.info("Shutdown cancel of job %s not needed: %s", job_id, e.detail)
        except Exception:
            # Still wait for, then end, the worker below: a shutdown is not optional.
            logger.exception("Shutdown cancel of job %s failed", job_id)

    _join_all(list(running.values()), grace_s)

    # Every stuck worker at once, so the bound holds whatever their number.
    stuck = {jid: process for jid, process in running.items() if process.is_alive()}
    for job_id, process in stuck.items():
        logger.warning("Job %s did not stop within %.1f s; terminating it", job_id, grace_s)
        process.terminate()
    _join_all(list(stuck.values()), TERMINATE_WAIT_SECONDS)
    survivors = [process for process in stuck.values() if process.is_alive()]
    for process in survivors:
        logger.warning("Worker %s survived terminate; killing it", process.pid)
        process.kill()
    _join_all(survivors, KILL_WAIT_SECONDS)

    for job_id, process in running.items():
        if process.is_alive():
            logger.error("Job %s's worker is still alive after a kill; its slot expires", job_id)
            continue
        try:
            _end_stopped_job(job_id, process, killed=job_id in stuck, tracker=tracker)
        except Exception:
            logger.exception("Could not record the end of job %s; its slot expires", job_id)
        _launched.pop(job_id, None)


def _join_all(processes: list[BaseProcess], timeout: float) -> None:
    """Join ``processes`` within one shared ``timeout``, not ``timeout`` each."""
    deadline = time.monotonic() + timeout
    for process in processes:
        process.join(max(0.0, deadline - time.monotonic()))


def _end_stopped_job(
    job_id: str, process: BaseProcess, *, killed: bool, tracker: JobTracker
) -> None:
    """Record the end of a job whose worker has exited, unless the worker did.

    A worker that stopped through the cancel path set its own terminal status;
    that is left as is. One that was killed, or exited without recording an
    outcome, leaves the job non-terminal: it is ended here with the reason.
    The status CAS keeps a terminal status from being overwritten.
    """
    writer = get_optimization_writer(job_id)
    if killed:
        new, message = JobStatus.CANCELLED, SHUTDOWN_KILLED_MESSAGE
    else:
        new = JobStatus.ERROR
        message = (
            f"The worker exited (code {process.exitcode}) without recording an outcome while "
            "the service shut down; check the machine before the next run."
        )
    # Re-read on a lost CAS: only a concurrent API cancel (RUNNING -> CANCELLING)
    # can move a job whose worker is gone.
    for _ in range(3):
        current = writer.get_status()
        if current.is_terminal:
            break
        if writer.set_status_cas(current, new):
            writer.set_error_message(message)
            logger.warning("Job %s ended %s on shutdown: %s", job_id, new.value, message)
            break
    tracker.mark_completed(job_id)
