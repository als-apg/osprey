"""Redis-based job tracking for optimization jobs."""

from __future__ import annotations

import logging
import os
import time
import traceback
from types import TracebackType

from pydantic import BaseModel, ValidationError
from redis import Redis
from redis.commands.core import Script

from osprey.services.optimizer.core.state import TERMINAL_STATE_TTL_SECONDS
from osprey.services.optimizer.models.enums import JobStatus, JobType
from osprey.services.optimizer.models.exceptions import CancellationError
from osprey.services.optimizer.models.protocols import JobStateWriter
from osprey.services.optimizer.optimization.state import get_optimization_writer
from osprey.services.optimizer.utils import redis_client as _redis_client
from osprey.services.optimizer.utils.interval_thread import IntervalThread
from osprey.services.optimizer.utils.job_keys import (
    all_meta_scan_pattern,
    job_scan_pattern,
    meta_key,
)
from osprey.services.optimizer.utils.redis_state_store import (
    CREATED_STATE_TTL_SECONDS,
    RUNNING_STATE_TTL_SECONDS,
)

logger = logging.getLogger(__name__)

# Message recorded for any cancelled job, regardless of job type. JobLifecycle
# is the single owner of terminal status, so the cancel message lives here too.
CANCELLED_MESSAGE = "Cancelled by user"

# Global single-active-slot pointer. NOT per-job: it is deleted explicitly and
# must stay outside the ``job:{id}:*`` namespace so a per-job sweep can never
# match it. Every other per-job key lives under that namespace (see job_keys).
ACTIVE_JOB_KEY = "api:active_job"

# Explicitly-settable "current job" focus pointer for the UI to follow. Separate
# from ACTIVE_JOB_KEY (the auto-managed running slot): any job id may be set
# here, including a CREATED draft. get_current_job self-heals a vanished target.
CURRENT_JOB_KEY = "api:current_job"

# Shared "current environment" pointer: the single environment all clients work
# in. Lives outside the job namespace like the other api:* pointers. Switching is
# Lua-guarded against ACTIVE_JOB_KEY so it cannot move while a job drives hardware.
CURRENT_ENVIRONMENT_KEY = "api:current_environment"

# Atomic single-active-slot claim. KEYS[1]=ACTIVE_JOB_KEY, KEYS[2]=meta_key.
# ARGV[1]=job_id, ARGV[2]=running-meta JSON, ARGV[3]=ttl seconds.
# Only writes the RUNNING meta if the NX claim on the active slot succeeds, so a
# lost race leaves any pre-existing (e.g. CREATED) meta at KEYS[2] untouched.
_CLAIM_ACTIVE_JOB_LUA = """
if redis.call('SET', KEYS[1], ARGV[1], 'NX') then
    redis.call('SET', KEYS[2], ARGV[2], 'EX', tonumber(ARGV[3]))
    return 1
end
return 0
"""

# Set the current-environment pointer iff no job holds the active slot.
# KEYS[1]=ACTIVE_JOB_KEY, KEYS[2]=CURRENT_ENVIRONMENT_KEY, ARGV[1]=env name.
# Returns 1 if set, 0 if a job is active. One round-trip closes the
# check-to-write window a read-then-SET would leave open. NB: raw GET, no
# self-heal — a lingering active key (crashed worker) makes a switch reject
# until get_active_job() reaps it; erring toward reject is the safe direction.
_SET_CURRENT_ENV_LUA = """
if redis.call('GET', KEYS[1]) then
    return 0
end
redis.call('SET', KEYS[2], ARGV[1])
return 1
"""

# Heartbeat: a running job re-applies RUNNING_STATE_TTL_SECONDS across its whole
# job:{id}:* namespace every HEARTBEAT_INTERVAL_SECONDS. If the process dies
# (SIGKILL/OOM/redeploy) it stops refreshing, every key expires, and
# get_active_job() self-heals the otherwise-permanently-wedged active slot — and
# no state/cancel key leaks. The 6x margin (10s beat / 60s TTL) tolerates a brief
# Redis blip without freeing a slot out from under a live job.
HEARTBEAT_INTERVAL_SECONDS = 10


class JobMeta(BaseModel):
    """A job's tracker record — the value stored at ``job:{id}:meta``.

    Internal Redis state, not a wire contract (nothing outside this process
    reads it), so it lives here rather than in ``osprey.services.optimizer.models``. The
    JSON field names and values match what the tracker has always written:
    ``status``/``job_type`` serialize as their enum value strings.
    """

    job_id: str
    pid: int
    status: JobStatus
    created_at: float
    job_type: JobType
    completed_at: float | None = None


def _meta_json(job_id: str, pid: int, status: JobStatus, job_type: JobType) -> str:
    """Serialize a job's metadata record — the payload every registration writes."""
    return JobMeta(
        job_id=job_id,
        pid=pid,
        status=status,
        created_at=time.time(),
        job_type=job_type,
    ).model_dump_json()


class JobTracker:
    """Tracks optimization jobs using Redis for distributed state management."""

    def __init__(self) -> None:
        self._scripts: dict[str, Script] = {}

    def _get_redis(self) -> Redis[str]:
        """The shared JSON/string Redis client.

        No instance-level cache: ``get_redis_client`` is already memoized per
        process, and holding our own handle would keep a tracker built before a
        ``reset_redis_clients()`` talking to the dropped client.
        """
        return _redis_client.get_redis_client(binary=False)

    def _script(self, source: str) -> Script:
        """Lazily register a Lua script on the redis client, memoized by source.

        A concurrent init race under FastAPI's sync threadpool is benign:
        register_script is idempotent, so re-registering the same source has no
        correctness impact (worst case two handles for one server-side script).
        """
        if source not in self._scripts:
            self._scripts[source] = self._get_redis().register_script(source)
        return self._scripts[source]

    def register_job(self, job_id: str, pid: int, job_type: JobType = JobType.OPTIMIZATION) -> None:
        """Register a job as active, overwriting any current active marker.

        Used for in-place updates of an already-claimed job (e.g. writing the
        real PID after spawn). To claim the active slot for a brand-new job, use
        :meth:`claim_active_job`, which is atomic against concurrent starts.

        Args:
            job_id: Unique job identifier
            pid: Process ID of the job
            job_type: the kind of job being tracked
        """
        redis = self._get_redis()
        # Store job metadata under a heartbeat TTL so a crashed process's slot
        # self-heals once it stops refreshing (see refresh_heartbeat).
        redis.set(
            meta_key(job_id),
            _meta_json(job_id, pid, JobStatus.RUNNING, job_type),
            ex=RUNNING_STATE_TTL_SECONDS,
        )
        redis.set(ACTIVE_JOB_KEY, job_id)
        logger.info("Registered job %s with PID %s", job_id, pid)

    def register_created_job(self, job_id: str, job_type: JobType = JobType.OPTIMIZATION) -> None:
        """Register a CREATED-but-not-started job WITHOUT claiming the active slot.

        Non-reserving counterpart to :meth:`claim_active_job`: it records the job
        so status polls resolve, but leaves ``ACTIVE_JOB_KEY`` untouched so the
        single-running-job guard only fires at start. The meta carries a CREATED
        status and a created-job TTL so an abandoned create self-reclaims.
        """
        redis = self._get_redis()
        redis.set(
            meta_key(job_id),
            _meta_json(job_id, 0, JobStatus.CREATED, job_type),
            ex=CREATED_STATE_TTL_SECONDS,
        )
        logger.info("Registered CREATED job %s (slot not claimed)", job_id)

    def claim_active_job(
        self, job_id: str, pid: int, job_type: JobType = JobType.OPTIMIZATION
    ) -> bool:
        """Atomically claim the single active-job slot for a brand-new job.

        Runs one Lua script: ``SET ACTIVE_JOB_KEY job_id NX`` and, only if that
        succeeds, ``SET meta_key running-meta EX ttl`` — so a concurrent
        :meth:`get_active_job` observes a RUNNING job the instant the active key
        appears, while a *lost* race writes nothing and leaves the job's
        pre-existing (CREATED) metadata intact. Returns False if another job
        already holds the slot.
        """
        claimed = self._script(_CLAIM_ACTIVE_JOB_LUA)(
            keys=[ACTIVE_JOB_KEY, meta_key(job_id)],
            args=[
                job_id,
                _meta_json(job_id, pid, JobStatus.RUNNING, job_type),
                RUNNING_STATE_TTL_SECONDS,
            ],
        )
        if claimed == 1:
            logger.info("Claimed active slot for job %s with PID %s", job_id, pid)
            return True
        logger.info("Job %s could not claim active slot; another job is already active", job_id)
        return False

    def get_active_job(self) -> str | None:
        """Get the currently active job ID, if any.

        Returns:
            The active job_id or None if no job is active.
        """
        redis = self._get_redis()
        active_job_id = redis.get(ACTIVE_JOB_KEY)
        if not active_job_id:
            return None

        job_data = self.get_job(active_job_id)
        if job_data and job_data.status is JobStatus.RUNNING:
            return active_job_id

        redis.delete(ACTIVE_JOB_KEY)
        return None

    def set_current_job(self, job_id: str) -> None:
        """Set the UI 'current job' pointer to any job id."""
        self._get_redis().set(CURRENT_JOB_KEY, job_id)

    def get_current_job(self) -> str | None:
        """Get the current-job pointer, self-healing a vanished target.

        Returns the stored id for any still-existing job (incl. terminal). If
        the pointer is unset, or the target job's keys are gone, returns None
        (and clears a dangling pointer).
        """
        redis = self._get_redis()
        job_id = redis.get(CURRENT_JOB_KEY)
        if not job_id:
            return None
        if not self.job_exists(job_id):
            redis.delete(CURRENT_JOB_KEY)
            return None
        return job_id

    def get_current_environment(self) -> str | None:
        """Return the shared current-environment pointer, or None if unset."""
        return self._get_redis().get(CURRENT_ENVIRONMENT_KEY)

    def set_current_environment_if_idle(self, name: str) -> bool:
        """Atomically set the pointer iff no job is active. False if one is."""
        result = self._script(_SET_CURRENT_ENV_LUA)(
            keys=[ACTIVE_JOB_KEY, CURRENT_ENVIRONMENT_KEY],
            args=[name],
        )
        return bool(result == 1)

    def get_job(self, job_id: str) -> JobMeta | None:
        """Get job metadata.

        Args:
            job_id: The job identifier

        Returns:
            The job's :class:`JobMeta` or None if not found. Raises
            ``ValidationError`` on a corrupt record (mapped to a 500 by the
            ``job_status`` read seam).
        """
        redis = self._get_redis()
        data = redis.get(meta_key(job_id))
        return JobMeta.model_validate_json(data) if data else None

    def list_job_metas(self) -> list[JobMeta]:
        """Return every tracked job's metadata record (one prefix scan).

        Used by ``GET /jobs`` to surface created-but-not-started jobs, which
        deliberately do not hold the single active slot. Terminal filtering is
        the caller's job — this returns every parsable record.
        """
        redis = self._get_redis()
        keys = list(redis.scan_iter(match=all_meta_scan_pattern()))
        if not keys:
            return []
        # One MGET rather than a GET per key; a key that expired between the
        # scan and the fetch comes back as None and is skipped.
        metas = []
        for data in redis.mget(keys):
            if not data:
                continue
            try:
                metas.append(JobMeta.model_validate_json(data))
            except ValidationError:
                # One corrupt record must not sink the whole listing — the
                # per-job read seam still 500s if that job is asked for by id.
                logger.warning("Skipping unparsable job metadata record: %s", data)
        return metas

    def job_exists(self, job_id: str) -> bool:
        """Check if a job exists in the tracker.

        Args:
            job_id: The job identifier

        Returns:
            True if job exists, False otherwise.
        """
        redis = self._get_redis()
        return bool(redis.exists(meta_key(job_id)))

    def mark_completed(self, job_id: str) -> None:
        """Mark a job as completed.

        Args:
            job_id: The job identifier
        """
        redis = self._get_redis()
        job_data = self.get_job(job_id)
        if job_data and job_data.status is not JobStatus.COMPLETED:
            job_data.status = JobStatus.COMPLETED
            job_data.completed_at = time.time()
            redis.set(meta_key(job_id), job_data.model_dump_json())
            logger.info("Marked job %s as completed", job_id)

        # Sweep the job's whole job:{id}:* footprint (meta + state + cancel) under
        # the terminal TTL in one pass, so it self-reclaims. Applied on every
        # terminal pass, not just the first flip: a job cancelled mid-run keeps
        # writing state keys AFTER the API-side mark_completed, and the worker's
        # own JobLifecycle.__exit__ -> mark_completed must re-apply the TTL to
        # those late writes (the set() above also clears the meta key's TTL). Not
        # gated on job_data: once the meta key has lapsed the rest of the
        # namespace can still be alive and must be swept too (#107).
        self.expire_job(job_id, TERMINAL_STATE_TTL_SECONDS)

        active = redis.get(ACTIVE_JOB_KEY)
        if active and active == job_id:
            redis.delete(ACTIVE_JOB_KEY)

    def expire_job(self, job_id: str, ttl_seconds: int) -> None:
        """Set a TTL on every Redis key the job owns (``job:{id}:*``).

        One prefix scan reclaims metadata, state, and the cancellation token
        together. Call only once a job is terminal — these keys are unbounded
        while it runs (the heartbeat keeps :func:`meta_key` alive). The tracker's
        str client scans/expires fine even though state values are binary: SCAN
        and EXPIRE act on key names, not values, and both clients share the DB.
        """
        redis = self._get_redis()
        pipe = redis.pipeline()
        for key in redis.scan_iter(match=job_scan_pattern(job_id)):
            pipe.expire(key, ttl_seconds)
        pipe.execute()

    def refresh_heartbeat(self, job_id: str) -> None:
        """Re-apply the running TTL across the job's whole ``job:{id}:*`` namespace.

        Called periodically from the job process (see :class:`JobLifecycle`).
        Refreshing the entire namespace (one SCAN per beat — the slight per-beat
        cost) keeps not just the metadata slot but every state and cancellation
        key alive while the job runs. A process that dies (SIGKILL/OOM/redeploy)
        stops calling this, so all its keys expire after
        ``RUNNING_STATE_TTL_SECONDS``: :meth:`get_active_job` reclaims the wedged
        slot, and no state/cancel key leaks (it never reaches the terminal sweep).
        """
        self.expire_job(job_id, RUNNING_STATE_TTL_SECONDS)


def create_job_state(job_type: JobType, job_id: str) -> JobStateWriter:
    """Return a job-type-dispatched state writer fully wired to Redis.

    Thin dispatcher over the per-domain factories — tests that mock Redis
    can patch ``osprey.services.optimizer.utils.redis_client.get_redis_client``.
    """
    del job_type  # one job type today; kept so callers stay type-dispatched
    return get_optimization_writer(job_id)


class JobLifecycle:
    """Context manager for background job lifecycle management.

    Handles registration in JobTracker, setting initial state,
    capturing crashes with full stack traces, and de-registering
    the job upon completion or failure.

    Works with both optimization and script jobs via ``job_type``.
    """

    writer: JobStateWriter  # set in __enter__

    def __init__(
        self,
        job_id: str,
        *,
        job_type: JobType = JobType.OPTIMIZATION,
    ) -> None:
        self.job_id = job_id
        self._job_type = job_type
        self.tracker = JobTracker()
        # Runs on its own thread so a long blocking call on the main thread (e.g.
        # a multi-second EPICS batch read) can't stall the heartbeat and make a
        # live job look dead. swallow_errors: one failed refresh must never kill
        # the thread — it is all that keeps this job's keys (and its active slot)
        # alive. Started in __enter__, stopped in __exit__.
        self._heartbeat = IntervalThread(
            self._refresh_heartbeat,
            interval=HEARTBEAT_INTERVAL_SECONDS,
            name=f"job-heartbeat-{job_id}",
            swallow_errors=True,
        )

    def __enter__(self) -> JobLifecycle:
        """Start the job lifecycle."""
        self.writer = create_job_state(self._job_type, self.job_id)
        self.tracker.register_job(self.job_id, os.getpid(), job_type=self._job_type)
        self.writer.set_status(JobStatus.RUNNING)
        self._heartbeat.start()

        logger.info(
            "Lifecycle started for job %s (type=%s, PID: %s)",
            self.job_id,
            self._job_type,
            os.getpid(),
        )
        return self

    def _refresh_heartbeat(self) -> None:
        """One heartbeat beat: re-apply the running TTL across the job's keys."""
        self.tracker.refresh_heartbeat(self.job_id)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """End the job lifecycle, classifying the outcome into terminal status.

        Single owner of the writer's terminal status for every job type. Runners
        run their body and let the outcome propagate: no exception → COMPLETED,
        a CancellationError → CANCELLED, anything else → ERROR. Backend-specific
        exceptions are translated to domain errors at their own adapter (e.g.
        xopt failures become ``OptimizerError`` in ``xopt_backend``), so this
        classifier stays generic — it uses ``str(exc)`` for the error message
        (the full traceback still goes to the log for diagnosis). Does not
        suppress: the exception still propagates.
        """
        # Stop the heartbeat first so it can't re-shorten the metadata TTL after
        # mark_completed() applies the long terminal TTL below.
        self._heartbeat.stop(timeout=HEARTBEAT_INTERVAL_SECONDS + 1)

        try:
            if exc_type is None:
                self.writer.set_status(JobStatus.COMPLETED)
                logger.info("Job %s finished successfully", self.job_id)
            elif issubclass(exc_type, CancellationError):
                self.writer.set_status(JobStatus.CANCELLED)
                self.writer.set_error_message(CANCELLED_MESSAGE)
                logger.info("Job %s cancelled by user", self.job_id)
            else:
                error_msg = str(exc_val) or exc_type.__name__
                logger.error(
                    "Job %s failed with unhandled exception:\n%s",
                    self.job_id,
                    traceback.format_exc(),
                )
                self.writer.set_status(JobStatus.ERROR)
                self.writer.set_error_message(error_msg)

            # mark_completed() sweeps the job's entire job:{id}:* footprint
            # (meta + state + cancellation) under the terminal TTL in one pass —
            # the heartbeat is already stopped above so it can't re-shorten it,
            # and the status/message just written above are covered by the sweep.
            self.tracker.mark_completed(self.job_id)

        except Exception:
            logger.exception("Error during lifecycle cleanup for job %s", self.job_id)
