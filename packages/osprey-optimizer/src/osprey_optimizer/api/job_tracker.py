"""Redis-based job tracking for optimization jobs."""

from __future__ import annotations

import logging
import os
import time
import traceback
from enum import Enum
from types import TracebackType

from pydantic import BaseModel, ValidationError
from redis.commands.core import Script

from osprey_optimizer.models.enums import JobStatus
from osprey_optimizer.models.exceptions import CancellationError
from osprey_optimizer.utils import redis_client as _redis_client
from osprey_optimizer.utils.interval_thread import IntervalThread
from osprey_optimizer.utils.job_keys import (
    all_meta_scan_pattern,
    config_key,
    heartbeat_key,
    job_scan_pattern,
    meta_key,
    state_prefix,
)
from osprey_optimizer.utils.redis_client import StrRedis, as_str
from osprey_optimizer.utils.redis_state_store import (
    CREATED_STATE_TTL_SECONDS,
    RUNNING_STATE_TTL_SECONDS,
    TERMINAL_STATE_TTL_SECONDS,
)
from osprey_optimizer.utils.serialization import serialize_obj

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

# Atomic single-active-slot claim, for a draft only. KEYS[1]=ACTIVE_JOB_KEY,
# KEYS[2]=meta_key. ARGV[1]=job_id, ARGV[2]=running-meta JSON, ARGV[3]=ttl
# seconds, ARGV[4]=the CREATED status value. Returns -1 when the job has a
# record that is not a draft's (it has started or ended), 0 when another job
# holds the slot, and writes nothing in either case; else takes the slot, writes
# the RUNNING meta and returns 1. A lost race leaves any pre-existing (e.g.
# CREATED) meta at KEYS[2] untouched.
_CLAIM_ACTIVE_JOB_LUA = """
local meta = redis.call('GET', KEYS[2])
if meta and cjson.decode(meta)['status'] ~= ARGV[4] then
    return -1
end
if redis.call('SET', KEYS[1], ARGV[1], 'NX') then
    redis.call('SET', KEYS[2], ARGV[2], 'EX', tonumber(ARGV[3]))
    return 1
end
return 0
"""

# Update a claimed job's RUNNING meta (its real PID) only while it still holds
# the active slot. KEYS[1]=ACTIVE_JOB_KEY, KEYS[2]=meta_key. ARGV[1]=job_id,
# ARGV[2]=running-meta JSON, ARGV[3]=ttl seconds. Never writes the slot itself:
# a job that lost it (its claim was released or expired, and another job may now
# hold it) must not take it back. Never re-creates a record that lapsed: a job
# whose meta is gone is left to its worker's refusal, the record's only writer.
_REGISTER_IF_ACTIVE_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] and redis.call('EXISTS', KEYS[2]) == 1 then
    redis.call('SET', KEYS[2], ARGV[2], 'EX', tonumber(ARGV[3]))
    return 1
end
return 0
"""

# A spawned worker registers itself, in one step: only while its job still holds
# the active slot and its record still exists. KEYS: ACTIVE_JOB_KEY, meta_key,
# heartbeat_key. ARGV: job_id, running-meta JSON, ttl seconds. Returns 0 when
# the slot does not name the job, -1 when its record is gone (its keys lapsed
# before the worker started), else writes the RUNNING meta and the heartbeat key,
# which hands the keys' upkeep from the launcher to the worker's heartbeat, and
# returns 1. Never re-creates a lost record and never writes the slot.
_REGISTER_WORKER_LUA = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then
    return 0
end
if redis.call('EXISTS', KEYS[2]) == 0 then
    return -1
end
redis.call('SET', KEYS[2], ARGV[2], 'EX', tonumber(ARGV[3]))
redis.call('SET', KEYS[3], '1', 'EX', tonumber(ARGV[3]))
return 1
"""

# The launcher's keep-alive of a job whose worker has not registered yet.
# KEYS[1]=meta_key, KEYS[2]=heartbeat_key, KEYS[3..]=every key the job owns.
# ARGV[1]=ttl seconds, ARGV[2]=the RUNNING status value. Returns 0 when the
# worker's heartbeat has taken over or the job is no longer running (a late tick
# never shortens the terminal TTL), -1 when the record is gone, else re-applies
# the running TTL to every key of the job and returns 1.
_KEEP_CLAIM_ALIVE_LUA = """
if redis.call('EXISTS', KEYS[2]) == 1 then
    return 0
end
local meta = redis.call('GET', KEYS[1])
if not meta then
    return -1
end
if cjson.decode(meta)['status'] ~= ARGV[2] then
    return 0
end
for i = 3, #KEYS do
    redis.call('EXPIRE', KEYS[i], tonumber(ARGV[1]))
end
return 1
"""

# The end of a job, in one step: its outcome, its record, its lifetime and the
# slot. KEYS[1]=meta_key, KEYS[2]=the job's pickled status key, KEYS[3]=its
# pickled error-message key, KEYS[4]=ACTIVE_JOB_KEY, KEYS[5]=CURRENT_JOB_KEY,
# KEYS[6..]=every key the job owns. ARGV[1]=job_id, ARGV[2]=pickled terminal
# status, ARGV[3]=pickled message or "" for none, ARGV[4]=terminal ttl,
# ARGV[5]=the end time, ARGV[6]=pickled status the job must hold or "" for any,
# ARGV[7]=meta JSON to create a lapsed record with or "" to leave it lapsed,
# ARGV[8]=the COMPLETED record status value, ARGV[9..]=every pickled terminal
# status.
# Returns -1 when the record is gone and is not to be created, -2 when the job
# does not hold the expected status, 0 when the job has already ended (all three
# write nothing), 1 when it wrote the status, 2 when the status was already
# terminal and only the record was completed. A terminal status and its message
# are never overwritten, an end is never re-dated or extended, and a slot or a
# pointer another job holds is never touched; the pointer is set only when the
# record is created and none is set.
_END_JOB_LUA = """
local meta = redis.call('GET', KEYS[1])
if not meta and ARGV[7] == '' then
    return -1
end
local status = redis.call('GET', KEYS[2])
if ARGV[6] ~= '' and status ~= ARGV[6] then
    return -2
end
local created = not meta
if created then
    meta = ARGV[7]
end
local terminal = false
for i = 9, #ARGV do
    if status == ARGV[i] then
        terminal = true
    end
end
local record = cjson.decode(meta)
if not created and terminal and record['status'] == ARGV[8] then
    return 0
end
if not terminal then
    redis.call('SET', KEYS[2], ARGV[2])
    if ARGV[3] ~= '' then
        redis.call('SET', KEYS[3], ARGV[3])
    end
end
record['status'] = ARGV[8]
if record['completed_at'] == nil or record['completed_at'] == cjson.null then
    record['completed_at'] = tonumber(ARGV[5])
end
redis.call('SET', KEYS[1], cjson.encode(record))
local ttl = tonumber(ARGV[4])
for i = 1, 3 do
    redis.call('EXPIRE', KEYS[i], ttl)
end
for i = 6, #KEYS do
    redis.call('EXPIRE', KEYS[i], ttl)
end
if redis.call('GET', KEYS[4]) == ARGV[1] then
    redis.call('DEL', KEYS[4])
end
if created then
    redis.call('SET', KEYS[5], ARGV[1], 'NX')
end
if terminal then
    return 2
end
return 1
"""

# Start a job that has just claimed the slot: CREATED -> RUNNING only while its
# stored config is still the exact JSON the start checked. KEYS[1]=the job's
# pickled status key, KEYS[2]=its config key, KEYS[3]=ACTIVE_JOB_KEY,
# KEYS[4]=meta_key. ARGV[1]=pickled CREATED, ARGV[2]=pickled RUNNING, ARGV[3]=the
# config JSON, ARGV[4]=running ttl, ARGV[5]=job_id, ARGV[6]=CREATED-meta JSON,
# ARGV[7]=created ttl. On a miss (a cancel or an edit won) the claim is undone in
# the same step: the slot is released if this job holds it, and a draft still
# CREATED gets its CREATED meta back. Returns 1 on start, 0 on a miss.
_START_CLAIMED_JOB_LUA = """
local status = redis.call('GET', KEYS[1])
if status == ARGV[1] and redis.call('GET', KEYS[2]) == ARGV[3] then
    redis.call('SET', KEYS[1], ARGV[2], 'EX', tonumber(ARGV[4]))
    return 1
end
if redis.call('GET', KEYS[3]) == ARGV[5] then
    redis.call('DEL', KEYS[3])
end
if status == ARGV[1] then
    redis.call('SET', KEYS[4], ARGV[6], 'EX', tonumber(ARGV[7]))
end
return 0
"""

# Heartbeat: a running job re-applies RUNNING_STATE_TTL_SECONDS across its whole
# job:{id}:* namespace every HEARTBEAT_INTERVAL_SECONDS. If the process dies
# (SIGKILL/OOM/redeploy) it stops refreshing, unless the launcher's watch ends
# the job first (job_control.end_exited_worker_job): every key expires, and
# get_active_job() self-heals the otherwise-permanently-wedged active slot — and
# no state/cancel key leaks. The 6x margin (10s beat / 60s TTL) tolerates a brief
# Redis blip without freeing a slot out from under a live job. Until the worker
# registers, the launcher keeps the keys alive on the same interval (see
# job_control.keep_launch_alive).
HEARTBEAT_INTERVAL_SECONDS = 10


class JobSlotLostError(RuntimeError):
    """A spawned job found that it no longer holds the single active slot."""


class JobRecordLostError(RuntimeError):
    """A spawned job found its job record gone before it started."""


class WorkerRegistration(Enum):
    """The outcome of a spawned worker registering itself."""

    REGISTERED = "registered"
    SLOT_LOST = "slot_lost"
    RECORD_LOST = "record_lost"


class JobEnd(Enum):
    """The outcome of ending a job (:meth:`JobTracker.end_job`)."""

    ENDED = 1
    """The status, the record, its lifetime and the slot were written."""
    RECORD_COMPLETED = 2
    """The status was already terminal; the record, its lifetime and the slot were written."""
    ALREADY_ENDED = 0
    """The job had already ended; nothing was written."""
    RECORD_GONE = -1
    """The job's record had lapsed; nothing was written."""
    NOT_EXPECTED = -2
    """The job did not hold the expected status; nothing was written."""


class JobMeta(BaseModel):
    """A job's tracker record — the value stored at ``job:{id}:meta``.

    Internal Redis state, not a wire contract (nothing outside this process
    reads it), so it lives here rather than in ``osprey_optimizer.models``. The
    JSON field names and values match what the tracker has always written:
    ``status`` serializes as its enum value string. Records written before the
    ``job_type`` field was dropped still parse: pydantic ignores the extra key.
    """

    job_id: str
    pid: int
    status: JobStatus
    created_at: float
    completed_at: float | None = None


def _meta_json(job_id: str, pid: int, status: JobStatus) -> str:
    """Serialize a job's metadata record — the payload every registration writes."""
    return JobMeta(
        job_id=job_id,
        pid=pid,
        status=status,
        created_at=time.time(),
    ).model_dump_json()


class JobTracker:
    """Tracks optimization jobs using Redis for distributed state management."""

    def __init__(self) -> None:
        self._scripts: dict[str, Script] = {}

    def _get_redis(self) -> StrRedis:
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

    def register_job(self, job_id: str, pid: int) -> bool:
        """Record the real PID of a job that holds the active slot.

        Used by the launcher for an in-place update of an already-claimed job
        (writing the PID after spawn). It never takes the slot: to claim it for
        a brand-new job, use :meth:`claim_active_job`, which is atomic against
        concurrent starts. The worker registers through :meth:`register_worker`.

        Args:
            job_id: Unique job identifier
            pid: Process ID of the job

        Returns:
            True if the job still holds the slot and was updated; False if it
            does not, or its record is gone; nothing was written.
        """
        # The meta carries a heartbeat TTL so a crashed process's slot
        # self-heals once it stops refreshing (see refresh_heartbeat).
        updated = self._script(_REGISTER_IF_ACTIVE_LUA)(
            keys=[ACTIVE_JOB_KEY, meta_key(job_id)],
            args=[
                job_id,
                _meta_json(job_id, pid, JobStatus.RUNNING),
                RUNNING_STATE_TTL_SECONDS,
            ],
        )
        if updated != 1:
            logger.warning(
                "Job %s does not hold the active slot; PID %s not registered", job_id, pid
            )
            return False
        logger.info("Registered job %s with PID %s", job_id, pid)
        return True

    def register_worker(self, job_id: str, pid: int) -> WorkerRegistration:
        """Register a spawned worker, only while its job holds the slot and has a record.

        On success the RUNNING meta carries the worker's PID and the heartbeat
        key is set, so the launcher's keep-alive hands over to the worker's
        heartbeat. A job that lost the slot, or whose record lapsed before the
        worker started, is refused and the registration writes nothing.
        """
        result = self._script(_REGISTER_WORKER_LUA)(
            keys=[ACTIVE_JOB_KEY, meta_key(job_id), heartbeat_key(job_id)],
            args=[
                job_id,
                _meta_json(job_id, pid, JobStatus.RUNNING),
                RUNNING_STATE_TTL_SECONDS,
            ],
        )
        if result == 1:
            logger.info("Registered job %s with PID %s", job_id, pid)
            return WorkerRegistration.REGISTERED
        if result == -1:
            logger.warning("Job %s has no job record; worker PID %s not registered", job_id, pid)
            return WorkerRegistration.RECORD_LOST
        logger.warning("Job %s does not hold the active slot; PID %s not registered", job_id, pid)
        return WorkerRegistration.SLOT_LOST

    def keep_claim_alive(self, job_id: str) -> bool:
        """Re-apply the running TTL to a launched job's keys until its worker registers.

        Returns True while the keys are still the launcher's to keep; False once
        the worker's heartbeat has taken over, the job is no longer running, or
        its record is gone. Atomic against :meth:`end_job`, so a late call
        never shortens the terminal TTL.
        """
        redis = self._get_redis()
        keys = list(redis.scan_iter(match=job_scan_pattern(job_id)))
        kept = self._script(_KEEP_CLAIM_ALIVE_LUA)(
            keys=[meta_key(job_id), heartbeat_key(job_id), *keys],
            args=[RUNNING_STATE_TTL_SECONDS, JobStatus.RUNNING.value],
        )
        return bool(kept == 1)

    def end_job(
        self,
        job_id: str,
        status: JobStatus,
        message: str | None,
        *,
        expect: JobStatus | None = None,
        create_if_missing: bool = False,
    ) -> JobEnd:
        """End a job in one step: its outcome, its record, its lifetime and the slot.

        Every side that ends a job calls this, and it is one Redis script: the
        terminal ``status`` and ``message`` (none when ``None``), the record
        marked COMPLETED with its end time, every key of the job under the
        terminal TTL, and the active slot released when it names this job.

        Invariants:

        - A terminal status and its message are never overwritten: a job whose
          status is already terminal keeps it, and only its record is completed.
        - An end is written once: a job already ended is left as it is, never
          re-dated and its lifetime never extended.
        - A lapsed record is never re-created, unless ``create_if_missing``:
          then it is written as a record that ended now, and the current-job
          pointer names it when none is set.
        - A slot or a pointer another job holds is never touched.
        - With ``expect``, nothing is written unless the job holds that status.

        Raises:
            ValueError: ``status`` is not terminal.
        """
        if not status.is_terminal:
            raise ValueError(f"A job ends with a terminal status, not {status.value!r}")
        redis = self._get_redis()
        keys = list(redis.scan_iter(match=job_scan_pattern(job_id)))
        now = time.time()
        created = (
            JobMeta(
                job_id=job_id,
                pid=os.getpid(),
                status=JobStatus.COMPLETED,
                created_at=now,
                completed_at=now,
            ).model_dump_json()
            if create_if_missing
            else ""
        )
        result = self._script(_END_JOB_LUA)(
            keys=[
                meta_key(job_id),
                f"{state_prefix(job_id)}:status",
                f"{state_prefix(job_id)}:error_message",
                ACTIVE_JOB_KEY,
                CURRENT_JOB_KEY,
                *keys,
            ],
            args=[
                job_id,
                serialize_obj(status),
                serialize_obj(message) if message is not None else "",
                TERMINAL_STATE_TTL_SECONDS,
                now,
                serialize_obj(expect) if expect is not None else "",
                created,
                JobStatus.COMPLETED.value,
                *(serialize_obj(terminal) for terminal in JobStatus if terminal.is_terminal),
            ],
        )
        outcome = JobEnd(int(result))
        if outcome is JobEnd.ENDED:
            logger.info("Ended job %s as %s", job_id, status.value)
        elif outcome is JobEnd.RECORD_COMPLETED:
            logger.info("Completed the record of job %s, which had already ended", job_id)
        return outcome

    def register_created_job(self, job_id: str) -> None:
        """Register a CREATED-but-not-started job WITHOUT claiming the active slot.

        Non-reserving counterpart to :meth:`claim_active_job`: it records the job
        so status polls resolve, but leaves ``ACTIVE_JOB_KEY`` untouched so the
        single-running-job guard only fires at start. The meta carries a CREATED
        status and a created-job TTL so an abandoned create self-reclaims.
        """
        redis = self._get_redis()
        redis.set(
            meta_key(job_id),
            _meta_json(job_id, 0, JobStatus.CREATED),
            ex=CREATED_STATE_TTL_SECONDS,
        )
        logger.info("Registered CREATED job %s (slot not claimed)", job_id)

    def claim_active_job(self, job_id: str, pid: int) -> bool:
        """Atomically claim the single active-job slot for a draft.

        Runs one Lua script: ``SET ACTIVE_JOB_KEY job_id NX`` and, only if that
        succeeds, ``SET meta_key running-meta EX ttl`` — so a concurrent
        :meth:`get_active_job` observes a RUNNING job the instant the active key
        appears, while a *lost* race writes nothing and leaves the job's
        pre-existing (CREATED) metadata intact. A job whose record is not a
        draft's (it has started or ended) is refused before the slot is tried,
        and nothing is written. Returns False when another job holds the slot or
        the job is not a draft.

        A slot left by a job that died without releasing it (its keys expired
        with its heartbeat) is cleared first, by :meth:`get_active_job`; the
        NX claim alone would refuse every start after such a crash.
        """
        self.get_active_job()
        claimed = self._script(_CLAIM_ACTIVE_JOB_LUA)(
            keys=[ACTIVE_JOB_KEY, meta_key(job_id)],
            args=[
                job_id,
                _meta_json(job_id, pid, JobStatus.RUNNING),
                RUNNING_STATE_TTL_SECONDS,
                JobStatus.CREATED.value,
            ],
        )
        if claimed == 1:
            logger.info("Claimed active slot for job %s with PID %s", job_id, pid)
            return True
        if claimed == -1:
            logger.info("Job %s is not a draft; the active slot was not claimed", job_id)
            return False
        logger.info("Job %s could not claim active slot; another job is already active", job_id)
        return False

    def start_claimed_job(self, job_id: str, config_json: str) -> bool:
        """Move a job that holds the slot from CREATED to RUNNING, in one step.

        Only while its status is still CREATED and its stored config is still
        ``config_json``, the exact value the start read. Otherwise (a cancel or
        an edit landed first) nothing starts and the claim is undone: the slot
        is released if this job holds it, and a still-CREATED draft gets its
        CREATED meta back. Returns True when the job is now RUNNING.
        """
        started = self._script(_START_CLAIMED_JOB_LUA)(
            keys=[
                f"{state_prefix(job_id)}:status",
                config_key(job_id),
                ACTIVE_JOB_KEY,
                meta_key(job_id),
            ],
            args=[
                serialize_obj(JobStatus.CREATED),
                serialize_obj(JobStatus.RUNNING),
                config_json,
                RUNNING_STATE_TTL_SECONDS,
                job_id,
                _meta_json(job_id, 0, JobStatus.CREATED),
                CREATED_STATE_TTL_SECONDS,
            ],
        )
        return bool(started == 1)

    def get_active_job(self) -> str | None:
        """Get the currently active job ID, if any.

        Returns:
            The active job_id or None if no job is active.
        """
        redis = self._get_redis()
        active_job_id = as_str(redis.get(ACTIVE_JOB_KEY))
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
        job_id = as_str(redis.get(CURRENT_JOB_KEY))
        if not job_id:
            return None
        if not self.job_exists(job_id):
            redis.delete(CURRENT_JOB_KEY)
            return None
        return job_id

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

    def expire_job(self, job_id: str, ttl_seconds: int) -> None:
        """Set a TTL on every Redis key the job owns (``job:{id}:*``).

        One prefix scan covers metadata, state, and the cancellation token
        together. The tracker's str client scans/expires fine even though state
        values are binary: SCAN and EXPIRE act on key names, not values, and
        both clients share the DB.
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
        stops calling this, so, unless the launcher's watch ends the job first
        (``job_control.end_exited_worker_job``), all its keys expire after
        ``RUNNING_STATE_TTL_SECONDS``: :meth:`get_active_job` reclaims the wedged
        slot, and no state/cancel key leaks (it never reaches the terminal sweep).
        """
        self.expire_job(job_id, RUNNING_STATE_TTL_SECONDS)


class JobLifecycle:
    """Context manager for background job lifecycle management.

    Handles registration in JobTracker, setting initial state,
    capturing crashes with full stack traces, and de-registering
    the job upon completion or failure.
    """

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
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
        """Start the job lifecycle.

        The launcher already claimed the slot and set RUNNING before spawning
        this process, and a pause or cancel may have landed during the spawn:
        the status is left as found, so neither is lost.

        Raises:
            JobSlotLostError: The job no longer holds the active slot, so another
                job may be driving the machine; this one must not start.
            JobRecordLostError: The job's record lapsed before the worker
                started; the job ends with a terminal error record and this one
                must not start.
        """
        registration = self.tracker.register_worker(self.job_id, os.getpid())
        if registration is WorkerRegistration.SLOT_LOST:
            message = (
                f"Job {self.job_id} lost the active-job slot before it started; "
                "it wrote nothing to the machine."
            )
            self.tracker.end_job(self.job_id, JobStatus.ERROR, message)
            raise JobSlotLostError(message)
        if registration is WorkerRegistration.RECORD_LOST:
            message = "The job's record expired before its worker started."
            self.tracker.end_job(self.job_id, JobStatus.ERROR, message, create_if_missing=True)
            raise JobRecordLostError(message)
        self._heartbeat.start()

        logger.info(
            "Lifecycle started for job %s (PID: %s)",
            self.job_id,
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

        Single owner of a run's terminal status, written through
        :meth:`JobTracker.end_job`, which keeps a terminal status already
        recorded. Runners run their body and let the outcome propagate: no
        exception → COMPLETED, a CancellationError → CANCELLED, anything else →
        ERROR. Backend-specific
        exceptions are translated to domain errors at their own adapter (e.g.
        xopt failures become ``OptimizerError`` in ``xopt_backend``), so this
        classifier stays generic — it uses ``str(exc)`` for the error message
        (the full traceback still goes to the log for diagnosis). Does not
        suppress: the exception still propagates.
        """
        # Stop the heartbeat first so it can't re-shorten the TTL after end_job()
        # applies the terminal TTL below.
        self._heartbeat.stop(timeout=HEARTBEAT_INTERVAL_SECONDS + 1)

        status: JobStatus
        message: str | None
        if exc_type is None:
            status, message = JobStatus.COMPLETED, None
            logger.info("Job %s finished successfully", self.job_id)
        elif issubclass(exc_type, CancellationError):
            status, message = JobStatus.CANCELLED, CANCELLED_MESSAGE
            logger.info("Job %s cancelled by user", self.job_id)
        else:
            status, message = JobStatus.ERROR, str(exc_val) or exc_type.__name__
            logger.error(
                "Job %s failed with unhandled exception:\n%s",
                self.job_id,
                traceback.format_exc(),
            )

        try:
            # One step: the outcome, the record, every key of the job under the
            # terminal TTL, and the slot.
            self.tracker.end_job(self.job_id, status, message)
        except Exception:
            logger.exception("Error during lifecycle cleanup for job %s", self.job_id)
