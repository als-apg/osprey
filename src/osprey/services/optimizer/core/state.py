"""Shared base for job state classes (status, logs, error message)."""

from __future__ import annotations

import logging
from typing import TypeVar

from redis.commands.core import Script

from osprey.services.optimizer.models.enums import JobStatus
from osprey.services.optimizer.utils import redis_client
from osprey.services.optimizer.utils.job_keys import state_prefix
from osprey.services.optimizer.utils.redis_state_store import (
    RUNNING_STATE_TTL_SECONDS,
    RedisStateStore,
)
from osprey.services.optimizer.utils.serialization import serialize_obj

logger = logging.getLogger(__name__)

__all__ = ["TERMINAL_STATE_TTL_SECONDS", "JobStateBase", "wire_state"]

# How long a finished job's Redis footprint (tracker metadata + state keys)
# lingers before self-reclaiming. Long enough for the UI to fetch final
# status/logs after a run ends, but bounded so completed-run keys don't
# accumulate forever. Only applied once a job is terminal — never to a running
# job, whose keys back the single active-slot guard.
TERMINAL_STATE_TTL_SECONDS = 3600

# Byte-compare-and-swap on the pickled status key. KEYS[1]=full status key,
# ARGV[1]=expected pickle bytes, ARGV[2]=new pickle bytes, ARGV[3]=ttl. Pickle of
# a given enum member is byte-stable, so byte equality is a valid CAS. Only swaps
# when the stored bytes equal ARGV[1]; returns 1 on swap, 0 otherwise.
_STATUS_CAS_LUA = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    redis.call('SET', KEYS[1], ARGV[2], 'EX', tonumber(ARGV[3]))
    return 1
end
return 0
"""


class JobStateBase(RedisStateStore):
    """Base class for job state.

    Adds the job-lifecycle surface (status, logs, error message) on top of
    the raw Redis keyed-namespace store. Concrete job types
    (``_OptimizationState``) subclass this and add their own
    domain-specific getters and setters.
    """

    def get_status(self) -> JobStatus:
        """Get current job status."""
        return self._expect_type(self._load_from_redis("status"), JobStatus, "status")

    def get_logs(self) -> str:
        """Get captured logs.

        Returns ``""`` when the key is absent — a legitimate early-lifecycle
        state before the running process has flushed its first log batch.
        A non-str non-None value indicates genuine corruption and still raises.
        """
        logs = self._load_from_redis("logs")
        if logs is None:
            return ""
        return self._expect_type(logs, str, "logs")

    def get_error_message(self) -> str | None:
        """Get error message if job failed."""
        return self._load_typed("error_message", str)

    def set_status(self, status: JobStatus) -> None:
        """Set job status."""
        self._save_to_redis("status", status)
        logger.debug("State updated to %s", status)

    def set_status_cas(self, expected: JobStatus, new: JobStatus) -> bool:
        """Compare-and-swap the status: set ``new`` only if the stored value still equals ``expected``.

        Status is a dual-writer field (worker + API). A plain read-check-write
        can clobber a terminal status the worker set in the gap, resurrecting a
        finished job. This runs a single Lua script that byte-compares the stored
        pickle against ``serialize_obj(expected)`` and swaps only on a match —
        the swap and the comparison are one atomic Redis operation.

        Returns True if the swap happened, False if the stored value differed
        (the caller should re-read to report the real current status).
        """
        client = self._require_redis_client()
        key = f"{self._redis_key_prefix}:status"
        script: Script = client.register_script(_STATUS_CAS_LUA)
        result = script(
            keys=[key],
            args=[serialize_obj(expected), serialize_obj(new), RUNNING_STATE_TTL_SECONDS],
        )
        matched = bool(result == 1)
        logger.debug("CAS status %s -> %s: matched=%s", expected, new, matched)
        return matched

    def set_error_message(self, message: str | None) -> None:
        """Set error message."""
        self._save_to_redis("error_message", message)

    def save_logs(self, logs: str) -> None:
        """Save log text directly to Redis."""
        self._save_to_redis("logs", logs)


_S = TypeVar("_S", bound=JobStateBase)


def wire_state(cls: type[_S], job_id: str) -> _S:
    """Build a job-state handle for ``job_id`` bound to that job's Redis namespace."""
    state = cls()
    state.set_redis_client(redis_client.get_redis_client(binary=True), state_prefix(job_id))
    return state
