"""State class and protocols for optimization runs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import polars as pl
from redis.commands.core import Script

from osprey_optimizer.models.enums import JobStatus
from osprey_optimizer.models.protocols import JobStateReader, JobStateWriter
from osprey_optimizer.models.types import DIRECTIONS, Direction
from osprey_optimizer.utils import redis_client
from osprey_optimizer.utils.job_keys import state_prefix
from osprey_optimizer.utils.redis_client import BytesRedis
from osprey_optimizer.utils.redis_state_store import (
    RUNNING_STATE_TTL_SECONDS,
    RedisStateStore,
    _truncate_repr,
)
from osprey_optimizer.utils.serialization import (
    deserialize_obj,
    serialize_obj,
)

__all__ = [
    "OptimizationStateWriter",
    "get_optimization_reader",
    "get_optimization_writer",
]

logger = logging.getLogger(__name__)

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


class _OptimizationState(RedisStateStore):
    """Concrete state class for optimization runs.

    The job-lifecycle surface (status, logs, error message) plus the
    optimization fields, on top of the raw Redis keyed-namespace store.

    Private: all construction must go through ``get_optimization_writer`` /
    ``get_optimization_reader``. Consumers type against the Reader/Writer
    protocols instead of this class.
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

    def _increment_data_version(self) -> None:
        """Increment data version counter atomically via Redis INCR."""
        client = self._require_redis_client()
        client.incr(f"{self._redis_key_prefix}:data_version")

    def set_last_results_path(self, path: Path | None) -> None:
        """Set path to last completed optimization results.

        Args:
            path: Path to results directory or None
        """
        self._save_to_redis("last_results_path", path)

    def set_data(self, data: pl.DataFrame | None) -> None:
        """Set optimization data.

        Args:
            data: Optimization data as polars DataFrame or None to clear
        """
        self._save_dataframe_to_redis("data", data)
        self._increment_data_version()

    def set_total_evaluations(self, total: int | None) -> None:
        """Set the total planned evaluations (resolved initial samples + iterations).

        Args:
            total: Total evaluation count, or None to clear
        """
        self._save_to_redis("total_evaluations", total)

    def set_write_authorization(self, state: dict[str, Any] | None) -> None:
        """Set the run's write authorization as its status shows it.

        Args:
            state: ``{"state": "requesting" | "awaiting" | "granted" | "ended" |
                "released", "id": ..., "steps": ..., "completion": ...,
                "reason": ...}``, or None to clear.
        """
        self._save_to_redis("write_authorization", state)

    def get_write_authorization(self) -> dict[str, Any] | None:
        """Get the run's write authorization state, or None before it asks for one."""
        return self._load_typed("write_authorization", dict)

    def get_start_notice(self) -> str | None:
        """Get the draft's warning that the run may not move from its start, if any.

        Written with the draft's config by ``api.job_control.store_created_config``.
        """
        return self._load_typed("start_notice", str)

    def set_objective_name(self, name: str | None) -> None:
        """Set objective name.

        Args:
            name: Objective name or None to clear
        """
        self._save_to_redis("objective_name", name)

    def set_environment_name(self, name: str | None) -> None:
        """Set environment name.

        Args:
            name: Environment name or None to clear
        """
        self._save_to_redis("environment_name", name)

    def set_variable_names(self, names: list[str] | None) -> None:
        """Set variable names used in optimization.

        Args:
            names: List of variable names or None to clear
        """
        self._save_to_redis("variable_names", names)

    def set_variable_bounds(self, bounds: dict[str, list[float]] | None) -> None:
        """Set resolved [min, max] bounds per active variable.

        Args:
            bounds: Mapping of variable name to ``[min, max]`` (the resolved,
                safety-clamped, grid-snapped search space), or None to clear.
        """
        self._save_to_redis("variable_bounds", bounds)

    def set_objective_direction(self, direction: str | None) -> None:
        """Set objective optimization direction ('MAXIMIZE' or 'MINIMIZE')."""
        if direction is not None and direction not in DIRECTIONS:
            raise ValueError(
                f"Invalid objective direction {direction!r}; expected 'MAXIMIZE', 'MINIMIZE', or None"
            )
        self._save_to_redis("objective_direction", direction)

    def save_metadata(self, metadata: dict[str, Any]) -> None:
        """Save variable and observable metadata for summary enrichment.

        Args:
            metadata: Dict with keys ``"variables"`` and ``"observables"``,
                each mapping PV name to a dict of relevant metadata fields.
        """
        self._save_to_redis("metadata", metadata)

    def save_snapshot(self, name: str, data: dict[str, float]) -> None:
        """Save a snapshot of variable values (atomic per-field via Redis HSET).

        Args:
            name: Name of the snapshot (e.g., 'initial', 'lhs_best', 'bo_best')
            data: Dictionary of variable names and values
        """
        client = self._require_redis_client()
        client.hset(
            f"{self._redis_key_prefix}:snapshots",
            name,
            serialize_obj(data),
        )

    def get_last_results_path(self) -> Path | None:
        """Get path to last completed optimization results.

        Returns:
            Path to results directory or None
        """
        return self._load_typed("last_results_path", Path)

    def get_data(self) -> pl.DataFrame | None:
        """Get optimization data from Redis.

        Returns:
            Optimization data as polars DataFrame or None if not found
        """
        return self._load_dataframe_from_redis("data")

    def get_total_evaluations(self) -> int | None:
        """Get the total planned evaluations (resolved initial samples + iterations).

        Returns:
            Total evaluation count or None if not set
        """
        return self._load_typed("total_evaluations", int)

    def get_objective_name(self) -> str | None:
        """Get objective name.

        Returns:
            Objective name or None
        """
        return self._load_typed("objective_name", str)

    def get_environment_name(self) -> str | None:
        """Get environment name.

        Returns:
            Environment name or None
        """
        return self._load_typed("environment_name", str)

    def get_variable_names(self) -> list[str] | None:
        """Get variable names used in optimization.

        Returns:
            List of variable names or None
        """
        return self._load_typed("variable_names", list)

    def get_variable_bounds(self) -> dict[str, list[float]] | None:
        """Get resolved [min, max] bounds per active variable.

        Returns:
            Mapping of variable name to ``[min, max]`` or None if not set.
        """
        return self._load_typed("variable_bounds", dict)

    def get_objective_direction(self) -> Direction | None:
        """Get objective optimization direction ('MAXIMIZE' or 'MINIMIZE')."""
        result = self._load_from_redis("objective_direction")
        if result is None:
            return None
        narrowed = self._expect_type(result, str, "objective_direction")
        if narrowed not in DIRECTIONS:
            raise ValueError(
                f"Invalid objective_direction at {self._redis_key_prefix}:objective_direction: "
                f"got {_truncate_repr(narrowed)}, expected 'MAXIMIZE' or 'MINIMIZE'"
            )
        return Direction(narrowed)

    def get_data_version(self) -> int:
        """Get current data version.

        Returns:
            Current data version number. Returns 0 when the key has never been
            set; raises TypeError/ValueError if the stored value is malformed.
        """
        client = self._require_redis_client()
        key = f"{self._redis_key_prefix}:data_version"
        value = client.get(key)
        if value is None:
            return 0
        if not isinstance(value, bytes):
            raise TypeError(f"Expected bytes at {key}, got {type(value).__name__}")
        try:
            return int(value.decode())
        except (ValueError, UnicodeDecodeError) as e:
            raise ValueError(
                f"Invalid data_version at {key}: got {_truncate_repr(value)}, expected decimal bytes"
            ) from e

    def get_metadata(self) -> dict[str, Any] | None:
        """Get variable and observable metadata stored during initial state capture."""
        return self._load_typed("metadata", dict)

    def get_available_snapshots(self) -> dict[str, dict[str, float]]:
        """Get all available snapshots via Redis HGETALL.

        Returns:
            Dictionary of snapshot name -> snapshot data. Empty dict when no
            snapshots have been saved.
        """
        client = self._require_redis_client()
        key = f"{self._redis_key_prefix}:snapshots"
        raw = client.hgetall(key)
        if not raw:
            return {}
        result: dict[str, dict[str, float]] = {}
        for field, value in raw.items():
            if not isinstance(field, bytes):
                raise TypeError(f"Expected bytes field at {key}, got {type(field).__name__}")
            field_name = field.decode()
            if not isinstance(value, bytes):
                raise TypeError(
                    f"Expected bytes at {key}[{field_name}], got {type(value).__name__}"
                )
            deserialized = deserialize_obj(value)
            if not isinstance(deserialized, dict):
                raise TypeError(
                    f"Expected dict at {key}[{field_name}], got {type(deserialized).__name__}"
                )
            result[field_name] = deserialized
        return result


@runtime_checkable
class OptimizationStateReader(JobStateReader, Protocol):
    """Protocol for reading optimization state (extends JobStateReader)."""

    def get_last_results_path(self) -> Path | None: ...
    def get_data(self) -> pl.DataFrame | None: ...
    def get_objective_name(self) -> str | None: ...
    def get_environment_name(self) -> str | None: ...
    def get_variable_names(self) -> list[str] | None: ...
    def get_variable_bounds(self) -> dict[str, list[float]] | None: ...
    def get_objective_direction(self) -> Direction | None: ...
    def get_data_version(self) -> int: ...
    def get_total_evaluations(self) -> int | None: ...
    def get_metadata(self) -> dict[str, Any] | None: ...
    def get_available_snapshots(self) -> dict[str, dict[str, float]]: ...
    def get_write_authorization(self) -> dict[str, Any] | None: ...
    def get_start_notice(self) -> str | None: ...


@runtime_checkable
class OptimizationStateWriter(OptimizationStateReader, JobStateWriter, Protocol):
    """Protocol for writing optimization state (extends reader + JobStateWriter)."""

    def set_redis_client(self, client: BytesRedis, key_prefix: str) -> None: ...
    def set_last_results_path(self, path: Path | None) -> None: ...
    def set_data(self, data: pl.DataFrame | None) -> None: ...
    def set_total_evaluations(self, total: int | None) -> None: ...
    def set_objective_name(self, name: str | None) -> None: ...
    def set_environment_name(self, name: str | None) -> None: ...
    def set_variable_names(self, names: list[str] | None) -> None: ...
    def set_variable_bounds(self, bounds: dict[str, list[float]] | None) -> None: ...
    def set_objective_direction(self, direction: str | None) -> None: ...
    def save_metadata(self, metadata: dict[str, Any]) -> None: ...
    def save_snapshot(self, name: str, data: dict[str, float]) -> None: ...
    def set_write_authorization(self, state: dict[str, Any] | None) -> None: ...


def get_optimization_writer(job_id: str) -> OptimizationStateWriter:
    """Return a writable handle to the optimization state for ``job_id``."""
    state = _OptimizationState()
    state.set_redis_client(redis_client.get_redis_client(binary=True), state_prefix(job_id))
    return state


def get_optimization_reader(job_id: str) -> OptimizationStateReader:
    """Return a read-only view of the optimization state for ``job_id``."""
    return get_optimization_writer(job_id)
