"""State class and protocols for optimization runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import polars as pl

from osprey.services.optimizer.core.state import JobStateBase, wire_state
from osprey.services.optimizer.models.protocols import JobStateReader, JobStateWriter
from osprey.services.optimizer.models.types import DIRECTIONS, Direction
from osprey.services.optimizer.utils.redis_client import BytesRedis
from osprey.services.optimizer.utils.redis_state_store import _truncate_repr
from osprey.services.optimizer.utils.serialization import (
    deserialize_obj,
    serialize_obj,
)

__all__ = [
    "OptimizationStateWriter",
    "get_optimization_reader",
    "get_optimization_writer",
]


class _OptimizationState(JobStateBase):
    """Concrete state class for optimization runs.

    Private: all construction must go through ``get_optimization_writer`` /
    ``get_optimization_reader``. Consumers type against the Reader/Writer
    protocols instead of this class.
    """

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


def get_optimization_writer(job_id: str) -> OptimizationStateWriter:
    """Return a writable handle to the optimization state for ``job_id``."""
    return wire_state(_OptimizationState, job_id)


def get_optimization_reader(job_id: str) -> OptimizationStateReader:
    """Return a read-only view of the optimization state for ``job_id``."""
    return get_optimization_writer(job_id)
