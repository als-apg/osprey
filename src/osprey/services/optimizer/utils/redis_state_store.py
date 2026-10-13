"""Redis keyed-namespace store.

Provides the shared Redis plumbing used by job state classes: a client +
key-prefix pair and typed load/save primitives for serialized Python values
and polars DataFrames.
"""

from __future__ import annotations

from typing import Any, TypeVar

import polars as pl
from redis import Redis

from osprey.services.optimizer.utils.serialization import (
    deserialize_dataframe,
    deserialize_obj,
    serialize_dataframe,
    serialize_obj,
)

T = TypeVar("T")

_TRUNCATE_LIMIT = 64

# TTL every per-job key carries while its job is running. The job's heartbeat
# (see JobTracker.refresh_heartbeat) re-applies it across the whole job:{id}:*
# namespace each beat, so a live job's keys never lapse — but a hard-killed
# worker stops beating, and its entire footprint self-reclaims within this
# window with no terminal sweep. Every SET write stamps it inline (SET clears
# any prior TTL), so hot keys like status/logs/data are never left unbounded.
# Must stay >> the heartbeat interval so a single missed beat can't drop a key.
RUNNING_STATE_TTL_SECONDS = 60
# TTL for a CREATED-but-not-started job's whole job:{id}:* footprint. A created
# job has no heartbeat, so this is the window in which an abandoned create
# (never started) self-reclaims. Comfortably longer than RUNNING_STATE_TTL so a
# human can create then start without the config expiring underneath them.
CREATED_STATE_TTL_SECONDS = 600


def _truncate_repr(value: Any, limit: int = _TRUNCATE_LIMIT) -> str:
    """Bounded ``repr`` for embedding untrusted Redis-loaded values in error messages.

    Corrupt keys can hold arbitrarily large blobs; the raise-site messages
    propagate into logs and HTTP 503 response bodies, so the raw value must
    not be inlined unbounded. Returns ``repr(value)`` truncated to ``limit``
    chars with a trailing ellipsis when clipped.
    """
    s = repr(value)
    return s if len(s) <= limit else s[:limit] + "..."


class RedisStateStore:
    """Redis client + key prefix wrapper with serialization helpers.

    Subclasses access Redis through the protected ``_load_*`` / ``_save_*``
    primitives, which scope every operation to ``_redis_key_prefix``. The
    client is attached after construction via ``set_redis_client``; calling a
    read/write before then raises ``RuntimeError``.
    """

    def __init__(self) -> None:
        self._redis_client: Redis[bytes] | None = None
        self._redis_key_prefix = ""

    def set_redis_client(self, client: Redis[bytes], key_prefix: str) -> None:
        """Set Redis client and key prefix for state persistence.

        Args:
            client: Redis client instance (must not be None)
            key_prefix: Prefix for Redis keys (e.g. ``optimization_state:job-123``)
        """
        self._redis_client = client
        self._redis_key_prefix = key_prefix

    def _require_redis_client(self) -> Redis[bytes]:
        """Return the Redis client or raise if not initialized."""
        if self._redis_client is None:
            raise RuntimeError(
                "Redis client not initialized — call set_redis_client() before using state read/write methods"
            )
        return self._redis_client

    def _load_bytes(self, key_suffix: str) -> bytes | None:
        """Load raw bytes for a key, or ``None`` if absent. Raises on a non-bytes value."""
        client = self._require_redis_client()
        key = f"{self._redis_key_prefix}:{key_suffix}"
        data = client.get(key)
        if data is None:
            return None
        if isinstance(data, bytes):
            return data
        raise ValueError(f"Redis key {key} contained invalid type: {type(data).__name__}")

    def _load_from_redis(self, key_suffix: str) -> Any | None:
        """Load a pickled Python value from Redis."""
        data = self._load_bytes(key_suffix)
        return deserialize_obj(data) if data is not None else None

    def _load_dataframe_from_redis(self, key_suffix: str) -> pl.DataFrame | None:
        """Load a polars DataFrame from Redis."""
        data = self._load_bytes(key_suffix)
        return deserialize_dataframe(data) if data is not None else None

    def _load_typed(self, key_suffix: str, tp: type[T]) -> T | None:
        """Load a pickled value, passing ``None`` through and narrowing to ``tp``.

        The common getter shape: ``None`` (absent key) stays ``None``; any
        present value is checked against ``tp`` via ``_expect_type`` (whose
        ``TypeError`` message is asserted on by tests).
        """
        data = self._load_from_redis(key_suffix)
        return None if data is None else self._expect_type(data, tp, key_suffix)

    def _save_to_redis(self, key_suffix: str, data: Any) -> None:
        """Save a pickled Python value to Redis."""
        client = self._require_redis_client()
        key = f"{self._redis_key_prefix}:{key_suffix}"
        client.set(key, serialize_obj(data), ex=RUNNING_STATE_TTL_SECONDS)

    def _save_dataframe_to_redis(self, key_suffix: str, df: pl.DataFrame | None) -> None:
        """Save a polars DataFrame to Redis, or delete the key when ``df`` is None."""
        client = self._require_redis_client()
        key = f"{self._redis_key_prefix}:{key_suffix}"
        if df is None:
            client.delete(key)
        else:
            client.set(key, serialize_dataframe(df), ex=RUNNING_STATE_TTL_SECONDS)

    def _expect_type(self, value: Any, tp: type[T], key_suffix: str) -> T:
        """Narrow a Redis-loaded value to ``tp`` or raise ``TypeError``.

        Used by getters *after* their own ``None``/domain handling. The
        ``Expected {T} at {prefix}:{suffix}, got {actual}`` phrasing is
        load-bearing — tests assert on it via ``pytest.raises(match=...)``.
        Hash-field variants (``{prefix}:snap[{field}]``) keep their raises
        inlined to avoid a second helper whose signature would be awkward.
        """
        if not isinstance(value, tp):
            raise TypeError(
                f"Expected {tp.__name__} at {self._redis_key_prefix}:{key_suffix}, got {type(value).__name__}"
            )
        return value
