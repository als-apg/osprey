from __future__ import annotations

import os
from functools import cache
from typing import Literal, overload

from redis import Redis


def get_redis_url() -> str:
    """Get Redis URL from environment variable or default."""
    return os.environ.get("REDIS_URL", "redis://localhost:6379/0")


@cache
def _str_client() -> Redis[str]:
    """The process-wide decode_responses=True client (created once)."""
    return Redis.from_url(get_redis_url(), decode_responses=True)


@cache
def _binary_client() -> Redis[bytes]:
    """The process-wide decode_responses=False client (created once)."""
    return Redis.from_url(get_redis_url(), decode_responses=False)


@overload
def get_redis_client(binary: Literal[False] = False) -> Redis[str]: ...


@overload
def get_redis_client(binary: Literal[True]) -> Redis[bytes]: ...


@overload
def get_redis_client(binary: bool) -> Redis[str] | Redis[bytes]: ...


def get_redis_client(binary: bool = False) -> Redis[str] | Redis[bytes]:
    """Get or create a Redis client instance.

    Args:
        binary: If True, returns a client with decode_responses=False (for bytes).
               If False, returns a client with decode_responses=True (for str).

    The two variants are memoized behind ``@cache`` factories (the public
    function itself cannot be cached — ``_lru_cache_wrapper`` does not compose
    with the overloads above). Call :func:`reset_redis_clients` to drop them.
    """
    return _binary_client() if binary else _str_client()


def reset_redis_clients() -> None:
    """Drop the cached clients so the next call reconnects.

    Needed whenever ``REDIS_URL`` changes after a client was created (tests
    pointing at an embedded server); the URL is read at creation time only.
    """
    _str_client.cache_clear()
    _binary_client.cache_clear()


def ping_redis() -> None:
    """Verify Redis connectivity; raises on failure.

    Redis clients are lazy — ``from_url`` does not open a connection. Call this
    at startup so an unreachable broker surfaces before we register singletons.
    """
    get_redis_client().ping()


def ping_url(url: str) -> None:
    """Open a throwaway client to ``url``, ping it, then close it.

    Propagates ``redis.ConnectionError`` (and any other ping error) so callers
    can retry, fall back to an embedded server, or fail. Unlike
    :func:`ping_redis` this does not touch the module singletons — it is for
    probing an arbitrary URL (startup readiness, existing-server detection).
    """
    client = Redis.from_url(url)
    try:
        client.ping()
    finally:
        client.close()
