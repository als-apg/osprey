"""ARIEL database connection management.

This module provides async connection pool management for the ARIEL database.
"""

import asyncio
import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from psycopg import AsyncConnection
    from psycopg_pool import AsyncConnectionPool

    from osprey.services.ariel_search.config import DatabaseConfig
    from osprey.services.ariel_search.exceptions import StoreSilentError


class _Opening:
    """What one open learns from the pool's own connects.

    The pool connects in its worker; this is where that connect is seen. It
    keeps the first error the store answered with, so the open can hand that
    error itself to the caller, and it holds each connect to the caller's
    deadline while the open lasts, so a store that never answers ends the
    connect rather than leaving it waiting.
    """

    def __init__(self, deadline_s: float | None) -> None:
        loop = asyncio.get_running_loop()
        self.until: float | None = None if deadline_s is None else loop.time() + deadline_s
        self.first_error: Exception | None = None
        self.gave_up = asyncio.Event()
        self.open = True

    def over(self) -> None:
        """The open has returned: later connects carry no deadline and are not recorded."""
        self.open = False
        self.until = None

    def connection_class(self, base: "type[AsyncConnection[Any]]") -> "type[AsyncConnection[Any]]":
        """The pool's connection class: *base*, connecting under this open's watch."""
        opening = self

        class _WatchedConnection(base):  # type: ignore[valid-type,misc]
            @classmethod
            async def connect(cls, conninfo: str = "", **kwargs: Any) -> Any:
                # A silent store is cancelled here at the deadline; the
                # resulting TimeoutError is no answer of the store's, so it is
                # not recorded as one.
                async with asyncio.timeout_at(opening.until if opening.open else None):
                    try:
                        return await super().connect(conninfo, **kwargs)
                    except Exception as exc:
                        if opening.open and opening.first_error is None:
                            opening.first_error = exc
                        raise

        return _WatchedConnection


def _store_silent(conninfo: str, deadline_s: float) -> "StoreSilentError":
    """The error for a store at *conninfo* that did not answer within *deadline_s*."""
    from psycopg.conninfo import conninfo_to_dict

    from osprey.services.ariel_search.exceptions import StoreSilentError

    params = conninfo_to_dict(conninfo)
    return StoreSilentError(
        host=str(params.get("host") or params.get("hostaddr") or "localhost"),
        port=str(params.get("port") or "5432"),
        # With no database named the server takes the login's name for it; the
        # login stays out of the words, so the message says only that.
        database=str(params.get("dbname") or "(default)"),
        deadline_s=deadline_s,
    )


async def create_connection_pool(
    config: "DatabaseConfig",
    *,
    uri: str | None = None,
    max_size: int = 10,
    deadline_s: float | None = None,
) -> "AsyncConnectionPool":
    """Create async connection pool for ARIEL repository.

    The open keeps no clock of its own: *deadline_s* is the caller's, and the
    only one. The pool is returned once its first connection is in it; a store
    that refuses -- nothing listening, a login it rejects -- ends the open at
    once with the error the pool's first connect got; a store that takes the
    connection and does not answer within the deadline ends it with
    :class:`~osprey.services.ariel_search.exceptions.StoreSilentError`.

    Args:
        config: Database configuration with connection URI
        uri: Connect with this DSN instead of ``config.uri``. The one caller
            that passes it is the read-only pool the agent's raw-SQL path uses,
            which reaches the SAME store as a different role, so everything
            else about the pool is unchanged.
        max_size: Upper bound on pooled connections. The default sizes the
            ingestion and search pool; a pool serving one tool wants far fewer.
        deadline_s: Seconds the caller gives the store to answer. ``None``
            means the caller states none, and the open waits as long as the
            store takes.

    Returns:
        Configured AsyncConnectionPool ready for use

    Raises:
        ImportError: If psycopg_pool is not installed
        psycopg.OperationalError: If the store refuses the first connection
        StoreSilentError: If the store does not answer within *deadline_s*
    """
    try:
        from psycopg import AsyncConnection
        from psycopg_pool import AsyncConnectionPool
    except ImportError as e:
        raise ImportError(
            "psycopg[pool] is required for ARIEL database support. "
            "Install with: pip install 'psycopg[pool]'"
        ) from e

    conninfo = uri if uri is not None else config.uri
    opening = _Opening(deadline_s)
    # ``prepare_threshold=None`` keeps every statement unprepared on the
    # server. A migration may ADD COLUMN to a table while this process's other
    # pools are serving readers; a server-prepared ``SELECT *`` on a pooled
    # connection then fails with "cached plan must not change result type"
    # until that connection is recycled. Unprepared statements are planned
    # afresh and see the new column.
    connect_kwargs: dict[str, Any] = {"autocommit": True, "prepare_threshold": None}
    pool = AsyncConnectionPool(
        conninfo=conninfo,
        min_size=1,
        max_size=max_size,
        kwargs=connect_kwargs,
        connection_class=opening.connection_class(AsyncConnection),
        open=False,  # Don't open immediately
        reconnect_timeout=0,  # Don't retry on failure
        # With no retries the pool gives up after its first failed connection,
        # which is what ends a refused open before the deadline.
        reconnect_failed=lambda _pool: opening.gave_up.set(),
    )
    try:
        await pool.open()
        filled = asyncio.ensure_future(pool.wait(timeout=math.inf))
        refused = asyncio.ensure_future(opening.gave_up.wait())
        try:
            done, _pending = await asyncio.wait(
                {filled, refused}, timeout=deadline_s, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            refused.cancel()
            filled.cancel()
        if filled in done and filled.exception() is None:
            opening.over()
            return pool
        if opening.first_error is not None:
            raise opening.first_error
        if deadline_s is None:  # the pool gave up on no error it recorded
            raise RuntimeError("the ARIEL store's pool gave up on its first connection")
        raise _store_silent(conninfo, deadline_s)
    except BaseException:
        opening.over()
        await pool.close()
        raise


@asynccontextmanager
async def try_advisory_lock(
    conninfo: str,
    key: str,
    *,
    wait: bool = False,
) -> AsyncIterator[bool]:
    """Hold a session-level PostgreSQL advisory lock for the ``async with`` body.

    The lock lives on a dedicated, non-pooled autocommit connection opened from
    *conninfo*. The body's exit unlocks it, so the next caller finds it free;
    closing that connection releases it on any path where the unlock does not
    land. Keeping it off the pool means no exit path -- an exception, a task
    cancel -- can hand a connection still holding the lock back to a pool for an
    unrelated caller.
    Callers pass the pool's own ``conninfo`` so the lock lives in the database
    that pool works on.

    Args:
        conninfo: DSN of the database to lock in. Required: an empty value
            would let libpq fall back to its environment defaults and lock a
            different database than the caller works on.
        key: Lock name; hashed server-side to the 64-bit advisory-lock key.
        wait: Block until the lock is free when True. When False, return at
            once whether or not it was obtained.

    Yields:
        True when the lock is held for the body, False when ``wait`` is False
        and another session holds it.

    Raises:
        ValueError: If *conninfo* is empty.
    """
    if not conninfo:
        raise ValueError("try_advisory_lock needs the conninfo of the database to lock in")

    from psycopg import AsyncConnection

    conn = await AsyncConnection.connect(conninfo, autocommit=True)
    try:
        try:
            if wait:
                await conn.execute(
                    "SELECT pg_advisory_lock(hashtextextended(%(key)s, 0))", {"key": key}
                )
                held = True
            else:
                cur = await conn.execute(
                    "SELECT pg_try_advisory_lock(hashtextextended(%(key)s, 0))", {"key": key}
                )
                row = await cur.fetchone()
                held = bool(row and row[0])
        except BaseException:
            # A cancel while blocked in pg_advisory_lock must not leave the
            # server-side wait running: closing the socket alone does not stop
            # a backend that is waiting on a lock, which would then take the
            # lock the moment it frees and hold it until it noticed the client
            # had gone. Cancel the statement first, then close.
            # Best effort: the close below runs whether or not the cancel lands.
            with suppress(Exception):
                await conn.cancel_safe()
            raise
        try:
            yield held
        finally:
            # Closing only asks the backend to exit; the server drops the lock
            # once that exit completes, which can be after the next caller has
            # already asked for it. Unlock first so the lock is free on return.
            # Best effort: the close below releases it whether or not this lands.
            if held:
                with suppress(Exception):
                    await conn.execute(
                        "SELECT pg_advisory_unlock(hashtextextended(%(key)s, 0))", {"key": key}
                    )
    finally:
        await conn.close()
