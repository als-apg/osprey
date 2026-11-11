"""ARIEL database connection management.

This module provides async connection pool management for the ARIEL database.
"""

import asyncio
import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from psycopg_pool import AsyncConnectionPool

    from osprey.services.ariel_search.config import DatabaseConfig


async def create_connection_pool(
    config: "DatabaseConfig",
    *,
    uri: str | None = None,
    max_size: int = 10,
) -> "AsyncConnectionPool":
    """Create async connection pool for ARIEL repository.

    The open keeps no clock of its own: the caller's deadline (a request, a CLI
    command, a start-up probe) is the only one. The pool is returned once its
    first connection is in it, however long a loaded store takes to accept
    it; a store that refuses -- nothing listening, a login it rejects -- ends
    the open at once with that connection's error.

    Args:
        config: Database configuration with connection URI
        uri: Connect with this DSN instead of ``config.uri``. The one caller
            that passes it is the read-only pool the agent's raw-SQL path uses,
            which reaches the SAME store as a different role, so everything
            else about the pool is unchanged.
        max_size: Upper bound on pooled connections. The default sizes the
            ingestion and search pool; a pool serving one tool wants far fewer.

    Returns:
        Configured AsyncConnectionPool ready for use

    Raises:
        ImportError: If psycopg_pool is not installed
        psycopg.OperationalError: If the store refuses the first connection
    """
    try:
        from psycopg import AsyncConnection, OperationalError
        from psycopg_pool import AsyncConnectionPool
    except ImportError as e:
        raise ImportError(
            "psycopg[pool] is required for ARIEL database support. "
            "Install with: pip install 'psycopg[pool]'"
        ) from e

    conninfo = uri if uri is not None else config.uri
    gave_up = asyncio.Event()
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
        open=False,  # Don't open immediately
        reconnect_timeout=0,  # Don't retry on failure
        # With no retries the pool gives up after its first failed connection;
        # that, not a deadline, is what ends a refused open.
        reconnect_failed=lambda _pool: gave_up.set(),
    )
    try:
        await pool.open()
        filled = asyncio.ensure_future(pool.wait(timeout=math.inf))
        refused = asyncio.ensure_future(gave_up.wait())
        try:
            done, _pending = await asyncio.wait(
                {filled, refused}, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            refused.cancel()
            filled.cancel()
        if filled in done and filled.exception() is None:
            return pool
        # The pool keeps the error it gave up on to its log; connecting once
        # more, as the pool connects, raises the store's own answer to the
        # caller. This attempt runs on the caller's clock like the open itself.
        await (await AsyncConnection.connect(conninfo, **connect_kwargs)).close()
        raise OperationalError(
            "the ARIEL store refused the pool's first connection and accepted the next; "
            "the error the pool gave up on is in its log"
        )
    except BaseException:
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
