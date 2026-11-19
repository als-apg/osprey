"""The migration lock and the non-queueing table lock against real PostgreSQL.

Every test works in its own scratch database: advisory locks are per-database,
so the ``ariel_migrate`` lock taken here can never meet another module's.
"""

from __future__ import annotations

import asyncio
import itertools
import time

import psycopg
import pytest

from osprey.services.ariel_search.config import ARIELConfig, DatabaseConfig
from osprey.services.ariel_search.database.connection import create_connection_pool
from osprey.services.ariel_search.database.migrations import (
    MIGRATION_LOCK_KEY,
    BaseMigration,
    MigrationRunner,
    acquire_nonqueueing_lock,
    run_migrations_detailed,
)
from tests._live import async_wait_until

# xdist_group("docker"): pins every container-starting test file onto one worker, so
# a run has a single testcontainers session and a single ryuk reaper -- concurrent
# reaper starts race the Docker daemon's port mapper. Each test session (each xdist
# worker) writes to a database it created itself (``database_url`` /
# ``own_database``), on a server it found or in a container it started.
#
# func_only: the 30 s bounds the test body, not its setup. The first database test
# on a worker starts the session's Postgres container inside its own setup, which
# has taken ~30 s on a loaded runner and left a one-second test no time to run.
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.xdist_group("docker"),
    pytest.mark.timeout(30, func_only=True),
]

LOCK_SQL = "SELECT pg_advisory_lock(hashtextextended(%(key)s, 0))"
TRY_LOCK_SQL = "SELECT pg_try_advisory_lock(hashtextextended(%(key)s, 0))"
UNLOCK_SQL = "SELECT pg_advisory_unlock(hashtextextended(%(key)s, 0))"
#: Sessions waiting on a lock that the session running this query holds.
WAITERS_SQL = (
    "SELECT count(*) FROM pg_stat_activity WHERE pg_backend_pid() = ANY(pg_blocking_pids(pid))"
)
#: Failure deadline of a wait for a state of the lock; the waits return as soon as it holds.
LOCK_STATE_TIMEOUT_S = 10.0
#: How long a run must go on waiting after it sends its blocking lock request:
#: a wait that gives up within this, and goes on without the lock, is caught.
BLOCKED_WINDOW_S = 1.0


class ProbeMigration(BaseMigration):
    """Migration whose ``up()`` needs an ACCESS EXCLUSIVE lock on ``lock_probe``."""

    @property
    def name(self) -> str:
        return "lock_probe_column"

    async def up(self, conn) -> None:
        await acquire_nonqueueing_lock(conn, "lock_probe", "ACCESS EXCLUSIVE")
        await conn.execute("ALTER TABLE lock_probe ADD COLUMN added text")


@pytest.fixture
def probe_database(scratch_database: str) -> str:
    """Scratch database holding the bookkeeping table and the probe table."""
    with psycopg.connect(scratch_database, autocommit=True) as conn:
        conn.execute(
            "CREATE TABLE ariel_migrations (name text PRIMARY KEY, applied_at timestamptz)"
        )
        conn.execute("CREATE TABLE lock_probe (id int)")
    return scratch_database


@pytest.fixture
async def probe_pool(probe_database: str):
    async with await create_connection_pool(DatabaseConfig(uri=probe_database)) as pool:
        yield pool


@pytest.fixture
def probe_config(probe_database: str) -> ARIELConfig:
    return ARIELConfig(database=DatabaseConfig(uri=probe_database))


@pytest.fixture(autouse=True)
def only_probe(monkeypatch) -> None:
    """Every run in this module sees the probe migration and nothing else."""
    monkeypatch.setattr(MigrationRunner, "_get_enabled_migrations", lambda self: [ProbeMigration()])


def _applied(uri: str) -> list[str]:
    with psycopg.connect(uri, autocommit=True) as conn:
        return [r[0] for r in conn.execute("SELECT name FROM ariel_migrations").fetchall()]


async def _hold_migration_lock(uri: str) -> psycopg.AsyncConnection:
    conn = await psycopg.AsyncConnection.connect(uri, autocommit=True)
    await conn.execute(LOCK_SQL, {"key": MIGRATION_LOCK_KEY})
    return conn


async def _blocked_run(pool, config, monkeypatch) -> asyncio.Task:
    """Start a waiting run against the held migration lock; return it once it is blocked.

    The run counts as blocked once its lock connection sends the blocking
    ``pg_advisory_lock`` and the run is still not done :data:`BLOCKED_WINDOW_S`
    later. A wait that gives up and goes on without the lock is done by then.
    """
    sent = asyncio.Event()
    real_execute = psycopg.AsyncConnection.execute

    async def execute(self, query, *args, **kwargs):
        if str(query).startswith("SELECT pg_advisory_lock("):
            sent.set()
        return await real_execute(self, query, *args, **kwargs)

    monkeypatch.setattr(psycopg.AsyncConnection, "execute", execute)
    run = asyncio.create_task(run_migrations_detailed(pool, config))
    await async_wait_until(
        sent.is_set, timeout=LOCK_STATE_TIMEOUT_S, what="the run's blocking pg_advisory_lock"
    )
    await asyncio.sleep(BLOCKED_WINDOW_S)
    assert not run.done()
    return run


async def _waiters(holder: psycopg.AsyncConnection) -> int:
    """How many sessions wait on a lock *holder* holds."""
    cur = await holder.execute(WAITERS_SQL)
    row = await cur.fetchone()
    return int(row[0]) if row else 0


async def _lock_is_free(uri: str) -> bool:
    async with await psycopg.AsyncConnection.connect(uri, autocommit=True) as conn:
        cur = await conn.execute(TRY_LOCK_SQL, {"key": MIGRATION_LOCK_KEY})
        got = (await cur.fetchone())[0]
        if got:
            await conn.execute(UNLOCK_SQL, {"key": MIGRATION_LOCK_KEY})
        return bool(got)


async def test_try_with_the_lock_free_applies_a_busy_then_free_migration(
    probe_pool, probe_config, probe_database, monkeypatch
) -> None:
    """No self-deadlock: the runner's own lock never blocks its migration.

    Another session reads ``lock_probe`` until every attempt but the last has
    been refused, so the last is granted -- all inside one ``lock='try'`` run.
    The shipped constants must span the about five seconds of retrying they
    document, with at least two refusals so a pause is measured; the run then
    goes with the pause patched to 0.05 s.
    """
    from osprey.services.ariel_search.database import migrations as migrations_module

    attempts = migrations_module.LOCK_ATTEMPTS
    assert attempts >= 3
    assert (attempts - 1) * migrations_module.LOCK_RETRY_DELAY_S >= 4.5
    monkeypatch.setattr(migrations_module, "LOCK_RETRY_DELAY_S", 0.05)
    reader = await psycopg.AsyncConnection.connect(probe_database)
    await reader.execute("SELECT * FROM lock_probe")  # opens a transaction holding ACCESS SHARE
    refused_at: list[float] = []
    real_execute = psycopg.AsyncConnection.execute

    async def execute(self, *args, **kwargs):
        try:
            return await real_execute(self, *args, **kwargs)
        except psycopg.errors.LockNotAvailable:
            # Before the refusal reaches the runner, so before its next attempt.
            refused_at.append(time.monotonic())
            if len(refused_at) == attempts - 1:
                await reader.commit()
            raise

    monkeypatch.setattr(psycopg.AsyncConnection, "execute", execute)
    try:
        result = await run_migrations_detailed(probe_pool, probe_config, lock="try")
    finally:
        await reader.close()

    assert result.lock_held is False
    assert result.applied == ["lock_probe_column"]
    assert result.busy_skipped == []
    assert _applied(probe_database) == ["lock_probe_column"]
    assert len(refused_at) == attempts - 1
    # asyncio may run a timer up to its clock resolution early; 1 ms covers it.
    pauses = [later - earlier for earlier, later in itertools.pairwise(refused_at)]
    assert min(pauses) >= migrations_module.LOCK_RETRY_DELAY_S - 0.001


async def test_try_with_the_lock_held_returns_at_once(
    probe_pool, probe_config, probe_database
) -> None:
    holder = await _hold_migration_lock(probe_database)
    try:
        started = time.monotonic()
        result = await run_migrations_detailed(probe_pool, probe_config, lock="try")
        elapsed = time.monotonic() - started
    finally:
        await holder.close()

    assert result.lock_held is True
    assert result.applied == []
    assert elapsed < 5
    assert _applied(probe_database) == []


async def test_wait_blocks_until_the_other_session_lets_go(
    probe_pool, probe_config, probe_database, monkeypatch
) -> None:
    holder = await _hold_migration_lock(probe_database)
    try:
        run = await _blocked_run(probe_pool, probe_config, monkeypatch)
        assert _applied(probe_database) == []
        assert await _waiters(holder) == 1
    finally:
        await holder.close()

    result = await asyncio.wait_for(run, timeout=10)
    assert result.applied == ["lock_probe_column"]
    assert result.lock_held is False


async def test_cancel_during_the_wait_leaves_no_lock(
    probe_pool, probe_config, probe_database, monkeypatch
) -> None:
    """A run cancelled while waiting must not take the lock once it frees."""
    holder = await _hold_migration_lock(probe_database)
    try:
        run = await _blocked_run(probe_pool, probe_config, monkeypatch)
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run

        # Nothing waits behind the holder once the cancel reaches the backend,
        # so nothing can take the lock when it lets go. The unlock returns with
        # the lock released, where a close only asks the backend to exit.
        async def nobody_waits() -> bool:
            return await _waiters(holder) == 0

        await async_wait_until(
            nobody_waits, timeout=LOCK_STATE_TIMEOUT_S, what="no session waiting on the lock"
        )
        await holder.execute(UNLOCK_SQL, {"key": MIGRATION_LOCK_KEY})
    finally:
        await holder.close()

    assert await _lock_is_free(probe_database)
    assert _applied(probe_database) == []


async def test_lock_is_released_after_a_run(probe_pool, probe_config, probe_database) -> None:
    await run_migrations_detailed(probe_pool, probe_config)

    assert await _lock_is_free(probe_database)


async def test_table_held_past_the_retries_is_a_busy_skip(
    probe_pool, probe_config, probe_database, monkeypatch
) -> None:
    """A reader that never lets go turns the migration into a busy skip, unmarked."""
    from osprey.services.ariel_search.database import migrations as migrations_module

    monkeypatch.setattr(migrations_module, "LOCK_RETRY_DELAY_S", 0.05)
    reader = await psycopg.AsyncConnection.connect(probe_database)
    try:
        await reader.execute("SELECT * FROM lock_probe")
        result = await run_migrations_detailed(probe_pool, probe_config)
    finally:
        await reader.close()

    assert result.applied == []
    assert result.busy_skipped == ["lock_probe_column"]
    assert _applied(probe_database) == []


async def test_share_lock_lets_readers_through(probe_database) -> None:
    """SHARE, the index-build mode, conflicts with no reader."""
    async with await psycopg.AsyncConnection.connect(probe_database) as builder:
        async with builder.transaction():
            await acquire_nonqueueing_lock(builder, "lock_probe", "SHARE")
            async with await psycopg.AsyncConnection.connect(
                probe_database, autocommit=True
            ) as reader:
                await reader.execute("SET lock_timeout = '2s'")
                cur = await reader.execute("SELECT count(*) FROM lock_probe")
                assert (await cur.fetchone())[0] == 0


async def test_try_advisory_lock_lets_go_before_its_connection_closes(
    probe_database, monkeypatch
) -> None:
    """The next caller finds the lock free as soon as the body exits.

    Closing a connection only asks its backend to exit; the server drops the
    session's locks when that exit completes, after ``close()`` has returned.
    The lock is checked from another session at the moment the lock connection
    is closed, so a release left to the close alone is seen as still held.
    """
    from osprey.services.ariel_search.database.connection import try_advisory_lock

    key = "ariel_enhance:probe"
    free_at_close: list[bool] = []
    real_close = psycopg.AsyncConnection.close

    async def checked_close(self) -> None:
        with psycopg.connect(probe_database, autocommit=True) as probe:
            got = probe.execute(TRY_LOCK_SQL, {"key": key}).fetchone()[0]
            if got:
                probe.execute(UNLOCK_SQL, {"key": key})
        free_at_close.append(bool(got))
        await real_close(self)

    monkeypatch.setattr(psycopg.AsyncConnection, "close", checked_close)
    async with try_advisory_lock(probe_database, key) as held:
        assert held is True

    assert free_at_close == [True]
