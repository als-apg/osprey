"""The migration lock and the non-queueing table lock against real PostgreSQL.

Every test works in its own scratch database: advisory locks are per-database,
so the ``ariel_migrate`` lock taken here can never meet another module's.
"""

from __future__ import annotations

import asyncio
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

# xdist_group("docker"): pins every container-starting test file onto one worker, so
# a run has a single testcontainers session and a single ryuk reaper -- concurrent
# reaper starts race the Docker daemon's port mapper. It also serializes the shared
# database: the session ``database_url`` fixture prefers a running dev Postgres with
# ONE shared ``ariel_test`` database over a per-worker container, so parallel workers
# would otherwise collide on migrations/seed/truncate.
pytestmark = [pytest.mark.asyncio, pytest.mark.xdist_group("docker"), pytest.mark.timeout(30)]

LOCK_SQL = "SELECT pg_advisory_lock(hashtextextended(%(key)s, 0))"
TRY_LOCK_SQL = "SELECT pg_try_advisory_lock(hashtextextended(%(key)s, 0))"
UNLOCK_SQL = "SELECT pg_advisory_unlock(hashtextextended(%(key)s, 0))"


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
    pool = await create_connection_pool(DatabaseConfig(uri=probe_database))
    try:
        yield pool
    finally:
        await pool.close()


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


async def _lock_is_free(uri: str) -> bool:
    async with await psycopg.AsyncConnection.connect(uri, autocommit=True) as conn:
        cur = await conn.execute(TRY_LOCK_SQL, {"key": MIGRATION_LOCK_KEY})
        got = (await cur.fetchone())[0]
        if got:
            await conn.execute(UNLOCK_SQL, {"key": MIGRATION_LOCK_KEY})
        return bool(got)


async def test_try_with_the_lock_free_applies_a_busy_then_free_migration(
    probe_pool, probe_config, probe_database
) -> None:
    """No self-deadlock: the runner's own lock never blocks its migration.

    ``lock_probe`` is read by another session for about a second, so the
    migration's ACCESS EXCLUSIVE lock is refused at first and granted once the
    reader commits -- all inside one ``lock='try'`` run.
    """
    reader = await psycopg.AsyncConnection.connect(probe_database)
    await reader.execute("SELECT * FROM lock_probe")  # opens a transaction holding ACCESS SHARE

    async def release_later() -> None:
        await asyncio.sleep(1.0)
        await reader.commit()

    releaser = asyncio.create_task(release_later())
    try:
        result = await run_migrations_detailed(probe_pool, probe_config, lock="try")
    finally:
        await releaser
        await reader.close()

    assert result.lock_held is False
    assert result.applied == ["lock_probe_column"]
    assert result.busy_skipped == []
    assert _applied(probe_database) == ["lock_probe_column"]


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
    probe_pool, probe_config, probe_database
) -> None:
    holder = await _hold_migration_lock(probe_database)
    try:
        run = asyncio.create_task(run_migrations_detailed(probe_pool, probe_config))
        await asyncio.sleep(1.0)
        assert not run.done()
        assert _applied(probe_database) == []
    finally:
        await holder.close()

    result = await asyncio.wait_for(run, timeout=10)
    assert result.applied == ["lock_probe_column"]
    assert result.lock_held is False


async def test_cancel_during_the_wait_leaves_no_lock(
    probe_pool, probe_config, probe_database
) -> None:
    """A run cancelled while waiting must not take the lock once it frees."""
    holder = await _hold_migration_lock(probe_database)
    try:
        run = asyncio.create_task(run_migrations_detailed(probe_pool, probe_config))
        await asyncio.sleep(1.0)
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run
    finally:
        await holder.close()

    # Give a backend that kept waiting the chance to take the freed lock.
    await asyncio.sleep(1.0)
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
