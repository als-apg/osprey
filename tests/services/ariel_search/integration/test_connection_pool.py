"""ARIEL pools survive a column added while they serve readers.

A migration can ADD COLUMN to ``enhanced_entries`` in-process while the web and
MCP pools keep serving ``SELECT *`` readers. A server-prepared statement on a
pooled connection would then fail with "cached plan must not change result
type" until the connection is recycled, so ARIEL pools never prepare.
"""

from __future__ import annotations

import psycopg
import pytest

from osprey.services.ariel_search.config import DatabaseConfig
from osprey.services.ariel_search.database.connection import create_connection_pool

# xdist_group("docker"): pins every container-starting test file onto one worker, so
# a run has a single testcontainers session and a single ryuk reaper -- concurrent
# reaper starts race the Docker daemon's port mapper. It also serializes the shared
# database: the session ``database_url`` fixture prefers a running dev Postgres with
# ONE shared ``ariel_test`` database over a per-worker container, so parallel workers
# would otherwise collide on migrations/seed/truncate.
pytestmark = [pytest.mark.asyncio, pytest.mark.xdist_group("docker")]

SELECT = "SELECT * FROM enhanced_entries WHERE entry_id = %(id)s"
#: One more than psycopg's default prepare threshold (5), so a preparing
#: connection has the statement prepared on the server by the last run.
WARM_RUNS = 6


@pytest.fixture
def entries_database(scratch_database: str) -> str:
    with psycopg.connect(scratch_database, autocommit=True) as conn:
        conn.execute("CREATE TABLE enhanced_entries (entry_id text PRIMARY KEY, raw_text text)")
        conn.execute("INSERT INTO enhanced_entries VALUES ('e1', 'beam dump')")
    return scratch_database


def _add_column(uri: str) -> None:
    with psycopg.connect(uri, autocommit=True) as conn:
        conn.execute("ALTER TABLE enhanced_entries ADD COLUMN probe_col text")


async def test_pooled_reader_survives_add_column(entries_database: str) -> None:
    pool = await create_connection_pool(DatabaseConfig(uri=entries_database), max_size=1)
    try:
        async with pool.connection() as conn:
            for _ in range(WARM_RUNS):
                await (await conn.execute(SELECT, {"id": "e1"})).fetchall()

            _add_column(entries_database)

            rows = await (await conn.execute(SELECT, {"id": "e1"})).fetchall()
    finally:
        await pool.close()

    assert rows == [("e1", "beam dump", None)]


async def test_a_preparing_connection_does_hit_the_stale_plan(entries_database: str) -> None:
    """The hazard is real: psycopg's default threshold breaks on the same steps."""
    async with await psycopg.AsyncConnection.connect(entries_database, autocommit=True) as conn:
        for _ in range(WARM_RUNS):
            await (await conn.execute(SELECT, {"id": "e1"})).fetchall()

        _add_column(entries_database)

        with pytest.raises(psycopg.errors.FeatureNotSupported):
            await conn.execute(SELECT, {"id": "e1"})
