"""Integration tests for the ``ariel_only_sync_status`` migration.

Runs against a fresh scratch database per test. Rows are seeded on a schema
without the migration: ARIEL's own entries stored without ``sync_status`` (an
ARIEL MCP native entry, an old Logbook compose-sheet save under either name, a
simulation seed), and rows the migration must leave alone (a facility entry, an
ARIEL row that already records a status). Only ARIEL's own rows missing the key
gain ``local_only``; every other column is unchanged, and a second run changes
nothing.
"""

from __future__ import annotations

import json
import logging

import pytest

# xdist_group("docker"): every container-starting file shares one worker; see
# test_migrations.py for the full reason.
pytestmark = [pytest.mark.asyncio, pytest.mark.xdist_group("docker")]

MIGRATION = "ariel_only_sync_status"

#: (entry_id, source_system, metadata, metadata after the migration)
ROWS: list[tuple[str, str, object, object]] = [
    (
        "mcp-1",
        "ARIEL MCP",
        {"created_via": "ariel-mcp"},
        {"created_via": "ariel-mcp", "sync_status": "local_only"},
    ),
    ("web-1", "OSPREY Logbook Web", {}, {"sync_status": "local_only"}),
    ("web-old", "ARIEL Web", {"tags": ["vac"]}, {"tags": ["vac"], "sync_status": "local_only"}),
    ("sim-1", "Simulation", "not-an-object", {"sync_status": "local_only"}),
    ("web-2", "OSPREY Logbook Web", {"sync_status": "synced"}, {"sync_status": "synced"}),
    ("fac-1", "Facility eLog", {"logbook": "Ops"}, {"logbook": "Ops"}),
    ("near-1", "ariel mcp", {}, {}),
]


@pytest.fixture
async def scratch_pool(scratch_config):
    """Pool on the empty scratch database."""
    from osprey.services.ariel_search.database import create_connection_pool

    async with await create_connection_pool(scratch_config.database) as pool:
        yield pool


async def _seeded_store(pool, config, monkeypatch) -> None:
    """Migrate to everything but this migration, then seed ``ROWS``."""
    from osprey.services.ariel_search.database import migrations

    monkeypatch.setattr(
        migrations,
        "KNOWN_MIGRATIONS",
        [row for row in migrations.KNOWN_MIGRATIONS if row[0] != MIGRATION],
    )
    await migrations.run_migrations(pool, config)
    monkeypatch.undo()
    async with pool.connection() as conn:
        for entry_id, source, metadata, _after in ROWS:
            await conn.execute(
                """
                INSERT INTO enhanced_entries
                    (entry_id, source_system, timestamp, author, raw_text,
                     metadata, enhancement_status, subject)
                VALUES (%s, %s, NOW(), 'a', 'Subject\n\nBody', %s::jsonb, '{}'::jsonb,
                        'Subject')
                """,
                [entry_id, source, json.dumps(metadata)],
            )


async def _rows(pool) -> dict[str, tuple]:
    async with pool.connection() as conn:
        result = await conn.execute(
            "SELECT entry_id, metadata, source_system, raw_text, subject, enhancement_status "
            "FROM enhanced_entries ORDER BY entry_id"
        )
        return {row[0]: row[1:] for row in await result.fetchall()}


class TestRegistration:
    async def test_registered_always_on_after_the_core_schema(self):
        from osprey.services.ariel_search.database import migrations
        from osprey.services.ariel_search.database.ariel_only_sync_status_migration import (
            ArielOnlySyncStatusMigration,
        )

        (row,) = [r for r in migrations.KNOWN_MIGRATIONS if r[0] == MIGRATION]
        assert row == (
            MIGRATION,
            "osprey.services.ariel_search.database.ariel_only_sync_status_migration",
            "ArielOnlySyncStatusMigration",
            None,
        )
        assert ArielOnlySyncStatusMigration().depends_on == ["core_schema"]


class TestStamp:
    async def test_only_ariel_rows_missing_the_key_change_and_twice_is_once(
        self, scratch_pool, scratch_config, monkeypatch, caplog
    ):
        from osprey.services.ariel_search.database import run_migrations
        from osprey.services.ariel_search.database.ariel_only_sync_status_migration import (
            ArielOnlySyncStatusMigration,
        )

        await _seeded_store(scratch_pool, scratch_config, monkeypatch)
        before = await _rows(scratch_pool)

        caplog.set_level(logging.INFO, logger="ariel")
        assert await run_migrations(scratch_pool, scratch_config) == [MIGRATION]
        assert f"{MIGRATION}: 4 entries stamped local_only" in caplog.text
        first = await _rows(scratch_pool)

        assert await run_migrations(scratch_pool, scratch_config) == []
        async with scratch_pool.connection() as conn, conn.transaction():
            await ArielOnlySyncStatusMigration().up(conn)
        assert await _rows(scratch_pool) == first

        for entry_id, _source, _metadata, after in ROWS:
            assert first[entry_id][0] == after, entry_id
            # source_system, raw_text, subject and enhancement_status never change.
            assert first[entry_id][1:] == before[entry_id][1:], entry_id
