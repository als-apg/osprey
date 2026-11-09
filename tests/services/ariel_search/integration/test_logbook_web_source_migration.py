"""Integration tests for the ``logbook_web_source_system`` migration.

Runs against a fresh scratch database per test. Rows are seeded on a schema
without the migration: entries the Logbook's compose sheet saved under the old
name ``ARIEL Web``, and entries of every other source. The migration renames
exactly ``ARIEL Web`` to ``OSPREY Logbook Web`` and touches nothing else; a
renamed row's ``updated_at`` moves, so the qmd mirror's resync re-exports its
``Source:`` line, and every other row keeps its own. A second run changes
nothing.
"""

from __future__ import annotations

import json
import logging

import pytest

# xdist_group("docker"): every container-starting file shares one worker; see
# test_migrations.py for the full reason.
pytestmark = [pytest.mark.asyncio, pytest.mark.xdist_group("docker")]

MIGRATION = "logbook_web_source_system"

OLD = "ARIEL Web"
NEW = "OSPREY Logbook Web"

#: A fixed, old ``updated_at`` every seeded row carries.
OLD_UPDATED_AT = "2020-01-01 00:00:00+00"

#: (entry_id, source_system, metadata)
ROWS: list[tuple[str, str, dict]] = [
    ("web-1", OLD, {"sync_status": "local_only", "tags": ["vac"]}),
    ("web-2", OLD, {}),
    ("mcp-1", "ARIEL MCP", {"created_via": "ariel-mcp"}),
    ("sim-1", "Simulation", {}),
    ("near-1", "ariel web", {}),
    ("near-2", "ARIEL Web Extra", {}),
]


@pytest.fixture
async def scratch_pool(scratch_config):
    """Pool on the empty scratch database."""
    from osprey.services.ariel_search.database import create_connection_pool

    async with await create_connection_pool(scratch_config.database) as pool:
        yield pool


def _without(*names: str) -> list:
    from osprey.services.ariel_search.database import migrations

    return [row for row in migrations.KNOWN_MIGRATIONS if row[0] not in names]


async def _seeded_store(pool, config, monkeypatch) -> None:
    """Migrate to everything but the rename, then seed ``ROWS``."""
    from osprey.services.ariel_search.database import migrations

    monkeypatch.setattr(migrations, "KNOWN_MIGRATIONS", _without(MIGRATION))
    await migrations.run_migrations(pool, config)
    monkeypatch.undo()
    async with pool.connection() as conn:
        for entry_id, source, metadata in ROWS:
            await conn.execute(
                """
                INSERT INTO enhanced_entries
                    (entry_id, source_system, timestamp, author, raw_text,
                     metadata, enhancement_status, subject, updated_at)
                VALUES (%s, %s, NOW(), 'a', 'Subject\n\nBody', %s::jsonb, '{}'::jsonb,
                        'Subject', %s::timestamptz)
                """,
                [entry_id, source, json.dumps(metadata), OLD_UPDATED_AT],
            )


async def _rows(pool) -> dict[str, tuple]:
    async with pool.connection() as conn:
        result = await conn.execute(
            "SELECT entry_id, source_system, updated_at, raw_text, metadata, subject, "
            "enhancement_status FROM enhanced_entries ORDER BY entry_id"
        )
        return {row[0]: row[1:] for row in await result.fetchall()}


class TestRegistration:
    async def test_registered_always_on_after_the_core_schema(self):
        from osprey.services.ariel_search.database import migrations
        from osprey.services.ariel_search.database.logbook_web_source_migration import (
            LogbookWebSourceMigration,
        )

        (row,) = [r for r in migrations.KNOWN_MIGRATIONS if r[0] == MIGRATION]
        assert row == (
            MIGRATION,
            "osprey.services.ariel_search.database.logbook_web_source_migration",
            "LogbookWebSourceMigration",
            None,
        )
        assert LogbookWebSourceMigration().depends_on == ["core_schema"]

    async def test_the_new_name_is_the_one_the_logbook_writes(self):
        from osprey.services.ariel_search.database.logbook_web_source_migration import (
            NEW_SOURCE_SYSTEM,
            OLD_SOURCE_SYSTEM,
        )
        from osprey.services.ariel_search.entry_url import LOGBOOK_WEB_SOURCE_SYSTEM

        assert (OLD_SOURCE_SYSTEM, NEW_SOURCE_SYSTEM) == (OLD, LOGBOOK_WEB_SOURCE_SYSTEM)


class TestRename:
    async def test_only_the_exact_old_name_changes_and_twice_is_once(
        self, scratch_pool, scratch_config, monkeypatch, caplog
    ):
        from osprey.services.ariel_search.database import run_migrations
        from osprey.services.ariel_search.database.logbook_web_source_migration import (
            LogbookWebSourceMigration,
        )

        await _seeded_store(scratch_pool, scratch_config, monkeypatch)
        before = await _rows(scratch_pool)

        caplog.set_level(logging.INFO, logger="ariel")
        assert await run_migrations(scratch_pool, scratch_config) == [MIGRATION]
        assert f"{MIGRATION}: 2 entries renamed" in caplog.text
        first = await _rows(scratch_pool)

        # The runner does not run an applied migration again, and running its
        # body again by hand finds nothing left to rename.
        assert await run_migrations(scratch_pool, scratch_config) == []
        async with scratch_pool.connection() as conn, conn.transaction():
            await LogbookWebSourceMigration().up(conn)
        assert await _rows(scratch_pool) == first

        for entry_id, source, _metadata in ROWS:
            expected = NEW if source == OLD else source
            assert first[entry_id][0] == expected, entry_id
            # raw_text, metadata, subject and enhancement_status never change.
            assert first[entry_id][2:] == before[entry_id][2:], entry_id
            moved = first[entry_id][1] != before[entry_id][1]
            assert moved == (source == OLD), (entry_id, "updated_at")
