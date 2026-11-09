"""Integration tests for the ``entry_subject_and_keyset`` migration.

Runs against a fresh scratch database per test: the column, the three indexes,
idempotence, the ``has_subject`` schema fact, ingest on the pre-migration schema,
and the ``ariel.preserve_updated_at`` guard on the ``updated_at`` trigger.
"""

from __future__ import annotations

import pytest

# xdist_group("docker"): every container-starting file shares one worker; see
# test_migrations.py for the full reason.
pytestmark = [pytest.mark.asyncio, pytest.mark.xdist_group("docker")]

MIGRATION = "entry_subject_and_keyset"


@pytest.fixture
async def scratch_pool(scratch_config):
    """Pool on the empty scratch database."""
    from osprey.services.ariel_search.database import create_connection_pool

    async with await create_connection_pool(scratch_config.database) as pool:
        yield pool


def _without_migration() -> list:
    from osprey.services.ariel_search.database import migrations

    return [row for row in migrations.KNOWN_MIGRATIONS if row[0] != MIGRATION]


async def _subject_column(pool) -> tuple | None:
    async with pool.connection() as conn:
        result = await conn.execute("""
            SELECT data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = current_schema()
            AND table_name = 'enhanced_entries' AND column_name = 'subject'
        """)
        return await result.fetchone()


async def _index_definitions(pool) -> dict[str, str]:
    from osprey.services.ariel_search.database.subject_migration import SUBJECT_KEYSET_INDEXES

    async with pool.connection() as conn:
        result = await conn.execute(
            "SELECT indexname, indexdef FROM pg_indexes WHERE indexname = ANY(%s)",
            [list(SUBJECT_KEYSET_INDEXES)],
        )
        return {row[0]: row[1] for row in await result.fetchall()}


class TestRegistration:
    async def test_registered_always_on_after_core_schema(self):
        from osprey.services.ariel_search.database import migrations
        from osprey.services.ariel_search.database.subject_migration import (
            EntrySubjectAndKeysetMigration,
        )

        (row,) = [r for r in migrations.KNOWN_MIGRATIONS if r[0] == MIGRATION]
        assert row == (
            MIGRATION,
            "osprey.services.ariel_search.database.subject_migration",
            "EntrySubjectAndKeysetMigration",
            None,
        )
        migration = EntrySubjectAndKeysetMigration()
        assert migration.name == MIGRATION
        assert migration.depends_on == ["core_schema"]


class TestApply:
    async def test_full_run_adds_the_column_and_three_indexes(self, scratch_pool, scratch_config):
        from osprey.services.ariel_search.database import run_migrations

        applied = await run_migrations(scratch_pool, scratch_config)

        assert MIGRATION in applied
        assert applied.index("core_schema") < applied.index(MIGRATION)
        assert await _subject_column(scratch_pool) == ("text", "YES", None)
        indexes = await _index_definitions(scratch_pool)
        assert set(indexes) == {
            "idx_entries_timestamp_entry_id",
            "idx_entries_author_lower",
            "idx_entries_source_system_lower",
        }
        assert '("timestamp" DESC, entry_id DESC)' in indexes["idx_entries_timestamp_entry_id"]
        assert "lower(author)" in indexes["idx_entries_author_lower"]
        assert "lower(source_system)" in indexes["idx_entries_source_system_lower"]

    async def test_rerun_is_a_noop(self, scratch_pool, scratch_config):
        from osprey.services.ariel_search.database import run_migrations

        await run_migrations(scratch_pool, scratch_config)
        assert await run_migrations(scratch_pool, scratch_config) == []

    async def test_up_twice_is_idempotent(self, scratch_pool, scratch_config):
        """Calling ``up()`` again on a migrated store changes and raises nothing."""
        from osprey.services.ariel_search.database import run_migrations
        from osprey.services.ariel_search.database.subject_migration import (
            EntrySubjectAndKeysetMigration,
        )

        await run_migrations(scratch_pool, scratch_config)
        async with scratch_pool.connection() as conn, conn.transaction():
            await EntrySubjectAndKeysetMigration().up(conn)

        assert await _subject_column(scratch_pool) == ("text", "YES", None)
        assert len(await _index_definitions(scratch_pool)) == 3

    async def test_schema_facts_has_subject_flips(self, scratch_pool, scratch_config, monkeypatch):
        from osprey.services.ariel_search.database import ARIELRepository, migrations

        monkeypatch.setattr(migrations, "KNOWN_MIGRATIONS", _without_migration())
        await migrations.run_migrations(scratch_pool, scratch_config)
        repo = ARIELRepository(scratch_pool, scratch_config)
        assert (await repo.schema_facts()).has_subject is False

        monkeypatch.undo()
        assert await migrations.run_migrations(scratch_pool, scratch_config) == [
            MIGRATION,
            "entry_subject_backfill",
        ]
        repo.invalidate_schema_facts()
        assert (await repo.schema_facts()).has_subject is True


class TestPreMigrationIngest:
    async def test_upsert_on_the_pre_migration_schema_succeeds(
        self, scratch_pool, scratch_config, seed_entry_factory, monkeypatch
    ):
        from osprey.services.ariel_search.database import ARIELRepository, migrations

        monkeypatch.setattr(migrations, "KNOWN_MIGRATIONS", _without_migration())
        await migrations.run_migrations(scratch_pool, scratch_config)
        assert await _subject_column(scratch_pool) is None

        repo = ARIELRepository(scratch_pool, scratch_config)
        entry = seed_entry_factory(entry_id="subject-pre-001", raw_text="pre-migration")
        await repo.upsert_entry(entry)
        await repo.upsert_entry(entry)  # the ON CONFLICT path too

        stored = await repo.get_entry("subject-pre-001")
        assert stored is not None
        assert stored["raw_text"] == "pre-migration"


class TestPreserveUpdatedAt:
    async def _seed(self, pool, config) -> None:
        from osprey.services.ariel_search.database import run_migrations

        await run_migrations(pool, config)
        async with pool.connection() as conn:
            await conn.execute("""
                INSERT INTO enhanced_entries
                    (entry_id, source_system, timestamp, author, raw_text, updated_at)
                VALUES ('subject-ts-001', 'test', NOW(), 'a', 'x',
                        TIMESTAMPTZ '2020-01-01 00:00:00+00')
            """)

    @staticmethod
    async def _updated_at(pool):
        async with pool.connection() as conn:
            result = await conn.execute(
                "SELECT updated_at FROM enhanced_entries WHERE entry_id = 'subject-ts-001'"
            )
            return (await result.fetchone())[0]

    async def test_update_moves_updated_at(self, scratch_pool, scratch_config):
        await self._seed(scratch_pool, scratch_config)
        before = await self._updated_at(scratch_pool)

        async with scratch_pool.connection() as conn:
            await conn.execute(
                "UPDATE enhanced_entries SET subject = 's' WHERE entry_id = 'subject-ts-001'"
            )

        assert await self._updated_at(scratch_pool) > before

    async def test_preserve_setting_keeps_updated_at(self, scratch_pool, scratch_config):
        await self._seed(scratch_pool, scratch_config)
        before = await self._updated_at(scratch_pool)

        async with scratch_pool.connection() as conn, conn.transaction():
            await conn.execute("SET LOCAL ariel.preserve_updated_at = 'on'")
            await conn.execute(
                "UPDATE enhanced_entries SET subject = 's' WHERE entry_id = 'subject-ts-001'"
            )

        assert await self._updated_at(scratch_pool) == before

    async def test_setting_is_transaction_local(self, scratch_pool, scratch_config):
        """After the preserving transaction ends, a later update moves ``updated_at`` again."""
        await self._seed(scratch_pool, scratch_config)
        before = await self._updated_at(scratch_pool)

        async with scratch_pool.connection() as conn:
            async with conn.transaction():
                await conn.execute("SET LOCAL ariel.preserve_updated_at = 'on'")
            await conn.execute(
                "UPDATE enhanced_entries SET subject = 't' WHERE entry_id = 'subject-ts-001'"
            )

        assert await self._updated_at(scratch_pool) > before
