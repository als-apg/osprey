"""Integration tests for the ``entry_subject_backfill`` migration.

Runs against a fresh scratch database per test. Rows are seeded on a schema
that has the ``subject`` column but not the backfill, then the backfill runs:
every row of each shape gets :func:`derive_subject`'s value, ``updated_at``,
``metadata`` and ``enhancement_status`` stay as they were, a stored subject is
never overwritten, a second run changes nothing, and publish sends the stored
subject.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import patch

import pytest

# xdist_group("docker"): every container-starting file shares one worker; see
# test_migrations.py for the full reason.
pytestmark = [pytest.mark.asyncio, pytest.mark.xdist_group("docker")]

MIGRATION = "entry_subject_backfill"
COLUMN_MIGRATION = "entry_subject_and_keyset"

#: A fixed, old ``updated_at`` every seeded row carries; the backfill must keep it.
OLD_UPDATED_AT = "2020-01-01 00:00:00+00"

LONG_LINE = "x" * 250

#: (entry_id, raw_text, metadata, enhancement_status, preset subject)
SHAPES: list[tuple[str, str, dict, dict, str | None]] = [
    ("bf-als", "Beam dump\n\nDetails", {"subject": "Beam dump", "tags": ["ops"]}, {}, None),
    ("bf-jlab", "Body only", {"title": "JLab title"}, {"text_embedding": {"status": "done"}}, None),
    ("bf-untitled", "\n   \n  First line  \nsecond", {}, {}, None),
    ("bf-blank", "   \n\t\n", {}, {"qmd_export": {"status": "done"}}, None),
    ("bf-long", LONG_LINE, {}, {}, None),
    ("bf-blank-subject", "Fallback line", {"subject": "  ", "title": ""}, {}, None),
    ("bf-preset", "Would derive this", {"subject": "Would derive this"}, {}, "Kept as stored"),
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


async def _seed(pool, shapes=SHAPES) -> None:
    async with pool.connection() as conn:
        for entry_id, raw_text, metadata, status, subject in shapes:
            await conn.execute(
                """
                INSERT INTO enhanced_entries
                    (entry_id, source_system, timestamp, author, raw_text,
                     metadata, enhancement_status, subject, updated_at)
                VALUES (%s, 'test', NOW(), 'a', %s, %s::jsonb, %s::jsonb, %s,
                        %s::timestamptz)
                """,
                [
                    entry_id,
                    raw_text,
                    json.dumps(metadata),
                    json.dumps(status),
                    subject,
                    OLD_UPDATED_AT,
                ],
            )


async def _rows(pool) -> dict[str, tuple]:
    async with pool.connection() as conn:
        result = await conn.execute(
            "SELECT entry_id, subject, updated_at, metadata, enhancement_status "
            "FROM enhanced_entries ORDER BY entry_id"
        )
        return {row[0]: row[1:] for row in await result.fetchall()}


async def _seeded_store(pool, config, monkeypatch, shapes=SHAPES) -> None:
    """Migrate to everything but the backfill, then seed rows with ``subject`` NULL."""
    from osprey.services.ariel_search.database import migrations

    monkeypatch.setattr(migrations, "KNOWN_MIGRATIONS", _without(MIGRATION))
    await migrations.run_migrations(pool, config)
    monkeypatch.undo()
    await _seed(pool, shapes)


def _expected(raw_text: str, metadata: dict, preset: str | None) -> str:
    from osprey.services.ariel_search.subject import derive_subject

    if preset is not None:
        return preset
    return derive_subject({"metadata": metadata, "raw_text": raw_text})


class TestRegistration:
    async def test_registered_always_on_after_the_column(self):
        from osprey.services.ariel_search.database import migrations
        from osprey.services.ariel_search.database.subject_backfill import (
            EntrySubjectBackfillMigration,
        )

        names = [row[0] for row in migrations.KNOWN_MIGRATIONS]
        (row,) = [r for r in migrations.KNOWN_MIGRATIONS if r[0] == MIGRATION]
        assert row == (
            MIGRATION,
            "osprey.services.ariel_search.database.subject_backfill",
            "EntrySubjectBackfillMigration",
            None,
        )
        assert names.index(COLUMN_MIGRATION) < names.index(MIGRATION)
        migration = EntrySubjectBackfillMigration()
        assert migration.name == MIGRATION
        assert migration.depends_on == [COLUMN_MIGRATION]


class TestBackfill:
    async def test_each_shape_gets_the_derived_subject(
        self, scratch_pool, scratch_config, monkeypatch
    ):
        from osprey.services.ariel_search.database import run_migrations

        await _seeded_store(scratch_pool, scratch_config, monkeypatch)
        before = await _rows(scratch_pool)
        assert all(before[s[0]][0] is None for s in SHAPES if s[4] is None)

        assert await run_migrations(scratch_pool, scratch_config) == [MIGRATION]

        after = await _rows(scratch_pool)
        for entry_id, raw_text, metadata, _status, preset in SHAPES:
            assert after[entry_id][0] == _expected(raw_text, metadata, preset), entry_id
        assert after["bf-als"][0] == "Beam dump"
        assert after["bf-jlab"][0] == "JLab title"
        assert after["bf-untitled"][0] == "First line"
        assert after["bf-blank"][0] == ""
        assert after["bf-long"][0] == "x" * 200
        assert after["bf-blank-subject"][0] == "Fallback line"
        assert after["bf-preset"][0] == "Kept as stored"

    async def test_updated_at_metadata_and_status_unchanged(
        self, scratch_pool, scratch_config, monkeypatch
    ):
        from osprey.services.ariel_search.database import run_migrations

        await _seeded_store(scratch_pool, scratch_config, monkeypatch)
        before = await _rows(scratch_pool)

        await run_migrations(scratch_pool, scratch_config)

        after = await _rows(scratch_pool)
        for entry_id in before:
            assert after[entry_id][1:] == before[entry_id][1:], entry_id
        assert after["bf-jlab"][3] == {"text_embedding": {"status": "done"}}
        assert after["bf-als"][2] == {"subject": "Beam dump", "tags": ["ops"]}

    async def test_pages_by_keyset(self, scratch_pool, scratch_config, monkeypatch):
        """Pages smaller than the store still reach every row, exactly once."""
        from osprey.services.ariel_search.database import run_migrations, subject_backfill

        shapes = [(f"bf-page-{i:02d}", f"Line {i}", {}, {}, None) for i in range(7)]
        await _seeded_store(scratch_pool, scratch_config, monkeypatch, shapes)
        monkeypatch.setattr(subject_backfill, "BACKFILL_PAGE_SIZE", 3)

        assert await run_migrations(scratch_pool, scratch_config) == [MIGRATION]

        after = await _rows(scratch_pool)
        assert {k: v[0] for k, v in after.items()} == {s[0]: s[1] for s in shapes}

    async def test_logs_row_count_and_lock_scope(
        self, scratch_pool, scratch_config, monkeypatch, caplog
    ):
        from osprey.services.ariel_search.database import run_migrations

        await _seeded_store(scratch_pool, scratch_config, monkeypatch)
        caplog.set_level(logging.INFO, logger="ariel")

        await run_migrations(scratch_pool, scratch_config)

        backfilled = sum(1 for s in SHAPES if s[4] is None)
        assert f"{MIGRATION}: {backfilled} rows backfilled" in caplog.text
        assert "row locks held until the migration commits" in caplog.text


class TestIdempotence:
    async def test_second_run_is_a_noop(self, scratch_pool, scratch_config, monkeypatch, caplog):
        from osprey.services.ariel_search.database import run_migrations
        from osprey.services.ariel_search.database.subject_backfill import (
            EntrySubjectBackfillMigration,
        )

        await _seeded_store(scratch_pool, scratch_config, monkeypatch)
        await run_migrations(scratch_pool, scratch_config)
        first = await _rows(scratch_pool)

        assert await run_migrations(scratch_pool, scratch_config) == []
        caplog.set_level(logging.INFO, logger="ariel")
        async with scratch_pool.connection() as conn, conn.transaction():
            await EntrySubjectBackfillMigration().up(conn)

        assert await _rows(scratch_pool) == first
        assert f"{MIGRATION}: 0 rows backfilled" in caplog.text

    async def test_skipped_without_the_column(self, scratch_pool, scratch_config, monkeypatch):
        """On a schema without ``subject`` the backfill skips, and runs once the column exists."""
        from osprey.services.ariel_search.database import migrations

        monkeypatch.setattr(migrations, "KNOWN_MIGRATIONS", _without(COLUMN_MIGRATION))
        applied = await migrations.run_migrations(scratch_pool, scratch_config)
        assert MIGRATION not in applied

        monkeypatch.undo()
        assert await migrations.run_migrations(scratch_pool, scratch_config) == [
            COLUMN_MIGRATION,
            MIGRATION,
        ]


class TestPublish:
    async def test_publish_sends_the_stored_subject(
        self, scratch_pool, scratch_config, monkeypatch
    ):
        from osprey.services.ariel_search.database import ARIELRepository, run_migrations
        from tests.services.ariel_search._service_doubles import make_mock_service

        await _seeded_store(scratch_pool, scratch_config, monkeypatch)
        await run_migrations(scratch_pool, scratch_config)
        repo = ARIELRepository(scratch_pool, scratch_config)

        service, adapter, mock_repository = make_mock_service(adapter_supports_write=True)
        mock_repository.get_entry = repo.get_entry
        with patch("osprey.services.ariel_search.ingestion.get_adapter", return_value=adapter):
            await service.publish_entry("bf-jlab")
            await service.publish_entry("bf-preset")

        subjects = [call.args[0].subject for call in adapter.create_entry.call_args_list]
        assert subjects == ["JLab title", "Kept as stored"]
        assert (await repo.get_entry("bf-jlab"))["subject"] == "JLab title"


class TestMigrateCommand:
    """The backfill as an operator runs it: ``osprey ariel migrate``."""

    async def _migrate_via_cli(self, scratch_database: str) -> None:
        """Invoke the ``migrate`` click command (it owns its own event loop)."""
        import asyncio

        from click.testing import CliRunner

        from osprey.cli import ariel as ariel_cli

        def _invoke():
            with patch.object(
                ariel_cli,
                "_load_ariel_config",
                return_value={"database": {"uri": scratch_database}},
            ):
                return CliRunner().invoke(
                    ariel_cli.ariel_group, ["migrate"], catch_exceptions=False
                )

        result = await asyncio.to_thread(_invoke)
        assert result.exit_code == 0, result.output

    async def test_every_entry_has_a_subject_list_and_reader_agree(
        self, scratch_pool, scratch_config, scratch_database, monkeypatch
    ):
        from osprey.interfaces.ariel.api import routes
        from osprey.services.ariel_search.database import ARIELRepository
        from osprey.services.ariel_search.service import ARIELSearchService

        await _seeded_store(scratch_pool, scratch_config, monkeypatch)

        await self._migrate_via_cli(scratch_database)

        stored = {k: v[0] for k, v in (await _rows(scratch_pool)).items()}
        assert all(subject is not None for subject in stored.values()), stored
        repo = ARIELRepository(scratch_pool, scratch_config)
        service = ARIELSearchService(config=scratch_config, pool=scratch_pool, repository=repo)
        listed = await repo.search_by_time_range(limit=100)
        list_rows = {
            r.entry_id: r.subject for r in await routes._page_responses(service, listed, "row")
        }
        for entry_id, raw_text, metadata, _status, preset in SHAPES:
            expected = _expected(raw_text, metadata, preset)
            reader = (await routes._entry_responses(service, [await repo.get_entry(entry_id)]))[0]
            assert stored[entry_id] == expected, entry_id
            assert list_rows[entry_id] == expected, entry_id
            assert reader.subject == expected, entry_id

    async def test_reingest_after_the_backfill_keeps_the_subject(
        self, scratch_pool, scratch_config, scratch_database, monkeypatch, seed_entry_factory
    ):
        """Upserting each entry again, as ingestion would, leaves its backfilled subject."""
        from osprey.services.ariel_search.database import ARIELRepository

        await _seeded_store(scratch_pool, scratch_config, monkeypatch)
        await self._migrate_via_cli(scratch_database)
        backfilled = {k: v[0] for k, v in (await _rows(scratch_pool)).items()}
        repo = ARIELRepository(scratch_pool, scratch_config)

        for entry_id, raw_text, metadata, status, preset in SHAPES:
            if preset is not None:
                continue  # its stored subject never came from the backfill
            entry = seed_entry_factory(
                entry_id=entry_id, raw_text=raw_text, metadata=metadata, enhancement_status=status
            )
            await repo.upsert_entry(entry)

        after = {k: v[0] for k, v in (await _rows(scratch_pool)).items()}
        assert after == backfilled
