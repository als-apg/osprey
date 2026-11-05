"""Integration tests for the stored-subject write and the ``has_images`` read.

Runs against a fresh scratch database per test: both upsert paths write the
``subject`` column (C-SUBJ), a re-upsert without a subject keeps the stored one,
the upsert degrades on the pre-migration schema, and ``get_entry`` /
``get_entries_by_ids`` select ``has_images`` (C-ENH).
"""

from __future__ import annotations

import pytest

from tests.services.ariel_search.integration._support import insert_picture

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


@pytest.fixture
async def repo(scratch_pool, scratch_config):
    """A repository on the scratch database at today's schema."""
    from osprey.services.ariel_search.database import ARIELRepository, run_migrations

    await run_migrations(scratch_pool, scratch_config)
    repository = ARIELRepository(scratch_pool, scratch_config)
    assert (await repository.schema_facts()).has_subject
    return repository


async def _stored_subject(pool, entry_id: str):
    async with pool.connection() as conn:
        result = await conn.execute(
            "SELECT subject FROM enhanced_entries WHERE entry_id = %s", [entry_id]
        )
        row = await result.fetchone()
        return row[0] if row else None


UPSERTS = ["upsert_entry", "upsert_entry_returning"]


@pytest.mark.parametrize("method", UPSERTS)
class TestSubjectWrite:
    async def test_als_subject_from_metadata(self, repo, scratch_pool, seed_entry_factory, method):
        entry = seed_entry_factory(
            entry_id="subj-als",
            raw_text="Beam dump\n\nThe beam dumped.",
            metadata={"subject": "Beam dump"},
        )
        await getattr(repo, method)(entry)
        assert await _stored_subject(scratch_pool, "subj-als") == "Beam dump"
        stored = await repo.get_entry("subj-als")
        assert stored["subject"] == "Beam dump"
        # the column, never metadata.subject written by the upsert
        assert stored["metadata"] == {"subject": "Beam dump"}

    async def test_title_facility(self, repo, scratch_pool, seed_entry_factory, method):
        entry = seed_entry_factory(
            entry_id="subj-jlab", raw_text="Body line", metadata={"title": "JLab title"}
        )
        await getattr(repo, method)(entry)
        assert await _stored_subject(scratch_pool, "subj-jlab") == "JLab title"
        assert "subject" not in (await repo.get_entry("subj-jlab"))["metadata"]

    async def test_untitled_first_line(self, repo, scratch_pool, seed_entry_factory, method):
        entry = seed_entry_factory(entry_id="subj-untitled", raw_text="\n  First line \nmore")
        await getattr(repo, method)(entry)
        assert await _stored_subject(scratch_pool, "subj-untitled") == "First line"

    async def test_reupsert_without_subject_keeps_stored(
        self, repo, scratch_pool, seed_entry_factory, method
    ):
        upsert = getattr(repo, method)
        await upsert(
            seed_entry_factory(entry_id="subj-keep", raw_text="Kept", metadata={"subject": "Kept"})
        )
        await upsert(seed_entry_factory(entry_id="subj-keep", raw_text="   \n"))
        assert await _stored_subject(scratch_pool, "subj-keep") == "Kept"

    async def test_reupsert_with_subject_replaces(
        self, repo, scratch_pool, seed_entry_factory, method
    ):
        upsert = getattr(repo, method)
        await upsert(seed_entry_factory(entry_id="subj-new", raw_text="Old"))
        await upsert(seed_entry_factory(entry_id="subj-new", raw_text="New"))
        assert await _stored_subject(scratch_pool, "subj-new") == "New"


@pytest.mark.parametrize("method", UPSERTS)
async def test_upsert_on_the_pre_migration_schema(
    scratch_pool, scratch_config, seed_entry_factory, monkeypatch, method
):
    from osprey.services.ariel_search.database import ARIELRepository, migrations

    monkeypatch.setattr(
        migrations,
        "KNOWN_MIGRATIONS",
        [row for row in migrations.KNOWN_MIGRATIONS if row[0] != MIGRATION],
    )
    await migrations.run_migrations(scratch_pool, scratch_config)
    repo = ARIELRepository(scratch_pool, scratch_config)
    assert not (await repo.schema_facts()).has_subject

    entry = seed_entry_factory(entry_id="subj-pre", raw_text="pre", metadata={"subject": "S"})
    await getattr(repo, method)(entry)
    await getattr(repo, method)(entry)  # the ON CONFLICT path too
    stored = await repo.get_entry("subj-pre")
    assert stored is not None
    assert "subject" not in stored
    assert stored["has_images"] is False


class TestHasImages:
    async def test_get_entry_true_for_viewable_picture(
        self, repo, scratch_database, seed_entry_factory
    ):
        await repo.upsert_entry(seed_entry_factory(entry_id="img-yes"))
        await repo.upsert_entry(seed_entry_factory(entry_id="img-pending"))
        await repo.upsert_entry(seed_entry_factory(entry_id="img-none"))
        insert_picture(scratch_database, "att-yes", "img-yes", copy_status="copied")
        insert_picture(scratch_database, "att-pending", "img-pending", copy_status="pending")

        assert (await repo.get_entry("img-yes"))["has_images"] is True
        assert (await repo.get_entry("img-pending"))["has_images"] is False
        assert (await repo.get_entry("img-none"))["has_images"] is False

        by_id = {
            e["entry_id"]: e["has_images"]
            for e in await repo.get_entries_by_ids(["img-yes", "img-pending", "img-none"])
        }
        assert by_id == {"img-yes": True, "img-pending": False, "img-none": False}
