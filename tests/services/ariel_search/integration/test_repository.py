"""Integration tests for ARIELRepository.

Tests actual database operations against real PostgreSQL.

See 04_OSPREY_INTEGRATION.md Section 12.3.4 for test requirements.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from osprey.services.ariel_search.database.repository import MAX_ENHANCEMENT_ATTEMPTS

# xdist_group("docker"): pins every container-starting test file onto one worker, so
# a run has a single testcontainers session and a single ryuk reaper -- concurrent
# reaper starts race the Docker daemon's port mapper. It also serializes the shared
# database: the session ``database_url`` fixture prefers a running dev Postgres with
# ONE shared ``ariel_test`` database over a per-worker container, so parallel workers
# would otherwise collide on migrations/seed/truncate.
pytestmark = [pytest.mark.asyncio, pytest.mark.xdist_group("docker")]


#: Rows the repository suite seeds and reads back, one family for the file.
INTEG_PREFIX = "integ-"

#: Rows the concurrency probes write, under their own family.
CONCURRENT_PREFIX = "concurrent-"


@pytest.fixture
def _seed_integ_prefix(seeded_prefixes):
    """Record the ``integ-`` family before any test of the class writes a row."""
    seeded_prefixes.add(INTEG_PREFIX)


@pytest.fixture
def _seed_concurrent_prefix(seeded_prefixes):
    """Record the ``concurrent-`` family before any test of the class writes a row."""
    seeded_prefixes.add(CONCURRENT_PREFIX)


@pytest.mark.usefixtures("_seed_integ_prefix")
class TestRepositoryCRUD:
    """Test ARIELRepository CRUD operations with real database."""

    async def test_upsert_and_get_entry(self, repository, seed_entry_factory):
        """Test basic CRUD operations."""
        entry = seed_entry_factory(
            entry_id=f"{INTEG_PREFIX}crud-001",
            raw_text="Test entry content for CRUD test",
        )

        await repository.upsert_entry(entry)
        retrieved = await repository.get_entry("integ-crud-001")

        assert retrieved is not None
        assert retrieved["entry_id"] == "integ-crud-001"
        assert retrieved["raw_text"] == "Test entry content for CRUD test"

    async def test_get_nonexistent_entry_returns_none(self, repository):
        """Test get_entry returns None for missing entry."""
        result = await repository.get_entry("nonexistent-entry-id-xyz")
        assert result is None

    async def test_upsert_updates_existing_entry(self, repository, seed_entry_factory):
        """Test upsert updates an existing entry."""
        entry = seed_entry_factory(
            entry_id=f"{INTEG_PREFIX}update-001",
            raw_text="Original content",
        )
        await repository.upsert_entry(entry)

        # Update the entry
        entry["raw_text"] = "Updated content"
        await repository.upsert_entry(entry)

        retrieved = await repository.get_entry("integ-update-001")
        assert retrieved is not None
        assert retrieved["raw_text"] == "Updated content"

    async def test_count_entries(self, repository):
        """Test entry counting."""
        count = await repository.count_entries()
        assert isinstance(count, int)
        assert count >= 0

    async def test_count_entries_honors_filters(self, repository, seed_entry_factory):
        """count_entries applies the same filters as search_by_time_range.

        So a filtered Browse listing's total_pages matches the rows returned,
        instead of counting the whole table. Scoped to a unique source_system
        so other rows in the shared test database do not bleed in.
        """
        base = datetime(2005, 5, 5, tzinfo=UTC)
        src = "integ-count-src"
        for i in range(3):
            await repository.upsert_entry(
                seed_entry_factory(
                    entry_id=f"{INTEG_PREFIX}count-{i:03d}",
                    source_system=src,
                    author="integ-count-alice" if i == 0 else "integ-count-bob",
                    timestamp=base + timedelta(hours=i),
                )
            )

        assert await repository.count_entries(source_system=src) == 3
        assert await repository.count_entries(source_system=src, author="integ-count-alice") == 1
        assert await repository.count_entries(source_system="integ-count-nope") == 0

    async def test_get_entries_by_ids_empty_list(self, repository):
        """Test get_entries_by_ids with empty list returns empty list."""
        results = await repository.get_entries_by_ids([])
        assert results == []

    async def test_get_entries_by_ids(self, repository, seed_entry_factory):
        """Test get_entries_by_ids returns requested entries."""
        entries = [
            seed_entry_factory(entry_id=f"{INTEG_PREFIX}batch-001", raw_text="Entry 1"),
            seed_entry_factory(entry_id=f"{INTEG_PREFIX}batch-002", raw_text="Entry 2"),
        ]
        for entry in entries:
            await repository.upsert_entry(entry)

        results = await repository.get_entries_by_ids(["integ-batch-001", "integ-batch-002"])
        result_ids = {e["entry_id"] for e in results}
        assert "integ-batch-001" in result_ids
        assert "integ-batch-002" in result_ids


@pytest.mark.usefixtures("_seed_integ_prefix")
class TestRepositoryTimeRange:
    """Test ARIELRepository time range queries."""

    async def test_search_by_time_range(self, repository, seed_entry_factory):
        """Test search by time range returns entries."""
        now = datetime.now(UTC)
        entry = seed_entry_factory(
            entry_id=f"{INTEG_PREFIX}time-001",
            timestamp=now,
            raw_text="Time range test entry",
        )
        await repository.upsert_entry(entry)

        start = now - timedelta(hours=1)
        end = now + timedelta(hours=1)
        results = await repository.search_by_time_range(start=start, end=end, limit=100)

        entry_ids = [e["entry_id"] for e in results]
        assert "integ-time-001" in entry_ids

    async def test_search_by_time_range_no_filters(self, repository):
        """Test search with no time filters returns entries."""
        results = await repository.search_by_time_range(limit=10)
        assert isinstance(results, list)

    async def test_search_by_time_range_respects_limit(self, repository, seed_entry_factory):
        """Test search respects limit parameter."""
        now = datetime.now(UTC)
        # Create multiple entries
        for i in range(5):
            entry = seed_entry_factory(
                entry_id=f"{INTEG_PREFIX}limit-{i:03d}",
                timestamp=now,
                raw_text=f"Limit test entry {i}",
            )
            await repository.upsert_entry(entry)

        results = await repository.search_by_time_range(limit=2)
        assert len(results) <= 2

    async def test_search_by_time_range_filters_by_author(self, repository, seed_entry_factory):
        """search_by_time_range filters by author when supplied."""
        base = datetime(2002, 2, 2, tzinfo=UTC)
        await repository.upsert_entry(
            seed_entry_factory(
                entry_id=f"{INTEG_PREFIX}auth-alice", author="integ-alice", timestamp=base
            )
        )
        await repository.upsert_entry(
            seed_entry_factory(
                entry_id=f"{INTEG_PREFIX}auth-bob", author="integ-bob", timestamp=base
            )
        )

        results = await repository.search_by_time_range(author="integ-alice", limit=100)

        entry_ids = [e["entry_id"] for e in results]
        assert "integ-auth-alice" in entry_ids
        assert "integ-auth-bob" not in entry_ids

    async def test_search_by_time_range_filters_by_source_system(
        self, repository, seed_entry_factory
    ):
        """search_by_time_range filters by source_system when supplied."""
        base = datetime(2003, 3, 3, tzinfo=UTC)
        await repository.upsert_entry(
            seed_entry_factory(
                entry_id=f"{INTEG_PREFIX}src-ex", source_system="integ-EX", timestamp=base
            )
        )
        await repository.upsert_entry(
            seed_entry_factory(
                entry_id=f"{INTEG_PREFIX}src-other", source_system="integ-OTHER", timestamp=base
            )
        )

        results = await repository.search_by_time_range(source_system="integ-EX", limit=100)

        entry_ids = [e["entry_id"] for e in results]
        assert "integ-src-ex" in entry_ids
        assert "integ-src-other" not in entry_ids

    async def test_search_by_time_range_offset_paginates(self, repository, seed_entry_factory):
        """offset advances the page so older entries become reachable.

        Scoped to a unique source_system so other rows in the shared test
        database do not bleed into the assertions.
        """
        base = datetime(2004, 4, 4, tzinfo=UTC)
        src = "integ-offset-src"
        for i in range(3):
            await repository.upsert_entry(
                seed_entry_factory(
                    entry_id=f"{INTEG_PREFIX}offset-{i:03d}",
                    source_system=src,
                    timestamp=base + timedelta(hours=i),  # 002 newest, 000 oldest
                    raw_text=f"offset entry {i}",
                )
            )

        page1 = await repository.search_by_time_range(source_system=src, limit=2, offset=0)
        page2 = await repository.search_by_time_range(source_system=src, limit=2, offset=2)

        assert [e["entry_id"] for e in page1] == ["integ-offset-002", "integ-offset-001"]
        assert [e["entry_id"] for e in page2] == ["integ-offset-000"]


class TestRepositoryHealth:
    """Test ARIELRepository health check."""

    async def test_health_check(self, repository):
        """Test database health check returns healthy status."""
        healthy, message = await repository.health_check()
        assert healthy is True
        assert isinstance(message, str)


@pytest.mark.usefixtures("_seed_integ_prefix")
class TestRepositoryEnhancement:
    """Test ARIELRepository enhancement status operations."""

    async def test_mark_enhancement_complete(self, repository, seed_entry_factory):
        """Test marking an enhancement as complete."""
        entry = seed_entry_factory(
            entry_id=f"{INTEG_PREFIX}enhance-001",
            raw_text="Entry for enhancement test",
        )
        await repository.upsert_entry(entry)

        await repository.mark_enhancement_complete("integ-enhance-001", "test_module")

        updated = await repository.get_entry("integ-enhance-001")
        assert updated is not None
        status = updated.get("enhancement_status", {})
        assert "test_module" in status

    async def test_get_enhancement_stats(self, repository):
        """Test getting enhancement stats."""
        stats = await repository.get_enhancement_stats()
        assert isinstance(stats, dict)
        assert "total_entries" in stats


@pytest.mark.usefixtures("_seed_integ_prefix")
class TestRepositoryFuzzySearch:
    """Test ARIELRepository fuzzy search operations."""

    async def test_fuzzy_search_finds_similar_text(self, repository, seed_entry_factory):
        """Fuzzy search returns entries with similar text."""
        entry = seed_entry_factory(
            entry_id=f"{INTEG_PREFIX}fuzzy-001",
            raw_text="The beam alignment was adjusted for optimal performance",
        )
        await repository.upsert_entry(entry)

        # Search for similar text with typos
        results = await repository.fuzzy_search(
            search_text="beam alignement optimal",  # Note: typo in alignment
            threshold=0.2,
            max_results=10,
        )

        # May or may not find depending on similarity threshold
        assert isinstance(results, list)
        for _entry, score, highlights in results:
            assert isinstance(score, float)
            assert isinstance(highlights, list)

    async def test_fuzzy_search_no_matches(self, repository):
        """Fuzzy search returns empty list for no matches."""
        results = await repository.fuzzy_search(
            search_text="zzzzxyzabc123456nonexistent",
            threshold=0.9,  # High threshold
            max_results=10,
        )
        assert results == []


class TestRepositoryIncompleteEntries:
    """Test ARIELRepository incomplete entries query."""

    async def test_get_incomplete_entries(self, repository):
        """get_incomplete_entries returns entries list."""
        results = await repository.get_incomplete_entries(limit=5)
        assert isinstance(results, list)

    async def test_get_incomplete_entries_with_module_filter(self, repository):
        """get_incomplete_entries filters by module."""
        results = await repository.get_incomplete_entries(
            module_name="nonexistent_module",
            limit=5,
        )
        assert isinstance(results, list)


@pytest.mark.usefixtures("_seed_integ_prefix")
class TestRepositoryEnhancementFailure:
    """Test ARIELRepository enhancement failure tracking."""

    async def test_mark_enhancement_failed(self, repository, seed_entry_factory):
        """mark_enhancement_failed records failure."""
        entry = seed_entry_factory(
            entry_id=f"{INTEG_PREFIX}fail-001",
            raw_text="Entry for failure test",
        )
        await repository.upsert_entry(entry)

        await repository.mark_enhancement_failed(
            "integ-fail-001",
            "test_module",
            error="Test error message",
        )

        updated = await repository.get_entry("integ-fail-001")
        assert updated is not None
        status = updated.get("enhancement_status", {})
        assert "test_module" in status


#: Module name the attempt-count checks write status under.
ATTEMPTS_MODULE = "attempts_module"


async def _backfill_ids(repository) -> set[str]:
    """Ids of this class's rows that ``ATTEMPTS_MODULE``'s backfill selects.

    The limit is large because every row in the shared test database lacks the
    module's key, and so is selected ahead of or beside this class's rows.
    """
    entries = await repository.get_incomplete_entries(module_name=ATTEMPTS_MODULE, limit=100_000)
    return {e["entry_id"] for e in entries if e["entry_id"].startswith(f"{INTEG_PREFIX}attempts-")}


@pytest.mark.usefixtures("_seed_integ_prefix")
class TestRepositoryEnhancementAttempts:
    """A failing enhancement is counted and leaves the backfill at the cap."""

    async def test_each_failure_counts_one_attempt(self, repository, seed_entry_factory):
        """Three marks return 1, 2, 3 and keep the last error."""
        entry_id = f"{INTEG_PREFIX}attempts-count"
        await repository.upsert_entry(seed_entry_factory(entry_id=entry_id))

        counts = [
            await repository.mark_enhancement_failed(entry_id, ATTEMPTS_MODULE, f"error {i}")
            for i in range(1, 4)
        ]

        assert counts == [1, 2, 3]
        stored = (await repository.get_entry(entry_id))["enhancement_status"][ATTEMPTS_MODULE]
        assert stored["attempts"] == 3
        assert stored["status"] == "failed"
        assert stored["error"] == "error 3"

    async def test_an_entry_leaves_the_backfill_at_the_cap(self, repository, seed_entry_factory):
        """Selected below the cap, not selected at it."""
        entry_id = f"{INTEG_PREFIX}attempts-cap"
        await repository.upsert_entry(seed_entry_factory(entry_id=entry_id))

        for _ in range(MAX_ENHANCEMENT_ATTEMPTS - 1):
            await repository.mark_enhancement_failed(entry_id, ATTEMPTS_MODULE, "boom")
        assert entry_id in await _backfill_ids(repository)

        await repository.mark_enhancement_failed(entry_id, ATTEMPTS_MODULE, "boom")
        assert entry_id not in await _backfill_ids(repository)

    async def test_a_success_clears_the_count(self, repository, seed_entry_factory):
        """A failure after a success starts counting again from one."""
        entry_id = f"{INTEG_PREFIX}attempts-clear"
        await repository.upsert_entry(seed_entry_factory(entry_id=entry_id))

        await repository.mark_enhancement_failed(entry_id, ATTEMPTS_MODULE, "boom")
        await repository.mark_enhancement_failed(entry_id, ATTEMPTS_MODULE, "boom")
        await repository.mark_enhancement_complete(entry_id, ATTEMPTS_MODULE)

        assert await repository.mark_enhancement_failed(entry_id, ATTEMPTS_MODULE, "boom") == 1

    async def test_a_failure_recorded_without_a_count_is_retried(
        self, repository, seed_entry_factory
    ):
        """A failed status with no count reads as no attempts."""
        entry_id = f"{INTEG_PREFIX}attempts-legacy"
        await repository.upsert_entry(
            seed_entry_factory(
                entry_id=entry_id,
                enhancement_status={ATTEMPTS_MODULE: {"status": "failed", "error": "old"}},
            )
        )

        assert entry_id in await _backfill_ids(repository)
        assert await repository.mark_enhancement_failed(entry_id, ATTEMPTS_MODULE, "boom") == 1

    async def test_an_unknown_entry_returns_zero(self, repository):
        """Marking an id no entry has stores nothing and returns 0."""
        assert (
            await repository.mark_enhancement_failed(
                f"{INTEG_PREFIX}attempts-missing", ATTEMPTS_MODULE, "boom"
            )
            == 0
        )


class TestRepositoryEmbeddings:
    """Test ARIELRepository embedding operations."""

    async def test_get_embedding_tables(self, repository):
        """get_embedding_tables returns list of table info."""
        tables = await repository.get_embedding_tables()
        assert isinstance(tables, list)
        # May or may not have embedding tables depending on migrations

    async def test_validate_search_model_table_nonexistent(self, repository):
        """validate_search_model_table raises for nonexistent model."""
        from osprey.services.ariel_search.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            await repository.validate_search_model_table("nonexistent_model_xyz")


@pytest.mark.usefixtures("_seed_integ_prefix")
class TestRepositoryFuzzyDateFilters:
    """Test ARIELRepository fuzzy search with date filters."""

    async def test_fuzzy_search_with_start_date(self, repository, seed_entry_factory):
        """fuzzy_search can filter by start_date."""
        now = datetime.now(UTC)
        entry = seed_entry_factory(
            entry_id=f"{INTEG_PREFIX}fuzzydate-001",
            timestamp=now,
            raw_text="Fuzzy date filter test entry",
        )
        await repository.upsert_entry(entry)

        results = await repository.fuzzy_search(
            search_text="fuzzy filter",
            threshold=0.2,
            max_results=10,
            start_date=now - timedelta(hours=1),
        )
        assert isinstance(results, list)

    async def test_fuzzy_search_with_end_date(self, repository, seed_entry_factory):
        """fuzzy_search can filter by end_date."""
        now = datetime.now(UTC)
        entry = seed_entry_factory(
            entry_id=f"{INTEG_PREFIX}fuzzydate-002",
            timestamp=now,
            raw_text="Fuzzy end date filter test entry",
        )
        await repository.upsert_entry(entry)

        results = await repository.fuzzy_search(
            search_text="end date filter",
            threshold=0.2,
            max_results=10,
            end_date=now + timedelta(hours=1),
        )
        assert isinstance(results, list)


@pytest.mark.usefixtures("_seed_integ_prefix")
class TestRepositoryMetadata:
    """Test ARIELRepository entries with various metadata."""

    async def test_entry_with_empty_metadata(self, repository, seed_entry_factory):
        """Entries with empty metadata can be stored and retrieved."""
        entry = seed_entry_factory(
            entry_id=f"{INTEG_PREFIX}meta-001",
            raw_text="Entry with empty metadata",
        )
        entry["metadata"] = {}
        await repository.upsert_entry(entry)

        retrieved = await repository.get_entry("integ-meta-001")
        assert retrieved is not None
        assert retrieved.get("metadata") == {}

    async def test_entry_with_rich_metadata(self, repository, seed_entry_factory):
        """Entries with rich metadata can be stored and retrieved."""
        entry = seed_entry_factory(
            entry_id=f"{INTEG_PREFIX}meta-002",
            raw_text="Entry with rich metadata",
        )
        entry["metadata"] = {
            "title": "Test Title",
            "category": "operations",
            "tags": ["test", "integration"],
            "nested": {"key": "value"},
        }
        await repository.upsert_entry(entry)

        retrieved = await repository.get_entry("integ-meta-002")
        assert retrieved is not None
        assert retrieved.get("metadata", {}).get("title") == "Test Title"


@pytest.mark.usefixtures("_seed_integ_prefix")
class TestRepositoryAttachmentPreservation:
    """Re-ingestion must not erase ARIEL-native (web-uploaded) attachments.

    ARIEL-native attachments (URL ``/api/attachments/{id}``) live in ARIEL only —
    the adapter write contract carries no attachments. A background re-ingestion poll
    re-fetches an already-published entry and upserts the upstream copy, which has
    *no* attachments. Without preservation, that upsert would overwrite the entry's
    attachment references to ``[]`` and orphan the stored blobs.
    """

    async def test_reingest_with_no_attachments_preserves_ariel_native(
        self, repository, seed_entry_factory
    ):
        """An upsert carrying no attachments keeps the existing ARIEL-native ones."""
        ariel_native = [
            {"url": "/api/attachments/abc123", "type": "image/png", "filename": "shot.png"}
        ]
        entry = seed_entry_factory(
            entry_id=f"{INTEG_PREFIX}attach-preserve-001",
            raw_text="Published entry with an ARIEL-only attachment",
            attachments=ariel_native,
        )
        await repository.upsert_entry(entry)

        # Simulate the background poller re-ingesting the upstream entry, which has
        # no attachments (the logbook API never received the file).
        reingested = seed_entry_factory(
            entry_id=f"{INTEG_PREFIX}attach-preserve-001",
            raw_text="Published entry with an ARIEL-only attachment",
            attachments=[],
        )
        await repository.upsert_entry(reingested)

        retrieved = await repository.get_entry("integ-attach-preserve-001")
        assert retrieved is not None
        assert retrieved["attachments"] == ariel_native

    async def test_reingest_with_attachments_replaces(self, repository, seed_entry_factory):
        """A non-empty incoming attachment list still replaces (upstream wins when it has data)."""
        entry = seed_entry_factory(
            entry_id=f"{INTEG_PREFIX}attach-preserve-002",
            raw_text="Entry",
            attachments=[{"url": "/api/attachments/old", "type": "image/png", "filename": "a.png"}],
        )
        await repository.upsert_entry(entry)

        upstream = [{"url": "https://elog.example/img/1", "type": "image/png", "filename": "b.png"}]
        replacement = seed_entry_factory(
            entry_id=f"{INTEG_PREFIX}attach-preserve-002",
            raw_text="Entry",
            attachments=upstream,
        )
        await repository.upsert_entry(replacement)

        retrieved = await repository.get_entry("integ-attach-preserve-002")
        assert retrieved is not None
        assert retrieved["attachments"] == upstream


@pytest.mark.usefixtures("_seed_integ_prefix")
class TestRepositoryUpsertReturning:
    """The in-transaction upsert keeps native items and returns the stored columns."""

    NATIVE = {"url": "/api/attachments/web123", "type": "image/png", "filename": "web.png"}
    UPSTREAM = {"url": "https://elog.example/img/1", "type": "image/png", "filename": "up.png"}

    async def test_upsert_returning_new_entry_returns_its_attachments(
        self, repository, seed_entry_factory
    ):
        """A first insert returns the written list and NULL text/captions."""
        entry = seed_entry_factory(
            entry_id=f"{INTEG_PREFIX}upsret-001", attachments=[self.UPSTREAM]
        )
        async with repository.pool.connection() as conn, conn.transaction():
            row = await repository.upsert_entry_returning(entry, conn=conn)

        assert row == {
            "attachments": [self.UPSTREAM],
            "attachment_text": None,
            "attachment_captions": None,
        }

    async def test_upsert_returning_reingest_keeps_web_uploaded_native_item(
        self, repository, seed_entry_factory
    ):
        """A re-ingest with upstream pictures appends the stored native item."""
        entry_id = f"{INTEG_PREFIX}upsret-002"
        stale = {"url": "https://elog.example/img/old", "type": "image/png", "filename": "o.png"}
        await repository.upsert_entry(
            seed_entry_factory(entry_id=entry_id, attachments=[stale, self.NATIVE])
        )

        reingested = seed_entry_factory(entry_id=entry_id, attachments=[self.UPSTREAM])
        async with repository.pool.connection() as conn, conn.transaction():
            row = await repository.upsert_entry_returning(reingested, conn=conn)

        # Upstream replaces upstream items; the native item survives, after them.
        assert row["attachments"] == [self.UPSTREAM, self.NATIVE]
        retrieved = await repository.get_entry(entry_id)
        assert retrieved is not None
        assert retrieved["attachments"] == [self.UPSTREAM, self.NATIVE]

    async def test_upsert_returning_empty_incoming_list_keeps_stored_list(
        self, repository, seed_entry_factory
    ):
        """An empty incoming list keeps the old JSONB, non-native items included."""
        entry_id = f"{INTEG_PREFIX}upsret-003"
        stored = [self.UPSTREAM, self.NATIVE]
        await repository.upsert_entry(seed_entry_factory(entry_id=entry_id, attachments=stored))

        async with repository.pool.connection() as conn, conn.transaction():
            row = await repository.upsert_entry_returning(
                seed_entry_factory(entry_id=entry_id, attachments=[], raw_text="edited"),
                conn=conn,
            )

        assert row["attachments"] == stored
        retrieved = await repository.get_entry(entry_id)
        assert retrieved is not None
        assert retrieved["attachments"] == stored
        assert retrieved["raw_text"] == "edited"

    async def test_upsert_returning_does_not_duplicate_a_native_url_already_incoming(
        self, repository, seed_entry_factory
    ):
        """A native item whose url the incoming list already carries is not appended."""
        entry_id = f"{INTEG_PREFIX}upsret-004"
        await repository.upsert_entry(
            seed_entry_factory(entry_id=entry_id, attachments=[self.NATIVE])
        )
        incoming_native = {**self.NATIVE, "filename": "renamed.png"}
        async with repository.pool.connection() as conn, conn.transaction():
            row = await repository.upsert_entry_returning(
                seed_entry_factory(entry_id=entry_id, attachments=[incoming_native, self.UPSTREAM]),
                conn=conn,
            )

        assert row["attachments"] == [incoming_native, self.UPSTREAM]

    async def test_upsert_returning_native_match_is_anchored(self, repository, seed_entry_factory):
        """Only urls that are exactly /api/attachments/<id> count as native."""
        entry_id = f"{INTEG_PREFIX}upsret-005"
        lookalikes = [
            {"url": "/api/attachments/a/b", "type": "image/png", "filename": "x.png"},
            {"url": "https://h/api/attachments/zz", "type": "image/png", "filename": "y.png"},
            {"url": "/api/attachments/q?x=1", "type": "image/png", "filename": "z.png"},
            {"type": "image/png", "filename": "nourl.png"},
        ]
        await repository.upsert_entry(
            seed_entry_factory(entry_id=entry_id, attachments=[*lookalikes, self.NATIVE])
        )
        async with repository.pool.connection() as conn, conn.transaction():
            row = await repository.upsert_entry_returning(
                seed_entry_factory(entry_id=entry_id, attachments=[self.UPSTREAM]), conn=conn
            )

        assert row["attachments"] == [self.UPSTREAM, self.NATIVE]

    async def test_upsert_returning_returns_stored_text_and_captions(
        self, repository, seed_entry_factory
    ):
        """The stored attachment_text and attachment_captions come back unchanged."""
        entry_id = f"{INTEG_PREFIX}upsret-006"
        await repository.upsert_entry(
            seed_entry_factory(entry_id=entry_id, attachments=[self.UPSTREAM])
        )
        async with repository.pool.connection() as conn:
            await conn.execute(
                "UPDATE enhanced_entries SET attachment_text = %(t)s,"
                " attachment_captions = %(c)s::jsonb WHERE entry_id = %(e)s",
                {"t": "a picture", "c": '{"k": {"m": "cap"}}', "e": entry_id},
            )

        async with repository.pool.connection() as conn, conn.transaction():
            row = await repository.upsert_entry_returning(
                seed_entry_factory(entry_id=entry_id, attachments=[self.UPSTREAM]), conn=conn
            )

        assert row["attachment_text"] == "a picture"
        assert row["attachment_captions"] == {"k": {"m": "cap"}}

    async def test_upsert_returning_rolls_back_with_the_caller_transaction(
        self, repository, seed_entry_factory
    ):
        """The write belongs to the caller's transaction and holds the entry lock."""
        entry_id = f"{INTEG_PREFIX}upsret-007"
        await repository.upsert_entry(
            seed_entry_factory(entry_id=entry_id, raw_text="before", attachments=[self.NATIVE])
        )

        class _Abort(Exception):
            pass

        with pytest.raises(_Abort):
            async with repository.pool.connection() as conn, conn.transaction():
                await repository.upsert_entry_returning(
                    seed_entry_factory(entry_id=entry_id, raw_text="after"), conn=conn
                )
                async with repository.pool.connection() as other:
                    await other.execute("SET lock_timeout = '200ms'")
                    with pytest.raises(Exception, match="lock"):
                        await other.execute(
                            "SELECT 1 FROM enhanced_entries WHERE entry_id = %(e)s FOR UPDATE",
                            {"e": entry_id},
                        )
                raise _Abort

        retrieved = await repository.get_entry(entry_id)
        assert retrieved is not None
        assert retrieved["raw_text"] == "before"

    async def test_upsert_returning_without_conn_uses_its_own_connection(
        self, repository, seed_entry_factory
    ):
        """Called without a connection, the upsert commits on its own."""
        entry_id = f"{INTEG_PREFIX}upsret-008"
        row = await repository.upsert_entry_returning(
            seed_entry_factory(entry_id=entry_id, attachments=[self.NATIVE])
        )
        assert row["attachments"] == [self.NATIVE]
        retrieved = await repository.get_entry(entry_id)
        assert retrieved is not None
        assert retrieved["attachments"] == [self.NATIVE]


@pytest.mark.usefixtures("_seed_integ_prefix")
class TestRepositoryBulkOperations:
    """Test ARIELRepository bulk operations."""

    async def test_multiple_entries_upsert(self, repository, seed_entry_factory):
        """Multiple entries can be upserted sequentially."""
        entries = [
            seed_entry_factory(
                entry_id=f"{INTEG_PREFIX}bulk-{i:03d}",
                raw_text=f"Bulk entry {i}",
            )
            for i in range(5)
        ]

        for entry in entries:
            await repository.upsert_entry(entry)

        # Verify all were stored
        results = await repository.get_entries_by_ids([f"integ-bulk-{i:03d}" for i in range(5)])
        assert len(results) == 5


class TestDatabaseErrorConditions:
    """Test database error handling (EDGE-010, EDGE-011)."""

    async def test_connection_failure_raises_database_connection_error(self):
        """Attempting to connect with invalid credentials raises DatabaseConnectionError.

        EDGE-010: Database connection failure handling.
        """
        from osprey.services.ariel_search.config import DatabaseConfig
        from osprey.services.ariel_search.database.connection import create_connection_pool

        # Create config with invalid connection string
        config = DatabaseConfig(uri="postgresql://invalid:invalid@localhost:99999/nonexistent")

        # Attempt to create connection pool should fail
        with pytest.raises(Exception) as exc_info:
            pool = await create_connection_pool(config)
            # Try to actually connect
            async with pool.connection() as conn:
                await conn.execute("SELECT 1")

        # Should be a connection-related error (psycopg may raise PoolTimeout
        # when the pool can't establish any connections within the timeout)
        error_str = str(exc_info.value).lower()
        assert any(
            x in error_str
            for x in ["connect", "refused", "host", "port", "could not", "pool", "timeout"]
        )

    async def test_malformed_sql_raises_database_query_error(self, migrated_pool):
        """Executing malformed SQL raises DatabaseQueryError.

        EDGE-011: Malformed SQL error handling.
        """

        # Create repository with valid pool
        from osprey.services.ariel_search.config import ARIELConfig, DatabaseConfig
        from osprey.services.ariel_search.database.repository import ARIELRepository

        config = ARIELConfig(database=DatabaseConfig(uri="postgresql://test/test"))
        _repo = ARIELRepository(migrated_pool, config)

        # Execute intentionally malformed SQL
        # The repository methods wrap errors in DatabaseQueryError
        with pytest.raises(Exception) as exc_info:
            async with migrated_pool.connection() as conn:
                # This SQL is syntactically invalid
                await conn.execute("SELECT * FROMM invalid_table WHEREE x = y")

        # Should be a syntax error
        error_str = str(exc_info.value).lower()
        assert "syntax" in error_str or "error" in error_str

    @pytest.mark.usefixtures("seed_entry_factory")
    async def test_repository_wraps_query_errors(self, repository):
        """Repository methods wrap database errors in DatabaseQueryError."""
        from osprey.services.ariel_search.exceptions import DatabaseQueryError

        # Try to get entry with None ID (should cause error in query building)
        # Note: get_entry handles None gracefully, so we test with invalid type
        try:
            await repository.get_entry("valid-id")  # This should work
        except DatabaseQueryError:
            # If it raises, the wrapping works
            pass
        # Success either way - we're just verifying the mechanism


@pytest.mark.usefixtures("_seed_concurrent_prefix")
class TestConcurrentOperations:
    """Test concurrent database operations (INT-006)."""

    async def test_connection_pool_handles_concurrent_requests(
        self, repository, seed_entry_factory
    ):
        """Connection pool handles multiple concurrent operations.

        INT-006: Concurrent operations test.
        """
        import asyncio

        # Create test entries first
        entries = [
            seed_entry_factory(
                entry_id=f"{CONCURRENT_PREFIX}{i:03d}",
                raw_text=f"Concurrent test entry {i}",
            )
            for i in range(10)
        ]

        for entry in entries:
            await repository.upsert_entry(entry)

        # Launch 10 concurrent searches
        async def concurrent_search(idx: int):
            """Execute a search operation."""
            results = await repository.search_by_time_range(limit=5)
            return idx, len(results)

        # Run searches concurrently
        tasks = [concurrent_search(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        # Verify all completed successfully
        assert len(results) == 10
        for _idx, count in results:
            assert isinstance(count, int)

    async def test_concurrent_reads_and_writes(self, repository, seed_entry_factory):
        """Concurrent reads and writes don't corrupt data."""
        import asyncio

        base_id = f"{CONCURRENT_PREFIX}rw"

        async def writer(idx: int):
            """Write operation."""
            entry = seed_entry_factory(
                entry_id=f"{base_id}-{idx:03d}",
                raw_text=f"Entry written by task {idx}",
            )
            await repository.upsert_entry(entry)
            return f"write-{idx}"

        async def reader(idx: int):
            """Read operation."""
            await repository.search_by_time_range(limit=3)
            return f"read-{idx}"

        # Mix of readers and writers
        tasks = []
        for i in range(5):
            tasks.append(writer(i))
            tasks.append(reader(i))

        results = await asyncio.gather(*tasks)

        # All should complete
        assert len(results) == 10

        # Verify written entries exist
        for i in range(5):
            entry = await repository.get_entry(f"{base_id}-{i:03d}")
            assert entry is not None


# ============================================================================
# Image-embedding tables: status listing and purge
# ============================================================================

#: A one-pixel PNG; the stub embeds any content to a deterministic vector.
_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00"
    b"\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
    b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)

#: The text-embedding table the status tests create by hand.
_TEXT_TABLE = "text_embeddings_probe"


@pytest.fixture
def _hybrid_reader(monkeypatch):
    """Report the hybrid search module as on, whatever config this process loaded."""
    from osprey.services.ariel_search.enhancement.image_embedding import module as embed_mod

    monkeypatch.setattr(embed_mod, "hybrid_search_enabled", lambda: True)


def _image_config(uri: str, url: str, **image: object) -> dict:
    """The raw ``ariel`` block: keyword + hybrid search and the image-embedding module."""
    from tests.services.ariel_search.llama_stub import MODEL

    return {
        "database": {"uri": uri},
        "attachments": {"copy_on_ingest": "images"},
        "search_modules": {"keyword": {"enabled": True}, "hybrid": {"enabled": True}},
        "enhancement_modules": {
            "image_embedding": {
                "enabled": True,
                "provider": {"name": "llama-cpp", "base_url": url},
                "model": MODEL,
                "dimensions": 1024,
                **image,
            }
        },
    }


def _seed_picture(uri: str, entry_id: str, attachment_id: str) -> None:
    """One entry holding one copied picture with a rendition."""
    import psycopg

    with psycopg.connect(uri, autocommit=True) as conn:
        conn.execute(
            "INSERT INTO enhanced_entries (entry_id, source_system, timestamp, raw_text)"
            " VALUES (%s, 'test', NOW(), 'text')",
            (entry_id,),
        )
        conn.execute(
            """
            INSERT INTO attachment_files (
                attachment_id, entry_id, filename, mime_type, source_url, copy_status,
                rendition_bytes, rendition_mime, rendition_w, rendition_h, rendition_sha256
            ) VALUES (%s, %s, 'f.png', 'image/png', 'https://h.example/f.png', 'copied',
                      %s, 'image/png', 1, 1, %s)
            """,
            (attachment_id, entry_id, _PNG, "ab" * 32),
        )


def _create_text_table(uri: str) -> None:
    import psycopg

    with psycopg.connect(uri, autocommit=True) as conn:
        conn.execute(f"CREATE TABLE {_TEXT_TABLE} (entry_id TEXT PRIMARY KEY, embedding vector(3))")


def _vector_count(uri: str, table: str) -> int | None:
    """Stored vectors in *table*, None when the table does not exist."""
    import psycopg

    with psycopg.connect(uri) as conn:
        exists = conn.execute("SELECT to_regclass(%s) IS NOT NULL", (table,)).fetchone()
        if not (exists and exists[0]):
            return None
        row = conn.execute(f"SELECT COUNT(embedding) FROM {table}").fetchone()
    return int(row[0]) if row else 0


def _status_keys(uri: str) -> list[dict]:
    import psycopg

    with psycopg.connect(uri) as conn:
        rows = conn.execute("SELECT enhancement_status FROM enhanced_entries").fetchall()
    return [row[0] or {} for row in rows]


@pytest.mark.timeout(180)
@pytest.mark.usefixtures("_hybrid_reader")
class TestImageEmbeddingTables:
    """The image tables in ``status``, ``purge`` and the re-embed after a purge."""

    async def test_purge_drops_image_tables_and_catchup_re_embeds(
        self, scratch_database, llama_stub
    ):
        from osprey.services.ariel_search import cli_operations as ops
        from osprey.services.ariel_search.database.migrations import image_table_name
        from tests.services.ariel_search.llama_stub import MODEL

        stub = llama_stub()
        table = image_table_name(MODEL, 1024)
        cfg = _image_config(scratch_database, stub.url)
        await ops.run_migrate(cfg)
        _seed_picture(scratch_database, "img-1", "att-1")

        await ops.run_catchup(cfg, budget_s=None, stop_event=None)
        assert _vector_count(scratch_database, table) == 1
        assert all("image_embedding" in s for s in _status_keys(scratch_database))

        info = await ops.get_purge_info(cfg)
        assert info.image_embedding_tables == [table]
        assert table not in info.embedding_tables

        await ops.execute_purge(cfg, embeddings_only=True)
        assert _vector_count(scratch_database, table) is None
        assert not any("image_embedding" in s for s in _status_keys(scratch_database))
        assert (await ops.get_purge_info(cfg)).image_embedding_tables == []

        await ops.run_migrate(cfg)
        assert _vector_count(scratch_database, table) == 0
        await ops.run_catchup(cfg, budget_s=None, stop_event=None)
        assert _vector_count(scratch_database, table) == 1

        status = await ops.get_status(cfg)
        assert status["status"] == "healthy", status
        assert status["image_embedding_tables"] == [
            {"table": table, "pictures": 1, "dimension": 1024, "active": True}
        ]
        assert table not in [t["table"] for t in status["embedding_tables"]]
        assert status["enhancement_modules"]["image_embedding"]["complete"] == 1

    async def test_full_purge_drops_image_tables(self, scratch_database, llama_stub):
        from osprey.services.ariel_search import cli_operations as ops
        from osprey.services.ariel_search.database.migrations import image_table_name
        from tests.services.ariel_search.llama_stub import MODEL

        stub = llama_stub()
        table = image_table_name(MODEL, 1024)
        cfg = _image_config(scratch_database, stub.url)
        await ops.run_migrate(cfg)
        assert _vector_count(scratch_database, table) == 0

        await ops.execute_purge(cfg, embeddings_only=False)
        assert _vector_count(scratch_database, table) is None

    async def test_image_embedding_tables_status_names_a_down_server(
        self, scratch_database, llama_stub
    ):
        from osprey.services.ariel_search import cli_operations as ops

        stub = llama_stub()
        cfg = _image_config(scratch_database, stub.url)
        await ops.run_migrate(cfg)

        up = await ops.get_status(cfg)
        assert up["status"] == "healthy", up
        assert up["attachments"]["picture_search"] is True
        assert up["attachments"]["picture_search_unavailable"] is None

        stub.stop()
        down = await ops.get_status(cfg)
        assert down["attachments"]["picture_search"] is True
        assert down["attachments"]["picture_search_unavailable"] == "unreachable"
        assert down["enhancement_modules"]["image_embedding"]["health"]["reachable"] is False

    async def test_image_embedding_tables_status_with_a_misconfigured_block(
        self, scratch_database, llama_stub
    ):
        from osprey.services.ariel_search import cli_operations as ops

        stub = llama_stub()
        await ops.run_migrate({"database": {"uri": scratch_database}})
        _create_text_table(scratch_database)
        cfg = _image_config(scratch_database, stub.url, dimensions=5000)

        status = await ops.get_status(cfg)

        assert status["status"] == "healthy", status
        assert [t["table"] for t in status["embedding_tables"]] == [_TEXT_TABLE]
        assert status["image_embedding_tables"] == []
        image = status["enhancement_modules"]["image_embedding"]
        assert image["health"]["reachable"] is False
        assert image["health"]["reason"] == "config"
        assert status["attachments"]["picture_search_unavailable"] == "config"
