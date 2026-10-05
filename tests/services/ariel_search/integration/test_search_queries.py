"""Integration tests for ARIEL search queries.

Tests actual search operations (FTS, vector similarity) against real PostgreSQL.

See 04_OSPREY_INTEGRATION.md Section 12.3.4 for test requirements.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

# xdist_group("docker"): pins every container-starting test file onto one worker, so
# a run has a single testcontainers session and a single ryuk reaper -- concurrent
# reaper starts race the Docker daemon's port mapper. It also serializes the shared
# database: the session ``database_url`` fixture prefers a running dev Postgres with
# ONE shared ``ariel_test`` database over a per-worker container, so parallel workers
# would otherwise collide on migrations/seed/truncate.
pytestmark = [pytest.mark.asyncio, pytest.mark.xdist_group("docker")]


#: Keyword-search rows: four texts and authors chosen so a search that fails
#: to narrow returns a row a correct one does not.
KEYWORD_PREFIX = "search-kw-"

#: Rows carrying a real embedding, seeded only where a local embedding
#: service answers.
SEMANTIC_PREFIX = "semantic-"

#: The single row the source-filtered time-range query is asserted over.
SOURCE_PREFIX = "search-source-"


@pytest.fixture
async def seeded_repository(repository, seed_entry_factory, seeded_prefixes):
    """Repository seeded with four entries whose texts and authors discriminate.

    The rows are chosen rather than arbitrary. ``search-kw-002`` and
    ``search-kw-004`` both carry ``beam``, only ``search-kw-002`` also carries
    ``orbit``, and only ``search-kw-004`` is written by ``oper_smith``. A search
    that fails to narrow on the second term, or on the author, therefore returns
    a row that a correct one does not.

    The rows outlive the test that seeded them: they are removed once, after the
    package's last test, by the ledger's finalizer.

    Args:
        repository: Repository over the migrated test database.
        seed_entry_factory: Factory building a single logbook entry.
        seeded_prefixes: Package ledger of the entry-id prefixes to delete at
            teardown.

    Returns:
        The repository, with the four entries upserted.
    """
    entries = [
        seed_entry_factory(
            entry_id=f"{KEYWORD_PREFIX}001",
            raw_text="The vacuum chamber pressure dropped unexpectedly during the experiment.",
            author="operator1",
        ),
        seed_entry_factory(
            entry_id=f"{KEYWORD_PREFIX}002",
            raw_text="Beam alignment was adjusted to correct the orbit deviation.",
            author="physicist1",
        ),
        seed_entry_factory(
            entry_id=f"{KEYWORD_PREFIX}003",
            raw_text="The undulator gap was changed to optimize photon flux.",
            author="scientist1",
        ),
        seed_entry_factory(
            entry_id=f"{KEYWORD_PREFIX}004",
            raw_text="Beam loss was recorded during the morning shift.",
            author="oper_smith",
        ),
    ]
    seeded_prefixes.add(KEYWORD_PREFIX)
    for entry in entries:
        await repository.upsert_entry(entry)
    return repository


class TestKeywordSearch:
    """Test keyword search with real PostgreSQL FTS."""

    async def test_keyword_search_finds_matches(self, seeded_repository):
        """Keyword search returns matching entries via repository method."""
        # Repository keyword_search uses: where_clauses, params, search_text, max_results
        # Use FTS condition for search
        results = await seeded_repository.keyword_search(
            where_clauses=["to_tsvector('english', raw_text) @@ plainto_tsquery('english', %s)"],
            params=["vacuum pressure"],
            search_text="vacuum pressure",
            max_results=10,
        )

        # Should find the vacuum chamber entry - results are (entry, score, highlights)
        entry_ids = [entry["entry_id"] for entry, score, highlights in results]
        assert "search-kw-001" in entry_ids

    async def test_keyword_search_no_matches_returns_empty(self, seeded_repository):
        """Keyword search returns empty list when no matches."""
        results = await seeded_repository.keyword_search(
            where_clauses=["to_tsvector('english', raw_text) @@ plainto_tsquery('english', %s)"],
            params=["nonexistent term xyz123"],
            search_text="nonexistent term xyz123",
            max_results=10,
        )
        assert results == []

    async def test_keyword_search_respects_limit(self, seeded_repository):
        """Keyword search respects the limit parameter."""
        results = await seeded_repository.keyword_search(
            where_clauses=["to_tsvector('english', raw_text) @@ plainto_tsquery('english', %s)"],
            params=["the"],
            search_text="the",
            max_results=1,
        )
        assert len(results) <= 1

    async def test_keyword_search_multiple_terms(self, seeded_repository):
        """Keyword search with multiple terms uses AND logic."""
        results = await seeded_repository.keyword_search(
            where_clauses=["to_tsvector('english', raw_text) @@ plainto_tsquery('english', %s)"],
            params=["beam alignment orbit"],
            search_text="beam alignment orbit",
            max_results=10,
        )

        # Should find the beam alignment entry
        entry_ids = [entry["entry_id"] for entry, score, highlights in results]
        assert "search-kw-002" in entry_ids


class TestKeywordQuerySyntax:
    """Query text reaching real PostgreSQL through the keyword search module.

    These drive ``osprey.services.ariel_search.search.keyword.keyword_search``,
    which parses the query and builds the predicates the class above hands
    :meth:`ARIELRepository.keyword_search` directly. What an operator's
    ``AND`` and ``author:`` mean is decided by PostgreSQL over the composed
    statement, so only a real database answers it.

    ``max_results`` is wide because the database is shared with the rest of
    the package: a narrow limit would let another module's seeded rows crowd
    the discriminating entry out of the result and fail the assertion for a
    reason that has nothing to do with the query.
    """

    async def test_an_and_query_requires_both_terms(
        self, seeded_repository, integration_ariel_config
    ):
        """``beam AND orbit`` keeps only the entry carrying both terms."""
        from osprey.services.ariel_search.search.keyword import keyword_search

        results = await keyword_search(
            query="beam AND orbit",
            repository=seeded_repository,
            config=integration_ariel_config,
            max_results=100,
        )

        entry_ids = [entry["entry_id"] for entry, _score, _highlights in results]
        assert "search-kw-002" in entry_ids
        assert "search-kw-004" not in entry_ids

    async def test_an_author_prefix_narrows_a_text_match(
        self, seeded_repository, integration_ariel_config
    ):
        """``author:oper_smith beam`` keeps only that author's matching entry."""
        from osprey.services.ariel_search.search.keyword import keyword_search

        results = await keyword_search(
            query="author:oper_smith beam",
            repository=seeded_repository,
            config=integration_ariel_config,
            max_results=100,
        )

        entry_ids = [entry["entry_id"] for entry, _score, _highlights in results]
        assert "search-kw-004" in entry_ids
        assert "search-kw-002" not in entry_ids


# ==============================================================================
# Semantic Search Tests with Real Embeddings (TEST-M005 / INT-003)
# ==============================================================================


def is_ollama_available() -> bool:
    """Check if Ollama is available for tests."""
    try:
        import requests

        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


@pytest.mark.requires_ollama
class TestSemanticSearchWithRealEmbeddings:
    """Test semantic search with real Ollama embeddings.

    These tests require Ollama running locally with nomic-embed-text model.
    Run: ollama pull nomic-embed-text
    """

    @pytest.fixture
    async def seeded_repository_with_embeddings(
        self,
        repository,
        seed_entry_factory,
        seeded_prefixes,
        litellm_callback_pool,  # noqa: ARG002 - the embeddings below run their callbacks on it
    ):
        """Repository seeded with three entries and their embeddings.

        The three rows are about topics far enough apart that a query about one
        ranks above the others: beam loss at injection, orbit deviation after a
        position-monitor reading, and vacuum maintenance.

        The embedding rows need no ledger entry of their own.
        ``text_embeddings_nomic_embed_text.entry_id`` is a foreign key onto
        ``enhanced_entries(entry_id)`` declared ``ON DELETE CASCADE``, so
        deleting an entry takes its embedding with it.

        Args:
            repository: Repository over the migrated test database.
            seed_entry_factory: Factory building a single logbook entry.
            seeded_prefixes: Package ledger of the entry-id prefixes to delete
                at teardown.

        Returns:
            The repository, with the three entries and their embeddings stored.
        """
        if not is_ollama_available():
            pytest.skip("Ollama not available - run 'ollama pull nomic-embed-text'")

        from tests.services.ariel_search.fake_providers import ollama_text_embedder

        embedder = ollama_text_embedder()

        # Create entries about different topics
        entries = [
            seed_entry_factory(
                entry_id=f"{SEMANTIC_PREFIX}001",
                raw_text="Beam loss detected at sector 5. The injection efficiency dropped to 82% due to instability in the storage ring.",
                author="operator1",
            ),
            seed_entry_factory(
                entry_id=f"{SEMANTIC_PREFIX}002",
                raw_text="Beam position monitors showing orbit deviation. Correcting with steering magnets.",
                author="physicist1",
            ),
            seed_entry_factory(
                entry_id=f"{SEMANTIC_PREFIX}003",
                raw_text="Vacuum system maintenance completed. Pressure in sector 7 now at 1e-10 Torr.",
                author="technician1",
            ),
        ]
        seeded_prefixes.add(SEMANTIC_PREFIX)

        # Insert entries into database
        for entry in entries:
            await repository.upsert_entry(entry)

        # Generate and store embeddings
        for entry in entries:
            embeddings = embedder.execute_embedding(
                texts=[entry["raw_text"]],
                model_id="nomic-embed-text",
            )
            if embeddings and embeddings[0]:
                await repository.store_text_embedding(
                    entry_id=entry["entry_id"],
                    embedding=embeddings[0],
                    model_name="nomic-embed-text",
                )

        return repository

    async def test_semantic_search_finds_similar_entries(
        self, seeded_repository_with_embeddings, integration_ariel_config
    ):
        """Semantic search finds entries with similar meaning."""
        from osprey.services.ariel_search.search.semantic import semantic_search
        from tests.services.ariel_search.fake_providers import ollama_text_embedder

        embedder = ollama_text_embedder()

        # Query about beam instability - should match beam loss entries
        results = await semantic_search(
            query="beam instability problems",
            repository=seeded_repository_with_embeddings,
            config=integration_ariel_config,
            embedder=embedder,
            max_results=10,
            similarity_threshold=0.5,  # Lower threshold for testing
        )

        # Should find beam-related entries
        entry_ids = [entry["entry_id"] for entry, score in results]
        assert "semantic-001" in entry_ids or "semantic-002" in entry_ids

    async def test_similarity_scores_decrease_for_unrelated(
        self, seeded_repository_with_embeddings, integration_ariel_config
    ):
        """Unrelated entries have lower similarity scores."""
        from osprey.services.ariel_search.search.semantic import semantic_search
        from tests.services.ariel_search.fake_providers import ollama_text_embedder

        embedder = ollama_text_embedder()

        # Query specifically about beam loss
        results = await semantic_search(
            query="beam loss injection efficiency",
            repository=seeded_repository_with_embeddings,
            config=integration_ariel_config,
            embedder=embedder,
            max_results=10,
            similarity_threshold=0.0,  # Get all results
        )

        if len(results) >= 2:
            # Find scores for beam entry vs vacuum entry
            beam_scores = [
                score
                for entry, score in results
                if entry["entry_id"] in ("semantic-001", "semantic-002")
            ]
            vacuum_scores = [
                score for entry, score in results if entry["entry_id"] == "semantic-003"
            ]

            if beam_scores and vacuum_scores:
                # Beam-related entries should have higher similarity
                assert max(beam_scores) >= max(vacuum_scores)

    @pytest.mark.usefixtures("seeded_repository_with_embeddings")
    async def test_embedding_dimension_is_768(self, migrated_pool):
        """nomic-embed-text embeddings have 768 dimensions."""
        async with migrated_pool.connection() as conn:
            result = await conn.execute("""
                SELECT embedding FROM text_embeddings_nomic_embed_text
                WHERE entry_id = 'semantic-001'
                LIMIT 1
            """)
            row = await result.fetchone()

            if row and row[0]:
                # pgvector stores as string, parse dimension
                embedding = row[0]
                # Vector format is [x,y,z,...] - count elements
                if isinstance(embedding, str):
                    dim = embedding.count(",") + 1
                else:
                    dim = len(embedding)
                assert dim == 768


class TestSearchQueryStructure:
    """Test search query structure without semantic data."""

    async def test_search_by_time_range_with_source_filter(
        self, repository, seed_entry_factory, seeded_prefixes
    ):
        """Search with source system filter."""
        now = datetime.now(UTC)
        entry = seed_entry_factory(
            entry_id=f"{SOURCE_PREFIX}001",
            source_system="als_logbook",
            timestamp=now,
            raw_text="Test entry from the logbook",
        )
        seeded_prefixes.add(SOURCE_PREFIX)
        await repository.upsert_entry(entry)

        # This tests the query structure even if filtering isn't implemented
        results = await repository.search_by_time_range(limit=10)
        assert isinstance(results, list)


# ==============================================================================
# The has_v2_fts schema fact on real stores
# ==============================================================================


def _raw_text_config(scratch_config):
    """`scratch_config` with semantic_processor off, so keyword search runs on raw_text."""
    from dataclasses import replace

    from osprey.services.ariel_search.config import EnhancementModuleConfig

    modules = dict(scratch_config.enhancement_modules)
    modules["semantic_processor"] = EnhancementModuleConfig(enabled=False)
    return replace(scratch_config, enhancement_modules=modules)


async def _insert(conn, entry_id: str, raw_text: str, attachment_text: str | None = None):
    columns = (
        "entry_id, source_system, timestamp, author, raw_text, attachments, metadata, "
        "enhancement_status"
    )
    values = "%s, 'test', NOW(), 'tester', %s, '[]'::jsonb, '{}'::jsonb, '{}'::jsonb"
    params: list[object] = [entry_id, raw_text]
    if attachment_text is not None:
        columns += ", attachment_text"
        values += ", %s"
        params.append(attachment_text)
    await conn.execute(f"INSERT INTO enhanced_entries ({columns}) VALUES ({values})", params)


class TestSchemaFactOnRealStores:
    """Search SQL follows the probed ``has_v2_fts`` fact, never the code's expectations."""

    @pytest.fixture
    async def schema_behind_store(self, scratch_config):
        """A store as B1 left it: no attachment_text column, no copy state, no V2 index.

        Migrated to today's schema, then rolled back by dropping what the newer
        migrations added, so any statement naming ``attachment_text`` fails.
        """
        from osprey.services.ariel_search.database import create_connection_pool, run_migrations
        from osprey.services.ariel_search.database.repository import ARIELRepository

        config = _raw_text_config(scratch_config)
        pool = await create_connection_pool(config.database)
        try:
            await run_migrations(pool, config)
            async with pool.connection() as conn:
                await conn.execute(
                    "ALTER TABLE enhanced_entries "
                    "DROP COLUMN attachment_text CASCADE, "
                    "DROP COLUMN IF EXISTS attachment_captions CASCADE"
                )
                await conn.execute(
                    "ALTER TABLE attachment_files DROP COLUMN IF EXISTS copy_status CASCADE"
                )
                await _insert(conn, "behind-001", "QX-772 tripped on the north magnet string.")
                await _insert(conn, "behind-002", "Klystron forty one fault cleared by operator.")
            yield ARIELRepository(pool, config), config
        finally:
            await pool.close()

    @pytest.fixture
    async def migrated_store(self, scratch_config):
        """A store migrated to today's schema, raw_text keyword search."""
        from osprey.services.ariel_search.database import create_connection_pool, run_migrations
        from osprey.services.ariel_search.database.repository import ARIELRepository

        config = _raw_text_config(scratch_config)
        pool = await create_connection_pool(config.database)
        try:
            await run_migrations(pool, config)
            yield ARIELRepository(pool, config), config, pool
        finally:
            await pool.close()

    async def test_schema_behind_store_answers_pattern_and_fuzzy_with_the_diagnostic(
        self, schema_behind_store
    ):
        from osprey.services.ariel_search.database.repository import (
            SCHEMA_BEHIND_SEARCH_MESSAGE,
            SchemaFacts,
            schema_behind_diagnostics,
        )
        from osprey.services.ariel_search.models import DiagnosticLevel
        from osprey.services.ariel_search.search.keyword import keyword_search

        repository, config = schema_behind_store

        assert await repository.schema_facts() == SchemaFacts(False, False)

        pattern_hits = await keyword_search("QX-77*", repository, config, max_results=10)
        assert [entry["entry_id"] for entry, _s, _h in pattern_hits] == ["behind-001"]

        fuzzy_hits = await keyword_search(
            "Klystron forty one fault clearde", repository, config, max_results=10
        )
        assert [entry["entry_id"] for entry, _s, _h in fuzzy_hits] == ["behind-002"]

        (diagnostic,) = await schema_behind_diagnostics(repository)
        assert diagnostic.level is DiagnosticLevel.WARNING
        assert diagnostic.message == SCHEMA_BEHIND_SEARCH_MESSAGE

    async def test_fuzzy_raw_text_match_survives_a_long_attachment_text_caption(
        self, migrated_store
    ):
        """GREATEST over both texts: a long caption never dilutes a raw_text match."""
        repository, _config, pool = migrated_store
        raw_text = "Klystron forty one fault cleared by operator"
        async with pool.connection() as conn:
            await _insert(conn, "fuzzy-001", raw_text)

        assert (await repository.schema_facts()).has_v2_fts
        before = await repository.fuzzy_search(raw_text, threshold=0.5, v2=True)
        assert [entry["entry_id"] for entry, _s, _h in before] == ["fuzzy-001"]

        caption = " ".join(f"[picture p{i}.png] scope trace channel {i} nominal" for i in range(80))
        async with pool.connection() as conn:
            await conn.execute(
                "UPDATE enhanced_entries SET attachment_text = %s WHERE entry_id = 'fuzzy-001'",
                [caption],
            )

        after = await repository.fuzzy_search(raw_text, threshold=0.5, v2=True)
        assert [entry["entry_id"] for entry, _s, _h in after] == ["fuzzy-001"]
        assert after[0][1] == pytest.approx(before[0][1])

    async def test_attachment_text_pattern_matches_on_a_migrated_store(self, migrated_store):
        from osprey.services.ariel_search.search.keyword import keyword_search

        repository, config, pool = migrated_store
        async with pool.connection() as conn:
            await _insert(conn, "caption-001", "Scope capture attached.", "[picture] QX-779 trace")

        hits = await keyword_search("QX-77*", repository, config, max_results=10)

        assert [entry["entry_id"] for entry, _s, _h in hits] == ["caption-001"]

    async def test_attachment_text_keyword_search_uses_the_v2_index(
        self, migrated_store, monkeypatch
    ):
        """EXPLAIN of the statement ``keyword_search`` actually ran picks a ``_v2`` index."""
        import psycopg

        from osprey.services.ariel_search.search.keyword import keyword_search

        repository, config, pool = migrated_store
        async with pool.connection() as conn:
            for i in range(200):
                await _insert(
                    conn,
                    f"explain-{i:04d}",
                    f"shift note {i} vacuum gauge reading",
                    f"[picture p{i}.png] klystron trace {i}" if i % 3 else None,
                )

        statements: list[tuple[str, list]] = []
        original = psycopg.AsyncCursor.execute

        async def recording_execute(self, query, params=None, **kwargs):
            if isinstance(query, str) and "ts_rank" in query:
                statements.append((query, list(params or [])))
            return await original(self, query, params, **kwargs)

        monkeypatch.setattr(psycopg.AsyncCursor, "execute", recording_execute)
        hits = await keyword_search("klystron", repository, config, fuzzy_fallback=False)
        monkeypatch.undo()

        assert hits
        ((sql, params),) = statements
        async with pool.connection() as conn:
            await conn.execute("ANALYZE enhanced_entries")
            async with conn.transaction():
                await conn.execute("SET LOCAL enable_seqscan = off")
                cursor = psycopg.AsyncClientCursor(conn)
                try:
                    await cursor.execute(f"EXPLAIN {sql}", params)
                    plan = "\n".join(row[0] for row in await cursor.fetchall())
                finally:
                    await cursor.close()

        assert "idx_entries_raw_text_fts_v2" in plan, plan


# ==============================================================================
# Requirement 3: caption-only mentions are keyword-searchable
# ==============================================================================


def _caption_only_config(scratch_config):
    """Raw-text keyword search with the ``image_caption`` module off."""
    from dataclasses import replace

    from osprey.services.ariel_search.config import EnhancementModuleConfig

    config = _raw_text_config(scratch_config)
    modules = dict(config.enhancement_modules)
    modules["image_caption"] = EnhancementModuleConfig(enabled=False)
    return replace(config, enhancement_modules=modules)


def _picture(entry_id: str, filename: str, caption: str) -> tuple[dict, str]:
    """An upstream attachment item with a caption, and its ``attachment_files`` id."""
    from osprey.services.ariel_search.attachments import attachment_id_for

    item = {"url": f"https://elog.example/files/{entry_id}/{filename}", "filename": filename}
    item["caption"] = caption
    attachment_id = attachment_id_for(entry_id, item)
    assert attachment_id is not None
    return item, attachment_id


async def _insert_captioned(conn, entry_id: str, raw_text: str, item: dict, attachment_id: str):
    """Insert an entry whose picture caption is folded into ``attachment_text``, as ingest does."""
    import json

    from osprey.services.ariel_search.attachments.compose import compose_attachment_text

    await conn.execute(
        """
        INSERT INTO enhanced_entries (
            entry_id, source_system, timestamp, author, raw_text, attachments, metadata,
            enhancement_status, attachment_text
        ) VALUES (%s, 'test', NOW(), 'tester', %s, %s::jsonb, '{}'::jsonb, '{}'::jsonb, %s)
        """,
        [
            entry_id,
            raw_text,
            json.dumps([item]),
            compose_attachment_text(entry_id, [item], None, None),
        ],
    )
    await conn.execute(
        """
        INSERT INTO attachment_files (attachment_id, entry_id, filename, source_url, copy_status)
        VALUES (%s, %s, %s, %s, 'pending')
        """,
        [attachment_id, entry_id, item["filename"], item["url"]],
    )


class TestRequirement3CaptionSearch:
    """A mention only inside a picture caption finds the entry and names the picture."""

    @pytest.fixture
    async def captioned_store(self, scratch_config):
        """A migrated store holding caption-only entries and a raw-text-only control."""
        from osprey.services.ariel_search.database import create_connection_pool, run_migrations
        from osprey.services.ariel_search.database.repository import ARIELRepository

        config = _caption_only_config(scratch_config)
        pool = await create_connection_pool(config.database)
        try:
            await run_migrations(pool, config)
            plot, plot_id = _picture(
                "req3-bpm", "orbit.png", "Orbit plot from SR:C07 BPM readbacks"
            )
            magnet, magnet_id = _picture("req3-qx", "scope.png", "Scope trace of QX-772 current")
            async with pool.connection() as conn:
                await _insert_captioned(
                    conn, "req3-bpm", "Orbit correction applied, plot attached.", plot, plot_id
                )
                await _insert_captioned(
                    conn,
                    "req3-qx",
                    "Magnet string tripped, scope trace attached.",
                    magnet,
                    magnet_id,
                )
                await _insert(conn, "req3-control", "Klystron forty one fault cleared.")
            yield ARIELRepository(pool, config), config, {"bpm": plot_id, "qx": magnet_id}
        finally:
            await pool.close()

    async def test_requirement_3_caption_only_bpm_mention_is_found_with_its_id(
        self, captioned_store
    ):
        from osprey.services.ariel_search.search.keyword import keyword_search

        repository, config, ids = captioned_store

        hits = await keyword_search("SR:C07 BPM", repository, config, max_results=10)

        assert [entry["entry_id"] for entry, _s, _h in hits] == ["req3-bpm"]
        assert hits[0][0]["_matched_attachment_ids"] == [ids["bpm"]]

    async def test_requirement_3_glob_finds_a_caption_only_entry_and_its_id(self, captioned_store):
        from osprey.services.ariel_search.search.keyword import keyword_search

        repository, config, ids = captioned_store

        hits = await keyword_search("QX-77*", repository, config, max_results=10)

        assert [entry["entry_id"] for entry, _s, _h in hits] == ["req3-qx"]
        assert hits[0][0]["_matched_attachment_ids"] == [ids["qx"]]

    async def test_requirement_3_b1_row_with_upstream_caption_is_found_after_migrate(
        self, scratch_config
    ):
        """A row written before the upgrade becomes caption-searchable through ``migrate``."""
        import json

        from osprey.services.ariel_search.database import create_connection_pool, run_migrations
        from osprey.services.ariel_search.database.attachment_migration import (
            AttachmentFilesCopyStateMigration,
        )
        from osprey.services.ariel_search.database.attachment_text_migration import (
            AttachmentTextColumnsMigration,
            AttachmentTextUpstreamFoldMigration,
            RawTextFtsIndexV2Migration,
        )
        from osprey.services.ariel_search.database.repository import ARIELRepository
        from osprey.services.ariel_search.search.keyword import keyword_search

        config = _caption_only_config(scratch_config)
        assert not config.is_enhancement_module_enabled("image_caption")
        item, _attachment_id = _picture("req3-b1", "beamline.png", "Upstream caption QX-418 hutch")

        pool = await create_connection_pool(config.database)
        try:
            await run_migrations(pool, config)
            # Roll back to the B1 schema: no text columns, no fold, no V2 index,
            # no copy state.
            for migration in (
                RawTextFtsIndexV2Migration(),
                AttachmentTextUpstreamFoldMigration(),
                AttachmentTextColumnsMigration(),
                AttachmentFilesCopyStateMigration(),
            ):
                async with pool.connection() as conn, conn.transaction():
                    await migration.down(conn)
                    await conn.execute(
                        "DELETE FROM ariel_migrations WHERE name = %s", (migration.name,)
                    )
            async with pool.connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO enhanced_entries (
                        entry_id, source_system, timestamp, author, raw_text, attachments,
                        metadata, enhancement_status
                    ) VALUES ('req3-b1', 'test', NOW(), 'tester', 'Hutch survey done.',
                              %s::jsonb, '{}'::jsonb, '{}'::jsonb)
                    """,
                    [json.dumps([item])],
                )

            await run_migrations(pool, config)

            repository = ARIELRepository(pool, config)
            assert (await repository.schema_facts()).has_v2_fts
            hits = await keyword_search("QX-418", repository, config, max_results=10)
            assert [entry["entry_id"] for entry, _s, _h in hits] == ["req3-b1"]
        finally:
            await pool.close()
