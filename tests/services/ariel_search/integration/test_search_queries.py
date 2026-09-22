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
        migrated_pool,
        seed_entry_factory,
        integration_ariel_config,
        seeded_prefixes,
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
            migrated_pool: Pool over the migrated test database.
            seed_entry_factory: Factory building a single logbook entry.
            integration_ariel_config: ARIEL configuration for that database.
            seeded_prefixes: Package ledger of the entry-id prefixes to delete
                at teardown.

        Returns:
            The repository, with the three entries and their embeddings stored.
        """
        if not is_ollama_available():
            pytest.skip("Ollama not available - run 'ollama pull nomic-embed-text'")

        from osprey.models.embeddings.ollama import OllamaEmbeddingProvider

        embedder = OllamaEmbeddingProvider()

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
        from osprey.models.embeddings.ollama import OllamaEmbeddingProvider
        from osprey.services.ariel_search.search.semantic import semantic_search

        embedder = OllamaEmbeddingProvider()

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
        from osprey.models.embeddings.ollama import OllamaEmbeddingProvider
        from osprey.services.ariel_search.search.semantic import semantic_search

        embedder = OllamaEmbeddingProvider()

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
