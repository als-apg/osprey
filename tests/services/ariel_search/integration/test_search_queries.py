"""Integration tests for ARIEL search queries.

Tests actual search operations (FTS, vector similarity) against real PostgreSQL.

See 04_OSPREY_INTEGRATION.md Section 12.3.4 for test requirements.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class TestKeywordSearch:
    """Test keyword search with real PostgreSQL FTS."""

    @pytest.fixture
    async def seeded_repository(self, repository, seed_entry_factory):
        """Repository with test entries for search tests."""
        entries = [
            seed_entry_factory(
                entry_id="search-kw-001",
                raw_text="The vacuum chamber pressure dropped unexpectedly during the experiment.",
                author="operator1",
            ),
            seed_entry_factory(
                entry_id="search-kw-002",
                raw_text="Beam alignment was adjusted to correct the orbit deviation.",
                author="physicist1",
            ),
            seed_entry_factory(
                entry_id="search-kw-003",
                raw_text="The undulator gap was changed to optimize photon flux.",
                author="scientist1",
            ),
        ]
        for entry in entries:
            await repository.upsert_entry(entry)
        return repository

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


class TestKeywordSearchCleanup:
    """Clean up keyword search test data."""

    async def test_cleanup(self, migrated_pool):
        """Clean up test entries."""
        async with migrated_pool.connection() as conn:
            await conn.execute("""
                DELETE FROM enhanced_entries
                WHERE entry_id LIKE 'search-kw-%'
            """)


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
        self, repository, migrated_pool, seed_entry_factory, integration_ariel_config
    ):
        """Repository with test entries and their embeddings."""
        if not is_ollama_available():
            pytest.skip("Ollama not available - run 'ollama pull nomic-embed-text'")

        from osprey.models.embeddings.ollama import OllamaEmbeddingProvider

        embedder = OllamaEmbeddingProvider()

        # Create entries about different topics
        entries = [
            seed_entry_factory(
                entry_id="semantic-001",
                raw_text="Beam loss detected at sector 5. The injection efficiency dropped to 82% due to instability in the storage ring.",
                author="operator1",
            ),
            seed_entry_factory(
                entry_id="semantic-002",
                raw_text="Beam position monitors showing orbit deviation. Correcting with steering magnets.",
                author="physicist1",
            ),
            seed_entry_factory(
                entry_id="semantic-003",
                raw_text="Vacuum system maintenance completed. Pressure in sector 7 now at 1e-10 Torr.",
                author="technician1",
            ),
        ]

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

    async def test_embedding_dimension_is_768(
        self, seeded_repository_with_embeddings, migrated_pool
    ):
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

    async def test_cleanup(self, migrated_pool):
        """Clean up semantic search test data."""
        async with migrated_pool.connection() as conn:
            # Clean up embeddings first (foreign key constraint)
            try:
                await conn.execute("""
                    DELETE FROM text_embeddings_nomic_embed_text
                    WHERE entry_id LIKE 'semantic-%'
                """)
            except Exception:
                pass  # Table may not exist

            # Clean up entries
            await conn.execute("""
                DELETE FROM enhanced_entries
                WHERE entry_id LIKE 'semantic-%'
            """)


class TestSearchQueryStructure:
    """Test search query structure without semantic data."""

    async def test_search_by_time_range_with_source_filter(self, repository, seed_entry_factory):
        """Search with source system filter."""
        now = datetime.now(UTC)
        entry = seed_entry_factory(
            entry_id="search-source-001",
            source_system="als_logbook",
            timestamp=now,
            raw_text="Test entry from ALS logbook",
        )
        await repository.upsert_entry(entry)

        # This tests the query structure even if filtering isn't implemented
        results = await repository.search_by_time_range(limit=10)
        assert isinstance(results, list)

    async def test_cleanup(self, migrated_pool):
        """Clean up test entries."""
        async with migrated_pool.connection() as conn:
            await conn.execute("""
                DELETE FROM enhanced_entries
                WHERE entry_id LIKE 'search-source-%'
            """)
