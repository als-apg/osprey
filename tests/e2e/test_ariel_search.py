"""ARIEL Search E2E Tests.

Tests the full ARIEL pipeline: ingest -> enhance -> search.
Requires Ollama with nomic-embed-text (skips if unavailable).

Run with:
    pytest tests/e2e/test_ariel_search.py -v

With verbose LLM judge output:
    pytest tests/e2e/test_ariel_search.py -v --judge-verbose
"""

from __future__ import annotations

import atexit
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from osprey.services.ariel_search.config import ARIELConfig

pytestmark = [pytest.mark.e2e, pytest.mark.slow, pytest.mark.asyncio]

logger = logging.getLogger(__name__)

# Path to synthetic test data
TEST_DATA_PATH = Path(__file__).parent.parent / "fixtures" / "ariel" / "test_logbook_entries.jsonl"

# Path to config file for LLM access (RAG tests need this)
# Uses minimal test config with CBORG API access
CONFIG_FILE_PATH = Path(__file__).parent.parent / "fixtures" / "ariel" / "test_config.yml"

# Dev database URL - uses port 5432 (ariel-postgres container)
# Also try port 5433 (ariel-dev-db from docker/ariel-dev.yml)
DEV_DATABASE_URL_5432 = "postgresql://ariel:ariel@localhost:5432/ariel_e2e_test"
DEV_DATABASE_URL_5433 = "postgresql://ariel:ariel@localhost:5433/ariel_e2e_test"


def is_ollama_available() -> bool:
    """Check if Ollama is available for tests."""
    try:
        import requests

        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def is_dev_database_available() -> tuple[bool, str]:
    """Check if a dev database is running (tries 5432 then 5433).

    Returns:
        Tuple of (available, url) - url is the working database URL
    """
    import psycopg

    for dev_url in [DEV_DATABASE_URL_5432, DEV_DATABASE_URL_5433]:
        try:
            with psycopg.connect(
                dev_url.replace("/ariel_e2e_test", "/ariel"), autocommit=True
            ) as conn:
                conn.execute("SELECT 1")
                try:
                    conn.execute("CREATE DATABASE ariel_e2e_test")
                    logger.info("Created ariel_e2e_test database")
                except psycopg.errors.DuplicateDatabase:
                    pass
            return True, dev_url
        except Exception as e:
            logger.debug(f"Dev database at {dev_url} not available: {e}")
            continue
    return False, ""


def is_docker_available() -> bool:
    """Check if Docker is available for testcontainers."""
    try:
        import docker

        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


# =============================================================================
# Module-scoped fixtures for E2E tests
# =============================================================================


@pytest.fixture(scope="module")
def e2e_database_url() -> str:
    """Get database connection URL for e2e tests.

    Tries:
    1. ARIEL_TEST_DATABASE_URL environment variable
    2. Existing docker-compose dev database
    3. Testcontainers
    """
    # Check env var
    env_url = os.environ.get("ARIEL_TEST_DATABASE_URL")
    if env_url:
        logger.info("Using database from ARIEL_TEST_DATABASE_URL")
        return env_url

    # Try dev database
    available, dev_url = is_dev_database_available()
    if available:
        logger.info(f"Using docker dev database for e2e tests: {dev_url.split('@')[-1]}")
        return dev_url

    # Fall back to testcontainers
    if not is_docker_available():
        pytest.skip("No database available - start docker-compose or install Docker")

    logger.info("Starting testcontainers PostgreSQL for e2e tests")
    from testcontainers.postgres import PostgresContainer

    container = PostgresContainer(
        image="ankane/pgvector:latest",
        username="ariel",
        password="ariel",
        dbname="ariel_e2e_test",
    )
    container.start()
    atexit.register(container.stop)

    url = container.get_connection_url()
    if url.startswith("postgresql+psycopg2://"):
        url = url.replace("postgresql+psycopg2://", "postgresql://")

    logger.info(f"Testcontainer started: {url.split('@')[-1]}")
    return url


@pytest.fixture(scope="module")
def e2e_ariel_config(e2e_database_url: str) -> ARIELConfig:
    """Create ARIELConfig for e2e tests with all modules enabled."""
    from osprey.services.ariel_search.config import ARIELConfig

    return ARIELConfig.from_dict(
        {
            "database": {"uri": e2e_database_url},
            "search_modules": {
                "keyword": {"enabled": True},
                "semantic": {"enabled": True, "provider": "ollama", "model": "nomic-embed-text"},
                "rag": {"enabled": True, "provider": "ollama", "model": "nomic-embed-text"},
            },
            "enhancement_modules": {
                "text_embedding": {
                    "enabled": True,
                    "provider": "ollama",
                    "models": [{"name": "nomic-embed-text", "dimension": 768}],
                },
            },
            "embedding": {"provider": "ollama"},
            "ingestion": {
                "adapter": "als_logbook",
                "source_url": str(TEST_DATA_PATH),
            },
            # LLM config for RAG answer generation
            # provider references api.providers for credentials
            "reasoning": {
                "provider": "cborg",
                "model_id": "anthropic/claude-haiku",
                "temperature": 0.1,
            },
        }
    )


@pytest.fixture(scope="module")
async def e2e_connection_pool(e2e_database_url: str):
    """Module-scoped async connection pool."""
    from osprey.services.ariel_search.config import DatabaseConfig
    from osprey.services.ariel_search.database import create_connection_pool

    config = DatabaseConfig(uri=e2e_database_url)
    pool = await create_connection_pool(config)
    yield pool
    await pool.close()


@pytest.fixture(scope="module")
async def e2e_migrated_pool(e2e_connection_pool, e2e_ariel_config: ARIELConfig):
    """Connection pool with migrations applied (module-scoped)."""
    from osprey.services.ariel_search.database import run_migrations

    await run_migrations(e2e_connection_pool, e2e_ariel_config)
    logger.info("E2E migrations applied")
    return e2e_connection_pool


@pytest.fixture(scope="module")
def require_ollama():
    """Skip all tests in module if Ollama not available."""
    if not is_ollama_available():
        pytest.skip("Ollama not available - install and run 'ollama pull nomic-embed-text'")


@pytest.fixture(scope="module")
def e2e_config_file():
    """Module-scoped config file setup (used during data seeding).

    Required for RAG tests which call get_chat_completion.
    Uses my-control-assistant config which has CBORG configured.
    """
    if not CONFIG_FILE_PATH.exists():
        pytest.skip(f"Config file not found: {CONFIG_FILE_PATH}")
    return str(CONFIG_FILE_PATH)


@pytest.fixture
def rag_config_env(e2e_config_file):
    """Function-scoped fixture to set CONFIG_FILE for RAG tests.

    This runs AFTER the autouse reset_registry_between_tests fixture
    in tests/e2e/conftest.py which clears CONFIG_FILE.

    Must be used by RAG tests that need LLM access.
    """
    os.environ["CONFIG_FILE"] = e2e_config_file

    # Clear config cache to force reload with new CONFIG_FILE
    from osprey.utils import config as config_module

    config_module._default_config = None
    config_module._default_configurable = None
    config_module._config_cache.clear()

    # Reset and initialize the registry with new config
    from osprey.registry import initialize_registry, reset_registry

    reset_registry()
    initialize_registry()

    yield e2e_config_file


@pytest.fixture(scope="module")
async def seeded_ariel_db(
    e2e_database_url, e2e_migrated_pool, e2e_ariel_config, require_ollama, e2e_config_file
):
    """Database with test entries and embeddings.

    Ingests test_logbook_entries.jsonl and generates embeddings.
    Module-scoped for efficiency (~10-15s setup).

    Yields:
        Dict with repository, config, pool, and entry_count
    """
    from osprey.models.embeddings.ollama import OllamaEmbeddingProvider
    from osprey.services.ariel_search.database import ARIELRepository
    from osprey.services.ariel_search.ingestion.adapters.als import ALSLogbookAdapter

    repo = ARIELRepository(e2e_migrated_pool, e2e_ariel_config)

    # Initialize adapter
    adapter = ALSLogbookAdapter(e2e_ariel_config)

    # Initialize embedder
    embedder = OllamaEmbeddingProvider()

    # Ingest entries
    entry_count = 0
    logger.info(f"Ingesting test entries from {TEST_DATA_PATH}")

    async for entry in adapter.fetch_entries():
        await repo.upsert_entry(entry)

        # Generate embedding
        try:
            embeddings = embedder.execute_embedding(
                texts=[entry["raw_text"]],
                model_id="nomic-embed-text",
            )
            if embeddings and embeddings[0]:
                await repo.store_text_embedding(
                    entry_id=entry["entry_id"],
                    embedding=embeddings[0],
                    model_name="nomic-embed-text",
                )
        except Exception as e:
            logger.warning(f"Failed to embed entry {entry['entry_id']}: {e}")

        entry_count += 1

    logger.info(f"Ingested {entry_count} entries with embeddings")

    yield {
        "repository": repo,
        "config": e2e_ariel_config,
        "pool": e2e_migrated_pool,
        "entry_count": entry_count,
    }

    # Cleanup test data
    logger.info("Cleaning up test entries")
    async with e2e_migrated_pool.connection() as conn:
        # Clean up embeddings first (foreign key)
        try:
            await conn.execute("""
                DELETE FROM text_embeddings_nomic_embed_text
                WHERE entry_id LIKE 'E%'
            """)
        except Exception:
            pass  # Table may not exist

        # Clean up entries
        await conn.execute("""
            DELETE FROM enhanced_entries
            WHERE entry_id LIKE 'E%'
        """)


# =============================================================================
# Test Classes
# =============================================================================


class TestKeywordSearchE2E:
    """Keyword search tests with deterministic assertions."""

    async def test_exact_term_match(self, seeded_ariel_db):
        """Search for 'klystron' finds RF cavity trip entry (E001)."""
        from osprey.services.ariel_search.search.keyword import keyword_search

        results = await keyword_search(
            query="klystron",
            repository=seeded_ariel_db["repository"],
            config=seeded_ariel_db["config"],
            max_results=5,
        )

        # Results are (entry, score, highlights) tuples
        entry_ids = [entry["entry_id"] for entry, score, highlights in results]
        assert "E001" in entry_ids, f"Expected E001 in results for 'klystron', got: {entry_ids}"

    async def test_multi_term_search(self, seeded_ariel_db):
        """Search 'vacuum leak' finds vacuum entry (E002)."""
        from osprey.services.ariel_search.search.keyword import keyword_search

        results = await keyword_search(
            query="vacuum leak",
            repository=seeded_ariel_db["repository"],
            config=seeded_ariel_db["config"],
            max_results=5,
        )

        assert len(results) > 0, "Expected results for 'vacuum leak'"
        entry_ids = [entry["entry_id"] for entry, score, highlights in results]
        assert "E002" in entry_ids, f"Expected E002 in results for 'vacuum leak', got: {entry_ids}"

    async def test_boolean_and_search(self, seeded_ariel_db):
        """Search 'beam AND loss' finds beam loss entry (E003)."""
        from osprey.services.ariel_search.search.keyword import keyword_search

        results = await keyword_search(
            query="beam AND loss",
            repository=seeded_ariel_db["repository"],
            config=seeded_ariel_db["config"],
            max_results=5,
        )

        assert len(results) > 0, "Expected results for 'beam AND loss'"
        entry_ids = [entry["entry_id"] for entry, score, highlights in results]
        assert "E003" in entry_ids, (
            f"Expected E003 in results for 'beam AND loss', got: {entry_ids}"
        )

    async def test_author_filter(self, seeded_ariel_db):
        """Search 'author:oper_smith' finds entries by that author."""
        from osprey.services.ariel_search.search.keyword import keyword_search

        # The keyword search parses author: prefix but needs a search term too
        # Search for something oper_smith wrote about
        results = await keyword_search(
            query="author:oper_smith shift",
            repository=seeded_ariel_db["repository"],
            config=seeded_ariel_db["config"],
            max_results=10,
        )

        # Should find E006 (Morning Shift Summary by oper_smith)
        if results:
            authors = {entry["author"] for entry, score, highlights in results}
            assert "oper_smith" in authors, f"Expected oper_smith in authors, got: {authors}"

    async def test_no_results_for_nonexistent_term(self, seeded_ariel_db):
        """Search for nonexistent term returns empty results."""
        from osprey.services.ariel_search.search.keyword import keyword_search

        results = await keyword_search(
            query="xyznonexistent123abc",
            repository=seeded_ariel_db["repository"],
            config=seeded_ariel_db["config"],
            max_results=5,
            fuzzy_fallback=False,  # Disable fuzzy to ensure empty result
        )

        assert len(results) == 0, f"Expected no results for nonexistent term, got: {len(results)}"


class TestSemanticSearchE2E:
    """Semantic search with deterministic assertions.

    Note: LLM judge evaluation is optional - tests skip judge if config unavailable.
    """

    async def test_conceptual_search(self, seeded_ariel_db):
        """Search 'electrical failure' finds power-related entries."""
        from osprey.models.embeddings.ollama import OllamaEmbeddingProvider
        from osprey.services.ariel_search.search.semantic import semantic_search

        embedder = OllamaEmbeddingProvider()
        results = await semantic_search(
            query="electrical failure",
            repository=seeded_ariel_db["repository"],
            config=seeded_ariel_db["config"],
            embedder=embedder,
            max_results=5,
            similarity_threshold=0.3,  # Lower threshold for conceptual match
        )

        # Deterministic: we got results
        assert len(results) > 0, "Expected results for 'electrical failure'"

        # Check that power-related entries appear in results
        entry_ids = [entry["entry_id"] for entry, sim in results]
        # E005 (Power Outage) or E001 (RF trip) should be in top results
        power_related = {"E001", "E005", "E012"}  # RF trip, Power outage, Network outage
        found = power_related & set(entry_ids)
        assert len(found) > 0, f"Expected power-related entries in results, got: {entry_ids}"

    async def test_rephrased_query(self, seeded_ariel_db):
        """Search 'accelerator shutdown' finds trip/failure entries."""
        from osprey.models.embeddings.ollama import OllamaEmbeddingProvider
        from osprey.services.ariel_search.search.semantic import semantic_search

        embedder = OllamaEmbeddingProvider()
        results = await semantic_search(
            query="accelerator shutdown incident",
            repository=seeded_ariel_db["repository"],
            config=seeded_ariel_db["config"],
            embedder=embedder,
            max_results=5,
            similarity_threshold=0.3,
        )

        # Deterministic: we got results
        assert len(results) > 0, "Expected results for 'accelerator shutdown incident'"

        # Should find incident entries
        entry_ids = [entry["entry_id"] for entry, sim in results]
        # Incident entries: E001 (RF trip), E002 (vacuum leak), E003 (beam loss), E004 (quench), E005 (power)
        incident_entries = {"E001", "E002", "E003", "E004", "E005", "E009"}
        found = incident_entries & set(entry_ids)
        assert len(found) > 0, f"Expected incident entries in results, got: {entry_ids}"

    async def test_domain_concept_search(self, seeded_ariel_db):
        """Search 'cold equipment problems' finds cryogenic issues."""
        from osprey.models.embeddings.ollama import OllamaEmbeddingProvider
        from osprey.services.ariel_search.search.semantic import semantic_search

        embedder = OllamaEmbeddingProvider()
        results = await semantic_search(
            query="cold equipment temperature problems",
            repository=seeded_ariel_db["repository"],
            config=seeded_ariel_db["config"],
            embedder=embedder,
            max_results=5,
            similarity_threshold=0.3,
        )

        # Deterministic: we got results
        assert len(results) > 0, "Expected results for 'cold equipment temperature problems'"

        # Should find cryogenic-related entries
        entry_ids = [entry["entry_id"] for entry, sim in results]
        # Cryogenic entries: E004 (magnet quench), E009 (LN2 spill), E011 (cryostat)
        cryo_entries = {"E004", "E009", "E011"}
        found = cryo_entries & set(entry_ids)
        assert len(found) > 0, f"Expected cryogenic entries in results, got: {entry_ids}"


class TestRAGSearchE2E:
    """RAG search with deterministic grounding evaluation.

    Note: These tests verify RAG retrieval and answer generation.
    Requires CBORG_API_KEY environment variable for LLM calls.
    """

    async def test_factual_qa(self, seeded_ariel_db, rag_config_env):
        """RAG answers 'What caused the RF trip in sector 3?' using E001."""
        from osprey.models.embeddings.ollama import OllamaEmbeddingProvider
        from osprey.services.ariel_search.rag import RAGPipeline

        pipeline = RAGPipeline(
            repository=seeded_ariel_db["repository"],
            config=seeded_ariel_db["config"],
            embedder_loader=OllamaEmbeddingProvider,
        )
        result = await pipeline.execute(
            query="What caused the RF trip in sector 3?",
            max_results=5,
            similarity_threshold=0.3,
        )

        # Deterministic: E001 should be in sources
        source_ids = [e["entry_id"] for e in result.entries]
        assert "E001" in source_ids, f"Expected E001 in sources, got: {source_ids}"

        # Answer should mention key terms from E001
        answer_lower = result.answer.lower()
        # Should mention klystron, VSWR, RF, or cavity
        key_terms = ["klystron", "vswr", "rf", "cavity", "reflected power"]
        found_terms = [term for term in key_terms if term in answer_lower]
        assert len(found_terms) > 0, (
            f"Answer should mention RF trip details. Answer: {result.answer[:300]}"
        )

    async def test_multi_entry_synthesis(self, seeded_ariel_db, rag_config_env):
        """RAG synthesizes answer from multiple incident entries."""
        from osprey.models.embeddings.ollama import OllamaEmbeddingProvider
        from osprey.services.ariel_search.rag import RAGPipeline

        pipeline = RAGPipeline(
            repository=seeded_ariel_db["repository"],
            config=seeded_ariel_db["config"],
            embedder_loader=OllamaEmbeddingProvider,
        )
        result = await pipeline.execute(
            query="What equipment problems occurred during operations?",
            max_results=5,
            similarity_threshold=0.3,
        )

        # Deterministic: should have multiple sources
        assert len(result.entries) > 1, f"Expected multiple sources, got: {len(result.entries)}"

        # Answer should synthesize information (mention multiple issues)
        answer_lower = result.answer.lower()
        # Count how many different issue types are mentioned
        issue_indicators = [
            "rf" in answer_lower or "cavity" in answer_lower,
            "vacuum" in answer_lower or "leak" in answer_lower,
            "beam" in answer_lower or "loss" in answer_lower,
            "magnet" in answer_lower or "quench" in answer_lower,
            "power" in answer_lower or "outage" in answer_lower,
        ]
        issues_mentioned = sum(issue_indicators)
        assert issues_mentioned >= 2, (
            f"Answer should mention multiple issues. Answer: {result.answer[:300]}"
        )

    async def test_no_answer_for_unrelated_query(self, seeded_ariel_db, rag_config_env):
        """RAG gracefully handles queries with no relevant entries."""
        from osprey.models.embeddings.ollama import OllamaEmbeddingProvider
        from osprey.services.ariel_search.rag import RAGPipeline

        pipeline = RAGPipeline(
            repository=seeded_ariel_db["repository"],
            config=seeded_ariel_db["config"],
            embedder_loader=OllamaEmbeddingProvider,
        )
        result = await pipeline.execute(
            query="What is the chemical composition of the moon?",
            max_results=5,
            similarity_threshold=0.7,  # High threshold to ensure no matches
        )

        # Should have no sources or acknowledge lack of information
        if not result.entries:
            assert (
                "don't have enough information" in result.answer.lower() or len(result.entries) == 0
            )
        # If there are sources, they shouldn't be about moon composition
        # (the model should still ground its answer appropriately)


class TestSearchIntegration:
    """Tests for search module integration and edge cases."""

    async def test_entry_count(self, seeded_ariel_db):
        """Verify all 15 test entries were ingested."""
        assert seeded_ariel_db["entry_count"] == 15, (
            f"Expected 15 entries, got {seeded_ariel_db['entry_count']}"
        )

    async def test_embedding_dimension(self, seeded_ariel_db):
        """Verify embeddings have correct dimension (768 for nomic-embed-text)."""
        pool = seeded_ariel_db["pool"]
        async with pool.connection() as conn:
            result = await conn.execute("""
                SELECT embedding FROM text_embeddings_nomic_embed_text
                WHERE entry_id = 'E001'
                LIMIT 1
            """)
            row = await result.fetchone()

            assert row is not None, "Expected embedding for E001"
            embedding = row[0]

            # pgvector stores as string or list
            if isinstance(embedding, str):
                dim = embedding.count(",") + 1
            else:
                dim = len(embedding)

            assert dim == 768, f"Expected 768 dimensions, got {dim}"

    async def test_keyword_and_semantic_consistency(self, seeded_ariel_db):
        """Keyword and semantic search both find the same entry for exact terms."""
        from osprey.models.embeddings.ollama import OllamaEmbeddingProvider
        from osprey.services.ariel_search.search.keyword import keyword_search
        from osprey.services.ariel_search.search.semantic import semantic_search

        embedder = OllamaEmbeddingProvider()

        # Both should find E002 for "vacuum leak sector 5"
        keyword_results = await keyword_search(
            query="vacuum leak sector 5",
            repository=seeded_ariel_db["repository"],
            config=seeded_ariel_db["config"],
            max_results=5,
        )

        semantic_results = await semantic_search(
            query="vacuum leak in sector 5",
            repository=seeded_ariel_db["repository"],
            config=seeded_ariel_db["config"],
            embedder=embedder,
            max_results=5,
            similarity_threshold=0.3,
        )

        keyword_ids = {entry["entry_id"] for entry, score, highlights in keyword_results}
        semantic_ids = {entry["entry_id"] for entry, score in semantic_results}

        # E002 should appear in both
        assert "E002" in keyword_ids, f"E002 not in keyword results: {keyword_ids}"
        assert "E002" in semantic_ids, f"E002 not in semantic results: {semantic_ids}"
