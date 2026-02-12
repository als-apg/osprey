"""E2E Test: Full Osprey Pipeline with ARIEL Logbook Search.

This test validates the real user experience end-to-end:
  osprey init → database setup → ingest demo logbook → ask agent a logbook question →
  classifier routes → orchestrator plans → LogbookSearchCapability executes →
  ARIELSearchService searches → response generated.

Complements tests/e2e/test_ariel_search.py which tests the ARIEL search layer
in isolation (no Osprey framework). This test validates the full integration.

Run with:
    pytest tests/e2e/test_ariel_e2e_pipeline.py -v

With verbose output:
    pytest tests/e2e/test_ariel_e2e_pipeline.py -v --e2e-verbose

Requirements:
    - Docker (for PostgreSQL) - needed for all tests
    - CBORG_API_KEY env var  - needed for Phase 1.5 (ReAct agent) and Phase 2 (full pipeline)
"""

from __future__ import annotations

import atexit
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml

if TYPE_CHECKING:
    from osprey.services.ariel_search.config import ARIELConfig

pytestmark = [pytest.mark.e2e, pytest.mark.slow, pytest.mark.asyncio]

logger = logging.getLogger(__name__)

# Dev database URLs - try port 5432 (ariel-postgres), then 5433 (ariel-dev-db)
DEV_DATABASE_URL_5432 = "postgresql://ariel:ariel@localhost:5432/ariel_pipeline_test"
DEV_DATABASE_URL_5433 = "postgresql://ariel:ariel@localhost:5433/ariel_pipeline_test"


# =============================================================================
# Database Discovery (same pattern as test_ariel_search.py)
# =============================================================================


def _is_dev_database_available() -> tuple[bool, str]:
    """Check if a dev database is running (tries 5432 then 5433)."""
    import psycopg

    for dev_url in [DEV_DATABASE_URL_5432, DEV_DATABASE_URL_5433]:
        base_url = dev_url.rsplit("/", 1)[0] + "/ariel"
        try:
            with psycopg.connect(base_url, autocommit=True) as conn:
                conn.execute("SELECT 1")
                try:
                    conn.execute("CREATE DATABASE ariel_pipeline_test")
                    logger.info("Created ariel_pipeline_test database")
                except psycopg.errors.DuplicateDatabase:
                    pass
            return True, dev_url
        except Exception as e:
            logger.debug(f"Dev database at {dev_url} not available: {e}")
            continue
    return False, ""


def _is_docker_available() -> bool:
    """Check if Docker is available for testcontainers."""
    try:
        import docker

        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


# =============================================================================
# Module-Scoped Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def e2e_ariel_database_url() -> str:
    """Get database URL for pipeline E2E tests.

    Tries: env var → docker-compose dev DB → testcontainers.
    """
    env_url = os.environ.get("ARIEL_TEST_DATABASE_URL")
    if env_url:
        logger.info("Using database from ARIEL_TEST_DATABASE_URL")
        return env_url

    available, dev_url = _is_dev_database_available()
    if available:
        logger.info(f"Using docker dev database: {dev_url.split('@')[-1]}")
        return dev_url

    if not _is_docker_available():
        pytest.skip("No database available - start docker-compose or install Docker")

    logger.info("Starting testcontainers PostgreSQL for pipeline e2e tests")
    from testcontainers.postgres import PostgresContainer

    container = PostgresContainer(
        image="ankane/pgvector:latest",
        username="ariel",
        password="ariel",
        dbname="ariel_pipeline_test",
    )
    container.start()
    atexit.register(container.stop)

    url = container.get_connection_url()
    if url.startswith("postgresql+psycopg2://"):
        url = url.replace("postgresql+psycopg2://", "postgresql://")

    logger.info(f"Testcontainer started: {url.split('@')[-1]}")
    return url


@pytest.fixture(scope="module")
def e2e_ariel_config(e2e_ariel_database_url: str) -> ARIELConfig:
    """Create ARIELConfig with keyword search only (no Ollama dependency)."""
    from osprey.services.ariel_search.config import ARIELConfig

    return ARIELConfig.from_dict(
        {
            "database": {"uri": e2e_ariel_database_url},
            "search_modules": {
                "keyword": {"enabled": True},
                "semantic": {"enabled": False},
                "rag": {"enabled": False},
            },
            "enhancement_modules": {
                "text_embedding": {"enabled": False},
            },
            "embedding": {"provider": "ollama"},
            "ingestion": {
                "adapter": "generic_json",
                "source_url": str(
                    Path(__file__).parent.parent.parent
                    / "src"
                    / "osprey"
                    / "templates"
                    / "apps"
                    / "control_assistant"
                    / "data"
                    / "logbook_seed"
                    / "demo_logbook.json"
                ),
            },
            "reasoning": {
                "provider": "cborg",
                "model_id": "anthropic/claude-haiku",
                "temperature": 0.1,
            },
        }
    )


@pytest.fixture(scope="module")
async def e2e_ariel_migrated_pool(e2e_ariel_database_url: str, e2e_ariel_config: ARIELConfig):
    """Connection pool with migrations applied (module-scoped)."""
    from osprey.services.ariel_search.config import DatabaseConfig
    from osprey.services.ariel_search.database import create_connection_pool, run_migrations

    config = DatabaseConfig(uri=e2e_ariel_database_url)
    pool = await create_connection_pool(config)
    await run_migrations(pool, e2e_ariel_config)

    # Ensure pg_trgm is available for keyword fuzzy fallback.
    # Normally installed by SemanticProcessorMigration, but these tests
    # disable semantic search to avoid an Ollama dependency.
    async with pool.connection() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    logger.info("Pipeline E2E migrations applied")
    yield pool
    await pool.close()


@pytest.fixture(scope="module")
async def e2e_ariel_seeded_db(
    e2e_ariel_database_url: str,
    e2e_ariel_migrated_pool,
    e2e_ariel_config: ARIELConfig,
):
    """Database with demo logbook entries ingested (25 entries, no embeddings).

    Uses GenericJSONAdapter to ingest demo_logbook.json.
    Module-scoped for efficiency.

    Yields:
        Dict with repository, config, pool, and entry_count.
    """
    from osprey.services.ariel_search.database import ARIELRepository
    from osprey.services.ariel_search.ingestion.adapters.generic import GenericJSONAdapter

    repo = ARIELRepository(e2e_ariel_migrated_pool, e2e_ariel_config)
    adapter = GenericJSONAdapter(e2e_ariel_config)

    entry_count = 0
    logger.info(f"Ingesting demo logbook entries from {e2e_ariel_config.ingestion.source_url}")

    async for entry in adapter.fetch_entries():
        await repo.upsert_entry(entry)
        entry_count += 1

    logger.info(f"Ingested {entry_count} demo logbook entries")

    yield {
        "repository": repo,
        "config": e2e_ariel_config,
        "pool": e2e_ariel_migrated_pool,
        "entry_count": entry_count,
        "database_url": e2e_ariel_database_url,
    }

    # Cleanup
    logger.info("Cleaning up demo logbook entries")
    async with e2e_ariel_migrated_pool.connection() as conn:
        await conn.execute("""
            DELETE FROM enhanced_entries
            WHERE entry_id LIKE 'DEMO-%'
        """)


# =============================================================================
# Phase 1: Database + Surgical Search Tests (no LLM needed)
# =============================================================================


class TestDemoDataIngestion:
    """Verify demo data is correctly ingested and searchable."""

    async def test_demo_data_ingested(self, e2e_ariel_seeded_db):
        """All 25 demo logbook entries are in the database."""
        assert e2e_ariel_seeded_db["entry_count"] == 25, (
            f"Expected 25 entries, got {e2e_ariel_seeded_db['entry_count']}"
        )

        # Also verify via repository count
        count = await e2e_ariel_seeded_db["repository"].count_entries()
        assert count >= 25, f"Expected at least 25 entries in DB, got {count}"

    async def test_entry_has_expected_fields(self, e2e_ariel_seeded_db):
        """Verify ingested entries have all expected fields."""
        repo = e2e_ariel_seeded_db["repository"]
        entry = await repo.get_entry("DEMO-001")

        assert entry is not None, "DEMO-001 not found in database"
        assert entry["entry_id"] == "DEMO-001"
        assert entry["author"] == "J. Smith"
        assert "RF cavity" in entry["raw_text"] or "rf cavity" in entry["raw_text"].lower()
        assert entry["timestamp"] is not None


class TestKeywordSearchDemoData:
    """Keyword search tests against demo logbook data (deterministic, no LLM)."""

    async def test_keyword_search_rf_cavity(self, e2e_ariel_seeded_db):
        """Search 'RF cavity' finds DEMO-001 (RF trip) and DEMO-010 (RF inspection)."""
        from osprey.services.ariel_search.search.keyword import keyword_search

        results = await keyword_search(
            query="RF cavity",
            repository=e2e_ariel_seeded_db["repository"],
            config=e2e_ariel_seeded_db["config"],
            max_results=10,
        )

        entry_ids = [entry["entry_id"] for entry, score, highlights in results]
        assert "DEMO-001" in entry_ids, (
            f"Expected DEMO-001 in results for 'RF cavity', got: {entry_ids}"
        )
        assert "DEMO-010" in entry_ids, (
            f"Expected DEMO-010 in results for 'RF cavity', got: {entry_ids}"
        )

    async def test_keyword_search_vacuum(self, e2e_ariel_seeded_db):
        """Search 'vacuum' finds entries mentioning vacuum events."""
        from osprey.services.ariel_search.search.keyword import keyword_search

        results = await keyword_search(
            query="vacuum",
            repository=e2e_ariel_seeded_db["repository"],
            config=e2e_ariel_seeded_db["config"],
            max_results=10,
        )

        entry_ids = [entry["entry_id"] for entry, score, highlights in results]
        # DEMO-001 (vacuum spike), DEMO-005 (vacuum valve maintenance), DEMO-013 (vacuum chamber)
        expected = {"DEMO-001", "DEMO-005", "DEMO-013"}
        found = expected & set(entry_ids)
        assert len(found) >= 2, (
            f"Expected at least 2 of {expected} in results for 'vacuum', got: {entry_ids}"
        )

    async def test_keyword_search_shift_summary(self, e2e_ariel_seeded_db):
        """Search 'shift summary' finds all shift summary entries."""
        from osprey.services.ariel_search.search.keyword import keyword_search

        results = await keyword_search(
            query="shift summary",
            repository=e2e_ariel_seeded_db["repository"],
            config=e2e_ariel_seeded_db["config"],
            max_results=10,
        )

        entry_ids = [entry["entry_id"] for entry, score, highlights in results]
        # DEMO-003, DEMO-009, DEMO-015, DEMO-021 are shift summaries
        expected = {"DEMO-003", "DEMO-009", "DEMO-015", "DEMO-021"}
        found = expected & set(entry_ids)
        assert len(found) >= 3, (
            f"Expected at least 3 of {expected} in results for 'shift summary', got: {entry_ids}"
        )

    async def test_keyword_search_loto(self, e2e_ariel_seeded_db):
        """Search 'LOTO' finds entries with LOTO procedures."""
        from osprey.services.ariel_search.search.keyword import keyword_search

        results = await keyword_search(
            query="LOTO",
            repository=e2e_ariel_seeded_db["repository"],
            config=e2e_ariel_seeded_db["config"],
            max_results=10,
        )

        entry_ids = [entry["entry_id"] for entry, score, highlights in results]
        # DEMO-005 (LOTO-2024-0312), DEMO-020 (LOTO-2024-0318), DEMO-022 (references LOTO-0318)
        expected = {"DEMO-005", "DEMO-020"}
        found = expected & set(entry_ids)
        assert len(found) >= 1, (
            f"Expected at least 1 of {expected} in results for 'LOTO', got: {entry_ids}"
        )

    async def test_keyword_search_author_filter(self, e2e_ariel_seeded_db):
        """Search with 'author:Smith' filters to J. Smith entries only."""
        from osprey.services.ariel_search.search.keyword import keyword_search

        results = await keyword_search(
            query="author:Smith beam",
            repository=e2e_ariel_seeded_db["repository"],
            config=e2e_ariel_seeded_db["config"],
            max_results=10,
        )

        if results:
            authors = {entry["author"] for entry, score, highlights in results}
            # All results should be from an author containing "Smith"
            for author in authors:
                assert "Smith" in author, f"Expected author containing 'Smith', got: {author}"

    async def test_keyword_search_with_time_range(self, e2e_ariel_seeded_db):
        """Search with date range filters to entries in that window."""
        from osprey.services.ariel_search.search.keyword import keyword_search

        # March 20-22, 2024 should contain DEMO-011 through DEMO-016
        start = datetime(2024, 3, 20, 0, 0, 0, tzinfo=UTC)
        end = datetime(2024, 3, 22, 23, 59, 59, tzinfo=UTC)

        results = await keyword_search(
            query="beam",
            repository=e2e_ariel_seeded_db["repository"],
            config=e2e_ariel_seeded_db["config"],
            max_results=25,
            start_date=start,
            end_date=end,
        )

        # All returned entries should be within the time window
        for entry, _score, _highlights in results:
            ts = entry["timestamp"]
            if hasattr(ts, "tzinfo") and ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            assert start <= ts <= end, (
                f"Entry {entry['entry_id']} timestamp {ts} outside range [{start}, {end}]"
            )

    async def test_keyword_search_no_results(self, e2e_ariel_seeded_db):
        """Search for nonexistent term returns empty results."""
        from osprey.services.ariel_search.search.keyword import keyword_search

        results = await keyword_search(
            query="xyznonexistent123abc",
            repository=e2e_ariel_seeded_db["repository"],
            config=e2e_ariel_seeded_db["config"],
            max_results=5,
            fuzzy_fallback=False,
        )

        assert len(results) == 0, f"Expected no results, got: {len(results)}"


# =============================================================================
# Phase 1.5: ARIEL ReAct Agent E2E Tests (requires CBORG API key)
# =============================================================================


@pytest.fixture
def e2e_agent_config_env():
    """Set CONFIG_FILE for agent executor LLM access (CBORG).

    The AgentExecutor's _get_llm() calls get_provider_config("cborg") which
    reads from Osprey's config system. This fixture points CONFIG_FILE at the
    test config that has CBORG credentials.

    Runs after the autouse reset_registry_between_tests fixture (which clears
    CONFIG_FILE), so the env var is set fresh for each test.
    """
    config_path = Path(__file__).parent.parent / "fixtures" / "ariel" / "test_config.yml"
    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")

    os.environ["CONFIG_FILE"] = str(config_path)
    from osprey.utils import config as config_module

    config_module._default_config = None
    config_module._default_configurable = None
    config_module._config_cache.clear()

    # Reset and initialize the registry so get_chat_completion works (LLM judge)
    from osprey.registry import initialize_registry, reset_registry

    reset_registry()
    initialize_registry()

    yield

    if "CONFIG_FILE" in os.environ:
        del os.environ["CONFIG_FILE"]
    config_module._default_config = None
    config_module._default_configurable = None
    config_module._config_cache.clear()


@pytest.mark.requires_cborg
class TestAgentExecutorE2E:
    """ARIEL ReAct agent e2e tests against demo logbook data.

    Tests the AgentExecutor (LangGraph ReAct agent) directly via
    ARIELSearchService.search(mode=AGENT). Only keyword_search tool
    is available (semantic disabled to avoid Ollama dependency).

    Requires CBORG_API_KEY for LLM calls.
    """

    async def test_agent_finds_safety_procedures(
        self, e2e_ariel_seeded_db, e2e_agent_config_env, llm_judge
    ):
        """Agent searches for safety procedures and synthesizes answer.

        Query: "What safety procedures were followed for the power supply work in Sector 8?"
        Expected: Agent finds DEMO-020 (LOTO applied) and DEMO-022 (PS replacement),
        synthesizes answer about LOTO procedures and safety checks.
        """
        from osprey.services.ariel_search.models import SearchMode
        from osprey.services.ariel_search.service import ARIELSearchService

        db = e2e_ariel_seeded_db
        service = ARIELSearchService(
            config=db["config"],
            pool=db["pool"],
            repository=db["repository"],
        )

        query = "What safety procedures were followed for the power supply work in Sector 8?"
        result = await service.search(query, mode=SearchMode.AGENT)

        # --- Deterministic assertions ---

        # Agent should have produced an answer
        assert result.answer is not None, "Agent did not produce an answer"
        assert len(result.answer) > 50, f"Answer too short: {result.answer}"

        # Agent should have used keyword_search tool
        assert SearchMode.KEYWORD in result.search_modes_used, (
            f"Expected KEYWORD in search modes, got: {result.search_modes_used}"
        )

        # --- LLM judge evaluation ---
        expectations = """
        The answer should:
        1. Mention LOTO (Lock-Out/Tag-Out) procedures — specifically LOTO-2024-0318
        2. Reference safety checks like zero energy verification or circuit breaker lockout
        3. Be grounded in actual logbook entries (not hallucinated)
        4. Mention the power supply replacement work on quadrupole magnet QF-08C
        """
        evaluation = await llm_judge.evaluate_text(
            result_text=result.answer,
            expectations=expectations,
            query=query,
        )
        assert evaluation.passed, (
            f"LLM judge failed (confidence={evaluation.confidence}):\n{evaluation.reasoning}"
        )

    async def test_agent_handles_irrelevant_query(self, e2e_ariel_seeded_db, e2e_agent_config_env):
        """Agent handles queries with no relevant logbook entries gracefully."""
        from osprey.services.ariel_search.models import SearchMode
        from osprey.services.ariel_search.service import ARIELSearchService

        db = e2e_ariel_seeded_db
        service = ARIELSearchService(
            config=db["config"],
            pool=db["pool"],
            repository=db["repository"],
        )

        result = await service.search(
            "What is the recipe for chocolate cake?",
            mode=SearchMode.AGENT,
        )

        # Agent should still produce an answer (not crash)
        assert result.answer is not None, (
            "Agent should produce an answer even for irrelevant queries"
        )


# =============================================================================
# Phase 2: Full Osprey Agent Pipeline Tests (requires CBORG API key)
# =============================================================================


def _patch_config_for_test_db(config_path: Path, database_url: str) -> None:
    """Patch a project's config.yml to point ARIEL at the test database.

    Also ensures keyword search is enabled and semantic/rag are disabled
    (no Ollama dependency for pipeline tests).
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Patch database URI
    if "ariel" not in config:
        config["ariel"] = {}
    if "database" not in config["ariel"]:
        config["ariel"]["database"] = {}
    config["ariel"]["database"]["uri"] = database_url

    # Ensure keyword is enabled, semantic disabled
    if "search_modules" not in config["ariel"]:
        config["ariel"]["search_modules"] = {}
    config["ariel"]["search_modules"]["keyword"] = {"enabled": True}
    config["ariel"]["search_modules"]["semantic"] = {"enabled": False}

    # Disable enhancement modules (no Ollama)
    if "enhancement_modules" not in config["ariel"]:
        config["ariel"]["enhancement_modules"] = {}
    config["ariel"]["enhancement_modules"]["text_embedding"] = {"enabled": False}

    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)


@pytest.mark.requires_cborg
class TestAgentLogbookPipeline:
    """Full Osprey agent pipeline tests with ARIEL logbook search.

    Each test:
    1. Scaffolds a control_assistant project via e2e_project_factory
    2. Patches config.yml to point at the test database
    3. Initializes the full framework
    4. Executes a query through the complete agent graph
    5. Verifies the response with deterministic assertions + LLM judge
    """

    async def test_agent_routes_logbook_query(
        self, e2e_ariel_seeded_db, e2e_project_factory, llm_judge
    ):
        """Agent routes an RF cavity query to logbook_search capability.

        Query: "Search the logbook for any RF cavity trips or issues"
        Expected: Classifier routes to logbook_search, finds DEMO-001/DEMO-010

        What this test checks (deterministic):
          - No workflow errors
          - Classifier routes to logbook_search capability
          - Response references specific RF cavity entries from the demo data
            (DEMO-001: RF cavity trip, DEMO-010: RF inspection)
          - Response is substantive (>100 chars)

        Optional (when CBORG_API_KEY is set):
          - LLM judge evaluates overall workflow quality
        """
        # Reset ARIEL service singleton before test
        from osprey.services.ariel_search.capability import reset_ariel_service

        reset_ariel_service()

        # 1. Scaffold project
        project = await e2e_project_factory(
            name="ariel-rf-test",
            template="control_assistant",
            registry_style="extend",
        )

        # 2. Patch config to use test database
        _patch_config_for_test_db(project.config_path, e2e_ariel_seeded_db["database_url"])

        # 3. Initialize framework
        await project.initialize()

        # 4. Execute logbook query
        result = await project.query(
            "Search the logbook for any RF cavity trips or issues that were reported"
        )

        # --- Deterministic assertions ---

        # A. No errors
        assert result.error is None, f"Workflow error: {result.error}"

        # B. Capability routing: logbook_search must appear in the execution trace
        trace_lower = result.execution_trace.lower()
        assert "logbook_search" in trace_lower, (
            "Classifier did not route to logbook_search capability.\n"
            f"Execution trace excerpt:\n{result.execution_trace[:500]}"
        )

        # C. Response should reference specific demo entries about RF cavities
        response_lower = result.response.lower()

        # The demo data has DEMO-001 (RF cavity trip) and DEMO-010 (RF inspection).
        # The agent's response should cite at least one of them or their content.
        rf_evidence = any(
            [
                "demo-001" in response_lower,
                "demo-010" in response_lower,
                "reflected power" in response_lower,
                "rf cavity trip" in response_lower,
                "rf inspection" in response_lower,
                "cavity inspection" in response_lower,
                # The agent might quote the entry text directly
                "j. smith" in response_lower and "rf" in response_lower,
                "m. chen" in response_lower and "rf" in response_lower,
            ]
        )
        assert rf_evidence, (
            "Response does not reference any RF cavity entries from the demo data.\n"
            "Expected mention of DEMO-001 (RF cavity trip) or DEMO-010 (RF inspection),\n"
            "or specific details like 'reflected power', 'RF cavity trip', 'RF inspection'.\n"
            f"Response preview:\n{result.response[:500]}"
        )

        # D. Response should be substantive (not just an error message or empty)
        assert len(result.response) > 100, (
            f"Response too short ({len(result.response)} chars), expected substantive answer.\n"
            f"Full response: {result.response}"
        )

        # --- Optional LLM judge (runs when available, does not gate the test) ---
        try:
            expectations = """
            The workflow should:
            1. Route the query to the logbook_search capability (not channel_finding or python)
            2. Search the logbook database for RF cavity related entries
            3. Find at least one relevant entry about RF cavity issues
            4. Return a coherent response that mentions RF cavity details from the logbook
            5. Complete without critical errors
            """
            evaluation = await llm_judge.evaluate(result=result, expectations=expectations)
            if not evaluation.passed:
                import warnings

                warnings.warn(
                    f"LLM judge flagged potential issue (confidence={evaluation.confidence}):\n"
                    f"{evaluation.reasoning}",
                    UserWarning,
                    stacklevel=2,
                )
        except Exception as e:
            # Judge unavailable (no API key, network issue) — not a test failure
            import warnings

            warnings.warn(f"LLM judge skipped: {e}", UserWarning, stacklevel=2)

        # Cleanup
        reset_ariel_service()


# =============================================================================
# Phase 3: Custom Search + Enhancement Module E2E Tests
# =============================================================================


@pytest.fixture
async def e2e_ariel_watermarked_db(e2e_ariel_seeded_db):
    """Database with DEMO-001 stamped with a watermark in metadata.

    Depends on the module-scoped ``e2e_ariel_seeded_db`` (data already ingested).
    Applies the watermark, yields, then removes it to avoid polluting the
    shared module-scoped fixture.
    """
    from tests.e2e._e2e_watermark_enhancer import WATERMARK_ENTRY_ID, WATERMARK_VALUE

    pool = e2e_ariel_seeded_db["pool"]

    # Stamp the watermark
    async with pool.connection() as conn:
        await conn.execute(
            """
            UPDATE enhanced_entries
            SET metadata = metadata || %s::jsonb
            WHERE entry_id = %s
            """,
            [f'{{"e2e_watermark": "{WATERMARK_VALUE}"}}', WATERMARK_ENTRY_ID],
        )

    yield e2e_ariel_seeded_db

    # Cleanup: remove the watermark key
    async with pool.connection() as conn:
        await conn.execute(
            """
            UPDATE enhanced_entries
            SET metadata = metadata - 'e2e_watermark'
            WHERE entry_id = %s
            """,
            [WATERMARK_ENTRY_ID],
        )


@pytest.mark.requires_cborg
class TestCustomModuleIntegration:
    """Custom search + enhancement module e2e integration.

    Verifies that a user-provided search module, registered through the
    public registry API, is discovered by the ReAct agent and produces
    correct results against real data.
    """

    async def test_agent_uses_custom_search_module(
        self, e2e_ariel_watermarked_db, e2e_agent_config_env
    ):
        """Agent discovers and uses custom watermark_search tool.

        Flow:
        1. DEMO-001 already has ``e2e_watermark: CANARY_12345`` in metadata
        2. Register the watermark_search module in the Osprey registry
        3. Create ARIELSearchService with config enabling watermark_search
        4. Ask the agent about watermarked entries
        5. Assert the agent found DEMO-001 and mentions the watermark
        """
        import sys

        from osprey.registry import (
            ArielSearchModuleRegistration,
            get_registry,
        )
        from osprey.services.ariel_search.config import ARIELConfig
        from osprey.services.ariel_search.models import SearchMode
        from osprey.services.ariel_search.service import ARIELSearchService

        db = e2e_ariel_watermarked_db

        # --- 1. Make the custom module importable by the registry ---
        # The registry calls importlib.import_module(module_path), so we
        # insert the module under a known dotted name.
        import tests.e2e._e2e_watermark_search as wm_module

        module_path = "tests.e2e._e2e_watermark_search"
        sys.modules[module_path] = wm_module

        # --- 2. Register the custom search module in the live registry ---
        registry = get_registry()
        registration = ArielSearchModuleRegistration(
            name="watermark_search",
            module_path=module_path,
            description="E2E watermark search module",
        )
        registry.config.ariel_search_modules.append(registration)
        # Import and store the module directly (same as _initialize_ariel_search_modules)
        registry._registries["ariel_search_modules"]["watermark_search"] = wm_module

        # --- 3. Build an ARIELConfig with watermark_search enabled ---
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": db["database_url"]},
                "search_modules": {
                    "keyword": {"enabled": True},
                    "watermark_search": {"enabled": True},
                },
                "enhancement_modules": {
                    "text_embedding": {"enabled": False},
                },
                "pipelines": {
                    "agent": {
                        "enabled": True,
                        "retrieval_modules": ["keyword", "watermark_search"],
                    },
                },
                "embedding": {"provider": "ollama"},
                "reasoning": db["config"].reasoning.__dict__.copy(),
            }
        )

        # --- 4. Create service and ask the agent ---
        service = ARIELSearchService(
            config=config,
            pool=db["pool"],
            repository=db["repository"],
        )

        result = await service.search(
            "Are there any watermarked entries in the logbook?",
            mode=SearchMode.AGENT,
        )

        # --- 5. Assertions ---
        assert result.answer is not None, "Agent did not produce an answer"
        assert len(result.answer) > 50, f"Answer too short: {result.answer}"

        answer_lower = result.answer.lower()

        # The agent should reference DEMO-001 or its content (RF cavity)
        entry_evidence = any(
            [
                "demo-001" in answer_lower,
                "rf cavity" in answer_lower,
                "j. smith" in answer_lower,
            ]
        )
        assert entry_evidence, (
            "Response does not reference DEMO-001 or its content.\n"
            f"Response preview:\n{result.answer[:500]}"
        )

        # The watermark value should appear in the response
        assert "canary_12345" in answer_lower or "canary" in answer_lower, (
            "Response does not mention the watermark value CANARY_12345.\n"
            f"Response preview:\n{result.answer[:500]}"
        )
