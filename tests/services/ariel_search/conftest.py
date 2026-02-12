"""ARIEL Search test fixtures.

Provides database access for integration tests. Prefers using the existing
docker-compose dev database (ariel-dev.yml) for speed, falling back to
testcontainers if unavailable.

See 04_OSPREY_INTEGRATION.md Section 12.3.4 for test requirements.
"""

from __future__ import annotations

import logging
import os
import types
from datetime import UTC
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.services.ariel_search.enhancement.semantic_processor.processor import (
    SemanticProcessorModule,
)
from osprey.services.ariel_search.enhancement.text_embedding.embedder import (
    TextEmbeddingModule,
)

if TYPE_CHECKING:
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.database.repository import ARIELRepository

logger = logging.getLogger(__name__)


def _build_ariel_mock_registry():
    """Build a mock Osprey registry with ARIEL search/enhancement/pipeline/ingestion modules."""
    from osprey.registry.base import (
        ArielEnhancementModuleRegistration,
        ArielIngestionAdapterRegistration,
    )

    registry = MagicMock()

    # --- Search modules ---
    from osprey.services.ariel_search.search import keyword as kw_real
    from osprey.services.ariel_search.search import semantic as sem_real

    kw_mod = types.ModuleType("keyword")
    kw_mod.get_tool_descriptor = kw_real.get_tool_descriptor  # type: ignore[attr-defined]
    kw_mod.get_parameter_descriptors = getattr(kw_real, "get_parameter_descriptors", None)  # type: ignore[attr-defined]

    sem_mod = types.ModuleType("semantic")
    sem_mod.get_tool_descriptor = sem_real.get_tool_descriptor  # type: ignore[attr-defined]
    sem_mod.get_parameter_descriptors = getattr(sem_real, "get_parameter_descriptors", None)  # type: ignore[attr-defined]

    _search_modules = {"keyword": kw_mod, "semantic": sem_mod}
    registry.list_ariel_search_modules.return_value = list(_search_modules)
    registry.get_ariel_search_module.side_effect = _search_modules.get

    # --- Enhancement modules ---
    sp_reg = ArielEnhancementModuleRegistration(
        name="semantic_processor",
        module_path="osprey.services.ariel_search.enhancement.semantic_processor.processor",
        class_name="SemanticProcessorModule",
        description="Semantic processor",
        execution_order=10,
    )
    te_reg = ArielEnhancementModuleRegistration(
        name="text_embedding",
        module_path="osprey.services.ariel_search.enhancement.text_embedding.embedder",
        class_name="TextEmbeddingModule",
        description="Text embedding",
        execution_order=20,
    )
    _enhancement_modules = {
        "semantic_processor": (SemanticProcessorModule, sp_reg),
        "text_embedding": (TextEmbeddingModule, te_reg),
    }
    registry.list_ariel_enhancement_modules.return_value = [
        "semantic_processor",
        "text_embedding",
    ]
    registry.get_ariel_enhancement_module.side_effect = _enhancement_modules.get

    # --- Pipelines ---
    from osprey.services.ariel_search.pipelines import get_pipeline_descriptor

    rag_mod = types.ModuleType("rag")
    rag_mod.get_pipeline_descriptor = get_pipeline_descriptor  # type: ignore[attr-defined]

    agent_mod = types.ModuleType("agent")
    agent_mod.get_pipeline_descriptor = get_pipeline_descriptor  # type: ignore[attr-defined]

    _pipelines = {"rag": rag_mod, "agent": agent_mod}
    registry.list_ariel_pipelines.return_value = list(_pipelines)
    registry.get_ariel_pipeline.side_effect = _pipelines.get

    # --- Ingestion adapters ---
    from osprey.services.ariel_search.ingestion.adapters.als import ALSLogbookAdapter
    from osprey.services.ariel_search.ingestion.adapters.generic import GenericJSONAdapter
    from osprey.services.ariel_search.ingestion.adapters.jlab import JLabLogbookAdapter
    from osprey.services.ariel_search.ingestion.adapters.ornl import ORNLLogbookAdapter

    _ingestion_adapters = {
        "als_logbook": (
            ALSLogbookAdapter,
            ArielIngestionAdapterRegistration(
                name="als_logbook",
                module_path="osprey.services.ariel_search.ingestion.adapters.als",
                class_name="ALSLogbookAdapter",
                description="ALS eLog adapter",
            ),
        ),
        "jlab_logbook": (
            JLabLogbookAdapter,
            ArielIngestionAdapterRegistration(
                name="jlab_logbook",
                module_path="osprey.services.ariel_search.ingestion.adapters.jlab",
                class_name="JLabLogbookAdapter",
                description="JLab logbook adapter",
            ),
        ),
        "ornl_logbook": (
            ORNLLogbookAdapter,
            ArielIngestionAdapterRegistration(
                name="ornl_logbook",
                module_path="osprey.services.ariel_search.ingestion.adapters.ornl",
                class_name="ORNLLogbookAdapter",
                description="ORNL logbook adapter",
            ),
        ),
        "generic_json": (
            GenericJSONAdapter,
            ArielIngestionAdapterRegistration(
                name="generic_json",
                module_path="osprey.services.ariel_search.ingestion.adapters.generic",
                class_name="GenericJSONAdapter",
                description="Generic JSON adapter",
            ),
        ),
    }
    registry.list_ariel_ingestion_adapters.return_value = list(_ingestion_adapters)
    registry.get_ariel_ingestion_adapter.side_effect = _ingestion_adapters.get

    return registry


@pytest.fixture(autouse=True)
def _mock_ariel_registry():
    """Provide a mock Osprey registry with ARIEL modules for all ARIEL tests."""
    registry = _build_ariel_mock_registry()
    with patch("osprey.registry.get_registry", return_value=registry):
        yield


# Dev database URL (from docker/ariel-dev.yml)
DEV_DATABASE_URL = "postgresql://ariel:ariel@localhost:5433/ariel_test"


def is_dev_database_available() -> bool:
    """Check if the docker-compose dev database is running.

    Returns:
        True if ariel-dev-db container is running and accessible.
    """
    try:
        import psycopg

        # Try to connect to the dev database
        with psycopg.connect(
            DEV_DATABASE_URL.replace("/ariel_test", "/ariel"), autocommit=True
        ) as conn:
            # Create test database if it doesn't exist
            conn.execute("SELECT 1")
            try:
                conn.execute("CREATE DATABASE ariel_test")
                logger.info("Created ariel_test database")
            except psycopg.errors.DuplicateDatabase:
                pass  # Already exists
        return True
    except Exception as e:
        logger.debug(f"Dev database not available: {e}")
        return False


def is_docker_available() -> bool:
    """Check if Docker is available for testcontainers.

    Returns:
        True if Docker daemon is running and accessible, False otherwise.
    """
    try:
        import docker

        client = docker.from_env()
        client.ping()
        logger.info("Docker is available for integration tests")
        return True
    except Exception as e:
        logger.warning(f"Docker not available: {e}")
        return False


# ============================================================================
# Integration test fixtures (require Docker)
# ============================================================================


@pytest.fixture(scope="session")
def database_url() -> str:
    """Get database connection URL for tests.

    Tries in order:
    1. ARIEL_TEST_DATABASE_URL environment variable
    2. Existing docker-compose dev database (ariel-dev-db on port 5433)
    3. Testcontainers (spins up fresh container)

    Returns:
        PostgreSQL connection URL

    Skips:
        If no database is available
    """
    # 1. Check for explicit env var
    env_url = os.environ.get("ARIEL_TEST_DATABASE_URL")
    if env_url:
        logger.info(f"Using database from ARIEL_TEST_DATABASE_URL: {env_url.split('@')[-1]}")
        return env_url

    # 2. Try existing docker-compose dev database (fastest)
    if is_dev_database_available():
        logger.info("Using existing docker-compose dev database (ariel-dev-db)")
        return DEV_DATABASE_URL

    # 3. Fall back to testcontainers
    if not is_docker_available():
        pytest.skip(
            "No database available - either start docker-compose "
            "(docker compose -f docker/ariel-dev.yml up -d) or install Docker"
        )

    logger.info("Starting testcontainers PostgreSQL (dev database not running)")
    from testcontainers.postgres import PostgresContainer

    # Use pgvector image for vector search support
    container = PostgresContainer(
        image="ankane/pgvector:latest",
        username="ariel",
        password="ariel",
        dbname="ariel_test",
    )
    container.start()

    # Register cleanup
    import atexit

    atexit.register(container.stop)

    url = container.get_connection_url()
    # Convert psycopg2 format to psycopg (v3) format
    if url.startswith("postgresql+psycopg2://"):
        url = url.replace("postgresql+psycopg2://", "postgresql://")

    logger.info(f"Testcontainer started: {url.split('@')[-1]}")
    return url


@pytest.fixture(scope="session")
def integration_ariel_config(database_url: str) -> ARIELConfig:
    """Create ARIELConfig pointing to test database.

    This is a session-scoped config for integration tests.

    Args:
        database_url: Database connection URL from container

    Returns:
        ARIELConfig configured for test database
    """
    from osprey.services.ariel_search.config import ARIELConfig

    return ARIELConfig.from_dict(
        {
            "database": {"uri": database_url},
            "search_modules": {
                "keyword": {"enabled": True},
                "semantic": {"enabled": True, "model": "nomic-embed-text"},
                "rag": {"enabled": False},
            },
            "enhancement_modules": {
                "text_embedding": {
                    "enabled": True,
                    "models": [{"name": "nomic-embed-text", "dimension": 768}],
                },
                "semantic_processor": {"enabled": True},
            },
        }
    )


# Track if migrations have been applied in this session
_migrations_applied: bool = False


@pytest.fixture
async def connection_pool(database_url: str):
    """Create async connection pool to test database.

    Args:
        database_url: Database connection URL

    Yields:
        AsyncConnectionPool to test database
    """
    from osprey.services.ariel_search.config import DatabaseConfig
    from osprey.services.ariel_search.database import create_connection_pool

    config = DatabaseConfig(uri=database_url)
    pool = await create_connection_pool(config)
    yield pool
    await pool.close()


@pytest.fixture
async def migrated_pool(connection_pool, integration_ariel_config: ARIELConfig):
    """Connection pool with migrations applied.

    Migrations only run once per session (idempotent).

    Args:
        connection_pool: Async connection pool
        integration_ariel_config: ARIEL configuration

    Returns:
        Connection pool with schema migrations applied
    """
    global _migrations_applied
    from osprey.services.ariel_search.database import run_migrations

    if not _migrations_applied:
        await run_migrations(connection_pool, integration_ariel_config)
        _migrations_applied = True
        logger.info("Migrations applied (first test in session)")

    return connection_pool


@pytest.fixture
async def repository(migrated_pool, integration_ariel_config: ARIELConfig) -> ARIELRepository:
    """ARIELRepository with real database connection.

    Function-scoped for test isolation, but uses shared pool.

    Args:
        migrated_pool: Connection pool with migrations applied
        integration_ariel_config: ARIEL configuration

    Returns:
        ARIELRepository connected to test database
    """
    from osprey.services.ariel_search.database import ARIELRepository

    return ARIELRepository(migrated_pool, integration_ariel_config)


# ============================================================================
# Unit test fixtures (no Docker required)
# ============================================================================


@pytest.fixture
def mock_ariel_config() -> ARIELConfig:
    """Create ARIELConfig for unit tests (mocked database).

    Returns:
        ARIELConfig with mock database URI
    """
    from osprey.services.ariel_search.config import ARIELConfig, DatabaseConfig

    return ARIELConfig(database=DatabaseConfig(uri="postgresql://mock/test"))


@pytest.fixture
def mock_repository() -> MagicMock:
    """Mocked ARIELRepository for unit tests.

    Returns:
        MagicMock with common repository methods stubbed
    """
    from osprey.services.ariel_search.config import ARIELConfig, DatabaseConfig

    config = ARIELConfig(database=DatabaseConfig(uri="postgresql://mock/test"))
    repo = MagicMock()
    repo.config = config
    repo.get_entry = AsyncMock(return_value=None)
    repo.upsert_entry = AsyncMock()
    repo.keyword_search = AsyncMock(return_value=[])
    repo.semantic_search = AsyncMock(return_value=[])
    repo.search_by_time_range = AsyncMock(return_value=[])
    repo.get_entries_by_ids = AsyncMock(return_value=[])
    repo.get_enhancement_stats = AsyncMock(return_value={"total_entries": 0})
    repo.count_entries = AsyncMock(return_value=0)
    repo.health_check = AsyncMock(return_value=(True, "OK"))
    repo.mark_enhancement_complete = AsyncMock()
    repo.mark_enhancement_failed = AsyncMock()
    return repo


@pytest.fixture
def mock_search_service() -> MagicMock:
    """Mocked ARIELSearchService for capability tests.

    Returns:
        MagicMock with common service methods stubbed
    """
    from osprey.services.ariel_search.models import ARIELSearchResult

    service = MagicMock()
    service.health_check = AsyncMock(return_value=(True, "OK"))
    service.search = AsyncMock(
        return_value=ARIELSearchResult(
            entries=[],
            total_count=0,
            search_mode="keyword",
            query="test",
        )
    )
    return service


@pytest.fixture
def seed_entry_factory():
    """Factory for creating test EnhancedLogbookEntry instances.

    Returns:
        Callable that creates EnhancedLogbookEntry with customizable fields
    """
    from datetime import datetime

    from osprey.services.ariel_search.models import EnhancedLogbookEntry

    def _create_entry(
        entry_id: str = "test-001",
        source_system: str = "test",
        timestamp: datetime | None = None,
        author: str = "test_user",
        raw_text: str = "Test entry content",
        attachments: list | None = None,
        metadata: dict | None = None,
        enhancement_status: dict | None = None,
    ) -> EnhancedLogbookEntry:
        return {
            "entry_id": entry_id,
            "source_system": source_system,
            "timestamp": timestamp or datetime.now(UTC),
            "author": author,
            "raw_text": raw_text,
            "attachments": attachments or [],
            "metadata": metadata or {},
            "enhancement_status": enhancement_status or {},
        }

    return _create_entry
