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
import warnings
from datetime import UTC
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.services.ariel_search.enhancement.qmd_export.exporter import (
    QmdExportModule,
)
from osprey.services.ariel_search.enhancement.semantic_processor.processor import (
    SemanticProcessorModule,
)
from osprey.services.ariel_search.enhancement.text_embedding.embedder import (
    TextEmbeddingModule,
)
from tests._container_support import is_docker_available, start_or_skip, stop_quietly

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
    from osprey.services.ariel_search.search import qmd as qmd_real
    from osprey.services.ariel_search.search import semantic as sem_real

    kw_mod = types.ModuleType("keyword")
    kw_mod.get_tool_descriptor = kw_real.get_tool_descriptor  # type: ignore[attr-defined]
    kw_mod.get_parameter_descriptors = getattr(kw_real, "get_parameter_descriptors", None)  # type: ignore[attr-defined]

    sem_mod = types.ModuleType("semantic")
    sem_mod.get_tool_descriptor = sem_real.get_tool_descriptor  # type: ignore[attr-defined]
    sem_mod.get_parameter_descriptors = getattr(sem_real, "get_parameter_descriptors", None)  # type: ignore[attr-defined]

    hybrid_mod = types.ModuleType("hybrid")
    hybrid_mod.get_tool_descriptor = qmd_real.get_tool_descriptor  # type: ignore[attr-defined]
    hybrid_mod.get_parameter_descriptors = getattr(qmd_real, "get_parameter_descriptors", None)  # type: ignore[attr-defined]

    _search_modules = {"keyword": kw_mod, "semantic": sem_mod, "hybrid": hybrid_mod}
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
    qe_reg = ArielEnhancementModuleRegistration(
        name="qmd_export",
        module_path="osprey.services.ariel_search.enhancement.qmd_export.exporter",
        class_name="QmdExportModule",
        description="qmd markdown mirror export",
        execution_order=30,
    )
    _enhancement_modules = {
        "semantic_processor": (SemanticProcessorModule, sp_reg),
        "text_embedding": (TextEmbeddingModule, te_reg),
        "qmd_export": (QmdExportModule, qe_reg),
    }
    registry.list_ariel_enhancement_modules.return_value = [
        "semantic_processor",
        "text_embedding",
        "qmd_export",
    ]
    registry.get_ariel_enhancement_module.side_effect = _enhancement_modules.get

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


@pytest.fixture(autouse=True)
def _reset_ariel_service_singleton():
    """Reset the ARIEL service singleton around every test in this package.

    The leak: ``capability._ariel_service_instance`` is a module-global
    ``ARIELSearchService`` holding an open connection pool. A test that builds
    one and does not close it leaves both the instance and its pool live for
    whatever runs next on the same worker, so the next caller of
    ``get_ariel_search_service()`` silently gets the previous test's service.

    Reset happens *before* yield so a test never inherits a stale instance. At
    teardown a still-live instance is reported with a ``ResourceWarning`` naming
    ``close_ariel_service()`` before being dropped -- resetting a live service
    silently would leak its pool with no trace. Tests that legitimately build a
    real service (``integration/test_capability.py``) close it in their own
    fixture; autouse fixtures finalize last, so that cleanup runs first and no
    spurious warning fires.
    """
    from osprey.services.ariel_search import capability

    capability.reset_ariel_service()
    yield
    if capability._ariel_service_instance is not None:
        warnings.warn(
            "ariel service singleton left live; use close_ariel_service()",
            ResourceWarning,
            stacklevel=1,
        )
        capability.reset_ariel_service()


# Dev database URLs - try port 5432 (ariel-postgres), then 5433 (ariel-dev-db)
DEV_DATABASE_URL_5432 = "postgresql://ariel:ariel@localhost:5432/ariel_test"
DEV_DATABASE_URL_5433 = "postgresql://ariel:ariel@localhost:5433/ariel_test"


def is_dev_database_available() -> tuple[bool, str]:
    """Check if a dev database is running (tries 5432 then 5433).

    Returns:
        Tuple of (available, url).
    """
    try:
        import psycopg
    except ImportError:
        logger.debug("psycopg not available (libpq missing?), skipping dev database check")
        return False, ""

    for dev_url in [DEV_DATABASE_URL_5432, DEV_DATABASE_URL_5433]:
        try:
            base_url = dev_url.replace("/ariel_test", "/ariel")
            with psycopg.connect(base_url, autocommit=True) as conn:
                conn.execute("SELECT 1")
                try:
                    conn.execute("CREATE DATABASE ariel_test")
                    logger.info("Created ariel_test database")
                except psycopg.errors.DuplicateDatabase:
                    pass  # Already exists
            return True, dev_url
        except Exception as e:
            logger.debug(f"Dev database at {dev_url} not available: {e}")
            continue
    return False, ""


# ============================================================================
# Integration test fixtures (require Docker)
# ============================================================================


@pytest.fixture(scope="session")
def database_url(request: pytest.FixtureRequest) -> str:
    """Get database connection URL for tests.

    Tries in order:
    1. ARIEL_TEST_DATABASE_URL environment variable
    2. Existing docker-compose dev database (ariel-dev-db on port 5433)
    3. Testcontainers (spins up fresh container)

    Deliberately a plain (non-generator) fixture: only the third exit owns a
    container, while the first two return a URL to a database this fixture did
    not create. A ``yield`` would make *all three* paths generator paths, so the
    two resource-free exits would raise "did not yield a value". Container
    teardown is registered with ``request.addfinalizer`` instead, which runs at
    session teardown exactly as a ``finally`` would.

    Args:
        request: Fixture request, used to register container teardown.

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
    available, dev_url = is_dev_database_available()
    if available:
        logger.info(f"Using existing dev database: {dev_url.split('@')[-1]}")
        return dev_url

    # 3. Fall back to testcontainers
    if not is_docker_available():
        pytest.skip(
            "No database available - either start docker-compose "
            "(docker compose -f docker/ariel-dev.yml up -d) or install Docker"
        )

    logger.info("Starting testcontainers PostgreSQL (dev database not running)")
    try:
        from testcontainers.postgres import PostgresContainer
    except ImportError:
        pytest.skip("testcontainers[postgres] not installed")

    # Use pgvector image for vector search support
    container = start_or_skip(
        lambda: PostgresContainer(
            image="ankane/pgvector:latest",
            username="ariel",
            password="ariel",
            dbname="ariel_test",
        ),
        label="postgres",
    )

    request.addfinalizer(lambda: stop_quietly(container))

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


# Track if migrations have been applied in this session.
# Intentionally session-scoped with no reset seam: migrations are idempotent and
# every test in the docker group shares the one `ariel_test` database, so a
# per-test reset would buy nothing and re-run the full migration set once per
# test on the docker worker.
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
#
# Database fakes. The package acquires results in three different shapes, and
# the fakes below serve all three from one shared call log:
#
#   (a) result = await conn.execute(sql, params)      -> result.fetchone/fetchall
#   (b) async with conn.cursor(row_factory=dict_row)  -> dict rows
#   (c) async with conn.cursor()                      -> positional tuple rows
#
# Tests script the results and then assert on the recorded ``(sql, params)``
# sequence; nothing here talks to Postgres.


def _normalize_sql(sql: str) -> str:
    """Collapse SQL whitespace so multi-line literals match single-line patterns."""
    return " ".join(str(sql).split())


class _SQLRecorder:
    """Canned-result script plus the ordered ``(sql, params)`` call log.

    A pool, its connection and every cursor handed out from it share one
    recorder, so ``recorder.calls`` is the whole conversation with the database
    in order, whichever acquisition shape the code under test used.

    Result lookup per ``execute``, in order:

    1. ``rows_for`` -- the first pattern found in the (whitespace-normalized)
       SQL wins. Not consumed, so a repeated query keeps returning it.
    2. ``results`` -- a queue popped from the left, for scripts where the same
       SQL must return different rows on successive calls. This queue is popped
       by *every* ``execute``, including bookkeeping statements the code under
       test issues around the query you care about (``BEGIN READ ONLY``,
       ``SET LOCAL``, ``ROLLBACK``, ...), so a positional ``results`` script is
       easy to misalign. Prefer ``rows_for`` whenever the code under test issues
       such bookkeeping SQL (e.g. ``search/sql_query.py``).
    3. ``[]``.

    A scripted result that is an ``Exception`` instance is raised rather than
    returned; that is how a failure at one specific query is injected.
    """

    def __init__(
        self,
        results: list[Any] | None = None,
        rows_for: dict[str, Any] | None = None,
    ) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.results: list[Any] = list(results or [])
        self.rows_for: dict[str, Any] = dict(rows_for or {})

    @property
    def sql(self) -> list[str]:
        """SQL text of every recorded call, in order."""
        return [sql for sql, _ in self.calls]

    def matching(self, pattern: str) -> list[tuple[str, Any]]:
        """Recorded calls whose SQL contains `pattern`, ignoring whitespace."""
        needle = _normalize_sql(pattern)
        return [call for call in self.calls if needle in _normalize_sql(call[0])]

    def record(self, sql: str, params: Any = None) -> list[Any]:
        """Log one execute and return (or raise) its scripted result."""
        self.calls.append((sql, params))
        rows = self._next_rows(sql)
        if isinstance(rows, Exception):
            raise rows
        return list(rows)

    def _next_rows(self, sql: str) -> Any:
        normalized = _normalize_sql(sql)
        for pattern, rows in self.rows_for.items():
            if _normalize_sql(pattern) in normalized:
                return rows
        if self.results:
            return self.results.pop(0)
        return []


class _FakeCursor:
    """Cursor stand-in serving both ``conn.cursor()`` shapes and ``conn.execute()``.

    Usable as an async context manager (``async with conn.cursor(...) as cur``)
    and as the plain handle returned by ``await conn.execute(...)``. Rows come
    back exactly as scripted -- supply dicts for a ``row_factory=dict_row``
    cursor and tuples for a bare one; the requested factory is kept on
    ``row_factory`` for tests that assert which shape was asked for.

    Fetching is non-consuming: ``fetchone()`` always returns ``rows[0]`` and
    ``fetchmany(n)`` always returns ``rows[:n]``, with no cursor advance
    between calls. A drain loop written as
    ``while (row := await cur.fetchone()): ...`` will spin forever against
    this fake -- iterate ``fetchall()`` instead.
    """

    def __init__(self, recorder: _SQLRecorder, row_factory: Any = None) -> None:
        self.recorder = recorder
        self.row_factory = row_factory
        self.rows: list[Any] = []

    async def __aenter__(self) -> _FakeCursor:
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return False

    async def execute(self, sql: str, params: Any = None) -> _FakeCursor:
        self.rows = self.recorder.record(sql, params)
        return self

    async def fetchone(self) -> Any:
        return self.rows[0] if self.rows else None

    async def fetchall(self) -> list[Any]:
        return list(self.rows)

    async def fetchmany(self, size: int | None = None) -> list[Any]:
        return list(self.rows) if size is None else list(self.rows[:size])


class _FakeConnection:
    """Connection stand-in supporting every acquisition shape in the package.

    Doubles as the object yielded by :meth:`_FakePool.connection` and as the
    recording DDL connection that migration ``up``/``down`` bodies write
    through -- pass one straight to a migration and read ``conn.sql`` for the
    ordered DDL text.

    Args:
        recorder: Share the pool's recorder. Omit for a standalone connection,
            which builds its own from `results`/`rows_for`.
        error: Raised on entering ``async with pool.connection()``.
    """

    def __init__(
        self,
        recorder: _SQLRecorder | None = None,
        results: list[Any] | None = None,
        rows_for: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.recorder = recorder if recorder is not None else _SQLRecorder(results, rows_for)
        self.error = error
        self.cursors: list[_FakeCursor] = []

    @property
    def calls(self) -> list[tuple[str, Any]]:
        """Ordered ``(sql, params)`` of everything executed on this connection."""
        return self.recorder.calls

    @property
    def sql(self) -> list[str]:
        """SQL text of every call on this connection, in order."""
        return self.recorder.sql

    async def __aenter__(self) -> _FakeConnection:
        if self.error is not None:
            raise self.error
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return False

    def cursor(self, row_factory: Any = None) -> _FakeCursor:
        """Hand out a cursor; ``row_factory`` is recorded, never applied."""
        cur = _FakeCursor(self.recorder, row_factory=row_factory)
        self.cursors.append(cur)
        return cur

    async def execute(self, sql: str, params: Any = None) -> _FakeCursor:
        """Run one statement and return the cursor holding its result."""
        cur = self.cursor()
        await cur.execute(sql, params)
        return cur


class _FakePool:
    """Duck-typed stand-in for the psycopg ``AsyncConnectionPool``.

    ``async with pool.connection() as conn`` yields one shared
    :class:`_FakeConnection`, so ``pool.calls`` / ``pool.sql`` stay the ordered
    record across every block the code under test opens. ``await pool.close()``
    is recorded on ``closed`` and ``close_calls``.

    Args:
        results: Result queue, consumed one entry per ``execute``.
        rows_for: SQL-substring -> rows, matched before the queue.
        error: The raising variant for error-wrap tests. Entering the
            connection block raises it, so every repository method funnels
            into its ``except`` branch.
    """

    def __init__(
        self,
        results: list[Any] | None = None,
        rows_for: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.recorder = _SQLRecorder(results, rows_for)
        self.conn = _FakeConnection(self.recorder, error=error)
        self.close_calls = 0

    @property
    def calls(self) -> list[tuple[str, Any]]:
        """Ordered ``(sql, params)`` of everything executed through this pool."""
        return self.recorder.calls

    @property
    def sql(self) -> list[str]:
        """SQL text of every call through this pool, in order."""
        return self.recorder.sql

    @property
    def closed(self) -> bool:
        """True once ``close()`` has been awaited at least once."""
        return self.close_calls > 0

    def matching(self, pattern: str) -> list[tuple[str, Any]]:
        """Recorded calls whose SQL contains `pattern`, ignoring whitespace."""
        return self.recorder.matching(pattern)

    def connection(self) -> _FakeConnection:
        return self.conn

    async def close(self) -> None:
        self.close_calls += 1


class _FakeEmbeddingProvider:
    """Stand-in for the ``BaseEmbeddingProvider`` that ``get_embedding_provider`` returns.

    ``execute_embedding`` is synchronous, matching the real provider: ARIEL
    calls it from async code without awaiting. It returns one `dimension`-long
    vector per text (or the scripted `vectors`) and appends every call to
    ``calls``; pass `error` to make it raise, which is how the
    embedding-failure branches are reached.
    """

    name = "fake"

    def __init__(
        self,
        dimension: int = 4,
        default_base_url: str = "http://fake-embeddings.invalid:11434",
        default_model_id: str = "nomic-embed-text",
        vectors: list[list[float]] | None = None,
        error: Exception | None = None,
        healthy: tuple[bool, str] = (True, "ok"),
    ) -> None:
        self.dimension = dimension
        self.default_base_url = default_base_url
        self.default_model_id = default_model_id
        self.vectors = vectors
        self.error = error
        self.healthy = healthy
        self.calls: list[dict[str, Any]] = []

    def execute_embedding(
        self,
        texts: list[str],
        model_id: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        self.calls.append(
            {
                "texts": list(texts),
                "model_id": model_id,
                "api_key": api_key,
                "base_url": base_url,
                **kwargs,
            }
        )
        if self.error is not None:
            raise self.error
        if self.vectors is not None:
            return [list(vector) for vector in self.vectors]
        return [[0.1] * self.dimension for _ in texts]

    def check_health(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_id: str | None = None,
        timeout: float = 10.0,
    ) -> tuple[bool, str]:
        self.calls.append({"check_health": True, "api_key": api_key, "base_url": base_url})
        return self.healthy


@pytest.fixture
def fake_pool_factory():
    """Factory for scripted :class:`_FakePool` instances.

    Returns:
        Callable ``(results=None, rows_for=None, error=None) -> _FakePool``.
        Pass ``error=`` for the raising variant used by error-wrap tests.
    """

    def _make(
        results: list[Any] | None = None,
        rows_for: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> _FakePool:
        return _FakePool(results=results, rows_for=rows_for, error=error)

    return _make


@pytest.fixture
def fake_pool(fake_pool_factory) -> _FakePool:
    """Empty-script :class:`_FakePool`; every query returns no rows.

    Script it after the fact via ``fake_pool.recorder.results`` /
    ``fake_pool.recorder.rows_for`` when the shape is only known mid-test.
    """
    return fake_pool_factory()


@pytest.fixture
def ddl_conn() -> _FakeConnection:
    """Recording connection for migration ``up``/``down`` bodies.

    Migrations take a ``conn`` parameter, so they can be driven directly:
    ``await migration.up(ddl_conn)``, then assert on ``ddl_conn.sql``.
    """
    return _FakeConnection()


@pytest.fixture
def fake_embedding_provider() -> _FakeEmbeddingProvider:
    """Default :class:`_FakeEmbeddingProvider` (4-dimensional vectors)."""
    return _FakeEmbeddingProvider()


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

    ``repo.pool`` is a real :class:`_FakePool`, so code that opens
    ``async with repo.pool.connection() as conn`` works and its SQL lands in
    ``repo.pool.calls``. Script it through ``repo.pool.recorder``.

    Returns:
        MagicMock with common repository methods stubbed
    """
    from osprey.services.ariel_search.config import ARIELConfig, DatabaseConfig

    config = ARIELConfig(database=DatabaseConfig(uri="postgresql://mock/test"))
    repo = MagicMock()
    repo.config = config
    repo.pool = _FakePool()
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
    repo.start_ingestion_run = AsyncMock(return_value=1)
    repo.complete_ingestion_run = AsyncMock()
    repo.fail_ingestion_run = AsyncMock()
    repo.get_last_successful_run = AsyncMock(return_value=None)
    return repo


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
