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

from osprey.services.ariel_search.enhancement.image_caption.module import (
    ImageCaptionModule,
)
from osprey.services.ariel_search.enhancement.image_embedding.module import (
    ImageEmbeddingModule,
)
from osprey.services.ariel_search.enhancement.qmd_export.exporter import (
    QmdExportModule,
)
from osprey.services.ariel_search.enhancement.semantic_processor.processor import (
    SemanticProcessorModule,
)
from osprey.services.ariel_search.enhancement.text_embedding.embedder import (
    TextEmbeddingModule,
)
from tests import _litellm_callbacks
from tests._container_support import is_docker_available, start_or_skip, stop_quietly
from tests.services.ariel_search.fake_providers import make_fake_embedding_provider

if TYPE_CHECKING:
    from collections.abc import Iterator
    from concurrent.futures import ThreadPoolExecutor

    from osprey.models.providers.base import BaseProvider
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
    ic_reg = ArielEnhancementModuleRegistration(
        name="image_caption",
        module_path="osprey.services.ariel_search.enhancement.image_caption.module",
        class_name="ImageCaptionModule",
        description="Image caption",
        execution_order=5,
    )
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
    ie_reg = ArielEnhancementModuleRegistration(
        name="image_embedding",
        module_path="osprey.services.ariel_search.enhancement.image_embedding.module",
        class_name="ImageEmbeddingModule",
        description="Image embedding",
        execution_order=25,
    )
    qe_reg = ArielEnhancementModuleRegistration(
        name="qmd_export",
        module_path="osprey.services.ariel_search.enhancement.qmd_export.exporter",
        class_name="QmdExportModule",
        description="qmd markdown mirror export",
        execution_order=30,
    )
    _enhancement_modules = {
        "image_caption": (ImageCaptionModule, ic_reg),
        "semantic_processor": (SemanticProcessorModule, sp_reg),
        "text_embedding": (TextEmbeddingModule, te_reg),
        "image_embedding": (ImageEmbeddingModule, ie_reg),
        "qmd_export": (QmdExportModule, qe_reg),
    }
    registry.list_ariel_enhancement_modules.return_value = [
        "image_caption",
        "semantic_processor",
        "text_embedding",
        "image_embedding",
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


def reset_image_lane_state() -> None:
    """Forget the picture lane's breaker, reason, verdict and settings, and what they rest on.

    Clears the lane's own state, the shared local-server resolution cache and
    the availability tracker, so no test sees a breaker or a reason another
    test on the same worker left behind.
    """
    from osprey.models.providers import _local_server
    from osprey.services.ariel_search.enhancement import availability
    from osprey.services.ariel_search.search import image_lane

    image_lane._reset_state()
    _local_server.reset_cache()
    availability.reset_availability()


@pytest.fixture(autouse=True)
def _reset_image_lane():
    """Every ARIEL service test starts and ends with a closed, reason-free picture lane."""
    reset_image_lane_state()
    yield
    reset_image_lane_state()


@pytest.fixture
def llama_stub(monkeypatch):
    """Start :class:`~tests.services.ariel_search.llama_stub.LlamaStub` servers on 127.0.0.1.

    Calling the fixture starts one stub and returns it; every stub is stopped at
    teardown. ``LLAMA_CPP_HOST`` is removed and the container fallbacks emptied,
    so the adapter's reachability walk only ever probes the stub's own URL, and
    the shared local-server cache is cleared before and after.
    """
    from osprey.models.providers import _local_server
    from tests.services.ariel_search.llama_stub import LlamaStub

    monkeypatch.delenv("LLAMA_CPP_HOST", raising=False)
    monkeypatch.setattr(_local_server, "container_fallback_urls", lambda base_url, default_port: [])
    _local_server.reset_cache()
    started: list[LlamaStub] = []

    def _start() -> LlamaStub:
        stub = LlamaStub()
        started.append(stub)
        return stub

    yield _start
    for stub in started:
        stub.stop()
    _local_server.reset_cache()


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


def database_required() -> bool:
    """Whether ``ARIEL_TEST_REQUIRE_DB=1`` asks for a missing database to fail.

    A gate that runs an integration file sets it, so the gate cannot pass with
    every database test skipped.
    """
    return os.environ.get("ARIEL_TEST_REQUIRE_DB") == "1"


def skip_or_fail(reason: str) -> None:
    """Skip for want of a database -- or fail, when one is required."""
    if database_required():
        pytest.fail(f"{reason} (ARIEL_TEST_REQUIRE_DB=1)")
    pytest.skip(reason)


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
        If no database is available -- or fails instead under
        ``ARIEL_TEST_REQUIRE_DB=1``.
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
        skip_or_fail(
            "No database available - either start docker-compose "
            "(docker compose -f docker/ariel-dev.yml up -d) or install Docker"
        )

    logger.info("Starting testcontainers PostgreSQL (dev database not running)")
    try:
        from testcontainers.postgres import PostgresContainer
    except ImportError:
        skip_or_fail("testcontainers[postgres] not installed")

    # Use pgvector image for vector search support
    try:
        container = start_or_skip(
            lambda: PostgresContainer(
                image="pgvector/pgvector:pg16",
                username="ariel",
                password="ariel",
                dbname="ariel_test",
            ),
            label="postgres",
        )
    except pytest.skip.Exception as skipped:
        if database_required():
            pytest.fail(f"{skipped.msg} (ARIEL_TEST_REQUIRE_DB=1)")
        raise

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


class _FakeTransaction:
    """``conn.transaction()`` stand-in that records how the block ENDED.

    Constructing the object is not entering it, so the outcome is appended from
    ``__aexit__``: ``COMMIT`` when the block left cleanly, ``ROLLBACK`` when an
    exception carried it out. That distinction is the whole point of the double
    -- the migration runner's contract is that a failing ``up()`` leaves nothing
    behind, and only the rollback record proves it.

    Exceptions propagate (``__aexit__`` returns False), matching psycopg: the
    transaction undoes its work and the error still reaches the caller.
    """

    def __init__(self, log: list[str]) -> None:
        self._log = log

    async def __aenter__(self) -> _FakeTransaction:
        self._log.append("BEGIN")
        return self

    async def __aexit__(self, exc_type: object, *_rest: object) -> bool:
        self._log.append("ROLLBACK" if exc_type is not None else "COMMIT")
        return False


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
        self.transactions: list[str] = []

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

    def transaction(self) -> _FakeTransaction:
        """Hand out a transaction block that records COMMIT vs ROLLBACK.

        Read ``conn.transactions`` for the ordered outcomes.
        """
        return _FakeTransaction(self.transactions)

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
        self.connection_timeouts: list[float | None] = []

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

    def connection(self, timeout: float | None = None) -> _FakeConnection:
        self.connection_timeouts.append(timeout)
        return self.conn

    async def close(self) -> None:
        self.close_calls += 1


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
def fake_embedding_provider() -> BaseProvider:
    """An instance of a fresh ``make_fake_embedding_provider()`` class (4-d vectors)."""
    return make_fake_embedding_provider()()


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
    from osprey.services.ariel_search.database.repository import SchemaFacts

    config = ARIELConfig(database=DatabaseConfig(uri="postgresql://mock/test"))
    repo = MagicMock()
    repo.config = config
    repo.pool = _FakePool()
    # A store without the copy state: ``ingest_one`` takes its plain-upsert
    # branch, so ingest callers' counts match ``upsert_entry`` calls one to one.
    repo.schema_facts = AsyncMock(return_value=SchemaFacts(has_v2_fts=False, has_copy_state=False))
    repo.get_copy_retry_candidates = AsyncMock(return_value=[])
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
    repo.mark_enhancement_failed = AsyncMock(return_value=1)
    repo.start_ingestion_run = AsyncMock(return_value=1)
    repo.complete_ingestion_run = AsyncMock()
    repo.fail_ingestion_run = AsyncMock()
    repo.get_last_successful_run = AsyncMock(return_value=None)
    repo.get_last_ingestion = AsyncMock(return_value=None)
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


@pytest.fixture
def litellm_callback_pool() -> Iterator[ThreadPoolExecutor]:
    """Run litellm's success handlers on a pool this test owns, shut down after it.

    Requested by name by every test or fixture that reaches a live model, so the
    scope is open before the first call is made.
    """
    with _litellm_callbacks.litellm_callback_pool() as pool:
        yield pool


# -- attachment fetch fake ------------------------------------------------------


class AttachmentFetchFake:
    """Stands in for ``fetch_attachment_bytes`` at every consumer's reference.

    The copy path classifies and absorbs fetch failures, so a fetch a test did
    not expect would otherwise pass silently. Every call is recorded; a test
    that has not opted in by calling :meth:`respond` has its calls collected in
    :attr:`unopted`, answered with a transient outcome, and fails at fixture
    teardown.

    Attributes:
        calls: Every call, as a dict of the arguments it was given.
        unopted: The calls made before the test opted in.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.unopted: list[dict[str, Any]] = []
        self._handler: Any = None

    def respond(self, answer: Any) -> None:
        """Opt in: answer every later call with ``answer``.

        Args:
            answer: A ``FetchOutcome`` returned as is, or a callable taking the
                fetcher's arguments and returning one (sync or async).
        """
        self._handler = answer

    async def __call__(
        self,
        url: str,
        cap: int,
        origins: frozenset,
        adapter: Any,
        method: str = "GET",
        **kwargs: Any,
    ) -> Any:
        import inspect

        from osprey.services.ariel_search.attachments.fetch import FetchOutcome

        call = {
            "url": url,
            "cap": cap,
            "origins": origins,
            "adapter": adapter,
            "method": method,
            **kwargs,
        }
        self.calls.append(call)
        if self._handler is None:
            self.unopted.append(call)
            return FetchOutcome(transient=True)
        if isinstance(self._handler, FetchOutcome):
            return self._handler
        out = self._handler(url, cap, origins, adapter, method, **kwargs)
        return await out if inspect.isawaitable(out) else out

    def verify(self) -> None:
        """Fail the running test when any call came before it opted in."""
        if self.unopted:
            calls = [(c["method"], c["url"]) for c in self.unopted]
            pytest.fail(f"unopted attachment fetch: {calls}")


def install_attachment_fetch_fake(request: pytest.FixtureRequest, monkeypatch: Any) -> Any:
    """Patch the consumers' fetcher references with a fresh fake, or not at all.

    The consumers bind the fetcher at module level, so the fake replaces the
    ``attachments.copy`` and ``ingestion.metadata_attachment`` attributes and
    leaves ``fetch.fetch_attachment_bytes`` itself real. A test marked
    ``real_fetch`` gets no fake.

    Returns:
        The installed :class:`AttachmentFetchFake`, or ``None`` under ``real_fetch``.
    """
    if request.node.get_closest_marker("real_fetch"):
        return None
    from osprey.services.ariel_search.attachments import copy as copy_mod
    from osprey.services.ariel_search.ingestion import metadata_attachment as metadata_mod

    fake = AttachmentFetchFake()
    monkeypatch.setattr(copy_mod, "fetch_attachment_bytes", fake)
    monkeypatch.setattr(metadata_mod, "fetch_attachment_bytes", fake)
    return fake


@pytest.fixture(autouse=True)
def attachment_fetch(request, monkeypatch) -> Iterator[AttachmentFetchFake | None]:
    """No ARIEL test reaches a real attachment source unless it opts in.

    Yields the :class:`AttachmentFetchFake`; a test opts in to fetching by
    calling its ``respond``, or to the real fetcher with ``real_fetch``.
    """
    fake = install_attachment_fetch_fake(request, monkeypatch)
    yield fake
    if fake is not None:
        fake.verify()
