"""ARIEL Search Service.

This module provides the main ARIELSearchService class that orchestrates
search execution. A request names a search module (``"keyword"``,
``"semantic"``, ...); the service resolves that name against the registered
ARIEL search modules and invokes the module's own ``execute``. Adding a
search module therefore needs no change here.

Higher-level reasoning is handled by the Osprey agent layer.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from osprey.services.ariel_search.exceptions import (
    ARIELException,
    ConfigurationError,
    SearchExecutionError,
    SearchTimeoutError,
)
from osprey.services.ariel_search.models import (
    DEFAULT_SEARCH_MODE,
    ARIELSearchRequest,
    ARIELSearchResult,
    ARIELStatusResult,
    DiagnosticLevel,
    EmbeddingTableInfo,
    FacilityEntryCreateRequest,
    FacilityEntryCreateResult,
    SearchDiagnostic,
    SyncStatus,
)
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from psycopg_pool import AsyncConnectionPool

    from osprey.models.embeddings.base import BaseEmbeddingProvider
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.database.repository import ARIELRepository
    from osprey.services.ariel_search.search.base import SearchToolDescriptor

logger = get_logger("ariel")


class ARIELSearchService:
    """Main service class for ARIEL search functionality.

    Routes a request's mode name to the matching registered search module and
    calls that module's ``execute``. The registry is the only source of
    routable modes, so the service, the capabilities API, and the agent tools
    always agree on which modes exist.

    Higher-level reasoning is handled by the Osprey agent layer.

    Usage:
        config = ARIELConfig.from_dict(config_dict)
        async with create_ariel_service(config) as service:
            result = await service.search("What happened yesterday?")
    """

    def __init__(
        self,
        config: ARIELConfig,
        pool: AsyncConnectionPool,
        repository: ARIELRepository,
    ) -> None:
        """Initialize the service.

        Args:
            config: ARIEL configuration
            pool: Database connection pool
            repository: Database repository
        """
        self.config = config
        self.pool = pool
        self.repository = repository
        self._embedder: BaseEmbeddingProvider | None = None
        self._validated_search_model = False

    def _get_embedder(self) -> BaseEmbeddingProvider:
        """Lazy-load the embedding provider.

        Uses Osprey's provider configuration system to select the appropriate
        embedding provider based on config.embedding.provider.

        Returns:
            Configured embedding provider instance
        """
        if self._embedder is None:
            from osprey.models.embeddings import get_embedding_provider

            self._embedder = get_embedding_provider(self.config.embedding.provider)
        return self._embedder

    @staticmethod
    def _diagnostic_result(
        *,
        reasoning: str,
        level: DiagnosticLevel,
        source: str,
        category: str,
        message: str | None = None,
        modes: tuple[str, ...] = (),
    ) -> ARIELSearchResult:
        """Build an empty result carrying a single diagnostic.

        Shared shape for the non-result outcomes -- error, timeout, and
        graceful degradation -- which all return no entries, one diagnostic,
        and a human-readable ``reasoning``. ``message`` defaults to
        ``reasoning`` when the diagnostic text matches the caller-facing text.
        """
        return ARIELSearchResult(
            entries=(),
            answer=None,
            sources=(),
            search_modes_used=modes,
            reasoning=reasoning,
            diagnostics=(
                SearchDiagnostic(
                    level=level,
                    source=source,
                    message=reasoning if message is None else message,
                    category=category,
                ),
            ),
        )

    @staticmethod
    def _error_result(
        mode: str,
        source: str,
        error: Exception,
    ) -> ARIELSearchResult:
        return ARIELSearchService._diagnostic_result(
            reasoning=f"{mode.capitalize()} search failed: {error}",
            level=DiagnosticLevel.ERROR,
            source=source,
            category="search",
            modes=(mode,),
        )

    async def _validate_search_model(self) -> None:
        """Validate that the configured search model's table exists.

        Called lazily on first semantic search. If validation fails (e.g.,
        pgvector migration was skipped), disables semantic search at runtime
        so other search modes continue working.
        """
        if self._validated_search_model:
            return

        model = self.config.get_search_model()
        if model:
            try:
                await self.repository.validate_search_model_table(model)
            except ConfigurationError as e:
                logger.warning(
                    f"Semantic search disabled: embedding table not found ({e}). "
                    f"Install pgvector and re-run 'osprey ariel quickstart' to enable."
                )
                self.config.search_modules["semantic"].enabled = False

        self._validated_search_model = True

    async def search(
        self,
        query: str,
        *,
        max_results: int | None = None,
        time_range: tuple[Any, Any] | None = None,
        mode: str | None = None,
        advanced_params: dict[str, Any] | None = None,
    ) -> ARIELSearchResult:
        """Execute a search.

        This is the main entry point for searching the logbook.
        Routes to the appropriate execution mode.

        Args:
            query: Natural language query
            max_results: Maximum results (default: ``ARIELSearchRequest``'s)
            time_range: Optional (start, end) datetime tuple
            mode: Optional search module name (default: ``"keyword"``)
            advanced_params: Mode-specific advanced parameters from the frontend

        Returns:
            ARIELSearchResult with entries, answer, and sources
        """
        # Build the search request
        request = ARIELSearchRequest(
            query=query,
            max_results=(
                max_results if max_results is not None else ARIELSearchRequest.max_results
            ),
            time_range=time_range,
            modes=[mode] if mode else [DEFAULT_SEARCH_MODE],
            advanced_params=advanced_params or {},
        )

        return await self.ainvoke(request)

    async def ainvoke(
        self,
        request: ARIELSearchRequest,
    ) -> ARIELSearchResult:
        """Invoke ARIEL with a search request.

        Resolves the requested mode against the registered search modules and
        runs that module.

        Args:
            request: Search request with query and parameters

        Returns:
            ARIELSearchResult with entries, answer, and sources

        Raises:
            ConfigurationError: If the requested mode is not registered or is
                registered but disabled.
        """
        try:
            if self.config.is_search_module_enabled("semantic"):
                await self._validate_search_model()

            mode = request.modes[0] if request.modes else DEFAULT_SEARCH_MODE

            return await self._run_module(mode, request)

        except SearchTimeoutError as e:
            # Return graceful timeout result instead of propagating exception
            return self._diagnostic_result(
                reasoning=(
                    f"Search timed out before completion. "
                    f"{e.operation} timeout ({e.timeout_seconds}s) exceeded"
                ),
                level=DiagnosticLevel.ERROR,
                source="service.timeout",
                category="timeout",
                message=f"Search timed out: {e.operation} exceeded {e.timeout_seconds}s limit",
            )
        except ARIELException:
            raise
        except Exception as e:
            logger.exception(f"Search failed: {e}")
            mode = request.modes[0] if request.modes else DEFAULT_SEARCH_MODE
            raise SearchExecutionError(
                f"Search execution failed: {e}",
                search_mode=mode,
                query=request.query,
            ) from e

    @staticmethod
    def _registered_descriptors() -> dict[str, SearchToolDescriptor]:
        """Collect the tool descriptor of every registered search module.

        Uses the same registry listing as the capabilities API so the routable
        modes and the advertised modes can never drift apart.

        Returns:
            Mapping of module name to descriptor, in registry order. A module
            name is what a request's ``modes`` carries and what
            ``search_modules.<name>`` configures.
        """
        from osprey.registry import get_registry

        registry = get_registry()
        descriptors: dict[str, SearchToolDescriptor] = {}
        for name in registry.list_ariel_search_modules():
            module = registry.get_ariel_search_module(name)
            if module is None:
                continue
            descriptors[name] = module.get_tool_descriptor()
        return descriptors

    def _enabled_mode_names(self, descriptors: dict[str, SearchToolDescriptor]) -> str:
        """Render the modes a caller may actually ask for, for error messages."""
        enabled = [name for name in descriptors if self.config.is_search_module_enabled(name)]
        return ", ".join(enabled) if enabled else "(none enabled)"

    async def _run_module(self, mode: str, request: ARIELSearchRequest) -> ARIELSearchResult:
        """Run the registered search module named by ``mode``.

        Args:
            mode: Search module name, normalized to lowercase by the request.
            request: Search request

        Returns:
            ARIELSearchResult with matching entries, or a diagnostic-only
            result when the module failed or is gracefully unavailable.

        Raises:
            ConfigurationError: If ``mode`` names no registered module, or
                names one that is disabled in configuration.
        """
        descriptors = self._registered_descriptors()
        descriptor = descriptors.get(mode)

        if descriptor is None:
            raise ConfigurationError(
                f"Unknown search mode '{mode}'. "
                f"Available modes: {self._enabled_mode_names(descriptors)}",
                config_key="modes",
            )

        if not self.config.is_search_module_enabled(mode):
            if descriptor.needs_embedder:
                # Contract: embedding-backed search "degrades gracefully to
                # keyword-only" when embeddings are unavailable -- whether
                # disabled in config or auto-disabled at runtime by
                # _validate_search_model (missing pgvector table / Ollama).
                # Return a non-error result that steers the caller to keyword
                # search rather than raising, which would otherwise surface as
                # a hard MCP tool error (#276).
                return self._diagnostic_result(
                    reasoning=(
                        f"{mode.capitalize()} search is unavailable "
                        f"(embeddings not configured). Use keyword search instead."
                    ),
                    level=DiagnosticLevel.INFO,
                    source=f"service.{mode}",
                    category="search",
                )
            raise ConfigurationError(
                f"Search mode '{mode}' is not enabled. "
                f"Available modes: {self._enabled_mode_names(descriptors)}",
                config_key=f"search_modules.{mode}.enabled",
            )

        start_date, end_date = request.time_range if request.time_range else (None, None)

        args: list[Any] = [request.query, self.repository, self.config]
        if descriptor.needs_embedder:
            args.append(self._get_embedder())

        # Advanced params come first so the request's own fields win on collision.
        kwargs: dict[str, Any] = {
            **request.advanced_params,
            "max_results": request.max_results,
            "start_date": start_date,
            "end_date": end_date,
        }

        try:
            results = await descriptor.execute(*args, **kwargs)
        except Exception as e:
            logger.warning(f"{mode.capitalize()} search failed: {e}")
            return self._error_result(mode, f"service.{mode}", e)

        entries: list[dict[str, Any]] = []
        sources: list[Any] = []
        for entry, score, *extra in results:
            # Modules return (entry, score) or (entry, score, highlights).
            shaped = {**dict(entry), "_score": score}
            if extra:
                shaped["_highlights"] = extra[0]
            entries.append(shaped)
            sources.append(entry["entry_id"])

        return ARIELSearchResult(
            entries=tuple(entries),
            answer=None,
            sources=tuple(sources),
            search_modes_used=(mode,),
            reasoning=f"{mode.capitalize()} search: {len(entries)} results",
        )

    async def create_entry(
        self,
        request: FacilityEntryCreateRequest,
    ) -> FacilityEntryCreateResult:
        """Create a logbook entry through the facility adapter.

        Flow:
        1. Get the configured adapter
        2. Write to facility logbook via adapter.create_entry()
        3. Optimistic local upsert into ARIEL database
        4. For non-local adapters, attempt re-ingestion to sync

        Args:
            request: Entry creation request

        Returns:
            FacilityEntryCreateResult with entry ID and sync status

        Raises:
            NotImplementedError: If the adapter doesn't support writes
        """
        from datetime import UTC, datetime, timedelta

        from osprey.services.ariel_search.ingestion import get_adapter

        adapter = get_adapter(self.config)

        if not adapter.supports_write:
            raise NotImplementedError(
                f"{adapter.source_system_name} adapter does not support creating entries"
            )

        # Write to facility logbook
        facility_entry_id = await adapter.create_entry(request)

        now = datetime.now(UTC)
        source_system = adapter.source_system_name

        # Determine sync status based on adapter type
        is_local = source_system == "Generic JSON"
        sync_status = SyncStatus.LOCAL_ONLY if is_local else SyncStatus.PENDING_SYNC

        # Optimistic local upsert
        raw_text = f"{request.subject}\n\n{request.details}" if request.details else request.subject
        entry = {
            "entry_id": facility_entry_id,
            "source_system": source_system,
            "timestamp": now,
            "author": request.author or "",
            "raw_text": raw_text,
            "attachments": [],
            "metadata": {
                "logbook": request.logbook,
                "shift": request.shift,
                "tags": request.tags,
                "sync_status": sync_status.value,
            },
            "created_at": now,
            "updated_at": now,
        }

        await self.repository.upsert_entry(entry)

        # For non-local adapters, try to re-ingest the new entry to sync
        if not is_local:
            try:
                since = now - timedelta(minutes=5)
                async for fetched_entry in adapter.fetch_entries(since=since):
                    if fetched_entry["entry_id"] == facility_entry_id:
                        await self.repository.upsert_entry(fetched_entry)
                        sync_status = SyncStatus.SYNCED
                        break
            except Exception as e:
                logger.warning(
                    f"Re-ingestion after write failed for {facility_entry_id}: {e}. "
                    f"Entry will sync on next poll."
                )

        return FacilityEntryCreateResult(
            entry_id=facility_entry_id,
            source_system=source_system,
            sync_status=sync_status,
            message=f"Entry {facility_entry_id} created in {source_system}",
        )

    async def publish_entry(
        self,
        entry_id: str,
        *,
        logbook: str | None = None,
    ) -> FacilityEntryCreateResult:
        """Publish an existing ARIEL entry to the configured facility logbook.

        Writes through to the upstream source via create_entry(), which handles
        the adapter call, optimistic upsert, and re-ingestion. The ARIEL DB is
        a derived view — the upstream source is always the authority.

        Args:
            entry_id: ID of the existing ARIEL entry to publish
            logbook: Target logbook name (required by some facility APIs)

        Returns:
            FacilityEntryCreateResult with the facility-assigned entry ID

        Raises:
            KeyError: If entry_id not found in ARIEL database
            NotImplementedError: If the adapter doesn't support writes
        """
        entry = await self.repository.get_entry(entry_id)
        if entry is None:
            raise KeyError(f"Entry {entry_id} not found")

        subject = entry["raw_text"].split("\n", 1)[0].strip()
        details = entry["raw_text"]

        request = FacilityEntryCreateRequest(
            subject=subject,
            details=details,
            author=entry["author"],
            logbook=logbook,
            tags=entry["metadata"].get("tags", []),
        )

        return await self.create_entry(request)

    async def health_check(self) -> tuple[bool, str]:
        """Check service health.

        Returns:
            Tuple of (healthy, message)
        """
        db_healthy, db_msg = await self.repository.health_check()
        if not db_healthy:
            return (False, f"Database: {db_msg}")

        return (True, "ARIEL service healthy")

    async def get_status(self) -> ARIELStatusResult:
        """Get detailed ARIEL service status.

        Returns comprehensive service state including database connectivity,
        entry counts, embedding tables, and enabled modules.

        Returns:
            ARIELStatusResult with comprehensive service state.
        """
        errors: list[str] = []
        database_connected = False
        entry_count = None
        embedding_tables: list[EmbeddingTableInfo] = []
        last_ingestion = None

        masked_uri = self._mask_database_uri(self.config.database.uri)

        try:
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT 1")
                    database_connected = True

                    await cur.execute("SELECT COUNT(*) FROM enhanced_entries")
                    row = await cur.fetchone()
                    entry_count = row[0] if row else 0

                    embedding_tables = await self.repository.get_embedding_tables()

                    await cur.execute(
                        "SELECT MAX(completed_at) FROM ingestion_runs WHERE status = 'success'"
                    )
                    row = await cur.fetchone()
                    if row and row[0]:
                        last_ingestion = row[0]

        except Exception as e:
            errors.append(f"Database error: {e}")

        active_model = self.config.get_search_model()

        return ARIELStatusResult(
            healthy=database_connected and len(errors) == 0,
            database_connected=database_connected,
            database_uri=masked_uri,
            entry_count=entry_count,
            embedding_tables=embedding_tables,
            active_embedding_model=active_model,
            enabled_search_modules=self.config.get_enabled_search_modules(),
            enabled_enhancement_modules=self.config.get_enabled_enhancement_modules(),
            last_ingestion=last_ingestion,
            errors=errors,
        )

    def _mask_database_uri(self, uri: str) -> str:
        """Mask credentials in database URI for display.

        postgresql://user:password@host:5432/db -> postgresql://***@host:5432/db
        """
        import re

        return re.sub(r"://[^@]+@", "://***@", uri)

    async def __aenter__(self) -> ARIELSearchService:
        """Enter async context."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context and cleanup."""
        await self.pool.close()


async def create_ariel_service(
    config: ARIELConfig,
) -> ARIELSearchService:
    """Create and initialize an ARIEL search service.

    Factory function that sets up the database pool and repository.

    Args:
        config: ARIEL configuration

    Returns:
        Initialized ARIELSearchService

    Usage:
        async with create_ariel_service(config) as service:
            result = await service.search("What happened?")
    """
    from osprey.services.ariel_search.database.connection import create_connection_pool
    from osprey.services.ariel_search.database.repository import ARIELRepository

    pool = await create_connection_pool(config.database)
    repository = ARIELRepository(pool, config)

    return ARIELSearchService(
        config=config,
        pool=pool,
        repository=repository,
    )


__all__ = [
    "ARIELSearchService",
    "create_ariel_service",
]
