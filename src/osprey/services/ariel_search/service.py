"""ARIEL Search Service.

This module provides the main ARIELSearchService class that orchestrates
search execution. The service routes queries to one of four execution modes:
- KEYWORD / SEMANTIC: Direct calls to search functions
- RAG: Deterministic pipeline (retrieve → fuse → assemble → generate)
- AGENT: Non-deterministic ReAct agent

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
    ARIELSearchRequest,
    ARIELSearchResult,
    ARIELStatusResult,
    DiagnosticLevel,
    EmbeddingTableInfo,
    FacilityEntryCreateRequest,
    FacilityEntryCreateResult,
    SearchDiagnostic,
    SearchMode,
    SyncStatus,
)
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from psycopg_pool import AsyncConnectionPool

    from osprey.models.embeddings.base import BaseEmbeddingProvider
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.database.repository import ARIELRepository

logger = get_logger("ariel")


class ARIELSearchService:
    """Main service class for ARIEL search functionality.

    Routes queries based on SearchMode:
    - KEYWORD: Direct keyword_search() call
    - SEMANTIC: Direct semantic_search() call
    - RAG: RAGPipeline (hybrid retrieval + RRF + LLM generation)
    - AGENT: AgentExecutor (ReAct with search tools)

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
            provider_name = self.config.embedding.provider
            if provider_name != "ollama":
                logger.warning(
                    f"Embedding provider '{provider_name}' not yet supported, "
                    f"falling back to 'ollama'"
                )
            from osprey.models.embeddings.ollama import OllamaEmbeddingProvider

            self._embedder = OllamaEmbeddingProvider()
        return self._embedder

    @staticmethod
    def _error_result(
        mode: SearchMode,
        source: str,
        error: Exception,
    ) -> ARIELSearchResult:
        msg = f"{mode.value.capitalize()} search failed: {error}"
        return ARIELSearchResult(
            entries=(),
            answer=None,
            sources=(),
            search_modes_used=(mode,),
            reasoning=msg,
            diagnostics=(
                SearchDiagnostic(
                    level=DiagnosticLevel.ERROR,
                    source=source,
                    message=msg,
                    category="search",
                ),
            ),
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
        mode: SearchMode | None = None,
        advanced_params: dict[str, Any] | None = None,
    ) -> ARIELSearchResult:
        """Execute a search.

        This is the main entry point for searching the logbook.
        Routes to the appropriate execution mode.

        Args:
            query: Natural language query
            max_results: Maximum results (default from config)
            time_range: Optional (start, end) datetime tuple
            mode: Optional search mode (default: RAG)
            advanced_params: Mode-specific advanced parameters from the frontend

        Returns:
            ARIELSearchResult with entries, answer, and sources
        """
        # Build the search request
        request = ARIELSearchRequest(
            query=query,
            max_results=max_results or self.config.default_max_results,
            time_range=time_range,
            modes=[mode] if mode else [SearchMode.RAG],
            advanced_params=advanced_params or {},
        )

        return await self.ainvoke(request)

    async def ainvoke(
        self,
        request: ARIELSearchRequest,
    ) -> ARIELSearchResult:
        """Invoke ARIEL with a search request.

        Routes to the appropriate execution strategy based on mode.

        Args:
            request: Search request with query and parameters

        Returns:
            ARIELSearchResult with entries, answer, and sources
        """
        try:
            if self.config.is_search_module_enabled("semantic"):
                await self._validate_search_model()

            mode = request.modes[0] if request.modes else SearchMode.RAG

            match mode:
                case SearchMode.AGENT:
                    if not self.config.is_pipeline_enabled("agent"):
                        raise ConfigurationError(
                            "Agent pipeline not enabled",
                            config_key="pipelines.agent.enabled",
                        )
                    return await self._run_agent(request)
                case SearchMode.RAG:
                    if not self.config.is_pipeline_enabled("rag"):
                        raise ConfigurationError(
                            "RAG pipeline not enabled",
                            config_key="pipelines.rag.enabled",
                        )
                    return await self._run_rag(request)
                case SearchMode.KEYWORD:
                    return await self._run_keyword(request)
                case SearchMode.SEMANTIC:
                    return await self._run_semantic(request)
                case _:
                    raise ConfigurationError(
                        f"Unsupported mode: {mode.value}",
                        config_key="modes",
                    )

        except SearchTimeoutError as e:
            # Return graceful timeout result instead of propagating exception
            return ARIELSearchResult(
                entries=(),
                answer=None,
                sources=(),
                search_modes_used=(),
                reasoning=(
                    f"Search timed out before completion. "
                    f"{e.operation} timeout ({e.timeout_seconds}s) exceeded"
                ),
                diagnostics=(
                    SearchDiagnostic(
                        level=DiagnosticLevel.ERROR,
                        source="service.timeout",
                        message=(
                            f"Search timed out: {e.operation} exceeded {e.timeout_seconds}s limit"
                        ),
                        category="timeout",
                    ),
                ),
            )
        except ARIELException:
            raise
        except Exception as e:
            logger.exception(f"Search failed: {e}")
            mode = request.modes[0] if request.modes else SearchMode.RAG
            raise SearchExecutionError(
                f"Search execution failed: {e}",
                search_mode=mode.value,
                query=request.query,
            ) from e

    async def _run_keyword(self, request: ARIELSearchRequest) -> ARIELSearchResult:
        """Run keyword search directly.

        Args:
            request: Search request

        Returns:
            ARIELSearchResult with matching entries
        """
        if not self.config.is_search_module_enabled("keyword"):
            raise ConfigurationError(
                "Keyword search module not enabled",
                config_key="search_modules.keyword.enabled",
            )

        from osprey.services.ariel_search.search.keyword import keyword_search

        start_date, end_date = request.time_range if request.time_range else (None, None)

        ap = request.advanced_params
        include_highlights = ap.get("include_highlights", True)
        fuzzy_fallback = ap.get("fuzzy_fallback", True)

        try:
            results = await keyword_search(
                request.query,
                self.repository,
                self.config,
                max_results=request.max_results,
                start_date=start_date,
                end_date=end_date,
                author=ap.get("author"),
                source_system=ap.get("source_system"),
                include_highlights=include_highlights,
                fuzzy_fallback=fuzzy_fallback,
            )
        except Exception as e:
            logger.warning(f"Keyword search failed: {e}")
            return self._error_result(SearchMode.KEYWORD, "service.keyword", e)

        entries = tuple(
            {**dict(entry), "_score": score, "_highlights": highlights}
            for entry, score, highlights in results
        )
        sources = tuple(entry["entry_id"] for entry, _score, _highlights in results)

        return ARIELSearchResult(
            entries=entries,
            answer=None,
            sources=sources,
            search_modes_used=(SearchMode.KEYWORD,),
            reasoning=f"Keyword search: {len(results)} results",
        )

    async def _run_semantic(self, request: ARIELSearchRequest) -> ARIELSearchResult:
        """Run semantic search directly.

        Args:
            request: Search request

        Returns:
            ARIELSearchResult with matching entries
        """
        if not self.config.is_search_module_enabled("semantic"):
            raise ConfigurationError(
                "Semantic search module not enabled",
                config_key="search_modules.semantic.enabled",
            )

        from osprey.services.ariel_search.search.semantic import semantic_search

        start_date, end_date = request.time_range if request.time_range else (None, None)

        ap = request.advanced_params
        similarity_threshold = ap.get("similarity_threshold")

        try:
            results = await semantic_search(
                request.query,
                self.repository,
                self.config,
                self._get_embedder(),
                max_results=request.max_results,
                similarity_threshold=similarity_threshold,
                start_date=start_date,
                end_date=end_date,
                author=ap.get("author"),
                source_system=ap.get("source_system"),
            )
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
            return self._error_result(SearchMode.SEMANTIC, "service.semantic", e)

        entries = tuple({**dict(entry), "_score": similarity} for entry, similarity in results)
        sources = tuple(entry["entry_id"] for entry, _similarity in results)

        return ARIELSearchResult(
            entries=entries,
            answer=None,
            sources=sources,
            search_modes_used=(SearchMode.SEMANTIC,),
            reasoning=f"Semantic search: {len(results)} results",
        )

    async def _run_rag(self, request: ARIELSearchRequest) -> ARIELSearchResult:
        """Run the RAG pipeline.

        Args:
            request: Search request

        Returns:
            ARIELSearchResult with entries and LLM-generated answer
        """
        from osprey.services.ariel_search.rag import RAGPipeline

        ap = request.advanced_params
        max_context_chars = ap.get("max_context_chars", 12000)
        max_chars_per_entry = ap.get("max_chars_per_entry", 2000)
        similarity_threshold = ap.get("similarity_threshold")
        temperature = ap.get("temperature")

        prompt_template = None
        try:
            from osprey.prompts.loader import get_framework_prompts

            builder = get_framework_prompts().get_ariel_rag_prompt_builder()
            prompt_template = builder.get_prompt_template()  # type: ignore[attr-defined]
        except (ValueError, NotImplementedError, AttributeError):
            pass  # Falls back to hardcoded default inside RAGPipeline

        pipeline = RAGPipeline(
            repository=self.repository,
            config=self.config,
            embedder_loader=self._get_embedder,
            max_context_chars=max_context_chars,
            max_chars_per_entry=max_chars_per_entry,
            prompt_template=prompt_template,
        )

        start_date, end_date = request.time_range if request.time_range else (None, None)

        rag_result = await pipeline.execute(
            request.query,
            max_results=request.max_results,
            similarity_threshold=similarity_threshold,
            start_date=start_date,
            end_date=end_date,
            author=ap.get("author"),
            source_system=ap.get("source_system"),
            temperature=temperature,
        )

        return ARIELSearchResult(
            entries=rag_result.entries,
            answer=rag_result.answer,
            sources=rag_result.citations,
            search_modes_used=(SearchMode.RAG,),
            reasoning=f"RAG pipeline: {rag_result.retrieval_count} retrieved, "
            f"{len(rag_result.entries)} in context",
            diagnostics=rag_result.diagnostics,
            pipeline_details=rag_result.pipeline_details,
        )

    async def _run_agent(self, request: ARIELSearchRequest) -> ARIELSearchResult:
        """Run the AgentExecutor for agentic search.

        Args:
            request: Search request

        Returns:
            ARIELSearchResult
        """
        from osprey.services.ariel_search.agent import AgentExecutor

        # Load prompt from framework prompt system
        system_prompt = None
        try:
            from osprey.prompts.loader import get_framework_prompts

            builder = get_framework_prompts().get_ariel_agent_prompt_builder()
            system_prompt = builder.get_system_prompt()  # type: ignore[attr-defined]
        except (ValueError, NotImplementedError, AttributeError):
            pass  # Falls back to hardcoded default inside AgentExecutor

        executor = AgentExecutor(
            repository=self.repository,
            config=self.config,
            embedder_loader=self._get_embedder,
            system_prompt=system_prompt,
        )

        agent_result = await executor.execute(
            query=request.query,
            max_results=request.max_results,
            time_range=request.time_range,
        )

        return ARIELSearchResult(
            entries=agent_result.entries,
            answer=agent_result.answer,
            sources=agent_result.sources,
            search_modes_used=agent_result.search_modes_used,
            reasoning=agent_result.reasoning,
            diagnostics=agent_result.diagnostics,
            pipeline_details=agent_result.pipeline_details,
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
            enabled_pipelines=self.config.get_enabled_pipelines(),
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
