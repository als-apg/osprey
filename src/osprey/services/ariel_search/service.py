"""ARIEL Search Service.

This module provides the main ARIELSearchService class that orchestrates
search execution. The service routes queries to one of four execution modes:
- KEYWORD / SEMANTIC: Direct calls to search functions
- RAG: Deterministic pipeline (retrieve → fuse → assemble → generate)
- AGENT: Non-deterministic ReAct agent

See 04_OSPREY_INTEGRATION.md Section 5 for specification.
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
    EmbeddingTableInfo,
    SearchMode,
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

            # Dynamic provider selection based on config
            if provider_name == "ollama":
                from osprey.models.embeddings.ollama import OllamaEmbeddingProvider

                self._embedder = OllamaEmbeddingProvider()
            else:
                # For other providers, default to Ollama for now
                # Additional providers can be added here as they are implemented
                logger.warning(
                    f"Embedding provider '{provider_name}' not yet supported, "
                    f"falling back to 'ollama'"
                )
                from osprey.models.embeddings.ollama import OllamaEmbeddingProvider

                self._embedder = OllamaEmbeddingProvider()

        return self._embedder

    # === Validation ===

    async def _validate_search_model(self) -> None:
        """Validate that the configured search model's table exists.

        Called lazily on first semantic search.
        """
        if self._validated_search_model:
            return

        model = self.config.get_search_model()
        if model:
            await self.repository.validate_search_model_table(model)

        self._validated_search_model = True

    # === Main Search Interface ===

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
            # Validate search model table on first call (if semantic enabled)
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

    # === Mode-specific execution ===

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

        start_date = request.time_range[0] if request.time_range else None
        end_date = request.time_range[1] if request.time_range else None

        # Extract keyword-specific advanced params
        ap = request.advanced_params
        include_highlights = ap.get("include_highlights", True)
        fuzzy_fallback = ap.get("fuzzy_fallback", True)

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

        entries = tuple(
            {**dict(entry), "_highlights": highlights} for entry, _score, highlights in results
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

        start_date = request.time_range[0] if request.time_range else None
        end_date = request.time_range[1] if request.time_range else None

        # Extract semantic-specific advanced params
        ap = request.advanced_params
        similarity_threshold = ap.get("similarity_threshold")

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

        entries = tuple(dict(entry) for entry, _similarity in results)
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

        # Extract RAG-specific advanced params
        ap = request.advanced_params
        max_context_chars = ap.get("max_context_chars", 12000)
        max_chars_per_entry = ap.get("max_chars_per_entry", 2000)
        similarity_threshold = ap.get("similarity_threshold")
        temperature = ap.get("temperature")

        pipeline = RAGPipeline(
            repository=self.repository,
            config=self.config,
            embedder_loader=self._get_embedder,
            max_context_chars=max_context_chars,
            max_chars_per_entry=max_chars_per_entry,
        )

        start_date = request.time_range[0] if request.time_range else None
        end_date = request.time_range[1] if request.time_range else None

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
        )

    async def _run_agent(self, request: ARIELSearchRequest) -> ARIELSearchResult:
        """Run the AgentExecutor for agentic search.

        Args:
            request: Search request

        Returns:
            ARIELSearchResult
        """
        from osprey.services.ariel_search.agent import AgentExecutor

        # Create agent executor
        executor = AgentExecutor(
            repository=self.repository,
            config=self.config,
            embedder_loader=self._get_embedder,
        )

        # Execute agent
        agent_result = await executor.execute(
            query=request.query,
            max_results=request.max_results,
            time_range=request.time_range,
        )

        # Convert agent result to ARIELSearchResult
        return ARIELSearchResult(
            entries=agent_result.entries,
            answer=agent_result.answer,
            sources=agent_result.sources,
            search_modes_used=agent_result.search_modes_used,
            reasoning=agent_result.reasoning,
        )

    # === Health Check ===

    async def health_check(self) -> tuple[bool, str]:
        """Check service health.

        Returns:
            Tuple of (healthy, message)
        """
        # Check database
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

        # Mask database URI for security
        masked_uri = self._mask_database_uri(self.config.database.uri)

        # Check database connectivity and gather stats
        try:
            async with self.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT 1")
                    database_connected = True

                    # Get entry count
                    await cur.execute("SELECT COUNT(*) FROM enhanced_entries")
                    row = await cur.fetchone()
                    entry_count = row[0] if row else 0

                    # Get embedding tables info
                    embedding_tables = await self.repository.get_embedding_tables()

                    # Get last ingestion time
                    await cur.execute(
                        "SELECT MAX(completed_at) FROM ingestion_runs WHERE status = 'success'"
                    )
                    row = await cur.fetchone()
                    if row and row[0]:
                        last_ingestion = row[0]

        except Exception as e:
            errors.append(f"Database error: {e}")

        # Get active embedding model from config
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

    # === Context Manager ===

    async def __aenter__(self) -> ARIELSearchService:
        """Enter async context."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context and cleanup."""
        # Close the connection pool
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

    # Create connection pool
    pool = await create_connection_pool(config.database)

    # Create repository
    repository = ARIELRepository(pool, config)

    # Create and return service
    return ARIELSearchService(
        config=config,
        pool=pool,
        repository=repository,
    )


__all__ = [
    "ARIELSearchService",
    "create_ariel_service",
]
