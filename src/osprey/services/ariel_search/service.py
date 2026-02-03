"""ARIEL Search Service.

This module provides the main ARIELSearchService class that orchestrates
database, search modules, and the ReAct agent.

See 04_OSPREY_INTEGRATION.md Section 5 for specification.
"""

from __future__ import annotations

import asyncio
import logging
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
from osprey.services.ariel_search.prompts import get_system_prompt
from osprey.services.ariel_search.tools import create_search_tools

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from psycopg_pool import AsyncConnectionPool

    from osprey.models.embeddings.base import BaseEmbeddingProvider
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.database.repository import ARIELRepository

logger = logging.getLogger(__name__)


class ARIELSearchService:
    """Main service class for ARIEL search functionality.

    This service:
    - Manages the database connection pool
    - Creates and manages the ReAct agent
    - Provides the main search interface

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
        llm: BaseChatModel | None = None,
    ) -> None:
        """Initialize the service.

        Args:
            config: ARIEL configuration
            pool: Database connection pool
            repository: Database repository
            llm: Optional LLM for the agent (lazy-loaded if not provided)
        """
        self.config = config
        self.pool = pool
        self.repository = repository
        self._llm = llm
        self._embedder: BaseEmbeddingProvider | None = None
        self._agent = None
        self._validated_search_model = False

    # === Lazy-loaded Properties ===

    def _get_llm(self) -> BaseChatModel:
        """Lazy-load the LLM for the agent.

        Uses config.reasoning.llm_model_id and llm_provider (GAP-C007).

        Returns:
            Configured BaseChatModel instance
        """
        if self._llm is None:
            model_id = self.config.reasoning.llm_model_id
            provider = self.config.reasoning.llm_provider

            if provider == "openai":
                try:
                    from langchain_openai import ChatOpenAI

                    self._llm = ChatOpenAI(
                        model=model_id,
                        temperature=self.config.reasoning.temperature,
                    )
                except ImportError as err:
                    raise ConfigurationError(
                        "langchain_openai is required for ARIEL agent with OpenAI provider. "
                        "Install with: pip install langchain-openai",
                        config_key="reasoning.llm_provider",
                    ) from err
            elif provider == "anthropic":
                try:
                    from langchain_anthropic import ChatAnthropic

                    self._llm = ChatAnthropic(
                        model=model_id,
                        temperature=self.config.reasoning.temperature,
                    )
                except ImportError as err:
                    raise ConfigurationError(
                        "langchain_anthropic is required for ARIEL agent with Anthropic provider. "
                        "Install with: pip install langchain-anthropic",
                        config_key="reasoning.llm_provider",
                    ) from err
            else:
                raise ConfigurationError(
                    f"Unsupported LLM provider: {provider}. "
                    "Supported providers: openai, anthropic",
                    config_key="reasoning.llm_provider",
                )

        return self._llm

    def _get_embedder(self) -> BaseEmbeddingProvider:
        """Lazy-load the embedding provider.

        Returns:
            Configured embedding provider instance
        """
        if self._embedder is None:
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
    ) -> ARIELSearchResult:
        """Execute a search using the ARIEL agent.

        This is the main entry point for searching the logbook.

        Args:
            query: Natural language query
            max_results: Maximum results (default from config)
            time_range: Optional (start, end) datetime tuple
            mode: Optional search mode hint

        Returns:
            ARIELSearchResult with entries, answer, and sources
        """
        # Build the search request
        request = ARIELSearchRequest(
            query=query,
            max_results=max_results or self.config.default_max_results,
            time_range=time_range,
            modes=[mode] if mode else [SearchMode.MULTI],
        )

        return await self.ainvoke(request)

    async def ainvoke(
        self,
        request: ARIELSearchRequest,
    ) -> ARIELSearchResult:
        """Invoke the ARIEL agent with a search request.

        This method:
        1. Validates configuration on first call
        2. Creates search tools from enabled modules
        3. Runs the ReAct agent with timeout enforcement
        4. Returns structured results

        Args:
            request: Search request with query and parameters

        Returns:
            ARIELSearchResult with entries, answer, and sources
        """
        try:
            # Validate search model table on first call (if semantic enabled)
            if self.config.is_search_module_enabled("semantic"):
                await self._validate_search_model()

            # Create search tools with captured context
            tools = create_search_tools(
                config=self.config,
                repository=self.repository,
                embedder_loader=self._get_embedder,
                request=request,
            )

            if not tools:
                return ARIELSearchResult(
                    entries=(),
                    answer=None,
                    sources=(),
                    search_modes_used=(),
                    reasoning="No search modules enabled in configuration",
                )

            # Build and run the agent
            result = await self._run_agent(request, tools)
            return result

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
            raise SearchExecutionError(
                f"Search execution failed: {e}",
                search_mode="multi",
                query=request.query,
            ) from e

    async def _run_agent(
        self,
        request: ARIELSearchRequest,
        tools: list[Any],
    ) -> ARIELSearchResult:
        """Run the ReAct agent with the given request and tools.

        Uses asyncio.wait_for for timeout enforcement.

        Args:
            request: Search request
            tools: List of LangChain StructuredTool instances

        Returns:
            ARIELSearchResult
        """
        try:
            from langgraph.prebuilt import create_react_agent

            # Get LLM
            llm = self._get_llm()

            # Create the agent with system prompt
            # See 03_AGENTIC_REASONING.md Section 2.4
            system_prompt = get_system_prompt(self.config)
            agent = create_react_agent(
                model=llm,
                tools=tools,
                prompt=system_prompt,
            )

            # Build initial messages (just the user query, system prompt is in agent)
            initial_messages = [
                {"role": "user", "content": request.query},
            ]

            # Calculate recursion limit from max_iterations
            # LangGraph counts each model call and tool execution as separate steps
            # So we need to double max_iterations (model + tool = 1 iteration)
            # Add 1 for the final response
            # See 03_AGENTIC_REASONING.md Section 3.2
            recursion_limit = (self.config.reasoning.max_iterations * 2) + 1

            # Run with timeout and recursion limit
            # See 03_AGENTIC_REASONING.md Section 2.1 for timeout specification
            try:
                result = await asyncio.wait_for(
                    agent.ainvoke(
                        {"messages": initial_messages},
                        config={"recursion_limit": recursion_limit},
                    ),
                    timeout=self.config.reasoning.total_timeout_seconds,
                )
            except asyncio.TimeoutError as err:
                # Wrap in SearchTimeoutError per GAP-C005
                raise SearchTimeoutError(
                    message=f"Agent execution timed out after {self.config.reasoning.total_timeout_seconds}s",
                    timeout_seconds=self.config.reasoning.total_timeout_seconds,
                    operation="agent execution",
                ) from err

            # Extract results from agent response
            return self._parse_agent_result(result, request)

        except ImportError as err:
            raise ConfigurationError(
                "langgraph is required for ARIEL agent. "
                "Install with: pip install langgraph",
                config_key="reasoning",
            ) from err

    def _parse_agent_result(
        self,
        result: dict[str, Any],
        request: ARIELSearchRequest,
    ) -> ARIELSearchResult:
        """Parse the agent's result into ARIELSearchResult.

        Args:
            result: Raw agent result
            request: Original search request

        Returns:
            Structured ARIELSearchResult
        """
        messages = result.get("messages", [])

        # Extract the final answer from the last AI message
        answer = None
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.type == "ai":
                answer = msg.content
                break

        # Extract entry IDs from citations in the answer
        sources: list[str] = []
        if answer:
            import re

            # Find [entry-XXX] or [#XXX] patterns
            citation_pattern = r"\[(?:entry-)?#?(\w+)\]"
            matches = re.findall(citation_pattern, answer)
            sources = list(dict.fromkeys(matches))  # Dedupe preserving order

        # Determine which search modes were used from tool calls
        search_modes_used: list[SearchMode] = []
        for msg in messages:
            if hasattr(msg, "tool_calls"):
                for tool_call in msg.tool_calls:
                    tool_name = tool_call.get("name", "")
                    if tool_name == "keyword_search" and SearchMode.KEYWORD not in search_modes_used:
                        search_modes_used.append(SearchMode.KEYWORD)
                    elif tool_name == "semantic_search" and SearchMode.SEMANTIC not in search_modes_used:
                        search_modes_used.append(SearchMode.SEMANTIC)
                    elif tool_name == "rag_search" and SearchMode.RAG not in search_modes_used:
                        search_modes_used.append(SearchMode.RAG)

        return ARIELSearchResult(
            entries=(),  # V2: Populate from tool results
            answer=answer,
            sources=tuple(sources),
            search_modes_used=tuple(search_modes_used),
            reasoning="",
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
