"""ARIEL Agent Executor for ReAct-style agentic orchestration.

This module provides the AgentExecutor class that encapsulates ReAct agent
logic with search tools. The agent decides what to search and synthesizes
answers from multiple tool invocations.

See 03_AGENTIC_REASONING.md for specification.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from osprey.services.ariel_search.exceptions import (
    ConfigurationError,
    SearchTimeoutError,
)
from osprey.services.ariel_search.models import SearchMode

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import StructuredTool

    from osprey.models.embeddings.base import BaseEmbeddingProvider
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.database.repository import ARIELRepository
    from osprey.services.ariel_search.models import EnhancedLogbookEntry

logger = logging.getLogger(__name__)


# === System Prompt ===

AGENT_SYSTEM_PROMPT = """You are ARIEL, an AI assistant for searching and analyzing facility logbook entries.

Your purpose is to help users find relevant information in the electronic logbook system.

## Available Tools

You have access to the following search tools:

1. **keyword_search** - Fast text-based lookup using full-text search
   - Use when searching for specific terms, equipment names, PV names, or phrases
   - Supports quoted phrases ("beam loss") and boolean operators (AND, OR, NOT)

2. **semantic_search** - Find conceptually related entries using AI embeddings
   - Use when the query describes a concept or situation
   - Good for finding entries about similar events even if exact words don't match

## Guidelines

- Choose the appropriate tool based on the user's query
- You may use multiple tools if needed to gather complete information
- Always cite specific entry IDs when referencing information: [#12345]
- If no relevant entries are found, say so clearly
- Keep responses concise but informative
- Focus on factual information from the logbook entries

## Response Format

- For search results: Summarize key findings with entry ID citations
- For questions: Provide a direct answer citing source entries
- For "no results": "I don't have information about this in the logbook."
"""


# === Input Schemas for Tools ===


class KeywordSearchInput(BaseModel):
    """Input schema for keyword search tool."""

    query: str = Field(
        description="Search terms. Supports phrases in quotes, AND/OR/NOT operators."
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum results to return",
    )
    start_date: datetime | None = Field(
        default=None,
        description="Filter entries created after this time (inclusive)",
    )
    end_date: datetime | None = Field(
        default=None,
        description="Filter entries created before this time (inclusive)",
    )


class SemanticSearchInput(BaseModel):
    """Input schema for semantic search tool."""

    query: str = Field(description="Natural language description of what to find")
    max_results: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum results to return",
    )
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score (0-1)",
    )
    start_date: datetime | None = Field(
        default=None,
        description="Filter entries created after this time (inclusive)",
    )
    end_date: datetime | None = Field(
        default=None,
        description="Filter entries created before this time (inclusive)",
    )


# === Tool Output Formatting ===


def format_keyword_result(
    entry: EnhancedLogbookEntry,
    score: float,
    highlights: list[str],
) -> dict[str, Any]:
    """Format a keyword search result for agent consumption.

    Args:
        entry: EnhancedLogbookEntry
        score: Relevance score
        highlights: Highlighted snippets

    Returns:
        Formatted dict for agent
    """
    timestamp = entry.get("timestamp")
    return {
        "entry_id": entry.get("entry_id"),
        "timestamp": timestamp.isoformat() if timestamp is not None else None,
        "author": entry.get("author"),
        "text": entry.get("raw_text", "")[:500],  # Truncate for agent
        "title": entry.get("metadata", {}).get("title"),
        "score": score,
        "highlights": highlights,
    }


def format_semantic_result(
    entry: EnhancedLogbookEntry,
    similarity: float,
) -> dict[str, Any]:
    """Format a semantic search result for agent consumption.

    Args:
        entry: EnhancedLogbookEntry
        similarity: Cosine similarity score

    Returns:
        Formatted dict for agent
    """
    timestamp = entry.get("timestamp")
    return {
        "entry_id": entry.get("entry_id"),
        "timestamp": timestamp.isoformat() if timestamp is not None else None,
        "author": entry.get("author"),
        "text": entry.get("raw_text", "")[:500],
        "title": entry.get("metadata", {}).get("title"),
        "similarity": similarity,
    }


# === Agent Result ===


@dataclass(frozen=True)
class AgentResult:
    """Result from agent execution.

    Attributes:
        answer: Generated answer text
        entries: Matching entries (from tool calls)
        sources: Entry IDs used as sources
        search_modes_used: Which search modes were invoked
        reasoning: Explanation of results
    """

    answer: str | None = None
    entries: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    sources: tuple[str, ...] = field(default_factory=tuple)
    search_modes_used: tuple[SearchMode, ...] = field(default_factory=tuple)
    reasoning: str = ""


# === Agent Executor ===


class AgentExecutor:
    """Executor for ReAct-style agent with search tools.

    The AgentExecutor runs a ReAct agent that can use keyword_search and
    semantic_search tools to find information and synthesize answers.

    This is a separate interface from the Pipeline - use Agent for agentic
    orchestration (AGENT mode) and Pipeline for deterministic search modes.

    Usage:
        executor = AgentExecutor(repository, config, embedder_loader)
        result = await executor.execute("What happened yesterday?")
        print(result.answer)
    """

    def __init__(
        self,
        repository: ARIELRepository,
        config: ARIELConfig,
        embedder_loader: Callable[[], BaseEmbeddingProvider],
        llm: BaseChatModel | None = None,
    ) -> None:
        """Initialize the agent executor.

        Args:
            repository: Database repository instance
            config: ARIEL configuration
            embedder_loader: Callable that returns embedding model (lazy-loaded)
            llm: Optional LLM for the agent (lazy-loaded if not provided)
        """
        self.repository = repository
        self.config = config
        self._embedder_loader = embedder_loader
        self._llm = llm

    def _get_llm(self) -> BaseChatModel:
        """Lazy-load the LLM for the agent.

        Uses Osprey's provider configuration system for centralized credential management.
        The provider name in config.reasoning.provider references api.providers section.

        Returns:
            Configured BaseChatModel instance
        """
        if self._llm is None:
            from osprey.models.langchain import get_langchain_model

            provider_name = self.config.reasoning.provider
            model_id = self.config.reasoning.model_id

            try:
                # Get provider config from Osprey's central configuration
                # This may fail in test environments without config.yml
                from osprey.utils.config import get_provider_config

                try:
                    provider_config = get_provider_config(provider_name)
                except FileNotFoundError:
                    # Test environment without config.yml - use empty config
                    logger.debug(
                        f"No config.yml found, using empty provider config for '{provider_name}'"
                    )
                    provider_config = {}

                self._llm = get_langchain_model(
                    provider=provider_name,
                    model_id=model_id,
                    provider_config=provider_config,
                    temperature=self.config.reasoning.temperature,
                )
            except ImportError as err:
                raise ConfigurationError(
                    f"Required LangChain package not installed for provider '{provider_name}'. "
                    f"See error details: {err}",
                    config_key="reasoning.provider",
                ) from err
            except ValueError as err:
                raise ConfigurationError(
                    f"Invalid provider configuration for '{provider_name}': {err}",
                    config_key="reasoning.provider",
                ) from err

        return self._llm

    def _create_tools(
        self,
        time_range: tuple[datetime, datetime] | None = None,
    ) -> list[StructuredTool]:
        """Create LangChain tools for the agent.

        Creates keyword_search and semantic_search tools with captured context.

        Time Range Resolution (3-tier priority):
        1. Tool call parameter (highest) - Agent explicitly passes start_date/end_date
        2. Request context - From time_range parameter (default for session)
        3. No filter (lowest) - Search all entries

        Args:
            time_range: Optional default time range for searches

        Returns:
            List of StructuredTool instances
        """
        from langchain_core.tools import StructuredTool

        tools: list[StructuredTool] = []

        def _resolve_time_range(
            tool_start: datetime | None,
            tool_end: datetime | None,
        ) -> tuple[datetime | None, datetime | None]:
            """Resolve time range with 3-tier priority."""
            # Explicit tool params override request context
            if tool_start is not None or tool_end is not None:
                return (tool_start, tool_end)
            # Fall back to request context
            if time_range:
                return time_range
            # No filtering
            return (None, None)

        # Keyword Search Tool
        if self.config.is_search_module_enabled("keyword"):

            async def _keyword_search(
                query: str,
                max_results: int = 10,
                start_date: datetime | None = None,
                end_date: datetime | None = None,
            ) -> list[dict[str, Any]]:
                """Execute keyword search with captured dependencies."""
                from osprey.services.ariel_search.search.keyword import keyword_search

                resolved_start, resolved_end = _resolve_time_range(start_date, end_date)

                results = await keyword_search(
                    query=query,
                    repository=self.repository,
                    config=self.config,
                    max_results=max_results,
                    start_date=resolved_start,
                    end_date=resolved_end,
                )

                return [
                    format_keyword_result(entry, score, highlights)
                    for entry, score, highlights in results
                ]

            tools.append(
                StructuredTool.from_function(
                    func=_keyword_search,
                    coroutine=_keyword_search,
                    name="keyword_search",
                    description=(
                        "Fast text-based lookup using full-text search. "
                        "Use for specific terms, equipment names, PV names, or phrases. "
                        "Supports quoted phrases and AND/OR/NOT operators."
                    ),
                    args_schema=KeywordSearchInput,
                )
            )

        # Semantic Search Tool
        if self.config.is_search_module_enabled("semantic"):

            async def _semantic_search(
                query: str,
                max_results: int = 10,
                similarity_threshold: float = 0.7,
                start_date: datetime | None = None,
                end_date: datetime | None = None,
            ) -> list[dict[str, Any]]:
                """Execute semantic search with captured dependencies."""
                from osprey.services.ariel_search.search.semantic import semantic_search

                resolved_start, resolved_end = _resolve_time_range(start_date, end_date)
                embedder = self._embedder_loader()

                results = await semantic_search(
                    query=query,
                    repository=self.repository,
                    config=self.config,
                    embedder=embedder,
                    max_results=max_results,
                    similarity_threshold=similarity_threshold,
                    start_date=resolved_start,
                    end_date=resolved_end,
                )

                return [format_semantic_result(entry, similarity) for entry, similarity in results]

            tools.append(
                StructuredTool.from_function(
                    func=_semantic_search,
                    coroutine=_semantic_search,
                    name="semantic_search",
                    description=(
                        "Find conceptually related entries using AI embeddings. "
                        "Use for queries describing concepts, situations, or events "
                        "where exact words may not match."
                    ),
                    args_schema=SemanticSearchInput,
                )
            )

        return tools

    async def execute(
        self,
        query: str,
        *,
        max_results: int | None = None,
        time_range: tuple[datetime, datetime] | None = None,
    ) -> AgentResult:
        """Execute the agent with a search query.

        The agent will use search tools to find relevant entries and
        synthesize an answer based on the results.

        Args:
            query: Natural language query
            max_results: Maximum results to return (unused, for API compatibility)
            time_range: Optional (start, end) datetime tuple for filtering

        Returns:
            AgentResult with answer, entries, sources, and reasoning
        """
        try:
            # Create search tools with time range context
            tools = self._create_tools(time_range=time_range)

            if not tools:
                return AgentResult(
                    answer=None,
                    entries=(),
                    sources=(),
                    search_modes_used=(),
                    reasoning="No search modules enabled in configuration",
                )

            # Build and run the agent
            result = await self._run_agent(query, tools)
            return result

        except SearchTimeoutError:
            raise
        except Exception as e:
            logger.exception(f"Agent execution failed: {e}")
            raise

    async def _run_agent(
        self,
        query: str,
        tools: list[StructuredTool],
    ) -> AgentResult:
        """Run the ReAct agent with the given query and tools.

        Uses asyncio.wait_for for timeout enforcement.

        Args:
            query: Search query
            tools: List of LangChain StructuredTool instances

        Returns:
            AgentResult
        """
        try:
            from langgraph.prebuilt import create_react_agent

            # Get LLM
            llm = self._get_llm()

            # Create the agent with system prompt
            agent = create_react_agent(
                model=llm,
                tools=tools,
                prompt=AGENT_SYSTEM_PROMPT,
            )

            # Build initial messages
            initial_messages = [
                {"role": "user", "content": query},
            ]

            # Calculate recursion limit from max_iterations
            # LangGraph counts each model call and tool execution as separate steps
            # So we need to double max_iterations (model + tool = 1 iteration)
            # Add 1 for the final response
            recursion_limit = (self.config.reasoning.max_iterations * 2) + 1

            # Run with timeout and recursion limit
            try:
                result = await asyncio.wait_for(
                    agent.ainvoke(
                        {"messages": initial_messages},
                        config={"recursion_limit": recursion_limit},
                    ),
                    timeout=self.config.reasoning.total_timeout_seconds,
                )
            except TimeoutError as err:
                raise SearchTimeoutError(
                    message=f"Agent execution timed out after {self.config.reasoning.total_timeout_seconds}s",
                    timeout_seconds=self.config.reasoning.total_timeout_seconds,
                    operation="agent execution",
                ) from err

            # Extract results from agent response
            return self._parse_agent_result(result)

        except ImportError as err:
            raise ConfigurationError(
                "langgraph is required for ARIEL agent. Install with: pip install langgraph",
                config_key="reasoning",
            ) from err

    def _parse_agent_result(
        self,
        result: dict[str, Any],
    ) -> AgentResult:
        """Parse the agent's result into AgentResult.

        Args:
            result: Raw agent result

        Returns:
            Structured AgentResult
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
                    if (
                        tool_name == "keyword_search"
                        and SearchMode.KEYWORD not in search_modes_used
                    ):
                        search_modes_used.append(SearchMode.KEYWORD)
                    elif (
                        tool_name == "semantic_search"
                        and SearchMode.SEMANTIC not in search_modes_used
                    ):
                        search_modes_used.append(SearchMode.SEMANTIC)

        return AgentResult(
            answer=answer,
            entries=(),  # V2: Populate from tool results
            sources=tuple(sources),
            search_modes_used=tuple(search_modes_used),
            reasoning="",
        )


__all__ = [
    "AGENT_SYSTEM_PROMPT",
    "AgentExecutor",
    "AgentResult",
    "KeywordSearchInput",
    "SemanticSearchInput",
    "format_keyword_result",
    "format_semantic_result",
]
