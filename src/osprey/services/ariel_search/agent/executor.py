"""ARIEL Agent Executor for ReAct-style agentic orchestration.

This module provides the AgentExecutor class that encapsulates ReAct agent
logic with search tools. The agent decides what to search and synthesizes
answers from multiple tool invocations.

Search tools are auto-discovered from the Osprey registry via
SearchToolDescriptor — adding a new search module requires zero changes here.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from osprey.services.ariel_search.exceptions import (
    ConfigurationError,
    SearchTimeoutError,
)
from osprey.services.ariel_search.models import (
    AgentStep,
    AgentToolInvocation,
    DiagnosticLevel,
    EnhancedLogbookEntry,
    PipelineDetails,
    SearchDiagnostic,
    SearchMode,
)
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import StructuredTool

    from osprey.models.embeddings.base import BaseEmbeddingProvider
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.database.repository import ARIELRepository
    from osprey.services.ariel_search.search.base import SearchToolDescriptor

logger = get_logger("ariel")


AGENT_SYSTEM_PROMPT = """You are ARIEL, an AI assistant for searching and analyzing facility logbook entries.

Your purpose is to help users find relevant information in the electronic logbook system.

## Guidelines

- Use the available search tools to find relevant logbook entries
- You may call tools multiple times with different queries to gather complete information
- Always cite specific entry IDs when referencing information
- If no relevant entries are found, say so clearly
- Keep responses concise but informative
- Focus on factual information from the logbook entries

## Response Format

- Summarize key findings with entry ID citations
- Provide direct answers citing source entries
- If nothing is found: clearly state that no relevant information was found in the logbook
"""


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
    diagnostics: tuple[SearchDiagnostic, ...] = field(default_factory=tuple)
    pipeline_details: PipelineDetails | None = None


class AgentExecutor:
    """Executor for ReAct-style agent with search tools.

    The AgentExecutor auto-discovers search tools from the Osprey registry.
    Each search module provides a `get_tool_descriptor()` that describes its tool —
    the executor wraps descriptors into LangChain StructuredTools generically.

    Adding a new search module requires zero changes to this class.

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
        system_prompt: str | None = None,
    ) -> None:
        self.repository = repository
        self.config = config
        self._embedder_loader = embedder_loader
        self._llm = llm
        self._system_prompt = system_prompt or AGENT_SYSTEM_PROMPT

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
                from osprey.utils.config import get_provider_config

                try:
                    provider_config = get_provider_config(provider_name)
                except FileNotFoundError:
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

    def _load_descriptors(self) -> list[SearchToolDescriptor]:
        """Load tool descriptors from enabled search modules via the registry.

        Uses the agent pipeline's configured retrieval_modules if available,
        otherwise falls back to all enabled search modules.

        Returns:
            List of SearchToolDescriptor for each enabled and registered module
        """
        from osprey.registry import get_registry

        registry = get_registry()
        descriptors: list[SearchToolDescriptor] = []

        # Use pipeline-configured modules, filtered to only enabled ones
        retrieval_modules = self.config.get_pipeline_retrieval_modules("agent")
        for module_name in retrieval_modules:
            if not self.config.is_search_module_enabled(module_name):
                continue
            module = registry.get_ariel_search_module(module_name)
            if module is not None:
                descriptors.append(module.get_tool_descriptor())
        return descriptors

    def _build_tool(
        self,
        descriptor: SearchToolDescriptor,
        time_range: tuple[datetime, datetime] | None = None,
        collected_ids: list[str] | None = None,
    ) -> StructuredTool:
        """Build a LangChain StructuredTool from a descriptor.

        Creates an async closure that captures the repository, config, and
        optional embedder, then wraps it in a StructuredTool.

        Time Range Resolution (3-tier priority):
        1. Tool call parameter (highest) - Agent explicitly passes start_date/end_date
        2. Request context - From time_range parameter (default for session)
        3. No filter (lowest) - Search all entries

        Args:
            descriptor: Search tool descriptor
            time_range: Optional default time range for searches
            collected_ids: Mutable list to collect entry IDs from tool results

        Returns:
            Configured StructuredTool
        """
        from langchain_core.tools import StructuredTool

        from osprey.services.ariel_search.models import resolve_time_range

        _execute = descriptor.execute
        _format = descriptor.format_result
        _needs_embedder = descriptor.needs_embedder

        _collected_ids = collected_ids

        async def _tool_fn(**kwargs: Any) -> list[dict[str, Any]]:
            start_date = kwargs.pop("start_date", None)
            end_date = kwargs.pop("end_date", None)
            resolved_start, resolved_end = resolve_time_range(start_date, end_date, time_range)

            call_kwargs: dict[str, Any] = {
                "query": kwargs.pop("query"),
                "repository": self.repository,
                "config": self.config,
                "start_date": resolved_start,
                "end_date": resolved_end,
                **kwargs,
            }

            if _needs_embedder:
                call_kwargs["embedder"] = self._embedder_loader()

            results = await _execute(**call_kwargs)

            formatted = [
                _format(*item) if isinstance(item, tuple) else _format(item) for item in results
            ]

            if _collected_ids is not None:
                for item in formatted:
                    eid = item.get("entry_id")
                    if eid:
                        _collected_ids.append(eid)

            return formatted

        return StructuredTool.from_function(
            func=_tool_fn,
            coroutine=_tool_fn,
            name=descriptor.name,
            description=descriptor.description,
            args_schema=descriptor.args_schema,
        )

    def _create_tools(
        self,
        time_range: tuple[datetime, datetime] | None = None,
        collected_ids: list[str] | None = None,
    ) -> tuple[list[StructuredTool], list[SearchToolDescriptor]]:
        """Create LangChain tools from auto-discovered descriptors.

        Args:
            time_range: Optional default time range for searches
            collected_ids: Mutable list to collect entry IDs from tool results

        Returns:
            Tuple of (tools list, descriptors list)
        """
        descriptors = self._load_descriptors()
        tools = [self._build_tool(d, time_range, collected_ids) for d in descriptors]
        return tools, descriptors

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
            max_results: Maximum results to return (reserved for future use)
            time_range: Optional (start, end) datetime tuple for filtering

        Returns:
            AgentResult with answer, entries, sources, and reasoning
        """
        try:
            collected_ids: list[str] = []
            tools, descriptors = self._create_tools(
                time_range=time_range, collected_ids=collected_ids
            )

            if not tools:
                return AgentResult(
                    answer=None,
                    entries=(),
                    sources=(),
                    search_modes_used=(),
                    reasoning="No search modules enabled in configuration",
                    diagnostics=(
                        SearchDiagnostic(
                            level=DiagnosticLevel.WARNING,
                            source="agent.tools",
                            message="No search modules enabled",
                        ),
                    ),
                )

            result = await self._run_agent(query, tools)

            unique_ids = list(dict.fromkeys(collected_ids))
            entries: list[EnhancedLogbookEntry] = []
            if unique_ids:
                try:
                    entries = await self.repository.get_entries_by_ids(unique_ids)
                except Exception:
                    logger.warning("Failed to fetch full entries for agent results", exc_info=True)

            return self._parse_agent_result(result, descriptors, entries=entries)

        except SearchTimeoutError:
            raise
        except Exception as e:
            logger.exception(f"Agent execution failed: {e}")
            raise

    async def _run_agent(
        self,
        query: str,
        tools: list[StructuredTool],
    ) -> dict[str, Any]:
        """Run the ReAct agent with the given query and tools.

        Uses asyncio.wait_for for timeout enforcement.

        Args:
            query: Search query
            tools: List of LangChain StructuredTool instances

        Returns:
            Raw agent result dict
        """
        try:
            from langgraph.prebuilt import create_react_agent

            llm = self._get_llm()

            agent = create_react_agent(
                model=llm,
                tools=tools,
                prompt=self._system_prompt,
            )

            initial_messages = [
                {"role": "user", "content": query},
            ]

            recursion_limit = (self.config.reasoning.max_iterations * 2) + 1

            try:
                return await asyncio.wait_for(
                    agent.ainvoke(
                        {"messages": initial_messages},
                        config={"recursion_limit": recursion_limit},
                    ),
                    timeout=self.config.reasoning.total_timeout_seconds,
                )
            except TimeoutError as err:
                raise SearchTimeoutError(
                    message=(
                        f"Agent execution timed out after "
                        f"{self.config.reasoning.total_timeout_seconds}s"
                    ),
                    timeout_seconds=self.config.reasoning.total_timeout_seconds,
                    operation="agent execution",
                ) from err

        except ImportError as err:
            raise ConfigurationError(
                "langgraph is required for ARIEL agent. Install with: pip install langgraph",
                config_key="reasoning",
            ) from err

    def _parse_agent_result(
        self,
        result: dict[str, Any],
        descriptors: list[SearchToolDescriptor],
        entries: list[EnhancedLogbookEntry] | None = None,
    ) -> AgentResult:
        """Parse the agent's result into AgentResult.

        Uses descriptors to dynamically map tool names to SearchMode values.

        Args:
            result: Raw agent result
            descriptors: Loaded descriptors (for tool name → SearchMode mapping)
            entries: Full entries fetched from the repository

        Returns:
            Structured AgentResult
        """
        tool_name_to_mode = {d.name: d.search_mode for d in descriptors}
        messages = result.get("messages", [])

        answer = None
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.type == "ai":
                answer = msg.content
                break

        sources: list[str] = []
        if answer and entries:
            sources = [
                e["entry_id"] for e in entries if e.get("entry_id") and e["entry_id"] in answer
            ]

        search_modes_used: list[SearchMode] = []
        tool_invocations: list[AgentToolInvocation] = []
        steps: list[AgentStep] = []
        tool_call_order = 0
        step_order = 0

        call_id_to_name: dict[str, str] = {}

        def _process_tool_calls(
            tool_calls: list[dict],
        ) -> None:
            nonlocal tool_call_order, step_order
            for tc in tool_calls:
                t_name = tc.get("name", "unknown")
                t_args = tc.get("args", {})
                t_id = tc.get("id", "")
                call_id_to_name[t_id] = t_name

                tool_invocations.append(
                    AgentToolInvocation(
                        tool_name=t_name,
                        tool_args=t_args,
                        order=tool_call_order,
                    )
                )
                steps.append(
                    AgentStep(
                        step_type="tool_call",
                        content=str(t_args)[:200],
                        tool_name=t_name,
                        order=step_order,
                    )
                )
                tool_call_order += 1
                step_order += 1

                mode = tool_name_to_mode.get(t_name)
                if mode is not None and mode not in search_modes_used:
                    search_modes_used.append(mode)

        for msg in messages:
            msg_type = getattr(msg, "type", None)

            if msg_type == "human":
                content = getattr(msg, "content", "")
                steps.append(
                    AgentStep(
                        step_type="user_query",
                        content=str(content)[:200],
                        order=step_order,
                    )
                )
                step_order += 1

            elif msg_type == "ai":
                content = getattr(msg, "content", "")
                tool_calls = getattr(msg, "tool_calls", None) or []

                if content and isinstance(content, str):
                    if not tool_calls:
                        steps.append(
                            AgentStep(
                                step_type="final_answer" if msg is messages[-1] else "reasoning",
                                content=str(content)[:200],
                                order=step_order,
                            )
                        )
                        step_order += 1
                    else:
                        steps.append(
                            AgentStep(
                                step_type="reasoning",
                                content=str(content)[:200],
                                order=step_order,
                            )
                        )
                        step_order += 1

                _process_tool_calls(tool_calls)

            elif msg_type == "tool":
                content = getattr(msg, "content", "")
                t_id = getattr(msg, "tool_call_id", "")
                t_name = call_id_to_name.get(t_id)
                summary = str(content)[:200]

                for i, inv in enumerate(tool_invocations):
                    if inv.tool_name == t_name and inv.result_summary == "":
                        tool_invocations[i] = AgentToolInvocation(
                            tool_name=inv.tool_name,
                            tool_args=inv.tool_args,
                            result_summary=summary,
                            order=inv.order,
                        )
                        break

                steps.append(
                    AgentStep(
                        step_type="tool_result",
                        content=summary,
                        tool_name=t_name,
                        order=step_order,
                    )
                )
                step_order += 1

            else:
                tool_calls = getattr(msg, "tool_calls", None) or []
                _process_tool_calls(tool_calls)

        tool_names = [inv.tool_name for inv in tool_invocations]
        unique_tools = list(dict.fromkeys(tool_names))
        if tool_invocations:
            step_summary = f"{len(tool_invocations)} tool call(s): {', '.join(unique_tools)}"
        else:
            step_summary = "No tool calls"

        pd = PipelineDetails(
            pipeline_type="agent",
            agent_tool_invocations=tuple(tool_invocations),
            agent_steps=tuple(steps),
            step_summary=step_summary,
        )

        return AgentResult(
            answer=answer,
            entries=tuple(dict(e) for e in entries) if entries else (),
            sources=tuple(sources),
            search_modes_used=tuple(search_modes_used),
            reasoning="",
            pipeline_details=pd,
        )


__all__ = [
    "AGENT_SYSTEM_PROMPT",
    "AgentExecutor",
    "AgentResult",
]
