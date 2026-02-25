"""Self-contained ReAct agent executor for ARIEL search.

Implements a ReAct (Reason + Act) loop using Osprey's built-in
completion API (LiteLLM tool-calling) — no LangGraph or LangChain required.

Search tools are auto-discovered from the Osprey registry via
SearchToolDescriptor, same as the previous LangGraph-based agent.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from osprey.services.ariel_search.exceptions import SearchTimeoutError
from osprey.services.ariel_search.models import (
    AgentStep,
    AgentToolInvocation,
    DiagnosticLevel,
    EnhancedLogbookEntry,
    PipelineDetails,
    SearchDiagnostic,
    SearchMode,
    resolve_time_range,
)
from osprey.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from osprey.models.embeddings.base import BaseEmbeddingProvider
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.database.repository import ARIELRepository
    from osprey.services.ariel_search.search.base import SearchToolDescriptor

logger = get_logger("ariel")


AGENT_SYSTEM_PROMPT = """\
You are ARIEL, an AI assistant for searching and analyzing facility logbook entries.

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
- If nothing is found: clearly state that no relevant information was found
"""


@dataclass(frozen=True)
class AgentResult:
    """Result from agent execution."""

    answer: str | None = None
    entries: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    sources: tuple[str, ...] = field(default_factory=tuple)
    search_modes_used: tuple[SearchMode, ...] = field(default_factory=tuple)
    reasoning: str = ""
    diagnostics: tuple[SearchDiagnostic, ...] = field(default_factory=tuple)
    pipeline_details: PipelineDetails | None = None


def _descriptor_to_openai_tool(descriptor: SearchToolDescriptor) -> dict[str, Any]:
    """Convert a SearchToolDescriptor to an OpenAI-format tool definition."""
    schema = descriptor.args_schema.model_json_schema()

    # Strip pydantic metadata keys the LLM API doesn't need
    properties = {}
    for name, prop in schema.get("properties", {}).items():
        clean = {k: v for k, v in prop.items() if k not in ("title",)}
        if "anyOf" in clean:
            types = [t.get("type") for t in clean["anyOf"] if "type" in t]
            if types:
                clean["type"] = types[0]
            clean.pop("anyOf", None)
        properties[name] = clean

    required = schema.get("required", [])

    return {
        "type": "function",
        "function": {
            "name": descriptor.name,
            "description": descriptor.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


class AgentExecutor:
    """ReAct agent using Osprey's native completion API with tool calling.

    Usage:
        executor = AgentExecutor(repository, config, embedder_loader)
        result = await executor.execute("What happened yesterday?")
    """

    def __init__(
        self,
        repository: ARIELRepository,
        config: ARIELConfig,
        embedder_loader: Callable[[], BaseEmbeddingProvider],
        system_prompt: str | None = None,
    ) -> None:
        self.repository = repository
        self.config = config
        self._embedder_loader = embedder_loader
        self._system_prompt = system_prompt or AGENT_SYSTEM_PROMPT

    def _load_descriptors(self) -> list[SearchToolDescriptor]:
        """Load tool descriptors from enabled search modules via the registry."""
        from osprey.registry import get_registry

        registry = get_registry()
        descriptors: list[SearchToolDescriptor] = []

        retrieval_modules = self.config.get_pipeline_retrieval_modules("agent")
        for module_name in retrieval_modules:
            if not self.config.is_search_module_enabled(module_name):
                continue
            module = registry.get_ariel_search_module(module_name)
            if module is not None:
                descriptors.append(module.get_tool_descriptor())
        return descriptors

    async def _call_tool(
        self,
        descriptor: SearchToolDescriptor,
        args: dict[str, Any],
        time_range: tuple[datetime, datetime] | None,
        collected_ids: list[str],
    ) -> list[dict[str, Any]]:
        """Execute a single tool call and return formatted results."""
        start_date = args.pop("start_date", None)
        end_date = args.pop("end_date", None)
        resolved_start, resolved_end = resolve_time_range(start_date, end_date, time_range)

        call_kwargs: dict[str, Any] = {
            "query": args.pop("query"),
            "repository": self.repository,
            "config": self.config,
            "start_date": resolved_start,
            "end_date": resolved_end,
            **args,
        }

        if descriptor.needs_embedder:
            call_kwargs["embedder"] = self._embedder_loader()

        results = await descriptor.execute(**call_kwargs)

        formatted = [
            descriptor.format_result(*item)
            if isinstance(item, tuple)
            else descriptor.format_result(item)
            for item in results
        ]

        for item in formatted:
            eid = item.get("entry_id")
            if eid:
                collected_ids.append(eid)

        return formatted

    async def execute(
        self,
        query: str,
        *,
        max_results: int | None = None,
        time_range: tuple[datetime, datetime] | None = None,
    ) -> AgentResult:
        """Execute the ReAct agent loop."""
        try:
            collected_ids: list[str] = []
            descriptors = self._load_descriptors()

            if not descriptors:
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

            result = await self._run_react_loop(
                query,
                descriptors,
                time_range,
                collected_ids,
            )

            unique_ids = list(dict.fromkeys(collected_ids))
            entries: list[EnhancedLogbookEntry] = []
            if unique_ids:
                try:
                    entries = await self.repository.get_entries_by_ids(unique_ids)
                except Exception:
                    logger.warning(
                        "Failed to fetch full entries for agent results",
                        exc_info=True,
                    )

            return self._build_result(result, descriptors, entries)

        except SearchTimeoutError:
            raise
        except Exception as e:
            logger.exception(f"Agent execution failed: {e}")
            raise

    async def _run_react_loop(
        self,
        query: str,
        descriptors: list[SearchToolDescriptor],
        time_range: tuple[datetime, datetime] | None,
        collected_ids: list[str],
    ) -> dict[str, Any]:
        """Run the ReAct loop using Osprey's completion API with tools.

        Returns a dict with 'answer', 'tool_invocations', and 'steps'.
        """
        from osprey.models.completion import get_chat_completion
        from osprey.models.messages import ChatCompletionRequest, ChatMessage

        descriptor_map = {d.name: d for d in descriptors}
        openai_tools = [_descriptor_to_openai_tool(d) for d in descriptors]

        messages = ChatCompletionRequest(
            messages=[
                ChatMessage(role="system", content=self._system_prompt),
                ChatMessage(role="user", content=query),
            ]
        )

        tool_invocations: list[AgentToolInvocation] = []
        steps: list[AgentStep] = [
            AgentStep(step_type="user_query", content=query[:200], order=0),
        ]
        search_modes_used: list[SearchMode] = []
        step_order = 1
        tool_call_order = 0

        max_iterations = self.config.reasoning.max_iterations
        timeout = self.config.reasoning.total_timeout_seconds

        provider = self.config.reasoning.provider
        model_id = self.config.reasoning.model_id
        temperature = self.config.reasoning.temperature

        async def _one_iteration() -> str | None:
            """Run one LLM call; returns final answer text or None if tools were called."""
            nonlocal step_order, tool_call_order

            response = await asyncio.to_thread(
                get_chat_completion,
                chat_request=messages,
                provider=provider,
                model_id=model_id,
                temperature=temperature,
                max_tokens=4096,
                tools=openai_tools,
                tool_choice="auto",
            )

            # Tool calls come back as a list of dicts
            if isinstance(response, list):
                tool_calls = response

                # Build assistant message with tool_calls for the conversation
                assistant_tool_calls = []
                for tc in tool_calls:
                    tc_id = tc.get("id", f"call_{uuid.uuid4().hex[:8]}")
                    fn = tc.get("function", {})
                    assistant_tool_calls.append(
                        {
                            "id": tc_id,
                            "type": "function",
                            "function": {
                                "name": fn.get("name", ""),
                                "arguments": fn.get("arguments", "{}"),
                            },
                        }
                    )

                messages.messages.append(
                    ChatMessage(role="assistant", tool_calls=assistant_tool_calls)
                )

                for tc in tool_calls:
                    tc_id = tc.get("id", f"call_{uuid.uuid4().hex[:8]}")
                    fn = tc.get("function", {})
                    tool_name = fn.get("name", "unknown")
                    raw_args = fn.get("arguments", "{}")
                    try:
                        tool_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except json.JSONDecodeError:
                        tool_args = {}

                    steps.append(
                        AgentStep(
                            step_type="tool_call",
                            content=str(tool_args)[:200],
                            tool_name=tool_name,
                            order=step_order,
                        )
                    )
                    step_order += 1

                    desc = descriptor_map.get(tool_name)
                    if desc is None:
                        result_text = f"Unknown tool: {tool_name}"
                    else:
                        try:
                            results = await self._call_tool(
                                desc,
                                dict(tool_args),
                                time_range,
                                collected_ids,
                            )
                            result_text = json.dumps(results, default=str)[:4000]
                        except Exception as exc:
                            logger.warning(f"Tool {tool_name} failed: {exc}")
                            result_text = f"Error: {exc}"

                        mode = desc.search_mode
                        if mode not in search_modes_used:
                            search_modes_used.append(mode)

                    inv = AgentToolInvocation(
                        tool_name=tool_name,
                        tool_args=tool_args,
                        result_summary=result_text[:200],
                        order=tool_call_order,
                    )
                    tool_invocations.append(inv)
                    tool_call_order += 1

                    steps.append(
                        AgentStep(
                            step_type="tool_result",
                            content=result_text[:200],
                            tool_name=tool_name,
                            order=step_order,
                        )
                    )
                    step_order += 1

                    messages.messages.append(
                        ChatMessage(
                            role="tool",
                            content=result_text,
                            tool_call_id=tc_id,
                            name=tool_name,
                        )
                    )

                return None  # more iterations needed

            # Plain text response — final answer
            answer = str(response)
            steps.append(
                AgentStep(
                    step_type="final_answer",
                    content=answer[:200],
                    order=step_order,
                )
            )
            step_order += 1

            messages.messages.append(ChatMessage(role="assistant", content=answer))
            return answer

        try:
            answer = None
            for _i in range(max_iterations):
                result = await asyncio.wait_for(
                    _one_iteration(),
                    timeout=timeout,
                )
                if result is not None:
                    answer = result
                    break
            else:
                # Exhausted iterations — ask for a final answer without tools
                final_response = await asyncio.to_thread(
                    get_chat_completion,
                    chat_request=messages,
                    provider=provider,
                    model_id=model_id,
                    temperature=temperature,
                    max_tokens=4096,
                )
                answer = str(final_response)
                steps.append(
                    AgentStep(
                        step_type="final_answer",
                        content=answer[:200],
                        order=step_order,
                    )
                )

        except TimeoutError as err:
            raise SearchTimeoutError(
                message=f"Agent execution timed out after {timeout}s",
                timeout_seconds=timeout,
                operation="agent execution",
            ) from err

        tool_names = [inv.tool_name for inv in tool_invocations]
        unique_tools = list(dict.fromkeys(tool_names))
        step_summary = (
            f"{len(tool_invocations)} tool call(s): {', '.join(unique_tools)}"
            if tool_invocations
            else "No tool calls"
        )

        return {
            "answer": answer,
            "tool_invocations": tool_invocations,
            "steps": steps,
            "search_modes_used": search_modes_used,
            "step_summary": step_summary,
        }

    def _build_result(
        self,
        raw: dict[str, Any],
        descriptors: list[SearchToolDescriptor],
        entries: list[EnhancedLogbookEntry] | None = None,
    ) -> AgentResult:
        """Build AgentResult from the loop output."""
        answer = raw.get("answer")
        tool_invocations = raw.get("tool_invocations", [])
        steps = raw.get("steps", [])
        search_modes_used = raw.get("search_modes_used", [])
        step_summary = raw.get("step_summary", "")

        sources: list[str] = []
        if answer and entries:
            sources = [
                e["entry_id"] for e in entries if e.get("entry_id") and e["entry_id"] in answer
            ]

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


__all__ = ["AGENT_SYSTEM_PROMPT", "AgentExecutor", "AgentResult"]
