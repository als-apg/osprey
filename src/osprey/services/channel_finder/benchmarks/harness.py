"""ReAct agent harness for benchmarking LiteLLM-supported models via MCP.

Provides a ReAct agent loop that connects to OSPREY's channel-finder MCP
servers via fastmcp 3.x and uses ``litellm.acompletion()`` directly for
LLM calls.  This enables benchmarking any LiteLLM-supported model
(including local Ollama models) against the same evaluation pipeline used
by the SDK benchmark runner.

Key design decisions:
- Calls ``litellm.acompletion()`` directly (the Osprey adapter silently
  drops tools for Ollama providers).
- Uses ``fastmcp.Client`` with ``StdioTransport`` for MCP communication.
- Reuses ``ToolTrace`` from ``sdk.py`` for result format compatibility.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import litellm
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from osprey.services.channel_finder.benchmarks.sdk import ToolTrace
from osprey.services.channel_finder.rate_limiter import (
    get_rate_limiter,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


OLLAMA_OPENAI_BASE = os.environ.get("OLLAMA_OPENAI_BASE", "http://localhost:11434/v1")


def _litellm_call_kwargs(model: str) -> dict:
    """Resolve LiteLLM routing for the benchmark model string.

    Returns a kwargs dict with ``model`` and optionally ``api_base`` /
    ``api_key``.  ``ollama/*`` and ``ollama_chat/*`` are routed through
    Ollama's OpenAI-compatible endpoint (``/v1/chat/completions``) via
    LiteLLM's ``openai/*`` provider.  This bypasses LiteLLM's buggy
    ``ollama_chat`` translation which silently drops ``tool_calls`` on
    certain multi-turn responses (BerriAI/litellm #11104, #5245, #24091).
    """
    if model.startswith("ollama/") or model.startswith("ollama_chat/"):
        name = model.split("/", 1)[1]
        return {
            "model": f"openai/{name}",
            "api_base": OLLAMA_OPENAI_BASE,
            "api_key": "ollama-local",
        }
    return {"model": model}


def _litellm_model_name(model: str) -> str:
    """Return the LiteLLM model string for legacy callers."""
    return _litellm_call_kwargs(model)["model"]


def _mcp_tool_to_openai(tool: Any) -> dict:
    """Convert an MCP ``Tool`` object to OpenAI function-calling format."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema,
        },
    }


# ---------------------------------------------------------------------------
# Pre-flight validation
# ---------------------------------------------------------------------------


async def preflight_checks(model: str) -> None:
    """Validate environment before running benchmark queries.

    Checks:
    1. Ollama reachability (if using an ``ollama/`` model).
    2. Tool-calling support for the remapped model name.
    3. Availability of an API key for the LLM judge.

    Raises:
        RuntimeError: If Ollama is required but not reachable.
    """
    import httpx

    # 1. Check Ollama reachability for ollama/* models
    if model.startswith("ollama/"):
        tags_url = OLLAMA_OPENAI_BASE.replace("/v1", "") + "/api/tags"
        try:
            async with httpx.AsyncClient() as http:
                resp = await http.get(tags_url, timeout=5.0)
                resp.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                f"Ollama is not reachable at {tags_url} — is it running?  (error: {exc})"
            ) from exc

    # 2. Check tool-calling support after remapping
    remapped = _litellm_model_name(model)
    try:
        supported = litellm.get_supported_openai_params(model=remapped)
        if supported is not None and "tools" not in supported:
            logger.warning(
                "Model %r may not support tool calling — 'tools' not in supported OpenAI params.",
                remapped,
            )
    except Exception:
        logger.warning(
            "Could not determine supported params for model %r; proceeding anyway.",
            remapped,
        )

    # 3. Check for LLM judge API key
    if "ANTHROPIC_API_KEY" not in os.environ and "CBORG_API_KEY" not in os.environ:
        logger.warning(
            "Neither ANTHROPIC_API_KEY nor CBORG_API_KEY found in environment — "
            "LLM judge evaluation will not be available."
        )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ReactWorkflowResult:
    """Aggregated result from a ReAct agent query run.

    Mirrors the ``SDKWorkflowResult`` interface but stores values directly
    (no SDK ``ResultMessage`` dependency).
    """

    tool_traces: list[ToolTrace] = field(default_factory=list)
    text_blocks: list[str] = field(default_factory=list)
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    num_turns: int = 0


# ---------------------------------------------------------------------------
# MCP client session
# ---------------------------------------------------------------------------


@asynccontextmanager
async def mcp_client_session(
    project_dir: Path,
    paradigm: str,
    env: dict[str, str] | None = None,
):
    """Async context manager that yields a connected fastmcp ``Client``.

    Spawns the channel-finder MCP server for the given *paradigm*
    (``in_context``, ``hierarchical``, or ``middle_layer``) as a
    subprocess and connects via stdio transport.

    Args:
        env: Extra environment variables merged on top of the default
            ``{"OSPREY_CONFIG": ...}`` dict. Caller-supplied keys win on
            collision.
    """
    merged_env = {"OSPREY_CONFIG": str(project_dir / "config.yml")} | (env or {})
    transport = StdioTransport(
        command=sys.executable,
        args=["-m", f"osprey.mcp_server.channel_finder_{paradigm}"],
        env=merged_env,
        cwd=str(project_dir),
    )
    client = Client(transport=transport)
    async with client:
        yield client


# ---------------------------------------------------------------------------
# Core ReAct loop
# ---------------------------------------------------------------------------


async def run_react_query(
    client: Client,
    prompt: str,
    model: str,
    system_prompt: str | None,
    max_turns: int = 25,
    timeout: float = 300,
    call_kwargs_override: dict | None = None,
) -> ReactWorkflowResult:
    """Execute a ReAct agent loop against an MCP server.

    Discovers available tools from *client*, then iteratively calls
    ``litellm.acompletion()`` with tool-calling enabled until the model
    produces a final text response or *max_turns* is exhausted.

    Args:
        client: A connected fastmcp ``Client``.
        prompt: The user query to answer.
        model: LiteLLM model identifier (e.g. ``ollama/llama3.1``).
        system_prompt: Optional system prompt for the model.
        max_turns: Maximum number of tool-calling turns.
        timeout: Per-call timeout in seconds.
        call_kwargs_override: Optional dict merged into the LiteLLM call
            kwargs after the base routing is resolved. Used by backends
            that need to pin ``api_base`` / ``api_key`` (e.g. CBORG).

    Returns:
        A ``ReactWorkflowResult`` with tool traces, text blocks, and
        token/cost aggregates.
    """
    result = ReactWorkflowResult()
    call_kwargs = _litellm_call_kwargs(model)
    if call_kwargs_override:
        call_kwargs.update(call_kwargs_override)

    # 1. Discover tools from MCP server
    mcp_tools = await client.list_tools()
    openai_tools = [_mcp_tool_to_openai(t) for t in mcp_tools]

    # 2. Build initial messages
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # 3. ReAct loop
    for _turn in range(max_turns):
        # Throttle if a rate limiter is armed (CBORG free tier ≈ 20 req/min)
        limiter = get_rate_limiter()
        if limiter is not None:
            await limiter.acquire()

        try:
            response = await asyncio.wait_for(
                litellm.acompletion(
                    **call_kwargs,
                    messages=messages,
                    tools=openai_tools if openai_tools else None,
                    tool_choice="auto" if openai_tools else None,
                    # Let LiteLLM handle stragglers if the throttle still misses.
                    num_retries=3,
                ),
                timeout=timeout,
            )
        except TimeoutError as exc:
            # Re-raise so the runner counts this as num_failed instead of
            # silently treating it as a low-F1 query.
            logger.warning("LLM call timed out after %.0fs on turn %d", timeout, _turn + 1)
            raise RuntimeError(f"ReAct query aborted: timeout on turn {_turn + 1}") from exc
        except Exception as exc:
            # Re-raise with context. Includes RateLimitError, ContextWindowExceededError, etc.
            logger.error("LLM call failed on turn %d: %s", _turn + 1, exc)
            raise RuntimeError(
                f"ReAct query aborted: {type(exc).__name__} on turn {_turn + 1}"
            ) from exc

        choice = response.choices[0]
        msg = choice.message

        # Aggregate token usage
        usage = getattr(response, "usage", None)
        if usage is not None:
            result.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
            result.output_tokens += getattr(usage, "completion_tokens", 0) or 0

        # Aggregate cost
        hidden = getattr(response, "_hidden_params", None)
        if hidden is not None:
            result.cost_usd += hidden.get("response_cost", 0.0) or 0.0

        # Thinking models (Qwen 3.5, Deepseek-R1, etc.) emit chain-of-thought
        # on .reasoning_content while leaving .content empty.  Capture it on
        # every turn so evaluation sees model-stated answers even when they
        # live in the reasoning stream.
        reasoning = (
            getattr(msg, "reasoning_content", None)
            or getattr(msg, "thinking", None)
            or getattr(msg, "reasoning", None)
            or ""
        )
        if reasoning:
            result.text_blocks.append(reasoning)

        # Check for tool calls
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            # Append assistant message with tool calls
            messages.append(msg.model_dump())

            for tool_call in tool_calls:
                fn_name = tool_call.function.name
                raw_args = tool_call.function.arguments

                # Parse arguments
                try:
                    args = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError as json_err:
                    logger.error(
                        "Malformed tool call JSON for %r: %s (raw: %r)",
                        fn_name,
                        json_err,
                        raw_args,
                    )
                    error_msg = f"Error: malformed JSON arguments — {json_err}"
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": error_msg,
                        }
                    )
                    result.tool_traces.append(
                        ToolTrace(name=fn_name, input={}, result=error_msg, is_error=True)
                    )
                    continue

                # Execute tool via MCP
                try:
                    tool_result = await client.call_tool(fn_name, args)
                    tool_result_str = str(tool_result)
                except Exception as tool_err:
                    logger.error("Tool call %r failed: %s", fn_name, tool_err)
                    tool_result_str = f"Error: tool '{fn_name}' failed — {tool_err}"
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result_str,
                        }
                    )
                    result.tool_traces.append(
                        ToolTrace(
                            name=fn_name,
                            input=args,
                            result=tool_result_str,
                            is_error=True,
                        )
                    )
                    continue

                # Success — record result
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result_str,
                    }
                )
                result.tool_traces.append(
                    ToolTrace(name=fn_name, input=args, result=tool_result_str)
                )

            result.num_turns += 1

        else:
            # No tool calls — final text response
            content = getattr(msg, "content", None) or ""
            if content:
                result.text_blocks.append(content)
            break

        # Also break on explicit stop
        if choice.finish_reason == "stop" and not tool_calls:
            break

    return result


# ---------------------------------------------------------------------------
# Text extraction helper
# ---------------------------------------------------------------------------


def combined_text_from_react(result: ReactWorkflowResult) -> str:
    """Combine all text blocks and tool results into a single searchable string.

    Mirrors :func:`sdk.combined_text` for result format compatibility.
    """
    parts = list(result.text_blocks)
    for trace in result.tool_traces:
        if trace.result:
            parts.append(trace.result)
    return " ".join(parts).lower()
