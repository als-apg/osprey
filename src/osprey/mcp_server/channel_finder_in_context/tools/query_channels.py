"""MCP tool: query_channels -- answer a natural-language channel query via an inner LLM."""

from __future__ import annotations

import logging
from typing import TypedDict

import litellm

from osprey.mcp_server.channel_finder_in_context.server import mcp
from osprey.mcp_server.channel_finder_in_context.server_context import get_cf_ic_context
from osprey.models import ChatCompletionRequest, ChatMessage, aget_chat_completion
from osprey.services.channel_finder.rate_limiter import get_rate_limiter

logger = logging.getLogger("osprey.mcp_server.channel_finder_in_context.tools.query_channels")


class QueryChannelsResult(TypedDict):
    """Structured tool result for query_channels.

    Returned as fastmcp ``structuredContent`` so programmatic callers (the
    benchmark harness) can read token counts without the LLM-friendly text
    becoming unparseable. LLM callers see the same payload as JSON text and
    handle it as a normal tool result.
    """

    text: str
    input_tokens: int
    output_tokens: int


def _safe_token_count(model: str, text: str) -> int:
    """Tokenize ``text`` for ``model``; return 0 if litellm has no tokenizer for it."""
    try:
        return int(litellm.token_counter(model=model, text=text))
    except Exception:  # noqa: BLE001
        return 0


@mcp.tool()
async def query_channels(query: str) -> QueryChannelsResult:
    """Answer a natural-language question about control-system channels.

    Calls an inner LLM with the full channel database in its context.
    The answer is returned in ``text`` with the inner LLM's <final>...</final>
    tag wrapping its final answer per its system prompt. ``input_tokens``
    and ``output_tokens`` are tokenizer-estimated counts (not the upstream
    API's billed counts, which most providers don't return on this code path);
    they are used by the benchmark harness to infer cost via list price.

    Args:
        query: Natural-language question about channels or PV addresses.
    """
    ctx = get_cf_ic_context()

    limiter = get_rate_limiter()
    if limiter is not None:
        await limiter.acquire()

    req = ChatCompletionRequest(
        messages=[
            ChatMessage(role="system", content=ctx.system_prompt_with_db),
            ChatMessage(role="user", content=query),
        ]
    )

    # Common token-accounting prefix for both happy-path and error returns.
    user_query_tokens = _safe_token_count(ctx.subagent_model_id, query)
    input_tokens = ctx.system_prompt_input_tokens + user_query_tokens

    try:
        text = await aget_chat_completion(
            provider=ctx.subagent_provider,
            model_id=ctx.subagent_model_id,
            chat_request=req,
            max_tokens=4096,
            temperature=0.0,
        )
    except litellm.ContextWindowExceededError:
        logger.warning(
            "query_channels: context window exceeded for model %s", ctx.subagent_model_id
        )
        return QueryChannelsResult(
            text="ERROR: context_window_exceeded",
            input_tokens=input_tokens,
            output_tokens=0,
        )
    except litellm.BadRequestError as exc:
        # Some upstream gateways (e.g. Bedrock via OpenAI-compatible proxies)
        # surface context-window overflow as a generic BadRequestError that
        # only mentions ContextWindowExceededError in the message string.
        # ContextWindowExceededError IS a BadRequestError subclass, but the
        # mapping in litellm.exception_mapping_utils may not always reclassify
        # the wrapped error — fall back to substring matching here so the
        # caller still gets the friendly ERROR string instead of a traceback.
        if "ContextWindowExceededError" in str(exc) or "context window" in str(exc).lower():
            logger.warning(
                "query_channels: context window exceeded (wrapped BadRequestError) for model %s",
                ctx.subagent_model_id,
            )
            return QueryChannelsResult(
                text="ERROR: context_window_exceeded",
                input_tokens=input_tokens,
                output_tokens=0,
            )
        raise
    except litellm.RateLimitError:
        logger.warning("query_channels: rate limit hit for provider %s", ctx.subagent_provider)
        return QueryChannelsResult(
            text="ERROR: rate_limited",
            input_tokens=input_tokens,
            output_tokens=0,
        )

    output_tokens = _safe_token_count(ctx.subagent_model_id, text)
    return QueryChannelsResult(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
