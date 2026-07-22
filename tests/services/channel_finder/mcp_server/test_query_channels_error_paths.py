"""Supplemental coverage for the in-context query_channels error branches.

The happy path, ContextWindowExceededError, and RateLimitError are covered by
test_query_channels.py. This file closes the remaining branches: the wrapped
BadRequestError context-window fallback (and its re-raise for unrelated
BadRequestErrors), the rate-limiter acquire hook, and the tokenizer helper's
exception guard.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest

_MOD = "osprey.mcp_server.channel_finder_in_context.tools.query_channels"


@pytest.fixture
def mock_ctx():
    ctx = MagicMock()
    ctx.subagent_provider = "anthropic"
    ctx.subagent_model_id = "anthropic/claude-haiku"
    ctx.system_prompt_with_db = "You are a channel finder."
    ctx.system_prompt_input_tokens = 100
    return ctx


def _bad_request(message: str) -> litellm.BadRequestError:
    return litellm.BadRequestError(
        message=message, model="anthropic/claude-haiku", llm_provider="anthropic"
    )


@pytest.mark.asyncio
async def test_wrapped_bad_request_context_window_returns_friendly_error(mock_ctx):
    """A BadRequestError whose message mentions the context window is softened."""
    mock_aget = AsyncMock(side_effect=_bad_request("ContextWindowExceededError: too big"))
    with (
        patch(f"{_MOD}.get_cf_ic_context", return_value=mock_ctx),
        patch(f"{_MOD}.get_rate_limiter", return_value=None),
        patch(f"{_MOD}.aget_chat_completion", mock_aget),
    ):
        from osprey.mcp_server.channel_finder_in_context.tools.query_channels import query_channels

        result = await query_channels("find BPMs")

    assert result["text"] == "ERROR: context_window_exceeded"
    assert result["output_tokens"] == 0
    assert result["input_tokens"] >= 100


@pytest.mark.asyncio
async def test_unrelated_bad_request_is_reraised(mock_ctx):
    """A BadRequestError with no context-window signal propagates unchanged."""
    mock_aget = AsyncMock(side_effect=_bad_request("invalid parameter: temperature"))
    with (
        patch(f"{_MOD}.get_cf_ic_context", return_value=mock_ctx),
        patch(f"{_MOD}.get_rate_limiter", return_value=None),
        patch(f"{_MOD}.aget_chat_completion", mock_aget),
    ):
        from osprey.mcp_server.channel_finder_in_context.tools.query_channels import query_channels

        with pytest.raises(litellm.BadRequestError):
            await query_channels("find BPMs")


@pytest.mark.asyncio
async def test_rate_limiter_is_acquired_when_present(mock_ctx):
    """When a limiter is configured, the tool awaits acquire() before the call."""
    limiter = MagicMock()
    limiter.acquire = AsyncMock()
    mock_aget = AsyncMock(return_value="<final>PV:1</final>")
    with (
        patch(f"{_MOD}.get_cf_ic_context", return_value=mock_ctx),
        patch(f"{_MOD}.get_rate_limiter", return_value=limiter),
        patch(f"{_MOD}.aget_chat_completion", mock_aget),
    ):
        from osprey.mcp_server.channel_finder_in_context.tools.query_channels import query_channels

        result = await query_channels("beam current PV?")

    limiter.acquire.assert_awaited_once()
    assert result["text"] == "<final>PV:1</final>"


def test_safe_token_count_returns_zero_on_tokenizer_failure():
    """The tokenizer helper swallows litellm errors and reports 0 tokens."""
    from osprey.mcp_server.channel_finder_in_context.tools.query_channels import _safe_token_count

    with patch("litellm.token_counter", side_effect=Exception("no tokenizer")):
        assert _safe_token_count("mystery/model", "some text") == 0
