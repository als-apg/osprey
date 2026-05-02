"""Unit tests for query_channels MCP tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest


@pytest.fixture()
def mock_ctx():
    ctx = MagicMock()
    ctx.subagent_provider = "anthropic"
    ctx.subagent_model_id = "anthropic/claude-haiku"
    ctx.system_prompt_with_db = "You are a channel finder. <final>...</final>\nCH1 | PV:1 | desc"
    # Concrete int (not a MagicMock) so query_channels can add it to the
    # tokenized user-query count without producing a MagicMock arithmetic
    # result that breaks downstream assertions.
    ctx.system_prompt_input_tokens = 100
    return ctx


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    with patch(
        "osprey.mcp_server.channel_finder_in_context.tools.query_channels.get_rate_limiter",
        return_value=None,
    ):
        yield


class TestQueryChannelsHappyPath:
    @pytest.mark.asyncio
    async def test_returns_llm_response(self, mock_ctx):
        mock_aget = AsyncMock(return_value="<final>PV:1</final>")
        with (
            patch(
                "osprey.mcp_server.channel_finder_in_context.tools.query_channels.get_cf_ic_context",
                return_value=mock_ctx,
            ),
            patch(
                "osprey.mcp_server.channel_finder_in_context.tools.query_channels.aget_chat_completion",
                mock_aget,
            ),
        ):
            from osprey.mcp_server.channel_finder_in_context.tools.query_channels import (
                query_channels,
            )

            result = await query_channels("what is the beam current PV?")

        # query_channels returns a QueryChannelsResult TypedDict with
        # text + tokenizer-estimated input/output token counts.
        assert result["text"] == "<final>PV:1</final>"
        assert result["input_tokens"] >= 100  # at least the system-prompt base
        assert "output_tokens" in result
        mock_aget.assert_awaited_once()
        call_kwargs = mock_aget.call_args.kwargs
        assert call_kwargs["provider"] == mock_ctx.subagent_provider
        assert call_kwargs["model_id"] == mock_ctx.subagent_model_id
        chat_request = call_kwargs["chat_request"]
        assert chat_request.messages[0].role == "system"
        assert "<final>" in chat_request.messages[0].content
        assert chat_request.messages[1].role == "user"
        assert chat_request.messages[1].content == "what is the beam current PV?"


class TestQueryChannelsContextWindowExceeded:
    @pytest.mark.asyncio
    async def test_returns_error_string(self, mock_ctx):
        mock_aget = AsyncMock(
            side_effect=litellm.ContextWindowExceededError(
                message="context too long",
                model="anthropic/claude-haiku",
                llm_provider="anthropic",
            )
        )
        with (
            patch(
                "osprey.mcp_server.channel_finder_in_context.tools.query_channels.get_cf_ic_context",
                return_value=mock_ctx,
            ),
            patch(
                "osprey.mcp_server.channel_finder_in_context.tools.query_channels.aget_chat_completion",
                mock_aget,
            ),
        ):
            from osprey.mcp_server.channel_finder_in_context.tools.query_channels import (
                query_channels,
            )

            result = await query_channels("find all BPM channels")

        assert result["text"] == "ERROR: context_window_exceeded"
        assert result["output_tokens"] == 0


class TestQueryChannelsRateLimit:
    @pytest.mark.asyncio
    async def test_returns_error_string(self, mock_ctx):
        mock_aget = AsyncMock(
            side_effect=litellm.RateLimitError(
                message="rate limit hit",
                llm_provider="anthropic",
                model="anthropic/claude-haiku",
            )
        )
        with (
            patch(
                "osprey.mcp_server.channel_finder_in_context.tools.query_channels.get_cf_ic_context",
                return_value=mock_ctx,
            ),
            patch(
                "osprey.mcp_server.channel_finder_in_context.tools.query_channels.aget_chat_completion",
                mock_aget,
            ),
        ):
            from osprey.mcp_server.channel_finder_in_context.tools.query_channels import (
                query_channels,
            )

            result = await query_channels("find corrector magnets")

        assert result["text"] == "ERROR: rate_limited"
        assert result["output_tokens"] == 0
