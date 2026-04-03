"""Anthropic-to-OpenAI protocol translation proxy for Claude Code.

When ``claude_code.provider`` is set to an OpenAI-compatible backend (e.g.
Stanford AI Playground), Claude Code cannot communicate directly because it
speaks the Anthropic Messages API.  This package provides a lightweight local
proxy that accepts Anthropic-format requests, translates them via LiteLLM,
and returns Anthropic-format responses.
"""

from osprey.infrastructure.proxy.lifecycle import (
    is_proxy_needed,
    start_proxy,
    stop_proxy,
)

__all__ = ["is_proxy_needed", "start_proxy", "stop_proxy"]
