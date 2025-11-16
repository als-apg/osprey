"""LLM client for multi-provider model access."""

from .completion import get_chat_completion, get_provider_config

__all__ = ["get_chat_completion", "get_provider_config"]

