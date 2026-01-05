"""Unified LLM Model Management Framework.

This module provides a comprehensive interface for LLM model access across 100+ providers
via LiteLLM. It supports direct chat completion requests with advanced features including
extended thinking, structured outputs, and automatic TypedDict to Pydantic conversion.

Key features:
- Direct inference via LiteLLM (100+ provider support)
- Extended thinking for Anthropic and Google models
- Structured output generation with Pydantic models or TypedDict
- HTTP proxy support via environment variables
- Automatic provider configuration loading

.. seealso::
   :func:`get_chat_completion` : Direct chat completion requests
   :mod:`configs.config` : Provider configuration management
"""

from .completion import get_chat_completion
from .logging import set_api_call_context

__all__ = ["get_chat_completion", "set_api_call_context"]
