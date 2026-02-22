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
   :func:`get_chat_completion` : Direct chat completion requests (LiteLLM-based)
   :mod:`configs.config` : Provider configuration management
"""

import warnings

# Suppress Pydantic serialization warnings from LiteLLM response types.
# LiteLLM's response objects have model_dump() but non-standard schemas that
# cause harmless Pydantic validation warnings. Applied early to catch all cases.
warnings.filterwarnings(
    "ignore",
    message=r"Pydantic serializer warnings.*",
    category=UserWarning,
)

from .completion import aget_chat_completion, get_chat_completion  # noqa: E402
from .logging import set_api_call_context  # noqa: E402
from .messages import ChatCompletionRequest, ChatMessage  # noqa: E402
from .provider_registry import ProviderRegistry, get_provider_registry  # noqa: E402

__all__ = [
    "ChatCompletionRequest",
    "ChatMessage",
    "ProviderRegistry",
    "aget_chat_completion",
    "get_chat_completion",
    "get_provider_registry",
    "set_api_call_context",
]
