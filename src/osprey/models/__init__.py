"""Unified LLM Model Management Framework.

This module provides a comprehensive interface for LLM model access across 100+ providers
via LiteLLM. It supports direct chat completion requests with advanced features including
extended thinking, structured outputs, and automatic TypedDict to Pydantic conversion.

Key features:
- Direct inference via LiteLLM (100+ provider support)
- LangChain model factory for LangGraph integration
- Extended thinking for Anthropic and Google models
- Structured output generation with Pydantic models or TypedDict
- HTTP proxy support via environment variables
- Automatic provider configuration loading

.. seealso::
   :func:`get_chat_completion` : Direct chat completion requests (LiteLLM-based)
   :func:`get_langchain_model` : LangChain model factory for LangGraph
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

from .completion import get_chat_completion  # noqa: E402
from .langchain import (  # noqa: E402
    SUPPORTED_PROVIDERS,
    get_langchain_model,
    get_langchain_model_from_name,
    list_supported_providers,
)
from .logging import set_api_call_context  # noqa: E402

__all__ = [
    "get_chat_completion",
    "get_langchain_model",
    "get_langchain_model_from_name",
    "list_supported_providers",
    "set_api_call_context",
    "SUPPORTED_PROVIDERS",
]
