"""Unified LLM Model Management Framework.

This module provides a comprehensive interface for creating and managing LLM model instances
across multiple providers (Anthropic, OpenAI, Google, Ollama, CBORG). It maintains clean
separation between structured model instance creation for PydanticAI agents and direct
chat completion requests for immediate inference needs.

The module supports advanced features including:
- HTTP proxy configuration for enterprise environments
- Timeout management and connection pooling
- TypedDict to Pydantic model conversion for structured outputs
- Extended thinking capabilities for Anthropic and Google models
- Automatic provider configuration loading from config system

.. note::
   Model instances created by :func:`get_model` are designed for use with PydanticAI
   agents and structured generation workflows, while :func:`get_chat_completion`
   provides direct access to model inference for simpler use cases.

.. seealso::
   :func:`get_model` : Create model instances for structured generation
   :func:`get_chat_completion` : Direct chat completion requests
   :mod:`configs.config` : Provider configuration management
   :doc:`/developer-guides/01_understanding-the-framework/02_convention-over-configuration` : Model setup and configuration guide
"""

from .completion import get_chat_completion
from .factory import get_model
from .logging import set_api_call_context

__all__ = ["get_model", "get_chat_completion", "set_api_call_context"]
