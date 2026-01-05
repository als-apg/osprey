"""AI Model Provider Adapters.

This module provides a unified registry-based system for AI model providers,
centralizing all provider logic through the LiteLLM adapter layer.
"""

from .base import BaseProvider

__all__ = ["BaseProvider"]
