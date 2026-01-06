"""AI Model Provider Adapters.

This module provides a unified registry-based system for AI model providers,
centralizing all provider logic for both PydanticAI models and direct API calls.
"""

from .base import BaseProvider

__all__ = ["BaseProvider"]
