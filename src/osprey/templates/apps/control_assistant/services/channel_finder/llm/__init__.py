"""LLM client for multi-provider model access.

This module re-exports Osprey's completion interface for backward compatibility.
New code should import directly from osprey.models.completion.
"""

from osprey.models.completion import get_chat_completion

__all__ = ["get_chat_completion"]
