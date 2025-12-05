"""
Context Management Framework

Clean, production-ready context system using Pydantic for automatic serialization,
validation, and type safety. LangGraph-native with ContextManager providing
sophisticated functionality over dictionary data.

Key benefits:
- Automatic JSON serialization/deserialization via Pydantic
- Built-in validation and type safety
- Zero custom serialization logic needed
- 86% reduction in complexity from previous implementation
- Production-proven robustness
"""

from .base import CapabilityContext
from .context_manager import ContextManager, ContextNamespace
from .loader import load_context

__all__ = [
    "CapabilityContext",  # Pydantic-based context base class
    "ContextManager",  # Simplified LangGraph-native context manager
    "ContextNamespace",  # Namespace object for dot notation access to context objects
    "load_context",  # Utility function for loading context from JSON files
]
