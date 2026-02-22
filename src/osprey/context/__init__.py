"""
Context Management Framework

Clean, production-ready context system using Pydantic for automatic serialization,
validation, and type safety.

Key benefits:
- Automatic JSON serialization/deserialization via Pydantic
- Built-in validation and type safety
- Zero custom serialization logic needed
"""

from .base import CapabilityContext
from .loader import load_context

__all__ = [
    "CapabilityContext",  # Pydantic-based context base class
    "load_context",  # Utility function for loading context from JSON files
]
