"""Framework Base Module - Error Classification and Handling.

This module provides the foundational error types for the Osprey Framework.
"""

from .errors import ErrorSeverity, ExecutionError

__all__ = [
    "ErrorSeverity",
    "ExecutionError",
]
