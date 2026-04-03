"""
Core abstractions and shared models for Channel Finder.

This module provides the base interfaces and data structures used across
channel finder components.
"""

from .base_database import BaseDatabase
from .exceptions import (
    ChannelFinderError,
    ConfigurationError,
    DatabaseLoadError,
    HierarchicalNavigationError,
    PipelineModeError,
    QueryProcessingError,
)
from .models import (
    ChannelFinderResult,
    ChannelInfo,
)

__all__ = [
    # Exceptions
    "ChannelFinderError",
    "PipelineModeError",
    "DatabaseLoadError",
    "ConfigurationError",
    "HierarchicalNavigationError",
    "QueryProcessingError",
    # Base classes
    "BaseDatabase",
    # Models
    "ChannelInfo",
    "ChannelFinderResult",
]
