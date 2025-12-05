"""
Core abstractions and shared models for Channel Finder.

This module provides the base interfaces and data structures used across
all pipeline implementations.
"""

from .base_database import BaseDatabase
from .base_pipeline import BasePipeline
from .exceptions import (
    ChannelFinderError,
    ConfigurationError,
    DatabaseLoadError,
    HierarchicalNavigationError,
    PipelineModeError,
    QueryProcessingError,
)
from .models import (
    ChannelCorrectionOutput,
    ChannelFinderResult,
    ChannelInfo,
    ChannelMatchOutput,
    QuerySplitterOutput,
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
    "BasePipeline",
    "BaseDatabase",
    # Models
    "QuerySplitterOutput",
    "ChannelMatchOutput",
    "ChannelCorrectionOutput",
    "ChannelInfo",
    "ChannelFinderResult",
]
