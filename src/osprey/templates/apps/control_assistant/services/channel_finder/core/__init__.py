"""
Core abstractions and shared models for Channel Finder.

This module provides the base interfaces and data structures used across
all pipeline implementations.
"""

from .exceptions import (
    ChannelFinderError,
    PipelineModeError,
    DatabaseLoadError,
    ConfigurationError,
    HierarchicalNavigationError,
    QueryProcessingError
)
from .base_pipeline import BasePipeline
from .base_database import BaseDatabase
from .models import (
    QuerySplitterOutput,
    ChannelMatchOutput,
    ChannelCorrectionOutput,
    ChannelInfo,
    ChannelFinderResult
)

__all__ = [
    # Exceptions
    'ChannelFinderError',
    'PipelineModeError',
    'DatabaseLoadError',
    'ConfigurationError',
    'HierarchicalNavigationError',
    'QueryProcessingError',
    # Base classes
    'BasePipeline',
    'BaseDatabase',
    # Models
    'QuerySplitterOutput',
    'ChannelMatchOutput',
    'ChannelCorrectionOutput',
    'ChannelInfo',
    'ChannelFinderResult',
]

