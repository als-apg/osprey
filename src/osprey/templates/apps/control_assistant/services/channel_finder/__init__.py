"""
Channel Finder - Generic In-Context Retrieval System

A facility-agnostic system for finding control system channels using natural language queries.
Configure which facility to use in config.yml.
"""

from .core.exceptions import (
    ChannelFinderError,
    ConfigurationError,
    DatabaseLoadError,
    HierarchicalNavigationError,
    PipelineModeError,
    QueryProcessingError,
)

# Core models and base classes
from .core.models import (
    ChannelCorrectionOutput,
    ChannelFinderResult,
    ChannelInfo,
    ChannelMatchOutput,
    QuerySplitterOutput,
)

# Databases
from .databases import HierarchicalChannelDatabase, LegacyChannelDatabase, TemplateChannelDatabase

# Pipelines
from .pipelines.in_context import InContextPipeline

# Service (high-level interface)
from .service import ChannelFinderService

__version__ = "2.0.0"

__all__ = [
    # Service (high-level interface)
    "ChannelFinderService",
    # Pipelines
    "InContextPipeline",
    # Database classes
    "TemplateChannelDatabase",
    "LegacyChannelDatabase",
    "HierarchicalChannelDatabase",
    # Data models
    "QuerySplitterOutput",
    "ChannelMatchOutput",
    "ChannelCorrectionOutput",
    "ChannelFinderResult",
    "ChannelInfo",
    # Exceptions
    "ChannelFinderError",
    "PipelineModeError",
    "DatabaseLoadError",
    "ConfigurationError",
    "HierarchicalNavigationError",
    "QueryProcessingError",
]
