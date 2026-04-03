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
from .core.models import (
    ChannelFinderResult,
    ChannelInfo,
)
from .databases import FlatChannelDatabase, HierarchicalChannelDatabase, TemplateChannelDatabase

__version__ = "2.0.0"

__all__ = [
    "FlatChannelDatabase",
    "TemplateChannelDatabase",
    "HierarchicalChannelDatabase",
    "ChannelFinderResult",
    "ChannelInfo",
    "ChannelFinderError",
    "PipelineModeError",
    "DatabaseLoadError",
    "ConfigurationError",
    "HierarchicalNavigationError",
    "QueryProcessingError",
]
