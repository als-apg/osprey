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
    ChannelCorrectionOutput,
    ChannelFinderResult,
    ChannelInfo,
    ChannelMatchOutput,
    QuerySplitterOutput,
)
from .databases import FlatChannelDatabase, HierarchicalChannelDatabase, TemplateChannelDatabase
from .pipelines.in_context import InContextPipeline
from .service import ChannelFinderService

__version__ = "2.0.0"

__all__ = [
    "ChannelFinderService",
    "InContextPipeline",
    "FlatChannelDatabase",
    "TemplateChannelDatabase",
    "HierarchicalChannelDatabase",
    "QuerySplitterOutput",
    "ChannelMatchOutput",
    "ChannelCorrectionOutput",
    "ChannelFinderResult",
    "ChannelInfo",
    "ChannelFinderError",
    "PipelineModeError",
    "DatabaseLoadError",
    "ConfigurationError",
    "HierarchicalNavigationError",
    "QueryProcessingError",
]
