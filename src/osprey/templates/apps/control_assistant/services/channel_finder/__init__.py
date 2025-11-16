"""
Channel Finder - Generic In-Context Retrieval System

A facility-agnostic system for finding control system channels using natural language queries.
Configure which facility to use in config.yml.
"""

# Service (high-level interface)
from .service import ChannelFinderService

# Core models and base classes
from .core.models import (
    QuerySplitterOutput,
    ChannelMatchOutput,
    ChannelCorrectionOutput,
    ChannelFinderResult,
    ChannelInfo
)
from .core.exceptions import (
    ChannelFinderError,
    PipelineModeError,
    DatabaseLoadError,
    ConfigurationError,
    HierarchicalNavigationError,
    QueryProcessingError
)

# Pipelines
from .pipelines.in_context import InContextPipeline

# Databases
from .databases import (
    LegacyChannelDatabase,
    TemplateChannelDatabase,
    HierarchicalChannelDatabase
)

__version__ = "2.0.0"

__all__ = [
    # Service (high-level interface)
    'ChannelFinderService',
    
    # Pipelines
    'InContextPipeline',
    
    # Database classes
    'TemplateChannelDatabase',
    'LegacyChannelDatabase',
    'HierarchicalChannelDatabase',
    
    # Data models
    'QuerySplitterOutput',
    'ChannelMatchOutput',
    'ChannelCorrectionOutput',
    'ChannelFinderResult',
    'ChannelInfo',
    
    # Exceptions
    'ChannelFinderError',
    'PipelineModeError',
    'DatabaseLoadError',
    'ConfigurationError',
    'HierarchicalNavigationError',
    'QueryProcessingError',
]

