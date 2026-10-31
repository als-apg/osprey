"""
Custom exceptions for channel finder.

Provides specific exception types for better error handling and debugging.
"""

from osprey_connectors.errors import ConfigurationError as _FrameworkConfigurationError


class ChannelFinderError(Exception):
    """Base exception for all channel finder errors."""


class PipelineModeError(ChannelFinderError):
    """Raised when an invalid pipeline mode is specified."""


class DatabaseLoadError(ChannelFinderError):
    """Raised when a database file cannot be loaded."""


class ConfigurationError(ChannelFinderError, _FrameworkConfigurationError):
    """Raised when configuration is invalid or incomplete."""


class HierarchicalNavigationError(ChannelFinderError):
    """Raised when hierarchical navigation fails (e.g., combinatorial explosion)."""


class QueryProcessingError(ChannelFinderError):
    """Raised when query processing fails."""


class CoverageJudgeError(ChannelFinderError):
    """Raised when the benchmark's coverage judge cannot score a query.

    Either the judge cannot be resolved from the project's configuration, or a
    judge call fails or returns no verdict. A query the judge could not score is
    not scored another way. The message says which of the two it was.
    """


class GraphIndexBuildError(ChannelFinderError):
    """Raised when the graph search index cannot be built from the corpus."""
