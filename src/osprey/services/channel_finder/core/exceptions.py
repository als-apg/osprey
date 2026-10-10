"""
Custom exceptions for channel finder.

Provides specific exception types for better error handling and debugging.
"""

from osprey.errors import ConfigurationError as _FrameworkConfigurationError


class ChannelFinderError(Exception):
    """Base exception for all channel finder errors."""

    pass


class PipelineModeError(ChannelFinderError):
    """Raised when an invalid pipeline mode is specified."""

    pass


class DatabaseLoadError(ChannelFinderError):
    """Raised when a database file cannot be loaded."""

    pass


class ConfigurationError(ChannelFinderError, _FrameworkConfigurationError):
    """Raised when configuration is invalid or incomplete."""

    pass


class HierarchicalNavigationError(ChannelFinderError):
    """Raised when hierarchical navigation fails (e.g., combinatorial explosion)."""

    pass


class QueryProcessingError(ChannelFinderError):
    """Raised when query processing fails."""

    pass


class CoverageJudgeError(ChannelFinderError):
    """Raised when the benchmark's coverage judge cannot score a query.

    Either the judge cannot be resolved from the project's configuration, or a
    judge call fails or returns no verdict. A query the judge could not score is
    not scored another way. The message says which of the two it was.
    """

    pass


class GraphIndexBuildError(ChannelFinderError):
    """Raised when the graph search index cannot be built from the corpus."""

    pass
