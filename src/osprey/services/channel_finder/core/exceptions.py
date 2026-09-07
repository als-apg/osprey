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


class GraphIndexBuildError(ChannelFinderError):
    """Raised when the graph search index cannot be built from the corpus."""

    pass


class AddressPatternError(ChannelFinderError):
    """Raised when a device family's address pattern cannot be expanded.

    The pattern is the family's own address column, so a family that gives two
    of them, or one naming a placeholder the expander has no value for, would
    otherwise reach the database as channels whose address is the literal
    pattern text.  It is a defect in the input, not a family to skip.
    """
