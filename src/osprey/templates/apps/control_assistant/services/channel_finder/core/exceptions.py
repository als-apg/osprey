"""
Custom exceptions for channel finder.

Provides specific exception types for better error handling and debugging.
"""


class ChannelFinderError(Exception):
    """Base exception for all channel finder errors."""
    pass


class PipelineModeError(ChannelFinderError):
    """Raised when an invalid pipeline mode is specified."""
    pass


class DatabaseLoadError(ChannelFinderError):
    """Raised when a database file cannot be loaded."""
    pass


class ConfigurationError(ChannelFinderError):
    """Raised when configuration is invalid or incomplete."""
    pass


class HierarchicalNavigationError(ChannelFinderError):
    """Raised when hierarchical navigation fails (e.g., combinatorial explosion)."""
    pass


class QueryProcessingError(ChannelFinderError):
    """Raised when query processing fails."""
    pass

