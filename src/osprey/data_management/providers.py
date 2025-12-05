"""
Data Source Abstraction Layer

This module provides the base abstractions for integrating external data sources
into the Osprey Agentic Framework. Data sources can include user memory, knowledge graphs,
databases, APIs, and custom user-defined sources.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .request import DataSourceRequest

logger = logging.getLogger(__name__)


@dataclass
class DataSourceContext:
    """
    Container for data source retrieval results.

    This standardized format allows different data sources to return results
    in a consistent way while preserving source-specific metadata.
    """

    source_name: str  # Unique identifier for the data source
    context_type: str  # Type of context data (for validation)
    data: Any  # The actual retrieved data
    metadata: dict[str, Any] = field(default_factory=dict)  # Additional source metadata
    provider: Optional["DataSourceProvider"] = (
        None  # Reference to the provider that created this context
    )

    def format_for_prompt(self) -> str:
        """
        Format this context for inclusion in LLM prompts.

        Delegates to the provider's format_for_prompt method if available,
        otherwise falls back to default formatting.
        """
        if self.provider:
            return self.provider.format_for_prompt(self)

        # Fallback formatting if no provider reference
        if hasattr(self.data, "format_for_prompt"):
            return self.data.format_for_prompt()
        elif hasattr(self.data, "format_for_llm"):
            return self.data.format_for_llm()
        else:
            return str(self.data)

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of this data source context for logging/debugging."""
        return {
            "source_name": self.source_name,
            "context_type": self.context_type,
            "data_type": type(self.data).__name__,
            "metadata": self.metadata,
            "has_data": self.data is not None,
        }


class DataSourceProvider(ABC):
    """
    Abstract base class for all data source providers.

    Data source providers are responsible for:
    1. Determining if they can provide data for the current context
    2. Retrieving data from their specific source
    3. Returning data in a standardized format
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this data source provider."""
        pass

    @property
    @abstractmethod
    def context_type(self) -> str:
        """
        Context type this provider creates.

        Should match a registered context type in the context registry
        for type validation and LLM prompt formatting.
        """
        pass

    @abstractmethod
    async def retrieve_data(self, request: "DataSourceRequest") -> DataSourceContext | None:
        """
        Retrieve data from this source given the current request.

        Args:
            request: Data source request containing user info, session context, and requester details

        Returns:
            DataSourceContext with retrieved data, or None if no data available

        Raises:
            Should handle all internal exceptions and return None rather than raising,
            unless the exception represents a critical system failure.
        """
        pass

    @abstractmethod
    def should_respond(self, request: "DataSourceRequest") -> bool:
        """
        Determine if this data source should respond to the given request.

        This should be a fast check (no I/O) that determines whether it makes
        sense to call retrieve_data() for the given request.

        Args:
            request: Data source request with requester information

        Returns:
            True if this data source should provide data for this request
        """
        pass

    @property
    def description(self) -> str:
        """Human-readable description of this data source."""
        return f"Data source: {self.name}"

    def get_config_requirements(self) -> dict[str, Any]:
        """
        Get configuration requirements for this data source.

        Returns a dictionary describing what configuration this data source needs.
        This can be used for validation and documentation.
        """
        return {}

    async def health_check(self) -> bool:
        """
        Perform a health check for this data source.

        This is an optional method that can be implemented by data sources
        that need to verify connectivity or service availability.

        Returns:
            True if the data source is healthy and available
        """
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

    def format_for_prompt(self, context: DataSourceContext) -> str:
        """
        Format this data source's context for inclusion in LLM prompts.

        Each data source provider can override this to control exactly how
        their data appears in LLM prompts, including section headers and formatting.

        Args:
            context: The DataSourceContext returned by retrieve_data()

        Returns:
            Formatted string ready for inclusion in LLM prompts
        """
        if not context or not context.data:
            return ""

        # Default formatting with source name header
        source_title = context.source_name.replace("_", " ").title()

        try:
            # Try the data object's formatting methods first
            if hasattr(context.data, "format_for_prompt"):
                formatted_data = context.data.format_for_prompt()
            elif hasattr(context.data, "format_for_llm"):
                formatted_data = context.data.format_for_llm()
            else:
                formatted_data = str(context.data)

            if formatted_data.strip():
                return f"**{source_title}:**\n{formatted_data}"
            else:
                return f"**{source_title}:**\n(No content available)"

        except Exception as e:
            logger.warning(f"Failed to format context from {self.name}: {e}")
            return f"**{source_title}:**\n(Error formatting content: {e})"
