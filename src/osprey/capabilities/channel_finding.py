"""Channel Finding Capability.

Resolves natural-language queries into control system channel addresses
using the Channel Finder service. This is the primary entry point for
users who describe what they want to monitor or control in plain English.

The capability delegates to the Channel Finder service which supports
multiple pipeline architectures (hierarchical, in-context, middle-layer)
and database formats.

Context Flow:
    Input:  User query (natural language)
    Output: CHANNEL_ADDRESSES (list of resolved channel names)
"""

from __future__ import annotations

from typing import Any, ClassVar

from osprey.base.capability import BaseCapability
from osprey.base.errors import ErrorClassification, ErrorSeverity
from osprey.context import CapabilityContext

# ---------------------------------------------------------------------------
# Context class (inlined per v0.11 convention)
# ---------------------------------------------------------------------------


class ChannelAddressesContext(CapabilityContext):
    """Context containing resolved control system channel addresses.

    Produced by the channel_finding capability after resolving a
    natural-language query into concrete channel names.
    """

    CONTEXT_TYPE: ClassVar[str] = "CHANNEL_ADDRESSES"
    CONTEXT_CATEGORY: ClassVar[str] = "control_system"

    channels: list[str]
    original_query: str

    def get_summary(self) -> str:
        """Return a human-readable summary."""
        count = len(self.channels)
        preview = ", ".join(self.channels[:3])
        if count > 3:
            preview += f", ... ({count} total)"
        return f"Resolved channels: {preview}"

    def get_access_details(self, key: str) -> dict:
        """Return access details for this context."""
        return {
            "context_type": self.CONTEXT_TYPE,
            "key": key,
            "channel_count": len(self.channels),
            "channels": self.channels,
            "original_query": self.original_query,
        }


# ---------------------------------------------------------------------------
# Capability
# ---------------------------------------------------------------------------


class ChannelFindingCapability(BaseCapability):
    """Resolve natural-language queries to control system channel addresses.

    Uses the Channel Finder service to map user descriptions (e.g.,
    "beam position monitors in sector 4") to concrete PV / channel names.
    """

    name = "channel_finding"
    description = "Resolve natural-language queries to control system channel addresses"
    provides = ["CHANNEL_ADDRESSES"]
    requires: list[str] = []

    @staticmethod
    def classify_error(exc: Exception, context: dict) -> ErrorClassification:
        """Classify channel-finding errors."""
        from osprey.services.channel_finder.core.exceptions import (
            ChannelFinderError,
            ConfigurationError,
            DatabaseLoadError,
        )

        if isinstance(exc, ConfigurationError):
            return ErrorClassification(
                severity=ErrorSeverity.CRITICAL,
                user_message="Channel finder configuration error. Check your database setup.",
                metadata={"technical_details": str(exc)},
            )
        if isinstance(exc, DatabaseLoadError):
            return ErrorClassification(
                severity=ErrorSeverity.CRITICAL,
                user_message="Could not load the channel database.",
                metadata={"technical_details": str(exc)},
            )
        if isinstance(exc, ChannelFinderError):
            return ErrorClassification(
                severity=ErrorSeverity.RETRIABLE,
                user_message="Channel finding failed, retrying...",
                metadata={"technical_details": str(exc)},
            )
        return ErrorClassification(
            severity=ErrorSeverity.RETRIABLE,
            user_message=f"Unexpected error during channel finding: {exc}",
            metadata={"technical_details": str(exc)},
        )

    @staticmethod
    async def execute(state: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Execute channel finding against the configured service.

        This is a stub implementation. In production, the MCP-based
        channel finder tools handle the actual resolution. This capability
        exists primarily for registry metadata, eject support, and as a
        reference implementation for custom overrides.
        """
        return {"messages": state.get("messages", [])}
