"""Archiver Retrieval Capability.

Retrieves historical time-series data from the archiver system for
specified channels and time ranges. Useful for trend analysis, post-mortem
investigation, and data-driven decision making.

Context Flow:
    Input:  CHANNEL_ADDRESSES (from channel_finding), TIME_RANGE
    Output: ARCHIVER_DATA (historical time-series data)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar

from osprey.base.capability import BaseCapability
from osprey.base.errors import ErrorClassification, ErrorSeverity
from osprey.context import CapabilityContext

# ---------------------------------------------------------------------------
# Context class (inlined per v0.11 convention)
# ---------------------------------------------------------------------------


class ArchiverDataContext(CapabilityContext):
    """Context containing historical archiver data.

    Produced by the archiver_retrieval capability after fetching
    time-series data from the archiver system.
    """

    CONTEXT_TYPE: ClassVar[str] = "ARCHIVER_DATA"
    CONTEXT_CATEGORY: ClassVar[str] = "control_system"

    timestamps: list[datetime]
    precision_ms: int
    time_series_data: dict[str, list[float]]
    available_channels: list[str]
    timezone_name: str = (
        ""  # Human-readable timezone name (e.g. "EST", "PST") copied from TIME_RANGE
    )

    def get_summary(self) -> str:
        """Return a human-readable summary."""
        n_channels = len(self.available_channels)
        n_points = len(self.timestamps)
        return f"Archiver data: {n_channels} channel(s), {n_points} time point(s)"

    def get_access_details(self, key: str) -> dict:
        """Return access details for this context."""
        return {
            "context_type": self.CONTEXT_TYPE,
            "key": key,
            "channel_count": len(self.available_channels),
            "channels": self.available_channels,
            "timestamp_count": len(self.timestamps),
            "precision_ms": self.precision_ms,
            "timezone_name": self.timezone_name or "",
        }


# ---------------------------------------------------------------------------
# Capability
# ---------------------------------------------------------------------------


class ArchiverRetrievalCapability(BaseCapability):
    """Retrieve historical time-series data from archiver systems.

    Queries the configured archiver connector for historical PV data
    over a specified time range.
    """

    name = "archiver_retrieval"
    description = "Retrieve historical time-series data from archiver systems"
    provides = ["ARCHIVER_DATA"]
    requires = ["CHANNEL_ADDRESSES", "TIME_RANGE"]

    @staticmethod
    def classify_error(exc: Exception, context: dict) -> ErrorClassification:
        """Classify archiver-retrieval errors."""
        if isinstance(exc, TimeoutError):
            return ErrorClassification(
                severity=ErrorSeverity.RETRIABLE,
                user_message="Archiver query timed out, retrying...",
                metadata={"technical_details": str(exc)},
            )
        if isinstance(exc, ConnectionError):
            return ErrorClassification(
                severity=ErrorSeverity.RETRIABLE,
                user_message="Lost connection to archiver, retrying...",
                metadata={"technical_details": str(exc)},
            )
        return ErrorClassification(
            severity=ErrorSeverity.RETRIABLE,
            user_message=f"Unexpected error during archiver retrieval: {exc}",
            metadata={"technical_details": str(exc)},
        )

    @staticmethod
    async def execute(state: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Execute archiver retrieval via the configured connector.

        This is a stub implementation. In production, the MCP-based
        control system tools handle the actual archiver queries. This
        capability exists primarily for registry metadata, eject support,
        and as a reference implementation for custom overrides.
        """
        return {"messages": state.get("messages", [])}
