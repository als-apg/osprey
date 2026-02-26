"""Channel Read Capability.

Reads current values from control system channels via the configured
connector (EPICS, mock, etc.). Requires channel addresses to have been
resolved first by the channel_finding capability.

Context flow: CHANNEL_ADDRESSES -> CHANNEL_VALUES
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar

from pydantic import BaseModel

from osprey.base.capability import BaseCapability
from osprey.base.errors import ErrorClassification, ErrorSeverity
from osprey.context import CapabilityContext


class ChannelValue(BaseModel):
    """A single channel reading with metadata."""

    value: str
    timestamp: datetime
    units: str


class ChannelValuesContext(CapabilityContext):
    """Context containing current channel readings.

    Produced by the channel_read capability after reading values from
    the control system.
    """

    CONTEXT_TYPE: ClassVar[str] = "CHANNEL_VALUES"
    CONTEXT_CATEGORY: ClassVar[str] = "control_system"

    channel_values: dict[str, ChannelValue]

    def get_summary(self) -> str:
        count = len(self.channel_values)
        return f"Read {count} channel value(s)"

    def get_access_details(self, key: str) -> dict:
        return {
            "context_type": self.CONTEXT_TYPE,
            "key": key,
            "channel_count": len(self.channel_values),
            "channels": list(self.channel_values.keys()),
        }


class ChannelReadCapability(BaseCapability):
    """Read current values from control system channels.

    Reads live PV values through the configured control system connector.
    """

    name = "channel_read"
    description = "Read current values from control system channels"
    provides = ["CHANNEL_VALUES"]
    requires = ["CHANNEL_ADDRESSES"]

    @staticmethod
    def classify_error(exc: Exception, context: dict) -> ErrorClassification:
        if isinstance(exc, TimeoutError):
            return ErrorClassification(
                severity=ErrorSeverity.RETRIABLE,
                user_message="Channel read timed out, retrying...",
                metadata={"technical_details": str(exc)},
            )
        if isinstance(exc, ConnectionError):
            return ErrorClassification(
                severity=ErrorSeverity.RETRIABLE,
                user_message="Lost connection to control system, retrying...",
                metadata={"technical_details": str(exc)},
            )
        return ErrorClassification(
            severity=ErrorSeverity.RETRIABLE,
            user_message=f"Unexpected error during channel read: {exc}",
            metadata={"technical_details": str(exc)},
        )

    @staticmethod
    async def execute(state: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Execute channel read via the configured connector.

        This is a stub implementation. In production, the MCP-based
        control system tools handle the actual reads. This capability
        exists primarily for registry metadata, eject support, and as a
        reference implementation for custom overrides.
        """
        return {"messages": state.get("messages", [])}
