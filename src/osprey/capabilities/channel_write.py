"""Channel Write Capability.

Writes values to control system channels via the configured connector.
All writes require human approval before execution, enforcing the
safety-first design of the Osprey framework.

Context Flow:
    Input:  CHANNEL_ADDRESSES (from channel_finding)
    Output: CHANNEL_WRITE_RESULTS (success/failure per channel)
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel

from osprey.base.capability import BaseCapability
from osprey.base.errors import ErrorClassification, ErrorSeverity
from osprey.context import CapabilityContext

# ---------------------------------------------------------------------------
# Context classes (inlined per v0.11 convention)
# ---------------------------------------------------------------------------


class WriteVerificationInfo(BaseModel):
    """Verification details for a channel write operation."""

    level: str
    verified: bool


class ChannelWriteResult(BaseModel):
    """Result of a single channel write operation."""

    channel_address: str
    value_written: Any
    success: bool


class ChannelWriteResultsContext(CapabilityContext):
    """Context containing results of channel write operations.

    Produced by the channel_write capability after writing values
    to the control system.
    """

    CONTEXT_TYPE: ClassVar[str] = "CHANNEL_WRITE_RESULTS"
    CONTEXT_CATEGORY: ClassVar[str] = "control_system"

    results: list[ChannelWriteResult]
    total_writes: int
    successful_count: int
    failed_count: int

    def get_summary(self) -> str:
        """Return a human-readable summary."""
        return (
            f"Wrote to {self.total_writes} channel(s): "
            f"{self.successful_count} succeeded, {self.failed_count} failed"
        )

    def get_access_details(self, key: str) -> dict:
        """Return access details for this context."""
        return {
            "context_type": self.CONTEXT_TYPE,
            "key": key,
            "total_writes": self.total_writes,
            "successful_count": self.successful_count,
            "failed_count": self.failed_count,
        }


# ---------------------------------------------------------------------------
# Capability
# ---------------------------------------------------------------------------


class ChannelWriteCapability(BaseCapability):
    """Write values to control system channels with human approval.

    All writes are gated by the human-in-the-loop approval system.
    """

    name = "channel_write"
    description = "Write values to control system channels (requires approval)"
    provides = ["CHANNEL_WRITE_RESULTS"]
    requires = ["CHANNEL_ADDRESSES"]

    @staticmethod
    def classify_error(exc: Exception, context: dict) -> ErrorClassification:
        """Classify channel-write errors."""
        if isinstance(exc, TimeoutError):
            return ErrorClassification(
                severity=ErrorSeverity.RETRIABLE,
                user_message="Channel write timed out, retrying...",
                metadata={"technical_details": str(exc)},
            )
        if isinstance(exc, ConnectionError):
            return ErrorClassification(
                severity=ErrorSeverity.CRITICAL,
                user_message="Lost connection to control system during write.",
                metadata={"technical_details": str(exc)},
            )
        return ErrorClassification(
            severity=ErrorSeverity.CRITICAL,
            user_message=f"Error during channel write: {exc}",
            metadata={"technical_details": str(exc)},
        )

    @staticmethod
    async def execute(state: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Execute channel write via the configured connector.

        This is a stub implementation. In production, the MCP-based
        control system tools handle the actual writes. This capability
        exists primarily for registry metadata, eject support, and as a
        reference implementation for custom overrides.
        """
        return {"messages": state.get("messages", [])}
