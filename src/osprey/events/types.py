"""Typed event classes for Osprey event streaming.

This module defines the event types used in Osprey's event streaming system.
Events are dataclasses that serialize to dicts for transport.

Event Categories:
- Status Events: General status updates during execution
- Result Events: Errors during execution

Usage:
    from osprey.events.types import StatusEvent, OspreyEvent

    event = StatusEvent(message="Processing...", component="channel_finder")
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass
class BaseEvent:
    """Base class for all Osprey events.

    Attributes:
        timestamp: When the event was created
        component: The component that emitted this event (e.g., "router", "classifier")
    """

    timestamp: datetime = field(default_factory=datetime.now)
    component: str = ""


# -----------------------------------------------------------------------------
# Status Events
# -----------------------------------------------------------------------------


@dataclass
class StatusEvent(BaseEvent):
    """Status update during execution.

    Used for general progress updates, log messages, and status changes.

    Attributes:
        message: The status message
        level: Severity/type of the status (info, warning, error, debug, success, status)
        phase: Current execution phase (e.g., "Task Preparation", "Execution")
        step: Current step number (1-based)
        total_steps: Total number of steps
    """

    message: str = ""
    level: Literal["info", "warning", "error", "debug", "success", "status", "key_info"] = "info"
    phase: str | None = None
    step: int | None = None
    total_steps: int | None = None


# -----------------------------------------------------------------------------
# Error Events
# -----------------------------------------------------------------------------


@dataclass
class ErrorEvent(BaseEvent):
    """Error during execution.

    Emitted when an error occurs during execution.

    Attributes:
        error_type: Classification of the error (e.g., "ValidationError", "TimeoutError")
        error_message: Human-readable error message
        recoverable: Whether the error is recoverable
        stack_trace: Optional stack trace for debugging
    """

    error_type: str = ""
    error_message: str = ""
    recoverable: bool = False
    stack_trace: str | None = None


# -----------------------------------------------------------------------------
# Union Type
# -----------------------------------------------------------------------------

OspreyEvent = StatusEvent | ErrorEvent
"""Union type of all Osprey events for type hints."""
