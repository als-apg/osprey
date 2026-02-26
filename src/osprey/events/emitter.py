"""Event emitter for Osprey event streaming.

This module provides the EventEmitter class that handles event emission
via registered handlers.

Events are serialized to dicts and dispatched to all registered handlers.

Usage:
    from osprey.events.emitter import EventEmitter
    from osprey.events.types import StatusEvent

    # In a component
    emitter = EventEmitter("my_component")
    emitter.emit(StatusEvent(message="Processing..."))
"""

from dataclasses import asdict
from datetime import datetime
from typing import Any

from .types import OspreyEvent


class EventEmitter:
    """Emits typed events via registered handlers.

    Events are serialized and dispatched to all registered handlers.

    Used by ComponentLogger for status and error event emission.

    Attributes:
        component: Default component name for events without one set
    """

    def __init__(self, component: str):
        """Initialize the event emitter.

        Args:
            component: Default component name for events
        """
        self.component = component

    def emit(self, event: OspreyEvent) -> None:
        """Emit typed event.

        Currently a no-op (no handlers registered). Retained as the
        public API so that ComponentLogger and infrastructure nodes
        continue to compile without changes.

        Args:
            event: The typed event to emit
        """
        # Ensure component is set
        if not event.component:
            event.component = self.component

    def _serialize(self, event: OspreyEvent) -> dict[str, Any]:
        """Convert typed event to dict for transport.

        Args:
            event: The typed event to serialize

        Returns:
            Dict suitable for JSON serialization and transport
        """
        result = asdict(event)

        # Add event class name for reconstruction on the receiving end
        result["event_class"] = type(event).__name__

        # Convert timestamp to ISO format string for JSON serialization
        if isinstance(result.get("timestamp"), datetime):
            result["timestamp"] = result["timestamp"].isoformat()

        return result
