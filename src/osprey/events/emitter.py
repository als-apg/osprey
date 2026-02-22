"""Event emitter for Osprey event streaming.

This module provides the EventEmitter class that handles event emission
via registered fallback handlers.

Events are serialized to dicts and dispatched to all registered handlers.
UI contexts (TUI, CLI, Web) register handlers to receive events.

Usage:
    from osprey.events.emitter import EventEmitter, register_fallback_handler
    from osprey.events.types import StatusEvent

    # In a component
    emitter = EventEmitter("my_component")
    emitter.emit(StatusEvent(message="Processing..."))

    # For TUI/UI that needs events
    def my_handler(event_dict):
        queue.put_nowait(event_dict)

    unregister = register_fallback_handler(my_handler)
    # ... run UI ...
    unregister()  # Cleanup on exit
"""

from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime
from typing import Any

from .types import OspreyEvent

# Global handlers for event dispatch
# UI contexts register here to receive all emitted events
_fallback_handlers: list[Callable[[dict[str, Any]], None]] = []


def register_fallback_handler(handler: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
    """Register a handler to receive emitted events.

    UI contexts (TUI, CLI, Web) register handlers here to receive all events.

    Args:
        handler: Callable that receives serialized event dicts

    Returns:
        Unregister function to remove the handler

    Example:
        def my_handler(event_dict):
            queue.put_nowait(event_dict)

        unregister = register_fallback_handler(my_handler)

        # ... run UI ...

        unregister()  # Cleanup on exit
    """
    _fallback_handlers.append(handler)

    def unregister() -> None:
        if handler in _fallback_handlers:
            _fallback_handlers.remove(handler)

    return unregister


def clear_fallback_handlers() -> None:
    """Clear all registered fallback handlers.

    Useful for testing or complete reset scenarios.
    """
    _fallback_handlers.clear()


class EventEmitter:
    """Emits typed events via registered handlers.

    Events are serialized and dispatched to all registered handlers.

    Used by ComponentLogger and infrastructure nodes.

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
        """Emit typed event to all registered handlers.

        Silent no-op if no handlers are registered (safe default).

        Args:
            event: The typed event to emit
        """
        # Ensure component is set
        if not event.component:
            event.component = self.component

        # Serialize and dispatch to handlers
        serialized = self._serialize(event)
        self._emit_to_fallback_handlers(serialized)

    def _emit_to_fallback_handlers(self, serialized: dict[str, Any]) -> None:
        """Emit to all registered fallback handlers.

        Args:
            serialized: Serialized event dict
        """
        for handler in _fallback_handlers:
            try:
                handler(serialized)
            except Exception:
                # Don't crash on fallback failures
                pass

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
