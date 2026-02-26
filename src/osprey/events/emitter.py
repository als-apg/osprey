"""Event emitter for Osprey event streaming.

This module provides the EventEmitter class that handles event emission
using a LangGraph-first approach with fallback support.

The emitter tries to use LangGraph's get_stream_writer() for native streaming
during graph execution, and falls back to registered handlers for events
that occur outside of graph.astream() (e.g., during startup/shutdown).

Usage:
    from osprey.events.emitter import EventEmitter, register_fallback_handler
    from osprey.events.types import StatusEvent

    # In a component
    emitter = EventEmitter("my_component")
    emitter.emit(StatusEvent(message="Processing..."))

    # For TUI/UI that needs startup events
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

# Global fallback handlers for events outside LangGraph context
# UI contexts that need ALL events (including startup) can register here
_fallback_handlers: list[Callable[[dict[str, Any]], None]] = []


def register_fallback_handler(handler: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
    """Register a handler for events emitted outside LangGraph context.

    Use this when a UI needs to capture startup/shutdown events that happen
    before or after graph.astream(). Most interfaces won't need this.

    Args:
        handler: Callable that receives serialized event dicts

    Returns:
        Unregister function to remove the handler

    Example:
        # TUI startup - capture pre-execution events
        def tui_fallback(event_dict):
            queue.put_nowait(event_dict)

        unregister = register_fallback_handler(tui_fallback)

        # ... run TUI ...

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
    """Emits typed events using LangGraph streaming when available.

    LangGraph-First: Tries get_stream_writer() for native transport.
    Fallback-Ready: Routes to registered handlers when outside graph execution.

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
        self._writer: Callable[[dict[str, Any]], None] | None = None

    def emit(self, event: OspreyEvent) -> None:
        """Emit typed event via best available transport.

        1. Try LangGraph's get_stream_writer() (works during graph.astream())
        2. Fall back to registered handlers (for startup/shutdown events)
        3. Silent no-op if neither available (safe default)

        Args:
            event: The typed event to emit
        """
        # Ensure component is set
        if not event.component:
            event.component = self.component

        # Serialize once for either transport
        serialized = self._serialize(event)

        # Try LangGraph native streaming first (the happy path)
        if self._try_langgraph_emit(serialized):
            return  # Success via LangGraph

        # Fallback for outside-execution events
        self._emit_to_fallback_handlers(serialized)

    def _try_langgraph_emit(self, serialized: dict[str, Any]) -> bool:
        """Try to emit via LangGraph streaming.

        Args:
            serialized: Serialized event dict

        Returns:
            True if emission succeeded, False if not in LangGraph context
        """
        try:
            from langgraph.config import get_stream_writer

            writer = get_stream_writer()
            writer(serialized)
            return True
        except RuntimeError:
            # Not in LangGraph context - use fallback
            return False
        except ImportError:
            # LangGraph not available - use fallback
            return False

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
