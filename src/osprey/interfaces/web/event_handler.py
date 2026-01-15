"""Web Event Handler for Osprey Debug UI.

This module provides the WebEventHandler class that converts typed Osprey events
to JSON and sends them to connected WebSocket clients for real-time display.

The handler serializes all event types (StatusEvent, LLMRequestEvent, etc.)
to a consistent JSON format that the frontend can render.
"""

import asyncio
from dataclasses import asdict
from datetime import datetime
from typing import TYPE_CHECKING

from osprey.events import OspreyEvent, parse_event

if TYPE_CHECKING:
    from fastapi import WebSocket


class WebEventHandler:
    """Sends typed Osprey events to connected WebSocket clients.

    This handler converts OspreyEvent dataclasses to JSON-serializable
    dictionaries and sends them via WebSocket for real-time display.

    Attributes:
        websocket: The FastAPI WebSocket connection
        include_raw: Whether to include raw event data in output
    """

    def __init__(self, websocket: "WebSocket", include_raw: bool = False):
        """Initialize the web event handler.

        Args:
            websocket: FastAPI WebSocket connection to send events to
            include_raw: Include raw dataclass dict in event payload
        """
        self.websocket = websocket
        self.include_raw = include_raw
        self._event_count = 0

    async def handle(self, event: OspreyEvent) -> None:
        """Process and send a typed event to the WebSocket client.

        Converts the event to a JSON-serializable format with metadata
        and sends it via the WebSocket connection.

        Args:
            event: The typed OspreyEvent to process and send
        """
        self._event_count += 1

        # Build event payload
        event_data = {
            "id": self._event_count,
            "type": event.__class__.__name__,
            "timestamp": datetime.now().isoformat(),
            "component": getattr(event, "component", ""),
        }

        # Add event-specific fields based on type
        event_dict = asdict(event)

        # Remove common fields already in event_data
        event_dict.pop("component", None)

        # Convert datetime objects to ISO strings for JSON serialization
        def serialize_value(v):
            if isinstance(v, datetime):
                return v.isoformat()
            elif isinstance(v, dict):
                return {k: serialize_value(val) for k, val in v.items()}
            elif isinstance(v, list):
                return [serialize_value(item) for item in v]
            return v

        event_dict = {k: serialize_value(v) for k, v in event_dict.items()}

        # Add remaining fields
        event_data["data"] = event_dict

        # Optionally include raw event for debugging
        if self.include_raw:
            raw_dict = asdict(event)
            event_data["raw"] = {k: serialize_value(v) for k, v in raw_dict.items()}

        try:
            await self.websocket.send_json(event_data)
        except Exception as e:
            # Log the error for debugging (don't silently swallow)
            import logging
            logging.getLogger(__name__).warning(f"Failed to send event: {e}")

    async def handle_dict(self, event_dict: dict) -> None:
        """Process a raw event dictionary and send to client.

        Parses the dictionary to a typed event first, then handles it.

        Args:
            event_dict: Raw event dictionary from LangGraph streaming
        """
        event = parse_event(event_dict)
        if event:
            await self.handle(event)

    def create_fallback_handler(self):
        """Create a callback function for fallback handler registration.

        Returns a synchronous callback that schedules async event handling.
        Used with register_fallback_handler() for events outside graph execution.

        Returns:
            Callback function suitable for register_fallback_handler()
        """

        def callback(event_dict: dict) -> None:
            """Fallback callback that schedules async handling."""
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.handle_dict(event_dict))
                else:
                    loop.run_until_complete(self.handle_dict(event_dict))
            except RuntimeError:
                # No event loop available
                pass

        return callback

    @property
    def event_count(self) -> int:
        """Return the total number of events handled."""
        return self._event_count
