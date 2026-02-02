"""Event parser for reconstructing typed events from stream dicts.

This module provides the parse_event() function that reconstructs typed
OspreyEvent instances from serialized dictionaries received via streaming.

Usage:
    from osprey.events.parser import parse_event
    from osprey.events.types import StatusEvent

    # In a stream consumer
    async for chunk in graph.astream(..., stream_mode="custom"):
        event = parse_event(chunk)
        if event:
            match event:
                case StatusEvent(message=msg):
                    print(f"Status: {msg}")
"""

import dataclasses
from datetime import datetime
from typing import Any

from .types import (
    ApprovalReceivedEvent,
    ApprovalRequiredEvent,
    CapabilitiesSelectedEvent,
    CapabilityCompleteEvent,
    CapabilityStartEvent,
    CodeExecutedEvent,
    CodeGeneratedEvent,
    CodeGenerationStartEvent,
    ErrorEvent,
    LLMRequestEvent,
    LLMResponseEvent,
    OspreyEvent,
    PhaseCompleteEvent,
    PhaseStartEvent,
    PlanCreatedEvent,
    ResultEvent,
    StatusEvent,
    TaskExtractedEvent,
    ToolResultEvent,
    ToolUseEvent,
)

# Mapping of event class names to their classes
EVENT_CLASSES: dict[str, type] = {
    "StatusEvent": StatusEvent,
    "PhaseStartEvent": PhaseStartEvent,
    "PhaseCompleteEvent": PhaseCompleteEvent,
    "TaskExtractedEvent": TaskExtractedEvent,
    "CapabilitiesSelectedEvent": CapabilitiesSelectedEvent,
    "PlanCreatedEvent": PlanCreatedEvent,
    "CapabilityStartEvent": CapabilityStartEvent,
    "CapabilityCompleteEvent": CapabilityCompleteEvent,
    "LLMRequestEvent": LLMRequestEvent,
    "LLMResponseEvent": LLMResponseEvent,
    "ToolUseEvent": ToolUseEvent,
    "ToolResultEvent": ToolResultEvent,
    "CodeGeneratedEvent": CodeGeneratedEvent,
    "CodeExecutedEvent": CodeExecutedEvent,
    "CodeGenerationStartEvent": CodeGenerationStartEvent,
    "ApprovalRequiredEvent": ApprovalRequiredEvent,
    "ApprovalReceivedEvent": ApprovalReceivedEvent,
    "ResultEvent": ResultEvent,
    "ErrorEvent": ErrorEvent,
}


def parse_event(data: dict[str, Any]) -> OspreyEvent | None:
    """Reconstruct typed event from stream dict.

    Takes a dictionary (typically from LangGraph streaming) and reconstructs
    the original typed event. Returns None for non-Osprey events (state updates,
    LLM tokens, legacy dict events, etc.)

    Args:
        data: Dictionary containing serialized event data.
              Must have "event_class" key for typed events.

    Returns:
        Reconstructed OspreyEvent instance, or None if not a valid Osprey event

    Example:
        async for chunk in graph.astream(..., stream_mode="custom"):
            event = parse_event(chunk)
            if event:
                match event:
                    case StatusEvent(message=msg, level="error"):
                        handle_error(msg)
                    case CapabilityStartEvent(capability_name=name):
                        show_capability_start(name)
    """
    if not isinstance(data, dict):
        return None

    # Make a copy to avoid mutating the original
    data = data.copy()

    # Extract event class name
    event_class_name = data.pop("event_class", None)
    if not event_class_name:
        return None

    # Look up the event class
    event_class = EVENT_CLASSES.get(event_class_name)
    if not event_class:
        return None

    # Parse timestamp from ISO format string
    if "timestamp" in data and isinstance(data["timestamp"], str):
        try:
            data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        except ValueError:
            # If parsing fails, use current time
            data["timestamp"] = datetime.now()

    # Filter to only fields the dataclass accepts
    valid_fields = {f.name for f in dataclasses.fields(event_class)}
    filtered_data = {k: v for k, v in data.items() if k in valid_fields}

    try:
        return event_class(**filtered_data)
    except TypeError:
        # If construction fails (e.g., missing required field), return None
        return None


def is_osprey_event(data: dict[str, Any]) -> bool:
    """Check if a dict represents an Osprey typed event.

    Args:
        data: Dictionary to check

    Returns:
        True if the dict has a valid event_class key
    """
    if not isinstance(data, dict):
        return False
    event_class_name = data.get("event_class")
    return event_class_name in EVENT_CLASSES
