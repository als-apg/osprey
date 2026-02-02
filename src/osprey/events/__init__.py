"""Osprey Event Streaming System.

This package provides a unified, type-safe event streaming system for Osprey.
It replaces the previous dict-based event system with typed dataclass events
that provide better validation, IDE support, and pattern matching capabilities.

Architecture:
    - Events are typed dataclasses defined in types.py
    - EventEmitter in emitter.py handles emission via LangGraph streaming
    - parse_event() in parser.py reconstructs typed events from stream dicts
    - Interface handlers (TUI, CLI, Web) use pattern matching for clean routing

Usage:
    # Emitting events (in components/nodes)
    from osprey.events import EventEmitter, StatusEvent

    emitter = EventEmitter("my_component")
    emitter.emit(StatusEvent(message="Processing...", level="status"))

    # Consuming events (in interfaces)
    from osprey.events import parse_event, StatusEvent, CapabilityStartEvent

    async for chunk in graph.astream(..., stream_mode="custom"):
        event = parse_event(chunk)
        if event:
            match event:
                case StatusEvent(message=msg):
                    display_status(msg)
                case CapabilityStartEvent(capability_name=name):
                    show_capability_start(name)

    # Fallback handlers for events outside graph execution
    from osprey.events import register_fallback_handler

    unregister = register_fallback_handler(lambda e: queue.put_nowait(e))
    # ... run UI ...
    unregister()

    # Multi-mode streaming with LLM tokens
    from osprey.events import consume_stream, LLMToken

    async for output in consume_stream(graph, input_data, config):
        match output:
            case StatusEvent(message=msg):
                display_status(msg)
            case LLMToken(content=text):
                print(text, end="", flush=True)
"""

# Event types
# Event emission
from .emitter import (
    EventEmitter,
    clear_fallback_handlers,
    register_fallback_handler,
)

# Event parsing
from .parser import (
    EVENT_CLASSES,
    is_osprey_event,
    parse_event,
)

# Multi-mode streaming
from .streaming import (
    LLMToken,
    StateUpdate,
    StreamMode,
    consume_custom_events,
    consume_stream,
)
from .types import (
    ApprovalReceivedEvent,
    ApprovalRequiredEvent,
    BaseEvent,
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

__all__ = [
    # Base
    "BaseEvent",
    "OspreyEvent",
    # Status
    "StatusEvent",
    # Phase Lifecycle
    "PhaseStartEvent",
    "PhaseCompleteEvent",
    # Data Output
    "TaskExtractedEvent",
    "CapabilitiesSelectedEvent",
    "PlanCreatedEvent",
    # Capability
    "CapabilityStartEvent",
    "CapabilityCompleteEvent",
    # LLM
    "LLMRequestEvent",
    "LLMResponseEvent",
    # Tool/Code
    "ToolUseEvent",
    "ToolResultEvent",
    "CodeGeneratedEvent",
    "CodeGenerationStartEvent",
    "CodeExecutedEvent",
    # Control Flow
    "ApprovalRequiredEvent",
    "ApprovalReceivedEvent",
    # Results
    "ResultEvent",
    "ErrorEvent",
    # Emitter
    "EventEmitter",
    "register_fallback_handler",
    "clear_fallback_handlers",
    # Parser
    "parse_event",
    "is_osprey_event",
    "EVENT_CLASSES",
    # Streaming
    "consume_stream",
    "consume_custom_events",
    "LLMToken",
    "StateUpdate",
    "StreamMode",
]
