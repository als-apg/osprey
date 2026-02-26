"""Osprey Event Streaming System.

This package provides a unified, type-safe event streaming system for Osprey.
Events are typed dataclasses defined in types.py. EventEmitter in emitter.py
handles emission via registered handlers.

Usage:
    from osprey.events import EventEmitter, StatusEvent

    emitter = EventEmitter("my_component")
    emitter.emit(StatusEvent(message="Processing...", level="status"))
"""

# Event emission
from .emitter import (
    EventEmitter,
    clear_fallback_handlers,
    register_fallback_handler,
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
]
