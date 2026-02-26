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
from .emitter import EventEmitter
from .types import (
    BaseEvent,
    ErrorEvent,
    OspreyEvent,
    StatusEvent,
)

__all__ = [
    # Base
    "BaseEvent",
    "OspreyEvent",
    # Status
    "StatusEvent",
    # Results
    "ErrorEvent",
    # Emitter
    "EventEmitter",
]
