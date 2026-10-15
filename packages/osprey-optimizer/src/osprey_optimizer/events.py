"""Event system for optimization progress tracking.

Provides an event bus for phases and optimization logic to emit progress
updates without tight coupling to the UI layer.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import TypeAlias

import polars as pl
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of events emitted during optimization."""

    OPTIMIZATION_STARTED = "optimization_started"
    PHASE_STARTED = "phase_started"
    ITERATION_COMPLETED = "iteration_completed"
    PHASE_COMPLETED = "phase_completed"
    PHASE_FAILED = "phase_failed"
    OPTIMIZATION_COMPLETED = "optimization_completed"


class OptimizationStartedData(BaseModel):
    """Data structure for OPTIMIZATION_STARTED events."""

    objective: str | None = None
    # Resolved [min, max] per active variable (the optimizer's VOCS search space),
    # known only after bounds resolution — so this event is emitted once the grid
    # is computed. Consumed to persist the live figure's normalization bounds.
    variable_bounds: dict[str, list[float]] | None = None


class PhaseStartedData(BaseModel):
    """Data structure for PHASE_STARTED events."""

    phase: str
    n_samples: int | None = None  # For LHS phase


class IterationCompletedData(BaseModel):
    """Data structure for ITERATION_COMPLETED events."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    phase: str
    iteration: int
    total: int
    best_value: float
    data: pl.DataFrame


class PhaseCompletedData(BaseModel):
    """Data structure for PHASE_COMPLETED events."""

    phase: str
    best_value: float | None = None  # For LHS and BO phases


class PhaseFailedData(BaseModel):
    """Data structure for PHASE_FAILED events."""

    phase: str
    error_message: str


class OptimizationCompletedData(BaseModel):
    """Data structure for OPTIMIZATION_COMPLETED events."""

    objective: str | None = None


EventData = (
    OptimizationStartedData
    | PhaseStartedData
    | IterationCompletedData
    | PhaseCompletedData
    | PhaseFailedData
    | OptimizationCompletedData
)


@dataclass
class OptimizationEvent:
    """An event emitted during optimization."""

    type: EventType
    data: EventData
    timestamp: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        return f"OptimizationEvent(type={self.type.value}, data={self.data})"


EventCallback: TypeAlias = Callable[[OptimizationEvent], None]


def emit_event(bus: "EventBus | None", event_type: EventType, data: EventData) -> None:
    """Wrap *data* in an OptimizationEvent and emit it; no-op when *bus* is None."""
    if bus:
        bus.emit(OptimizationEvent(type=event_type, data=data))


class EventBus:
    """Pub/sub event system for optimization progress.

    Thread-safe event bus that allows multiple listeners to subscribe
    to optimization events without coupling the event producers to
    the event consumers.

    Example:
        >>> bus = EventBus()
        >>> def on_iteration(event: OptimizationEvent) -> None:
        ...     print(f"Iteration {event.data.iteration}: {event.data.best_value}")
        >>> bus.subscribe(EventType.ITERATION_COMPLETED, on_iteration)
        >>> bus.emit(OptimizationEvent(
        ...     type=EventType.ITERATION_COMPLETED,
        ...     data=IterationCompletedData(
        ...         phase="BO", iteration=1, total=20, best_value=42.0, data=pl.DataFrame()
        ...     ),
        ... ))
    """

    def __init__(self) -> None:
        """Initialize the event bus."""
        self._subscribers: dict[EventType, list[EventCallback]] = {
            event_type: [] for event_type in EventType
        }
        self._lock = Lock()

    def subscribe(self, event_type: EventType, callback: EventCallback) -> None:
        """Subscribe to events of a specific type.

        Args:
            event_type: Type of event to subscribe to
            callback: Function to call when event is emitted
        """
        with self._lock:
            self._subscribers[event_type].append(callback)
            logger.debug(
                "Subscriber added for %s (total: %d)",
                event_type.value,
                len(self._subscribers[event_type]),
            )

    def subscribe_all(self, callback: EventCallback) -> None:
        """Subscribe a callback to every ``EventType``.

        Forwarders that fan events out to a single listener should use this so
        new event types reach them automatically; downstream handlers filter by
        type internally.
        """
        for event_type in EventType:
            self.subscribe(event_type, callback)

    def emit(self, event: OptimizationEvent) -> None:
        """Emit an event to all subscribers.

        Args:
            event: Event to emit
        """
        with self._lock:
            callbacks = self._subscribers[event.type].copy()

        for callback in callbacks:
            try:
                callback(event)
            except Exception:
                logger.exception("Error in event callback for %s", event.type.value)
