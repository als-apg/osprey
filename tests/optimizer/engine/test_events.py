import logging

import pytest

from osprey_optimizer.events import (
    EventBus,
    EventType,
    OptimizationEvent,
    OptimizationStartedData,
    PhaseCompletedData,
)


class TestEventSystem:
    """Tests for the EventBus implementation logic."""

    def test_event_bus_routing(self) -> None:
        """Verify events are routed to correct subscribers."""
        bus = EventBus()
        received = []

        bus.subscribe(EventType.OPTIMIZATION_STARTED, lambda e: received.append("start"))
        bus.subscribe(EventType.PHASE_COMPLETED, lambda e: received.append("phase"))

        bus.emit(
            OptimizationEvent(
                type=EventType.OPTIMIZATION_STARTED, data=OptimizationStartedData(objective="test")
            )
        )
        assert received == ["start"]

        bus.emit(
            OptimizationEvent(
                type=EventType.PHASE_COMPLETED,
                data=PhaseCompletedData(phase="LHS", best_value=0.5),
            )
        )
        assert received == ["start", "phase"]

    def test_event_bus_error_isolation(self, caplog: pytest.LogCaptureFixture) -> None:
        """Ensure one crashing listener doesn't break the bus — and that the
        swallowed exception is logged with its traceback (a silent `pass` must
        fail here: a subscriber crashing unseen has bitten us before)."""
        bus = EventBus()

        def crasher(_event: OptimizationEvent) -> None:
            raise RuntimeError("Crash")

        bus.subscribe(EventType.OPTIMIZATION_STARTED, crasher)

        worked = False

        def success(_event: OptimizationEvent) -> None:
            nonlocal worked
            worked = True

        bus.subscribe(EventType.OPTIMIZATION_STARTED, success)

        try:
            with caplog.at_level(logging.ERROR, logger="osprey_optimizer.events"):
                bus.emit(
                    OptimizationEvent(
                        type=EventType.OPTIMIZATION_STARTED, data=OptimizationStartedData()
                    )
                )
        except Exception:
            pytest.fail("EventBus let an exception propagate out of emit()")

        assert worked
        # Select by level rather than unpacking caplog.records: emit() also logs
        # DEBUG lines, and whether those are captured depends on the ambient log
        # level left behind by earlier tests.
        [record] = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert record.exc_info is not None  # logger.exception, not logger.error
        assert EventType.OPTIMIZATION_STARTED.value in record.getMessage()
