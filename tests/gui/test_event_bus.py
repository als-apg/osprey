"""
Unit tests for Event Bus

Tests the event bus pattern implementation for decoupled component communication.
"""

import pytest

from osprey.interfaces.pyqt.event_bus import EventBus


class TestEventBus:
    """Test suite for EventBus class."""

    def test_initialization(self):
        """Test event bus initializes correctly."""
        bus = EventBus()
        assert bus._subscribers == {}
        assert bus._event_log == []
        assert not bus._logging_enabled

    def test_subscribe_single_handler(self):
        """Test subscribing a single handler to an event."""
        bus = EventBus()
        called = []

        def handler(data):
            called.append(data)

        bus.subscribe("test_event", handler)
        assert bus.get_subscriber_count("test_event") == 1

    def test_subscribe_multiple_handlers(self):
        """Test subscribing multiple handlers to same event."""
        bus = EventBus()
        calls = {"h1": [], "h2": []}

        def handler1(data):
            calls["h1"].append(data)

        def handler2(data):
            calls["h2"].append(data)

        bus.subscribe("test_event", handler1)
        bus.subscribe("test_event", handler2)

        assert bus.get_subscriber_count("test_event") == 2

    def test_subscribe_duplicate_handler(self):
        """Test that subscribing same handler twice doesn't duplicate."""
        bus = EventBus()
        called = []

        def handler(data):
            called.append(data)

        bus.subscribe("test_event", handler)
        bus.subscribe("test_event", handler)  # Subscribe again

        assert bus.get_subscriber_count("test_event") == 1

    def test_publish_with_data(self):
        """Test publishing event with data."""
        bus = EventBus()
        received_data = []

        def handler(data):
            received_data.append(data)

        bus.subscribe("test_event", handler)
        bus.publish("test_event", {"key": "value"})

        assert len(received_data) == 1
        assert received_data[0] == {"key": "value"}

    def test_publish_without_data(self):
        """Test publishing event without data."""
        bus = EventBus()
        called = []

        def handler():
            called.append(True)

        bus.subscribe("test_event", handler)
        bus.publish("test_event")

        assert len(called) == 1

    def test_publish_to_multiple_handlers(self):
        """Test that all subscribed handlers receive the event."""
        bus = EventBus()
        calls = {"h1": [], "h2": [], "h3": []}

        def handler1(data):
            calls["h1"].append(data)

        def handler2(data):
            calls["h2"].append(data)

        def handler3(data):
            calls["h3"].append(data)

        bus.subscribe("test_event", handler1)
        bus.subscribe("test_event", handler2)
        bus.subscribe("test_event", handler3)

        bus.publish("test_event", "test_data")

        assert calls["h1"] == ["test_data"]
        assert calls["h2"] == ["test_data"]
        assert calls["h3"] == ["test_data"]

    def test_publish_no_subscribers(self):
        """Test publishing to event with no subscribers doesn't error."""
        bus = EventBus()
        # Should not raise any exception
        bus.publish("nonexistent_event", "data")

    def test_unsubscribe(self):
        """Test unsubscribing a handler."""
        bus = EventBus()
        called = []

        def handler(data):
            called.append(data)

        bus.subscribe("test_event", handler)
        bus.publish("test_event", "first")

        bus.unsubscribe("test_event", handler)
        bus.publish("test_event", "second")

        # Should only have received first event
        assert called == ["first"]
        assert bus.get_subscriber_count("test_event") == 0

    def test_unsubscribe_nonexistent_handler(self):
        """Test unsubscribing handler that wasn't subscribed."""
        bus = EventBus()

        def handler(data):
            pass

        # Should not raise exception
        bus.unsubscribe("test_event", handler)

    def test_clear_subscribers_specific_event(self):
        """Test clearing subscribers for specific event."""
        bus = EventBus()

        def handler1(data):
            pass

        def handler2(data):
            pass

        bus.subscribe("event1", handler1)
        bus.subscribe("event2", handler2)

        bus.clear_subscribers("event1")

        assert bus.get_subscriber_count("event1") == 0
        assert bus.get_subscriber_count("event2") == 1

    def test_clear_all_subscribers(self):
        """Test clearing all subscribers."""
        bus = EventBus()

        def handler(data):
            pass

        bus.subscribe("event1", handler)
        bus.subscribe("event2", handler)
        bus.subscribe("event3", handler)

        bus.clear_subscribers()

        assert bus.get_subscriber_count("event1") == 0
        assert bus.get_subscriber_count("event2") == 0
        assert bus.get_subscriber_count("event3") == 0

    def test_event_logging_disabled_by_default(self):
        """Test that event logging is disabled by default."""
        bus = EventBus()

        def handler(data):
            pass

        bus.subscribe("test_event", handler)
        bus.publish("test_event", "data")

        assert len(bus.get_event_log()) == 0

    def test_event_logging_when_enabled(self):
        """Test event logging when enabled."""
        bus = EventBus()
        bus.enable_logging(True)

        def handler(data):
            pass

        bus.subscribe("test_event", handler)
        bus.publish("test_event", "data1")
        bus.publish("test_event", "data2")

        log = bus.get_event_log()
        assert len(log) == 2
        assert log[0] == {"type": "test_event", "data": "data1"}
        assert log[1] == {"type": "test_event", "data": "data2"}

    def test_event_logging_disable_clears_log(self):
        """Test that disabling logging clears the log."""
        bus = EventBus()
        bus.enable_logging(True)

        def handler(data):
            pass

        bus.subscribe("test_event", handler)
        bus.publish("test_event", "data")

        assert len(bus.get_event_log()) == 1

        bus.enable_logging(False)
        assert len(bus.get_event_log()) == 0

    def test_handler_exception_doesnt_stop_other_handlers(self):
        """Test that exception in one handler doesn't prevent others from running."""
        bus = EventBus()
        calls = {"h1": [], "h2": []}

        def handler1(data):
            calls["h1"].append(data)
            raise ValueError("Test error")

        def handler2(data):
            calls["h2"].append(data)

        bus.subscribe("test_event", handler1)
        bus.subscribe("test_event", handler2)

        bus.publish("test_event", "data")

        # Both handlers should have been called despite h1 raising exception
        assert calls["h1"] == ["data"]
        assert calls["h2"] == ["data"]

    def test_multiple_event_types(self):
        """Test handling multiple different event types."""
        bus = EventBus()
        calls = {"event1": [], "event2": [], "event3": []}

        def handler1(data):
            calls["event1"].append(data)

        def handler2(data):
            calls["event2"].append(data)

        def handler3(data):
            calls["event3"].append(data)

        bus.subscribe("event1", handler1)
        bus.subscribe("event2", handler2)
        bus.subscribe("event3", handler3)

        bus.publish("event1", "data1")
        bus.publish("event2", "data2")
        bus.publish("event3", "data3")

        assert calls["event1"] == ["data1"]
        assert calls["event2"] == ["data2"]
        assert calls["event3"] == ["data3"]

    def test_get_subscriber_count_nonexistent_event(self):
        """Test getting subscriber count for event with no subscribers."""
        bus = EventBus()
        assert bus.get_subscriber_count("nonexistent") == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
