"""Tests for EventEmitter class."""

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

from osprey.events import (
    EventEmitter,
    StatusEvent,
    clear_fallback_handlers,
    register_fallback_handler,
)

# =============================================================================
# Test EventEmitter Initialization
# =============================================================================


class TestEventEmitterInitialization:
    """Test EventEmitter initialization."""

    def test_emitter_stores_component_name(self):
        """Verify component name is stored."""
        emitter = EventEmitter("test_component")
        assert emitter.component == "test_component"

    def test_emitter_with_different_components(self):
        """Test multiple emitters with different component names."""
        emitter1 = EventEmitter("router")
        emitter2 = EventEmitter("classifier")
        assert emitter1.component == "router"
        assert emitter2.component == "classifier"


# =============================================================================
# Test Event Emission
# =============================================================================


class TestEventEmission:
    """Test event emission behavior."""

    def test_emit_sets_component_if_missing(self, captured_events, fallback_handler_with_capture):
        """Test that emit sets component if event has none."""
        emitter = EventEmitter("default_component")
        event = StatusEvent(message="Test message")
        assert event.component == ""

        emitter.emit(event)

        # After emit, component should be set on the event
        assert event.component == "default_component"
        # And in the captured event
        assert len(captured_events) == 1
        assert captured_events[0]["component"] == "default_component"

    def test_emit_preserves_existing_component(
        self, captured_events, fallback_handler_with_capture
    ):
        """Test that emit preserves existing component."""
        emitter = EventEmitter("default_component")
        event = StatusEvent(message="Test message", component="custom_component")

        emitter.emit(event)

        assert len(captured_events) == 1
        assert captured_events[0]["component"] == "custom_component"

    def test_emit_serializes_event_with_class_name(
        self, captured_events, fallback_handler_with_capture
    ):
        """Test that serialized event includes event_class field."""
        emitter = EventEmitter("test")
        event = StatusEvent(message="Test")

        emitter.emit(event)

        assert len(captured_events) == 1
        assert captured_events[0]["event_class"] == "StatusEvent"

    def test_emit_converts_timestamp_to_iso(self, captured_events, fallback_handler_with_capture):
        """Test that timestamp is converted to ISO format string."""
        emitter = EventEmitter("test")
        now = datetime.now()
        event = StatusEvent(message="Test", timestamp=now)

        emitter.emit(event)

        assert len(captured_events) == 1
        assert isinstance(captured_events[0]["timestamp"], str)
        # Should be parseable back to datetime
        parsed = datetime.fromisoformat(captured_events[0]["timestamp"])
        assert parsed == now


# =============================================================================
# Test Fallback Handlers
# =============================================================================


class TestFallbackHandlers:
    """Test fallback handler registration and behavior."""

    def test_register_fallback_handler(self):
        """Test registering a fallback handler."""
        captured = []

        def handler(event_dict: dict[str, Any]) -> None:
            captured.append(event_dict)

        unregister = register_fallback_handler(handler)

        # Emit an event
        emitter = EventEmitter("test")
        emitter.emit(StatusEvent(message="Test"))

        assert len(captured) == 1
        assert captured[0]["message"] == "Test"

        unregister()

    def test_unregister_fallback_handler(self):
        """Test that unregistering removes the handler."""
        captured = []

        def handler(event_dict: dict[str, Any]) -> None:
            captured.append(event_dict)

        unregister = register_fallback_handler(handler)
        unregister()

        # Emit an event after unregistering
        emitter = EventEmitter("test")
        emitter.emit(StatusEvent(message="Test"))

        # Should not be captured
        assert len(captured) == 0

    def test_multiple_fallback_handlers(self):
        """Test that multiple handlers are all called."""
        captured1 = []
        captured2 = []

        def handler1(event_dict: dict[str, Any]) -> None:
            captured1.append(event_dict)

        def handler2(event_dict: dict[str, Any]) -> None:
            captured2.append(event_dict)

        unregister1 = register_fallback_handler(handler1)
        unregister2 = register_fallback_handler(handler2)

        emitter = EventEmitter("test")
        emitter.emit(StatusEvent(message="Test"))

        assert len(captured1) == 1
        assert len(captured2) == 1

        unregister1()
        unregister2()

    def test_clear_fallback_handlers(self):
        """Test that clear removes all handlers."""
        captured = []

        def handler(event_dict: dict[str, Any]) -> None:
            captured.append(event_dict)

        register_fallback_handler(handler)
        register_fallback_handler(handler)

        clear_fallback_handlers()

        emitter = EventEmitter("test")
        emitter.emit(StatusEvent(message="Test"))

        assert len(captured) == 0

    def test_fallback_handler_exception_doesnt_crash(self):
        """Test that exception in handler doesn't crash emission."""
        captured = []

        def bad_handler(event_dict: dict[str, Any]) -> None:
            raise RuntimeError("Handler error")

        def good_handler(event_dict: dict[str, Any]) -> None:
            captured.append(event_dict)

        unregister1 = register_fallback_handler(bad_handler)
        unregister2 = register_fallback_handler(good_handler)

        emitter = EventEmitter("test")
        # Should not raise
        emitter.emit(StatusEvent(message="Test"))

        # Good handler should still receive event
        assert len(captured) == 1

        unregister1()
        unregister2()


# =============================================================================
# Test LangGraph Integration
# =============================================================================


class TestLangGraphIntegration:
    """Test LangGraph streaming integration."""

    def test_emit_uses_langgraph_when_available(self):
        """Test that emit uses get_stream_writer when available."""
        mock_writer = MagicMock()

        # Patch at langgraph.config where it's imported from
        with patch("langgraph.config.get_stream_writer", return_value=mock_writer):
            emitter = EventEmitter("test")
            emitter.emit(StatusEvent(message="Test"))

            mock_writer.assert_called_once()
            call_args = mock_writer.call_args[0][0]
            assert call_args["message"] == "Test"
            assert call_args["event_class"] == "StatusEvent"

    def test_emit_falls_back_when_langgraph_not_available(
        self, captured_events, fallback_handler_with_capture
    ):
        """Test that emit uses fallback when LangGraph raises RuntimeError."""
        with patch(
            "langgraph.config.get_stream_writer",
            side_effect=RuntimeError("Not in graph context"),
        ):
            emitter = EventEmitter("test")
            emitter.emit(StatusEvent(message="Test"))

            # Should use fallback handler
            assert len(captured_events) == 1
            assert captured_events[0]["message"] == "Test"

    def test_emit_falls_back_when_langgraph_import_fails(
        self, captured_events, fallback_handler_with_capture
    ):
        """Test that emit uses fallback when LangGraph import fails."""
        with patch(
            "langgraph.config.get_stream_writer",
            side_effect=ImportError("No module named langgraph"),
        ):
            emitter = EventEmitter("test")
            emitter.emit(StatusEvent(message="Test"))

            # Should use fallback handler
            assert len(captured_events) == 1

    def test_emit_silent_noop_without_handlers(self):
        """Test that emit is silent when no handlers and no LangGraph."""
        with patch(
            "langgraph.config.get_stream_writer",
            side_effect=RuntimeError("Not in graph context"),
        ):
            clear_fallback_handlers()
            emitter = EventEmitter("test")
            # Should not raise - just silent no-op
            emitter.emit(StatusEvent(message="Test"))


# =============================================================================
# Test Serialization
# =============================================================================


class TestEventSerialization:
    """Test event serialization."""

    def test_serialize_includes_all_fields(self, captured_events, fallback_handler_with_capture):
        """Test that serialization includes all event fields."""
        emitter = EventEmitter("test")
        event = StatusEvent(
            message="Test message",
            level="warning",
            phase="execution",
            step=1,
            total_steps=3,
        )

        emitter.emit(event)

        serialized = captured_events[0]
        assert serialized["message"] == "Test message"
        assert serialized["level"] == "warning"
        assert serialized["phase"] == "execution"
        assert serialized["step"] == 1
        assert serialized["total_steps"] == 3
        assert "timestamp" in serialized
        assert "event_class" in serialized

    def test_serialize_handles_none_values(self, captured_events, fallback_handler_with_capture):
        """Test that serialization handles None values correctly."""
        emitter = EventEmitter("test")
        event = StatusEvent(
            message="Test",
            phase=None,
            step=None,
        )

        emitter.emit(event)

        serialized = captured_events[0]
        assert serialized["phase"] is None
        assert serialized["step"] is None

    def test_serialize_handles_list_fields(self, captured_events, fallback_handler_with_capture):
        """Test that serialization handles list fields."""
        from osprey.events import ResultEvent

        emitter = EventEmitter("test")
        event = ResultEvent(
            success=True,
            response="Done",
            capabilities_used=["python", "search"],
        )

        emitter.emit(event)

        serialized = captured_events[0]
        assert serialized["capabilities_used"] == ["python", "search"]

    def test_serialize_handles_dict_fields(self, captured_events, fallback_handler_with_capture):
        """Test that serialization handles dict fields."""
        from osprey.events import ToolUseEvent

        emitter = EventEmitter("test")
        event = ToolUseEvent(
            tool_name="search",
            tool_input={"query": "test", "limit": 10},
        )

        emitter.emit(event)

        serialized = captured_events[0]
        assert serialized["tool_input"] == {"query": "test", "limit": 10}
