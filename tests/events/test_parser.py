"""Tests for event parser (parse_event function)."""

from datetime import datetime

from osprey.events import (
    CapabilitiesSelectedEvent,
    CapabilityCompleteEvent,
    CapabilityStartEvent,
    ErrorEvent,
    PhaseCompleteEvent,
    PhaseStartEvent,
    PlanCreatedEvent,
    ResultEvent,
    StatusEvent,
    TaskExtractedEvent,
    is_osprey_event,
    parse_event,
)
from osprey.events.parser import EVENT_CLASSES

# =============================================================================
# Test parse_event Function
# =============================================================================


class TestParseEvent:
    """Test parse_event function."""

    def test_parse_status_event(self):
        """Test parsing StatusEvent from dict."""
        data = {
            "event_class": "StatusEvent",
            "message": "Processing data",
            "level": "info",
            "component": "router",
            "phase": "execution",
            "step": 1,
            "total_steps": 3,
            "timestamp": datetime.now().isoformat(),
        }

        event = parse_event(data)

        assert isinstance(event, StatusEvent)
        assert event.message == "Processing data"
        assert event.level == "info"
        assert event.component == "router"
        assert event.phase == "execution"
        assert event.step == 1
        assert event.total_steps == 3

    def test_parse_phase_start_event(self):
        """Test parsing PhaseStartEvent from dict."""
        data = {
            "event_class": "PhaseStartEvent",
            "phase": "classification",
            "description": "Classifying user request",
            "component": "classifier",
            "timestamp": datetime.now().isoformat(),
        }

        event = parse_event(data)

        assert isinstance(event, PhaseStartEvent)
        assert event.phase == "classification"
        assert event.description == "Classifying user request"

    def test_parse_phase_complete_event(self):
        """Test parsing PhaseCompleteEvent from dict."""
        data = {
            "event_class": "PhaseCompleteEvent",
            "phase": "classification",
            "duration_ms": 150,
            "success": True,
            "component": "classifier",
            "timestamp": datetime.now().isoformat(),
        }

        event = parse_event(data)

        assert isinstance(event, PhaseCompleteEvent)
        assert event.phase == "classification"
        assert event.duration_ms == 150
        assert event.success is True

    def test_parse_capability_start_event(self):
        """Test parsing CapabilityStartEvent from dict."""
        data = {
            "event_class": "CapabilityStartEvent",
            "capability_name": "python_executor",
            "step_number": 1,
            "total_steps": 3,
            "description": "Running Python code",
            "component": "python_executor",
            "timestamp": datetime.now().isoformat(),
        }

        event = parse_event(data)

        assert isinstance(event, CapabilityStartEvent)
        assert event.capability_name == "python_executor"
        assert event.step_number == 1
        assert event.total_steps == 3

    def test_parse_capability_complete_event(self):
        """Test parsing CapabilityCompleteEvent from dict."""
        data = {
            "event_class": "CapabilityCompleteEvent",
            "capability_name": "python_executor",
            "success": True,
            "duration_ms": 500,
            "error_message": None,
            "component": "python_executor",
            "timestamp": datetime.now().isoformat(),
        }

        event = parse_event(data)

        assert isinstance(event, CapabilityCompleteEvent)
        assert event.capability_name == "python_executor"
        assert event.success is True
        assert event.duration_ms == 500

    def test_parse_error_event(self):
        """Test parsing ErrorEvent with all fields."""
        data = {
            "event_class": "ErrorEvent",
            "error_type": "ValidationError",
            "error_message": "Invalid input format",
            "recoverable": True,
            "stack_trace": "Traceback...",
            "component": "router",
            "timestamp": datetime.now().isoformat(),
        }

        event = parse_event(data)

        assert isinstance(event, ErrorEvent)
        assert event.error_type == "ValidationError"
        assert event.error_message == "Invalid input format"
        assert event.recoverable is True
        assert event.stack_trace == "Traceback..."

    def test_parse_result_event_with_list(self):
        """Test parsing ResultEvent with list field."""
        data = {
            "event_class": "ResultEvent",
            "success": True,
            "response": "Task completed",
            "duration_ms": 5000,
            "total_cost_usd": 0.10,
            "capabilities_used": ["python_executor", "search"],
            "component": "respond",
            "timestamp": datetime.now().isoformat(),
        }

        event = parse_event(data)

        assert isinstance(event, ResultEvent)
        assert event.success is True
        assert event.capabilities_used == ["python_executor", "search"]

    def test_parse_task_extracted_event(self):
        """Test parsing TaskExtractedEvent from dict."""
        data = {
            "event_class": "TaskExtractedEvent",
            "task": "Generate a summary report",
            "depends_on_chat_history": True,
            "depends_on_user_memory": False,
            "component": "task_extraction",
            "timestamp": datetime.now().isoformat(),
        }

        event = parse_event(data)

        assert isinstance(event, TaskExtractedEvent)
        assert event.task == "Generate a summary report"
        assert event.depends_on_chat_history is True
        assert event.depends_on_user_memory is False
        assert event.component == "task_extraction"

    def test_parse_capabilities_selected_event(self):
        """Test parsing CapabilitiesSelectedEvent from dict."""
        data = {
            "event_class": "CapabilitiesSelectedEvent",
            "capability_names": ["python_executor", "search"],
            "all_capability_names": ["python_executor", "search", "web_browser"],
            "component": "classifier",
            "timestamp": datetime.now().isoformat(),
        }

        event = parse_event(data)

        assert isinstance(event, CapabilitiesSelectedEvent)
        assert event.capability_names == ["python_executor", "search"]
        assert event.all_capability_names == ["python_executor", "search", "web_browser"]
        assert event.component == "classifier"

    def test_parse_plan_created_event(self):
        """Test parsing PlanCreatedEvent from dict."""
        steps = [
            {"capability": "python_executor", "context_key": "step_1"},
            {"capability": "respond", "context_key": "step_2"},
        ]
        data = {
            "event_class": "PlanCreatedEvent",
            "steps": steps,
            "component": "orchestrator",
            "timestamp": datetime.now().isoformat(),
        }

        event = parse_event(data)

        assert isinstance(event, PlanCreatedEvent)
        assert len(event.steps) == 2
        assert event.steps[0]["capability"] == "python_executor"
        assert event.component == "orchestrator"


# =============================================================================
# Test Invalid Input Handling
# =============================================================================


class TestInvalidInputHandling:
    """Test handling of invalid input."""

    def test_parse_returns_none_for_non_dict(self):
        """Test that parse_event returns None for non-dict input."""
        assert parse_event("not a dict") is None
        assert parse_event(123) is None
        assert parse_event(None) is None
        assert parse_event([]) is None

    def test_parse_returns_none_for_missing_class(self):
        """Test that parse_event returns None when event_class is missing."""
        data = {
            "message": "Test",
            "level": "info",
        }
        assert parse_event(data) is None

    def test_parse_returns_none_for_unknown_class(self):
        """Test that parse_event returns None for unknown event_class."""
        data = {
            "event_class": "UnknownEvent",
            "message": "Test",
        }
        assert parse_event(data) is None

    def test_parse_handles_extra_fields_gracefully(self):
        """Test that extra fields are ignored gracefully."""
        data = {
            "event_class": "StatusEvent",
            "message": "Test",
            "level": "info",
            "component": "test",
            "timestamp": datetime.now().isoformat(),
            "extra_field": "should be ignored",
            "another_extra": 123,
        }

        event = parse_event(data)

        assert isinstance(event, StatusEvent)
        assert event.message == "Test"
        # Extra fields should not appear on the event
        assert not hasattr(event, "extra_field")
        assert not hasattr(event, "another_extra")

    def test_parse_handles_missing_optional_fields(self):
        """Test that missing optional fields use defaults."""
        data = {
            "event_class": "StatusEvent",
            "message": "Test",
            # Missing: level, phase, step, total_steps
            "timestamp": datetime.now().isoformat(),
        }

        event = parse_event(data)

        assert isinstance(event, StatusEvent)
        assert event.message == "Test"
        # Should use defaults
        assert event.level == "info"
        assert event.phase is None


# =============================================================================
# Test Timestamp Parsing
# =============================================================================


class TestTimestampParsing:
    """Test timestamp parsing from ISO format."""

    def test_parse_timestamp_from_iso(self):
        """Test that ISO timestamp string is converted back."""
        now = datetime(2024, 1, 15, 12, 30, 45)
        data = {
            "event_class": "StatusEvent",
            "message": "Test",
            "timestamp": now.isoformat(),
        }

        event = parse_event(data)

        assert isinstance(event.timestamp, datetime)
        assert event.timestamp == now

    def test_parse_handles_invalid_timestamp(self):
        """Test that invalid timestamp uses current time."""
        data = {
            "event_class": "StatusEvent",
            "message": "Test",
            "timestamp": "not-a-valid-timestamp",
        }

        before = datetime.now()
        event = parse_event(data)
        after = datetime.now()

        assert isinstance(event.timestamp, datetime)
        assert before <= event.timestamp <= after

    def test_parse_handles_datetime_object(self):
        """Test that datetime object is preserved."""
        now = datetime.now()
        data = {
            "event_class": "StatusEvent",
            "message": "Test",
            "timestamp": now,
        }

        event = parse_event(data)

        assert event.timestamp == now


# =============================================================================
# Test is_osprey_event Function
# =============================================================================


class TestIsOspreyEvent:
    """Test is_osprey_event function."""

    def test_is_osprey_event_true_for_valid(self):
        """Test that is_osprey_event returns True for valid events."""
        for event_class_name in EVENT_CLASSES:
            data = {"event_class": event_class_name, "message": "test"}
            assert is_osprey_event(data) is True

    def test_is_osprey_event_false_for_invalid(self):
        """Test that is_osprey_event returns False for invalid data."""
        # Non-dict
        assert is_osprey_event("string") is False
        assert is_osprey_event(123) is False
        assert is_osprey_event(None) is False

        # Missing event_class
        assert is_osprey_event({"message": "test"}) is False

        # Unknown event_class
        assert is_osprey_event({"event_class": "UnknownEvent"}) is False


# =============================================================================
# Test Roundtrip (Serialize -> Parse)
# =============================================================================


class TestEventRoundtrip:
    """Test that events can be serialized and parsed back."""

    def test_status_event_roundtrip(self):
        """Test StatusEvent roundtrip."""
        from osprey.events import EventEmitter

        original = StatusEvent(
            message="Test message",
            level="warning",
            component="test",
            phase="execution",
            step=1,
            total_steps=3,
        )

        # Serialize
        emitter = EventEmitter("test")
        serialized = emitter._serialize(original)

        # Parse back
        parsed = parse_event(serialized)

        assert isinstance(parsed, StatusEvent)
        assert parsed.message == original.message
        assert parsed.level == original.level
        assert parsed.component == original.component
        assert parsed.phase == original.phase
        assert parsed.step == original.step
        assert parsed.total_steps == original.total_steps

    def test_all_event_types_roundtrip(self, sample_events):
        """Test roundtrip for all event types."""
        from osprey.events import EventEmitter

        emitter = EventEmitter("test")

        for event_type, original in sample_events.items():
            # Serialize
            serialized = emitter._serialize(original)

            # Parse back
            parsed = parse_event(serialized)

            assert parsed is not None, f"Failed to parse {event_type}"
            assert type(parsed).__name__ == type(original).__name__
            assert parsed.component == original.component

    def test_roundtrip_preserves_all_fields(self):
        """Test that roundtrip preserves all field values."""
        from osprey.events import EventEmitter, ToolUseEvent

        emitter = EventEmitter("test")

        original = ToolUseEvent(
            tool_name="code_search",
            tool_input={"query": "function definition", "limit": 10},
            component="router",
        )

        serialized = emitter._serialize(original)
        parsed = parse_event(serialized)

        assert isinstance(parsed, ToolUseEvent)
        assert parsed.tool_name == original.tool_name
        assert parsed.tool_input == original.tool_input


# =============================================================================
# Test EVENT_CLASSES Registry
# =============================================================================


class TestEventClassesRegistry:
    """Test EVENT_CLASSES registry."""

    def test_all_event_types_registered(self):
        """Verify all expected event types are registered."""
        expected_classes = [
            "StatusEvent",
            "PhaseStartEvent",
            "PhaseCompleteEvent",
            "TaskExtractedEvent",
            "CapabilitiesSelectedEvent",
            "PlanCreatedEvent",
            "CapabilityStartEvent",
            "CapabilityCompleteEvent",
            "LLMRequestEvent",
            "LLMResponseEvent",
            "ToolUseEvent",
            "ToolResultEvent",
            "CodeGeneratedEvent",
            "CodeExecutedEvent",
            "ApprovalRequiredEvent",
            "ApprovalReceivedEvent",
            "ResultEvent",
            "ErrorEvent",
        ]

        for class_name in expected_classes:
            assert class_name in EVENT_CLASSES, f"{class_name} not in EVENT_CLASSES"

    def test_registry_maps_to_correct_classes(self):
        """Verify registry maps to correct classes."""
        assert EVENT_CLASSES["StatusEvent"] == StatusEvent
        assert EVENT_CLASSES["PhaseStartEvent"] == PhaseStartEvent
        assert EVENT_CLASSES["TaskExtractedEvent"] == TaskExtractedEvent
        assert EVENT_CLASSES["CapabilitiesSelectedEvent"] == CapabilitiesSelectedEvent
        assert EVENT_CLASSES["PlanCreatedEvent"] == PlanCreatedEvent
        assert EVENT_CLASSES["ErrorEvent"] == ErrorEvent
        assert EVENT_CLASSES["ResultEvent"] == ResultEvent
