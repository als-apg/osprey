"""Pytest fixtures for event streaming tests."""

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from osprey.events import (
    ApprovalReceivedEvent,
    ApprovalRequiredEvent,
    CapabilityCompleteEvent,
    CapabilityStartEvent,
    CodeExecutedEvent,
    CodeGeneratedEvent,
    ErrorEvent,
    EventEmitter,
    LLMRequestEvent,
    LLMResponseEvent,
    PhaseCompleteEvent,
    PhaseStartEvent,
    ResultEvent,
    StatusEvent,
    ToolResultEvent,
    ToolUseEvent,
    clear_fallback_handlers,
)


@pytest.fixture
def sample_events() -> dict[str, Any]:
    """Provide sample events of each type for testing."""
    now = datetime.now()
    return {
        "status": StatusEvent(
            message="Processing data",
            level="info",
            component="router",
            phase="execution",
            step=1,
            total_steps=3,
            timestamp=now,
        ),
        "phase_start": PhaseStartEvent(
            phase="classification",
            description="Classifying user request",
            component="classifier",
            timestamp=now,
        ),
        "phase_complete": PhaseCompleteEvent(
            phase="classification",
            duration_ms=150,
            success=True,
            component="classifier",
            timestamp=now,
        ),
        "capability_start": CapabilityStartEvent(
            capability_name="python_executor",
            step_number=1,
            total_steps=3,
            description="Running Python code",
            component="python_executor",
            timestamp=now,
        ),
        "capability_complete": CapabilityCompleteEvent(
            capability_name="python_executor",
            success=True,
            duration_ms=500,
            error_message=None,
            component="python_executor",
            timestamp=now,
        ),
        "llm_request": LLMRequestEvent(
            prompt_preview="What is the meaning of...",
            prompt_length=100,
            model="gpt-4",
            provider="openai",
            component="router",
            timestamp=now,
        ),
        "llm_response": LLMResponseEvent(
            response_preview="The meaning of...",
            response_length=500,
            input_tokens=50,
            output_tokens=200,
            thinking_tokens=None,
            cost_usd=0.05,
            duration_ms=2000,
            component="router",
            timestamp=now,
        ),
        "tool_use": ToolUseEvent(
            tool_name="code_search",
            tool_input={"query": "function definition"},
            component="router",
            timestamp=now,
        ),
        "tool_result": ToolResultEvent(
            tool_name="code_search",
            result_preview="Found 3 matches...",
            is_error=False,
            component="router",
            timestamp=now,
        ),
        "code_generated": CodeGeneratedEvent(
            language="python",
            code_preview="def hello(): ...",
            code_length=50,
            component="python_executor",
            timestamp=now,
        ),
        "code_executed": CodeExecutedEvent(
            success=True,
            output_preview="Hello, World!",
            error_message=None,
            component="python_executor",
            timestamp=now,
        ),
        "approval_required": ApprovalRequiredEvent(
            action_description="Execute shell command",
            approval_type="execution",
            component="approval_gateway",
            timestamp=now,
        ),
        "approval_received": ApprovalReceivedEvent(
            approved=True,
            user_message="Proceed with execution",
            component="approval_gateway",
            timestamp=now,
        ),
        "result": ResultEvent(
            success=True,
            response="Task completed successfully",
            duration_ms=5000,
            total_cost_usd=0.10,
            capabilities_used=["python_executor", "search"],
            component="respond",
            timestamp=now,
        ),
        "error": ErrorEvent(
            error_type="ValidationError",
            error_message="Invalid input format",
            recoverable=True,
            stack_trace="Traceback...",
            component="router",
            timestamp=now,
        ),
    }


@pytest.fixture
def event_emitter() -> EventEmitter:
    """Provide a pre-configured EventEmitter instance."""
    return EventEmitter("test_component")


@pytest.fixture
def captured_events() -> list[dict[str, Any]]:
    """Provide a list to capture emitted events."""
    return []


@pytest.fixture
def fallback_handler_with_capture(captured_events):
    """Register a fallback handler that captures events."""
    from osprey.events import register_fallback_handler

    def handler(event_dict: dict[str, Any]) -> None:
        captured_events.append(event_dict)

    unregister = register_fallback_handler(handler)
    yield handler
    unregister()


@pytest.fixture(autouse=True)
def clear_handlers_between_tests():
    """Ensure fallback handlers are cleared between tests."""
    clear_fallback_handlers()
    yield
    clear_fallback_handlers()


@pytest.fixture
def mock_stream_writer():
    """Provide a mock LangGraph stream writer."""
    return MagicMock()


@pytest.fixture
def mock_graph():
    """Provide a mock LangGraph graph for streaming tests."""
    mock = MagicMock()

    async def mock_astream(*args, **kwargs):
        """Async generator that yields test data."""
        # Yield a status event
        yield "custom", {
            "event_class": "StatusEvent",
            "message": "Test status",
            "level": "info",
            "component": "test",
            "timestamp": datetime.now().isoformat(),
        }
        # Yield an LLM token
        mock_message = MagicMock()
        mock_message.content = "Hello"
        yield "messages", (mock_message, {"run_id": "test"})

    mock.astream = mock_astream
    return mock
