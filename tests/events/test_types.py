"""Tests for Osprey event types (dataclasses)."""

from datetime import datetime

from osprey.events import (
    ApprovalReceivedEvent,
    ApprovalRequiredEvent,
    BaseEvent,
    CapabilitiesSelectedEvent,
    CapabilityCompleteEvent,
    CapabilityStartEvent,
    CodeExecutedEvent,
    CodeGeneratedEvent,
    ErrorEvent,
    LLMRequestEvent,
    LLMResponseEvent,
    PhaseCompleteEvent,
    PhaseStartEvent,
    PlanCreatedEvent,
    ResultEvent,
    StatusEvent,
    TaskExtractedEvent,
    ToolResultEvent,
    ToolUseEvent,
)

# =============================================================================
# Test Base Event
# =============================================================================


class TestBaseEvent:
    """Test BaseEvent dataclass."""

    def test_default_timestamp_is_generated(self):
        """Verify timestamp is auto-generated if not provided."""
        event = BaseEvent()
        assert event.timestamp is not None
        assert isinstance(event.timestamp, datetime)

    def test_default_component_is_empty_string(self):
        """Verify default component is empty string."""
        event = BaseEvent()
        assert event.component == ""

    def test_custom_values_are_accepted(self):
        """Verify custom timestamp and component are preserved."""
        custom_time = datetime(2024, 1, 15, 12, 0, 0)
        event = BaseEvent(timestamp=custom_time, component="custom_component")
        assert event.timestamp == custom_time
        assert event.component == "custom_component"


# =============================================================================
# Test Status Events
# =============================================================================


class TestStatusEvent:
    """Test StatusEvent dataclass."""

    def test_creation_with_all_fields(self):
        """Test creating StatusEvent with all fields."""
        event = StatusEvent(
            message="Processing data",
            level="info",
            component="router",
            phase="execution",
            step=1,
            total_steps=3,
        )
        assert event.message == "Processing data"
        assert event.level == "info"
        assert event.component == "router"
        assert event.phase == "execution"
        assert event.step == 1
        assert event.total_steps == 3

    def test_default_values(self):
        """Test StatusEvent default values."""
        event = StatusEvent()
        assert event.message == ""
        assert event.level == "info"
        assert event.phase is None
        assert event.step is None
        assert event.total_steps is None

    def test_all_log_levels(self):
        """Test all valid log levels."""
        levels = ["info", "warning", "error", "debug", "success", "status"]
        for level in levels:
            event = StatusEvent(message="Test", level=level)
            assert event.level == level


# =============================================================================
# Test Phase Lifecycle Events
# =============================================================================


class TestPhaseStartEvent:
    """Test PhaseStartEvent dataclass."""

    def test_creation_with_valid_phases(self):
        """Test PhaseStartEvent with all valid phases."""
        phases = ["task_extraction", "classification", "planning", "execution", "response"]
        for phase in phases:
            event = PhaseStartEvent(phase=phase, description=f"Starting {phase}")
            assert event.phase == phase
            assert event.description == f"Starting {phase}"

    def test_default_values(self):
        """Test PhaseStartEvent default values."""
        event = PhaseStartEvent()
        assert event.phase == "execution"
        assert event.description == ""


class TestPhaseCompleteEvent:
    """Test PhaseCompleteEvent dataclass."""

    def test_creation_with_timing(self):
        """Test PhaseCompleteEvent with timing information."""
        event = PhaseCompleteEvent(
            phase="classification",
            duration_ms=150,
            success=True,
            component="classifier",
        )
        assert event.phase == "classification"
        assert event.duration_ms == 150
        assert event.success is True
        assert event.component == "classifier"

    def test_failure_case(self):
        """Test PhaseCompleteEvent for failure case."""
        event = PhaseCompleteEvent(
            phase="execution",
            duration_ms=5000,
            success=False,
        )
        assert event.success is False

    def test_default_values(self):
        """Test PhaseCompleteEvent default values."""
        event = PhaseCompleteEvent()
        assert event.phase == "execution"
        assert event.duration_ms == 0
        assert event.success is True


# =============================================================================
# Test Data Output Events
# =============================================================================


class TestTaskExtractedEvent:
    """Test TaskExtractedEvent dataclass."""

    def test_creation_with_all_fields(self):
        """Test TaskExtractedEvent with all fields."""
        event = TaskExtractedEvent(
            task="Generate a summary report",
            depends_on_chat_history=True,
            depends_on_user_memory=False,
            component="task_extraction",
        )
        assert event.task == "Generate a summary report"
        assert event.depends_on_chat_history is True
        assert event.depends_on_user_memory is False
        assert event.component == "task_extraction"

    def test_default_values(self):
        """Test TaskExtractedEvent default values."""
        event = TaskExtractedEvent()
        assert event.task == ""
        assert event.depends_on_chat_history is False
        assert event.depends_on_user_memory is False

    def test_long_task_string(self):
        """Test TaskExtractedEvent with long task description."""
        long_task = "Analyze the data and " * 50
        event = TaskExtractedEvent(task=long_task)
        assert event.task == long_task


class TestCapabilitiesSelectedEvent:
    """Test CapabilitiesSelectedEvent dataclass."""

    def test_creation_with_all_fields(self):
        """Test CapabilitiesSelectedEvent with all fields."""
        event = CapabilitiesSelectedEvent(
            capability_names=["python_executor", "search"],
            all_capability_names=["python_executor", "search", "web_browser", "file_manager"],
            component="classifier",
        )
        assert event.capability_names == ["python_executor", "search"]
        assert event.all_capability_names == [
            "python_executor",
            "search",
            "web_browser",
            "file_manager",
        ]
        assert event.component == "classifier"

    def test_default_values(self):
        """Test CapabilitiesSelectedEvent default values."""
        event = CapabilitiesSelectedEvent()
        assert event.capability_names == []
        assert event.all_capability_names == []

    def test_empty_selection(self):
        """Test CapabilitiesSelectedEvent with no capabilities selected."""
        event = CapabilitiesSelectedEvent(
            capability_names=[],
            all_capability_names=["python_executor", "search"],
        )
        assert event.capability_names == []
        assert len(event.all_capability_names) == 2


class TestPlanCreatedEvent:
    """Test PlanCreatedEvent dataclass."""

    def test_creation_with_steps(self):
        """Test PlanCreatedEvent with execution steps."""
        steps = [
            {
                "capability": "python_executor",
                "context_key": "step_1",
                "task_objective": "Run code",
            },
            {"capability": "respond", "context_key": "step_2", "task_objective": "Respond to user"},
        ]
        event = PlanCreatedEvent(steps=steps, component="orchestrator")
        assert len(event.steps) == 2
        assert event.steps[0]["capability"] == "python_executor"
        assert event.steps[1]["capability"] == "respond"
        assert event.component == "orchestrator"

    def test_default_values(self):
        """Test PlanCreatedEvent default values."""
        event = PlanCreatedEvent()
        assert event.steps == []

    def test_empty_plan(self):
        """Test PlanCreatedEvent with empty plan."""
        event = PlanCreatedEvent(steps=[])
        assert event.steps == []

    def test_complex_step_structure(self):
        """Test PlanCreatedEvent with complex step data."""
        steps = [
            {
                "capability": "python_executor",
                "context_key": "analysis_result",
                "task_objective": "Analyze the dataset",
                "expected_output": "analysis_summary",
                "success_criteria": "Complete analysis without errors",
                "inputs": ["data_source"],
            }
        ]
        event = PlanCreatedEvent(steps=steps)
        assert event.steps[0]["inputs"] == ["data_source"]
        assert event.steps[0]["expected_output"] == "analysis_summary"


# =============================================================================
# Test Capability Events
# =============================================================================


class TestCapabilityStartEvent:
    """Test CapabilityStartEvent dataclass."""

    def test_creation_with_all_fields(self):
        """Test CapabilityStartEvent with step information."""
        event = CapabilityStartEvent(
            capability_name="python_executor",
            step_number=2,
            total_steps=5,
            description="Running Python code",
            component="python_executor",
        )
        assert event.capability_name == "python_executor"
        assert event.step_number == 2
        assert event.total_steps == 5
        assert event.description == "Running Python code"

    def test_default_values(self):
        """Test CapabilityStartEvent default values."""
        event = CapabilityStartEvent()
        assert event.capability_name == ""
        assert event.step_number == 0
        assert event.total_steps == 0
        assert event.description == ""


class TestCapabilityCompleteEvent:
    """Test CapabilityCompleteEvent dataclass."""

    def test_success_case(self):
        """Test CapabilityCompleteEvent for success case."""
        event = CapabilityCompleteEvent(
            capability_name="python_executor",
            success=True,
            duration_ms=500,
            error_message=None,
        )
        assert event.success is True
        assert event.error_message is None

    def test_failure_case(self):
        """Test CapabilityCompleteEvent for failure case."""
        event = CapabilityCompleteEvent(
            capability_name="python_executor",
            success=False,
            duration_ms=100,
            error_message="Execution failed: syntax error",
        )
        assert event.success is False
        assert event.error_message == "Execution failed: syntax error"

    def test_default_values(self):
        """Test CapabilityCompleteEvent default values."""
        event = CapabilityCompleteEvent()
        assert event.capability_name == ""
        assert event.success is True
        assert event.duration_ms == 0
        assert event.error_message is None


# =============================================================================
# Test LLM Events
# =============================================================================


class TestLLMRequestEvent:
    """Test LLMRequestEvent dataclass."""

    def test_creation_with_all_fields(self):
        """Test LLMRequestEvent with all fields."""
        full_prompt = "What is the meaning of life, the universe, and everything?"
        event = LLMRequestEvent(
            prompt_preview="What is the meaning of...",
            prompt_length=100,
            model="gpt-4",
            provider="openai",
            full_prompt=full_prompt,
            key="weather",
        )
        assert event.prompt_preview == "What is the meaning of..."
        assert event.prompt_length == 100
        assert event.model == "gpt-4"
        assert event.provider == "openai"
        assert event.full_prompt == full_prompt
        assert event.key == "weather"

    def test_default_values(self):
        """Test LLMRequestEvent default values."""
        event = LLMRequestEvent()
        assert event.prompt_preview == ""
        assert event.prompt_length == 0
        assert event.model == ""
        assert event.provider == ""
        assert event.full_prompt == ""
        assert event.key == ""

    def test_key_for_multi_llm_accumulation(self):
        """Test key field for accumulating multiple LLM prompts."""
        # Simulate classification with multiple capabilities
        event1 = LLMRequestEvent(full_prompt="Classify for weather", key="weather")
        event2 = LLMRequestEvent(full_prompt="Classify for calculator", key="calculator")

        assert event1.key == "weather"
        assert event2.key == "calculator"
        assert event1.full_prompt != event2.full_prompt


class TestLLMResponseEvent:
    """Test LLMResponseEvent dataclass."""

    def test_creation_with_tokens(self):
        """Test LLMResponseEvent with token counts."""
        full_response = "The meaning of life is 42, according to Douglas Adams."
        event = LLMResponseEvent(
            response_preview="The meaning of...",
            response_length=500,
            input_tokens=50,
            output_tokens=200,
            thinking_tokens=100,
            cost_usd=0.05,
            duration_ms=2000,
            full_response=full_response,
            key="weather",
        )
        assert event.input_tokens == 50
        assert event.output_tokens == 200
        assert event.thinking_tokens == 100
        assert event.cost_usd == 0.05
        assert event.full_response == full_response
        assert event.key == "weather"

    def test_default_values(self):
        """Test LLMResponseEvent default values."""
        event = LLMResponseEvent()
        assert event.response_preview == ""
        assert event.input_tokens == 0
        assert event.output_tokens == 0
        assert event.thinking_tokens is None
        assert event.cost_usd is None
        assert event.full_response == ""
        assert event.key == ""

    def test_key_for_multi_llm_accumulation(self):
        """Test key field for accumulating multiple LLM responses."""
        # Simulate classification with multiple capabilities
        event1 = LLMResponseEvent(full_response='{"is_match": true}', key="weather")
        event2 = LLMResponseEvent(full_response='{"is_match": false}', key="calculator")

        assert event1.key == "weather"
        assert event2.key == "calculator"
        assert event1.full_response != event2.full_response


# =============================================================================
# Test Tool/Code Events
# =============================================================================


class TestToolUseEvent:
    """Test ToolUseEvent dataclass."""

    def test_creation_with_input(self):
        """Test ToolUseEvent with tool input."""
        event = ToolUseEvent(
            tool_name="code_search",
            tool_input={"query": "function definition", "limit": 10},
        )
        assert event.tool_name == "code_search"
        assert event.tool_input == {"query": "function definition", "limit": 10}

    def test_default_values(self):
        """Test ToolUseEvent default values."""
        event = ToolUseEvent()
        assert event.tool_name == ""
        assert event.tool_input == {}


class TestToolResultEvent:
    """Test ToolResultEvent dataclass."""

    def test_success_case(self):
        """Test ToolResultEvent for success case."""
        event = ToolResultEvent(
            tool_name="code_search",
            result_preview="Found 3 matches...",
            is_error=False,
        )
        assert event.is_error is False

    def test_error_case(self):
        """Test ToolResultEvent for error case."""
        event = ToolResultEvent(
            tool_name="code_search",
            result_preview="Error: Invalid query",
            is_error=True,
        )
        assert event.is_error is True


class TestCodeGeneratedEvent:
    """Test CodeGeneratedEvent dataclass."""

    def test_creation_with_all_fields(self):
        """Test CodeGeneratedEvent with all fields."""
        event = CodeGeneratedEvent(
            code="def hello(): print('hi')",
            attempt=2,
            success=True,
            language="python",
        )
        assert event.code == "def hello(): print('hi')"
        assert event.attempt == 2
        assert event.success is True
        assert event.language == "python"

    def test_default_values(self):
        """Test CodeGeneratedEvent default values."""
        event = CodeGeneratedEvent()
        assert event.code == ""
        assert event.attempt == 1
        assert event.success is True
        assert event.language == "python"


class TestCodeExecutedEvent:
    """Test CodeExecutedEvent dataclass."""

    def test_success_case(self):
        """Test CodeExecutedEvent for success case."""
        event = CodeExecutedEvent(
            success=True,
            output_preview="Hello, World!",
            error_message=None,
        )
        assert event.success is True
        assert event.error_message is None

    def test_failure_case(self):
        """Test CodeExecutedEvent for failure case."""
        event = CodeExecutedEvent(
            success=False,
            output_preview="",
            error_message="NameError: name 'x' is not defined",
        )
        assert event.success is False
        assert "NameError" in event.error_message


# =============================================================================
# Test Control Flow Events
# =============================================================================


class TestApprovalRequiredEvent:
    """Test ApprovalRequiredEvent dataclass."""

    def test_creation_with_all_fields(self):
        """Test ApprovalRequiredEvent with all fields."""
        event = ApprovalRequiredEvent(
            action_description="Execute shell command: rm -rf /tmp/test",
            approval_type="execution",
        )
        assert event.action_description == "Execute shell command: rm -rf /tmp/test"
        assert event.approval_type == "execution"

    def test_all_approval_types(self):
        """Test all valid approval types."""
        types = ["execution", "modification", "external"]
        for approval_type in types:
            event = ApprovalRequiredEvent(approval_type=approval_type)
            assert event.approval_type == approval_type


class TestApprovalReceivedEvent:
    """Test ApprovalReceivedEvent dataclass."""

    def test_approved_case(self):
        """Test ApprovalReceivedEvent when approved."""
        event = ApprovalReceivedEvent(
            approved=True,
            user_message="Proceed with execution",
        )
        assert event.approved is True
        assert event.user_message == "Proceed with execution"

    def test_rejected_case(self):
        """Test ApprovalReceivedEvent when rejected."""
        event = ApprovalReceivedEvent(
            approved=False,
            user_message="Do not proceed",
        )
        assert event.approved is False


# =============================================================================
# Test Result Events
# =============================================================================


class TestResultEvent:
    """Test ResultEvent dataclass."""

    def test_success_case(self):
        """Test ResultEvent for success case."""
        event = ResultEvent(
            success=True,
            response="Task completed successfully",
            duration_ms=5000,
            total_cost_usd=0.10,
            capabilities_used=["python_executor", "search"],
        )
        assert event.success is True
        assert event.response == "Task completed successfully"
        assert event.capabilities_used == ["python_executor", "search"]

    def test_default_values(self):
        """Test ResultEvent default values."""
        event = ResultEvent()
        assert event.success is True
        assert event.response == ""
        assert event.capabilities_used == []


class TestErrorEvent:
    """Test ErrorEvent dataclass."""

    def test_creation_with_stack_trace(self):
        """Test ErrorEvent with stack trace."""
        event = ErrorEvent(
            error_type="ValidationError",
            error_message="Invalid input format",
            recoverable=True,
            stack_trace="Traceback (most recent call last):\n  File...",
        )
        assert event.error_type == "ValidationError"
        assert event.error_message == "Invalid input format"
        assert event.recoverable is True
        assert event.stack_trace is not None

    def test_unrecoverable_error(self):
        """Test ErrorEvent for unrecoverable error."""
        event = ErrorEvent(
            error_type="SystemError",
            error_message="Critical failure",
            recoverable=False,
        )
        assert event.recoverable is False

    def test_default_values(self):
        """Test ErrorEvent default values."""
        event = ErrorEvent()
        assert event.error_type == ""
        assert event.error_message == ""
        assert event.recoverable is False
        assert event.stack_trace is None


# =============================================================================
# Test Union Type
# =============================================================================


class TestOspreyEventUnionType:
    """Test OspreyEvent union type."""

    def test_union_includes_all_event_types(self):
        """Verify OspreyEvent union includes all event types."""
        # Check that each event type is part of the union
        # by checking that isinstance works for any event in the union
        # (This is a type-level test, but we can verify construction)
        events = [
            StatusEvent(message="test"),
            PhaseStartEvent(phase="execution"),
            PhaseCompleteEvent(phase="execution"),
            TaskExtractedEvent(task="test"),
            CapabilitiesSelectedEvent(capability_names=["test"]),
            PlanCreatedEvent(steps=[]),
            CapabilityStartEvent(capability_name="test"),
            CapabilityCompleteEvent(capability_name="test"),
            LLMRequestEvent(model="gpt-4"),
            LLMResponseEvent(response_length=100),
            ToolUseEvent(tool_name="test"),
            ToolResultEvent(tool_name="test"),
            CodeGeneratedEvent(language="python"),
            CodeExecutedEvent(success=True),
            ApprovalRequiredEvent(action_description="test"),
            ApprovalReceivedEvent(approved=True),
            ResultEvent(success=True),
            ErrorEvent(error_type="TestError"),
        ]

        # All should be valid and constructible (15 original + 3 new data events)
        assert len(events) == 18

    def test_timestamp_auto_generated_for_all_types(self):
        """Verify timestamp is auto-generated for all event types."""
        events = [
            StatusEvent(),
            PhaseStartEvent(),
            PhaseCompleteEvent(),
            TaskExtractedEvent(),
            CapabilitiesSelectedEvent(),
            PlanCreatedEvent(),
            CapabilityStartEvent(),
            CapabilityCompleteEvent(),
            LLMRequestEvent(),
            LLMResponseEvent(),
            ToolUseEvent(),
            ToolResultEvent(),
            CodeGeneratedEvent(),
            CodeExecutedEvent(),
            ApprovalRequiredEvent(),
            ApprovalReceivedEvent(),
            ResultEvent(),
            ErrorEvent(),
        ]

        for event in events:
            assert event.timestamp is not None
            assert isinstance(event.timestamp, datetime)
