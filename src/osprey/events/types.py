"""Typed event classes for Osprey event streaming.

This module defines all event types used in Osprey's unified event streaming system.
Events are dataclasses that serialize to dicts for transport via LangGraph streaming.

Event Categories:
- Status Events: General status updates during execution
- Phase Lifecycle Events: Phase transitions in agent execution
- Capability Events: Capability execution lifecycle
- LLM Events: LLM API call tracking
- Tool/Code Events: External tool and code execution
- Control Flow Events: Approval workflow
- Result Events: Final execution results and errors

Usage:
    from osprey.events.types import StatusEvent, OspreyEvent

    event = StatusEvent(message="Processing...", component="router")
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass
class BaseEvent:
    """Base class for all Osprey events.

    Attributes:
        timestamp: When the event was created
        component: The component that emitted this event (e.g., "router", "classifier")
    """

    timestamp: datetime = field(default_factory=datetime.now)
    component: str = ""


# -----------------------------------------------------------------------------
# Status Events
# -----------------------------------------------------------------------------


@dataclass
class StatusEvent(BaseEvent):
    """Status update during execution.

    Used for general progress updates, log messages, and status changes.

    Attributes:
        message: The status message
        level: Severity/type of the status (info, warning, error, debug, success, status)
        phase: Current execution phase (e.g., "Task Preparation", "Execution")
        step: Current step number (1-based)
        total_steps: Total number of steps
    """

    message: str = ""
    level: Literal["info", "warning", "error", "debug", "success", "status", "key_info"] = "info"
    phase: str | None = None
    step: int | None = None
    total_steps: int | None = None


# -----------------------------------------------------------------------------
# Phase Lifecycle Events
# -----------------------------------------------------------------------------


@dataclass
class PhaseStartEvent(BaseEvent):
    """Phase transition in agent execution.

    Emitted when a major phase of execution begins.

    Attributes:
        phase: The phase being started
        description: Human-readable description of the phase
    """

    phase: Literal["task_extraction", "classification", "planning", "execution", "response"] = (
        "execution"
    )
    description: str = ""


@dataclass
class PhaseCompleteEvent(BaseEvent):
    """Phase completion.

    Emitted when a major phase of execution completes.

    Attributes:
        phase: The phase that completed
        duration_ms: How long the phase took in milliseconds
        success: Whether the phase completed successfully
    """

    phase: Literal["task_extraction", "classification", "planning", "execution", "response"] = (
        "execution"
    )
    duration_ms: int = 0
    success: bool = True


# -----------------------------------------------------------------------------
# Data Output Events
# -----------------------------------------------------------------------------


@dataclass
class TaskExtractedEvent(BaseEvent):
    """Task extraction completed with output.

    Emitted when task extraction produces an actionable task.

    Attributes:
        task: The extracted actionable task string
        depends_on_chat_history: Whether task references prior conversation
        depends_on_user_memory: Whether task uses user memory/preferences
    """

    task: str = ""
    depends_on_chat_history: bool = False
    depends_on_user_memory: bool = False


@dataclass
class CapabilitiesSelectedEvent(BaseEvent):
    """Capability classification completed with output.

    Emitted when classification selects active capabilities.

    Attributes:
        capability_names: List of selected capability names
        all_capability_names: List of all available capability names (for UI)
    """

    capability_names: list[str] = field(default_factory=list)
    all_capability_names: list[str] = field(default_factory=list)


@dataclass
class PlanCreatedEvent(BaseEvent):
    """Orchestration completed with execution plan.

    Emitted when orchestration creates an execution plan.

    Attributes:
        steps: List of execution steps (each a dict with capability_name, etc.)
    """

    steps: list[dict[str, Any]] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Capability Events
# -----------------------------------------------------------------------------


@dataclass
class CapabilityStartEvent(BaseEvent):
    """Capability execution started.

    Emitted when a capability begins execution.

    Attributes:
        capability_name: Name of the capability being executed
        step_number: Current step in the execution plan (1-based)
        total_steps: Total number of steps in the plan
        description: Human-readable description of what the capability does
    """

    capability_name: str = ""
    step_number: int = 0
    total_steps: int = 0
    description: str = ""


@dataclass
class CapabilityCompleteEvent(BaseEvent):
    """Capability execution completed.

    Emitted when a capability finishes execution.

    Attributes:
        capability_name: Name of the capability that completed
        success: Whether the capability executed successfully
        duration_ms: How long the execution took in milliseconds
        error_message: Error message if success is False
    """

    capability_name: str = ""
    success: bool = True
    duration_ms: int = 0
    error_message: str | None = None


# -----------------------------------------------------------------------------
# LLM Events
# -----------------------------------------------------------------------------


@dataclass
class LLMRequestEvent(BaseEvent):
    """LLM API call started.

    Emitted when an LLM request is made.

    Attributes:
        prompt_preview: First N characters of the prompt (for logging)
        prompt_length: Total length of the prompt in characters
        model: Model identifier (e.g., "gpt-4", "claude-3-opus")
        provider: Provider name (e.g., "openai", "anthropic")
        full_prompt: Complete prompt text for TUI display
        key: Optional key for accumulating multiple prompts (e.g., capability name)
    """

    prompt_preview: str = ""
    prompt_length: int = 0
    model: str = ""
    provider: str = ""
    full_prompt: str = ""
    key: str = ""


@dataclass
class LLMResponseEvent(BaseEvent):
    """LLM API response received.

    Emitted when an LLM response is received.

    Attributes:
        response_preview: First N characters of the response (for logging)
        response_length: Total length of the response in characters
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        thinking_tokens: Number of thinking tokens (for reasoning models)
        cost_usd: Estimated cost in USD
        duration_ms: How long the request took in milliseconds
        full_response: Complete response text for TUI display
        key: Optional key for accumulating multiple responses (e.g., capability name)
    """

    response_preview: str = ""
    response_length: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int | None = None
    cost_usd: float | None = None
    duration_ms: int = 0
    full_response: str = ""
    key: str = ""


# -----------------------------------------------------------------------------
# Tool/Code Events
# -----------------------------------------------------------------------------


@dataclass
class ToolUseEvent(BaseEvent):
    """External tool invocation (MCP, Claude Code, etc.).

    Emitted when an external tool is called.

    Attributes:
        tool_name: Name of the tool being invoked
        tool_input: Input parameters passed to the tool
    """

    tool_name: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResultEvent(BaseEvent):
    """Tool execution result.

    Emitted when a tool returns a result.

    Attributes:
        tool_name: Name of the tool that executed
        result_preview: First N characters of the result
        is_error: Whether the tool returned an error
    """

    tool_name: str = ""
    result_preview: str = ""
    is_error: bool = False


@dataclass
class CodeGeneratedEvent(BaseEvent):
    """Code generation completed.

    Emitted after code generation finishes successfully.
    Signals TUI to finalize widget (close stream, update title, auto-collapse).

    Attributes:
        code: Full generated code
        attempt: Attempt number (1-based)
        success: Whether generation was successful
        language: Programming language (kept for backward compatibility)
    """

    code: str = ""
    attempt: int = 1
    success: bool = True
    language: str = "python"


@dataclass
class CodeExecutedEvent(BaseEvent):
    """Code was executed.

    Emitted when generated code is executed.

    Attributes:
        success: Whether the code executed successfully
        output_preview: First N characters of the output
        error_message: Error message if success is False
    """

    success: bool = True
    output_preview: str = ""
    error_message: str | None = None


@dataclass
class CodeGenerationStartEvent(BaseEvent):
    """Code generation attempt started.

    Emitted BEFORE any LLM tokens are streamed to enable TUI to create
    separate widgets per retry attempt without race conditions.

    This event is emitted synchronously before model.astream() begins,
    guaranteeing that the TUI can finalize the previous widget and create
    a new CollapsibleCodeMessage widget before any code tokens arrive.

    Attributes:
        attempt: The retry attempt number (1 for first attempt, 2+ for retries)
        is_retry: Whether this is a retry attempt (convenience flag)
    """

    attempt: int = 1
    is_retry: bool = False


# -----------------------------------------------------------------------------
# Control Flow Events
# -----------------------------------------------------------------------------


@dataclass
class ApprovalRequiredEvent(BaseEvent):
    """User approval required.

    Emitted when the system needs user approval to proceed.

    Attributes:
        action_description: Description of the action requiring approval
        approval_type: Type of approval needed
    """

    action_description: str = ""
    approval_type: Literal["execution", "modification", "external"] = "execution"


@dataclass
class ApprovalReceivedEvent(BaseEvent):
    """User approval received.

    Emitted when the user responds to an approval request.

    Attributes:
        approved: Whether the user approved the action
        user_message: Optional message from the user
    """

    approved: bool = True
    user_message: str | None = None


# -----------------------------------------------------------------------------
# Result Events
# -----------------------------------------------------------------------------


@dataclass
class ResultEvent(BaseEvent):
    """Final execution result.

    Emitted when execution completes with a final result.

    Attributes:
        success: Whether execution was successful
        response: The final response/result
        duration_ms: Total execution time in milliseconds
        total_cost_usd: Total estimated cost in USD
        capabilities_used: List of capabilities that were executed
    """

    success: bool = True
    response: str = ""
    duration_ms: int = 0
    total_cost_usd: float | None = None
    capabilities_used: list[str] = field(default_factory=list)


@dataclass
class ErrorEvent(BaseEvent):
    """Error during execution.

    Emitted when an error occurs during execution.

    Attributes:
        error_type: Classification of the error (e.g., "ValidationError", "TimeoutError")
        error_message: Human-readable error message
        recoverable: Whether the error is recoverable
        stack_trace: Optional stack trace for debugging
    """

    error_type: str = ""
    error_message: str = ""
    recoverable: bool = False
    stack_trace: str | None = None


# -----------------------------------------------------------------------------
# Union Type
# -----------------------------------------------------------------------------

OspreyEvent = (
    StatusEvent
    | PhaseStartEvent
    | PhaseCompleteEvent
    | TaskExtractedEvent
    | CapabilitiesSelectedEvent
    | PlanCreatedEvent
    | CapabilityStartEvent
    | CapabilityCompleteEvent
    | LLMRequestEvent
    | LLMResponseEvent
    | ToolUseEvent
    | ToolResultEvent
    | CodeGeneratedEvent
    | CodeExecutedEvent
    | CodeGenerationStartEvent
    | ApprovalRequiredEvent
    | ApprovalReceivedEvent
    | ResultEvent
    | ErrorEvent
)
"""Union type of all Osprey events for type hints."""
