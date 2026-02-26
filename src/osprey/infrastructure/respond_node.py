"""
Respond Capability

This capability responds to user queries by generating appropriate responses - both technical
queries requiring execution context and conversational queries that don't. It adaptively
chooses the appropriate response strategy based on query type and available context.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from osprey.base import BaseCapability
from osprey.base.decorators import capability_node
from osprey.base.errors import ErrorClassification, ErrorSeverity
from osprey.context.context_manager import ContextManager
from osprey.models import get_langchain_model
from osprey.prompts.loader import get_framework_prompts
from osprey.registry import get_registry
from osprey.state import (
    AgentState,
    ChatHistoryFormatter,
    StateManager,
    populate_legacy_fields_from_artifacts,
)
from osprey.utils.config import get_model_config


@dataclass
class ResponseContext:
    """Container for all information needed for response generation.

    Aggregates all relevant information from the agent state for
    comprehensive response generation.

    :param current_task: The current task being addressed
    :type current_task: str
    :param execution_history: List of executed steps
    :type execution_history: List[Any]
    :param relevant_context: Context data relevant to the response (list of summary dicts)
    :type relevant_context: List[Dict[str, Any]]
    :param is_killed: Whether execution was terminated
    :type is_killed: bool
    :param kill_reason: Reason for termination if applicable
    :type kill_reason: Optional[str]
    :param capabilities_overview: Overview of available capabilities
    :type capabilities_overview: Optional[str]
    :param total_steps_executed: Total number of steps executed
    :type total_steps_executed: int
    :param execution_start_time: When execution started
    :type execution_start_time: Optional[float]
    :param reclassification_count: Number of reclassification attempts
    :type reclassification_count: int
    :param current_date: Current date for temporal context
    :type current_date: str
    :param figures_available: Number of figures available for display
    :type figures_available: int
    :param commands_available: Number of launchable commands available
    :type commands_available: int
    :param notebooks_available: Number of notebook links available
    :type notebooks_available: int
    :param interface_context: Interface type (openwebui, cli, etc.)
    :type interface_context: str
    :param chat_history: Formatted chat history when task depends on conversation context
    :type chat_history: Optional[str]
    """

    current_task: str
    execution_history: list[Any]
    relevant_context: list[dict[str, Any]]
    is_killed: bool
    kill_reason: str | None
    capabilities_overview: str | None
    total_steps_executed: int
    execution_start_time: float | None
    reclassification_count: int
    current_date: str
    figures_available: int
    commands_available: int
    notebooks_available: int
    interface_context: str
    chat_history: str | None = None


# --- Convention-Based Capability Definition ---


@capability_node
class RespondCapability(BaseCapability):
    """Respond to user queries with appropriate response strategy.

    Generates comprehensive responses for both technical queries requiring
    execution context and conversational queries. Adapts response strategy
    based on available context and execution history.
    """

    name = "respond"
    description = "Respond to user queries by generating appropriate responses for both technical and conversational questions"
    provides = ["FINAL_RESPONSE"]
    requires = []  # Can work with any previous context, or none at all

    async def execute(self) -> dict[str, Any]:
        """Generate appropriate response using unified dynamic prompt construction.

        :return: State update with generated response
        :rtype: Dict[str, Any]
        """
        state = self._state

        # Get unified logger with automatic streaming support
        logger = self.get_logger()

        # Get step (injected by decorator)
        step = self._step

        try:
            logger.status("Gathering information for response...")

            # Gather all available information
            response_context = _gather_information(state, logger)

            logger.status("Generating response...")

            # Build prompt dynamically based on available information
            prompt = _get_base_system_prompt(response_context.current_task, response_context)

            # Emit LLM prompt event for TUI display
            logger.emit_llm_request(prompt)

            # Get streaming LangChain model for response generation
            model = get_langchain_model(model_config=get_model_config("response"))

            # Build messages for LangChain format
            messages = [HumanMessage(content=prompt)]

            # Stream response - LangGraph captures AIMessageChunks in messages mode
            # This enables real-time token streaming to frontends (CLI/TUI)
            response_chunks: list[str] = []
            async for chunk in model.astream(messages):
                if chunk.content:
                    response_chunks.append(chunk.content)

            response_text = "".join(response_chunks)

            if not response_text:
                raise Exception("No response from LLM, please try again.")

            # Note: LLM response is streamed via LangGraph's messages mode,
            # so we don't emit LLMResponseEvent here (would cause duplicate display)

            logger.status("Response generated")

            # Use actual task objective if available, otherwise describe response mode
            if step and step.get("task_objective"):
                task_objective = step.get("task_objective")
            else:
                # Use response context to determine correct mode description
                task_objective = (
                    "conversational query"
                    if response_context.execution_history == []
                    else "technical query"
                )
            logger.info(f"Generated response for: '{task_objective}'")

            # Populate legacy fields from unified artifacts for backward compatibility
            # This ensures OpenWebUI and other interfaces can access figures/notebooks/commands
            ui_artifacts = state.get("ui_artifacts", [])
            legacy_updates = {}
            if ui_artifacts:
                legacy_updates = populate_legacy_fields_from_artifacts(ui_artifacts)

            # Return native LangGraph pattern: AIMessage added to messages list
            # Include legacy field updates for backward compatibility
            return {"messages": [AIMessage(content=response_text)], **legacy_updates}

        except Exception as e:
            logger.error(f"Error in response generation: {e}")

            # Since respond node goes directly to END (bypasses router error handling),
            # we need to handle errors internally and provide a fallback response
            error_message = str(e)

            # Generic error fallback - avoid hardcoding specific error codes or providers
            fallback_response = (
                "❌ **Response Generation Failed**\n\n"
                "Your request may have been processed successfully, but the final "
                "response could not be generated.\n\n"
                "**Common causes:**\n"
                "• Rate limits or high demand on the LLM provider (usually temporary)\n"
                "• Network connectivity issues\n"
                "• Service availability problems\n\n"
                "**Next steps:**\n"
                "• Please try your request again in a few minutes\n"
                "• If the problem persists, please contact support\n\n"
                f"**Technical details:** {error_message}"
            )

            # Return the fallback response as an AIMessage instead of raising
            return {"messages": [AIMessage(content=fallback_response)]}

    # Optional: Add error classification if needed
    @staticmethod
    def classify_error(exc: Exception, context: dict):
        """Response generation error classification."""
        return ErrorClassification(
            severity=ErrorSeverity.CRITICAL,
            user_message=f"Failed to generate response: {str(exc)}",
            metadata={"technical_details": str(exc)},
        )

    def _create_orchestrator_guide(self):
        """Get orchestrator guide from prompt builder."""

        prompt_provider = get_framework_prompts()  # Registry will determine the right provider
        response_builder = prompt_provider.get_response_generation_prompt_builder()

        return response_builder.get_orchestrator_guide()

    def _create_classifier_guide(self):
        """Get classifier guide from prompt builder."""

        prompt_provider = get_framework_prompts()  # Registry will determine the right provider
        response_builder = prompt_provider.get_response_generation_prompt_builder()

        return response_builder.get_classifier_guide()


# --- Helper Functions ---


def _gather_information(state: AgentState, logger=None) -> ResponseContext:
    """Gather all relevant information for response generation.

    :param state: Current agent state
    :type state: AgentState
    :return: Complete response context
    :rtype: ResponseContext
    """

    # Extract context data and determine response mode
    context_manager = ContextManager(state)
    current_step = StateManager.get_current_step(state)
    relevant_context = context_manager.get_summaries(current_step)

    # Determine response mode and prepare appropriate data
    response_mode = _determine_response_mode(state, current_step)

    if response_mode == "conversational":
        execution_history = []
        capabilities_overview = _get_capabilities_overview()
        if logger:
            logger.info("Using conversational response mode (no execution context)")
    else:  # technical mode
        execution_history = _get_execution_history(state)
        capabilities_overview = None
        if logger:
            logger.info(f"Using technical response mode (context type: {response_mode})")

    # Populate legacy fields from unified artifacts (for backward compatibility)
    # This derives ui_captured_figures, ui_launchable_commands, ui_captured_notebooks
    # from the canonical ui_artifacts field
    ui_artifacts = state.get("ui_artifacts", [])
    if ui_artifacts:
        legacy_fields = populate_legacy_fields_from_artifacts(ui_artifacts)
        ui_figures = legacy_fields["ui_captured_figures"]
        ui_commands = legacy_fields["ui_launchable_commands"]
        ui_notebooks = legacy_fields["ui_captured_notebooks"]
    else:
        # Fall back to direct legacy field access (for old capabilities)
        ui_figures = state.get("ui_captured_figures", [])
        ui_commands = state.get("ui_launchable_commands", [])
        ui_notebooks = state.get("ui_captured_notebooks", [])

    figures_available = len(ui_figures)
    commands_available = len(ui_commands)
    notebooks_available = len(ui_notebooks)

    # Log notebook availability for debugging
    if logger:
        logger.debug(f"Respond node found {len(ui_notebooks)} notebook links")

    # Get interface context from configurable
    from osprey.utils.config import get_interface_context

    interface_context = get_interface_context()

    # Include chat history if task depends on conversation context
    chat_history = None
    if state.get("task_depends_on_chat_history", False):
        messages = state.get("messages", [])
        if messages:
            chat_history = ChatHistoryFormatter.format_for_llm(messages)
            if logger:
                logger.info(
                    f"Including chat history ({len(messages)} messages) for context-dependent task"
                )

    return ResponseContext(
        current_task=state.get("task_current_task", "General information request"),
        execution_history=execution_history,
        relevant_context=relevant_context,
        is_killed=state.get("control_is_killed", False),
        kill_reason=state.get("control_kill_reason"),
        capabilities_overview=capabilities_overview,
        total_steps_executed=StateManager.get_current_step_index(state),
        execution_start_time=state.get("execution_start_time"),
        reclassification_count=state.get("control_reclassification_count", 0),
        current_date=datetime.now().strftime("%Y-%m-%d"),
        figures_available=figures_available,
        commands_available=commands_available,
        notebooks_available=notebooks_available,
        interface_context=interface_context,
        chat_history=chat_history,
    )


def _determine_response_mode(state: AgentState, current_step: dict[str, Any]) -> str:
    """Determine the appropriate response mode based on available context.

    Args:
        state: Current agent state
        current_step: Current execution step

    Returns:
        Response mode: "conversational", "specific_context", or "general_context"
    """

    # Check if current step has specific context inputs assigned
    has_step_inputs = current_step and current_step.get("inputs")

    # Check if any capability context data exists in the system
    has_capability_data = bool(state.get("capability_context_data", {}))

    if not has_step_inputs and not has_capability_data:
        return "conversational"
    elif has_step_inputs:
        return "specific_context"
    else:
        return "general_context"


def _get_capabilities_overview() -> str:
    """Get capabilities overview for conversational mode."""
    try:
        return get_registry().get_capabilities_overview()
    except Exception:
        return "General AI Assistant capabilities available"


def _get_execution_history(state: AgentState) -> list[dict[str, Any]]:
    """Get execution history from state for technical mode."""
    execution_step_results = state.get("execution_step_results", {})
    ordered_results = sorted(
        execution_step_results.items(), key=lambda x: x[1].get("step_index", 0)
    )
    return [result for step_key, result in ordered_results]


def _get_base_system_prompt(current_task: str, info=None) -> str:
    """Get the base system prompt with task context.

    :param current_task: The current task being addressed
    :type current_task: str
    :param info: Optional response context information
    :type info: Optional[ResponseContext]
    :return: Complete system prompt
    :rtype: str
    """

    prompt_provider = get_framework_prompts()
    response_builder = prompt_provider.get_response_generation_prompt_builder()

    return response_builder.build_prompt(current_task=current_task, info=info)
