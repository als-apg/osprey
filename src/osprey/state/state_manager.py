"""State Manager - LangGraph Native State Management.

This module provides comprehensive state management utilities for the Osprey
Agent Framework, implementing LangGraph-native patterns for optimal performance and
compatibility. The StateManager class serves as the primary interface for state
creation, manipulation, and context storage operations.

**Architecture Overview:**

The state management system is built around LangGraph's native patterns with a focus
on simplicity and performance. The design eliminates complex custom reducers by
leveraging LangGraph's automatic message handling and selective persistence strategy.

**Key Components:**

- :class:`StateManager`: Primary state management utilities and operations
- :func:`get_agent_control_defaults`: Configuration defaults with error handling
- :func:`get_execution_steps_summary`: Execution history formatting utilities

**Design Principles:**

1. **LangGraph Native**: Full compatibility with LangGraph's state management
2. **Selective Persistence**: Only capability context data persists across turns
3. **Flat Structure**: Simplified state structure without complex nesting
4. **Type Safety**: Comprehensive type annotations and validation
5. **Error Resilience**: Robust error handling with safe fallbacks

**State Lifecycle Management:**

The StateManager handles the complete state lifecycle:

1. **Fresh State Creation**: Initialize new conversation states with defaults
2. **Context Preservation**: Maintain capability context across conversation turns
3. **State Updates**: Provide utilities for LangGraph-compatible state updates
4. **Context Storage**: One-liner context storage for capability results

**Usage Patterns:**

The StateManager is designed for use throughout the framework infrastructure
and capabilities, providing consistent patterns for state operations.

.. note::
   The StateManager is optimized for LangGraph's native patterns and should be
   the primary interface for all state operations. Direct state manipulation
   may interfere with LangGraph's checkpointing and serialization.

.. seealso::
   :class:`osprey.state.AgentState` : Main state structure definition
   :class:`osprey.context.ContextManager` : Context data management
   :mod:`osprey.infrastructure.gateway` : Main entry point using StateManager
"""

from typing import TYPE_CHECKING, Any

# LangGraph native imports
from langchain_core.messages import BaseMessage

from osprey.base.planning import ExecutionPlan, PlannedStep
from osprey.context.base import CapabilityContext
from osprey.context.context_manager import ContextManager
from osprey.utils.config import get_agent_control_defaults as _get_agent_control_defaults
from osprey.utils.logger import get_logger

from .artifacts import ArtifactType, create_artifact
from .messages import MessageUtils
from .state import AgentState, StateUpdate

if TYPE_CHECKING:
    from osprey.context.context_manager import ContextManager

logger = get_logger(name="StateManager", color="white")


def get_agent_control_defaults() -> dict[str, Any]:
    """Get agent control configuration defaults with robust error handling.

    This function retrieves the default agent control configuration from the
    configuration system with comprehensive error handling and safe fallbacks. The
    configuration includes planning mode settings, EPICS execution controls, approval
    workflows, and execution limits.

    The function provides resilient configuration loading that ensures the framework
    can operate even if the configuration system encounters errors, using safe
    defaults that prioritize security and controlled execution.

    :return: Dictionary containing agent control defaults with all required fields
    :rtype: Dict[str, Any]

    .. note::
       The function includes comprehensive error handling that logs configuration
       loading failures and provides safe defaults to ensure framework stability.

    .. warning::
       Safe defaults prioritize security by disabling potentially dangerous operations
       like EPICS writes and enabling approval requirements for sensitive operations.

    Examples:
        Normal configuration loading::

            >>> defaults = get_agent_control_defaults()
            >>> defaults['planning_mode_enabled']
            False
            >>> defaults['epics_writes_enabled']
            False

        Configuration with approval settings::

            >>> defaults = get_agent_control_defaults()
            >>> defaults['approval_global_mode']
            'selective'
            >>> defaults['python_execution_approval_enabled']
            True

    .. seealso::
       :class:`osprey.state.AgentControlState` : Agent control state structure
       :mod:`configs.config` : Configuration system
    """
    try:
        return _get_agent_control_defaults()
    except Exception as e:
        logger.warning(f"Could not load agent control defaults: {e}")
        # Return safe defaults if config fails
        return {
            "planning_mode_enabled": False,
            "epics_writes_enabled": False,
            "approval_global_mode": "selective",
            "python_execution_approval_enabled": True,
            "python_execution_approval_mode": "all_code",
            "memory_approval_enabled": True,
            "max_reclassifications": 1,
            "max_planning_attempts": 2,
            "max_step_retries": 0,
            "max_execution_time_seconds": 300,
            "max_concurrent_classifications": 5,
            # Bypass configuration defaults
            "task_extraction_bypass_enabled": False,
            "capability_selection_bypass_enabled": False,
        }


class StateManager:
    """LangGraph-native state management utilities for conversational AI agents.

    This class provides a comprehensive suite of static utilities for managing
    conversational agent state using LangGraph's native patterns. The StateManager
    serves as the primary interface for state creation, manipulation, and context
    storage operations throughout the Osprey Agent Framework.

    **Core Functionality:**

    The StateManager implements a simplified state management approach that leverages
    LangGraph's native patterns for optimal performance and compatibility:

    - **Fresh State Creation**: Initialize new conversation states with proper defaults
    - **Context Persistence**: Maintain capability context data across conversation turns
    - **State Updates**: Generate LangGraph-compatible state update dictionaries
    - **Context Storage**: One-liner context storage for capability results
    - **Execution Tracking**: Utilities for accessing execution plans and current steps

    **Design Philosophy:**

    The StateManager follows a "batteries included" approach where common state
    operations are provided as simple, one-liner utilities. This eliminates the need
    for capabilities and infrastructure components to implement complex state logic.

    **State Lifecycle Management:**

    1. **Initialization**: create_fresh_state() creates new conversation states
    2. **Execution**: Utilities provide access to current execution context
    3. **Context Storage**: store_context() handles capability result persistence
    4. **State Updates**: Methods generate proper LangGraph state update dictionaries

    .. note::
       All StateManager methods are static utilities that do not require instantiation.
       The class serves as a namespace for state management operations.

    .. warning::
       StateManager utilities are optimized for LangGraph's native patterns. Using
       these utilities ensures compatibility with checkpointing and serialization.
       Direct state manipulation may cause issues.

    Examples:
        Creating fresh state for new conversation::

            >>> state = StateManager.create_fresh_state(
            ...     user_input="Find beam current data",
            ...     current_state=previous_state
            ... )

        Storing capability context results::

            >>> updates = StateManager.store_context(
            ...     state, "PV_ADDRESSES", "step1", pv_data
            ... )
            >>> return updates  # Compatible with LangGraph

        Accessing current execution step::

            >>> step = StateManager.get_current_step(state)
            >>> task_objective = step.get('task_objective')

    .. seealso::
       :class:`osprey.state.AgentState` : Main state structure managed by this class
       :class:`osprey.context.ContextManager` : Context data management utilities
       :func:`get_agent_control_defaults` : Configuration defaults for state creation
    """

    @staticmethod
    def create_fresh_state(user_input: str, current_state: AgentState | None = None) -> AgentState:
        """Create fresh agent state for a new conversation turn with selective persistence.

        This method creates a complete fresh state for LangGraph execution while
        preserving only the capability context data from the previous state. All
        execution-scoped fields are reset to their default values, ensuring clean
        state for each conversation turn while maintaining accumulated context.

        The state creation process handles the complete lifecycle initialization:

        1. **Message Initialization**: Creates properly formatted user message
        2. **Context Preservation**: Extracts and preserves capability context data
        3. **Default Population**: Initializes all execution fields with safe defaults
        4. **Configuration Loading**: Applies agent control defaults from configuration

        This approach ensures optimal performance by avoiding state bloat while
        maintaining the persistent context needed for multi-turn conversations.

        :param user_input: The user's message content for this conversation turn
        :type user_input: str
        :param current_state: Previous agent state to preserve context from (optional)
        :type current_state: Optional[AgentState]
        :return: Complete fresh state ready for LangGraph graph invocation
        :rtype: AgentState

        .. note::
           Only capability_context_data and session_state persist across conversation
           turns. All other fields including execution results, control state, and UI
           data are reset to defaults for optimal performance and state clarity.

        .. warning::
           The current_state parameter should be the complete previous AgentState.
           Partial state dictionaries may cause context preservation to fail silently.

        Examples:
            Creating state for new conversation::

                >>> state = StateManager.create_fresh_state("Find beam current PVs")
                >>> state['task_current_task'] is None
                True
                >>> len(state['messages'])
                1

            Preserving context from previous conversation::

                >>> new_state = StateManager.create_fresh_state(
                ...     "Show me the latest data",
                ...     current_state=previous_state
                ... )
                >>> # Context preserved but execution state reset
                >>> len(new_state['capability_context_data'])
                3  # Preserved from previous
                >>> new_state['execution_step_results']
                {}  # Reset to empty

        .. seealso::
           :class:`AgentState` : State structure created by this method
           :func:`get_agent_control_defaults` : Configuration defaults applied
           :class:`osprey.state.MessageUtils` : Message creation utilities
        """
        # Create initial message
        initial_message = MessageUtils.create_user_message(user_input)

        # Preserve persistent fields from previous state if available
        preserved_context_data = {}
        preserved_session_state = {}
        if current_state:
            if "capability_context_data" in current_state:
                preserved_context_data = current_state["capability_context_data"]
                logger.debug("Preserved capability_context_data from previous state")
            if "session_state" in current_state:
                preserved_session_state = current_state["session_state"]
                logger.debug("Preserved session_state from previous state")

        # Create complete fresh state with persistent fields preserved
        state = AgentState(
            # Messages (from MessagesState)
            messages=[initial_message],
            # Persistent fields - preserved from previous state
            capability_context_data=preserved_context_data,
            session_state=preserved_session_state,
            # Agent control state - reset to defaults each conversation turn
            agent_control=get_agent_control_defaults(),
            # Event accumulation - reset each execution
            status_updates=[],
            progress_events=[],
            # Task processing fields - reset to defaults
            task_current_task=None,
            task_depends_on_chat_history=False,
            task_depends_on_user_memory=False,
            task_custom_message=None,
            # Planning fields - reset to defaults
            planning_active_capabilities=[],
            planning_execution_plan=None,
            planning_current_step_index=0,
            # Execution fields - reset to defaults
            execution_step_results={},
            execution_last_result=None,
            execution_pending_approvals={},
            execution_start_time=None,
            execution_total_time=None,
            # Approval handling fields - reset to defaults
            approval_approved=None,
            approved_payload=None,
            # Control flow fields - reset to defaults
            control_reclassification_reason=None,
            control_reclassification_count=0,
            control_plans_created_count=0,
            control_current_step_retry_count=0,
            control_retry_count=0,
            control_has_error=False,
            control_error_info=None,
            control_last_error=None,
            control_max_retries=3,
            control_is_killed=False,
            control_kill_reason=None,
            control_is_awaiting_validation=False,
            control_validation_context=None,
            control_validation_timestamp=None,
            # UI result fields - reset to defaults
            ui_artifacts=[],  # Unified artifact registry
            # Legacy fields (maintained for backward compatibility)
            ui_captured_notebooks=[],
            ui_captured_figures=[],
            ui_launchable_commands=[],
            ui_agent_context=None,
            # Runtime metadata fields - reset to defaults
            runtime_checkpoint_metadata=None,
            runtime_info=None,
        )

        return state

    # ===== UTILITY METHODS =====

    @staticmethod
    def get_current_task(state: AgentState) -> str | None:
        """Get current task from state."""
        return state.get("task_current_task")

    @staticmethod
    def get_user_query(state: AgentState) -> str | None:
        """Get the user's query from the current conversation.

        Extracts the most recent user message from the conversation history,
        which represents the original user query that started the current
        conversation turn.

        Args:
            state: Current conversation state

        Returns:
            The user's query string, or None if no user messages exist
        """
        from .messages import ChatHistoryFormatter

        return ChatHistoryFormatter.get_latest_user_message(state.get("messages", []))

    @staticmethod
    def store_context(
        state: AgentState, context_type: str, context_key: str, context_object: CapabilityContext
    ) -> StateUpdate:
        """Store capability context data and return LangGraph-compatible state updates.

        This is the primary utility function that capabilities should use for storing
        context data in the agent state. The method provides a one-liner interface that
        handles all the complexity of context management, serialization, and state
        update generation for seamless integration with LangGraph's state system.

        The function performs the complete context storage workflow:

        1. **Context Manager Creation**: Initializes ContextManager from current state
        2. **Context Serialization**: Converts CapabilityContext to dictionary format
        3. **State Integration**: Merges context data into existing state structure
        4. **Update Generation**: Returns LangGraph-compatible state update dictionary

        This approach ensures that capability results are properly persisted across
        conversation turns while maintaining optimal serialization performance.

        :param state: Current agent state containing existing context data
        :type state: AgentState
        :param context_type: Context type identifier (e.g., "PV_ADDRESSES", "ANALYSIS_RESULTS")
        :type context_type: str
        :param context_key: Unique key for this context instance within the type
        :type context_key: str
        :param context_object: CapabilityContext object containing data to store
        :type context_object: CapabilityContext
        :return: State update dictionary for LangGraph automatic merging
        :rtype: StateUpdate

        .. note::
           This is the recommended pattern for all capability context storage. The
           returned dictionary can be directly returned from capability execute methods
           and will be automatically merged by LangGraph.

        .. warning::
           The context_key should be unique within the context_type to avoid
           overwriting existing context data. Consider using step context keys
           or timestamp-based keys for uniqueness.

        Examples:
            Basic capability context storage::

                >>> @staticmethod
                >>> async def execute(state: AgentState, **kwargs):
                ...     # Capability logic here
                ...     result = PVAddresses(pvs=found_pvs, description="Found PVs")
                ...
                ...     # Store and return state updates (one-liner)
                ...     step = StateManager.get_current_step(state)
                ...     return StateManager.store_context(
                ...         state, "PV_ADDRESSES", step.get('context_key'), result
                ...     )

            Context storage with custom key::

                >>> analysis_result = DataAnalysis(
                ...     results=processed_data,
                ...     timestamp=datetime.now()
                ... )
                >>> updates = StateManager.store_context(
                ...     state, "ANALYSIS", f"beam_analysis_{timestamp}", analysis_result
                ... )
                >>> return updates

        .. seealso::
           :class:`osprey.context.ContextManager` : Underlying context management
           :class:`osprey.context.CapabilityContext` : Base class for context objects
           :func:`get_current_step` : Utility for getting current execution step
        """

        # Create context manager from state
        context_manager = ContextManager(state)

        # Store the context object
        context_manager.set_context(context_type, context_key, context_object)

        # Return state updates with the updated dictionary data
        return {"capability_context_data": context_manager.get_raw_data()}

    @staticmethod
    def get_messages(state: AgentState) -> list[BaseMessage]:
        """Get messages from state."""
        return state.get("messages", [])

    @staticmethod
    def create_response_update(response: str) -> StateUpdate:
        """Create a state update that adds an assistant response using native LangGraph pattern.

        Args:
            response: The assistant's response

        Returns:
            StateUpdate: Update that adds the response message to the conversation
        """
        return {"messages": [MessageUtils.create_assistant_message(response)]}

    # ===== PLANNING AND EXECUTION UTILITIES =====

    @staticmethod
    def get_execution_plan(state: AgentState) -> ExecutionPlan | None:
        """Get current execution plan from state with type validation.

        Performs type validation to ensure the returned ExecutionPlan is properly
        formed (TypedDict with 'get' method). This eliminates the need for
        downstream callers to perform hasattr checks.

        Args:
            state: Current conversation state

        Returns:
            ExecutionPlan if available and valid, None otherwise
        """
        execution_plan = state.get("planning_execution_plan")

        # Type validation - ensure it's a valid ExecutionPlan (TypedDict)
        if execution_plan and hasattr(execution_plan, "get"):
            return execution_plan

        return None

    @staticmethod
    def get_current_step_index(state: AgentState) -> int:
        """Get current step index from state.

        Args:
            state: Current conversation state

        Returns:
            Current step index (defaults to 0 if not set)
        """
        return state.get("planning_current_step_index", 0)

    @staticmethod
    def get_current_step(state: AgentState) -> PlannedStep:
        """Get current execution step from state.

        This is the unified utility for accessing the current step being executed.
        Replaces the old pattern of importing _extract_current_step from decorators.

        Args:
            state: Current conversation state

        Returns:
            PlannedStep: Current step dictionary with capability, task_objective, etc.

        Raises:
            RuntimeError: If execution plan is missing or step index is invalid

        Example:
            .. code-block:: python

                # In any capability or infrastructure component:
                from osprey.state import StateManager

                step = StateManager.get_current_step(state)
                task_objective = step.get('task_objective')
                capability = step.get('capability')
        """
        execution_plan = StateManager.get_execution_plan(state)
        current_step_index = StateManager.get_current_step_index(state)

        if execution_plan:
            # Type validation already done by get_execution_plan()
            plan_steps = execution_plan.get("steps", [])
            logger.debug(
                f"current_step_index={current_step_index}, plan_steps_length={len(plan_steps)}"
            )

            if current_step_index < len(plan_steps):
                step = plan_steps[current_step_index]
                logger.debug(
                    f"Successfully extracted step at index {current_step_index}: {step.get('capability', 'unknown')}"
                )
                return step
            else:
                # This should NEVER happen - router should have caught this first
                raise RuntimeError(
                    f"CRITICAL BUG: Step index {current_step_index} beyond plan length {len(plan_steps)}. "
                    f"Router should have caught this and raised an exception first. "
                    f"This indicates multiple bugs in orchestrator validation AND router logic."
                )
        else:
            # No execution plan available - this could happen in edge cases during initialization
            raise RuntimeError(
                "CRITICAL BUG: No execution plan available for step extraction. "
                "Router should ensure execution plan exists "
                "before routing to capabilities that need step extraction."
            )

    # ===== UNIFIED ARTIFACT REGISTRATION =====

    @staticmethod
    def register_artifact(
        state: AgentState,
        artifact_type: ArtifactType,
        capability: str,
        data: dict[str, Any],
        display_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        current_artifacts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Register an artifact in the unified UI artifact registry.

        This is the single entry point for all capabilities to register artifacts
        for UI display. It replaces the separate figure/command/notebook registration
        methods with a unified interface that supports any artifact type.

        The method creates a properly structured artifact with a unique ID and timestamp,
        then appends it to the artifact list. It supports accumulation for registering
        multiple artifacts within a single node execution.

        Args:
            state: Current agent state
            artifact_type: Type of artifact (IMAGE, NOTEBOOK, COMMAND, HTML, FILE)
            capability: Name of the capability generating this artifact
            data: Type-specific data payload (path, url, uri, etc.)
            display_name: Optional human-readable name for the artifact
            metadata: Optional additional metadata dictionary
            current_artifacts: Optional list to accumulate artifacts (for multiple
                registrations within same node). If None, reads from state.

        Returns:
            State update dictionary with ui_artifacts update

        Examples:
            Register an image artifact::

                >>> update = StateManager.register_artifact(
                ...     state,
                ...     ArtifactType.IMAGE,
                ...     "python_executor",
                ...     {"path": "/path/to/plot.png", "format": "png"},
                ...     display_name="Analysis Plot"
                ... )
                >>> return {**other_updates, **update}

            Register multiple artifacts in one node::

                >>> accumulating = None
                >>> for path in figure_paths:
                ...     update = StateManager.register_artifact(
                ...         state,
                ...         ArtifactType.IMAGE,
                ...         "python_executor",
                ...         {"path": str(path), "format": path.suffix[1:]},
                ...         current_artifacts=accumulating
                ...     )
                ...     accumulating = update["ui_artifacts"]
                >>> return update  # Contains all artifacts

        .. seealso::
           :class:`osprey.state.artifacts.ArtifactType` : Available artifact types
           :func:`osprey.state.artifacts.create_artifact` : Artifact creation factory
        """
        # Create the artifact using the factory function
        artifact = create_artifact(
            artifact_type=artifact_type,
            capability=capability,
            data=data,
            display_name=display_name,
            metadata=metadata,
        )

        # Use provided current_artifacts or get from state
        if current_artifacts is not None:
            artifacts_list = current_artifacts
        else:
            artifacts_list = list(state.get("ui_artifacts", []))

        artifacts_list.append(artifact)

        logger.info(
            f"StateManager: registered {artifact_type.value} artifact for {capability}: "
            f"{display_name or data.get('path') or data.get('uri') or data.get('url', 'unknown')}"
        )

        return {"ui_artifacts": artifacts_list}

    # ===== LEGACY REGISTRATION METHODS (delegate to register_artifact) =====

    @staticmethod
    def register_figure(
        state: AgentState,
        capability: str,
        figure_path: str,
        display_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        current_figures: list[dict[str, Any]] | None = None,
        current_artifacts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Register a figure in the UI registry.

        .. deprecated::
            Use :meth:`register_artifact` with ``ArtifactType.IMAGE`` instead.
            This method is maintained for backward compatibility.

        This method now delegates to register_artifact() for the unified artifact
        system while maintaining the legacy ui_captured_figures field for backward
        compatibility with existing interfaces (e.g., OpenWebUI pipeline).

        Args:
            state: Current agent state
            capability: Capability identifier (e.g., "python_executor", "r_executor")
            figure_path: Path to the figure file (absolute or relative)
            display_name: Optional human-readable figure name
            metadata: Optional capability-specific metadata dictionary
            current_figures: Optional list of current figures to accumulate
            current_artifacts: Optional list of current artifacts to accumulate (for
                multiple registrations in same node). If not provided but current_figures
                is, artifacts from previous update are retrieved from state.

        Returns:
            State update dictionary with both ui_artifacts and ui_captured_figures updates
        """
        from datetime import datetime
        from pathlib import Path

        # Determine format from file extension
        path = Path(figure_path)
        format_ext = path.suffix[1:].lower() if path.suffix else "unknown"

        # Determine current_artifacts for accumulation
        # If current_figures is provided (accumulation mode), we need matching artifacts
        if current_artifacts is None:
            current_artifacts = list(state.get("ui_artifacts", []))

        # Register in unified artifact system
        artifact_update = StateManager.register_artifact(
            state=state,
            artifact_type=ArtifactType.IMAGE,
            capability=capability,
            data={"path": figure_path, "format": format_ext},
            display_name=display_name,
            metadata=metadata,
            current_artifacts=current_artifacts,
        )

        # Also maintain legacy field for backward compatibility
        figure_entry = {
            "capability": capability,
            "figure_path": figure_path,
            "created_at": datetime.now().isoformat(),
        }
        if display_name:
            figure_entry["display_name"] = display_name
        if metadata:
            figure_entry["metadata"] = metadata

        if current_figures is not None:
            figures_list = current_figures
        else:
            figures_list = list(state.get("ui_captured_figures", []))
        figures_list.append(figure_entry)

        logger.info(f"StateManager: prepared figure registration for {capability}: {figure_path}")

        # Return both unified and legacy updates
        return {**artifact_update, "ui_captured_figures": figures_list}

    @staticmethod
    def register_command(
        state: AgentState,
        capability: str,
        launch_uri: str,
        display_name: str | None = None,
        command_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        current_commands: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Register a launchable command in the UI registry.

        .. deprecated::
            Use :meth:`register_artifact` with ``ArtifactType.COMMAND`` instead.
            This method is maintained for backward compatibility.

        This method now delegates to register_artifact() for the unified artifact
        system while maintaining the legacy ui_launchable_commands field for backward
        compatibility with existing interfaces.

        Args:
            state: Current agent state
            capability: Name of the capability that generated this command
            launch_uri: URI that can be used to launch the command
            display_name: Optional human-readable name for the command
            command_type: Optional type of command (e.g., 'web_app', 'desktop_app')
            metadata: Optional capability-specific metadata dictionary
            current_commands: Optional list of current commands to accumulate

        Returns:
            State update dictionary with both ui_artifacts and ui_launchable_commands updates
        """
        from datetime import datetime

        # Build data payload for artifact
        data: dict[str, Any] = {"uri": launch_uri}
        if command_type:
            data["command_type"] = command_type

        # Register in unified artifact system
        artifact_update = StateManager.register_artifact(
            state=state,
            artifact_type=ArtifactType.COMMAND,
            capability=capability,
            data=data,
            display_name=display_name,
            metadata=metadata,
            current_artifacts=list(state.get("ui_artifacts", [])),
        )

        # Also maintain legacy field for backward compatibility
        command_entry = {
            "capability": capability,
            "launch_uri": launch_uri,
            "created_at": datetime.now().isoformat(),
        }
        if display_name:
            command_entry["display_name"] = display_name
        if command_type:
            command_entry["command_type"] = command_type
        if metadata:
            command_entry["metadata"] = metadata

        if current_commands is not None:
            commands_list = current_commands
        else:
            commands_list = list(state.get("ui_launchable_commands", []))
        commands_list.append(command_entry)

        logger.info(
            f"StateManager: prepared command registration for {capability}: {display_name or launch_uri}"
        )

        # Return both unified and legacy updates
        return {**artifact_update, "ui_launchable_commands": commands_list}

    @staticmethod
    def register_notebook(
        state: AgentState,
        capability: str,
        notebook_path: str,
        notebook_link: str,
        display_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Register a notebook in the UI registry.

        .. deprecated::
            Use :meth:`register_artifact` with ``ArtifactType.NOTEBOOK`` instead.
            This method is maintained for backward compatibility.

        This method now delegates to register_artifact() for the unified artifact
        system while maintaining the legacy ui_captured_notebooks field for backward
        compatibility with existing interfaces.

        Args:
            state: Current agent state
            capability: Capability identifier (e.g., "python_executor")
            notebook_path: Path to the notebook file
            notebook_link: Link to access the notebook
            display_name: Optional human-readable notebook name
            metadata: Optional capability-specific metadata dictionary

        Returns:
            State update dictionary with both ui_artifacts and ui_captured_notebooks updates
        """
        # Register in unified artifact system
        artifact_update = StateManager.register_artifact(
            state=state,
            artifact_type=ArtifactType.NOTEBOOK,
            capability=capability,
            data={"path": notebook_path, "url": notebook_link},
            display_name=display_name,
            metadata=metadata,
            current_artifacts=list(state.get("ui_artifacts", [])),
        )

        # Also maintain legacy field for backward compatibility
        notebook_links = list(state.get("ui_captured_notebooks", []))
        notebook_links.append(notebook_link)

        logger.info(
            f"StateManager: prepared notebook registration for {capability}: {display_name or notebook_path}"
        )

        # Return both unified and legacy updates
        return {**artifact_update, "ui_captured_notebooks": notebook_links}


def get_execution_steps_summary(state: AgentState) -> list[str]:
    """Generate ordered execution steps summary for prompts and UI display.

    This utility function extracts and formats execution step information from the
    agent state to provide a clean summary of completed execution
    steps. The function is designed for use in capability prompts, error summaries,
    UI displays, and debugging contexts where execution history is needed.

    The function processes the execution_step_results dictionary to create an
    ordered list of step descriptions based on execution order. Each step is
    formatted with a step number and task objective for clear presentation.

    **Processing Logic:**

    1. **Results Extraction**: Retrieves execution_step_results from state
    2. **Ordering**: Sorts results by step_index to maintain execution order
    3. **Formatting**: Creates numbered step descriptions with task objectives
    4. **Fallback Handling**: Uses capability names when task objectives unavailable

    :param state: Current agent state containing execution_step_results
    :type state: AgentState
    :return: Ordered list of formatted step descriptions for display
    :rtype: List[str]

    .. note::
       The function handles missing or incomplete execution results gracefully,
       returning an empty list when no execution data is available. Step numbering
       starts from 1 for intuitive display.

    .. warning::
       The function relies on step_index values in execution results for ordering.
       If step_index is missing or incorrect, the order may not reflect actual
       execution sequence.

    Examples:
        Basic execution summary::

            >>> state = {
            ...     "execution_step_results": {
            ...         "step1": {"step_index": 0, "task_objective": "Find PV addresses"},
            ...         "step2": {"step_index": 1, "task_objective": "Retrieve values"}
            ...     }
            ... }
            >>> steps = get_execution_steps_summary(state)
            >>> steps[0]
            'Step 1: Find PV addresses'
            >>> steps[1]
            'Step 2: Retrieve values'

        Empty state handling::

            >>> empty_state = {"execution_step_results": {}}
            >>> get_execution_steps_summary(empty_state)
            []

        Fallback to capability names::

            >>> state_without_objectives = {
            ...     "execution_step_results": {
            ...         "step1": {"step_index": 0, "capability": "pv_finder"}
            ...     }
            ... }
            >>> steps = get_execution_steps_summary(state_without_objectives)
            >>> steps[0]
            'Step 1: pv_finder'

    .. seealso::
       :class:`AgentState` : State structure containing execution_step_results
       :mod:`osprey.base.decorators` : Capability execution tracking
       :mod:`osprey.infrastructure` : Infrastructure components using summaries
    """
    step_results = state.get("execution_step_results", {})
    if not step_results:
        return []

    # Sort by step_index to ensure correct execution order
    ordered_results = sorted(step_results.items(), key=lambda x: x[1].get("step_index", 0))

    execution_steps = []
    for _step_key, result in ordered_results:
        step_num = result.get("step_index", 0) + 1
        task_objective = result.get("task_objective", result.get("capability", "unknown"))
        execution_steps.append(f"Step {step_num}: {task_objective}")

    return execution_steps
