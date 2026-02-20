"""Framework State - Agent Control and Configuration Management.

This module provides comprehensive agent control and configuration management for the
Osprey Agent Framework. The control system manages execution flow parameters,
approval workflows, EPICS operations, and runtime overrides through a unified
configuration interface.

**Core Components:**

- :class:`AgentControlState`: Unified control configuration with runtime overrides
- :func:`apply_slash_commands_to_agent_control_state`: Apply command changes to agent control state

**Configuration Management:**

The control system provides centralized management of agent behavior through
configuration parameters that can be overridden at runtime through various sources:

- **Global Configuration**: Base configuration from config system
- **User Valves**: User-specific overrides through UI controls
- **Slash Commands**: Real-time overrides through chat commands
- **Runtime Updates**: Programmatic overrides during execution

**Control Categories:**

1. **Planning Control**: Planning mode enablement and orchestration behavior
2. **EPICS Control**: EPICS write operation permissions and safety controls
3. **Approval Control**: Approval workflow configuration and requirements
4. **Execution Control**: Retry limits, timeouts, and execution boundaries

**State Integration:**

The control state integrates seamlessly with the main AgentState structure,
providing runtime configuration that affects execution behavior throughout
the framework. All control parameters support partial updates for LangGraph
compatibility.

.. note::
   All control state fields are optional to support partial updates in LangGraph's
   state management system. Default values are applied when fields are not provided.

.. seealso::
   :class:`osprey.state.AgentState` : Main state structure using control state
   :mod:`configs.config` : Base configuration system
   :mod:`osprey.infrastructure.gateway` : Gateway applying slash commands
"""

from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict

from osprey.utils.config import get_execution_limits
from osprey.utils.logger import get_logger

logger = get_logger("state_control")


class AgentControlState(TypedDict, total=False):
    """Unified agent control configuration with comprehensive runtime override support.

    This TypedDict class serves as the single source of truth for all agent control
    parameters throughout the Osprey Agent Framework. The control state manages
    execution flow, approval workflows, EPICS operations, and planning modes with
    support for runtime overrides from multiple sources including user interfaces,
    slash commands, and programmatic updates.

    **Configuration Architecture:**

    The control state implements a layered configuration approach:

    1. **Base Configuration**: Default values from configuration system
    2. **User Overrides**: User-specific settings through UI valve controls
    3. **Slash Commands**: Real-time overrides through chat-based commands
    4. **Runtime Updates**: Programmatic overrides during execution

    **Control Categories:**

    - **Planning Control**: Orchestration and planning mode management
    - **EPICS Control**: EPICS write operation permissions and safety
    - **Approval Control**: Workflow approval requirements and modes
    - **Execution Control**: Retry limits, timeouts, and execution boundaries

    **LangGraph Integration:**

    All fields are optional (total=False) to support LangGraph's partial state
    update patterns. The control state integrates seamlessly with AgentState
    and supports automatic merging of configuration updates.

    **Default Values:**

    When fields are not provided, the following defaults apply:

    - planning_mode_enabled: False (disabled for safety)
    - epics_writes_enabled: False (disabled for safety)
    - approval_global_mode: "selective" (balanced approval approach)
    - python_execution_approval_enabled: True (security-first approach)
    - python_execution_approval_mode: "all_code" (comprehensive approval)
    - memory_approval_enabled: True (protect user memory)
    - max_reclassifications: 1 (prevent infinite reclassification loops)
    - max_planning_attempts: 2 (allow retry but prevent excessive attempts)
    - max_step_retries: 0 (fail fast by default)
    - max_execution_time_seconds: 300 (5-minute execution limit)
    - max_concurrent_classifications: 5 (prevent API flooding during classification)
    - task_extraction_bypass_enabled: False (use task extraction by default)
    - capability_selection_bypass_enabled: False (use capability selection by default)
    - approval_mode: "selective" (balanced approval mode from slash commands)

    .. note::
       Default values prioritize safety and security by disabling potentially
       dangerous operations and enabling approval requirements. Production
       deployments may override these defaults through configuration.

    .. warning::
       Changes to control state affect agent behavior immediately. Ensure that
       runtime overrides are validated and appropriate for the execution context.

    Examples:
        Basic control state with defaults::

            >>> control = AgentControlState()
            >>> # All fields optional, defaults applied by StateManager

        Control state with planning enabled::

            >>> control = AgentControlState(
            ...     planning_mode_enabled=True,
            ...     max_planning_attempts=3
            ... )

        Security-focused configuration::

            >>> control = AgentControlState(
            ...     epics_writes_enabled=False,
            ...     approval_global_mode="all_capabilities",
            ...     python_execution_approval_mode="all_code"
            ... )

    .. seealso::
       :func:`apply_slash_commands_to_agent_control_state` : Apply command changes to control state
       :class:`osprey.state.AgentState` : Main state containing control state
       :mod:`configs.config` : Base configuration system
    """

    # Planning control
    planning_mode_enabled: bool  # Whether planning mode is enabled for the agent

    # EPICS execution control
    epics_writes_enabled: bool  # Whether EPICS write operations are allowed

    # Approval control
    approval_global_mode: str  # Global approval mode setting (disabled/selective/all_capabilities)
    approval_mode: str  # Approval mode set via /approval slash command (enabled/disabled/selective)
    python_execution_approval_enabled: bool  # Whether Python execution requires approval
    python_execution_approval_mode: str  # Python approval mode (disabled/epics_writes/all_code)
    memory_approval_enabled: bool  # Whether memory operations require approval

    # Bypass control (performance optimizations via /task and /caps commands)
    task_extraction_bypass_enabled: bool  # Whether task extraction is bypassed
    capability_selection_bypass_enabled: bool  # Whether capability selection is bypassed

    # Execution flow control
    max_reclassifications: int  # Maximum number of task reclassifications allowed
    max_planning_attempts: int  # Maximum number of planning attempts before giving up
    max_step_retries: int  # Maximum number of retries per execution step
    max_execution_time_seconds: int  # Maximum execution time in seconds
    max_concurrent_classifications: int  # Maximum concurrent LLM classification requests


def apply_slash_commands_to_agent_control_state(
    agent_control_state: AgentControlState, command_changes: dict[str, Any]
) -> AgentControlState:
    """Apply processed command changes to agent control configuration.

    This function takes the output from the centralized command system and applies
    the changes to the agent control state. It ensures proper state merging with
    existing configuration while preserving all non-modified fields.

    The function handles the complete state update workflow:

    1. **State Copying**: Creates a new control state instance with all existing values
    2. **Change Application**: Applies only the specific changes from commands
    3. **Default Handling**: Ensures all fields have proper default values
    4. **Logging**: Records applied changes for debugging and audit

    **Command Integration:**

    This function works with the centralized command system where agent control
    commands return dictionaries of state changes (e.g., {"planning_mode_enabled": True}).

    :param agent_control_state: Current agent control state to update
    :type agent_control_state: AgentControlState
    :param command_changes: Dictionary of field changes from command handlers
    :type command_changes: Dict[str, Any]
    :return: New AgentControlState instance with command changes applied
    :rtype: AgentControlState

    .. note::
       The function creates a new AgentControlState instance rather than modifying
       the input state, ensuring immutability and preventing side effects.

    .. warning::
       Command validation is performed but invalid options are logged as warnings
       rather than raising exceptions to maintain execution continuity.

    Examples:
        Applying planning mode change::

            >>> current_state = AgentControlState(planning_mode_enabled=False)
            >>> changes = {"planning_mode_enabled": True}  # From command handler
            >>> new_state = apply_slash_commands_to_agent_control_state(
            ...     current_state, changes
            ... )
            >>> new_state['planning_mode_enabled']
            True

        Applying multiple changes::

            >>> changes = {
            ...     "planning_mode_enabled": True,
            ...     "debug_mode": True
            ... }
            >>> new_state = apply_slash_commands_to_agent_control_state(
            ...     current_state, changes
            ... )

        No changes (passthrough)::

            >>> new_state = apply_slash_commands_to_agent_control_state(
            ...     current_state, {}
            ... )
            >>> new_state == current_state
            True

    .. seealso::
       :class:`AgentControlState` : Control state structure being modified
       :mod:`osprey.infrastructure.gateway` : Gateway parsing slash commands
    """
    if not command_changes:
        return agent_control_state

    # Get execution limits from configuration for consistent defaults
    execution_limits = get_execution_limits()

    # Create a copy with all existing values preserved
    new_instance = AgentControlState(
        planning_mode_enabled=agent_control_state.get("planning_mode_enabled", False),
        epics_writes_enabled=agent_control_state.get("epics_writes_enabled", False),
        approval_global_mode=agent_control_state.get("approval_global_mode", "selective"),
        approval_mode=agent_control_state.get("approval_mode", "selective"),
        python_execution_approval_enabled=agent_control_state.get(
            "python_execution_approval_enabled", True
        ),
        python_execution_approval_mode=agent_control_state.get(
            "python_execution_approval_mode", "all_code"
        ),
        memory_approval_enabled=agent_control_state.get("memory_approval_enabled", True),
        task_extraction_bypass_enabled=agent_control_state.get(
            "task_extraction_bypass_enabled", False
        ),
        capability_selection_bypass_enabled=agent_control_state.get(
            "capability_selection_bypass_enabled", False
        ),
        max_reclassifications=agent_control_state.get(
            "max_reclassifications", execution_limits.get("max_reclassifications", 1)
        ),
        max_planning_attempts=agent_control_state.get(
            "max_planning_attempts", execution_limits.get("max_planning_attempts", 2)
        ),
        max_step_retries=agent_control_state.get(
            "max_step_retries", execution_limits.get("max_step_retries", 0)
        ),
        max_execution_time_seconds=agent_control_state.get(
            "max_execution_time_seconds", execution_limits.get("max_execution_time_seconds", 300)
        ),
        max_concurrent_classifications=agent_control_state.get(
            "max_concurrent_classifications",
            execution_limits.get("max_concurrent_classifications", 5),
        ),
    )

    # Apply the specific changes from command handlers
    for field, value in command_changes.items():
        if field in AgentControlState.__annotations__:
            new_instance[field] = value
            logger.info(f"Applied command change: {field} = {value}")
        else:
            logger.warning(f"Unknown agent control field from command: {field}")

    return new_instance
