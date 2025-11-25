"""
Orchestration Node - LangGraph Compatible

Creates execution plans from active capabilities and task requirements.
Convention-based implementation with native LangGraph interrupt support.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langgraph.types import interrupt

from osprey.approval.approval_system import (
    clear_approval_state,
    create_approval_type,
    create_plan_approval_interrupt,
    get_approval_resume_data,
)
from osprey.base.decorators import infrastructure_node
from osprey.base.errors import ErrorClassification, ErrorSeverity, ReclassificationRequiredError
from osprey.base.nodes import BaseInfrastructureNode
from osprey.base.planning import ExecutionPlan, PlannedStep
from osprey.context.context_manager import ContextManager
from osprey.models import get_chat_completion
from osprey.prompts.loader import get_framework_prompts
from osprey.registry import get_registry
from osprey.state import AgentState
from osprey.state.state import create_status_update
from osprey.state.state_manager import StateManager
from osprey.utils.config import get_agent_dir, get_model_config

# Factory code consolidated inline as helper function
from osprey.utils.logger import get_logger
from osprey.utils.streaming import get_streamer

if TYPE_CHECKING:
    from osprey.base.errors import ErrorClassification

logger = get_logger("orchestrator")


# =============================================================================
# EXECUTION PLAN VALIDATION
# =============================================================================


def _validate_and_fix_execution_plan(
    execution_plan: ExecutionPlan, current_task: str, logger
) -> ExecutionPlan:
    """Validate and fix execution plan to ensure all capabilities exist and it ends with respond or clarify step.

    This is the primary validation mechanism to:
    1. Check that all capabilities in the plan exist in the registry
    2. Ensure users always get a response by ending with respond/clarify

    If hallucinated capabilities are found, raises ValueError for re-planning.
    If the execution plan doesn't end with a respond or clarify step, we append a respond step.

    :param execution_plan: The execution plan to validate
    :param current_task: The current task for context
    :param logger: Logger instance
    :return: Fixed execution plan that ends with respond or clarify
    :raises ValueError: If hallucinated capabilities are found requiring re-planning
    """
    # Get fresh registry instance (not module-level cached)
    registry = get_registry()

    steps = execution_plan.get("steps", [])

    # Create generic respond step (used in multiple cases)
    generic_response = PlannedStep(
        context_key="user_response",
        capability="respond",
        task_objective=f"Respond to user request: {current_task}",
        expected_output="user_response",
        success_criteria="Provide helpful response to user query",
        inputs=[],
    )

    if not steps:
        logger.warning("Empty execution plan - adding default respond step")
        return {"steps": [generic_response]}

    # =====================================================================
    # STEP 1: VALIDATE ALL CAPABILITIES EXIST IN REGISTRY
    # =====================================================================

    hallucinated_capabilities = []

    for i, step in enumerate(steps):
        capability_name = step.get("capability", "")
        if not capability_name:
            logger.warning(f"Step {i+1} has no capability specified")
            continue

        # Check if capability exists in registry
        if not registry.get_node(capability_name):
            hallucinated_capabilities.append(capability_name)
            logger.error(f"Step {i+1}: Capability '{capability_name}' not found in registry")

    # If hallucinated capabilities found, trigger re-planning
    if hallucinated_capabilities:
        error_msg = f"Orchestrator hallucinated non-existent capabilities: {hallucinated_capabilities}. Available capabilities: {registry.get_stats()['capability_names']}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.debug("✅ All capabilities in execution plan exist in registry")

    # =====================================================================
    # STEP 2: ENSURE PLAN ENDS WITH RESPOND OR CLARIFY
    # =====================================================================

    # Check if last step is respond or clarify
    last_step = steps[-1]
    last_capability = last_step.get("capability", "").lower()

    if last_capability in ["respond", "clarify"]:
        logger.debug(f"Execution plan correctly ends with {last_capability} step")
        return execution_plan

    # Plan doesn't end with respond/clarify - add respond step
    logger.info(
        f"Execution plan ends with '{last_capability}' instead of respond/clarify - adding respond step"
    )

    # Append the respond step
    fixed_steps = steps + [generic_response]
    logger.success(f"Added respond step to execution plan (now {len(fixed_steps)} steps total)")

    return {"steps": fixed_steps}


# =============================================================================
# CONVENTION-BASED ORCHESTRATION NODE
# =============================================================================


@infrastructure_node
class OrchestrationNode(BaseInfrastructureNode):
    """Convention-based orchestration node with sophisticated execution planning logic.

    Creates detailed execution plans from task requirements and available capabilities.
    Handles both initial planning and replanning scenarios with approval workflows.

    Features:
    - Configuration-driven error classification and retry policies
    - LLM-based execution planning with fallback mechanisms
    - Approval workflow integration for execution plans
    - Context-aware planning with capability selection
    - Sophisticated error handling for LLM operations
    """

    name = "orchestrator"
    description = "Execution Planning and Orchestration"

    @staticmethod
    def classify_error(exc: Exception, context: dict):
        """Built-in error classification for orchestration operations."""
        # Retry LLM timeouts (orchestration uses LLM heavily)
        if hasattr(exc, "__class__") and "timeout" in exc.__class__.__name__.lower():
            return ErrorClassification(
                severity=ErrorSeverity.RETRIABLE,
                user_message="LLM timeout during execution planning, retrying...",
                metadata={"technical_details": str(exc)},
            )

        # Retry network/connection errors
        if isinstance(exc, (ConnectionError, TimeoutError)):
            return ErrorClassification(
                severity=ErrorSeverity.RETRIABLE,
                user_message="Network timeout during execution planning, retrying...",
                metadata={"technical_details": str(exc)},
            )

        # Retry Pydantic validation errors (LLM generation failures)
        # These occur when the LLM fails to generate valid structured output
        exc_str = str(exc).lower()
        if isinstance(exc, ValueError) and any(
            indicator in exc_str
            for indicator in ["validation error", "field required", "pydantic", "json", "parsing"]
        ):
            return ErrorClassification(
                severity=ErrorSeverity.RETRIABLE,
                user_message="LLM failed to generate valid execution plan format, retrying...",
                metadata={"technical_details": str(exc)},
            )

        # Don't retry true configuration/logic errors
        if isinstance(exc, (ValueError, TypeError)):
            return ErrorClassification(
                severity=ErrorSeverity.CRITICAL,
                user_message="Execution planning configuration error",
                metadata={"technical_details": str(exc)},
            )

        # Don't retry import/module errors (infrastructure issues)
        if isinstance(exc, (ImportError, ModuleNotFoundError)):
            return ErrorClassification(
                severity=ErrorSeverity.CRITICAL,
                user_message=f"Infrastructure dependency error: {str(exc)}",
                metadata={"technical_details": str(exc)},
            )

        # Default: CRITICAL for unknown errors (fail safe principle)
        # Handle reclassification requirement
        if isinstance(exc, ReclassificationRequiredError):
            return ErrorClassification(
                severity=ErrorSeverity.RECLASSIFICATION,
                user_message=f"Task needs reclassification: {str(exc)}",
                metadata={"technical_details": str(exc)},
            )

        # Only explicitly known errors should be RETRIABLE
        return ErrorClassification(
            severity=ErrorSeverity.CRITICAL,
            user_message=f"Unknown execution planning error: {str(exc)}",
            metadata={
                "technical_details": f"Error type: {type(exc).__name__}, Details: {str(exc)}"
            },
        )

    @staticmethod
    def get_retry_policy() -> dict[str, Any]:
        """Custom retry policy for LLM-based orchestration operations.

        Orchestration uses LLM calls heavily and can be flaky due to:
        - Network timeouts to LLM services
        - LLM provider rate limiting
        - Temporary LLM service unavailability

        Use longer delays and more attempts than default infrastructure policy.
        """
        return {
            "max_attempts": 4,  # More attempts for LLM operations
            "delay_seconds": 2.0,  # Longer initial delay for LLM services
            "backoff_factor": 2.0,  # Aggressive backoff for rate limiting
        }

    async def execute(self) -> dict[str, Any]:
        """Create execution plans with LangGraph native interrupt support.

        This implementation creates execution plans from task requirements and
        handles planning mode with native LangGraph interrupts for approval workflows.

        :return: Dictionary of state updates for LangGraph
        :rtype: Dict[str, Any]
        """
        state = self._state

        # Explicit logger retrieval - professional practice
        logger = get_logger("orchestrator")

        # Define streaming helper here for step awareness
        streamer = get_streamer("orchestrator", state)

        # =====================================================================
        # STEP 1: CHECK FOR APPROVED PLAN IN AGENT STATE (HIGHEST PRIORITY)
        # =====================================================================

        # Check for approved execution plan using centralized function
        has_approval_resume, approved_payload = get_approval_resume_data(
            state, create_approval_type("orchestrator", "plan")
        )

        if has_approval_resume and approved_payload:
            # Try to load execution plan from file first
            file_load_result = _load_execution_plan_from_file(logger=logger)

            if file_load_result["success"]:
                approved_plan = file_load_result["execution_plan"]
                plan_source = file_load_result["source"]
                logger.success(f"Using approved execution plan from file ({plan_source})")

                streamer.status(f"Using approved execution plan from file ({plan_source})")

                # Clean up processed plan files
                _cleanup_processed_plan_files(logger=logger)

                # Create state updates with explicit cleanup of approval and error state
                return {
                    **_create_state_updates(
                        state, approved_plan, f"approved_from_file_{plan_source}"
                    ),
                    **clear_approval_state(),
                }
            else:
                # Fallback to old in-memory approach if file loading fails
                approved_plan = approved_payload.get("execution_plan")
                if approved_plan:
                    logger.warning(
                        f"File loading failed ({file_load_result.get('error')}), using in-memory plan"
                    )

                    streamer.status("Using approved execution plan from memory")

                    # Create state updates with explicit cleanup of approval and error state
                    return {
                        **_create_state_updates(
                            state, approved_plan, "approved_from_memory_fallback"
                        ),
                        **clear_approval_state(),
                    }
                else:
                    logger.warning("Both file loading and in-memory plan failed")
        elif has_approval_resume:
            # Plan was rejected - clean up any pending files
            _cleanup_processed_plan_files(logger=logger)
            logger.info("Execution plan was rejected by user")

        # =====================================================================
        # STEP 2: EXTRACT CURRENT TASK AND ACTIVE CAPABILITIES
        # =====================================================================

        current_task = StateManager.get_current_task(state)
        if not current_task:
            raise ValueError("No current task available for orchestration")

        # Get active capabilities from state
        active_capability_names = state.get("planning_active_capabilities")

        if not active_capability_names:
            logger.error("No active capabilities found in state")
            raise ReclassificationRequiredError("No active capabilities found for task")

        # Get capability instances from registry using capability names
        # Get fresh registry instance (not module-level cached)
        registry = get_registry()

        active_capabilities = []
        for cap_name in active_capability_names:
            capability = registry.get_capability(cap_name)
            if capability:
                active_capabilities.append(capability)
            else:
                logger.warning(f"Capability '{cap_name}' not found in registry")

        if not active_capabilities:
            raise ValueError("No valid capability instances found for orchestration")

        logger.info(f"Planning for task: {current_task}")
        logger.info(f"Available capabilities: {[cap.name for cap in active_capabilities]}")

        # =====================================================================
        # STEP 3: CREATE NEW EXECUTION PLAN
        # =====================================================================

        # =====================================================================
        # HELPER FUNCTION: CREATE SYSTEM PROMPT
        # =====================================================================

        async def create_system_prompt() -> str:
            """Create orchestrator system prompt from capabilities and context."""
            logger.info(f'Creating orchestrator prompt for task: "{current_task[:100]}..."')

            # Extract capability names from active capabilities
            active_capability_names = [cap.name for cap in active_capabilities]
            logger.info(f"Active capabilities: {active_capability_names}")

            # Create ContextManager from state data
            context_manager = ContextManager(state)

            # Format error context for replanning if available
            error_info = state.get("control_error_info")
            error_context = None
            if error_info and isinstance(error_info, dict):
                classification = error_info.get("classification")
                if classification and hasattr(classification, "format_for_llm"):
                    error_context = classification.format_for_llm()

            if error_context:
                logger.info(
                    "Error context detected - enabling replanning mode with failure analysis"
                )
                logger.debug(f"Error context for replanning: {error_context}")

            # Use the prompt system to build the complete orchestrator prompt
            prompt_provider = get_framework_prompts()
            orchestrator_builder = prompt_provider.get_orchestrator_prompt_builder()

            system_instructions = orchestrator_builder.get_system_instructions(
                active_capabilities=active_capabilities,
                context_manager=context_manager,
                task_depends_on_chat_history=state.get("task_depends_on_chat_history", False),
                task_depends_on_user_memory=state.get("task_depends_on_user_memory", False),
                error_context=error_context,
            )

            if not system_instructions:
                logger.error("No prompt text generated. The instructions will be empty.")
                raise ValueError("No prompt text generated. The instructions will be empty.")

            # Count total examples across all capabilities
            total_examples = sum(
                len(cap.orchestrator_guide.examples)
                for cap in active_capabilities
                if cap.orchestrator_guide and hasattr(cap.orchestrator_guide, "examples")
            )

            # Get context data from ContextManager
            raw_data = context_manager.get_raw_data()
            context_types = len(raw_data) if raw_data else 0

            logger.info("Constructed orchestrator instructions using:")
            logger.info(f" - {len(active_capabilities)} capabilities")
            logger.info(f" - {total_examples} structured examples")
            logger.info(f" - {context_types} context types from state")
            if error_context:
                logger.info(" - Error context for replanning (previous failure analysis)")

            logger.debug(
                f"\n\n\n------------Orchestrator System Prompt:\n{system_instructions}\n------------\n\n\n"
            )

            return system_instructions

        # =====================================================================
        # GENERATE EXECUTION PLAN
        # =====================================================================

        # Create system prompt
        system_prompt = await create_system_prompt()

        streamer.status("Generating execution plan...")

        # Call LLM directly for execution planning
        logger.key_info("Creating execution plan with orchestrator LLM")
        plan_start_time = time.time()

        # Get model configuration and call LLM
        model_config = get_model_config("orchestrator")
        message = f"{system_prompt}\n\nTASK TO PLAN: {current_task}"

        # Run sync LLM call in thread pool to avoid blocking event loop for streaming
        execution_plan = await asyncio.to_thread(
            get_chat_completion,
            message=message,
            model_config=model_config,
            output_model=ExecutionPlan,
        )

        execution_time = time.time() - plan_start_time
        logger.info(f"Orchestrator LLM execution time: {execution_time:.2f} seconds")

        # =====================================================================
        # STEP 3.5: VALIDATE AND FIX EXECUTION PLAN
        # =====================================================================

        try:
            # Validate that all capabilities exist and plan ends with respond/clarify
            execution_plan = _validate_and_fix_execution_plan(execution_plan, current_task, logger)

            # Log final validated execution plan (after any modifications)
            _log_execution_plan(execution_plan, logger)

            logger.success(
                f"Final execution plan ready with {len(execution_plan.get('steps', []))} steps"
            )

        except ValueError as e:
            # Orchestrator hallucinated non-existent capabilities - trigger re-classification
            logger.error(f"Execution plan validation failed: {e}")
            logger.warning("Triggering re-classification due to hallucinated capabilities")
            streamer.status("Re-planning due to invalid capabilities...")

            # Raise exception to trigger reclassification through error system
            raise ReclassificationRequiredError(f"Orchestrator validation failed: {e}")

        # =====================================================================
        # STEP 4: HANDLE RESULTING EXECUTION PLAN
        # =====================================================================

        if _is_planning_mode_enabled(state):
            logger.info("PLANNING MODE DETECTED - entering approval workflow")
            # LangGraph handles caching automatically - no manual caching needed
            await _handle_planning_mode(execution_plan, current_task, state, logger, streamer)
        else:
            logger.info("Planning mode not enabled - proceeding with normal execution")

        streamer.status("Execution plan created")

        logger.key_info("Orchestration processing completed")

        return _create_state_updates(state, execution_plan, "llm_based")


# =============================================================================
# BUSINESS LOGIC HELPERS
# =============================================================================


def _clear_error_state() -> dict[str, Any]:
    """Clear error state to prevent router from staying in retry handling mode.

    When orchestrator creates a new plan, we need to clear previous error state
    so the router can execute the new plan instead of continuing to handle old errors.

    Returns:
        Dictionary with error state fields cleared
    """
    return {
        "control_has_error": False,
        "control_error_info": None,
        "control_last_error": None,
        "control_retry_count": 0,
        "control_current_step_retry_count": 0,
    }


def _log_execution_plan(execution_plan: ExecutionPlan, logger):
    """Log execution plan with clean formatting."""

    logger.key_info("=" * 50)
    for index, step in enumerate(execution_plan.get("steps", [])):
        logger.key_info(f" << Step {index + 1}")
        logger.info(f" << ├───── id: '{step.get('context_key', 'unknown')}'")
        logger.info(f" << ├─── node: '{step.get('capability', 'unknown')}'")
        logger.info(f" << ├─── task: '{step.get('task_objective', 'unknown')}'")
        logger.info(f" << └─ inputs: '{step.get('inputs', [])}'")
    logger.key_info("=" * 50)


def _save_execution_plan_to_file(
    execution_plan: ExecutionPlan, current_task: str, state: AgentState, logger=None
) -> dict[str, Any]:
    """Save execution plan to JSON file for human approval workflow.

    Args:
        execution_plan: The execution plan to save
        current_task: The extracted task from task extraction node
        state: Agent state containing original user message
        logger: Logger instance for logging

    Returns:
        Dictionary with success status and file path
    """
    try:
        # Get execution plans directory
        execution_plans_dir = get_agent_dir("execution_plans")
        pending_plans_dir = Path(execution_plans_dir) / "pending_plans"
        pending_plans_dir.mkdir(parents=True, exist_ok=True)

        # Extract original user query
        original_query = StateManager.get_user_query(state)
        if not original_query:
            original_query = current_task  # Fallback if no messages available

        # Create plan data with metadata including both original query and extracted task
        plan_data = {
            "__metadata__": {
                "current_task": current_task,  # Extracted task from task extraction
                "original_query": original_query,  # Original user input
                "created_at": datetime.datetime.now().isoformat(),
                "serialization_type": "pending_execution_plan",
            },
            "steps": execution_plan.get("steps", []),
        }

        # Save to pending plan file (used by editor)
        pending_plan_file = pending_plans_dir / "pending_execution_plan.json"
        with open(pending_plan_file, "w", encoding="utf-8") as f:
            json.dump(plan_data, f, indent=2, ensure_ascii=False)

        if logger:
            logger.info(f"Execution plan saved to {pending_plan_file}")

        return {
            "success": True,
            "file_path": str(pending_plan_file),
            "pending_plans_dir": str(pending_plans_dir),
        }

    except Exception as e:
        error_msg = f"Failed to save execution plan to file: {e}"
        if logger:
            logger.error(error_msg)
        return {"success": False, "error": error_msg}


def _load_execution_plan_from_file(logger=None) -> dict[str, Any]:
    """Load execution plan from JSON file after human approval.

    Args:
        logger: Logger instance for logging

    Returns:
        Dictionary with success status and execution plan data
    """
    try:
        # Get execution plans directory using config
        execution_plans_dir = get_agent_dir("execution_plans")
        pending_plans_dir = Path(execution_plans_dir) / "pending_plans"

        # Try to load modified plan first (if user modified the plan)
        modified_plan_file = pending_plans_dir / "modified_execution_plan.json"
        if modified_plan_file.exists():
            with open(modified_plan_file, encoding="utf-8") as f:
                plan_data = json.load(f)

            if logger:
                logger.info(f"Loaded modified execution plan from {modified_plan_file}")

            return {
                "success": True,
                "execution_plan": {"steps": plan_data.get("steps", [])},
                "metadata": plan_data.get("__metadata__", {}),
                "source": "modified_plan",
            }

        # Fall back to original pending plan
        pending_plan_file = pending_plans_dir / "pending_execution_plan.json"
        if pending_plan_file.exists():
            with open(pending_plan_file, encoding="utf-8") as f:
                plan_data = json.load(f)

            if logger:
                logger.info(f"Loaded original execution plan from {pending_plan_file}")

            return {
                "success": True,
                "execution_plan": {"steps": plan_data.get("steps", [])},
                "metadata": plan_data.get("__metadata__", {}),
                "source": "original_plan",
            }

        error_msg = "No execution plan file found for loading"
        if logger:
            logger.warning(error_msg)

        return {"success": False, "error": error_msg}

    except Exception as e:
        error_msg = f"Failed to load execution plan from file: {e}"
        if logger:
            logger.error(error_msg)
        return {"success": False, "error": error_msg}


def _cleanup_processed_plan_files(logger=None):
    """Clean up processed execution plan files after successful execution.

    Args:
        logger: Logger instance for logging
    """
    try:
        # Get execution plans directory using config
        execution_plans_dir = get_agent_dir("execution_plans")
        pending_plans_dir = Path(execution_plans_dir) / "pending_plans"

        files_to_remove = []

        # Remove pending plan file
        pending_plan_file = pending_plans_dir / "pending_execution_plan.json"
        if pending_plan_file.exists():
            files_to_remove.append(pending_plan_file)

        # Remove modified plan file if it exists
        modified_plan_file = pending_plans_dir / "modified_execution_plan.json"
        if modified_plan_file.exists():
            files_to_remove.append(modified_plan_file)

        # Remove the files
        for file_path in files_to_remove:
            file_path.unlink()
            if logger:
                logger.debug(f"Cleaned up plan file: {file_path}")

        if logger and files_to_remove:
            logger.info(f"Cleaned up {len(files_to_remove)} processed plan files")

    except Exception as e:
        if logger:
            logger.warning(f"Failed to cleanup plan files: {e}")


async def _handle_planning_mode(
    execution_plan: ExecutionPlan, current_task: str, state: AgentState, logger, streamer
):
    """Handle planning mode using structured approval system with file-based plan storage."""

    logger.approval("Planning mode enabled - requesting plan approval")

    streamer.status("Saving execution plan and requesting approval...")

    # Save execution plan to file for human approval workflow
    save_result = _save_execution_plan_to_file(
        execution_plan=execution_plan, current_task=current_task, state=state, logger=logger
    )

    if not save_result["success"]:
        logger.warning(f"Failed to save execution plan: {save_result.get('error')}")
        # Fallback to in-memory approach if file saving fails
        interrupt_data = create_plan_approval_interrupt(execution_plan=execution_plan)
    else:
        logger.approval(f"Execution plan saved to {save_result['file_path']}")

        # Create enhanced interrupt data with file path references
        interrupt_data = create_plan_approval_interrupt(
            execution_plan=execution_plan,
            plan_file_path=save_result["file_path"],
            pending_plans_dir=save_result["pending_plans_dir"],
        )

    logger.approval("Interrupting execution for plan approval")
    logger.debug(f"Interrupt data created with {len(execution_plan.get('steps', []))} steps")

    # LangGraph interrupt - execution stops here until user responds
    interrupt(interrupt_data)


def _is_planning_mode_enabled(state: AgentState) -> bool:
    """Check if planning mode is enabled in agent control state."""
    agent_control = state.get("agent_control", {})
    return agent_control.get("planning_mode_enabled", False)


def _create_state_updates(
    state: AgentState, execution_plan: ExecutionPlan, approach: str
) -> dict[str, Any]:
    """Create state updates based on orchestration results using proper LangGraph merging."""

    # Direct planning state update
    planning_update = {"planning_execution_plan": execution_plan, "planning_current_step_index": 0}

    # Increment plans created counter - orchestrator owns this responsibility
    current_plans_count = state.get("control_plans_created_count", 0)
    planning_control_update = {"control_plans_created_count": current_plans_count + 1}

    # CRITICAL: Clear error state so router can execute new plan instead of staying in retry mode
    error_state_cleanup = _clear_error_state()

    # Add status event using LangGraph's add reducer
    status_event = create_status_update(
        message=f"Execution plan created using {approach} (plan #{current_plans_count + 1})",
        progress=1.0,
        complete=True,
        node="orchestration",
        approach=approach,
        total_steps=len(execution_plan.get("steps", [])),
    )

    # Merge the updates - LangGraph will handle this properly
    return {**planning_update, **planning_control_update, **error_state_cleanup, **status_event}
