"""
Osprey Agentic Framework - Classification Node

Task classification and capability selection with sophisticated analysis.
Combines LangGraph infrastructure with core classification logic.

Analyzes user queries to determine required capabilities and data dependencies.
Convention-based LangGraph-native implementation with built-in error handling and retry policies.
"""

from __future__ import annotations

import asyncio
from typing import Any

from osprey.base import BaseCapability, CapabilityMatch, ClassifierExample
from osprey.base.decorators import infrastructure_node
from osprey.base.errors import ErrorClassification, ErrorSeverity, ReclassificationRequiredError
from osprey.base.nodes import BaseInfrastructureNode
from osprey.events import EventEmitter, StatusEvent
from osprey.models import get_chat_completion
from osprey.prompts.loader import get_framework_prompts
from osprey.registry import get_registry
from osprey.state import AgentState
from osprey.state.state import create_status_update
from osprey.utils.config import get_classification_config, get_model_config
from osprey.utils.logger import get_logger

# Module-level logger for helper functions
logger = get_logger("classifier")
emitter = EventEmitter("classifier")


@infrastructure_node
class ClassificationNode(BaseInfrastructureNode):
    """Convention-based classification node with sophisticated capability selection logic.

    Analyzes user tasks and selects appropriate capabilities using parallel
    LLM-based classification with few-shot examples. Handles both initial
    classification and reclassification scenarios.

    Uses LangGraph's sophisticated state merging with built-in error handling
    and retry policies optimized for LLM-based classification operations.
    """

    # Loaded through registry configuration
    name = "classifier"
    description = "Task Classification and Capability Selection"

    @staticmethod
    def classify_error(exc: Exception, context: dict[str, Any]) -> ErrorClassification:
        """Built-in error classification for classifier operations.

        :param exc: Exception that occurred
        :param context: Error context information
        :return: Classification with severity and retry guidance
        """

        # Retry LLM timeouts and network errors
        if hasattr(exc, "__class__") and "timeout" in exc.__class__.__name__.lower():
            return ErrorClassification(
                severity=ErrorSeverity.RETRIABLE,
                user_message="Classification service temporarily unavailable, retrying...",
                metadata={"technical_details": f"LLM timeout: {str(exc)}"},
            )

        if isinstance(exc, (ConnectionError, TimeoutError)):
            return ErrorClassification(
                severity=ErrorSeverity.RETRIABLE,
                user_message="Network connectivity issues during classification, retrying...",
                metadata={"technical_details": f"Network error: {str(exc)}"},
            )

        # Don't retry validation errors (data/logic issues)
        if isinstance(exc, (ValueError, TypeError)):
            return ErrorClassification(
                severity=ErrorSeverity.CRITICAL,
                user_message="Task classification configuration error",
                metadata={
                    "technical_details": f"Validation error: {str(exc)}",
                    "safety_abort_reason": "Classification system misconfiguration detected",
                },
            )

        # Don't retry import/module errors (missing dependencies or path issues)
        # Check both the exception itself and any chained exceptions
        def is_import_error(e):
            if isinstance(e, (ImportError, ModuleNotFoundError, NameError)):
                return True
            # Check chained exceptions (from "raise X from Y")
            if hasattr(e, "__cause__") and e.__cause__:
                return isinstance(e.__cause__, (ImportError, ModuleNotFoundError, NameError))
            return False

        if is_import_error(exc):
            return ErrorClassification(
                severity=ErrorSeverity.CRITICAL,
                user_message="Task classification dependencies not available",
                metadata={
                    "technical_details": f"Import error: {str(exc)}",
                    "safety_abort_reason": "Required classification dependencies missing",
                },
            )

        # Handle reclassification requirement
        if isinstance(exc, ReclassificationRequiredError):
            return ErrorClassification(
                severity=ErrorSeverity.RECLASSIFICATION,
                user_message=f"Task needs reclassification: {str(exc)}",
                metadata={"technical_details": str(exc)},
            )

        # Default: CRITICAL for unknown errors (fail safe principle)
        # Only explicitly known errors should be RETRIABLE
        return ErrorClassification(
            severity=ErrorSeverity.CRITICAL,
            user_message=f"Unknown classification error: {str(exc)}",
            metadata={
                "technical_details": f"Error type: {type(exc).__name__}, Details: {str(exc)}",
                "safety_abort_reason": "Unhandled classification system error",
            },
        )

    @staticmethod
    def get_retry_policy() -> dict[str, Any]:
        """Custom retry policy for LLM-based classification operations.

        Classification uses parallel LLM calls for capability selection and can be flaky due to:
        - Multiple concurrent LLM requests
        - Network timeouts to LLM services
        - LLM provider rate limiting
        - Classification model variability

        Use more attempts with moderate delays for better reliability.
        """
        return {
            "max_attempts": 4,  # More attempts for LLM classification
            "delay_seconds": 1.0,  # Moderate delay for parallel LLM calls
            "backoff_factor": 1.8,  # Moderate backoff to handle rate limiting
        }

    async def execute(self) -> dict[str, Any]:
        """Main classification logic with bypass support and sophisticated capability selection.

        Analyzes user tasks and selects appropriate capabilities using parallel
        LLM-based classification. Handles both initial classification and
        reclassification scenarios with state preservation.

        Supports bypass mode where all available capabilities are activated,
        skipping LLM-based classification for performance optimization.

        :return: Dictionary of state updates for LangGraph
        :rtype: Dict[str, Any]
        """
        import time

        from osprey.events import CapabilitiesSelectedEvent, PhaseCompleteEvent, PhaseStartEvent

        state = self._state
        start_time = time.time()

        # Get unified logger with automatic streaming support
        logger = self.get_logger()

        # Emit phase start event
        logger.emit_event(
            PhaseStartEvent(
                phase="classification",
                description="Analyzing task requirements and selecting capabilities",
            )
        )

        # Get the current task from state
        current_task = state.get("task_current_task")

        if not current_task:
            # Emit phase complete with failure
            duration_ms = int((time.time() - start_time) * 1000)
            logger.emit_event(
                PhaseCompleteEvent(phase="classification", duration_ms=duration_ms, success=False)
            )
            logger.error("No current task found in state")
            raise ReclassificationRequiredError("No current task found")

        # Check if capability selection bypass is enabled
        bypass_enabled = state.get("agent_control", {}).get(
            "capability_selection_bypass_enabled", False
        )

        if bypass_enabled:
            logger.info("Capability selection bypass enabled - activating all capabilities")

            # Get all capability names directly from registry
            registry = get_registry()
            active_capabilities = registry.get_stats()["capability_names"]

            logger.success(
                f"Bypass mode: activated all {len(active_capabilities)} capabilities",
                capability_names=active_capabilities,
            )

            # Emit data event with selected capabilities
            logger.emit_event(
                CapabilitiesSelectedEvent(
                    capability_names=active_capabilities,
                    all_capability_names=active_capabilities,  # In bypass mode, all are selected
                )
            )

            # Emit phase complete event
            duration_ms = int((time.time() - start_time) * 1000)
            logger.emit_event(
                PhaseCompleteEvent(phase="classification", duration_ms=duration_ms, success=True)
            )

            # Return standardized classification result
            return _create_classification_result(
                active_capabilities=active_capabilities,
                state=state,
                message=f"Bypass mode: activated all {len(active_capabilities)} capabilities",
                is_bypass=True,
            )

        # Original classification logic continues here...

        # Detect reclassification scenario from error state
        previous_failure = _detect_reclassification_scenario(state)

        reclassification_count = state.get("control_reclassification_count", 0)

        if previous_failure:
            logger.status(f"Reclassifying task (attempt {reclassification_count + 1})...")
            logger.warning(f"Previous failure reason: {previous_failure}")
        else:
            logger.status("Analyzing task requirements...")

        logger.info(f"Classifying task: {current_task}")

        # Get available capabilities from capability registry
        registry = get_registry()
        available_capabilities = registry.get_all_capabilities()

        logger.debug(f"Available capabilities: {len(available_capabilities)}")

        # Run capability selection using the task analyzer (core business logic)
        active_capabilities = await select_capabilities(
            task=current_task,  # Updated parameter name
            available_capabilities=available_capabilities,
            state=state,
            logger=logger,
            previous_failure=previous_failure,  # Pass failure context for reclassification
        )

        logger.success(
            f"Classification completed with {len(active_capabilities)} active capabilities",
            capability_names=active_capabilities,
        )
        logger.debug(f"Active capabilities: {active_capabilities}")

        # Emit data event with selected capabilities
        # Get all available capability names for UI display
        all_capability_names = [cap.name for cap in available_capabilities]
        logger.emit_event(
            CapabilitiesSelectedEvent(
                capability_names=active_capabilities,
                all_capability_names=all_capability_names,
            )
        )

        # Emit phase complete event
        duration_ms = int((time.time() - start_time) * 1000)
        logger.emit_event(
            PhaseCompleteEvent(phase="classification", duration_ms=duration_ms, success=True)
        )

        # Return standardized classification result
        return _create_classification_result(
            active_capabilities=active_capabilities,
            state=state,
            message=f"Classification completed with {len(active_capabilities)} capabilities",
            is_bypass=False,
            previous_failure=previous_failure,
        )


# ====================================================
# Classification helper functions
# ====================================================


def _create_classification_result(
    active_capabilities: list[str],
    state: AgentState,
    message: str,
    is_bypass: bool = False,
    previous_failure: str | None = None,
) -> dict[str, Any]:
    """Create standardized classification result with all required state updates.

    Consolidates the creation of planning fields, control flow updates, and status events
    into a single function to eliminate code duplication between bypass and normal
    classification paths.

    :param active_capabilities: List of capability names that were selected
    :param state: Current agent state for extracting reclassification count
    :param message: Status message to display
    :param is_bypass: Whether this is a bypass mode result
    :param previous_failure: Previous failure reason (for reclassification detection)
    :return: Complete state update dictionary for LangGraph
    """
    reclassification_count = state.get("control_reclassification_count", 0)

    # Initialize error state cleanup as empty - only populate if this is a reclassification
    error_state_cleanup = {}

    # Only increment and clear error state if this is actually a reclassification
    if previous_failure:
        reclassification_count += 1
        emitter.emit(
            StatusEvent(
                component="classifier",
                message=f"Incremented reclassification count to {reclassification_count} due to previous failure: {previous_failure}",
                level="info",
            )
        )

        # Clear error state since we're handling the reclassification
        # This is safe because classifier provides a fresh start with new capabilities
        error_state_cleanup = {
            "control_has_error": False,
            "control_error_info": None,
            "control_last_error": None,
            "control_retry_count": 0,
            "control_current_step_retry_count": 0,
        }

    # Planning state updates
    planning_fields = {
        "planning_active_capabilities": active_capabilities,
        "planning_execution_plan": None,
        "planning_current_step_index": 0,
    }

    # Control flow updates
    control_flow_update = {
        "control_reclassification_count": reclassification_count,
        "control_reclassification_reason": None,
    }

    # Status event with comprehensive metadata
    status_event = create_status_update(
        message=message,
        progress=1.0,
        complete=True,
        node="classifier",
        capabilities_selected=len(active_capabilities),
        capability_names=active_capabilities,
        reclassification=bool(previous_failure),
        reclassification_count=reclassification_count,
        bypass_mode=is_bypass,
    )

    return {**planning_fields, **control_flow_update, **error_state_cleanup, **status_event}


def _detect_reclassification_scenario(state: AgentState) -> str | None:
    """Detect if this classification is a reclassification due to a previous error.

    Analyzes the current agent state to determine if this classification run
    is happening because a previous capability (like orchestrator) failed and
    requested reclassification.

    :param state: Current agent state containing error information
    :type state: AgentState
    :return: Reclassification reason string if this is a reclassification, None otherwise
    :rtype: Optional[str]
    """
    # Check if there's an active error state
    has_error = state.get("control_has_error", False)
    if not has_error:
        return None

    # Extract error information
    error_info = state.get("control_error_info", {})
    error_classification = error_info.get("classification")

    # Validate error classification exists and has required attributes
    if not error_classification or not hasattr(error_classification, "severity"):
        return None

    # Check if this is specifically a reclassification error
    try:
        is_reclassification = error_classification.severity.name == "RECLASSIFICATION"
    except AttributeError:
        # Handle case where severity doesn't have .name attribute
        is_reclassification = str(error_classification.severity) == "RECLASSIFICATION"

    if not is_reclassification:
        return None

    # Build reclassification reason string
    capability_name = error_info.get("capability_name", "unknown")
    user_message = (
        getattr(error_classification, "user_message", None) or "Reclassification required"
    )

    return f"Capability {capability_name} requested reclassification: {user_message}"


class CapabilityClassifier:
    """Handles individual capability classification with proper resource management."""

    def __init__(self, task: str, state: AgentState, logger, previous_failure: str | None = None):
        self.task = task
        self.state = state
        self.logger = logger
        self.previous_failure = previous_failure

    async def classify(self, capability: BaseCapability, semaphore: asyncio.Semaphore) -> bool:
        """Classify a single capability with semaphore-controlled concurrency.

        :param capability: The capability to analyze
        :param semaphore: Semaphore for concurrency control
        :return: True if capability is required, False otherwise
        """
        async with semaphore:  # Proper semaphore usage
            return await self._perform_classification(capability)

    async def _perform_classification(self, capability: BaseCapability) -> bool:
        """Perform the actual classification logic."""
        # Validate classifier availability
        classifier = self._get_classifier(capability)
        if not classifier:
            return False

        # Build classification prompt
        message = self._build_classification_prompt(classifier)
        self.logger.debug(
            f"\n\nTask Analyzer System Prompt for capability '{capability.name}':\n{message}\n\n"
        )

        # Emit LLM prompt event for TUI display (key=capability.name for accumulation)
        self.logger.emit_llm_request(message, key=capability.name)

        # Execute classification
        try:
            # Set caller context for API call logging (propagates through asyncio.to_thread)
            from osprey.models import set_api_call_context

            set_api_call_context(
                function="_perform_classification",
                module="classification_node",
                class_name="CapabilityClassifier",
                line=387,
                extra={"capability": capability.name},
            )

            response_data = await asyncio.to_thread(
                get_chat_completion,
                model_config=get_model_config("classifier"),
                message=message,
                output_model=CapabilityMatch,
            )

            # Emit LLM response event for TUI display (key=capability.name for accumulation)
            if isinstance(response_data, CapabilityMatch):
                response_json = response_data.model_dump_json()
            else:
                response_json = str(response_data)
            self.logger.emit_llm_response(response_json, key=capability.name)

            result = self._process_classification_response(capability, response_data)
            self.logger.info(f" >>> Capability '{capability.name}' >>> {result}")
            return result

        except Exception as e:
            self.logger.error(f"Error in capability classification for '{capability.name}': {e}")
            return False

    def _get_classifier(self, capability: BaseCapability):
        """Get classifier with proper error handling."""
        try:
            classifier = capability.classifier_guide
            if not classifier:
                # Not a warning - capability may intentionally not have a classifier
                # (e.g., direct-chat-only capabilities that override _create_classifier_guide to return None)
                self.logger.debug(
                    f"No classifier guide for capability '{capability.name}' - skipping classification"
                )
                return None
            return classifier
        except Exception as e:
            self.logger.error(f"Error loading classifier for capability '{capability.name}': {e}")
            # For import errors, skip this capability instead of failing entire classification
            if isinstance(e, (ImportError, ModuleNotFoundError, NameError)):
                self.logger.warning(
                    f"Skipping capability '{capability.name}' due to import error: {e}"
                )
                return None
            # For other errors, re-raise with capability context for better error reporting
            raise Exception(f"Capability '{capability.name}' classifier failed: {e}") from e

    def _build_classification_prompt(self, classifier) -> str:
        """Build the classification prompt."""
        capability_instructions = classifier.instructions
        examples_string = ClassifierExample.join(classifier.examples, randomize=True)

        prompt_provider = get_framework_prompts()
        classification_builder = prompt_provider.get_classification_prompt_builder()
        system_prompt = classification_builder.get_system_instructions(
            capability_instructions=capability_instructions,
            classifier_examples=examples_string,
            context=None,
            previous_failure=self.previous_failure,
        )
        return f"{system_prompt}\n\nUser request:\n{self.task}"

    def _process_classification_response(self, capability: BaseCapability, response_data) -> bool:
        """Process and validate classification response."""
        if isinstance(response_data, CapabilityMatch):
            return response_data.is_match
        else:
            self.logger.error(
                f"Classification call for '{capability.name}' did not return a CapabilityMatch. Got: {type(response_data)}"
            )
            return False


async def select_capabilities(
    task: str,
    available_capabilities: list[BaseCapability],
    state: AgentState,
    logger,
    previous_failure: str | None = None,
) -> list[str]:
    """Select capabilities needed for the task by using classification.

    :param task: Task description for analysis
    :type task: str
    :param available_capabilities: Available capabilities to choose from
    :type available_capabilities: List[BaseCapability]
    :param state: Current agent state
    :type state: AgentState
    :param logger: Logger instance
    :param previous_failure: Previous failure reason for reclassification context
    :return: List of capability names needed for the task
    :rtype: List[str]
    """

    # Get registry to access always-active capability names
    registry = get_registry()
    always_active_names = registry.get_always_active_capability_names()

    active_capabilities: list[str] = []

    # Step 1: Add always-active capabilities from registry configuration
    for capability in available_capabilities:
        if capability.name in always_active_names:
            active_capabilities.append(capability.name)

    # Step 2: Classify remaining capabilities (those not marked as always_active)
    remaining_capabilities = [
        cap for cap in available_capabilities if cap.name not in always_active_names
    ]

    if remaining_capabilities:
        # Get classification configuration for concurrency control
        classification_config = get_classification_config()
        max_concurrent = classification_config["max_concurrent_classifications"]

        logger.info(
            f"Classifying {len(remaining_capabilities)} capabilities with max {max_concurrent} concurrent requests"
        )

        # Create classifier instance with shared context
        classifier = CapabilityClassifier(task, state, logger, previous_failure)

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        # Create classification tasks with proper semaphore usage
        classification_tasks = [
            classifier.classify(capability, semaphore) for capability in remaining_capabilities
        ]

        # Execute all classifications in parallel with semaphore control
        classification_results = await asyncio.gather(*classification_tasks, return_exceptions=True)

        # Process results and collect active capabilities
        for capability, result in zip(remaining_capabilities, classification_results, strict=False):
            if isinstance(result, Exception):
                logger.error(f"Classification failed for capability '{capability.name}': {result}")
                # Skip failed classifications - don't activate capability on error
                continue
            elif result is True:
                active_capabilities.append(capability.name)

    # Step 3: Expand dependencies — ensure provider capabilities are included
    active_capabilities = _expand_capability_dependencies(active_capabilities, state, logger)

    logger.info(f"{len(active_capabilities)} capabilities required: {active_capabilities}")
    return active_capabilities


def _expand_capability_dependencies(
    selected_names: list[str],
    state: AgentState,
    logger,
) -> list[str]:
    """Expand selected capabilities to include providers for unsatisfied dependencies.

    For each selected capability, checks its ``requires`` list.  If a required
    context type is not already available in ``capability_context_data`` (from a
    previous turn), the capability that ``provides`` it is added to the selected
    set.  This is applied transitively until no new capabilities are added.

    :param selected_names: Initially selected capability names
    :param state: Current agent state (used to check existing context data)
    :param logger: Logger instance
    :return: Expanded list of capability names
    """
    registry = get_registry()

    # Build provider map: context_type → capability name (from ALL registered)
    provider_map: dict[str, str] = {}
    for cap in registry.get_all_capabilities():
        for provided_type in getattr(cap, "provides", []) or []:
            provider_map[provided_type] = getattr(cap, "name", "unknown")

    context_data = state.get("capability_context_data", {})
    expanded = list(selected_names)
    expanded_set = set(selected_names)

    # Iterative transitive closure
    changed = True
    while changed:
        changed = False
        for name in list(expanded):
            cap = registry.get_capability(name)
            if not cap:
                continue
            requires = getattr(cap, "requires", []) or []
            for req in requires:
                context_type = req[0] if isinstance(req, tuple) else req
                # Skip if context already available from a previous turn
                type_contexts = context_data.get(context_type, {})
                if type_contexts and isinstance(type_contexts, dict):
                    continue
                # Look up provider and add if found
                provider = provider_map.get(context_type)
                if provider and provider not in expanded_set:
                    expanded.append(provider)
                    expanded_set.add(provider)
                    changed = True
                    logger.info(
                        f"Dependency expansion: added '{provider}' "
                        f"(provides {context_type} required by {name})"
                    )

    return expanded
