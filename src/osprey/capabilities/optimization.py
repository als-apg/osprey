"""Optimization Capability - Service Gateway for XOpt Machine Optimization.

This capability acts as the gateway between the main agent graph and the
XOpt optimizer service, providing seamless integration for autonomous
machine optimization workflows with human approval, machine state awareness,
and result analysis.

The capability provides a clean abstraction layer that:
1. **Service Integration**: Manages communication with the XOpt optimizer service
2. **Approval Workflows**: Integrates with the approval system for optimization control
3. **Context Management**: Handles context data passing and result context creation
4. **Error Handling**: Provides sophisticated error classification and recovery

Key architectural features:
    - Service gateway pattern for clean separation of concerns
    - LangGraph-native approval workflow integration
    - Comprehensive context management for cross-capability data flow
    - Structured result processing with optimization metadata
    - Error classification with domain-specific recovery strategies

.. note::
   This capability requires the XOpt optimizer service to be available in the
   framework registry. All optimization execution is managed by the separate service.

.. warning::
   Optimization operations may require user approval depending on the configured
   approval policies. Execution may be suspended pending user confirmation.

.. seealso::
   :class:`osprey.services.xopt_optimizer.XOptOptimizerService` : Optimization service
   :class:`OptimizationResultContext` : Optimization result context structure
"""

from typing import Any, ClassVar

from langgraph.types import Command
from pydantic import Field

from osprey.approval import (
    clear_approval_state,
    create_approval_type,
    get_approval_resume_data,
    handle_service_with_interrupts,
)
from osprey.base.capability import BaseCapability
from osprey.base.decorators import capability_node
from osprey.base.errors import ErrorClassification, ErrorSeverity
from osprey.base.examples import OrchestratorGuide, TaskClassifierGuide
from osprey.context.base import CapabilityContext
from osprey.prompts.loader import get_framework_prompts
from osprey.registry import get_registry
from osprey.services.xopt_optimizer import XOptExecutionRequest, XOptServiceResult
from osprey.state import StateManager
from osprey.utils.config import get_full_configuration
from osprey.utils.logger import get_logger

# Module-level logger for helper functions
logger = get_logger("optimization")


# ========================================================
# Context Class
# ========================================================


class OptimizationResultContext(CapabilityContext):
    """Context for XOpt optimization results.

    Provides structured context for optimization execution results including
    the run artifact, strategy used, iteration count, and analysis summary.
    This context enables other capabilities to access optimization outcomes
    for downstream processing or response generation.

    :param run_artifact: Optimization run output data
    :type run_artifact: Dict[str, Any]
    :param strategy: Strategy used (exploration/optimization)
    :type strategy: str
    :param total_iterations: Number of iterations completed
    :type total_iterations: int
    :param analysis_summary: Summary of optimization analysis
    :type analysis_summary: Dict[str, Any]
    :param recommendations: List of recommendations from analysis
    :type recommendations: List[str]
    :param generated_yaml: XOpt YAML configuration used
    :type generated_yaml: str

    .. note::
       The run_artifact contains the primary optimization outputs that
       other capabilities can use for further processing or analysis.

    .. seealso::
       :class:`osprey.context.base.CapabilityContext` : Base context functionality
       :class:`osprey.services.xopt_optimizer.XOptServiceResult` : Service result structure
    """

    run_artifact: dict[str, Any] = Field(default_factory=dict)
    strategy: str = ""
    total_iterations: int = 0
    analysis_summary: dict[str, Any] = Field(default_factory=dict)
    recommendations: list[str] = Field(default_factory=list)
    generated_yaml: str = ""

    CONTEXT_TYPE: ClassVar[str] = "OPTIMIZATION_RESULT"
    CONTEXT_CATEGORY: ClassVar[str] = "OPTIMIZATION_DATA"

    @property
    def context_type(self) -> str:
        """Return the context type identifier."""
        return self.CONTEXT_TYPE

    def get_access_details(self, key: str) -> dict[str, Any]:
        """Provide access information for optimization results.

        :param key: Context key name for access pattern generation
        :type key: str
        :return: Dictionary containing access details and patterns
        :rtype: Dict[str, Any]
        """
        return {
            "run_artifact": "Optimization run output data",
            "strategy": f"Strategy used: {self.strategy}",
            "total_iterations": f"Completed {self.total_iterations} iterations",
            "analysis_summary": "Summary of optimization analysis",
            "recommendations": "List of recommendations from analysis",
            "generated_yaml": "XOpt YAML configuration used",
            "access_pattern": f"context.OPTIMIZATION_RESULT.{key}",
        }

    def get_summary(self) -> dict[str, Any]:
        """Generate summary for display and LLM processing.

        :return: Dictionary containing summarized optimization results
        :rtype: Dict[str, Any]
        """
        return {
            "type": "Optimization Result",
            "strategy": self.strategy,
            "iterations": self.total_iterations,
            "recommendations_count": len(self.recommendations),
            "has_artifact": bool(self.run_artifact),
        }


def _create_optimization_context(service_result: XOptServiceResult) -> OptimizationResultContext:
    """Create OptimizationResultContext from service result.

    This helper function transforms the XOpt service result into a structured
    context object that can be stored in state and accessed by other capabilities.

    :param service_result: Result from XOpt optimizer service
    :type service_result: XOptServiceResult
    :return: Structured context for optimization results
    :rtype: OptimizationResultContext
    """
    return OptimizationResultContext(
        run_artifact=service_result.run_artifact,
        strategy=service_result.strategy.value,
        total_iterations=service_result.total_iterations,
        analysis_summary=service_result.analysis_summary,
        recommendations=list(service_result.recommendations),
        generated_yaml=service_result.generated_yaml,
    )


# ========================================================
# Convention-Based Capability Implementation
# ========================================================


@capability_node
class OptimizationCapability(BaseCapability):
    """Machine optimization capability using XOpt.

    Acts as the gateway between the main agent graph and the XOpt optimizer
    service, providing seamless integration for optimization workflows
    with approval handling and result processing.

    This is a framework-level capability that can be customized per facility
    through the prompt builder system and configuration.

    Key architectural features:
        - Service gateway pattern for clean separation between capability and service
        - Comprehensive context management for cross-capability data access
        - LangGraph-native approval workflow integration with interrupt handling
        - Structured result processing with optimization metadata

    .. note::
       Requires XOpt optimizer service availability in framework registry.
       All actual optimization execution is delegated to the service.

    .. warning::
       Optimization operations may trigger approval interrupts that suspend
       execution until user confirmation is received.
    """

    name = "optimization"
    description = "Optimize machine parameters using XOpt autonomous optimization"
    provides = ["OPTIMIZATION_RESULT"]
    requires = []  # Can optionally use CHANNEL_ADDRESSES if available

    # ========================================
    # ORCHESTRATOR / CLASSIFIER GUIDES
    # ========================================

    def _create_orchestrator_guide(self) -> OrchestratorGuide | None:
        """Create orchestrator guide from prompt builder system.

        Retrieves orchestrator guidance from the application's prompt builder
        system. This guide teaches the orchestrator when and how to include
        optimization steps in execution plans.

        :return: Orchestrator guide with examples and notes
        :rtype: Optional[OrchestratorGuide]
        """
        prompt_provider = get_framework_prompts()
        optimization_builder = prompt_provider.get_optimization_prompt_builder()

        return optimization_builder.get_orchestrator_guide()

    def _create_classifier_guide(self) -> TaskClassifierGuide | None:
        """Create task classification guide from prompt builder system.

        Retrieves task classification guidance from the application's prompt
        builder system. This guide teaches the classifier when user requests
        should be routed to optimization operations.

        :return: Classifier guide with examples
        :rtype: Optional[TaskClassifierGuide]
        """
        prompt_provider = get_framework_prompts()
        optimization_builder = prompt_provider.get_optimization_prompt_builder()

        return optimization_builder.get_classifier_guide()

    # ========================================
    # MAIN EXECUTION
    # ========================================

    async def execute(self) -> dict[str, Any]:
        """Execute optimization with service integration and approval handling.

        Implements the complete optimization workflow including service invocation,
        approval management, and result processing. The method handles both normal
        execution scenarios and approval resume scenarios with proper state management.

        :return: State updates with optimization results and context data
        :rtype: Dict[str, Any]

        :raises RuntimeError: If XOpt optimizer service is not available in registry
        :raises XOptExecutionError: If optimization execution fails
        """

        # ========================================
        # GENERIC SETUP (needed for both paths)
        # ========================================

        # Get unified logger with automatic streaming support
        cap_logger = self.get_logger()
        step = self._step

        cap_logger.status("Initializing XOpt optimizer service...")

        # Get XOpt service from registry (runtime lookup)
        registry = get_registry()
        xopt_service = registry.get_service("xopt_optimizer")

        if not xopt_service:
            raise RuntimeError("XOpt optimizer service not available in registry")

        # Get the full configurable from main graph
        main_configurable = get_full_configuration()

        # Create service config by extending main graph's configurable
        service_config = {
            "configurable": {
                **main_configurable,
                "thread_id": f"xopt_service_{step.get('context_key', 'default')}",
                "checkpoint_ns": "xopt_optimizer",
            }
        }

        # ========================================
        # APPROVAL CASE (handle first)
        # ========================================

        # Check if this is a resume from approval
        has_approval_resume, approved_payload = get_approval_resume_data(
            self._state, create_approval_type("xopt_optimizer")
        )

        if has_approval_resume:
            if approved_payload:
                cap_logger.resume("Sending approval response to XOpt optimizer service")
                resume_response = {"approved": True}
                resume_response.update(approved_payload)
            else:
                cap_logger.key_info("XOpt optimization was rejected by user")
                resume_response = {"approved": False}

            try:
                service_result = await xopt_service.ainvoke(
                    Command(resume=resume_response), config=service_config
                )

                cap_logger.info("XOpt optimizer service completed successfully after approval")
                approval_cleanup = clear_approval_state()

                # Process results
                results_context = _create_optimization_context(service_result)
                cap_logger.success(
                    f"Optimization complete - {service_result.total_iterations} iterations"
                )

                # Store context and merge cleanup
                result_updates = StateManager.store_context(
                    self._state,
                    "OPTIMIZATION_RESULT",
                    step.get("context_key"),
                    results_context,
                )
                result_updates.update(approval_cleanup)
                return result_updates

            except Exception as e:
                # Import here to avoid circular imports
                from langgraph.errors import GraphInterrupt
                from langgraph.types import interrupt

                # Check if this is a GraphInterrupt (service looped and needs approval for next iteration)
                if isinstance(e, GraphInterrupt):
                    cap_logger.info(
                        "XOptOptimizer: Service completed iteration and requests approval for next"
                    )

                    try:
                        # Extract interrupt data from GraphInterrupt
                        interrupt_data = e.args[0][0].value
                        cap_logger.debug(
                            f"XOptOptimizer: Extracted interrupt data with keys: {list(interrupt_data.keys())}"
                        )

                        # Re-raise interrupt in main graph context for next iteration
                        cap_logger.info(
                            "⏸️  XOptOptimizer: Creating approval interrupt for next iteration"
                        )
                        interrupt(interrupt_data)

                        # This line should never be reached
                        cap_logger.error(
                            "UNEXPECTED: interrupt() returned instead of pausing execution"
                        )
                        raise RuntimeError("Interrupt mechanism failed in XOptOptimizer")

                    except (IndexError, KeyError, AttributeError) as extract_error:
                        cap_logger.error(
                            f"XOptOptimizer: Failed to extract interrupt data: {extract_error}"
                        )
                        raise RuntimeError(
                            f"XOptOptimizer: Failed to handle service interrupt: {extract_error}"
                        ) from extract_error
                else:
                    # Re-raise non-interrupt exceptions
                    raise

        # ========================================
        # NORMAL EXECUTION (new request)
        # ========================================

        user_query = self._state.get("input_output", {}).get("user_query", "")
        task_objective = self.get_task_objective(default="")
        capability_contexts = self._state.get("capability_context_data", {})

        # Create execution request
        execution_request = XOptExecutionRequest(
            user_query=user_query,
            optimization_objective=task_objective,
            capability_context_data=capability_contexts,
            require_approval=True,
            max_iterations=3,
        )

        cap_logger.status("Invoking XOpt optimizer service...")

        # Handle service invocation with interrupt support
        service_result = await handle_service_with_interrupts(
            service=xopt_service,
            request=execution_request,
            config=service_config,
            logger=cap_logger,
            capability_name="XOptOptimizer",
        )

        # ========================================
        # RESULT PROCESSING
        # ========================================

        cap_logger.status("Processing optimization results...")

        results_context = _create_optimization_context(service_result)

        cap_logger.success(f"Optimization complete - {service_result.total_iterations} iterations")

        return StateManager.store_context(
            self._state,
            "OPTIMIZATION_RESULT",
            step.get("context_key"),
            results_context,
        )

    # ========================================
    # ERROR CLASSIFICATION
    # ========================================

    @staticmethod
    def classify_error(exc: Exception, context: dict) -> ErrorClassification:
        """Classify optimization errors for appropriate handling.

        :param exc: The exception that occurred
        :type exc: Exception
        :param context: Additional context about the error
        :type context: dict
        :return: Error classification with severity and user message
        :rtype: ErrorClassification
        """
        from osprey.services.xopt_optimizer.exceptions import (
            MachineStateAssessmentError,
            XOptExecutionError,
            YamlGenerationError,
        )

        if isinstance(exc, MachineStateAssessmentError):
            return ErrorClassification(
                severity=ErrorSeverity.REPLANNING,
                user_message=f"Machine not ready for optimization: {exc}",
                metadata={
                    "replanning_reason": str(exc),
                    "suggestions": ["Wait for machine conditions to improve", "Check interlocks"],
                },
            )

        elif isinstance(exc, YamlGenerationError):
            return ErrorClassification(
                severity=ErrorSeverity.REPLANNING,
                user_message=f"Failed to generate optimization configuration: {exc}",
                metadata={"replanning_reason": str(exc)},
            )

        elif isinstance(exc, XOptExecutionError):
            return ErrorClassification(
                severity=ErrorSeverity.CRITICAL,
                user_message=f"Optimization execution failed: {exc}",
                metadata={"safety_abort_reason": str(exc)},
            )

        else:
            return ErrorClassification(
                severity=ErrorSeverity.CRITICAL,
                user_message=f"Unexpected optimization error: {exc}",
                metadata={"safety_abort_reason": str(exc)},
            )
