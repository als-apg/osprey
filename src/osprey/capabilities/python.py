"""Python Capability - Service Gateway for Code Generation and Execution

This capability acts as the sophisticated gateway between the main agent graph and
the Python executor service, providing seamless integration for code generation,
execution, and result processing. It handles the complete Python execution workflow
including service invocation, approval management, interrupt propagation, and
structured result processing.

The capability provides a clean abstraction layer that:
1. **Service Integration**: Manages communication with the Python executor service
2. **Approval Workflows**: Integrates with the approval system for code execution control
3. **Context Management**: Handles context data passing and result context creation
4. **Error Handling**: Provides sophisticated error classification and recovery
5. **Result Processing**: Structures execution results for capability integration

Key architectural features:
    - Service gateway pattern for clean separation of concerns
    - LangGraph-native approval workflow integration
    - Comprehensive context management for cross-capability data flow
    - Structured result processing with execution metadata
    - Error classification with domain-specific recovery strategies

The capability uses the @capability_node decorator for full LangGraph integration
including streaming, configuration management, error handling, and checkpoint support.

.. note::
   This capability requires the Python executor service to be available in the
   framework registry. All code execution is managed by the separate service.

.. warning::
   Python code execution may require user approval depending on the configured
   approval policies. Execution may be suspended pending user confirmation.

.. seealso::
   :class:`osprey.services.python_executor.PythonExecutorService` : Execution service
   :class:`PythonResultsContext` : Execution result context structure
   :class:`osprey.approval.ApprovalManager` : Code execution approval workflows
"""

from typing import Any, ClassVar

from langgraph.types import Command

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
from osprey.context.context_manager import ContextManager, recursively_summarize_data
from osprey.prompts.loader import get_framework_prompts
from osprey.registry import get_registry
from osprey.services.python_executor import PythonServiceResult
from osprey.services.python_executor.models import PlanningMode, PythonExecutionRequest
from osprey.state import ArtifactType, StateManager
from osprey.utils.config import get_full_configuration
from osprey.utils.logger import get_logger

# Module-level logger for helper functions
logger = get_logger("python")


# ========================================================
# Context Class
# ========================================================


class PythonResultsContext(CapabilityContext):
    """Context for Python execution results with comprehensive execution metadata.

    Provides structured context for Python code execution results including
    generated code, execution output, computed results, performance metrics,
    and resource links. This context enables other capabilities to access
    both the execution process details and the computed results.

    The context maintains complete execution metadata including timing information,
    output logs, error details, and generated resources like figures and notebooks.
    This comprehensive tracking enables sophisticated debugging, result analysis,
    and cross-capability integration.

    :param code: Generated Python code that was executed
    :type code: str
    :param output: Complete stdout/stderr output from code execution
    :type output: str
    :param results: Structured results dictionary from execution (from results.json)
    :type results: Optional[Dict[str, Any]]
    :param error: Error message if execution failed
    :type error: Optional[str]
    :param execution_time: Total execution time in seconds
    :type execution_time: float
    :param folder_path: Path to execution folder containing generated files
    :type folder_path: Optional[str]
    :param notebook_path: Path to generated Jupyter notebook file
    :type notebook_path: Optional[str]
    :param notebook_link: URL link to access the generated notebook
    :type notebook_link: Optional[str]
    :param figure_paths: List of paths to generated figure/visualization files
    :type figure_paths: Optional[list]

    .. note::
       The results field contains structured data computed by the Python code,
       while output contains the raw execution logs and print statements.

    .. seealso::
       :class:`osprey.context.base.CapabilityContext` : Base context functionality
       :class:`osprey.services.python_executor.PythonServiceResult` : Service result structure
    """

    code: str
    output: str
    results: dict[str, Any] | None = None  # Actual computed results from results.json
    error: str | None = None
    execution_time: float = 0.0
    folder_path: str | None = None
    notebook_path: str | None = None
    notebook_link: str | None = None
    figure_paths: list | None = None

    CONTEXT_TYPE: ClassVar[str] = "PYTHON_RESULTS"
    CONTEXT_CATEGORY: ClassVar[str] = "COMPUTATIONAL_DATA"

    def __post_init__(self):
        if self.figure_paths is None:
            self.figure_paths = []

    @property
    def context_type(self) -> str:
        return self.CONTEXT_TYPE

    def get_access_details(self, key: str) -> dict[str, Any]:
        """Provide comprehensive access information for Python execution results.

        Generates detailed access information for other capabilities to understand
        how to interact with Python execution results. Includes access patterns,
        data structure descriptions, and usage examples for both computed results
        and execution metadata.

        :param key_name: Optional context key name for access pattern generation
        :type key_name: Optional[str]
        :return: Dictionary containing comprehensive access details and examples
        :rtype: Dict[str, Any]

        .. note::
           Access details distinguish between execution metadata (code, output, timing)
           and computed results (structured data from results.json).
        """
        return {
            "code": "Python code that was executed",
            "output": "Stdout/stderr logs from code execution",
            "results": (
                "Computed results dictionary from results.json"
                if self.results
                else "No computed results"
            ),
            "error": "Error message if execution failed" if self.error else "No errors",
            "execution_time": f"Execution time: {self.execution_time:.2f} seconds",
            "folder_path": "Path to execution folder",
            "notebook_link": "Jupyter notebook link for review",
            "access_pattern": f"context.{self.CONTEXT_TYPE}.{key}.results",
            "example_usage": f"context.{self.CONTEXT_TYPE}.{key}.results gives the computed results dictionary",
        }

    def get_summary(self) -> dict[str, Any]:
        """Generate summary of Python execution for display and analysis.

        Creates a comprehensive summary of the Python execution including both
        metadata (execution time, status, resource counts) and actual computed
        results for display in user interfaces and inclusion in agent responses.

        The summary includes structured data rather than pre-formatted strings
        to enable robust LLM processing and flexible presentation formatting.

        :param key_name: Optional context key name for reference
        :type key_name: Optional[str]
        :return: Dictionary containing execution summary with results
        :rtype: Dict[str, Any]

        .. note::
           Includes both execution metadata and computed results to provide
           complete context for response generation and analysis.
        """
        summary = {
            "type": "Python Results",
            "code_lines": len(self.code.split("\n")) if self.code else 0,
            "execution_time": f"{self.execution_time:.2f}s",
            "notebook_available": bool(self.notebook_link),
            "figure_count": len(self.figure_paths) if self.figure_paths else 0,
            "status": "Failed" if self.error else "Success",
        }

        # Include summarized execution data to prevent context overflow
        if self.results:
            summary["results"] = recursively_summarize_data(self.results)
        if self.output:
            # Truncate large output logs
            if len(self.output) > 1000:
                summary["output"] = (
                    f"{self.output[:500]}... (truncated from {len(self.output)} chars)"
                )
            else:
                summary["output"] = self.output
        if self.error:
            summary["error"] = self.error  # Errors are usually short
        if self.code:
            # Truncate large code blocks
            if len(self.code) > 2000:
                lines = self.code.split("\n")
                if len(lines) > 50:
                    summary["code"] = (
                        f"{chr(10).join(lines[:25])}... (truncated from {len(lines)} lines)"
                    )
                else:
                    summary["code"] = (
                        f"{self.code[:1000]}... (truncated from {len(self.code)} chars)"
                    )
            else:
                summary["code"] = self.code

        return summary


# ========================================================
# Private Helper Functions
# ========================================================


def _create_python_context(service_result: PythonServiceResult) -> PythonResultsContext:
    """Create Python results context from structured service execution result.

    Transforms the structured result from the Python executor service into a
    capability context object suitable for framework integration. The service
    guarantees result structure validation, enabling clean context creation
    without additional validation requirements.

    This helper function provides a clean abstraction between the service result
    format and the capability context structure, enabling easy maintenance if
    either structure evolves independently.

    :param service_result: Structured execution result from Python executor service
    :type service_result: PythonServiceResult
    :return: Ready-to-store context object containing all execution details
    :rtype: PythonResultsContext

    .. note::
       The service guarantees execution_result validity and structure,
       eliminating the need for additional validation in this helper.

    .. seealso::
       :class:`osprey.services.python_executor.PythonServiceResult` : Input structure
       :class:`PythonResultsContext` : Output context structure
       :func:`_create_python_capability_prompts` : Related prompt generation helper
       :meth:`PythonCapability.execute` : Main capability method that uses this helper
    """
    # Service guarantees execution_result is valid, so just extract fields directly
    execution_result = service_result.execution_result

    return PythonResultsContext(
        code=service_result.generated_code,
        output=execution_result.stdout,
        results=execution_result.results,
        execution_time=execution_result.execution_time,
        folder_path=str(execution_result.folder_path),
        notebook_path=str(execution_result.notebook_path),
        notebook_link=execution_result.notebook_link,
        figure_paths=execution_result.figure_paths,
    )


def _create_python_capability_prompts(
    task_objective: str, user_query: str, context_description: str = ""
) -> list[str]:
    """Create capability-specific prompts for Python code generation and execution.

    Builds structured prompts that provide the Python executor service with
    comprehensive context about the user's request, task objectives, and
    available data context. These prompts guide the LLM in generating
    appropriate Python code for the specific task.

    This function now integrates with the prompt builder system to inject
    domain-specific instructions from application-level prompt builders.

    :param task_objective: Specific task objective from the execution plan step
    :type task_objective: str
    :param user_query: Original user query or broader task description
    :type user_query: str
    :param context_description: Description of available context data for code access
    :type context_description: str
    :return: List of structured prompts for Python code generation
    :rtype: list[str]

    .. note::
       Prompts are structured to provide clear guidance while avoiding
       redundancy when task_objective and user_query contain similar information.

    .. seealso::
       :class:`osprey.services.python_executor.PythonExecutionRequest` : Request structure
       :func:`_create_python_context` : Related context creation helper
       :class:`osprey.context.context_manager.ContextManager` : Context access description source
       :meth:`PythonCapability.execute` : Main capability method that uses these prompts
    """
    prompts = []

    if task_objective:
        prompts.append(f"TASK: {task_objective}")
    if user_query and user_query != task_objective:
        prompts.append(f"USER REQUEST: {user_query}")
    if context_description:
        prompts.append(f"CONTEXT ACCESS DESCRIPTION: {context_description}")

    # Inject domain-specific instructions from prompt builder system
    try:
        python_builder = get_framework_prompts().get_python_prompt_builder()
        domain_instructions = python_builder.get_instructions()
        if domain_instructions:
            prompts.append(domain_instructions)
    except Exception as e:
        # Graceful degradation if no custom prompts or prompt system not initialized
        logger.debug(f"Could not load Python prompt builder instructions: {e}")
        pass

    return prompts


# ========================================================
# Convention-Based Capability Implementation
# ========================================================


@capability_node
class PythonCapability(BaseCapability):
    """Python execution capability providing comprehensive code generation and execution.

    Acts as the sophisticated gateway between the main agent graph and the Python
    executor service, providing seamless integration for Python code generation,
    execution, and result processing. The capability handles the complete Python
    execution workflow including approval management, context integration, and
    structured result processing.

    The capability implements a dual-execution pattern that handles both normal
    execution flows and approval resume scenarios:

    1. **Normal Execution**: Creates execution requests with context data and invokes
       the Python executor service with comprehensive prompt engineering
    2. **Approval Resume**: Handles resumption of execution after user approval
       with proper state management and cleanup

    Key architectural features:
        - Service gateway pattern for clean separation between capability and executor
        - Comprehensive context management for cross-capability data access
        - LangGraph-native approval workflow integration with interrupt handling
        - Structured result processing with execution metadata and computed results
        - Domain-specific error classification for sophisticated recovery strategies

    The capability leverages the framework's registry system for service discovery,
    configuration management for proper service setup, and context management for
    seamless data flow between capabilities.

    .. note::
       Requires Python executor service availability in framework registry.
       All actual code generation and execution is delegated to the service.

    .. warning::
       Python code execution may trigger approval interrupts that suspend
       execution until user confirmation is received.

    .. seealso::
       :class:`osprey.base.capability.BaseCapability` : Base capability functionality
       :class:`osprey.services.python_executor.PythonExecutorService` : Execution backend
       :class:`PythonResultsContext` : Execution result context structure
    """

    name = "python"
    description = "Generate and execute Python code using the Python executor service"
    provides = ["PYTHON_RESULTS"]
    requires = []

    async def execute(self) -> dict[str, Any]:
        """Execute Python capability with comprehensive service integration and approval handling.

        Implements the complete Python execution workflow including service invocation,
        approval management, and result processing. The method handles both normal
        execution scenarios and approval resume scenarios with proper state management.

        The execution follows this sophisticated pattern:
        1. **Service Setup**: Retrieves Python executor service and configures execution environment
        2. **Approval Check**: Determines if this is an approval resume or new execution
        3. **Request Creation**: Builds comprehensive execution request with context data
        4. **Service Invocation**: Invokes Python executor with proper configuration
        5. **Result Processing**: Creates structured context from execution results

        The method integrates with the framework's approval system to handle code
        execution approval workflows, ensuring user control over potentially sensitive
        code execution while maintaining seamless execution flow.

        :return: State updates with Python execution results and context data
        :rtype: Dict[str, Any]

        :raises RuntimeError: If Python executor service is not available in registry
        :raises CodeRuntimeError: If Python code execution fails
        :raises ServiceInvocationError: If service communication fails

        .. note::
           Uses the framework's configuration system to pass all necessary
           configuration to the Python executor service including thread IDs.

        .. warning::
           May trigger LangGraph interrupts for approval workflows that suspend
           execution until user responds to approval requests.
        """

        # ========================================
        # GENERIC SETUP (needed for both paths)
        # ========================================

        # Get unified logger with automatic streaming support
        logger = self.get_logger()

        # Current step is injected by decorator
        step = self._step

        logger.status("Initializing Python executor service...")

        # Get Python executor service from registry (call get_registry() at runtime, not module import time)
        registry = get_registry()
        python_service = registry.get_service("python_executor")

        if not python_service:
            raise RuntimeError("Python executor service not available in registry")

        # Get the full configurable from main graph (needed for both approval and normal cases)
        main_configurable = get_full_configuration()

        # Create service config by extending main graph's configurable
        service_config = {
            "configurable": {
                **main_configurable,  # Pass all main graph configuration to service
                "thread_id": f"python_service_{step.get('context_key', 'default')}",  # Override thread ID for service
                "checkpoint_ns": "python_executor",  # Add checkpoint namespace for service
            }
        }

        # ========================================
        # APPROVAL CASE (handle first)
        # ========================================

        # Check if this is a resume from approval using centralized function
        has_approval_resume, approved_payload = get_approval_resume_data(
            self._state, create_approval_type("python_executor")
        )

        if has_approval_resume:
            if approved_payload:
                logger.resume("Sending approval response to Python executor service")
                logger.debug(f"Additional payload keys: {list(approved_payload.keys())}")

                # Resume execution with approval response
                resume_response = {"approved": True}
                resume_response.update(approved_payload)
            else:
                # Explicitly rejected
                logger.key_info("Python execution was rejected by user")
                resume_response = {"approved": False}

            # Resume the service with full configurable
            service_result = await python_service.ainvoke(
                Command(resume=resume_response), config=service_config
            )

            logger.info("✅ Python executor service completed successfully after approval")

            # Add approval cleanup to prevent state pollution
            approval_cleanup = clear_approval_state()

        else:
            # ========================================
            # REGULAR EXECUTION CASE
            # ========================================

            # Create execution request
            # Build capability-specific prompts with task information
            user_query = self._state.get("input_output", {}).get("user_query", "")
            task_objective = self.get_task_objective(default="")

            # Build capability-specific prompts using helper methods
            step_inputs = self.get_step_inputs()
            context_manager = ContextManager(self._state)
            context_description = context_manager.get_context_access_description(step_inputs)

            # Create capability-specific prompts
            capability_prompts = _create_python_capability_prompts(
                task_objective=task_objective,
                user_query=user_query,
                context_description=context_description,
            )

            if step_inputs:
                logger.info(f"Added context access description for {len(step_inputs)} inputs")

            # Get main graph's context data (raw dictionary that contains context data)
            # Python service will recreate ContextManager from this dictionary data
            capability_contexts = self._state.get("capability_context_data", {})

            # DEBUG: Log context data availability
            logger.debug(f"capability_context_data keys: {list(capability_contexts.keys())}")
            logger.debug(f"full state keys: {list(self._state.keys())}")

            execution_request = PythonExecutionRequest(
                user_query=user_query,
                task_objective=task_objective,
                expected_results={},
                capability_prompts=capability_prompts,
                execution_folder_name="python_capability",
                capability_context_data=capability_contexts,
                config=self._state.get("config"),
                retries=3,
                planning_mode=PlanningMode.GENERATOR_DRIVEN,
            )

            logger.status("Invoking Python executor service...")

            # Normal service execution using centralized interrupt handler
            service_result = await handle_service_with_interrupts(
                service=python_service,
                request=execution_request,
                config=service_config,
                logger=logger,
                capability_name="Python",
            )

        # Process results - single path for both approval and normal execution
        logger.status("Processing Python execution results...")

        # Create context using private helper function - ultra-clean!
        results_context = _create_python_context(service_result)

        # Service only returns on success, so always provide success feedback
        execution_time = results_context.execution_time
        figure_count = len(results_context.figure_paths)
        logger.success(f"Python execution complete - {execution_time:.2f}s, {figure_count} figures")

        # Store context using StateManager
        result_updates = StateManager.store_context(
            self._state, "PYTHON_RESULTS", step.get("context_key"), results_context
        )

        # Register artifacts using unified artifact system
        # Single accumulation pattern - clean and simple
        artifacts = None

        # Register figures as IMAGE artifacts
        for figure_path in results_context.figure_paths:
            artifact_update = StateManager.register_artifact(
                self._state,
                artifact_type=ArtifactType.IMAGE,
                capability="python_executor",
                data={"path": str(figure_path), "format": figure_path.suffix[1:].lower()},
                display_name="Python Execution Figure",
                metadata={
                    "execution_folder": results_context.folder_path,
                    "notebook_link": results_context.notebook_link,
                    "execution_time": results_context.execution_time,
                    "context_key": step.get("context_key"),
                },
                current_artifacts=artifacts,
            )
            artifacts = artifact_update["ui_artifacts"]

        # Register notebook as NOTEBOOK artifact
        if results_context.notebook_link:
            artifact_update = StateManager.register_artifact(
                self._state,
                artifact_type=ArtifactType.NOTEBOOK,
                capability="python_executor",
                data={
                    "path": str(results_context.notebook_path),
                    "url": results_context.notebook_link,
                },
                display_name="Python Execution Notebook",
                metadata={
                    "execution_folder": results_context.folder_path,
                    "execution_time": results_context.execution_time,
                    "context_key": step.get("context_key"),
                    "code_lines": (
                        len(results_context.code.split("\n")) if results_context.code else 0
                    ),
                },
                current_artifacts=artifacts,
            )
            artifacts = artifact_update["ui_artifacts"]

        # Build artifact updates (only ui_artifacts - legacy fields populated at finalization)
        artifact_updates = {"ui_artifacts": artifacts} if artifacts else {}

        # Combine all updates
        if has_approval_resume:
            return {**result_updates, **approval_cleanup, **artifact_updates}
        else:
            return {**result_updates, **artifact_updates}

    @staticmethod
    def classify_error(exc: Exception, context: dict) -> ErrorClassification:
        """Classify Python execution errors for appropriate recovery strategies.

        Provides domain-specific error classification for Python execution failures,
        enabling appropriate recovery strategies based on the specific failure mode.
        Most Python execution errors are classified as RETRIABLE since they often
        represent transient service issues rather than fundamental capability problems.

        :param exc: The exception that occurred during Python execution
        :type exc: Exception
        :param context: Error context including capability info and execution state
        :type context: dict
        :return: Error classification with recovery strategy and user messaging
        :rtype: ErrorClassification

        .. note::
           Service-related errors are generally retriable since they often
           represent temporary communication or resource issues.

        .. seealso::
           :class:`osprey.base.errors.ErrorClassification` : Error classification structure
           :class:`osprey.services.python_executor.exceptions.CodeRuntimeError` : Service-specific error
           :func:`handle_service_with_interrupts` : Service invocation with error handling
           :class:`osprey.base.errors.ErrorSeverity` : Available error severity levels
           :meth:`PythonCapability.execute` : Main method that uses this error classification
        """

        # Service-related errors are generally retriable
        return ErrorClassification(
            severity=ErrorSeverity.RETRIABLE,
            user_message=f"Python execution service error: {exc}",
            metadata={"technical_details": str(exc)},
        )

    def _create_orchestrator_guide(self) -> OrchestratorGuide | None:
        """Create orchestrator integration guide from prompt builder system.

        Retrieves sophisticated orchestration guidance from the application's
        prompt builder system. This guide teaches the orchestrator when and how
        to invoke Python execution within execution plans.

        :return: Orchestrator guide for Python capability integration
        :rtype: Optional[OrchestratorGuide]

        .. note::
           Guide content is provided by the application layer through the
           framework's prompt builder system for domain-specific customization.

        .. seealso::
           :class:`osprey.base.examples.OrchestratorGuide` : Guide structure returned by this method
           :meth:`_create_classifier_guide` : Complementary classifier guide creation
           :class:`osprey.prompts.loader.FrameworkPrompts` : Prompt system integration
        """
        prompt_provider = get_framework_prompts()
        python_builder = prompt_provider.get_python_prompt_builder()

        return python_builder.get_orchestrator_guide()

    def _create_classifier_guide(self) -> TaskClassifierGuide | None:
        """Create task classification guide from prompt builder system.

        Retrieves task classification guidance from the application's prompt
        builder system. This guide teaches the classifier when user requests
        should be routed to Python code execution.

        :return: Classification guide for Python capability activation
        :rtype: Optional[TaskClassifierGuide]

        .. note::
           Guide content is provided by the application layer through the
           framework's prompt builder system for domain-specific examples.
        """
        prompt_provider = get_framework_prompts()
        python_builder = prompt_provider.get_python_prompt_builder()

        return python_builder.get_classifier_guide()
