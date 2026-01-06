"""
Python Executor Service - LangGraph Architecture

Main service class that orchestrates Python code generation, analysis, and execution
using LangGraph with human approval integration through interrupts.
"""

from typing import Any

from langgraph.graph import StateGraph
from langgraph.types import Command

from osprey.graph.graph_builder import (
    create_async_postgres_checkpointer,
    create_memory_checkpointer,
)
from osprey.utils.config import get_full_configuration
from osprey.utils.logger import get_logger

from .analysis import create_analyzer_node
from .approval import create_approval_node
from .config import PythonExecutorConfig
from .exceptions import CodeRuntimeError
from .execution import create_executor_node
from .generation import create_generator_node
from .models import PythonExecutionRequest, PythonExecutionState, PythonServiceResult

logger = get_logger("python")


class PythonExecutorService:
    """Advanced Python execution service with flexible deployment and human oversight capabilities.

    This service provides a production-ready, LangGraph-based workflow for Python code generation,
    static analysis, human approval, and secure execution. It implements three key innovations
    that make it particularly suitable for high-stakes scientific and industrial environments:

    ## 🎯 **Unique Capabilities**

    ### **1. Flexible Execution Environments**
    Switch between container and local execution with a single configuration change:
    - **Container Mode**: Secure, isolated Jupyter environments with full dependency management
    - **Local Mode**: Direct host execution with automatic Python environment detection
    - **Seamless Switching**: Same interface, same results, different isolation levels

    ### **2. Comprehensive Jupyter Notebook Generation**
    Automatic creation of rich, interactive notebooks for human evaluation:
    - **Multi-Stage Notebooks**: Generated at code creation, analysis, and execution phases
    - **Rich Context**: Complete execution metadata, analysis results, and error diagnostics
    - **Direct Access**: Click-to-open URLs for immediate notebook review in Jupyter
    - **Audit Trails**: Complete history of execution attempts with detailed context

    ### **3. Production-Ready Human-in-the-Loop Approval**
    Sophisticated approval workflows designed for high-stakes environments:
    - **LangGraph-Native Interrupts**: Seamless workflow suspension for human oversight
    - **Rich Approval Context**: Detailed safety assessments, code analysis, and execution plans
    - **Security Integration**: Automatic detection of potentially dangerous operations
    - **Resumable Workflows**: Checkpoint-based execution resumption after approval
    - **Configurable Policies**: Domain-specific approval rules for different operation types

    ## Execution Pipeline

    The service orchestrates a sophisticated multi-stage workflow:

    1. **Code Generation**: LLM-based Python code generation with context awareness and iterative improvement
    2. **Static Analysis**: Security and policy analysis with configurable domain-specific rules
    3. **Approval Workflows**: Human oversight system with rich context and safety assessments
    4. **Flexible Execution**: Container or local execution with unified result collection
    5. **Notebook Generation**: Comprehensive Jupyter notebook creation for human evaluation
    6. **Result Processing**: Structured result handling with artifact management and audit trails

    The service maintains complete compatibility with the existing capability
    interface while providing enhanced functionality through its internal
    LangGraph-based architecture. It supports both fresh execution requests
    and resumption of interrupted workflows (such as approval processes).

    Key architectural features:
        - **Exception-Based Flow Control**: Clean exception handling with categorized
          errors that determine appropriate retry strategies
        - **Checkpoint Support**: Full LangGraph checkpoint integration for workflow
          resumption and debugging
        - **Type-Safe Interfaces**: Pydantic models for request/response with
          comprehensive validation
        - **Service Isolation**: Self-contained service graph separate from the
          main agent workflow
        - **Comprehensive Logging**: Detailed execution tracking and debugging support

    The service integrates with the framework's configuration system, approval
    management, and context handling to provide seamless operation within the
    broader agent framework.

    .. note::
       This service is designed to be invoked through the PythonCapability class
       rather than directly. Direct invocation is supported for advanced use cases.

    .. warning::
       The service can execute arbitrary Python code within configured security
       constraints. Ensure proper approval policies are configured for production use.

    .. seealso::
       :class:`osprey.capabilities.python.PythonCapability` : Main capability interface
       :class:`PythonExecutionRequest` : Request model for service invocation
       :class:`PythonServiceResult` : Structured response from successful execution
       :class:`PythonExecutionState` : Internal LangGraph state management

    Examples:
        **Execution with automatic notebook generation**::

            >>> service = PythonExecutorService()
            >>> request = PythonExecutionRequest(
            ...     user_query="Analyze EPICS PV data and create trend plots",
            ...     task_objective="Generate comprehensive data analysis report",
            ...     execution_folder_name="epics_analysis"
            ... )
            >>> result = await service.ainvoke(request, config=service_config)
            >>>
            >>> # Rich results with notebook access
            >>> print(f"Generated code: {result.generated_code}")
            >>> print(f"Execution time: {result.execution_result.execution_time}s")
            >>> print(f"Review notebook: {result.execution_result.notebook_link}")
            >>> print(f"Generated figures: {len(result.execution_result.figure_paths)}")

        **Container vs Local execution** (same interface, different isolation)::

            >>> # Container execution (config: execution_method: "container")
            >>> result_container = await service.ainvoke(request, config=container_config)
            >>> # Executes in secure Jupyter container
            >>>
            >>> # Local execution (config: execution_method: "local")
            >>> result_local = await service.ainvoke(request, config=local_config)
            >>> # Executes on host Python - same results, faster execution

        **Human-in-the-loop approval workflow**::

            >>> # Request requiring approval automatically triggers interrupt
            >>> request = PythonExecutionRequest(
            ...     user_query="Adjust beam current setpoints",
            ...     task_objective="Optimize accelerator performance",
            ...     execution_folder_name="beam_optimization"
            ... )
            >>> # Service pauses execution, user receives rich approval context:
            >>> # - Generated code in reviewable notebook
            >>> # - Safety analysis and concerns
            >>> # - Execution environment details
            >>> # - Clear approve/reject options
            >>>
            >>> # After user approval, execution resumes automatically
            >>> resume_command = Command(resume={"approved": True})
            >>> result = await service.ainvoke(resume_command, config=service_config)
            >>> print(f"Approved operation completed: {result.execution_result.results}")
    """

    def __init__(self):
        self.config = self._load_config()
        self.executor_config = PythonExecutorConfig(self.config)
        self._compiled_graph = None

    def get_compiled_graph(self):
        """Get the compiled LangGraph for this service."""
        if self._compiled_graph is None:
            self._compiled_graph = self._build_and_compile_graph()
        return self._compiled_graph

    async def ainvoke(self, input_data, config):
        """Main service entry point handling execution requests and workflow resumption.

        This method serves as the primary interface for the Python executor service,
        accepting both fresh execution requests and workflow resumption commands.
        It implements comprehensive input validation, workflow orchestration, and
        structured result processing.

        The method handles two primary input types:

        1. **PythonExecutionRequest**: Fresh execution requests containing user queries,
           task objectives, and execution parameters. These trigger the complete
           code generation, analysis, and execution workflow.

        2. **Command**: Workflow resumption commands, typically containing approval
           responses from interrupted workflows. These resume execution from the
           appropriate checkpoint.

        The service automatically determines the appropriate workflow path based on
        the input type and manages the complete execution lifecycle including error
        handling, result processing, and exception propagation.

        :param input_data: Execution request or resumption command
        :type input_data: Union[PythonExecutionRequest, Command]
        :param config: LangGraph configuration including thread_id and service settings
        :type config: Dict[str, Any]
        :return: Structured execution results for successful completion
        :rtype: PythonServiceResult
        :raises CodeRuntimeError: If Python code execution fails
        :raises TypeError: If input_data is not a supported type
        :raises ValueError: If Command contains invalid resume data

        .. note::
           The service automatically raises appropriate exceptions for execution
           failures rather than returning error states, enabling clean error
           handling in calling code.

        .. warning::
           This method can execute arbitrary Python code. Ensure proper approval
           policies are configured and input validation is performed.

        Examples:
            Processing a fresh execution request::

                >>> service = PythonExecutorService()
                >>> request = PythonExecutionRequest(
                ...     user_query="Calculate data statistics",
                ...     task_objective="Generate summary statistics",
                ...     execution_folder_name="stats_analysis"
                ... )
                >>> config = {"thread_id": "session_123"}
                >>> result = await service.ainvoke(request, config)
                >>> print(f"Success: {result.execution_result.results}")

            Resuming after approval::

                >>> resume_cmd = Command(resume={"approved": True})
                >>> result = await service.ainvoke(resume_cmd, config)
        """

        if isinstance(input_data, Command):
            logger.debug(f"Service ainvoke received input_data type: {type(input_data)}")
            logger.debug(
                f"Service ainvoke input_data isinstance PythonExecutionRequest: {isinstance(input_data, PythonExecutionRequest)}"
            )

            # This is a resume command (approval response)
            if hasattr(input_data, "resume") and input_data.resume:
                logger.info("Resuming Python service execution after approval")
                approval_result = input_data.resume.get("approved", False)
                logger.info(f"Approval result: {approval_result}")
                logger.info(f"Full resume payload keys: {list(input_data.resume.keys())}")

                # Pass Command directly to let LangGraph handle checkpoint resume
                # This preserves the entire approval payload and resumes from the correct checkpoint
                compiled_graph = self.get_compiled_graph()
                result = await compiled_graph.ainvoke(input_data, config)

                # Check for execution failure and raise exception
                self._handle_execution_failure(result)

                return result
            else:
                raise ValueError(
                    "Invalid Command received by service - missing or invalid resume data"
                )

        elif isinstance(input_data, PythonExecutionRequest):
            logger.debug(f"Service ainvoke received input_data type: {type(input_data)}")
            logger.debug(
                f"Service ainvoke input_data isinstance PythonExecutionRequest: {isinstance(input_data, PythonExecutionRequest)}"
            )

            logger.debug("Converting PythonExecutionRequest to internal state")
            internal_state = self._create_internal_state(input_data)
            logger.debug(f"Created internal_state type: {type(internal_state)}")
            logger.debug(f"Internal state keys: {list(internal_state.keys())}")
            logger.debug(f"Internal state has request: {internal_state.get('request') is not None}")

            compiled_graph = self.get_compiled_graph()
            result = await compiled_graph.ainvoke(internal_state, config)

            # Check for execution failure and raise exception
            self._handle_execution_failure(result)

            # Transform to structured result - no more dict validation needed by capabilities!
            return PythonServiceResult(
                execution_result=result["execution_result"],
                generated_code=result.get("generated_code", ""),
                generation_attempt=result.get("generation_attempt", 1),
                analysis_warnings=result.get("analysis_warnings", []),
            )
        else:
            # Clean API: Only accept defined input types
            supported_types = [PythonExecutionRequest.__name__, "Command"]
            raise TypeError(
                f"Python executor service received unsupported input type: {type(input_data).__name__}. "
                f"Supported types: {', '.join(supported_types)}"
            )

    def _handle_execution_failure(self, result: dict) -> None:
        """Check result and raise exception if execution failed.

        :param result: Execution result dictionary to check
        :type result: dict
        :raises CodeRuntimeError: If execution was not successful
        """
        if not result.get("is_successful", False):
            failure_reason = result.get("failure_reason") or result.get(
                "execution_error", "Code execution failed"
            )
            logger.error(f"Python execution failed: {failure_reason}")
            raise CodeRuntimeError(
                message=f"Python code execution failed: {failure_reason}",
                traceback_info=result.get("execution_error", ""),
                execution_attempt=result.get("generation_attempt", 1),
            )

    def _build_and_compile_graph(self):
        """Build and compile the Python executor LangGraph."""

        # Create state graph with PythonExecutionState
        workflow = StateGraph(PythonExecutionState)

        # Add service nodes (no @capability_node decorator!)
        workflow.add_node("python_code_generator", create_generator_node())
        workflow.add_node("python_code_analyzer", create_analyzer_node())
        workflow.add_node("python_approval_node", create_approval_node())
        workflow.add_node("python_code_executor", create_executor_node())

        # Set up internal flow
        workflow.set_entry_point("python_code_generator")
        workflow.add_edge("python_code_generator", "python_code_analyzer")
        workflow.add_conditional_edges(
            "python_code_analyzer",
            self._analyzer_conditional_edge,
            {
                "approve": "python_approval_node",
                "retry": "python_code_generator",
                "execute": "python_code_executor",
                "__end__": "__end__",  # Handle permanent failures
            },
        )
        workflow.add_conditional_edges(
            "python_approval_node",
            self._approval_conditional_edge,
            {
                "approved": "python_code_executor",
                "rejected": "__end__",
                "retry": "python_code_generator",
            },
        )
        workflow.add_conditional_edges(
            "python_code_executor",
            self._executor_conditional_edge,
            {"retry": "python_code_generator", "__end__": "__end__"},
        )

        # Compile with checkpointer for interrupt support - use same pattern as main graph
        checkpointer = self._create_checkpointer()
        compiled = workflow.compile(checkpointer=checkpointer)

        logger.info("Python executor service graph compiled successfully")
        return compiled

    def _create_internal_state(self, request: PythonExecutionRequest) -> PythonExecutionState:
        """Convert PythonExecutionRequest to internal service state.

        This preserves the existing request interface while enabling internal
        LangGraph state management for the service nodes.

        """
        return PythonExecutionState(
            request=request,  # Store serializable request data
            # Extract capability context data to top level for ContextManager compatibility
            capability_context_data=request.capability_context_data,
            # Initialize execution state
            generation_attempt=0,
            error_chain=[],
            current_stage="generation",
            # Approval state
            requires_approval=None,
            approval_interrupt_data=None,
            approval_result=None,
            approved=None,
            # Runtime state
            generated_code=None,
            analysis_result=None,
            analysis_failed=None,
            execution_failed=None,
            execution_result=None,
            execution_folder=None,
            # Control flags
            is_successful=False,
            is_failed=False,
            failure_reason=None,
        )

    def _analyzer_conditional_edge(self, state: PythonExecutionState) -> str:
        """Route after static analysis.

        Pure routing function - no state mutations.
        Retry limit checking is done in the analyzer node itself.
        """
        # Check permanent failure first (set by nodes when retry limit exceeded)
        if state.get("is_failed", False):
            return "__end__"  # Permanently failed - don't retry

        # Route based on analysis results
        elif state.get("analysis_failed", False):
            # Node already checked retry limits and set is_failed if needed
            return "retry"
        elif state.get("requires_approval", False):
            return "approve"
        else:
            return "execute"

    def _approval_conditional_edge(self, state: PythonExecutionState) -> str:
        """Route after approval process."""
        if state.get("approved", False):
            return "approved"
        else:
            return "rejected"

    def _executor_conditional_edge(self, state: PythonExecutionState) -> str:
        """Route after code execution.

        Pure routing function - no state mutations.
        Retry limit checking is done in the executor node itself.
        """
        # Check permanent failure first (set by nodes when retry limit exceeded)
        if state.get("is_failed", False):
            return "__end__"  # Permanently failed - don't retry

        # Route based on execution results
        elif state.get("execution_failed", False):
            # Node already checked retry limits and set is_failed if needed
            return "retry"
        else:
            return "__end__"  # Success - complete the workflow

    def _create_checkpointer(self):
        """Create checkpointer using same logic as main graph."""
        # Check if we should use PostgreSQL (production mode)
        use_postgres = self.config.get("langgraph", {}).get("use_postgres", False)

        if use_postgres:
            try:
                # Try PostgreSQL when explicitly requested
                checkpointer = create_async_postgres_checkpointer()
                logger.info("Python executor service using async PostgreSQL checkpointer")
                return checkpointer
            except Exception as e:
                # Fall back to memory saver if PostgreSQL fails
                logger.warning(f"PostgreSQL checkpointer failed for Python executor service: {e}")
                logger.info("Python executor service falling back to in-memory checkpointer")
                return create_memory_checkpointer()
        else:
            # Default to memory saver for R&D mode
            logger.info("Python executor service using in-memory checkpointer")
            return create_memory_checkpointer()

    def _load_config(self) -> dict[str, Any]:
        """Load service configuration."""
        return get_full_configuration()
