"""XOpt Optimizer Service - LangGraph-based Orchestrator.

This module provides the main service class that orchestrates the XOpt optimization
workflow using LangGraph. It follows the same patterns as PythonExecutorService
for consistency across the Osprey framework.

The service implements a multi-stage workflow:
1. State Identification - Assess machine readiness
2. Decision - Select optimization strategy
3. YAML Generation - Create XOpt configuration
4. Approval - Human approval of configuration
5. Execution - Run XOpt optimization
6. Analysis - Analyze results and decide continuation

The workflow supports iteration loops where analysis can route back to
state identification for multi-iteration optimization campaigns.
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

from .analysis import create_analysis_node
from .approval import create_approval_node
from .decision import create_decision_node
from .exceptions import XOptExecutionError
from .execution import create_executor_node
from .models import (
    XOptExecutionRequest,
    XOptExecutionState,
    XOptServiceResult,
    XOptStrategy,
)
from .state_identification import create_state_identification_node
from .yaml_generation import create_yaml_generation_node

logger = get_logger("xopt_optimizer")


class XOptOptimizerService:
    """XOpt Optimizer Service - LangGraph-based orchestrator.

    Follows the same patterns as PythonExecutorService for consistency
    across the Osprey framework.

    The service provides:
    - Multi-stage optimization workflow with approval gates
    - Iterative optimization with configurable loop control
    - Machine state assessment before optimization
    - Configuration-driven strategy selection
    """

    def __init__(self):
        """Initialize the XOpt optimizer service."""
        self.config = self._load_config()
        self._compiled_graph = None

    def get_compiled_graph(self):
        """Get the compiled LangGraph for this service.

        Lazily compiles the graph on first access.

        Returns:
            CompiledGraph: The compiled LangGraph workflow
        """
        if self._compiled_graph is None:
            self._compiled_graph = self._build_and_compile_graph()
        return self._compiled_graph

    async def ainvoke(self, input_data, config):
        """Main service entry point handling execution requests and workflow resumption.

        This method serves as the primary interface for the XOpt optimizer service,
        accepting both fresh execution requests and workflow resumption commands.

        Args:
            input_data: XOptExecutionRequest for new execution, or Command for resumption
            config: LangGraph configuration including thread_id and service settings

        Returns:
            XOptServiceResult on success

        Raises:
            XOptExecutionError: If optimization fails
            TypeError: If input_data is not a supported type
        """
        if isinstance(input_data, Command):
            # This is a resume command (approval response)
            if hasattr(input_data, "resume") and input_data.resume:
                logger.info("Resuming XOpt service execution after approval")
                approval_result = input_data.resume.get("approved", False)
                logger.info(f"Approval result: {approval_result}")

                # Pass Command directly to let LangGraph handle checkpoint resume
                compiled_graph = self.get_compiled_graph()
                result = await compiled_graph.ainvoke(input_data, config)

                # Check for execution failure and raise exception
                self._handle_execution_failure(result)

                return self._create_service_result(result)
            else:
                raise ValueError(
                    "Invalid Command received by service - missing or invalid resume data"
                )

        elif isinstance(input_data, XOptExecutionRequest):
            logger.debug("Converting XOptExecutionRequest to internal state")
            internal_state = self._create_initial_state(input_data)

            compiled_graph = self.get_compiled_graph()
            result = await compiled_graph.ainvoke(internal_state, config)

            # Check for execution failure and raise exception
            self._handle_execution_failure(result)

            return self._create_service_result(result)

        else:
            supported_types = [XOptExecutionRequest.__name__, "Command"]
            raise TypeError(
                f"XOpt optimizer service received unsupported input type: {type(input_data).__name__}. "
                f"Supported types: {', '.join(supported_types)}"
            )

    def _create_initial_state(self, request: XOptExecutionRequest) -> XOptExecutionState:
        """Convert XOptExecutionRequest to internal service state.

        Initialize ALL state fields to avoid KeyError during execution.

        Args:
            request: The execution request from the capability

        Returns:
            XOptExecutionState: Initialized state for the LangGraph workflow
        """
        return XOptExecutionState(
            # Request (preserved via reducer)
            request=request,
            # Capability context
            capability_context_data=request.capability_context_data,
            # Error tracking
            error_chain=[],
            yaml_generation_attempt=0,
            # Machine state
            machine_state=None,
            machine_state_details=None,
            # Decision
            selected_strategy=None,
            decision_reasoning=None,
            # YAML
            generated_yaml=None,
            yaml_generation_failed=None,
            # Approval
            requires_approval=None,
            approval_interrupt_data=None,
            approval_result=None,
            approved=None,
            # Execution
            run_artifact=None,
            execution_error=None,
            execution_failed=None,
            # Analysis
            analysis_result=None,
            recommendations=None,
            # Loop control
            iteration_count=0,
            max_iterations=request.max_iterations,
            should_continue=False,
            # Control flags
            is_successful=False,
            is_failed=False,
            failure_reason=None,
            current_stage="state_id",
        )

    def _build_and_compile_graph(self):
        """Build and compile the XOpt optimizer LangGraph.

        Creates a StateGraph with all nodes and conditional edges for the
        optimization workflow, then compiles it with checkpointing support.

        Returns:
            CompiledGraph: The compiled workflow graph
        """
        workflow = StateGraph(XOptExecutionState)

        # Add nodes
        workflow.add_node("state_identification", create_state_identification_node())
        workflow.add_node("decision", create_decision_node())
        workflow.add_node("yaml_generation", create_yaml_generation_node())
        workflow.add_node("approval", create_approval_node())
        workflow.add_node("execution", create_executor_node())
        workflow.add_node("analysis", create_analysis_node())

        # Define flow
        workflow.set_entry_point("state_identification")
        workflow.add_edge("state_identification", "decision")

        workflow.add_conditional_edges(
            "decision",
            self._decision_router,
            {"continue": "yaml_generation", "abort": "__end__"},
        )

        workflow.add_conditional_edges(
            "yaml_generation",
            self._yaml_generation_router,
            {
                "approve": "approval",
                "execute": "execution",
                "retry": "yaml_generation",
                "__end__": "__end__",
            },
        )

        workflow.add_conditional_edges(
            "approval",
            self._approval_router,
            {"approved": "execution", "rejected": "__end__"},
        )

        workflow.add_edge("execution", "analysis")

        workflow.add_conditional_edges(
            "analysis",
            self._loop_router,
            {"continue": "state_identification", "complete": "__end__"},
        )

        # Compile with checkpointer for interrupt support
        checkpointer = self._create_checkpointer()
        compiled = workflow.compile(checkpointer=checkpointer)

        logger.info("XOpt optimizer service graph compiled successfully")
        return compiled

    def _decision_router(self, state: XOptExecutionState) -> str:
        """Route after machine state decision.

        Args:
            state: Current execution state

        Returns:
            str: "abort" if failed or abort strategy, "continue" otherwise
        """
        if state.get("is_failed"):
            return "abort"
        if state.get("selected_strategy") == XOptStrategy.ABORT:
            return "abort"
        return "continue"

    def _yaml_generation_router(self, state: XOptExecutionState) -> str:
        """Route after YAML generation.

        Args:
            state: Current execution state

        Returns:
            str: Routing decision based on generation result
        """
        if state.get("is_failed"):
            return "__end__"
        if state.get("yaml_generation_failed"):
            return "retry"
        if state.get("requires_approval"):
            return "approve"
        return "execute"

    def _approval_router(self, state: XOptExecutionState) -> str:
        """Route after approval process.

        Args:
            state: Current execution state

        Returns:
            str: "approved" if approved, "rejected" otherwise
        """
        return "approved" if state.get("approved") else "rejected"

    def _loop_router(self, state: XOptExecutionState) -> str:
        """Route after analysis - continue loop or complete.

        Args:
            state: Current execution state

        Returns:
            str: "continue" to loop back, "complete" to end
        """
        if state.get("is_failed"):
            return "complete"
        return "continue" if state.get("should_continue") else "complete"

    def _handle_execution_failure(self, result: dict) -> None:
        """Check result and raise exception if execution failed.

        Args:
            result: Final state from graph execution

        Raises:
            XOptExecutionError: If optimization failed
        """
        if not result.get("is_successful", False) and result.get("is_failed", False):
            failure_reason = result.get("failure_reason", "XOpt optimization failed")
            logger.error(f"XOpt execution failed: {failure_reason}")
            raise XOptExecutionError(
                message=f"XOpt optimization failed: {failure_reason}",
                xopt_error=result.get("execution_error"),
            )

    def _create_service_result(self, result: dict) -> XOptServiceResult:
        """Create structured service result from final state.

        Args:
            result: Final state from graph execution

        Returns:
            XOptServiceResult: Structured result for capability consumption
        """
        recommendations = result.get("recommendations") or []
        return XOptServiceResult(
            run_artifact=result.get("run_artifact", {}),
            generated_yaml=result.get("generated_yaml", ""),
            strategy=result.get("selected_strategy", XOptStrategy.EXPLORATION),
            total_iterations=result.get("iteration_count", 0),
            analysis_summary=result.get("analysis_result", {}),
            recommendations=tuple(recommendations),  # Convert to tuple for frozen dataclass
        )

    def _create_checkpointer(self):
        """Create checkpointer using same logic as main graph.

        Returns:
            Checkpointer: PostgreSQL or in-memory checkpointer
        """
        # Check if we should use PostgreSQL (production mode)
        use_postgres = self.config.get("langgraph", {}).get("use_postgres", False)

        if use_postgres:
            try:
                # Try PostgreSQL when explicitly requested
                checkpointer = create_async_postgres_checkpointer()
                logger.info("XOpt optimizer service using async PostgreSQL checkpointer")
                return checkpointer
            except Exception as e:
                # Fall back to memory saver if PostgreSQL fails
                logger.warning(f"PostgreSQL checkpointer failed for XOpt optimizer service: {e}")
                logger.info("XOpt optimizer service falling back to in-memory checkpointer")
                return create_memory_checkpointer()
        else:
            # Default to memory saver for R&D mode
            logger.info("XOpt optimizer service using in-memory checkpointer")
            return create_memory_checkpointer()

    def _load_config(self) -> dict[str, Any]:
        """Load service configuration.

        Returns:
            dict: Full configuration dictionary
        """
        return get_full_configuration()
