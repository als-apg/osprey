"""Base Infrastructure Classes - System Component Architecture

This module provides the BaseInfrastructureNode class and supporting components
for building infrastructure nodes in the Osprey Framework. Infrastructure
nodes handle system-level operations like orchestration, classification, routing,
and monitoring that are essential for agent operation but distinct from business
logic capabilities.

The infrastructure architecture emphasizes simplicity, reliability, and fast
failure detection. Infrastructure nodes are designed to fail fast with clear
error messages rather than attempting complex recovery strategies, since they
handle system-critical functions that require immediate attention.

Key Infrastructure Components:
    - Task extraction and processing
    - Request classification and routing
    - Execution orchestration and planning
    - Error handling and recovery coordination
    - State management and validation
    - Monitoring and performance tracking

The infrastructure pattern follows these principles:
1. **Convention-based validation**: Required components enforced through reflection
2. **LangGraph-native integration**: Full streaming, configuration, and checkpoint support
3. **Fast failure detection**: Conservative error handling with immediate failure
4. **State management**: Pure dictionary operations for LangGraph compatibility
5. **Execution tracking**: Comprehensive timing and performance monitoring

.. note::
   Infrastructure nodes use the @infrastructure_node decorator for LangGraph
   integration and should focus on orchestration rather than business logic.

.. warning::
   Infrastructure nodes use conservative retry policies since they handle
   system-critical functions. Most errors are treated as critical by default.

.. seealso::
   :func:`infrastructure_node` : Decorator for LangGraph integration
   :class:`BaseCapability` : Business logic component base class
   :mod:`osprey.state` : State management and structure definitions
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from osprey.base.errors import ErrorClassification, ErrorSeverity

# Import types for type hints
if TYPE_CHECKING:
    from osprey.state import AgentState


class BaseInfrastructureNode(ABC):
    """Base class for infrastructure nodes in the LangGraph-native architecture.

    This class provides the foundation for all infrastructure components in the
    Osprey Framework. Infrastructure nodes handle system-level operations
    that orchestrate, route, classify, and monitor agent execution. Unlike
    capabilities which contain business logic, infrastructure nodes focus on
    system coordination and management.

    The BaseInfrastructureNode class enforces a strict contract through
    reflection-based validation and provides standardized integration with
    LangGraph's execution model. Infrastructure nodes are designed for fast
    failure detection and minimal retry attempts since they handle system-critical
    functions.

    Infrastructure Node Responsibilities:

    - **Task Extraction**: Parse and structure user requests into actionable tasks
    - **Classification**: Determine which capabilities should handle specific requests
    - **Orchestration**: Plan and coordinate execution sequences across capabilities
    - **Routing**: Direct execution flow based on state conditions and results
    - **Monitoring**: Track execution progress and system health
    - **Error Coordination**: Handle system-level error recovery and routing

    Required Components (enforced at decoration time):

    - name: Infrastructure node identifier for routing and logging
    - description: Human-readable description for documentation
    - execute(): Async static method containing orchestration logic
    - classify_error(): Error classification method (inherited or custom)
    - get_retry_policy(): Retry configuration method (inherited or custom)

    Architecture Integration:

    Infrastructure nodes integrate with the framework through:

    1. **LangGraph Integration**: Via @infrastructure_node decorator
    2. **State Management**: Pure dictionary operations for serialization
    3. **Error Handling**: Conservative policies with fast failure detection
    4. **Streaming**: Native LangGraph streaming for real-time updates
    5. **Configuration**: Access to LangGraph's configuration system

    Example::

        @infrastructure_node
        class TaskExtractionNode(BaseInfrastructureNode):
            name = "task_extraction"
            description = "Task Extraction and Processing"

            @staticmethod
            async def execute(state: AgentState, **kwargs) -> Dict[str, Any]:
                # Explicit logger retrieval - professional practice
                from osprey.utils.logger import get_logger
                logger = get_logger("task_extraction")

                # Use get_stream_writer() for pure LangGraph streaming
                from langgraph.config import get_stream_writer
                streaming = get_stream_writer()

                if streaming:
                    streaming({"event_type": "status", "message": "Extracting task", "progress": 0.3})

                logger.info("Starting task extraction")

                # Extract and process task from flat state structure
                task = state.get("task_current_task", "")

                if streaming:
                    streaming({"event_type": "status", "message": "Task extraction complete", "progress": 1.0, "complete": True})

                # Return state updates for flat structure
                return {
                    "task_current_task": task,
                    "task_depends_on_chat_history": True,
                    "task_depends_on_user_memory": False
                }

            @staticmethod
            def classify_error(exc: Exception, context: dict) -> ErrorClassification:
                # Optional error classification
                return ErrorClassification(severity=ErrorSeverity.RETRIABLE, ...)


    :param name: Infrastructure node identifier (required class attribute)
    :type name: str
    :param description: Human-readable description (required class attribute)
    :type description: str

    .. note::
       Infrastructure nodes use the @infrastructure_node decorator which handles
       all LangGraph integration, parameter injection, and error handling.

    .. warning::
       The name and description class attributes are required. The execute method
       must be implemented as a static method and should return state updates.
    """

    # Required class attributes - must be overridden in subclasses
    name: str = None
    description: str = None

    @staticmethod
    @abstractmethod
    async def execute(
        state: 'AgentState',
        **kwargs
    ) -> dict[str, Any]:
        """Execute the infrastructure operation with comprehensive system coordination.

        This is the core method that all infrastructure nodes must implement.
        It contains the orchestration, routing, or monitoring logic and integrates
        with the framework's state management system. The method should be
        implemented as a static method to support LangGraph's execution model.

        Infrastructure nodes should focus on system coordination rather than
        business logic. They receive the complete agent state and return updates
        that LangGraph automatically merges. The @infrastructure_node decorator
        provides timing, error handling, and execution tracking.

        Common Infrastructure Patterns:
        1. **Task Extraction**: Parse user input into structured task information
        2. **Classification**: Analyze requests to determine capability routing
        3. **Orchestration**: Create execution plans with capability sequences
        4. **Routing**: Direct flow based on state conditions and results
        5. **Monitoring**: Track progress and system health metrics

        :param state: Current agent state containing all execution context and data
        :type state: AgentState
        :param kwargs: Additional parameters including logger and configuration
        :type kwargs: dict
        :return: Dictionary of state updates for LangGraph to merge into agent state
        :rtype: Dict[str, Any]

        :raises NotImplementedError: This is an abstract method that must be implemented
        :raises ValidationError: If required state data is missing or invalid
        :raises InfrastructureError: For infrastructure-specific operation failures

        Example::

            async def execute(self) -> Dict[str, Any]:
                # Get unified logger with automatic streaming
                logger = self.get_logger()
                logger.status("Starting orchestration")

                # Infrastructure logic
                current_task = self.get_current_task()
                plan = create_execution_plan(current_task)

                logger.success("Execution plan created")

                # Return state updates
                return {
                    "planning_execution_plan": plan,
                    "planning_ready_for_execution": True
                }

        .. note::
           Infrastructure nodes should focus on orchestration, routing, and state
           management logic. Use self.get_logger() for unified logging with automatic
           streaming support. The @infrastructure_node decorator handles timing,
           error handling, and retry policies.
        """
        pass

    def get_logger(self):
        """Get unified logger with automatic streaming support.

        Creates a logger that:
        - Uses this infrastructure node's name automatically
        - Has access to state for streaming via self._state
        - Streams high-level messages automatically when in LangGraph context
        - Logs to CLI with Rich formatting

        Returns:
            ComponentLogger instance with streaming capability
        """
        from osprey.utils.logger import get_logger
        return get_logger(self.name, state=self._state)

    @staticmethod
    def classify_error(exc: Exception, context: dict) -> 'ErrorClassification':
        """Classify errors for infrastructure-specific error handling and recovery.

        This method provides default error classification for all infrastructure
        nodes with a conservative approach that treats most errors as critical.
        Infrastructure nodes handle system-critical functions like orchestration
        and routing, so failures typically require immediate attention rather than
        automatic retry attempts.

        The default implementation prioritizes system stability by failing fast
        with clear error messages. Subclasses should override this method only
        when specific infrastructure components can benefit from retry logic
        (e.g., LLM-based orchestrators that may encounter temporary API issues).

        :param exc: The exception that occurred during infrastructure operation
        :type exc: Exception
        :param context: Error context including node info, execution state, and timing
        :type context: dict
        :return: Error classification with severity and recovery strategy
        :rtype: ErrorClassification

        .. note::
           The context dictionary includes:

           - ``infrastructure_node``: node name for identification
           - ``execution_time``: time spent before failure
           - ``current_state``: agent state at time of error

        Example::

            @staticmethod
            def classify_error(exc: Exception, context: dict) -> ErrorClassification:
                # Retry network timeouts for LLM-based infrastructure
                if isinstance(exc, (ConnectionError, TimeoutError)):
                    return ErrorClassification(
                        severity=ErrorSeverity.RETRIABLE,
                        user_message="Network timeout, retrying...",
                        metadata={"technical_details": str(exc)}
                    )
                return ErrorClassification(
                    severity=ErrorSeverity.CRITICAL,
                    user_message=f"Infrastructure error: {exc}",
                    metadata={"technical_details": str(exc)}
                )

        .. note::
           Infrastructure nodes should generally fail fast, so the default
           implementation treats most errors as critical. Override this method
           for infrastructure that can benefit from retries (e.g., LLM-based nodes).
        """
        node_name = context.get('infrastructure_node', 'unknown_infrastructure_node')
        return ErrorClassification(
            severity=ErrorSeverity.CRITICAL,
            user_message=f"Infrastructure error in {node_name}: {exc}",
            metadata={"technical_details": str(exc)}
        )

    @staticmethod
    def get_retry_policy() -> dict[str, Any]:
        """Get conservative retry policy configuration for infrastructure operations.

        This method provides retry configuration optimized for infrastructure
        nodes that handle system-critical functions. The default policy uses
        conservative settings with minimal retry attempts and fast failure
        detection to maintain system stability.

        Infrastructure nodes should generally fail fast rather than retry
        extensively, since failures often indicate system-level issues that
        require immediate attention. Override this method only for specific
        infrastructure components that can benefit from retry logic.

        :return: Dictionary containing conservative retry configuration parameters
        :rtype: Dict[str, Any]

        .. note::
           Infrastructure default policy: 2 attempts, 0.2s delay, minimal backoff.
           This prioritizes fast failure detection over retry persistence.

        Example::

            @staticmethod
            def get_retry_policy() -> Dict[str, Any]:
                return {
                    "max_attempts": 3,  # More retries for LLM-based infrastructure
                    "delay_seconds": 1.0,  # Longer delay for external service calls
                    "backoff_factor": 2.0  # Exponential backoff
                }

        .. note::
           The router uses this configuration to determine retry behavior.
           Infrastructure default: 2 attempts, 0.2s delay, minimal backoff.

        """
        return {
            "max_attempts": 2,  # Conservative for infrastructure
            "delay_seconds": 0.2,  # Fast retry for infrastructure
            "backoff_factor": 1.2  # Minimal backoff
        }

    # ===== STATE HELPER METHODS =====
    # These helper methods provide convenient access to StateManager utilities
    # using self._state (which is injected by the @infrastructure_node decorator)

    def get_current_task(self) -> str | None:
        """Get current task from state.

        Returns:
            Current task string, or None if not set

        Example:
            ```python
            async def execute(self) -> dict[str, Any]:
                current_task = self.get_current_task()
                if not current_task:
                    raise ValueError("No current task available")
            ```
        """
        from osprey.state import StateManager
        return StateManager.get_current_task(self._state)

    def get_user_query(self) -> str | None:
        """Get the user's query from the current conversation.

        Returns:
            The user's query string, or None if no user messages exist

        Example:
            ```python
            async def execute(self) -> dict[str, Any]:
                original_query = self.get_user_query()
            ```
        """
        from osprey.state import StateManager
        return StateManager.get_user_query(self._state)

    def get_execution_plan(self):
        """Get current execution plan from state with type validation.

        Returns:
            ExecutionPlan if available and valid, None otherwise

        Example:
            ```python
            async def execute(self) -> dict[str, Any]:
                execution_plan = self.get_execution_plan()
                if not execution_plan:
                    # Route to orchestrator
            ```
        """
        from osprey.state import StateManager
        return StateManager.get_execution_plan(self._state)

    def get_current_step_index(self) -> int:
        """Get current step index from state.

        Returns:
            Current step index (defaults to 0 if not set)

        Example:
            ```python
            async def execute(self) -> dict[str, Any]:
                current_index = self.get_current_step_index()
            ```
        """
        from osprey.state import StateManager
        return StateManager.get_current_step_index(self._state)

    def get_current_step(self):
        """Get current execution step from state.

        Returns:
            PlannedStep: Current step dictionary with capability, task_objective, etc.

        Raises:
            RuntimeError: If execution plan is missing or step index is invalid

        Example:
            ```python
            async def execute(self) -> dict[str, Any]:
                step = self.get_current_step()
                task_objective = step.get('task_objective')
            ```
        """
        from osprey.state import StateManager
        return StateManager.get_current_step(self._state)

    def __repr__(self) -> str:
        """Return a string representation of the infrastructure node for debugging.

        Provides a concise string representation that includes both the Python
        class name and the infrastructure node's registered name. This is useful
        for debugging, logging, and development workflows where infrastructure
        nodes need to be identified clearly.

        :return: String representation including class name and node name
        :rtype: str

        Example:
            >>> node = TaskExtractionNode()
            >>> repr(node)
            '<TaskExtractionNode: task_extraction>'

        .. note::
           The format follows the pattern '<ClassName: node_name>' for
           consistency across all framework components.
        """
        return f"<{self.__class__.__name__}: {self.name}>"
