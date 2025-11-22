"""LangGraph Integration Decorators - Infrastructure Injection System

This module provides the core decorator system that enables seamless LangGraph
integration for capabilities and infrastructure nodes. The decorators implement
reflection-based validation, automatic infrastructure injection, and standardized
execution patterns that eliminate boilerplate while ensuring consistent behavior
across all framework components.

The decorator system serves as the bridge between the framework's convention-based
architecture and LangGraph's execution model. It provides comprehensive error
handling, state management, execution tracking, and streaming support while
maintaining type safety and development-time validation.

Key Features:
    - **Reflection-based validation**: Ensures required components are implemented
    - **Automatic infrastructure injection**: Provides timing, logging, and error handling
    - **LangGraph-native integration**: Full streaming, configuration, and checkpoint support
    - **Manual retry coordination**: Consistent error classification and retry policies
    - **Development mode support**: Raw error re-raising for debugging
    - **Execution tracking**: Comprehensive performance and state monitoring

Decorator Architecture:
    1. **@capability_node**: Business logic components with comprehensive execution tracking
    2. **@infrastructure_node**: System components with fast failure detection
    3. **Validation patterns**: Reflection-based requirement checking at decoration time
    4. **Error coordination**: Manual retry system via router for consistent behavior
    5. **State management**: Pure dictionary operations for LangGraph compatibility

.. note::
   These decorators create LangGraph-compatible node functions while preserving
   the original classes for introspection and testing. The framework uses manual
   retry handling rather than LangGraph's native retry policies for consistency.

.. warning::
   Decorated classes must implement required components (name, description, execute)
   or decoration will fail with clear error messages during development.

.. seealso::
   :class:`BaseCapability` : Capability base class with decorator integration
   :class:`BaseInfrastructureNode` : Infrastructure node base class
   :mod:`osprey.state` : State management and execution tracking
"""

import inspect
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

from osprey.base.errors import ErrorSeverity
from osprey.utils.logger import get_logger

try:
    from langgraph.config import get_config, get_stream_writer
except ImportError:
    get_stream_writer = None
    get_config = None

# Import types for type hints
if TYPE_CHECKING:
    from osprey.state import AgentState

# Lazy imports to avoid circular dependencies
def _import_error_classification():
    try:
        from osprey.base.errors import ErrorClassification, ErrorSeverity
        return ErrorClassification, ErrorSeverity
    except ImportError:
        return None, None

def _is_graph_interrupt(exc: Exception) -> bool:
    """Check if exception is a LangGraph GraphInterrupt."""
    # Check class name to avoid import issues
    return exc.__class__.__name__ == 'GraphInterrupt'


# =============================================================================
# LANGGRAPH INTEGRATION DECORATORS
# =============================================================================




def capability_node(cls):
    """Decorator that validates capability conventions and injects comprehensive LangGraph infrastructure.

    This decorator serves as the primary integration point between capability classes
    and LangGraph's execution model. It performs reflection-based validation to ensure
    capability classes implement required components, then creates a LangGraph-compatible
    node function with complete infrastructure including error handling, retry coordination,
    execution tracking, and state management.

    The decorator implements the framework's convention-based architecture by:
    1. **Validation**: Ensures all required components are properly implemented
    2. **Infrastructure Injection**: Provides timing, logging, streaming, and error handling
    3. **LangGraph Integration**: Creates compatible node functions with state management
    4. **Error Coordination**: Routes all errors through manual retry system for consistency
    5. **Execution Tracking**: Comprehensive performance monitoring and state updates

    Required Components (validated through reflection):
        - name: Unique capability identifier for registry and routing
        - description: Human-readable description for documentation and logging
        - execute(): Async static method containing the main business logic
        - classify_error(): Error classification method (inherited from BaseCapability or custom)
        - get_retry_policy(): Retry configuration method (inherited from BaseCapability or custom)

    Infrastructure Features:
        - **Error Classification**: Domain-specific error analysis with recovery strategies
        - **Manual Retry System**: Consistent retry handling via router (no LangGraph retries)
        - **State Management**: Automatic state updates and step progression tracking
        - **Streaming Support**: Real-time status updates through LangGraph's streaming
        - **Development Mode**: Raw error re-raising for debugging when configured
        - **Execution Tracking**: Comprehensive timing and performance monitoring

    :param cls: The capability class to decorate with LangGraph infrastructure
    :type cls: type
    :return: Original class enhanced with langgraph_node attribute containing the LangGraph function
    :rtype: type
    :raises ValueError: If required class attributes (name, description) are missing
    :raises ValueError: If required methods (execute, classify_error, get_retry_policy) are missing

    .. note::
       The decorator creates a `langgraph_node` attribute on the class containing
       the LangGraph-compatible function. The original class remains unchanged for
       introspection and testing purposes.

    .. warning::
       All capability errors are routed through the manual retry system rather than
       using LangGraph's native retry policies to ensure consistent behavior across
       all framework components.

    Examples:
        Basic capability decoration::

            @capability_node
            class WeatherCapability(BaseCapability):
                name = "weather_data"
                description = "Retrieve current weather conditions"

                @staticmethod
                async def execute(state: AgentState, **kwargs) -> Dict[str, Any]:
                    # Business logic implementation
                    return {"weather_data": weather_info}

        Capability with custom error handling::

            @capability_node
            class DatabaseCapability(BaseCapability):
                name = "database_query"
                description = "Execute database queries"

                @staticmethod
                def classify_error(exc: Exception, context: dict) -> ErrorClassification:
                    if isinstance(exc, ConnectionError):
                        return ErrorClassification(severity=ErrorSeverity.RETRIABLE, ...)
                    return ErrorClassification(severity=ErrorSeverity.CRITICAL, ...)

                @staticmethod
                async def execute(state: AgentState, **kwargs) -> Dict[str, Any]:
                    # Database operation implementation
                    return {"query_results": results}

    .. seealso::
       :class:`BaseCapability` : Base class with required method implementations
       :func:`infrastructure_node` : Decorator for infrastructure components
       :class:`ErrorClassification` : Error classification system
    """

    # Extract required components using reflection
    capability_name = getattr(cls, 'name', None)
    description = getattr(cls, 'description', None)
    execute_func = getattr(cls, 'execute', None)
    error_classifier = getattr(cls, 'classify_error', None)
    retry_policy_func = getattr(cls, 'get_retry_policy', None)

    logger = get_logger(capability_name)

    # Validate required components
    if not capability_name:
        raise ValueError(f"Capability {cls.__name__} must define 'name' class attribute")
    if not description:
        raise ValueError(f"Capability class {cls.__name__} must define 'description' attribute")
    if not execute_func:
        raise ValueError(f"Capability class {cls.__name__} must define 'execute' method (static or instance)")
    if not error_classifier:
        raise ValueError(f"Capability class {cls.__name__} must have 'classify_error' method (inherit from BaseCapability or define manually)")
    if not retry_policy_func:
        raise ValueError(f"Capability class {cls.__name__} must have 'get_retry_policy' method (inherit from BaseCapability or define manually)")

    # Detect method type: static (legacy) or instance (new pattern)
    is_static = isinstance(inspect.getattr_static(cls, 'execute'), staticmethod)

    # Validate it's either static or regular method (not classmethod or property)
    if not is_static:
        execute_attr = inspect.getattr_static(cls, 'execute')
        if isinstance(execute_attr, classmethod):
            raise ValueError(
                f"Capability {cls.__name__}.execute() cannot be a classmethod. "
                f"Use either @staticmethod (legacy) or instance method (recommended)."
            )
        if isinstance(execute_attr, property):
            raise ValueError(
                f"Capability {cls.__name__}.execute() cannot be a property. "
                f"Must be a method."
            )
        if not callable(execute_attr):
            raise ValueError(
                f"Capability {cls.__name__}.execute must be callable."
            )

    # Create LangGraph-compatible node function
    async def langgraph_node(
        state: 'AgentState',
        **kwargs
    ) -> dict[str, Any]:
        """LangGraph-native node function with manual retry handling via router."""

        # Get streaming capability for status updates
        streaming = get_stream_writer() if get_stream_writer else None

        # Extract current step information using StateManager (lazy import to avoid circular imports)
        from osprey.state import StateManager
        step = StateManager.get_current_step(state)
        start_time = time.time()

        try:
            if streaming:
                streaming({"event_type": "status", "message": f"Executing {capability_name}...", "progress": 0.1})

            logger.info(f"Executing capability: {capability_name}")

            # Execute based on method type
            if is_static:
                # OLD: Static method (backward compatibility)
                # NOTE: Old static methods had **kwargs in signature but never used it
                result = await execute_func(state)
            else:
                # NEW: Instance method (recommended - unified interface)
                # Create instance WITHOUT state (registry-compatible!)
                instance = cls()

                # Inject state and step BEFORE calling execute
                instance._state = state
                instance._step = step

                # Now execute() can use self._state and self._step
                # NO kwargs - unified interface!
                result = await instance.execute()

            execution_time = time.time() - start_time

            # Handle state updates for step progression
            state_updates = _handle_capability_state_updates(
                state, result, step, capability_name, start_time, execution_time, logger
            )

            return state_updates

        except Exception as exc:
            execution_time = time.time() - start_time

            # Check for development mode - re-raise original exception for debugging
            try:
                if get_config:
                    config = get_config()
                    configurable = config.get("configurable", {})
                    if configurable.get('development', {}).get('raise_raw_errors', False):
                        logger.error("Development mode: Re-raising original exception directly for debugging")
                        raise exc
            except (RuntimeError, ImportError, AttributeError, KeyError):
                # If config access fails (outside runnable context), continue with normal error handling
                pass

            # Re-raise GraphInterrupt immediately - it's not an error!
            if _is_graph_interrupt(exc):
                logger.info(f"GraphInterrupt detected in {capability_name} - re-raising for LangGraph to handle")
                raise exc

            # Classify the error using domain-specific or default logic
            error_context = {
                "capability": capability_name,
                "current_step_index": StateManager.get_current_step_index(state),
                "execution_time": execution_time,
                "current_state": state
            }
            error_classification = error_classifier(exc, error_context)

            # Get retry policy for this capability
            retry_policy = retry_policy_func()

            logger.error(f"Error in {capability_name} after {execution_time:.2f}s: {str(exc)}")
            logger.error(f"Classification: {error_classification.user_message or str(exc)}")

            # Always use manual retry system via router - NO LangGraph retries

            # Track failure in execution_step_results for consistent tracking
            step = StateManager.get_current_step(state)
            step_results = state.get("execution_step_results", {}).copy()
            current_step_index = StateManager.get_current_step_index(state)
            step_key = step.get("context_key", f"{current_step_index}_{capability_name}")
            step_results[step_key] = {
                "step_index": current_step_index,  # For explicit ordering
                "capability": capability_name,
                "task_objective": step.get("task_objective", f"Execute {capability_name}"),  # Add step objective
                "success": False,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }

            return {
                "control_has_error": True,
                "control_error_info": {
                    "capability_name": capability_name,
                    "classification": error_classification,
                    "retry_policy": retry_policy,
                    "original_error": str(exc),
                    "user_message": error_classification.user_message or str(exc),
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat()
                },
                "control_retry_count": state.get("control_retry_count", 0) + 1,
                "execution_step_results": step_results,
                "execution_last_result": {
                    "capability": capability_name,
                    "success": False,
                    "error": str(exc),
                    "classification": error_classification.severity.value if hasattr(error_classification.severity, 'value') else str(error_classification.severity),
                    "timestamp": datetime.now().isoformat()
                }
            }

    # Attach metadata to the function for LangGraph integration
    langgraph_node.name = capability_name
    langgraph_node.capability_name = capability_name
    langgraph_node.description = description
    langgraph_node.error_classifier = error_classifier
    langgraph_node.original_class = cls

    # Set the LangGraph-native function on the class for registry discovery
    cls.langgraph_node = langgraph_node

    return cls


def _handle_capability_state_updates(
    state: dict[str, Any],
    result: dict[str, Any],
    step: dict[str, Any],
    capability_name: str,
    start_time: float,
    execution_time: float,
    logger
) -> dict[str, Any]:
    """Handle comprehensive state updates for capability execution."""

    # Lazy import to avoid circular imports
    from osprey.state import StateManager

    # Start with the capability's result (now includes capability_context_data instead of execution_context)
    state_updates = result.copy() if isinstance(result, dict) else {}

    # Step progression - advance to next step after successful execution
    current_step_index = StateManager.get_current_step_index(state)
    state_updates["planning_current_step_index"] = current_step_index + 1

    # Control flow updates
    state_updates["control_current_step_retry_count"] = 0  # Reset retry count


    # Clear retry state when capability succeeds
    state_updates["control_has_error"] = False
    state_updates["control_retry_count"] = 0
    state_updates["control_error_info"] = None

    # Store step results with step information
    step_results = state.get("execution_step_results", {}).copy()
    step_key = step.get("context_key",f"{current_step_index}_{capability_name}")
    step_results[step_key] = {
        "step_index": current_step_index,  # For explicit ordering
        "capability": capability_name,
        "task_objective": step.get("task_objective", f"Execute {capability_name}"),  # Add step objective
        "success": True,
        "execution_time": execution_time,
        "timestamp": datetime.now().isoformat()
    }
    state_updates["execution_step_results"] = step_results

    # Update last result for router decision-making
    state_updates["execution_last_result"] = {
        "capability": capability_name,
        "success": True,
        "execution_time": execution_time,
        "timestamp": datetime.now().isoformat()
    }

    logger.debug(f"State updates: step {current_step_index + 1}")

    return state_updates




def infrastructure_node(cls=None, *, quiet=False):
    """Decorator that validates infrastructure node conventions and injects comprehensive LangGraph infrastructure.

    This decorator serves as the primary integration point between infrastructure node
    classes and LangGraph's execution model. It performs reflection-based validation
    to ensure infrastructure classes implement required components, then creates a
    LangGraph-compatible node function with complete system infrastructure including
    error handling, performance monitoring, and state coordination.

    Infrastructure nodes handle system-critical operations like orchestration, routing,
    classification, and monitoring. The decorator emphasizes fast failure detection
    and conservative error handling since infrastructure failures typically indicate
    system-level issues requiring immediate attention.

    The decorator implements comprehensive system coordination by:
    1. **Validation**: Ensures all required components are properly implemented
    2. **Infrastructure Injection**: Provides timing, logging, streaming, and error handling
    3. **LangGraph Integration**: Creates compatible node functions with native features
    4. **Error Coordination**: Manual retry system with conservative failure policies
    5. **System Monitoring**: Performance tracking and infrastructure health monitoring

    Required Components (validated through reflection):
        - name: Infrastructure node identifier for routing and logging
        - description: Human-readable description for documentation and monitoring
        - execute(): Async static method containing orchestration/routing logic
        - classify_error(): Error classification method (inherited or custom)
        - get_retry_policy(): Retry configuration method (inherited or custom)

    Infrastructure Features:
        - **Conservative Error Handling**: Fast failure detection with minimal retry attempts
        - **System Monitoring**: Comprehensive timing and performance tracking
        - **LangGraph Native Integration**: Full streaming, configuration, and checkpoint support
        - **Development Mode Support**: Raw error re-raising for debugging when configured
        - **Optional Quiet Mode**: Suppressed logging for high-frequency routing operations
        - **Fatal Error Handling**: System-level failure detection with immediate termination

    :param cls: The infrastructure node class to decorate (None for parameterized usage)
    :type cls: Optional[type]
    :param quiet: If True, suppress start/completion logging (useful for routing nodes)
    :type quiet: bool
    :return: Enhanced infrastructure class with langgraph_node attribute
    :rtype: type
    :raises ValueError: If required class attributes (name, description) are missing
    :raises ValueError: If required methods (execute, classify_error, get_retry_policy) are missing

    .. note::
       The decorator supports both @infrastructure_node and @infrastructure_node(quiet=True)
       syntax. The quiet parameter is useful for high-frequency routing operations that
       would otherwise generate excessive logging.

    .. warning::
       Infrastructure nodes use conservative retry policies with fast failure detection.
       FATAL errors immediately terminate execution to prevent system-level issues.

    Examples:
        Basic infrastructure node::

            @infrastructure_node
            class TaskExtractionNode(BaseInfrastructureNode):
                name = "task_extraction"
                description = "Extract and structure user tasks"

                @staticmethod
                async def execute(state: AgentState, **kwargs) -> Dict[str, Any]:
                    # Task extraction logic
                    return {"task_current_task": extracted_task}

        Quiet routing node::

            @infrastructure_node(quiet=True)
            class RouterNode(BaseInfrastructureNode):
                name = "router"
                description = "Dynamic routing based on agent state"

                @staticmethod
                async def execute(state: AgentState, **kwargs) -> Dict[str, Any]:
                    # Routing logic without verbose logging
                    return {"control_next_node": next_node}

        Infrastructure node with custom error handling::

            @infrastructure_node
            class OrchestratorNode(BaseInfrastructureNode):
                name = "orchestrator"
                description = "Create execution plans"

                @staticmethod
                def classify_error(exc: Exception, context: dict) -> ErrorClassification:
                    # Retry LLM timeouts for planning operations
                    if isinstance(exc, TimeoutError):
                        return ErrorClassification(severity=ErrorSeverity.RETRIABLE, ...)
                    return ErrorClassification(severity=ErrorSeverity.CRITICAL, ...)

    .. seealso::
       :class:`BaseInfrastructureNode` : Base class with required method implementations
       :func:`capability_node` : Decorator for business logic components
       :class:`ErrorSeverity` : Error severity levels and recovery strategies

    Example::

        @infrastructure_node  # Validates requirements and injects infrastructure!
        class TaskExtractionNode(BaseInfrastructureNode):
            name = "task_extraction"
            description = "Task Extraction and Processing"

            @staticmethod
            async def execute(state: AgentState, **kwargs):
                # Explicit logger retrieval - professional practice
                from osprey.utils.logger import get_logger
                logger = get_logger("task_extraction")

                # Define streaming helper here for step awareness
                from osprey.utils.streaming import get_streamer
                streamer = get_streamer("task_extraction", state)
                streamer.status("Processing...")

                logger.info("Starting task extraction")

                # Main infrastructure logic
                result = await extract_task_from_conversation(state)
                return {"task_current_task": result.task}

            @staticmethod
            def classify_error(exc: Exception, context: dict) -> ErrorClassification:
                # Infrastructure-specific error classification
                return ErrorClassification(severity=ErrorSeverity.RETRIABLE, ...)


    :param cls: Infrastructure node class to enhance with LangGraph-native execution
    :type cls: Type[BaseInfrastructureNode]
    :param quiet: If True, suppress start/completion logging (useful for routing nodes)
    :type quiet: bool
    :return: Enhanced infrastructure class with LangGraph-native execution
    :rtype: Type[BaseInfrastructureNode]
    :raises ValueError: If required components are missing

    .. note::
       The decorator creates a `langgraph_node` attribute containing the LangGraph-compatible
       function. This is what the registry uses for actual execution.

       Infrastructure nodes use manual error handling for consistency with capability nodes.
       No automatic retry policies are created.
    """

    def decorator(cls):
        return _create_infrastructure_node(cls, quiet=quiet)

    # Handle both @infrastructure_node and @infrastructure_node(quiet=True) syntax
    if cls is None:
        # Called with parameters: @infrastructure_node(quiet=True)
        return decorator
    else:
        # Called without parameters: @infrastructure_node
        return decorator(cls)


def _create_infrastructure_node(cls, quiet=False):

    # Extract required components using reflection
    node_name = getattr(cls, 'name', None)
    description = getattr(cls, 'description', None)
    execute_func = getattr(cls, 'execute', None)
    error_classifier = getattr(cls, 'classify_error', None)
    retry_policy_func = getattr(cls, 'get_retry_policy', None)

    logger = get_logger(node_name)

    # Validate required components
    if not node_name:
        raise ValueError(f"Infrastructure node {cls.__name__} must define 'name' class attribute")
    if not description:
        raise ValueError(f"Infrastructure node {cls.__name__} must define 'description' class attribute")
    if not execute_func:
        raise ValueError(f"Infrastructure node {cls.__name__} must implement 'execute' static method")
    if not error_classifier:
        raise ValueError(f"Infrastructure node {cls.__name__} must have 'classify_error' method (inherit from BaseInfrastructureNode or define manually)")
    if not retry_policy_func:
        raise ValueError(f"Infrastructure node {cls.__name__} must have 'get_retry_policy' method (inherit from BaseInfrastructureNode or define manually)")

    # Create LangGraph-compatible node function
    async def langgraph_node(
        state: 'AgentState',
        **kwargs
    ) -> dict[str, Any]:
        """LangGraph-native node function with manual error handling.

        This function is called by LangGraph during execution. Infrastructure nodes
        now use get_stream_writer() and get_config() directly for pure LangGraph integration.

        :param state: Current agent state
        :type state: AgentState
        :param kwargs: Additional parameters from LangGraph
        :return: State updates dictionary
        :rtype: Dict[str, Any]
        """
        # Lazy import to avoid circular imports
        from osprey.state import StateManager

        # Execution timing
        start_time = time.time()

        try:
            # Only log start message if not quiet
            if not quiet:
                logger.info(f"Starting {description}")

            # Execute main infrastructure logic - nodes use get_config() internally
            result = await execute_func(
                state,
                logger=logger,
                **kwargs
            )

            execution_time = time.time() - start_time

            # Only log completion message if not quiet
            if not quiet:
                logger.success(f"Completed {description} in {execution_time:.2f}s")

            # Add execution tracking to result
            if isinstance(result, dict):
                if "control_flow" not in result:
                    result["control_flow"] = {}
                result["control_flow"]["last_execution_time"] = execution_time
                result["control_flow"]["last_infrastructure_node"] = node_name

            return result

        except Exception as exc:
            execution_time = time.time() - start_time

            # Check for development mode - re-raise original exception for debugging
            try:
                if get_config:
                    config = get_config()
                    configurable = config.get("configurable", {})
                    if configurable.get('development', {}).get('raise_raw_errors', False):
                        logger.error("Development mode: Re-raising original exception for debugging")
                        raise exc
            except (RuntimeError, ImportError, AttributeError, KeyError):
                # If config access fails (outside runnable context), continue with normal error handling
                pass

            # Re-raise GraphInterrupt immediately - it's not an error!
            if _is_graph_interrupt(exc):
                logger.info(f"GraphInterrupt detected in {node_name} - re-raising for LangGraph to handle")
                raise exc

            # Handle actual errors
            context = {
                "infrastructure_node": node_name,
                "execution_time": execution_time,
                "current_state": state
            }

            classification = error_classifier(exc, context)

            # Check for FATAL severity - raise exception to stop execution entirely
            if classification.severity == ErrorSeverity.FATAL:
                logger.error(f"FATAL error in {node_name} - Terminating execution to prevent system issues")
                technical_details = ""
                if classification.metadata and 'technical_details' in classification.metadata:
                    technical_details = f"Technical details: {classification.metadata['technical_details']}. "

                raise RuntimeError(
                    f"Fatal error in {node_name}: {classification.user_message or str(exc)}. "
                    f"{technical_details}"
                    f"Execution terminated due to system-level failure."
                )

            # Get retry policy for this infrastructure node
            retry_policy = retry_policy_func()

            logger.error(f"Error in {node_name} after {execution_time:.2f}s: {str(exc)}")
            logger.error(f"Classification: {classification.user_message or str(exc)}")

            # Use manual error handling for infrastructure nodes too

            # Track infrastructure failures in execution_step_results for consistent tracking
            step_results = state.get("execution_step_results", {}).copy()
            current_step_index = StateManager.get_current_step_index(state)
            step_key = f"infra_{current_step_index}_{node_name}"
            step_results[step_key] = {
                "step_index": current_step_index,  # For explicit ordering
                "capability": node_name,
                "task_objective": f"Infrastructure: {node_name}",  # Infrastructure nodes don't have step objectives
                "success": False,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }

            return {
                "control_has_error": True,
                "control_error_info": {
                    "node_name": node_name,
                    "classification": classification,
                    "retry_policy": retry_policy,
                    "original_error": str(exc),
                    "user_message": classification.user_message or str(exc),
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat()
                },
                "execution_step_results": step_results,
                "execution_last_result": {
                    "infrastructure_node": node_name,
                    "success": False,
                    "error": str(exc),
                    "classification": classification.severity.value if hasattr(classification.severity, 'value') else str(classification.severity),
                    "timestamp": datetime.now().isoformat()
                }
            }

    # Attach metadata to the function for LangGraph integration
    langgraph_node.name = node_name
    langgraph_node.node_name = node_name
    langgraph_node.description = description
    langgraph_node.error_classifier = error_classifier
    langgraph_node.original_class = cls

    # Set the LangGraph-native function on the class for registry discovery
    cls.langgraph_node = langgraph_node

    return cls
