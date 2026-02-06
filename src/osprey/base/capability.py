"""
Base Capability Class - LangGraph Migration

Convention-based base class for all capabilities in the Osprey Agent framework.
Implements the LangGraph-native architecture with configuration-driven patterns.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

from osprey.base.errors import ErrorClassification, ErrorSeverity

# Import types for type hints
if TYPE_CHECKING:
    from osprey.context import CapabilityContext
    from osprey.state import AgentState

# Direct imports - no circular dependencies in practice


def slash_command(name: str, state: AgentState) -> str | bool | None:
    """Read a capability-specific slash command from state.

    This is a module-level helper function for reading capability-specific slash
    commands that were parsed by the Gateway but not handled by registered commands.
    Capabilities can use these commands to customize their behavior.

    Args:
        name: Command name (without leading slash)
        state: Current agent state

    Returns:
        - str: If command was provided with a value (/beam:diagnostic -> "diagnostic")
        - True: If command was provided without value (/verbose -> True)
        - None: If command was not provided

    Examples:
        Using in a capability's execute method::

            from osprey.base.capability import slash_command

            async def execute(state: AgentState, **kwargs) -> dict[str, Any]:
                # Check for /beam:mode command
                if mode := slash_command("beam", state):
                    # mode is "diagnostic" if user typed /beam:diagnostic
                    pass

                # Check for /verbose flag
                if slash_command("verbose", state):
                    # User typed /verbose
                    pass
    """
    commands = state.get("_capability_slash_commands", {})
    return commands.get(name)


class RequiredContexts(dict):
    """Special dict that supports tuple unpacking in the order of requires field.

    This class enables elegant syntax like:
        channels, time_range = self.get_required_contexts()

    While maintaining backward compatibility with dict access:
        contexts = self.get_required_contexts()
        channels = contexts["CHANNEL_ADDRESSES"]

    The iteration order matches the order in the capability's requires field.
    """

    def __init__(self, data: dict, order: list[str]):
        """
        Initialize RequiredContexts with data and ordered keys.

        Args:
            data: Dictionary mapping context type names to context objects
            order: List of context type names in the order they appear in requires
        """
        super().__init__(data)
        self._order = order

    def __iter__(self):
        """Iterate in the order specified by requires field for tuple unpacking."""
        for key in self._order:
            if key in self:
                yield self[key]


class BaseCapability(ABC):
    """Base class for framework capabilities using convention-based configuration.

    This class provides the foundation for all capabilities in the Osprey Agent framework.
    Capabilities are self-contained business logic components that perform specific tasks
    and integrate seamlessly with the LangGraph execution model through convention-based
    patterns and automatic discovery.

    The BaseCapability class enforces a strict contract through reflection-based validation:
    capabilities must define required components and can optionally implement guidance
    systems for orchestration and classification. The @capability_node decorator provides
    complete LangGraph integration including error handling, retry policies, and
    execution tracking.

    Required Components (enforced at initialization):
        - name: Unique capability identifier used for registration and routing
        - description: Human-readable description for documentation and logging
        - execute(): Async static method containing the main business logic

    Optional Components (with defaults provided):
        - provides: List of data types this capability generates (default: [])
        - requires: List of data types this capability depends on (default: [])
        - classify_error(): Domain-specific error classification (default: all CRITICAL)
        - get_retry_policy(): Retry configuration for failure recovery (default: 3 attempts)
        - _create_orchestrator_guide(): Integration guidance for execution planning
        - _create_classifier_guide(): Task classification guidance for capability selection

    Architecture Integration:
        The capability integrates with multiple framework systems:

        1. **Execution System**: Via @capability_node decorator for LangGraph nodes
        2. **Planning System**: Via orchestrator guides for step planning
        3. **Classification System**: Via classifier guides for capability selection
        4. **Error Handling**: Via error classification for recovery strategies
        5. **Registry System**: Via convention-based configuration

    :param name: Unique capability identifier for registry and routing
    :type name: str
    :param description: Human-readable description for documentation
    :type description: str
    :param provides: Data types generated by this capability
    :type provides: List[str]
    :param requires: Data types required by this capability
    :type requires: List[str]

    :raises NotImplementedError: If required class attributes or methods are missing

    .. note::
       Use the @capability_node decorator to enable LangGraph integration with
       automatic error handling, retry policies, and execution tracking.

    .. warning::
       The execute method must be implemented as a static method and should
       return a dictionary of state updates for LangGraph to merge.

    Examples:
        Basic capability implementation::

            @capability_node
            class WeatherCapability(BaseCapability):
                name = "weather_data"
                description = "Retrieve current weather conditions"
                provides = ["WEATHER_DATA"]
                requires = ["LOCATION"]

                @staticmethod
                async def execute(state: AgentState, **kwargs) -> Dict[str, Any]:
                    location = state.get("location")
                    weather_data = await fetch_weather(location)
                    return {
                        "weather_current_conditions": weather_data,
                        "weather_last_updated": datetime.now().isoformat()
                    }

        Capability with custom error handling::

            @capability_node
            class DatabaseCapability(BaseCapability):
                name = "database_query"
                description = "Execute database queries with connection handling"

                @staticmethod
                def classify_error(exc: Exception, context: dict) -> ErrorClassification:
                    if isinstance(exc, ConnectionError):
                        return ErrorClassification(
                            severity=ErrorSeverity.RETRIABLE,
                            user_message="Database connection lost, retrying...",
                            metadata={"technical_details": str(exc)}
                        )
                    return ErrorClassification(
                        severity=ErrorSeverity.CRITICAL,
                        user_message=f"Database error: {exc}",
                        metadata={"technical_details": str(exc)}
                    )

                @staticmethod
                async def execute(state: AgentState, **kwargs) -> Dict[str, Any]:
                    # Implementation with database operations
                    pass

    .. seealso::
       :func:`capability_node` : Decorator for LangGraph integration
       :class:`BaseInfrastructureNode` : Infrastructure components base class
       :class:`ErrorClassification` : Error classification system
    """

    # Required class attributes - must be overridden in subclasses
    name: str = None
    description: str = None

    # Optional class attributes - defaults provided
    provides: list[str] = []
    requires: list[str | tuple[str, Literal["single", "multiple"]]] = []

    # Instance attributes (injected by @capability_node decorator at runtime)
    # These are set by the decorator before calling execute() and are available within execute()
    _state: AgentState | None = None
    _step: dict[str, Any] | None = None

    def __init__(self):
        """Initialize the capability and validate required components.

        Performs comprehensive validation of the capability class to ensure all
        required components are properly defined. This validation happens at
        initialization time to provide immediate feedback during development
        rather than waiting for runtime execution failures.

        The validation process checks:
        1. Required class attributes (name, description) are defined and non-None
        2. The execute method is implemented as a static method
        3. Optional attributes are properly initialized with defaults if missing
        4. The 'requires' field format is valid (strings or (context_type, cardinality) tuples)

        :raises NotImplementedError: If name or description class attributes are missing
        :raises NotImplementedError: If execute static method is not implemented
        :raises ValueError: If requires field contains invalid format

        .. note::
           This initialization performs validation only. The actual LangGraph
           integration happens through the @capability_node decorator.

        .. warning::
           Subclasses should not override this method unless they need additional
           validation. Override _create_orchestrator_guide() or _create_classifier_guide()
           for customization instead.
        """
        # Validate that subclass has defined required attributes
        if self.name is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define 'name' class attribute"
            )
        if self.description is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define 'description' class attribute"
            )

        # Validate that execute method is implemented
        if not hasattr(self.__class__, "execute"):
            raise NotImplementedError(
                f"{self.__class__.__name__} must implement 'execute' static method"
            )

        # Set defaults for optional attributes if not defined
        if not hasattr(self.__class__, "provides") or self.provides is None:
            self.__class__.provides = []
        if not hasattr(self.__class__, "requires") or self.requires is None:
            self.__class__.requires = []

        # Validate requires field format at initialization
        if self.requires:
            for idx, req in enumerate(self.requires):
                if isinstance(req, tuple):
                    if len(req) != 2:
                        raise ValueError(
                            f"{self.__class__.__name__}.requires[{idx}]: "
                            f"Invalid tuple format {req}. "
                            f"Expected (context_type: str, cardinality: 'single'|'multiple')"
                        )
                    context_type, cardinality = req
                    if not isinstance(context_type, str):
                        raise ValueError(
                            f"{self.__class__.__name__}.requires[{idx}]: "
                            f"Context type must be string, got {type(context_type).__name__}"
                        )
                    if cardinality not in ("single", "multiple"):
                        raise ValueError(
                            f"{self.__class__.__name__}.requires[{idx}]: "
                            f"Invalid cardinality '{cardinality}'. "
                            f"Must be 'single' or 'multiple'. "
                            f"\n\n"
                            f"NOTE: constraint_mode ('hard'/'soft') is NOT a tuple value!\n"
                            f"Use constraint_mode parameter in get_required_contexts() instead.\n"
                            f"\n"
                            f"Example:\n"
                            f"  requires = ['OPTIONAL_DATA_TYPE1', 'OPTIONAL_DATA_TYPE2']  # No tuple!\n"
                            f"  contexts = self.get_required_contexts(constraint_mode='soft')"
                        )
                elif not isinstance(req, str):
                    raise ValueError(
                        f"{self.__class__.__name__}.requires[{idx}]: "
                        f"Invalid type {type(req).__name__}. "
                        f"Expected string or (string, cardinality) tuple"
                    )

    # ========================================
    # NEW: Automatic Context Management Methods
    # ========================================

    def get_required_contexts(
        self, constraint_mode: Literal["hard", "soft"] = "hard"
    ) -> RequiredContexts:
        """
        Automatically extract contexts based on 'requires' field.

        The constraint_mode applies uniformly to ALL requirements.
        Use "hard" when all are required, "soft" when at least one is required.

        Tuple format is ONLY for cardinality constraints:
        - "single": Must be exactly one instance (not a list)
        - "multiple": Must be a list (not single instance)

        Args:
            constraint_mode: "hard" (all required) or "soft" (at least one required)

        Returns:
            RequiredContexts object supporting both dict and tuple unpacking access

        Raises:
            RuntimeError: If called outside execute() (state not injected)
            ValueError: If required contexts missing or cardinality violated
            AttributeError: If context type not found in registry

        Example:
            ```python
            # Define requirements
            requires = ["CHANNEL_ADDRESSES", ("TIME_RANGE", "single")]

            # Elegant tuple unpacking (matches order in requires)
            channels, time_range = self.get_required_contexts()

            # Traditional dict access (backward compatible)
            contexts = self.get_required_contexts()
            channels = contexts["CHANNEL_ADDRESSES"]
            time_range = contexts["TIME_RANGE"]
            ```

        .. note::
           Tuple unpacking only works reliably with constraint_mode="hard" (default).
           When using "soft" mode, use dict access instead since the number of
           returned contexts may vary:

               contexts = self.get_required_contexts(constraint_mode="soft")
               a = contexts.get("CONTEXT_A")
               b = contexts.get("CONTEXT_B")
        """
        if not self.requires:
            return RequiredContexts({}, [])

        # Ensure we're in execution context (state injected by decorator)
        if self._state is None:
            raise RuntimeError(
                f"{self.__class__.__name__}.get_required_contexts() requires self._state "
                f"to be injected by @capability_node decorator.\n"
                f"\n"
                f"Possible causes:\n"
                f"  1. Calling outside of execute() method\n"
                f"  2. Missing @capability_node decorator on class\n"
                f"  3. Manual instantiation without state injection\n"
                f"\n"
                f"Solution: Ensure @capability_node decorator is applied and only call "
                f"this method from within execute()"
            )

        # Import here to avoid circular dependencies
        from osprey.context.context_manager import ContextManager
        from osprey.registry import get_registry

        registry = get_registry()
        context_manager = ContextManager(self._state)

        # Parse requirements into constraints
        # Format: "CONTEXT_TYPE" or ("CONTEXT_TYPE", "single"|"multiple")
        constraints: list[str | tuple[str, str]] = []
        resolved_types: dict[str, Any] = {}  # Cache resolved types to avoid redundant lookups

        for req in self.requires:
            if isinstance(req, tuple):
                ctx_type_name, cardinality = req

                # Look up context type in registry with error handling
                try:
                    ctx_type = getattr(registry.context_types, ctx_type_name)
                except AttributeError:
                    available = [
                        attr for attr in dir(registry.context_types) if not attr.startswith("_")
                    ]
                    raise ValueError(
                        f"[{self.name}] Context type '{ctx_type_name}' not found in registry.\n"
                        f"Available types: {', '.join(available)}"
                    ) from None

                resolved_types[ctx_type_name] = ctx_type
                constraints.append((ctx_type, cardinality))
            else:
                # Simple string format
                ctx_type_name = req
                try:
                    ctx_type = getattr(registry.context_types, ctx_type_name)
                except AttributeError:
                    available = [
                        attr for attr in dir(registry.context_types) if not attr.startswith("_")
                    ]
                    raise ValueError(
                        f"[{self.name}] Context type '{ctx_type_name}' not found in registry.\n"
                        f"Available types: {', '.join(available)}"
                    ) from None

                resolved_types[ctx_type_name] = ctx_type
                constraints.append(ctx_type)

        # Extract contexts with uniform constraint mode
        try:
            raw_contexts = context_manager.extract_from_step(
                self._step,
                self._state,
                constraints=constraints,
                constraint_mode=constraint_mode,  # Applies uniformly to ALL
            )
        except ValueError as e:
            # Add capability context to errors
            raise ValueError(f"[{self.name}] Failed to extract required contexts: {e}") from e

        # Convert registry types to string keys for cleaner access
        # Note: In soft mode, raw_contexts might not have all requested keys
        string_keyed: dict[str, CapabilityContext | list[CapabilityContext]] = {}
        ordered_keys: list[str] = []  # Track order for tuple unpacking

        for req in self.requires:
            ctx_type_name = req[0] if isinstance(req, tuple) else req
            ctx_type = resolved_types[ctx_type_name]  # Reuse cached lookup
            # Only add if context was actually found (important for soft mode)
            if ctx_type in raw_contexts:
                string_keyed[ctx_type_name] = raw_contexts[ctx_type]
                ordered_keys.append(ctx_type_name)

        # Apply custom processing hook
        processed = self.process_extracted_contexts(string_keyed)

        # Return RequiredContexts with order preserved for tuple unpacking
        return RequiredContexts(processed, ordered_keys)

    def process_extracted_contexts(
        self, contexts: dict[str, CapabilityContext | list[CapabilityContext]]
    ) -> dict[str, CapabilityContext | list[CapabilityContext]]:
        """
        Override to customize extracted contexts (e.g., flatten lists).

        Args:
            contexts: Dict mapping context type names to extracted objects

        Returns:
            Processed contexts dict

        Example:
            ```python
            def process_extracted_contexts(self, contexts):
                '''Flatten list of CHANNEL_ADDRESSES.'''
                channels_raw = contexts["CHANNEL_ADDRESSES"]

                if isinstance(channels_raw, list):
                    flat = []
                    for ctx in channels_raw:
                        flat.extend(ctx.channels)
                    contexts["CHANNEL_ADDRESSES"] = flat
                else:
                    contexts["CHANNEL_ADDRESSES"] = channels_raw.channels

                return contexts
            ```
        """
        return contexts

    def get_parameters(self, default: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Get parameters from the current step.

        The orchestrator can provide optional parameters in the step definition
        that control capability behavior (e.g., precision, timeout, mode).

        Args:
            default: Default value to return if no parameters exist (defaults to empty dict)

        Returns:
            Parameters dictionary from the step

        Raises:
            RuntimeError: If called outside execute() (state not injected)

        Example:
            ```python
            async def execute(self) -> dict[str, Any]:
                params = self.get_parameters()
                precision_ms = params.get('precision_ms', 1000)
                timeout = params.get('timeout', 30)

                # Or with a custom default
                params = self.get_parameters(default={'precision_ms': 1000})
                ```
        """
        if self._step is None:
            raise RuntimeError(
                f"{self.__class__.__name__}.get_parameters() requires self._step "
                f"to be injected by @capability_node decorator.\n"
                f"\n"
                f"This method can only be called from within execute()."
            )

        if default is None:
            default = {}

        # Handle case where 'parameters' key exists but is None
        params = self._step.get("parameters", default)
        if params is None:
            return default
        return params

    def get_task_objective(self, default: str | None = None) -> str:
        """
        Get the task objective for the current step.

        The orchestrator provides task_objective in each step to describe what
        the capability should accomplish. This is commonly used for logging,
        search queries, and LLM prompts.

        Args:
            default: Default value if task_objective not in step.
                    If None, falls back to current task from state.

        Returns:
            Task objective string

        Raises:
            RuntimeError: If called outside execute() (state not injected)

        Example:
            ```python
            async def execute(self) -> dict[str, Any]:
                # Get task objective with automatic fallback
                task = self.get_task_objective()
                logger.info(f"Starting: {task}")

                # Or with custom default
                task = self.get_task_objective(default="unknown task")

                # Common pattern: use as search query
                search_query = self.get_task_objective().lower()
                ```
        """
        if self._step is None or self._state is None:
            raise RuntimeError(
                f"{self.__class__.__name__}.get_task_objective() requires self._step and self._state "
                f"to be injected by @capability_node decorator.\n"
                f"\n"
                f"This method can only be called from within execute()."
            )

        # Try to get from step first
        task_objective = self._step.get("task_objective")

        if task_objective:
            return task_objective

        # If not in step, use provided default or fall back to current task
        if default is not None:
            return default

        # Import here to avoid circular dependencies
        from osprey.state import StateManager

        return StateManager.get_current_task(self._state)

    def get_step_inputs(self, default: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
        """
        Get the inputs list from the current step.

        The orchestrator provides inputs in each step as a list of {context_type: context_key}
        mappings that specify which contexts are available for this step. This is commonly
        used for building context descriptions, validation, and informing the LLM about
        available data.

        Args:
            default: Default value to return if no inputs exist (defaults to empty list)

        Returns:
            List of input mappings from the step

        Raises:
            RuntimeError: If called outside execute() (state not injected)

        Example:
            ```python
            async def execute(self) -> dict[str, Any]:
                # Get step inputs
                step_inputs = self.get_step_inputs()

                # Use with ContextManager to build description
                from osprey.context import ContextManager
                context_manager = ContextManager(self._state)
                context_description = context_manager.get_context_access_description(step_inputs)

                # Or with a custom default
                step_inputs = self.get_step_inputs(default=[])
                ```
        """
        if self._step is None:
            raise RuntimeError(
                f"{self.__class__.__name__}.get_step_inputs() requires self._step "
                f"to be injected by @capability_node decorator.\n"
                f"\n"
                f"This method can only be called from within execute()."
            )

        if default is None:
            default = []

        # Handle case where 'inputs' key exists but is None
        inputs = self._step.get("inputs", default)
        if inputs is None:
            return default
        return inputs

    def store_output_context(self, context_data: CapabilityContext) -> dict[str, Any]:
        """
        Store single output context - uses context's CONTEXT_TYPE attribute.

        No need for provides field or state/step parameters!

        Args:
            context_data: Context object with CONTEXT_TYPE class variable

        Returns:
            State updates dict for LangGraph to merge

        Raises:
            AttributeError: If context_data lacks CONTEXT_TYPE class variable
            RuntimeError: If called outside execute() (state not injected)
            ValueError: If context_key missing from step

        Example:
            ```python
            return self.store_output_context(ArchiverDataContext(...))
            ```
        """
        # Delegate to the multiple contexts version
        return self.store_output_contexts(context_data)

    def store_output_contexts(self, *context_objects: CapabilityContext) -> dict[str, Any]:
        """
        Store multiple output contexts - all self-describing.

        Args:
            *context_objects: Context objects with CONTEXT_TYPE attributes

        Returns:
            Merged state updates dict for LangGraph

        Raises:
            AttributeError: If any context lacks CONTEXT_TYPE
            RuntimeError: If called outside execute()
            ValueError: If context types don't match provides field

        Example:
            ```python
            return self.store_output_contexts(
                ArchiverDataContext(...),
                MetadataContext(...),
                StatisticsContext(...)
            )
            ```
        """
        if self._state is None or self._step is None:
            raise RuntimeError(
                f"{self.__class__.__name__}.store_output_contexts() requires self._state and self._step "
                f"to be injected by @capability_node decorator.\n"
                f"\n"
                f"This method can only be called from within execute()."
            )

        # Import here to avoid circular dependencies
        from osprey.registry import get_registry
        from osprey.state import StateManager

        registry = get_registry()

        # Optional validation if provides field exists
        if self.provides and len(self.provides) > 0:
            context_types = {obj.CONTEXT_TYPE for obj in context_objects}
            if not context_types.issubset(set(self.provides)):
                raise ValueError(
                    f"[{self.name}] Context types {context_types} don't match provides: {self.provides}"
                )

        # Validate context_key exists
        context_key = self._step.get("context_key")
        if not context_key:
            raise ValueError(
                f"[{self.name}] No context_key in step - cannot store outputs.\n"
                f"\n"
                f"This indicates a framework issue: orchestrator must provide context_key.\n"
                f"Step contents: {self._step}"
            )

        # Extract task_objective for context metadata (enables orchestrator context reuse)
        task_objective = self._step.get("task_objective")

        # Store each and merge updates
        merged: dict[str, Any] = {}
        for obj in context_objects:
            if not hasattr(obj, "CONTEXT_TYPE"):
                raise AttributeError(
                    f"Context {type(obj).__name__} must have CONTEXT_TYPE class variable"
                )

            try:
                ctx_type = getattr(registry.context_types, obj.CONTEXT_TYPE)
            except AttributeError:
                available = [
                    attr for attr in dir(registry.context_types) if not attr.startswith("_")
                ]
                raise ValueError(
                    f"[{self.name}] Context type '{obj.CONTEXT_TYPE}' not found in registry.\n"
                    f"Available types: {', '.join(available)}"
                ) from None

            updates = StateManager.store_context(
                self._state, ctx_type, context_key, obj, task_objective=task_objective
            )
            merged = {**merged, **updates}

        return merged

    def get_logger(self):
        """Get unified logger with automatic streaming support.

        Creates a logger that:
        - Uses this capability's name automatically
        - Has access to state for streaming via self._state
        - Streams high-level messages automatically when in LangGraph context
        - Logs to CLI with Rich formatting

        The logger intelligently handles both CLI output and web UI streaming through
        a single API. High-level status updates (status, error, success) automatically
        stream to the web UI, while detailed logging (info, debug) goes to CLI only
        by default.

        Returns:
            ComponentLogger instance with streaming capability

        Example:
            ```python
            async def execute(self) -> dict[str, Any]:
                logger = self.get_logger()

                # High-level status - logs + streams automatically
                logger.status("Creating execution plan...")

                # Detailed info - logs only (unless explicitly requested)
                logger.info(f"Active capabilities: {capabilities}")

                # Explicit streaming for specific info
                logger.info("Step 1 of 5 complete", stream=True, progress=0.2)

                # Errors always stream
                logger.error("Validation failed", validation_errors=[...])

                # Success with metadata
                logger.success("Plan created", steps=5, total_time=2.3)

                return self.store_output_context(result)
            ```

        .. note::
           The logger uses lazy initialization for streaming, so it gracefully
           handles contexts where LangGraph streaming is not available (tests,
           utilities, CLI-only execution).

        .. seealso::
           :class:`ComponentLogger` : Logger class with streaming methods
           :func:`get_logger` : Underlying logger factory function
        """
        from osprey.utils.logger import get_logger

        return get_logger(self.name, state=self._state)

    def slash_command(self, name: str) -> str | bool | None:
        """Read a capability-specific slash command from state.

        Convenience method that uses self._state automatically. This allows
        capabilities to read custom slash commands that were not registered
        in the command registry.

        Args:
            name: Command name (without leading slash)

        Returns:
            - str: If command was provided with a value (/beam:diagnostic -> "diagnostic")
            - True: If command was provided without value (/verbose -> True)
            - None: If command was not provided

        Raises:
            RuntimeError: If called outside execute() (state not injected)

        Examples:
            Using in a capability's execute method::

                async def execute(self) -> dict[str, Any]:
                    # Check for /beam:mode command
                    if mode := self.slash_command("beam"):
                        # mode is "diagnostic" if user typed /beam:diagnostic
                        self._state["beam_mode"] = mode

                    # Check for /verbose flag
                    if self.slash_command("verbose"):
                        # User typed /verbose
                        self._state["beamline_verbose"] = True
        """
        if self._state is None:
            raise RuntimeError(
                f"{self.__class__.__name__}.slash_command() called before state injection. "
                f"This method can only be called from within execute()."
            )
        commands = self._state.get("_capability_slash_commands", {})
        return commands.get(name)

    @abstractmethod
    async def execute(self) -> dict[str, Any]:
        """Execute the main capability logic with comprehensive state management.

        This is the core method that all capabilities must implement. It contains
        the primary business logic and integrates with the framework's state
        management system.

        **NEW PATTERN** (Recommended - Simplified):
            - NO parameters needed - use self._state and self._step
            - Use self.get_required_contexts() for input extraction
            - Use self.store_output_context() for result storage
            - Access state via self._state, step via self._step
            - NO @staticmethod decorator

        **OLD PATTERN** (Legacy - Still Supported):
            - Static method with state parameter (and **kwargs that was never used)
            - Manual StateManager and ContextManager usage
            - Works during migration period

        The @capability_node decorator injects self._state and self._step before
        calling execute(), so they are always available in the new pattern.

        Returns:
            Dictionary of state updates for LangGraph to merge into agent state

        Raises:
            NotImplementedError: This is an abstract method that must be implemented
            ValidationError: If required state data is missing or invalid
            CapabilityError: For capability-specific execution failures

        Example (NEW PATTERN - Recommended):
            ```python
            async def execute(self) -> dict[str, Any]:
                # Auto-extract contexts using requires field
                contexts = self.get_required_contexts()
                input_data = contexts["INPUT_DATA"]

                # Business logic
                result = await process_data(input_data)

                # Auto-store with type inference
                return self.store_output_context(OutputContext(result))
            ```

        Example (OLD PATTERN - Still Supported):
            ```python
            @staticmethod
            async def execute(state: AgentState) -> dict[str, Any]:
                step = StateManager.get_current_step(state)
                # ... manual extraction and storage ...
                return state_updates
            ```

        .. note::
           The decorator provides automatic error handling, retry policies,
           timing, and execution tracking. Focus on the core business logic.

        .. seealso::
           :func:`capability_node` : Decorator that provides execution infrastructure
           :meth:`get_required_contexts` : Automatic context extraction
           :meth:`store_output_context` : Automatic context storage
        """
        logger = logging.getLogger(__name__)
        logger.warning(
            "⚠️  Capability is using the empty base execute() - consider implementing execute() for proper functionality."
        )
        pass

    # Optional methods for registry configuration - implement as needed

    @staticmethod
    def classify_error(exc: Exception, context: dict) -> ErrorClassification | None:
        """Classify errors for capability-specific error handling and recovery.

        This method provides domain-specific error classification to determine
        appropriate recovery strategies. The default implementation treats all
        errors as critical, but capabilities should override this method to
        provide sophisticated error handling based on their specific failure modes.

        The error classification determines how the framework responds to failures:
        - CRITICAL: End execution immediately
        - RETRIABLE: Retry with same parameters
        - REPLANNING: Create new execution plan
        - RECLASSIFICATION: Reclassify task capabilities
        - FATAL: System-level failure, terminate execution

        :param exc: The exception that occurred during capability execution
        :type exc: Exception
        :param context: Error context including capability info and execution state
        :type context: dict
        :return: Error classification with recovery strategy, or None to use default
        :rtype: Optional[ErrorClassification]

        .. note::
           The context dictionary contains useful information including:
           - 'capability': capability name
           - 'current_step_index': step being executed
           - 'execution_time': time spent before failure
           - 'current_state': agent state at time of error

        Examples:
            Network-aware error classification::

                @staticmethod
                def classify_error(exc: Exception, context: dict) -> ErrorClassification:
                    # Retry network timeouts and connection errors
                    if isinstance(exc, (ConnectionError, TimeoutError)):
                        return ErrorClassification(
                            severity=ErrorSeverity.RETRIABLE,
                            user_message="Network issue detected, retrying...",
                            metadata={"technical_details": str(exc)}
                        )


                    # Default to critical for unexpected errors
                    return ErrorClassification(
                        severity=ErrorSeverity.CRITICAL,
                        user_message=f"Unexpected error: {exc}",
                        metadata={"technical_details": str(exc)}
                    )

            Missing input data requiring replanning::

                @staticmethod
                def classify_error(exc: Exception, context: dict) -> ErrorClassification:
                    if isinstance(exc, KeyError) and "context" in str(exc):
                        return ErrorClassification(
                            severity=ErrorSeverity.REPLANNING,
                            user_message="Required data not available, trying different approach",
                            metadata={"technical_details": f"Missing context data: {str(exc)}"}
                        )
                    return BaseCapability.classify_error(exc, context)

        .. seealso::
           :class:`ErrorClassification` : Error classification result structure
           :class:`ErrorSeverity` : Available severity levels and their meanings
        """
        capability_name = context.get("capability", "unknown_capability")
        return ErrorClassification(
            severity=ErrorSeverity.CRITICAL,
            user_message=f"Unhandled error in {capability_name}: {exc}",
            metadata={"technical_details": str(exc)},
        )

    @staticmethod
    def get_retry_policy() -> dict[str, Any]:
        """Get retry policy configuration for failure recovery strategies.

        This method provides retry configuration that the framework uses for
        manual retry handling when capabilities fail with RETRIABLE errors.
        The default policy provides reasonable defaults for most capabilities,
        but should be overridden for capabilities with specific timing or
        retry requirements.

        The retry policy controls:
        - Maximum number of retry attempts before giving up
        - Initial delay between retry attempts
        - Backoff factor for exponential delay increase

        :return: Dictionary containing retry configuration parameters
        :rtype: Dict[str, Any]

        .. note::
           The framework uses manual retry handling rather than LangGraph's
           native retry policies to ensure consistent behavior across all
           components and to enable sophisticated error classification.

        Examples:
            Aggressive retry for network-dependent capability::

                @staticmethod
                def get_retry_policy() -> Dict[str, Any]:
                    return {
                        "max_attempts": 5,      # More attempts for network issues
                        "delay_seconds": 2.0,   # Longer delay for external services
                        "backoff_factor": 2.0   # Exponential backoff
                    }

            Conservative retry for expensive operations::

                @staticmethod
                def get_retry_policy() -> Dict[str, Any]:
                    return {
                        "max_attempts": 2,      # Minimal retries for expensive ops
                        "delay_seconds": 0.1,   # Quick retry for transient issues
                        "backoff_factor": 1.0   # No backoff for fast operations
                    }

        .. seealso::
           :func:`classify_error` : Error classification that determines when to retry
           :class:`ErrorSeverity` : RETRIABLE severity triggers retry policy usage
        """
        return {"max_attempts": 3, "delay_seconds": 0.5, "backoff_factor": 1.5}

    def _create_orchestrator_guide(self) -> Any | None:
        """Template Method: Create orchestrator guide for planning integration.

        IMPLEMENTATION APPROACHES (choose based on your needs):

        1. **Production (Recommended)**: Use prompt builders through registry::

               def _create_orchestrator_guide(self):
                   prompt_provider = get_framework_prompts()
                   builder = prompt_provider.get_my_capability_prompt_builder()
                   return builder.get_orchestrator_guide()

        2. **R&D/Experimentation**: Direct implementation for quick prototyping::

               def _create_orchestrator_guide(self):
                   return OrchestratorGuide(
                       instructions="Use when user mentions X, Y, Z...",
                       examples=[...], priority=10
                   )

        :return: Orchestrator snippet for planning integration, or None if not needed
        :rtype: Optional[OrchestratorGuide]

        Example::

            def _create_orchestrator_guide(self) -> Optional[OrchestratorGuide]:
                return OrchestratorGuide(
                    capability_name="time_range_parsing",
                    description="Parse time references into structured datetime ranges",
                    when_to_use="When user mentions time periods, dates, or relative time references",
                    provides_context="TIME_RANGE with start_date and end_date datetime objects",
                    example_usage="For 'show me data from last week' or 'yesterday's performance'"
                )
        """
        logger = logging.getLogger(__name__)
        logger.warning(
            f"⚠️  Capability '{self.name}' is using base _create_orchestrator_guide() - "
            "this may cause orchestrator hallucination. Consider implementing "
            "_create_orchestrator_guide() for proper integration."
        )
        return None

    def _create_classifier_guide(self) -> Any | None:
        """Template Method: Create classifier guide for capability activation.

        IMPLEMENTATION APPROACHES (choose based on your needs):

        1. **Production (Recommended)**: Use prompt builders through registry::

               def _create_classifier_guide(self):
                   prompt_provider = get_framework_prompts()
                   builder = prompt_provider.get_my_capability_prompt_builder()
                   return builder.get_classifier_guide()

        2. **R&D/Experimentation**: Direct implementation for quick testing::

               def _create_classifier_guide(self):
                   return TaskClassifierGuide(
                       instructions="Activate when user mentions time-related data requests",
                       examples=[
                           ClassifierExample(
                               query="Show me data from last week",
                               result=True,
                               reason="Contains time range requiring parsing"
                           ),
                           ClassifierExample(
                               query="What is the current status?",
                               result=False,
                               reason="Current status request, no time parsing needed"
                           )
                       ]
                   )

        :return: Classifier guide for capability selection, or None if not needed
        :rtype: Optional[TaskClassifierGuide]
        """
        logger = logging.getLogger(__name__)
        logger.warning(
            f"⚠️  Capability '{self.name}' is using base _create_classifier_guide() - "
            "this may cause classification issues. Consider implementing "
            "_create_classifier_guide() for proper task classification."
        )
        return None

    # Properties for compatibility and introspection

    @property
    def orchestrator_guide(self) -> Any | None:
        """Get the orchestrator guide for this capability (lazy-loaded).

        Standardized interface used by the framework. Automatically calls
        _create_orchestrator_guide() on first access and caches the result.

        :return: Orchestrator guide for execution planning integration, or None if not needed
        :rtype: Optional[OrchestratorGuide]
        """
        if not hasattr(self, "_orchestrator_guide"):
            try:
                self._orchestrator_guide = self._create_orchestrator_guide()
            except Exception as e:
                self.get_logger().warning(
                    f"Failed to create orchestrator guide for capability '{self.name}': {e}"
                )
                self._orchestrator_guide = None
        return self._orchestrator_guide

    @property
    def classifier_guide(self) -> Any | None:
        """Get the classifier guide for this capability (lazy-loaded).

        Standardized interface used by the framework. Automatically calls
        _create_classifier_guide() on first access and caches the result.

        :return: Classifier guide for capability activation, or None if not needed
        :rtype: Optional[TaskClassifierGuide]
        """
        if not hasattr(self, "_classifier_guide"):
            try:
                self._classifier_guide = self._create_classifier_guide()
            except Exception as e:
                self.get_logger().warning(
                    f"Failed to create classifier guide for capability '{self.name}': {e}"
                )
                self._classifier_guide = None
        return self._classifier_guide

    def __repr__(self) -> str:
        """Return a string representation of the capability.

        :return: String representation including class name and capability name
        :rtype: str
        """
        return f"<{self.__class__.__name__}: {self.name}>"
