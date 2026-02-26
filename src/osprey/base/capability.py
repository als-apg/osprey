"""
Base Capability Class

Convention-based base class for all capabilities in the Osprey Agent framework.
Implements the execution architecture with configuration-driven patterns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

from osprey.base.errors import ErrorClassification, ErrorSeverity

if TYPE_CHECKING:
    from osprey.context import CapabilityContext


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
    """Base class for capabilities using convention-based configuration.

    Subclasses must define ``name``, ``description``, and implement ``execute()``.
    The @capability_node decorator provides execution integration, error handling,
    retry policies, and execution tracking.

    Required class attributes:
        - name: Unique capability identifier for registration and routing.
        - description: Human-readable description.

    Optional class attributes:
        - provides: Context types this capability outputs (default: []).
        - requires: Context types this capability needs (default: []).

    Override ``classify_error()`` for domain-specific error handling
    and ``get_retry_policy()`` for custom retry behavior.

    Example::

        @capability_node
        class WeatherCapability(BaseCapability):
            name = "weather_data"
            description = "Retrieve current weather conditions"
            provides = ["WEATHER_DATA"]
            requires = ["LOCATION"]

            async def execute(self) -> dict[str, Any]:
                location = self.get_required_contexts()["LOCATION"]
                weather = await fetch_weather(location)
                return self.store_output_context(
                    WeatherDataContext(weather)
                )
    """

    # Required class attributes - must be overridden in subclasses
    name: str = None
    description: str = None

    # Optional class attributes - defaults provided
    provides: list[str] = []
    requires: list[str | tuple[str, Literal["single", "multiple"]]] = []

    _state: dict[str, Any] | None = None
    _step: dict[str, Any] | None = None

    def __init__(self):
        """Validate required class attributes and field formats."""
        if self.name is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define 'name' class attribute"
            )
        if self.description is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define 'description' class attribute"
            )

        if not hasattr(self.__class__, "execute"):
            raise NotImplementedError(
                f"{self.__class__.__name__} must implement 'execute' static method"
            )

        if not hasattr(self.__class__, "provides") or self.provides is None:
            self.__class__.provides = []
        if not hasattr(self.__class__, "requires") or self.requires is None:
            self.__class__.requires = []

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

        from osprey.context.context_manager import ContextManager
        from osprey.registry import get_registry

        registry = get_registry()
        context_manager = ContextManager(self._state)

        constraints: list[str | tuple[str, str]] = []
        resolved_types: dict[str, Any] = {}

        for req in self.requires:
            if isinstance(req, tuple):
                ctx_type_name, cardinality = req

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

        try:
            raw_contexts = context_manager.extract_from_step(
                self._step,
                self._state,
                constraints=constraints,
                constraint_mode=constraint_mode,  # Applies uniformly to ALL
            )
        except ValueError as e:
            raise ValueError(f"[{self.name}] Failed to extract required contexts: {e}") from e

        string_keyed: dict[str, CapabilityContext | list[CapabilityContext]] = {}
        ordered_keys: list[str] = []

        for req in self.requires:
            ctx_type_name = req[0] if isinstance(req, tuple) else req
            ctx_type = resolved_types[ctx_type_name]
            if ctx_type in raw_contexts:
                string_keyed[ctx_type_name] = raw_contexts[ctx_type]
                ordered_keys.append(ctx_type_name)

        processed = self.process_extracted_contexts(string_keyed)

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
                logger = self.get_logger()
                logger.status(f"Starting: {task}")  # Emits StatusEvent

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

        task_objective = self._step.get("task_objective")

        if task_objective:
            return task_objective

        if default is not None:
            return default

        return self._state.get("task_current_task", "")

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
            State updates dict to merge into agent state

        Raises:
            AttributeError: If context_data lacks CONTEXT_TYPE class variable
            RuntimeError: If called outside execute() (state not injected)
            ValueError: If context_key missing from step

        Example:
            ```python
            return self.store_output_context(ArchiverDataContext(...))
            ```
        """
        return self.store_output_contexts(context_data)

    def store_output_contexts(self, *context_objects: CapabilityContext) -> dict[str, Any]:
        """
        Store multiple output contexts - all self-describing.

        Args:
            *context_objects: Context objects with CONTEXT_TYPE attributes

        Returns:
            Merged state updates dict

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

        from osprey.registry import get_registry
        from osprey.state import StateManager

        registry = get_registry()

        if self.provides:
            context_types = {obj.CONTEXT_TYPE for obj in context_objects}
            if not context_types.issubset(set(self.provides)):
                raise ValueError(
                    f"[{self.name}] Context types {context_types} don't match provides: {self.provides}"
                )

        context_key = self._step.get("context_key")
        if not context_key:
            raise ValueError(
                f"[{self.name}] No context_key in step - cannot store outputs.\n"
                f"\n"
                f"This indicates a framework issue: orchestrator must provide context_key.\n"
                f"Step contents: {self._step}"
            )

        task_objective = self._step.get("task_objective")

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
        - Streams high-level messages automatically when in execution context
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

                # Status updates - emits StatusEvent
                logger.status("Creating execution plan...")

                # Info messages - emits StatusEvent with level="info"
                logger.info(f"Active capabilities: {capabilities}")

                # Debug messages - emits StatusEvent with level="debug"
                logger.debug("Detailed state information...")

                # Errors - emits ErrorEvent
                logger.error("Validation failed", validation_errors=[...])

                # Success with metadata - emits StatusEvent with level="success"
                logger.success("Plan created", steps=5, total_time=2.3)

                return self.store_output_context(result)
            ```

        .. note::
           All logger methods emit TypedEvents. The transport is handled automatically
           via event streaming (during execution) or fallback handlers. Downstream
           clients (CLI, TUI) filter and render events.

        .. seealso::
           :class:`ComponentLogger` : Logger class with streaming methods
           :func:`get_logger` : Underlying logger factory function
        """
        from osprey.utils.logger import get_logger

        return get_logger(self.name, state=self._state)

    @abstractmethod
    async def execute(self) -> dict[str, Any]:
        """Execute the main capability logic.

        The decorator injects ``self._state`` and ``self._step`` before calling.
        Use ``self.get_required_contexts()`` for input extraction and
        ``self.store_output_context()`` for result storage.

        Returns:
            Dictionary of state updates to merge into agent state.
        """
        pass

    @staticmethod
    def classify_error(exc: Exception, context: dict) -> ErrorClassification | None:
        """Classify an error as CRITICAL (default). Override for domain-specific handling.

        Args:
            exc: The exception that occurred.
            context: Error context with ``capability``, ``execution_time``,
                ``current_step_index``, and ``current_state`` keys.

        Returns:
            ErrorClassification with severity and recovery metadata,
            or None to use the default.

        Example override::

            @staticmethod
            def classify_error(exc, context):
                if isinstance(exc, (ConnectionError, TimeoutError)):
                    return ErrorClassification(
                        severity=ErrorSeverity.RETRIABLE,
                        user_message="Network issue, retrying...",
                    )
                if isinstance(exc, KeyError) and "context" in str(exc):
                    return ErrorClassification(
                        severity=ErrorSeverity.CRITICAL,
                        user_message="Required data missing",
                    )
                return ErrorClassification(
                    severity=ErrorSeverity.CRITICAL,
                    user_message=f"Unexpected error: {exc}",
                )
        """
        capability_name = context.get("capability", "unknown_capability")
        return ErrorClassification(
            severity=ErrorSeverity.CRITICAL,
            user_message=f"Unhandled error in {capability_name}: {exc}",
            metadata={"technical_details": str(exc)},
        )

    @staticmethod
    def get_retry_policy() -> dict[str, Any]:
        """Return default retry policy (3 attempts, 0.5s delay, 1.5x backoff).

        Override for capabilities with specific retry needs.
        """
        return {"max_attempts": 3, "delay_seconds": 0.5, "backoff_factor": 1.5}

    def __repr__(self) -> str:
        """Return a string representation of the capability.

        :return: String representation including class name and capability name
        :rtype: str
        """
        return f"<{self.__class__.__name__}: {self.name}>"
