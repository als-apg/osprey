"""Base class for infrastructure nodes (orchestration, classification, routing).

Infrastructure nodes handle system-level coordination and are designed to
fail fast with conservative retry policies. Use the @infrastructure_node
decorator for execution integration.
"""

from abc import ABC, abstractmethod
from typing import Any

from osprey.base.errors import ErrorClassification, ErrorSeverity


class BaseInfrastructureNode(ABC):
    """Base class for infrastructure nodes (orchestration, routing, classification).

    Subclasses must define ``name`` and ``description`` class attributes and
    implement ``execute()`` as an async static method. The @infrastructure_node
    decorator provides execution integration, parameter injection, timing,
    and error handling.

    Default error handling is conservative: most errors are classified as
    CRITICAL with minimal retries. Override ``classify_error()`` and
    ``get_retry_policy()`` for infrastructure that benefits from retries
    (e.g., LLM-based nodes with transient API failures).

    Example::

        @infrastructure_node
        class TaskExtractionNode(BaseInfrastructureNode):
            name = "task_extraction"
            description = "Task Extraction and Processing"

            @staticmethod
            async def execute(state: dict[str, Any], **kwargs) -> dict[str, Any]:
                task = state.get("task_current_task", "")
                return {"task_current_task": task}
    """

    # Required class attributes - must be overridden in subclasses
    name: str = None
    description: str = None

    @staticmethod
    @abstractmethod
    async def execute(state: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Execute the infrastructure operation.

        Args:
            state: Current agent state dictionary.
            **kwargs: Additional parameters (logger, configuration, etc.).

        Returns:
            Dictionary of state updates to merge into agent state.
        """
        pass

    def get_logger(self):
        """Get unified logger with automatic streaming support.

        Creates a logger that:
        - Uses this infrastructure node's name automatically
        - Has access to state for streaming via self._state
        - Streams high-level messages automatically when in execution context
        - Logs to CLI with Rich formatting

        Returns:
            ComponentLogger instance with streaming capability
        """
        from osprey.utils.logger import get_logger

        return get_logger(self.name, state=self._state)

    @staticmethod
    def classify_error(exc: Exception, context: dict) -> "ErrorClassification":
        """Classify an error as CRITICAL (default for infrastructure).

        Override for infrastructure nodes that benefit from retries
        (e.g., LLM-based orchestrators with transient API failures).

        Args:
            exc: The exception that occurred.
            context: Error context with ``infrastructure_node``, ``execution_time``,
                and ``current_state`` keys.

        Returns:
            ErrorClassification with severity and recovery metadata.

        Example override::

            @staticmethod
            def classify_error(exc, context):
                if isinstance(exc, (ConnectionError, TimeoutError)):
                    return ErrorClassification(
                        severity=ErrorSeverity.RETRIABLE,
                        user_message="Network timeout, retrying...",
                    )
                return ErrorClassification(
                    severity=ErrorSeverity.CRITICAL,
                    user_message=f"Infrastructure error: {exc}",
                )
        """
        node_name = context.get("infrastructure_node", "unknown_infrastructure_node")
        return ErrorClassification(
            severity=ErrorSeverity.CRITICAL,
            user_message=f"Infrastructure error in {node_name}: {exc}",
            metadata={"technical_details": str(exc)},
        )

    @staticmethod
    def get_retry_policy() -> dict[str, Any]:
        """Return conservative retry policy (2 attempts, 0.2s delay, 1.2x backoff).

        Override for infrastructure nodes that benefit from more retries.
        """
        return {
            "max_attempts": 2,  # Conservative for infrastructure
            "delay_seconds": 0.2,  # Fast retry for infrastructure
            "backoff_factor": 1.2,  # Minimal backoff
        }

    def get_current_task(self) -> str | None:
        """Get current task from state.

        Returns:
            Current task string, or None if not set

        Example::

            async def execute(self) -> dict[str, Any]:
                current_task = self.get_current_task()
                if not current_task:
                    raise ValueError("No current task available")
        """
        return self._state.get("task_current_task") if self._state else None

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
        if not self._state:
            return None
        messages = self._state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content")
        return None

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
        if not self._state:
            return None
        plan = self._state.get("planning_execution_plan")
        if isinstance(plan, dict) and "steps" in plan:
            return plan
        return None

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
        if not self._state:
            return 0
        return self._state.get("execution_current_step_index", 0)

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
        plan = self.get_execution_plan()
        if not plan:
            raise RuntimeError("No execution plan available in state")
        step_index = self.get_current_step_index()
        steps = plan.get("steps", [])
        if step_index >= len(steps):
            raise RuntimeError(
                f"Step index {step_index} out of range for plan with {len(steps)} steps"
            )
        return steps[step_index]

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"
