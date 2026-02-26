"""Result types for execution tracking.

Provides ExecutionResult (outcome of a single execution), ExecutionRecord
(historical step record with timing), and CapabilityMatch (classification
result for capability selection).
"""

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from .errors import ExecutionError

if TYPE_CHECKING:
    from osprey.base.planning import PlannedStep


@dataclass
class ExecutionResult:
    """Result of a single capability or infrastructure node execution.

    When ``success`` is True, ``data`` holds the output; when False, ``error``
    carries the structured failure information. Timing fields are optional.

    Example::

        result = ExecutionResult(
            success=True,
            data={"temperature": 72},
            execution_time=1.23,
        )
    """

    success: bool
    data: Any | None = None
    error: ExecutionError | None = None
    execution_time: float | None = None  # Duration in seconds
    start_time: datetime | None = None  # When execution started
    end_time: datetime | None = None  # When execution completed


@dataclass
class ExecutionRecord:
    """Historical record of a completed execution step with timing.

    Pairs the original PlannedStep with its ExecutionResult and timestamps
    for audit trails, performance analysis, and debugging.

    Example::

        record = ExecutionRecord(
            step=planned_step,
            start_time=datetime.utcnow(),
            result=ExecutionResult(success=True, data={...}, execution_time=1.2),
        )
    """

    step: "PlannedStep"  # Import handled at runtime
    start_time: datetime
    result: ExecutionResult
    end_time: datetime | None = None


class CapabilityMatch(BaseModel):
    """Binary classification result indicating whether a capability matches a task."""

    is_match: bool = Field(
        description="A boolean (true or false) indicating if the user's request matches the capability."
    )
