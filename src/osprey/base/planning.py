"""Execution planning types and utilities.

Defines PlannedStep and ExecutionPlan TypedDicts used by the orchestrator
to represent multi-step capability sequences, plus save/load helpers for
plan persistence.
"""

import json
from datetime import datetime
from pathlib import Path

from typing_extensions import TypedDict

from osprey.events import EventEmitter, StatusEvent
from osprey.utils.logger import get_logger

logger = get_logger("planning")


class PlannedStep(TypedDict, total=False):
    """A single capability execution step within an execution plan.

    All fields are optional (total=False) to support incremental construction.

    Example::

        step = PlannedStep(
            context_key="weather_data",
            capability="weather_retrieval",
            task_objective="Retrieve current weather for San Francisco",
            success_criteria="Weather data with temperature and conditions",
            expected_output="WEATHER_DATA",
            parameters={"location": "San Francisco", "units": "metric"},
        )
    """

    context_key: str  # Unique key for storing this step's results in execution context
    capability: str  # Name of the capability to execute
    task_objective: str  # Self-sufficient description of what this step must accomplish
    success_criteria: str  # How to determine successful completion
    expected_output: str | None  # Context type key for results (e.g., "PV_ADDRESSES")
    parameters: dict[str, str | int | float] | None  # Capability-specific configuration
    inputs: list[dict[str, str]] | None  # Step inputs as [{context_type: context_key}, ...]


class ExecutionPlan(TypedDict, total=False):
    """Ordered sequence of PlannedStep dicts representing a full execution plan.

    Example::

        plan = ExecutionPlan(steps=[step1, step2, step3])
    """

    steps: list[PlannedStep]


def save_execution_plan_to_file(plan: ExecutionPlan, file_path: str) -> None:
    """Save ExecutionPlan to JSON file for persistence or debugging.

    :param plan: ExecutionPlan dictionary to save
    :param file_path: Path where the execution plan should be saved
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    plan_with_metadata = {
        "__metadata__": {
            "version": "1.0",
            "serialization_type": "execution_plan",
            "created_at": datetime.now().isoformat(),
        },
        **plan,
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(plan_with_metadata, f, indent=2, ensure_ascii=False)

    emitter = EventEmitter("planning")
    emitter.emit(
        StatusEvent(
            component="planning",
            message=f"Saved ExecutionPlan with {len(plan.get('steps', []))} steps to: {file_path}",
            level="info",
        )
    )


def load_execution_plan_from_file(file_path: str) -> ExecutionPlan:
    """Load ExecutionPlan from JSON file.

    :param file_path: Path to the JSON file containing the execution plan
    :return: ExecutionPlan dictionary
    """
    file_path = Path(file_path)

    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    if "__metadata__" in data:
        del data["__metadata__"]

    emitter = EventEmitter("planning")
    emitter.emit(
        StatusEvent(
            component="planning",
            message=f"Loaded ExecutionPlan with {len(data.get('steps', []))} steps from: {file_path}",
            level="info",
        )
    )

    return ExecutionPlan(data)
