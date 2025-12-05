"""Execution Planning Framework - Agent Orchestration and Step Management

This module provides the comprehensive execution planning system for the Osprey
framework. It implements TypedDict-based execution planning that enables sophisticated
orchestration, step sequencing, and capability coordination while maintaining full
compatibility with LangGraph's checkpointing and serialization systems.

The planning system serves as the foundation for intelligent agent orchestration
by providing structured representations of execution steps, complete execution
plans, and utility functions for plan management. The TypedDict approach ensures
type safety, JSON serialization compatibility, and seamless integration with
LangGraph's state management and checkpointing systems.

Key Planning System Components:
    1. **PlannedStep**: Individual execution step with objectives and requirements
    2. **ExecutionPlan**: Complete execution sequence with ordered capability steps
    3. **Utility Functions**: Plan persistence, loading, and management operations
    4. **Type Safety**: TypedDict-based definitions for reliable serialization
    5. **LangGraph Integration**: Native compatibility with checkpointing and state

Planning Architecture:
    - **Structured Steps**: Complete step definitions with context and objectives
    - **Parameter Management**: Flexible parameter passing between capabilities
    - **Input/Output Mapping**: Clear data flow specification between steps
    - **Success Criteria**: Explicit success definitions for each execution step
    - **Context Management**: Unique context keys for result storage and retrieval

The planning system emphasizes clarity, type safety, and serialization compatibility
to ensure reliable execution orchestration and effective agent coordination. All
planning structures are designed for efficient storage, retrieval, and modification
throughout the execution lifecycle.

.. note::
   All planning structures use TypedDict with total=False to support partial
   updates in LangGraph state management. This enables incremental plan
   construction and modification during execution.

.. warning::
   Execution plans should maintain referential integrity between steps through
   proper context key management. Avoid circular dependencies in step sequencing.

.. seealso::
   :mod:`osprey.base.results` : Execution result tracking and management
   :class:`BaseCapability` : Capability integration with planning system
   :mod:`osprey.infrastructure.orchestration` : Plan creation and execution
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


class PlannedStep(TypedDict, total=False):
    """Individual execution step with comprehensive orchestration context.

    This TypedDict represents a single capability execution within an agent's
    execution plan. It provides complete context including objectives, success
    criteria, input requirements, and expected outputs to enable sophisticated
    orchestration and capability coordination throughout the framework.

    PlannedStep serves multiple critical functions:
    1. **Orchestration Guidance**: Clear objectives and success criteria for execution
    2. **Data Flow Management**: Input/output specifications for capability chaining
    3. **Context Management**: Unique keys for result storage and retrieval
    4. **Parameter Passing**: Flexible configuration for capability customization
    5. **Execution Tracking**: Complete context for monitoring and debugging

    The structure uses total=False to support partial updates in LangGraph's
    state management system, enabling incremental plan construction and
    modification during execution. All fields are optional to provide
    flexibility in plan creation and evolution.

    Field Definitions:
        - context_key: Unique identifier for storing step results in execution context
        - capability: Name of the capability to execute for this step
        - task_objective: Complete, self-sufficient description of step goals
        - success_criteria: Clear criteria for determining successful completion
        - expected_output: Context type key where results will be stored
        - parameters: Optional capability-specific configuration parameters
        - inputs: Step inputs as list of {context_type: context_key} mappings

    Default behaviors (when fields not provided):
        - expected_output: None (no specific output context expected)
        - parameters: None (no custom parameters required)
        - inputs: [] (no input dependencies)

    .. note::
       The task_objective should be complete and self-sufficient to enable
       capability execution without additional context. Success_criteria should
       be specific and measurable for reliable execution validation.

    .. warning::
       Context keys must be unique within an execution plan to prevent
       result collisions. Use descriptive, namespaced keys for clarity.

    Examples:
        Data retrieval step::

            step = PlannedStep(
                context_key="weather_data",
                capability="weather_retrieval",
                task_objective="Retrieve current weather conditions for San Francisco",
                success_criteria="Weather data retrieved with temperature and conditions",
                expected_output="WEATHER_DATA",
                parameters={"location": "San Francisco", "units": "metric"}
            )

        Data processing step with dependencies::

            step = PlannedStep(
                context_key="processed_data",
                capability="data_processor",
                task_objective="Process raw sensor data for trend analysis",
                success_criteria="Data processed with statistical summary available",
                expected_output="PROCESSED_DATA",
                inputs=[{"RAW_SENSOR_DATA": "sensor_readings"}],
                parameters={"analysis_type": "trend", "window_size": 24}
            )

    .. seealso::
       :class:`ExecutionPlan` : Complete execution plan containing multiple steps
       :class:`ExecutionRecord` : Historical record of completed step executions
    """

    context_key: str  # Unique identifier for storing step results in execution context
    capability: str  # Name of the capability to execute for this step
    task_objective: str  # Complete, self-sufficient description of what this step must accomplish
    success_criteria: str  # Criteria for determining successful step completion
    expected_output: (
        str | None
    )  # Context type key where results will be stored (e.g., "PV_ADDRESSES")
    parameters: (
        dict[str, str | int | float] | None
    )  # Optional capability-specific configuration parameters
    inputs: (
        list[dict[str, str]] | None
    )  # Step inputs as list of {context_type: context_key} mappings


class ExecutionPlan(TypedDict, total=False):
    """Complete execution plan with ordered capability sequence and orchestration context.

    This TypedDict represents the orchestrator's comprehensive plan for accomplishing
    a user's request through a coordinated sequence of capability executions. It
    provides the complete execution roadmap including step ordering, data flow,
    and coordination requirements for complex multi-capability tasks.

    ExecutionPlan serves as the primary coordination mechanism for:
    1. **Multi-Step Execution**: Ordered sequence of capability invocations
    2. **Data Flow Management**: Input/output coordination between capabilities
    3. **State Persistence**: LangGraph-compatible structure for checkpointing
    4. **Execution Tracking**: Foundation for monitoring and debugging
    5. **Plan Evolution**: Support for dynamic plan modification during execution

    The structure uses total=False to support incremental plan construction
    and modification in LangGraph's state management system. This enables
    dynamic planning where plans can be built progressively and modified
    based on execution results and changing requirements.

    Plan Structure:
        - steps: Ordered list of PlannedStep objects defining the execution sequence

    Default behaviors (when fields not provided):
        - steps: [] (empty plan requiring population)

    .. note::
       ExecutionPlan uses pure dictionary format for maximum compatibility with
       LangGraph's serialization and checkpointing systems. All plan data can
       be safely persisted and restored across execution sessions.

    .. warning::
       Step ordering is critical for proper execution flow. Ensure dependencies
       between steps are properly sequenced and context keys are unique to
       prevent execution conflicts.

    Examples:
        Simple two-step execution plan::

            plan = ExecutionPlan(
                steps=[
                    PlannedStep(
                        context_key="user_location",
                        capability="location_detection",
                        task_objective="Determine user's current location",
                        success_criteria="Location coordinates available",
                        expected_output="LOCATION_DATA"
                    ),
                    PlannedStep(
                        context_key="weather_report",
                        capability="weather_retrieval",
                        task_objective="Get weather for user's location",
                        success_criteria="Weather data retrieved successfully",
                        expected_output="WEATHER_DATA",
                        inputs=[{"LOCATION_DATA": "user_location"}]
                    )
                ]
            )

        Complex data processing pipeline::

            plan = ExecutionPlan(
                steps=[
                    PlannedStep(
                        context_key="raw_data",
                        capability="data_ingestion",
                        task_objective="Ingest sensor data from last 24 hours",
                        success_criteria="Raw data available for processing",
                        expected_output="RAW_SENSOR_DATA"
                    ),
                    PlannedStep(
                        context_key="cleaned_data",
                        capability="data_cleaning",
                        task_objective="Clean and validate sensor data",
                        success_criteria="Data cleaned with quality metrics",
                        expected_output="CLEANED_DATA",
                        inputs=[{"RAW_SENSOR_DATA": "raw_data"}]
                    ),
                    PlannedStep(
                        context_key="analysis_results",
                        capability="trend_analysis",
                        task_objective="Analyze trends in cleaned sensor data",
                        success_criteria="Trend analysis complete with insights",
                        expected_output="ANALYSIS_RESULTS",
                        inputs=[{"CLEANED_DATA": "cleaned_data"}]
                    )
                ]
            )

    .. seealso::
       :class:`PlannedStep` : Individual execution step structure
       :func:`save_execution_plan_to_file` : Plan persistence utilities
       :func:`load_execution_plan_from_file` : Plan loading utilities
    """

    steps: list[PlannedStep]  # Ordered list of execution steps comprising the plan


# Utility functions for working with execution plans (optional convenience functions)


def save_execution_plan_to_file(plan: ExecutionPlan, file_path: str) -> None:
    """Save ExecutionPlan to JSON file for persistence or debugging.

    :param plan: ExecutionPlan dictionary to save
    :param file_path: Path where the execution plan should be saved
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Add metadata for version tracking
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

    logger.info(f"Saved ExecutionPlan with {len(plan.get('steps', []))} steps to: {file_path}")


def load_execution_plan_from_file(file_path: str) -> ExecutionPlan:
    """Load ExecutionPlan from JSON file.

    :param file_path: Path to the JSON file containing the execution plan
    :return: ExecutionPlan dictionary
    """
    file_path = Path(file_path)

    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    # Remove metadata if present
    if "__metadata__" in data:
        del data["__metadata__"]

    logger.info(f"Loaded ExecutionPlan with {len(data.get('steps', []))} steps from: {file_path}")

    return ExecutionPlan(data)
