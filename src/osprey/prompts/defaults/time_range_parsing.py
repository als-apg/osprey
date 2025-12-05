"""
Time Range Parsing Prompt Builder

Default prompts for time range parsing capability.
"""

import textwrap

from osprey.base import (
    ClassifierActions,
    ClassifierExample,
    OrchestratorExample,
    OrchestratorGuide,
    PlannedStep,
    TaskClassifierGuide,
)
from osprey.prompts.base import FrameworkPromptBuilder
from osprey.registry import get_registry


class DefaultTimeRangeParsingPromptBuilder(FrameworkPromptBuilder):
    """Default time range parsing prompt builder."""

    PROMPT_TYPE = "time_range_parsing"

    def get_role_definition(self) -> str:
        """Get the role definition for time range parsing."""
        return "You are an expert time range parser that converts natural language time expressions into precise datetime ranges."

    def get_task_definition(self) -> str:
        """Get the task definition for time range parsing."""
        return "TASK: Parse time references from user queries and convert them to absolute datetime ranges."

    def get_instructions(self) -> str:
        """Get the instructions for time range parsing."""
        return textwrap.dedent(
            """
            INSTRUCTIONS:
            1. Identify time references in the user query (relative, absolute, or implicit)
            2. Convert all time expressions to absolute datetime values
            3. Ensure start_date is always before end_date
            4. Use current time as reference for relative expressions
            5. Return structured datetime objects in YYYY-MM-DD HH:MM:SS format

            SUPPORTED PATTERNS:
            - Relative: "last X hours/days", "yesterday", "this week"
            - Absolute: "from YYYY-MM-DD to YYYY-MM-DD"
            - Implicit: "current", "recent" (default to last few minutes)
            """
        ).strip()

    def get_orchestrator_guide(self) -> OrchestratorGuide | None:
        """Create orchestrator guide for time range parsing."""
        registry = get_registry()

        # Define structured examples using simplified dict format
        relative_time_example = OrchestratorExample(
            step=PlannedStep(
                context_key="last_week_timerange",
                capability="time_range_parsing",
                task_objective="Parse 'last week' time reference into absolute datetime objects",
                expected_output=registry.context_types.TIME_RANGE,
                success_criteria="Time range successfully parsed to absolute datetime objects",
                inputs=[],
            ),
            scenario_description="Parsing relative time references like 'last hour', 'yesterday'",
            notes=f"Output stored under {registry.context_types.TIME_RANGE} context type as datetime objects with full datetime functionality.",
        )

        absolute_time_example = OrchestratorExample(
            step=PlannedStep(
                context_key="explicit_timerange",
                capability="time_range_parsing",
                task_objective="Parse explicit datetime range '2024-01-15 09:00:00 to 2024-01-15 21:00:00' and validate format",
                expected_output=registry.context_types.TIME_RANGE,
                success_criteria="Explicit time range validated and converted to datetime objects",
                inputs=[],
            ),
            scenario_description="Parsing explicit time ranges in YYYY-MM-DD HH:MM:SS format",
            notes=f"Output stored under {registry.context_types.TIME_RANGE} context type. Validates and converts user-provided time ranges to datetime objects",
        )

        implicit_time_example = OrchestratorExample(
            step=PlannedStep(
                context_key="current_data_timerange",
                capability="time_range_parsing",
                task_objective="Infer appropriate time range for current beam energy data request (last 5 minutes)",
                expected_output=registry.context_types.TIME_RANGE,
                success_criteria="Appropriate time range inferred and converted to datetime objects",
                inputs=[],
            ),
            scenario_description="Inferring time ranges for 'current' or 'recent' data requests",
            notes=f"Output stored under {registry.context_types.TIME_RANGE} context type. Provides sensible defaults (e.g., last few minutes) as datetime objects",
        )

        return OrchestratorGuide(
            instructions=textwrap.dedent(
                f"""
                **When to plan "time_range_parsing" steps:**
                - When tasks require time-based data (historical trends, archiver data, logs)
                - When user queries contain time references that need to be converted to absolute datetime objects
                - As a prerequisite step before archiver data retrieval or time-based analysis

                **Step Structure:**
                - context_key: Unique identifier for output (e.g., "last_week_timerange", "explicit_timerange")
                - task_objective: The specific and self-contained time range parsing task to perform

                **Output: {registry.context_types.TIME_RANGE}**
                - Contains: start_date and end_date as datetime objects with full datetime functionality
                - Available to downstream steps via context system
                - Supports datetime arithmetic, comparison, and formatting operations

                **Time Pattern Support:**
                - Relative: "last X hours/minutes/days", "yesterday", "this week", "last week"
                - Absolute: "from YYYY-MM-DD HH:MM:SS to YYYY-MM-DD HH:MM:SS"
                - Implicit: "current", "recent" (defaults to last few minutes)

                **Dependencies and sequencing:**
                1. This step typically comes early when time-based data operations are needed
                2. Results feed into subsequent data retrieval capabilities that require time ranges
                3. Uses LLM to handle complex relative time references and natural language time expressions
                4. Downstream steps can use datetime objects directly without string parsing

                ALWAYS plan this step when any time-based data operations are needed,
                regardless of whether the user provides explicit time ranges or relative time descriptions.
                """
            ),
            examples=[relative_time_example, absolute_time_example, implicit_time_example],
            priority=5,
        )

    def get_classifier_guide(self) -> TaskClassifierGuide | None:
        """Create classifier guide for time range parsing."""
        return TaskClassifierGuide(
            instructions="Determine if the task involves time-based data requests that require parsing time ranges from user queries.",
            examples=[
                ClassifierExample(
                    query="Which tools do you have?",
                    result=False,
                    reason="This is a question about AI capabilities, no time range needed.",
                ),
                ClassifierExample(
                    query="Plot the beam current for the last 2 hours",
                    result=True,
                    reason="Request involves time range ('last 2 hours') that needs parsing.",
                ),
                ClassifierExample(
                    query="What is the current beam energy?",
                    result=False,
                    reason="Request is for current value, no time range needed.",
                ),
                ClassifierExample(
                    query="Show me vacuum trends from yesterday",
                    result=True,
                    reason="Request involves time range ('yesterday') that needs parsing.",
                ),
                ClassifierExample(
                    query="Get historical data from 2024-01-15 09:00:00 to 2024-01-15 21:00:00",
                    result=True,
                    reason="Request has explicit time range that needs parsing and validation.",
                ),
                ClassifierExample(
                    query="How does the accelerator work?",
                    result=False,
                    reason="This is a general question about accelerator principles, no time data needed.",
                ),
                ClassifierExample(
                    query="Show recent trends",
                    result=True,
                    reason="Request involves implicit time range ('recent') that needs parsing.",
                ),
                ClassifierExample(
                    query="Show me some data",
                    result=False,
                    reason="Request does not involve time range.",
                ),
            ],
            actions_if_true=ClassifierActions(),
        )
