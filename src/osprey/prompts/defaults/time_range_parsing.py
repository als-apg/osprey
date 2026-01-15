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

        # Define structured examples using descriptive context keys
        relative_time_example = OrchestratorExample(
            step=PlannedStep(
                context_key="tr_0103_0110",  # Descriptive: Jan 3 to Jan 10
                capability="time_range_parsing",
                task_objective="Parse 'last week' time reference into absolute datetime objects",
                expected_output=registry.context_types.TIME_RANGE,
                success_criteria="Time range successfully parsed to absolute datetime objects",
                inputs=[],
            ),
            scenario_description="Parsing relative time references like 'last week' (resolved to actual dates)",
            notes="Context key encodes the resolved dates (tr_MMDD_MMDD format) for easy comparison with new requests.",
        )

        absolute_time_example = OrchestratorExample(
            step=PlannedStep(
                context_key="tr_0115_9h_21h",  # Descriptive: Jan 15, 9am to 9pm
                capability="time_range_parsing",
                task_objective="Parse explicit datetime range '2024-01-15 09:00:00 to 2024-01-15 21:00:00' and validate format",
                expected_output=registry.context_types.TIME_RANGE,
                success_criteria="Explicit time range validated and converted to datetime objects",
                inputs=[],
            ),
            scenario_description="Parsing explicit time ranges (same-day with hours)",
            notes="For same-day ranges, include hours: tr_MMDD_Hh_Hh format.",
        )

        implicit_time_example = OrchestratorExample(
            step=PlannedStep(
                context_key="tr_recent",  # Descriptive: recent/current data
                capability="time_range_parsing",
                task_objective="Infer appropriate time range for current beam energy data request (last 5 minutes)",
                expected_output=registry.context_types.TIME_RANGE,
                success_criteria="Appropriate time range inferred and converted to datetime objects",
                inputs=[],
            ),
            scenario_description="Inferring time ranges for 'current' or 'recent' data requests",
            notes="Use 'tr_recent' or 'tr_now' for current/real-time requests.",
        )

        return OrchestratorGuide(
            instructions=textwrap.dedent(
                f"""
                **When to plan "time_range_parsing" steps:**
                - When tasks require time-based data (historical trends, archiver data, logs)
                - When user queries contain time references that need to be converted to absolute datetime objects
                - As a prerequisite step before archiver data retrieval or time-based analysis

                **Context Key Format (CRITICAL for context reuse):**
                Use descriptive keys encoding the actual dates for easy comparison:
                - Date ranges: tr_MMDD_MMDD (e.g., "12/5 to 12/10" → "tr_1205_1210")
                - Same-day with hours: tr_MMDD_Hh_Hh (e.g., "Dec 5 9am-5pm" → "tr_1205_9h_17h")
                - Current/recent: tr_recent or tr_now
                - Cross-month: tr_1128_0103 (Nov 28 to Jan 3)

                This enables the orchestrator to compare new requests against existing context keys
                and determine if a new time_range_parsing step is needed.

                **Output: {registry.context_types.TIME_RANGE}**
                - Contains: start_date and end_date as datetime objects
                - Available to downstream steps via context system

                **Time Pattern Support:**
                - Relative: "last X hours/days", "yesterday", "this week" → resolve to actual dates in key
                - Absolute: "from YYYY-MM-DD to YYYY-MM-DD" → encode dates in key
                - Implicit: "current", "recent" → use tr_recent or tr_now

                **Dependencies and sequencing:**
                1. This step typically comes early when time-based data operations are needed
                2. Results feed into subsequent data retrieval capabilities that require time ranges
                3. Downstream steps can use datetime objects directly without string parsing

                ALWAYS plan this step when time-based data operations are needed.
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
