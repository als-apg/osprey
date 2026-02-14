"""
Time Range Parsing Prompt Builder

Default prompts for time range parsing capability.
"""

import textwrap
from datetime import UTC, datetime, timedelta

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
    """Default time range parsing prompt builder.

    **Customization Points:**

    +---------------------------------+----------------------------------------------+
    | I want to...                    | Override...                                  |
    +=================================+==============================================+
    | Change the agent identity       | ``get_role()``                    |
    +---------------------------------+----------------------------------------------+
    | Change the task statement       | ``get_task()``                    |
    +---------------------------------+----------------------------------------------+
    | Change parsing instructions     | ``get_instructions()``                       |
    +---------------------------------+----------------------------------------------+
    | Change orchestrator guidance    | ``get_orchestrator_guide()``                 |
    +---------------------------------+----------------------------------------------+
    | Change classifier guidance      | ``get_classifier_guide()``                   |
    +---------------------------------+----------------------------------------------+
    """

    PROMPT_TYPE = "time_range_parsing"

    def get_role(self) -> str:
        """Get the role definition for time range parsing."""
        return "You are an expert time range parser that converts natural language time expressions into precise datetime ranges."

    def get_task(self) -> str:
        """Get the task definition for time range parsing."""
        return "TASK: Parse time references from user queries and convert them to absolute datetime ranges."

    def get_instructions(self) -> str:
        """Get the instructions for time range parsing.

        Contains the full static content for time parsing: critical requirements,
        parsing steps, common patterns, and calculation rules. Dynamic content
        (computed example dates, user query) is injected via build_dynamic_context().
        """
        return textwrap.dedent(
            """\
            CRITICAL REQUIREMENTS:
            - start_date and end_date must be valid datetime values in ISO format
            - Use format 'YYYY-MM-DD HH:MM:SS'
            - Return as datetime objects, not strings with extra text or descriptions
            - **CRITICAL**: start_date MUST be BEFORE end_date (start < end)
            - **CRITICAL**: Anchor ALL date calculations to the current datetime provided below — do NOT use your training data to infer what "now" is
            - For historical data requests, end_date should typically be close to current time

            Instructions:
            1. Parse the user query to identify time range references
            2. Convert relative time references to absolute datetime values
            3. Set found=true if you can identify a time range, found=false if no time reference exists
            4. If found=false, use current time for both start_date and end_date as placeholders

            Common patterns and their conversions:
            - "last X hours/minutes/days" → X time units BEFORE current time to NOW
            - "past X hours/minutes/days" → X time units BEFORE current time to NOW
            - "yesterday" → previous day from 00:00:00 to 23:59:59
            - "today" → current day from 00:00:00 to current time
            - "this week" → from start of current week to now
            - "last week" → previous week (Monday to Sunday)
            - Current/real-time requests → very recent time (last few minutes)

            CRITICAL CALCULATION RULES FOR RELATIVE TIMES:
            **STEP-BY-STEP for "past/last X hours":**
            1. start_date = current_time MINUS X hours (earlier time)
            2. end_date = current_time (later time)
            3. Verify: start_date < end_date

            **STEP-BY-STEP for "past/last X days":**
            1. start_date = current_time MINUS X days (earlier time)
            2. end_date = current_time (later time)
            3. Verify: start_date < end_date

            Respond with a JSON object containing start_date, end_date, and found.
            The start_date and end_date fields should be datetime values in YYYY-MM-DD HH:MM:SS format
            that will be automatically converted to Python datetime objects."""
        )

    def build_dynamic_context(self, **kwargs) -> str | None:
        """Inject runtime time context: current datetime, computed examples, and user query.

        This method computes the dynamic portions of the time parsing prompt that
        depend on the current time (example dates, calculation demonstrations) and
        appends the user query for parsing.

        Keyword Args:
            user_query: The user's natural language query containing time references.
        """
        now = datetime.now(UTC)
        current_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        current_weekday = now.strftime("%A")

        # Calculate example dates
        two_hours_ago = (now - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
        yesterday_start = (now - timedelta(days=1)).strftime("%Y-%m-%d") + " 00:00:00"
        yesterday_end = (now - timedelta(days=1)).strftime("%Y-%m-%d") + " 23:59:59"
        twenty_four_hours_ago = (now - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
        two_weeks_ago = (now - timedelta(days=14)).strftime("%Y-%m-%d %H:%M:%S")

        sections = []

        # Current time context
        sections.append(
            textwrap.dedent(f"""\
            Current time context:
            - Current datetime: {current_time_str}
            - Current weekday: {current_weekday}""")
        )

        # Example calculation
        sections.append(
            textwrap.dedent(f"""\
            **EXAMPLE CALCULATION for "past 24 hours" when current time is {current_time_str}:**
            1. start_date = {current_time_str} - 24 hours = {twenty_four_hours_ago}
            2. end_date = {current_time_str}
            3. Check: {twenty_four_hours_ago} < {current_time_str} ✓""")
        )

        # Examples with exact format
        sections.append(
            textwrap.dedent(f"""\
            EXAMPLES with exact format expected:
            - "last 2 hours" → start_date: "{two_hours_ago}", end_date: "{current_time_str}"
            - "yesterday" → start_date: "{yesterday_start}", end_date: "{yesterday_end}"
            - "last 24 hours" → start_date: "{twenty_four_hours_ago}", end_date: "{current_time_str}"
            - "past 24 hours" → start_date: "{twenty_four_hours_ago}", end_date: "{current_time_str}"
            - "past 2 weeks" → start_date: "{two_weeks_ago}", end_date: "{current_time_str}" """)
        )

        # User query
        user_query = kwargs.get("user_query")
        if user_query:
            sections.append(f"User query to parse: {user_query}")

        return "\n\n".join(sections)

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
