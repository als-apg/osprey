"""Channel Write Prompt Builder.

Provides orchestrator and classifier guidance for the ChannelWriteCapability.
"""

from __future__ import annotations

import textwrap

from osprey.base.examples import (
    ClassifierActions,
    ClassifierExample,
    OrchestratorExample,
    OrchestratorGuide,
    PlannedStep,
    TaskClassifierGuide,
)
from osprey.prompts.base import FrameworkPromptBuilder


class DefaultChannelWritePromptBuilder(FrameworkPromptBuilder):
    """Default channel write prompt builder.

    **Customization Points:**

    +---------------------------------+----------------------------------------------+
    | I want to...                    | Override...                                  |
    +=================================+==============================================+
    | Change the agent identity       | ``get_role()``                    |
    +---------------------------------+----------------------------------------------+
    | Change write instructions       | ``get_instructions()``                       |
    +---------------------------------+----------------------------------------------+
    | Change orchestrator guidance    | ``get_orchestrator_guide()``                 |
    +---------------------------------+----------------------------------------------+
    | Change classifier guidance      | ``get_classifier_guide()``                   |
    +---------------------------------+----------------------------------------------+
    """

    PROMPT_TYPE = "channel_write"

    def get_role(self) -> str:
        """Return role definition for channel write."""
        return "You are an expert at parsing control system write operations."

    def get_instructions(self) -> str:
        """Return instructions for channel write parsing.

        Contains the full static content for write parsing: the job description,
        value extraction examples, and critical rules. Dynamic content (task
        objective, channel mapping, available data) is injected via
        build_dynamic_context().
        """
        return textwrap.dedent(
            """\
            YOUR JOB:
            1. Identify which channels to write from the task
            2. Extract the value for each channel from:
               - Direct specification in task ("set X to 5")
               - Values in the available data above (computed results, statistics, etc.)
            3. Match channels using the semantic links shown above:
               - Use the original query ("beam energy", "HCM01", etc.) to find the right channel
               - The mapping shows: "original query" → CHANNEL:ADDRESS
            4. Return structured write operations with the exact CHANNEL:ADDRESS from the mapping

            VALUE EXTRACTION EXAMPLES:

            Example 1 - Direct values with semantic matching:
            Task: "Set HCM01 to 5.0 and HCM02 to -3.2"
            Available channels:
              "horizontal corrector HCM01" → HCM01:CURRENT:SP
              "horizontal corrector HCM02" → HCM02:CURRENT:SP
            Response:
            {{
                "write_operations": [
                    {{"channel_address": "HCM01:CURRENT:SP", "value": 5.0}},
                    {{"channel_address": "HCM02:CURRENT:SP", "value": -3.2}}
                ]
            }}

            Example 1b - Channel name with extra words:
            Task: "Set the TerminalVoltageSetPoint parameter to 50"
            Available channels:
              "terminal voltage" → TerminalVoltageSetPoint
            Response:
            {{
                "write_operations": [
                    {{"channel_address": "TerminalVoltageSetPoint", "value": 50.0}}
                ]
            }}
            Note: Task mentions "TerminalVoltageSetPoint" which matches the original query "terminal voltage".

            Example 2 - Extract from PYTHON_RESULTS:
            Task: "Set the magnet to the calculated optimal value"
            Available channels:
              "horizontal corrector magnet" → HCM01:CURRENT:SP
            Available data shows:
            PYTHON_RESULTS: {{"computed_results": {{"optimal_current": 12.5}}}}
            Response:
            {{
                "write_operations": [
                    {{"channel_address": "HCM01:CURRENT:SP", "value": 12.5,
                      "notes": "Using optimal_current from calculation"}}
                ]
            }}

            Example 3 - Value requires computation (return EMPTY list):
            Task: "Set magnet to sqrt(165)"
            Available channels:
              "corrector magnet" → HCM01:CURRENT:SP
            Available data: (none)
            Response:
            {{
                "write_operations": []
            }}
            Note: The value requires computation. Do NOT compute it yourself.
            Return an empty list so the orchestrator can plan a computation step first.

            Example 4 - Task references data not in context (return EMPTY list):
            Task: "Set corrector to yesterday's average current"
            Available channels:
              "horizontal corrector" → HCM01:CURRENT:SP
            Available data: (none)
            Response:
            {{
                "write_operations": []
            }}
            Note: "Yesterday's average current" refers to historical data, but no
            ARCHIVER_DATA is present in the available data. Do NOT guess or invent
            a value. Return an empty list so the orchestrator can retrieve the data first.

            Example 5 - Extract value from available context data:
            Task: "Set corrector to yesterday's average current"
            Available channels:
              "horizontal corrector" → HCM01:CURRENT:SP
            Available data shows:
            ARCHIVER_DATA: {{"channel_data": {{"HCM01:CURRENT:RB": {{"statistics": {{"mean_value": 8.5}}}}}}}}
            Response:
            {{
                "write_operations": [
                    {{"channel_address": "HCM01:CURRENT:SP", "value": 8.5,
                      "notes": "Using mean from historical data"}}
                ]
            }}
            Note: Same task as Example 4, but now ARCHIVER_DATA IS present in the
            available data, so the value can be extracted.

            CRITICAL RULES:
            - NEVER compute, calculate, or guess values. Only use values that are
              either stated literally in the task ("set X to 5") or present in the
              available data section above. If the value is not directly available,
              return an empty write_operations list.
            - Use the semantic mapping to match task references to channel addresses
            - Match task words ("beam energy", "magnet", "HCM01") to the original queries shown in the mapping
            - ALWAYS return the exact channel address (right side of →) in your response
            - Example: If task says "Set beam energy to 50" and mapping shows "beam energy" → SR:ENERGY:SP,
              then use "SR:ENERGY:SP" in your write_operations
            - All values must be numeric (float or int)
            - Extract units if mentioned in task (optional)
            - Add notes about value source if helpful (optional)

            IMPORTANT: Always use the exact channel address from the right side of the → mapping.

            Return JSON matching WriteOperationsOutput schema."""
        )

    def build_dynamic_context(self, **kwargs) -> str | None:
        """Inject runtime context: task objective, channel mapping, and available data.

        Keyword Args:
            task_objective: The write task description.
            channel_mapping: Formatted string of channel address mappings.
            available_data: Formatted string of available context data from previous steps.
        """
        sections = []

        task_objective = kwargs.get("task_objective")
        if task_objective:
            sections.append(f"TASK: {task_objective}")

        channel_mapping = kwargs.get("channel_mapping")
        if channel_mapping:
            sections.append(
                f"AVAILABLE CHANNEL ADDRESSES (with semantic links):\n{channel_mapping}"
            )

        available_data = kwargs.get("available_data")
        if available_data:
            sections.append(f"AVAILABLE DATA (from previous steps):\n{available_data}")

        return "\n\n".join(sections) if sections else None

    def get_orchestrator_guide(self) -> OrchestratorGuide | None:
        """Create orchestrator guide for channel write planning."""
        simple_write_example = OrchestratorExample(
            step=PlannedStep(
                context_key="write_setpoint_value",
                capability="channel_write",
                task_objective="Set the beam energy setpoint to 50 GeV",
                expected_output="CHANNEL_WRITE_RESULTS",
                success_criteria="Setpoint value successfully written",
                inputs=[{"CHANNEL_ADDRESSES": "energy_setpoint_address"}],
            ),
            scenario_description="Writing a single setpoint value",
            notes="Output stored under CHANNEL_WRITE_RESULTS context type. Can use additional inputs like PYTHON_RESULTS for calculated values.",
        )

        calculated_write_example = OrchestratorExample(
            step=PlannedStep(
                context_key="write_optimized_current",
                capability="channel_write",
                task_objective="Set the corrector magnet to the calculated optimal current value",
                expected_output="CHANNEL_WRITE_RESULTS",
                success_criteria="Optimal current value successfully written",
                inputs=[
                    {"CHANNEL_ADDRESSES": "corrector_address"},
                    {"PYTHON_RESULTS": "optimization_calculation"},
                ],
            ),
            scenario_description="Writing a calculated value from Python step",
            notes="This capability automatically extracts values from PYTHON_RESULTS context. Supports any context type the orchestrator provides.",
        )

        return OrchestratorGuide(
            instructions=textwrap.dedent("""
            **When to plan "channel_write" steps:**
            - When user requests to SET, CHANGE, or WRITE channel values
            - For direct value assignment to control system parameters
            - This capability handles parsing internally - just provide the task and relevant contexts

            **Context-Agnostic Value Extraction:**
            - This capability can extract values from ANY context type you provide in inputs
            - Common patterns:
              * Task only: "Set HCM01 to 5.0" (direct value in task)
              * Task + PYTHON_RESULTS: "Set magnet to calculated value" (extracts from Python results)
              * Task + ARCHIVER_DATA: "Set to yesterday's average" (extracts from statistics)
              * Any combination of contexts

            **Step Structure:**
            - context_key: Unique identifier for output (e.g., "write_magnet_current")
            - task_objective: Clear description of what to write (include values if direct, or reference to context data)
            - inputs: ALWAYS include CHANNEL_ADDRESSES, optionally include other contexts with values:
              [{"CHANNEL_ADDRESSES": "channel_step"}, {"PYTHON_RESULTS": "calc_step"}]

            **Required Inputs:**
            - CHANNEL_ADDRESSES: Required (from channel_finding step)
            - Optional: PYTHON_RESULTS, CHANNEL_VALUES, ARCHIVER_DATA, or any other context with values

            **Output: CHANNEL_WRITE_RESULTS**
            - Contains: Dictionary mapping channel addresses to write results (success/failure)
            - Available to downstream steps via context system

            **Dependencies and sequencing:**
            1. Channel finding step must precede this step
            2. If task requires calculation, add Python step before channel_write
            3. If task references historical data, add archiver_retrieval before channel_write

            **IMPORTANT Safety Notes:**
            - Write operations may trigger approval workflows
            - Limits checking may reject writes outside configured limits
            - This capability uses LLM to parse values from task and available contexts

            **NEVER** plan steps that would require making up channel addresses or values.
            """).strip(),
            examples=[simple_write_example, calculated_write_example],
            priority=10,
        )

    def get_classifier_guide(self) -> TaskClassifierGuide | None:
        """Create classifier for channel write capability."""
        return TaskClassifierGuide(
            instructions="Determine if task requires writing channel values. Look for SET, CHANGE, WRITE, ADJUST keywords. This capability handles both parsing and execution internally.",
            examples=[
                ClassifierExample(
                    query="Which tools do you have?",
                    result=False,
                    reason="Question about capabilities, not writing.",
                ),
                ClassifierExample(
                    query="Set the beam energy to 50 GeV",
                    result=True,
                    reason="Direct write request.",
                ),
                ClassifierExample(
                    query="Change the corrector magnet current to 2.5 A",
                    result=True,
                    reason="Write request.",
                ),
                ClassifierExample(
                    query="Set the magnet to the calculated optimal value",
                    result=True,
                    reason="Write request using calculated value (requires Python step first).",
                ),
                ClassifierExample(
                    query="What is the current beam energy?",
                    result=False,
                    reason="Reading value, not writing.",
                ),
                ClassifierExample(
                    query="Set magnet A to 5 and magnet B to 3",
                    result=True,
                    reason="Multiple write operations.",
                ),
                ClassifierExample(
                    query="Adjust correctors to compensate for drift",
                    result=True,
                    reason="Adjustment implies writing new values.",
                ),
            ],
            actions_if_true=ClassifierActions(),
        )


__all__ = ["DefaultChannelWritePromptBuilder"]
