"""Channel Read Prompt Builder.

Provides orchestrator and classifier guidance for the ChannelReadCapability.
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


class DefaultChannelReadPromptBuilder(FrameworkPromptBuilder):
    """Default channel read prompt builder.

    **Customization Points:**

    +---------------------------------+----------------------------------------------+
    | I want to...                    | Override...                                  |
    +=================================+==============================================+
    | Change the agent identity       | ``get_role()``                    |
    +---------------------------------+----------------------------------------------+
    | Change read instructions        | ``get_instructions()``                       |
    +---------------------------------+----------------------------------------------+
    | Change orchestrator guidance    | ``get_orchestrator_guide()``                 |
    +---------------------------------+----------------------------------------------+
    | Change classifier guidance      | ``get_classifier_guide()``                   |
    +---------------------------------+----------------------------------------------+
    """

    PROMPT_TYPE = "channel_read"

    def get_role(self) -> str:
        """Return role definition for channel read."""
        return "You are a channel value reader for control system parameters."

    def get_instructions(self) -> str:
        """Return instructions for channel read."""
        return "Read current values from control system channels."

    def get_orchestrator_guide(self) -> OrchestratorGuide | None:
        """Create orchestrator guide for channel read planning."""
        current_values_example = OrchestratorExample(
            step=PlannedStep(
                context_key="read_monitor_values",
                capability="channel_read",
                task_objective="Read current beam position monitor values for orbit analysis",
                expected_output="CHANNEL_VALUES",
                success_criteria="Position values successfully retrieved",
                inputs=[{"CHANNEL_ADDRESSES": "monitor_addresses"}],
            ),
            scenario_description="Reading beam monitor values for real-time monitoring",
            notes="Output stored under CHANNEL_VALUES context type. Requires CHANNEL_ADDRESSES input from previous step.",
        )

        status_check_example = OrchestratorExample(
            step=PlannedStep(
                context_key="system_status_values",
                capability="channel_read",
                task_objective="Check current values of critical power supply and magnet parameters for system health assessment",
                expected_output="CHANNEL_VALUES",
                success_criteria="Status values retrieved and within expected ranges",
                inputs=[{"CHANNEL_ADDRESSES": "critical_system_channels"}],
            ),
            scenario_description="Status checking for critical system parameters",
            notes="Output stored under CHANNEL_VALUES context type. Used for system health checks.",
        )

        return OrchestratorGuide(
            instructions=textwrap.dedent("""
            **When to plan "channel_read" steps:**
            - When the user requests current status, readings, or values of specific channels
            - For real-time monitoring of control system parameters
            - When current channel values are needed as inputs for subsequent steps

            **Step Structure:**
            - context_key: Unique identifier for output (e.g., "current_beam_values", "status_readings")
            - inputs: Specify required inputs as context type to context key mappings:
              {"CHANNEL_ADDRESSES": "context_key_with_channel_addresses"}

            **Required Inputs:**
            - CHANNEL_ADDRESSES data: typically from a "channel_finding" step

            **Output: CHANNEL_VALUES**
            - Contains: Dictionary mapping channel addresses to their current values and timestamps
            - Available to downstream steps via context system

            **Dependencies and sequencing:**
            1. Channel finding step must precede this step if CHANNEL_ADDRESSES data is not present already
            2. Channel values can serve as inputs for analysis or visualization steps
            3. Returns current values with timestamps for real-time monitoring

            Do NOT plan this for historical data; use "archiver_retrieval" for historical data.

            **NEVER** plan steps that would require making up channel addresses - always ensure addresses are obtained from previous steps.
            """).strip(),
            examples=[current_values_example, status_check_example],
            priority=10,
        )

    def get_classifier_guide(self) -> TaskClassifierGuide | None:
        """Create classifier for channel read capability."""
        return TaskClassifierGuide(
            instructions="Determine if the task requires fetching current channel values. Look for requests about current values, statuses, or readings of specific channels.",
            examples=[
                ClassifierExample(
                    query="Which tools do you have?",
                    result=False,
                    reason="This is a question about the AI's capabilities.",
                ),
                ClassifierExample(
                    query="What is the current beam energy?",
                    result=True,
                    reason="The query asks for a current value.",
                ),
                ClassifierExample(
                    query="Read the current magnet current.",
                    result=True,
                    reason="The query asks to read the current value.",
                ),
                ClassifierExample(
                    query="What's the beam current right now?",
                    result=True,
                    reason="The query asks for a current value, which requires reading channel values.",
                ),
                ClassifierExample(
                    query="Show me historical beam current data.",
                    result=False,
                    reason="This is asking for historical data, not current values.",
                ),
            ],
            actions_if_true=ClassifierActions(),
        )


__all__ = ["DefaultChannelReadPromptBuilder"]
