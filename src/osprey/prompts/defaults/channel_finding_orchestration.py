"""Channel Finding Orchestration Prompt Builder.

Provides orchestrator and classifier guidance for the ChannelFindingCapability.
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


class DefaultChannelFindingOrchestrationPromptBuilder(FrameworkPromptBuilder):
    """Default channel finding orchestration prompt builder.

    **Customization Points:**

    +---------------------------------+----------------------------------------------+
    | I want to...                    | Override...                                  |
    +=================================+==============================================+
    | Change the agent identity       | ``get_role()``                    |
    +---------------------------------+----------------------------------------------+
    | Change finding instructions     | ``get_instructions()``                       |
    +---------------------------------+----------------------------------------------+
    | Change orchestrator guidance    | ``get_orchestrator_guide()``                 |
    +---------------------------------+----------------------------------------------+
    | Change classifier guidance      | ``get_classifier_guide()``                   |
    +---------------------------------+----------------------------------------------+
    """

    PROMPT_TYPE = "channel_finding_orchestration"

    def get_role(self) -> str:
        """Return role definition for channel finding orchestration."""
        return "You are a channel address finder for control system parameters."

    def get_instructions(self) -> str:
        """Return instructions for channel finding orchestration."""
        return "Find and resolve channel addresses from natural language descriptions."

    def get_orchestrator_guide(self) -> OrchestratorGuide | None:
        """Create orchestrator guide for channel finding planning."""
        natural_language_example = OrchestratorExample(
            step=PlannedStep(
                context_key="beam_current_channels",
                capability="channel_finding",
                task_objective="Find channel addresses for beam current measurement",
                expected_output="CHANNEL_ADDRESSES",
                success_criteria="Relevant channel addresses found and validated",
                inputs=[],
            ),
            scenario_description="Natural language search for measurement types",
            notes="Output stored under CHANNEL_ADDRESSES context type.",
        )

        system_location_example = OrchestratorExample(
            step=PlannedStep(
                context_key="quadrupole_channels",
                capability="channel_finding",
                task_objective="Find channel addresses for quadrupole magnet currents in all sectors",
                expected_output="CHANNEL_ADDRESSES",
                success_criteria="System-specific channel addresses located",
                inputs=[],
            ),
            scenario_description="System and location-based channel discovery",
            notes="Output stored under CHANNEL_ADDRESSES context type.",
        )

        return OrchestratorGuide(
            instructions=textwrap.dedent("""
                **When to plan "channel_finding" steps:**
                - When the user mentions hardware, measurements, sensors, or devices without providing specific channel addresses
                - When fuzzy or descriptive names need to be resolved to exact channel addresses
                - As a prerequisite step before channel value retrieval or data analysis
                - When users reference systems or locations but not complete channel names

                **Step Structure:**
                - context_key: Unique identifier for output (e.g., "beam_current_channels", "validated_channels")
                - task_objective: The specific and self-contained channel address search task to perform

                **Output: CHANNEL_ADDRESSES**
                - Contains: List of channel addresses with description
                - Available to downstream steps via context system

                **Dependencies and sequencing:**
                - This step typically comes first when channel addresses are needed
                - Results feed into subsequent "channel_read", "channel_write", or "archiver_retrieval" steps

                ALWAYS plan this step when any channel-related operations are needed.
                """),
            examples=[natural_language_example, system_location_example],
            priority=1,
        )

    def get_classifier_guide(self) -> TaskClassifierGuide | None:
        """Create classifier guide for channel finding activation."""
        return TaskClassifierGuide(
            instructions="Determine if the task involves finding, extracting, or identifying channel addresses. This applies if the user is searching for channels based on descriptions, OR if they need any channel-related operations.",
            examples=[
                ClassifierExample(
                    query="Which tools do you have?",
                    result=False,
                    reason="This is a question about the AI's capabilities.",
                ),
                ClassifierExample(
                    query="Find channels related to beam position monitors.",
                    result=True,
                    reason="The query asks to find channels based on a description.",
                ),
                ClassifierExample(
                    query="I need the channel for the vacuum pressure.",
                    result=True,
                    reason="The query asks to find a channel based on a description.",
                ),
                ClassifierExample(
                    query="Can you plot the beam current for the last hour?",
                    result=True,
                    reason="The query asks to plot data, which requires channel finding first.",
                ),
                ClassifierExample(
                    query="What's the beam current right now?",
                    result=True,
                    reason="The query asks for a value without a specific channel address, requiring channel finding first.",
                ),
                ClassifierExample(
                    query="Set the main quad current to 5 amps.",
                    result=True,
                    reason="Setting a value requires finding the correct channel address first.",
                ),
            ],
            actions_if_true=ClassifierActions(),
        )


__all__ = ["DefaultChannelFindingOrchestrationPromptBuilder"]
