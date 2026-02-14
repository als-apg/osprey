"""Archiver Retrieval Prompt Builder.

Provides orchestrator and classifier guidance for the ArchiverRetrievalCapability.
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


class DefaultArchiverRetrievalPromptBuilder(FrameworkPromptBuilder):
    """Default archiver retrieval prompt builder.

    **Customization Points:**

    +---------------------------------+----------------------------------------------+
    | I want to...                    | Override...                                  |
    +=================================+==============================================+
    | Change the agent identity       | ``get_role()``                    |
    +---------------------------------+----------------------------------------------+
    | Change retrieval instructions   | ``get_instructions()``                       |
    +---------------------------------+----------------------------------------------+
    | Change orchestrator guidance    | ``get_orchestrator_guide()``                 |
    +---------------------------------+----------------------------------------------+
    | Change classifier guidance      | ``get_classifier_guide()``                   |
    +---------------------------------+----------------------------------------------+
    """

    PROMPT_TYPE = "archiver_retrieval"

    def get_role(self) -> str:
        """Return role definition for archiver retrieval."""
        return "You are a historical data retriever for control system archives."

    def get_instructions(self) -> str:
        """Return instructions for archiver retrieval."""
        return "Retrieve historical time-series data from the control system archiver."

    def get_orchestrator_guide(self) -> OrchestratorGuide | None:
        """Create orchestrator guide for archiver retrieval."""
        archiver_retrieval_example = OrchestratorExample(
            step=PlannedStep(
                context_key="historical_beam_current_data",
                capability="archiver_retrieval",
                task_objective="Retrieve historical beam current data from archiver for the last 24 hours",
                expected_output="ARCHIVER_DATA",
                success_criteria="Historical data retrieved successfully for specified time range",
                inputs=[
                    {"CHANNEL_ADDRESSES": "beam_current_channels"},
                    {"TIME_RANGE": "last_24_hours_timerange"},
                ],
            ),
            scenario_description="Retrieve historical time-series data from the archiver",
            notes="Requires channel addresses and time range from previous steps. Output stored under ARCHIVER_DATA context type. Optional parameter: precision_ms (default: 1000).",
        )

        plotting_workflow_python_step = OrchestratorExample(
            step=PlannedStep(
                context_key="beam_current_plot",
                capability="python",
                task_objective="Create a matplotlib time-series plot of the beam current data showing trends over the 24-hour period",
                expected_output="PYTHON_RESULTS",
                success_criteria="Time-series plot created with proper labels, showing beam current trends",
                inputs=[{"ARCHIVER_DATA": "historical_beam_current_data"}],
            ),
            scenario_description="WORKFLOW: Use python capability to plot archiver data from previous step",
            notes="Typical plotting workflow: archiver_retrieval (gets data) → python (creates plot) → respond (delivers to user). The python capability consumes the ARCHIVER_DATA from the archiver_retrieval step.",
        )

        analysis_workflow_python_step = OrchestratorExample(
            step=PlannedStep(
                context_key="beam_current_statistics",
                capability="python",
                task_objective="Calculate mean, standard deviation, min, and max values of the beam current data over the time period",
                expected_output="PYTHON_RESULTS",
                success_criteria="Statistical metrics calculated and displayed with clear labels",
                inputs=[{"ARCHIVER_DATA": "historical_beam_current_data"}],
            ),
            scenario_description="WORKFLOW: Use python capability to analyze archiver data from previous step",
            notes="Typical analysis workflow: archiver_retrieval (gets data) → python (calculates statistics) → respond (delivers results). The python capability consumes the ARCHIVER_DATA from the archiver_retrieval step.",
        )

        return OrchestratorGuide(
            instructions=textwrap.dedent("""
                **When to plan "archiver_retrieval" steps:**
                - When tasks require historical channel data
                - When retrieving past values from the archiver
                - When time-series data is needed from archived sources

                **Step Structure:**
                - context_key: Unique identifier for output (e.g., "historical_data", "trend_data")
                - inputs: Specify required inputs:
                {"CHANNEL_ADDRESSES": "context_key_with_channel_data", "TIME_RANGE": "context_key_with_time_range"}

                **Required Inputs:**
                - CHANNEL_ADDRESSES data: typically from a "channel_finding" step
                - TIME_RANGE data: typically from a "time_range_parsing" step

                **Input flow and sequencing:**
                1. "channel_finding" step must precede this step (if CHANNEL_ADDRESSES data is not present already)
                2. "time_range_parsing" step must precede this step (if TIME_RANGE data is not present already)

                **Output: ARCHIVER_DATA**
                - Contains: Structured historical data from the archiver
                - Available to downstream steps via context system

                **Common downstream workflow patterns:**
                - For plotting requests: archiver_retrieval → python (create plot) → respond
                - For analysis/statistics: archiver_retrieval → python (calculate stats) → respond
                - For complex analysis: archiver_retrieval → data_analysis → respond
                - Combined: archiver_retrieval → data_analysis → python (plot analysis) → respond

                Do NOT plan this for current values; use "channel_read" for real-time data.
                """),
            examples=[
                archiver_retrieval_example,
                plotting_workflow_python_step,
                analysis_workflow_python_step,
            ],
            priority=15,
        )

    def get_classifier_guide(self) -> TaskClassifierGuide | None:
        """Create classifier for archiver data capability."""
        return TaskClassifierGuide(
            instructions="Determines if the task requires accessing the archiver. This is relevant for requests involving historical data or trends.",
            examples=[
                ClassifierExample(
                    query="Which tools do you have?",
                    result=False,
                    reason="This is a question about the AI's capabilities.",
                ),
                ClassifierExample(
                    query="Plot the historical data for vacuum pressure for the last week.",
                    result=True,
                    reason="The query explicitly asks for historical data plotting.",
                ),
                ClassifierExample(
                    query="What is the current beam energy?",
                    result=False,
                    reason="The query asks for a current value, not historical data.",
                ),
                ClassifierExample(
                    query="Can you plot that over the last 4h?",
                    result=True,
                    reason="The query asks for historical data plotting.",
                ),
                ClassifierExample(
                    query="What was that value yesterday?",
                    result=True,
                    reason="The query asks for historical data.",
                ),
            ],
            actions_if_true=ClassifierActions(),
        )


__all__ = ["DefaultArchiverRetrievalPromptBuilder"]
