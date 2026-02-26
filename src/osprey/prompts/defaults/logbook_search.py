"""Logbook Search Prompt Builder.

Provides orchestrator and classifier guidance for the LogbookSearchCapability.
This is a guide-only builder — the capability relays queries directly to the
ARIEL search service, so ``build_prompt()`` is never called.

See 04_OSPREY_INTEGRATION.md Section 7.4 for specification.
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


class DefaultLogbookSearchPromptBuilder(FrameworkPromptBuilder):
    """Guide-only prompt builder for logbook search.

    The logbook search capability relays queries to the ARIEL search service,
    which uses its own internal ReAct agent and RAG prompts. This builder only
    provides orchestrator and classifier guidance — ``build_prompt()``
    is never called at runtime.

    **Customization Points:**

    +---------------------------------+----------------------------------------------+
    | I want to...                    | Override...                                  |
    +=================================+==============================================+
    | Change orchestrator guidance    | ``get_orchestrator_guide()``                 |
    +---------------------------------+----------------------------------------------+
    | Change classifier guidance      | ``get_classifier_guide()``                   |
    +---------------------------------+----------------------------------------------+
    """

    PROMPT_TYPE = "logbook_search"

    def get_role(self) -> str:
        """Not used at runtime — logbook search delegates to ARIEL."""
        return "Logbook search (delegates to ARIEL service)."

    def get_instructions(self) -> str:
        """Not used at runtime — logbook search delegates to ARIEL."""
        return "Relay query to the ARIEL logbook search service."

    def get_orchestrator_guide(self) -> OrchestratorGuide | None:
        """Create orchestrator guide for logbook search."""
        try:
            from osprey.registry import get_registry

            registry = get_registry()
            context_type = registry.context_types.LOGBOOK_SEARCH_RESULTS
        except (ImportError, AttributeError):
            context_type = "LOGBOOK_SEARCH_RESULTS"

        semantic_example = OrchestratorExample(
            step=PlannedStep(
                context_key="ls_injector_failures",
                capability="logbook_search",
                task_objective="Search logbook for historical injector failure incidents",
                expected_output=context_type,
                success_criteria="Found relevant logbook entries about injector failures",
                inputs=[{"TIME_RANGE": "tr_recent"}],  # Optional dependency
            ),
            scenario_description="When user asks about past equipment failures or incidents",
            context_requirements={"TIME_RANGE": "Optional - parsed time range from user query"},
            notes="Works with or without TIME_RANGE context. Agent selects search strategy.",
        )

        keyword_example = OrchestratorExample(
            step=PlannedStep(
                context_key="ls_bts_chicane",
                capability="logbook_search",
                task_objective="Find logbook entries containing 'BTS chicane alignment'",
                expected_output=context_type,
                success_criteria="Retrieved entries matching keyword search",
                inputs=[],
            ),
            scenario_description="When user searches for specific equipment or procedure names",
            notes="Keyword search excels at finding specific technical terms.",
        )

        rag_example = OrchestratorExample(
            step=PlannedStep(
                context_key="ls_rf_trip_handling",
                capability="logbook_search",
                task_objective="Answer: How do we typically handle RF trips?",
                expected_output=context_type,
                success_criteria="Generated answer with citations from logbook entries",
                inputs=[],
            ),
            scenario_description="When user asks a question requiring synthesized knowledge",
            notes="RAG search retrieves context and generates cited answer.",
        )

        return OrchestratorGuide(
            instructions=textwrap.dedent(f"""
                **When to plan "logbook_search" steps:**
                - When user asks about facility history, past incidents, or equipment behavior
                - When user wants to find specific logbook entries by keyword or topic
                - When user asks questions about operational procedures documented in logs
                - As a knowledge source for troubleshooting or analysis tasks

                **Context Key Format:**
                Use descriptive keys encoding the search topic:
                - ls_<topic> (e.g., "ls_injector_failures", "ls_rf_trips")
                - ls_<equipment> (e.g., "ls_bts_chicane", "ls_vacuum_pump")

                **Output: {context_type}**
                - entries: Ranked list of matching logbook entries
                - answer: RAG-generated response (if question-style query)
                - sources: Entry IDs cited in the answer

                **Optional TIME_RANGE Input:**
                - If TIME_RANGE context exists, it will be used to filter results
                - If no TIME_RANGE, searches all available entries
                - Plan time_range_parsing BEFORE logbook_search when query has time references

                **Search Strategy Selection:**
                The internal ReAct agent automatically selects the best search strategy:
                - Keyword: For specific terms, equipment names, exact phrases
                - Semantic: For conceptual queries, related events
                - RAG: For questions needing synthesized answers
            """),
            examples=[semantic_example, keyword_example, rag_example],
            priority=15,
        )

    def get_classifier_guide(self) -> TaskClassifierGuide | None:
        """Create classifier guide for logbook search."""
        return TaskClassifierGuide(
            instructions="Determine if the task involves searching or querying facility logbooks for historical information.",
            examples=[
                ClassifierExample(
                    query="What happened last time the injector failed?",
                    result=True,
                    reason="Request for historical incident information from logbook.",
                ),
                ClassifierExample(
                    query="Find entries about BTS chicane alignment",
                    result=True,
                    reason="Explicit request to search logbook entries.",
                ),
                ClassifierExample(
                    query="What is the current beam energy?",
                    result=False,
                    reason="Request for live data, not historical logbook search.",
                ),
                ClassifierExample(
                    query="How do we typically handle RF trips?",
                    result=True,
                    reason="Question about operational knowledge documented in logs.",
                ),
                ClassifierExample(
                    query="Set the quadrupole current to 5A",
                    result=False,
                    reason="Control action request, not logbook search.",
                ),
                ClassifierExample(
                    query="Show me the beam loss events from January",
                    result=True,
                    reason="Request for historical logbook entries with time filter.",
                ),
                ClassifierExample(
                    query="What did the night shift report about the vacuum issue?",
                    result=True,
                    reason="Request for shift log information.",
                ),
                ClassifierExample(
                    query="How does the accelerator work?",
                    result=False,
                    reason="General question, not facility-specific logbook search.",
                ),
            ],
            actions_if_true=ClassifierActions(),
        )


__all__ = ["DefaultLogbookSearchPromptBuilder"]
