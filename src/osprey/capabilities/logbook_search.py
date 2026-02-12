"""LogbookSearchCapability - ARIEL Logbook Search Integration.

This capability bridges Osprey's agent orchestration layer with the ARIELSearchService.
It provides intelligent logbook search functionality by interfacing with the ARIEL
agentic retrieval system.

Key Features:
    - Natural language queries about facility history, equipment incidents
    - Automatic search strategy selection (keyword, semantic, RAG)
    - Optional time range filtering from TimeRangeParsingCapability
    - RAG-generated answers with source citations

See 04_OSPREY_INTEGRATION.md Section 7 for the full specification.
"""

from __future__ import annotations

import textwrap
from typing import Any, ClassVar

from osprey.base.capability import BaseCapability
from osprey.base.decorators import capability_node
from osprey.base.errors import ErrorClassification, ErrorSeverity
from osprey.base.examples import (
    ClassifierActions,
    ClassifierExample,
    OrchestratorExample,
    OrchestratorGuide,
    PlannedStep,
    TaskClassifierGuide,
)
from osprey.context.base import CapabilityContext
from osprey.prompts.loader import get_framework_prompts


class LogbookSearchResultsContext(CapabilityContext):
    """Search results from ARIEL logbook search.

    Provides structured context for logbook search results including matched
    entries, RAG-generated answers, and search metadata for downstream
    capabilities and response generation.

    Attributes:
        entries: Matching entries, ranked by relevance.
        answer: RAG-generated answer (if RAG was used).
        sources: Entry IDs cited in answer.
        search_modes_used: Search modes that were invoked (e.g., ["semantic", "rag"]).
        query: Original query text.
        time_range_applied: Whether time filter was used.
    """

    CONTEXT_TYPE: ClassVar[str] = "LOGBOOK_SEARCH_RESULTS"
    CONTEXT_CATEGORY: ClassVar[str] = "DATA"

    # Search results
    entries: tuple[dict, ...]  # EnhancedLogbookEntry dicts, ranked by relevance
    answer: str | None  # RAG-generated answer (if RAG was used)
    sources: tuple[str, ...]  # Entry IDs cited in answer

    # Search metadata
    search_modes_used: tuple[str, ...]  # e.g., ("semantic", "rag")
    query: str  # Original query
    time_range_applied: bool  # Whether time filter was used

    @property
    def context_type(self) -> str:
        """Return context type identifier."""
        return self.CONTEXT_TYPE

    def get_access_details(self, key: str) -> dict[str, Any]:
        """Provide access information for other capabilities."""
        return {
            "access_pattern": f"context.LOGBOOK_SEARCH_RESULTS.{key}",
            "data_structure": "Logbook entries with optional RAG answer",
            "fields": ["entries", "answer", "sources", "search_modes_used", "query"],
            "example_usage": f"context.LOGBOOK_SEARCH_RESULTS.{key}.answer gives RAG response",
        }

    def get_summary(self) -> dict[str, Any]:
        """Generate summary for response generation including actual content.

        Returns both metadata and actual search results/answers to enable
        the RespondCapability to generate meaningful user responses.
        Follows the pattern established by PythonResultContext.
        """
        from osprey.context.context_manager import recursively_summarize_data

        summary: dict[str, Any] = {
            "type": "Logbook Search Results",
            "query": self.query,
            "entries_found": len(self.entries),
            "search_modes": list(self.search_modes_used),
            "time_filtered": self.time_range_applied,
        }

        # Include RAG answer if available (primary content for response)
        if self.answer:
            summary["answer"] = self.answer
            summary["sources"] = list(self.sources)

        # Include summarized entries for additional context
        # Use recursively_summarize_data to prevent context overflow
        if self.entries:
            summary["entries"] = recursively_summarize_data(
                [
                    {
                        "entry_id": e.get("entry_id", ""),
                        "timestamp": (
                            e["timestamp"].isoformat()
                            if hasattr(e.get("timestamp"), "isoformat")
                            else str(e.get("timestamp", ""))
                        ),
                        "author": e.get("author", ""),
                        "summary": e.get("summary", ""),  # LLM-generated summary
                        "raw_text": (
                            e.get("raw_text", "")[:200] + "..."
                            if len(e.get("raw_text", "")) > 200
                            else e.get("raw_text", "")
                        ),
                    }
                    for e in self.entries
                ]
            )

        return summary


@capability_node
class LogbookSearchCapability(BaseCapability):
    """Search facility logbooks using ARIEL's agentic search.

    Provides intelligent logbook search functionality by interfacing with the
    ARIELSearchService. The capability handles natural language queries about
    facility history, equipment incidents, and operational knowledge recorded
    in electronic logbooks.

    The capability:
    1. Receives parsed TIME_RANGE context from TimeRangeParsingCapability (optional)
    2. Invokes ARIELSearchService with the user's query
    3. Returns LOGBOOK_SEARCH_RESULTS context for downstream capabilities

    Attributes:
        name: "logbook_search"
        provides: ["LOGBOOK_SEARCH_RESULTS"]
        requires: [] (TIME_RANGE is optional via soft constraint)
    """

    name = "logbook_search"
    description = (
        "Search and query historical logbook entries. Use when the user asks about "
        "past events, equipment history, operational incidents, or wants to find "
        "specific logbook entries by keyword or time period."
    )
    provides = ["LOGBOOK_SEARCH_RESULTS"]
    requires = []  # TIME_RANGE is optional, handled via soft constraint  # type: ignore[assignment]

    async def execute(self) -> dict[str, Any]:
        """Execute logbook search via ARIELSearchService."""
        logger = self.get_logger()

        # Extract TIME_RANGE if available (soft constraint - not required)
        contexts = self.get_required_contexts(constraint_mode="soft")
        time_range = contexts.get("TIME_RANGE")

        # Get query from task objective
        query = self.get_task_objective()

        logger.status(f"Searching logbooks: {query[:50]}...")

        # Lazy imports
        from osprey.services.ariel_search import (
            ARIELSearchRequest,
            SearchMode,
        )
        from osprey.services.ariel_search.capability import get_ariel_search_service

        # Build search request
        time_range_tuple = None
        if time_range and hasattr(time_range, "start_date") and hasattr(time_range, "end_date"):
            time_range_tuple = (time_range.start_date, time_range.end_date)

        request = ARIELSearchRequest(
            query=query,
            time_range=time_range_tuple,
            modes=[SearchMode.RAG],
        )

        # Get service and execute
        service = await get_ariel_search_service()
        async with service:
            result = await service.search(
                query=request.query,
                max_results=request.max_results,
                mode=SearchMode.RAG,
                time_range=request.time_range,
            )

        # Build output context - convert entries to dicts if they aren't already
        # EnhancedLogbookEntry is a TypedDict, so we can cast them to dict
        entries_as_dicts: tuple[dict[str, Any], ...] = tuple(
            dict(e)
            for e in result.entries  # type: ignore[arg-type]
        )
        output = LogbookSearchResultsContext(
            entries=entries_as_dicts,
            answer=result.answer,
            sources=tuple(result.sources),
            search_modes_used=tuple(m.value for m in result.search_modes_used),
            query=query,
            time_range_applied=time_range is not None,
        )

        logger.success(f"Found {len(result.entries)} entries")
        return self.store_output_context(output)

    @staticmethod
    def classify_error(exc: Exception, context: dict) -> ErrorClassification:
        """Classify ARIEL errors for recovery strategies.

        Provides actionable guidance in user_message to help users
        resolve common setup issues (database not running, tables
        not created, embedding model unavailable).
        """
        from osprey.services.ariel_search.exceptions import (
            ConfigurationError,
            DatabaseConnectionError,
            DatabaseQueryError,
            EmbeddingGenerationError,
        )

        if isinstance(exc, DatabaseConnectionError):
            return ErrorClassification(
                severity=ErrorSeverity.CRITICAL,
                user_message=(
                    "Cannot connect to the logbook database. "
                    "Run 'osprey deploy up' to start the database, then "
                    "'osprey ariel migrate' and 'osprey ariel ingest' to set up data."
                ),
                metadata={"technical_details": str(exc)},
            )
        elif isinstance(exc, DatabaseQueryError):
            if "relation" in str(exc) and "does not exist" in str(exc):
                return ErrorClassification(
                    severity=ErrorSeverity.CRITICAL,
                    user_message=(
                        "Logbook database tables not found. "
                        "Run 'osprey ariel migrate' to create tables, then "
                        "'osprey ariel ingest' to populate data."
                    ),
                    metadata={"technical_details": str(exc)},
                )
            return ErrorClassification(
                severity=ErrorSeverity.RETRIABLE,
                user_message="Database query error, retrying...",
                metadata={"technical_details": str(exc)},
            )
        elif isinstance(exc, EmbeddingGenerationError):
            return ErrorClassification(
                severity=ErrorSeverity.CRITICAL,
                user_message=(
                    "Embedding model unavailable. If you don't need semantic search, "
                    "disable it in config.yml: search_modules.semantic.enabled: false"
                ),
                metadata={"technical_details": str(exc)},
            )
        elif isinstance(exc, ConfigurationError):
            return ErrorClassification(
                severity=ErrorSeverity.CRITICAL,
                user_message=f"Logbook search configuration error: {exc.message}",
                metadata={"technical_details": str(exc)},
            )
        else:
            return ErrorClassification(
                severity=ErrorSeverity.CRITICAL,
                user_message=f"Logbook search failed: {str(exc)}",
                metadata={"technical_details": str(exc)},
            )

    def _create_orchestrator_guide(self) -> OrchestratorGuide | None:
        """Create orchestrator guide from prompt builder system.

        Retrieves orchestration guidance from the application's prompt builder,
        enabling facility-specific customization of planning behavior.
        """
        try:
            prompt_provider = get_framework_prompts()
            logbook_builder = getattr(prompt_provider, "get_logbook_search_prompt_builder", None)
            if logbook_builder is None:
                return self._default_orchestrator_guide()
            builder = logbook_builder()
            result = builder.get_orchestrator_guide()
            return result if isinstance(result, OrchestratorGuide) else None
        except (AttributeError, ImportError):
            # Fallback if prompt builder not available
            return self._default_orchestrator_guide()

    def _create_classifier_guide(self) -> TaskClassifierGuide | None:
        """Create classifier guide from prompt builder system.

        Retrieves classification guidance from the application's prompt builder,
        enabling facility-specific customization of task routing.
        """
        try:
            prompt_provider = get_framework_prompts()
            logbook_builder = getattr(prompt_provider, "get_logbook_search_prompt_builder", None)
            if logbook_builder is None:
                return self._default_classifier_guide()
            builder = logbook_builder()
            result = builder.get_classifier_guide()
            return result if isinstance(result, TaskClassifierGuide) else None
        except (AttributeError, ImportError):
            # Fallback if prompt builder not available
            return self._default_classifier_guide()

    def _default_orchestrator_guide(self) -> OrchestratorGuide:
        """Provide default orchestrator guide when prompt builder unavailable."""
        semantic_example = OrchestratorExample(
            step=PlannedStep(
                context_key="ls_search_results",
                capability="logbook_search",
                task_objective="Search logbook for relevant historical information",
                expected_output="LOGBOOK_SEARCH_RESULTS",
                success_criteria="Found relevant logbook entries",
                inputs=[],
            ),
            scenario_description="When user asks about past events or equipment history",
            context_requirements={},
            notes="Agent automatically selects best search strategy",
        )

        return OrchestratorGuide(
            instructions=textwrap.dedent("""
                **When to plan "logbook_search" steps:**
                - When user asks about facility history, past incidents, or equipment behavior
                - When user wants to find specific logbook entries by keyword or topic
                - When user asks questions about operational procedures documented in logs

                **Search Strategy Selection:**
                The internal ReAct agent automatically selects the best search strategy:
                - Keyword: For specific terms, equipment names, exact phrases
                - Semantic: For conceptual queries, related events
                - RAG: For questions needing synthesized answers
            """),
            examples=[semantic_example],
            priority=15,
        )

    def _default_classifier_guide(self) -> TaskClassifierGuide:
        """Provide default classifier guide when prompt builder unavailable."""
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
            ],
            actions_if_true=ClassifierActions(),
        )


__all__ = [
    "LogbookSearchCapability",
    "LogbookSearchResultsContext",
]
