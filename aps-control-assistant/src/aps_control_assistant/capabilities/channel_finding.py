"""
Channel Finding Capability

This capability searches for control system channel addresses based on user descriptions.
It helps users find the correct channel names when they know what they want to measure
but don't know the exact channel address.

Based on ALS Assistant's PV Address Finding capability pattern.
"""
from __future__ import annotations

import logging
import re
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from osprey.state import AgentState

from osprey.base.capability import BaseCapability
from osprey.base.decorators import capability_node
from osprey.base.errors import ErrorClassification, ErrorSeverity
from osprey.base.examples import (
    ClassifierActions,
    ClassifierExample,
    OrchestratorExample,
    OrchestratorGuide,
    TaskClassifierGuide,
)
from osprey.base.planning import PlannedStep
from osprey.registry import get_registry

from ..context_classes import ChannelAddressesContext
from ..data.tools.pv_catalog import (
    PVEntry,
    get_pv_catalog,
)


PV_CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "channel_databases" / "PVs.mon"



# Available pipelines based on template configuration

AVAILABLE_PIPELINES = ['in_context', 'hierarchical']

DEFAULT_PIPELINE = 'hierarchical'


# ===================================================================
# Channel Finding Errors
# ===================================================================

class ChannelFindingError(Exception):
    """Base class for all channel finding errors."""
    pass

class ChannelNotFoundError(ChannelFindingError):
    """Raised when no channel addresses can be found for the given query."""
    pass

class ChannelFinderServiceError(ChannelFindingError):
    """Raised when the channel finder service is unavailable or fails."""
    pass


# ===================================================================
# Capability Implementation
# ===================================================================

@capability_node
class ChannelFindingCapability(BaseCapability):
    """
    Channel finding capability for resolving descriptions to channel addresses.

    
    Configuration: Both pipelines enabled
    - Switch between modes via config.yml
    
    """

    name = "channel_finding"
    description = "Find control system channel addresses based on descriptions or search terms"
    provides = ["CHANNEL_ADDRESSES"]
    requires = []  # No dependencies - extracts from task objective

    def _find_with_pv_catalog(self, search_query: str, logger: logging.Logger) -> List[PVEntry]:
        """
        Resolve PVs locally using the PV catalog (PVs.mon) aliases or srcorr.json fast-path.
        """
        try:
            catalog = get_pv_catalog(str(PV_CATALOG_PATH))
        except Exception as exc:
            logger.warning(f"PV catalog unavailable at {PV_CATALOG_PATH}: {exc}")
            return []

        # First, try SR corrector PV finding (uses srcorr.json)
        logger.info(f"🔍 Checking if SR corrector query...")
        sr_entries = catalog.find_sr_corrector_pvs(search_query)
        if sr_entries:
            logger.info(f"✅ Found {len(sr_entries)} SR corrector PVs via srcorr.json")
            logger.info(f"   First 5 PVs: {[e.pv_name for e in sr_entries[:5]]}")
            return sr_entries
        else:
            logger.info(f"ℹ️  Not an SR corrector query, using standard PV catalog")

        # Try to extract explicit PV-like tokens (e.g., "S10A:Q1") and search by prefix
        pv_tokens = [
            tok.lower()
            for tok in re.findall(r"[A-Za-z0-9:_-]+", search_query)
            if ":" in tok
        ]

        prefix_matches: List[PVEntry] = []
        if pv_tokens:
            seen = set()
            for entry in catalog.entries:
                name_lower = entry.pv_name.lower()
                if any(name_lower.startswith(tok) for tok in pv_tokens):
                    if entry.pv_name not in seen:
                        prefix_matches.append(entry)
                        seen.add(entry.pv_name)
                if len(prefix_matches) >= 100:
                    break  # avoid huge result sets
            if prefix_matches:
                logger.info(f"Found {len(prefix_matches)} PV(s) via PV prefix match ({', '.join(set(pv_tokens))})")
                return prefix_matches

        # Fallback to alias-based fuzzy matching
        matches = catalog.match_all(search_query)
        if not matches:
            fallback = catalog.match_query(search_query)
            if fallback:
                matches = [fallback]

        if matches:
            logger.info(f"Found {len(matches)} PV(s) via local PV catalog")
        else:
            logger.debug("No PVs found via local PV catalog; falling back to channel finder service.")

        return matches

    async def execute(self) -> Dict[str, Any]:
        """
        Find control system channel addresses based on search query.

        This method uses the PV catalog (including srcorr.json for SR devices) to find
        matching control system addresses. The search query is extracted from the task objective.

        Returns:
            State updates containing CHANNEL_ADDRESSES context with found channels.

        Raises:
            ChannelNotFoundError: If no matching channels are found.
        """
        # Extract search query from task objective
        search_query = self.get_task_objective(default='unknown')

        # Get unified logger with automatic streaming
        logger = self.get_logger()

        # Log the query
        logger.info(f'🆕 CODE UPDATED 2025-12-05 15:50 - Using PV catalog only (no LLM service)')
        logger.info(f'Channel finding query: "{search_query}"')

        # Some logger implementations in Osprey support `status()`, but if we are
        # given a standard logging.Logger it will not exist. Use a safe fallback.
        status_message = "Finding channel addresses..."
        if hasattr(logger, "status"):
            logger.status(status_message)
        else:
            logger.info(status_message)

        channel_list: List[str] = []
        description = ""

        # Use PV catalog exclusively (includes srcorr.json fast-path for SR devices)
        catalog_matches = self._find_with_pv_catalog(search_query, logger)
        if catalog_matches:
            channel_list = [entry.pv_name for entry in catalog_matches]
            description = f"Found {len(channel_list)} PV(s) via PV catalog for query: '{search_query}'."
            logger.info(f"✅ Found {len(channel_list)} channel addresses")
        else:
            # No matches found in PV catalog
            error_msg = f"No channel addresses found for query: '{search_query}'."
            logger.warning(f"❌ Channel address not found: {error_msg}")
            raise ChannelNotFoundError(error_msg)

        # Create framework context object
        channel_context = ChannelAddressesContext(
            channels=channel_list,
            description=description,
        )

        status_message = "Channel finding complete"
        if hasattr(logger, "status"):
            logger.status(status_message)
        else:
            logger.info(status_message)

        # Store result in execution context
        return self.store_output_context(channel_context)

    @staticmethod
    def classify_error(exc: Exception, context: dict) -> ErrorClassification:
        """Channel finding error classification with defensive approach."""

        if isinstance(exc, ChannelNotFoundError):
            return ErrorClassification(
                severity=ErrorSeverity.REPLANNING,
                user_message=f"No channel addresses found: {str(exc)}",
                metadata={
                    "technical_details": str(exc),
                    "replanning_reason": f"Channel search failed to find matches: {exc}",
                    "suggestions": [
                        "Try refining the search terms to be more specific",
                        "Check if the requested channels exist in the system",
                        "Consider using different keywords or system names",
                    ]
                }
            )

        elif isinstance(exc, ChannelFinderServiceError):
            return ErrorClassification(
                severity=ErrorSeverity.CRITICAL,
                user_message=f"Channel finder service unavailable: {str(exc)}",
                metadata={
                    "technical_details": str(exc),
                    "safety_abort_reason": f"Channel finder service failure: {exc}",
                    "suggestions": [
                        "Check if the channel finder service is running and accessible",
                        "Verify channel database is loaded properly",
                        "Contact system administrator if the service appears down",
                    ]
                }
            )

        else:
            # Default: critical for unknown errors (defensive approach)
            return ErrorClassification(
                severity=ErrorSeverity.CRITICAL,
                user_message=f"Unexpected error in channel finding: {exc}",
                metadata={
                    "technical_details": str(exc),
                    "safety_abort_reason": f"Unhandled channel finding error: {exc}"
                }
            )

    def _create_orchestrator_guide(self) -> Optional[OrchestratorGuide]:
        """
        Create orchestrator guide for channel finding planning.

        **Customization Point**: Modify the instructions and examples here to change
        how the orchestrator plans channel finding steps. The few-shot examples teach
        the LLM when and how to include this capability in execution plans.

        See: Framework docs on orchestrator planning for details.
        """

        # Define structured examples
        natural_language_example = OrchestratorExample(
            step=PlannedStep(
                context_key="beam_current_channels",
                capability="channel_finding",
                task_objective="Find channel addresses for beam current measurement",
                expected_output="CHANNEL_ADDRESSES",
                success_criteria="Relevant channel addresses found and validated",
                inputs=[]
            ),
            scenario_description="Natural language search for measurement types",
            notes="Output stored under CHANNEL_ADDRESSES context type."
        )

        system_location_example = OrchestratorExample(
            step=PlannedStep(
                context_key="quadrupole_channels",
                capability="channel_finding",
                task_objective="Find channel addresses for quadrupole magnet currents in all sectors",
                expected_output="CHANNEL_ADDRESSES",
                success_criteria="System-specific channel addresses located",
                inputs=[]
            ),
            scenario_description="System and location-based channel discovery",
            notes="Output stored under CHANNEL_ADDRESSES context type."
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
                - Results feed into subsequent "channel_value_retrieval" or "archiver_retrieval" steps

                ALWAYS plan this step when any channel-related operations are needed.
                """),
            examples=[natural_language_example, system_location_example],
            priority=1  # Should come early in the prompt ordering
        )

    def _create_classifier_guide(self) -> Optional[TaskClassifierGuide]:
        """
        Create classifier guide for channel finding activation.

        **Customization Point**: Modify the instructions and examples here to change
        when this capability is activated. The few-shot examples teach the LLM which
        user queries should trigger channel finding.

        See: Framework docs on classification and routing for details.
        """

        return TaskClassifierGuide(
            instructions="Determine if the task involves finding, extracting, or identifying channel addresses. This applies if the user is searching for channels based on descriptions, OR if they need any channel-related operations.",
            examples=[
                ClassifierExample(
                    query="Which tools do you have?",
                    result=False,
                    reason="This is a question about the AI's capabilities."
                ),
                ClassifierExample(
                    query="Find channels related to beam position monitors.",
                    result=True,
                    reason="The query asks to find channels based on a description."
                ),
                ClassifierExample(
                    query="I need the channel for the vacuum pressure.",
                    result=True,
                    reason="The query asks to find a channel based on a description."
                ),
                ClassifierExample(
                    query="Can you plot the beam current for the last hour?",
                    result=True,
                    reason="The query asks to plot data, which requires channel finding first."
                ),
                ClassifierExample(
                    query="What's the beam current right now?",
                    result=True,
                    reason="The query asks for a value without a specific channel address, requiring channel finding first."
                ),
            ],
            actions_if_true=ClassifierActions()
        )
