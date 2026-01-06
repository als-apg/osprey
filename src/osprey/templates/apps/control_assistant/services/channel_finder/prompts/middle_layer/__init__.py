"""
Example Middle Layer Facility Prompts

This package contains facility-specific prompts for the example middle layer accelerator.

Modular Structure:
- facility_description.py: Facility-specific description (CUSTOMIZE THIS)
- matching_rules.py: Channel matching rules and terminology (CUSTOMIZE AS NEEDED)
- system.py: Combines the above into facility_description
- query_splitter.py: Stage 1 query splitting

For middle_layer pipeline:
Note: Unlike in-context pipeline, middle_layer doesn't use channel_matcher or correction prompts.
The React agent uses database query tools to explore the functional hierarchy and find channels.

Architecture:
  - Database (middle_layer.json): Contains ONLY DATA (functional hierarchy, channel addresses)
  - Prompts: Contains INSTRUCTIONS (query splitting and agent system prompt)
  - Agent Tools: Provides database exploration capabilities

This maintains clean separation: data vs prompts vs tools.
"""

from . import query_splitter
from .facility_description import FACILITY_DESCRIPTION
from .matching_rules import MATCHING_RULES
from .system import facility_description

__all__ = [
    "facility_description",
    "FACILITY_DESCRIPTION",
    "MATCHING_RULES",
    "query_splitter",
]
