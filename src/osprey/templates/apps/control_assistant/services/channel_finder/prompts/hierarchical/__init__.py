"""
Example Hierarchical Facility Prompts

This package contains facility-specific prompts for the example hierarchical accelerator.

Modular Structure:
- facility_description.py: Facility-specific description (CUSTOMIZE THIS)
- matching_rules.py: Channel matching rules and terminology (CUSTOMIZE AS NEEDED)
- system.py: Combines the above into facility_description
- query_splitter.py: Stage 1 query splitting
- hierarchical_context.py: Hierarchical navigation context and level instructions

For hierarchical pipeline:
Note: Unlike in-context pipeline, hierarchical doesn't use channel_matcher or correction prompts.
The hierarchical navigation system handles matching and validation during tree traversal.

Architecture:
  - Database (hierarchical_database.json): Contains ONLY DATA (tree structure, descriptions)
  - Prompts: Contains INSTRUCTIONS (facility description, matching rules, navigation context)

This maintains clean separation: data vs prompts.
"""

from . import hierarchical_context, query_splitter
from .facility_description import FACILITY_DESCRIPTION
from .matching_rules import MATCHING_RULES
from .system import facility_description

__all__ = [
    "facility_description",
    "FACILITY_DESCRIPTION",
    "MATCHING_RULES",
    "query_splitter",
    "hierarchical_context",
]
