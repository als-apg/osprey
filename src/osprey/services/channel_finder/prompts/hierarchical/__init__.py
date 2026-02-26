"""Hierarchical facility prompts for channel finder.

Modules: facility_description, matching_rules, system (combined description),
query_splitter, and hierarchical_context (navigation instructions per level).
Unlike in-context, this pipeline does not use channel_matcher or correction
prompts -- the tree traversal handles matching during navigation.
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
