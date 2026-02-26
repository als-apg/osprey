"""Middle layer facility prompts for channel finder.

Modules: facility_description, matching_rules, system (combined description),
and query_splitter. Unlike in-context, this pipeline uses a React agent with
database query tools instead of channel_matcher or correction prompts.
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
