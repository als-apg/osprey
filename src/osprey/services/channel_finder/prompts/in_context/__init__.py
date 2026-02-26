"""In-context facility prompts (UCSB FEL example).

Modules: facility_description, matching_rules, system (combined description),
query_splitter, channel_matcher, and correction. Best suited for smaller
control systems (<1,000 channels) using semantic matching against the full
channel database.
"""

from . import channel_matcher, correction, query_splitter
from .facility_description import FACILITY_DESCRIPTION
from .matching_rules import MATCHING_RULES
from .system import facility_description

__all__ = [
    "facility_description",
    "FACILITY_DESCRIPTION",
    "MATCHING_RULES",
    "query_splitter",
    "channel_matcher",
    "correction",
]
