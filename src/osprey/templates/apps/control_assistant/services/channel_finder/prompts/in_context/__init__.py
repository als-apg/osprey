"""
UCSB FEL In-Context Facility Prompts

This package contains facility-specific prompts for the UCSB Free Electron Laser.

Modular Structure:
- facility_description.py: Facility-specific description (CUSTOMIZE THIS)
- matching_rules.py: Channel matching rules and terminology (CUSTOMIZE AS NEEDED)
- system.py: Combines the above into facility_description
- query_splitter.py: Stage 1 query splitting
- channel_matcher.py: Stage 2 channel matching
- correction.py: Stage 3 correction prompts

For in-context pipeline:
The in-context pipeline uses semantic matching against the full channel database.
Best suited for smaller control systems (<1,000 channels).
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
