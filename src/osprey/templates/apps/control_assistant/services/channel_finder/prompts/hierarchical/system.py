"""System-level prompt assembly for Example Hierarchical Accelerator.

This module combines facility description and matching rules into the
complete facility_description used by the channel finder pipeline.

To customize for your facility:
1. Edit facility_description.py with your facility's details
2. Optionally edit matching_rules.py to adjust terminology/synonyms
3. This file (system.py) typically doesn't need modification
"""

from .facility_description import FACILITY_DESCRIPTION
from .matching_rules import MATCHING_RULES

# Combine facility description and matching rules into complete prompt
facility_description = f"""
{FACILITY_DESCRIPTION}

{MATCHING_RULES}
""".strip()
