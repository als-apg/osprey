"""System-level prompt assembly for Example Middle Layer Accelerator.

Combines facility_description and matching_rules into the complete
facility_description string used by the pipeline. Edit the component
files rather than this module.
"""

from .facility_description import FACILITY_DESCRIPTION
from .matching_rules import MATCHING_RULES

# Combine facility description and matching rules into complete prompt
facility_description = f"""
{FACILITY_DESCRIPTION}

{MATCHING_RULES}
""".strip()
