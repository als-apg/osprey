"""Default in-context pipeline prompt builder.

Provides generic facility description and matching rules for the in-context
channel finder pipeline. Applications override this with facility-specific content.
"""

from .base import ChannelFinderPromptBuilder

# Default facility description for in-context pipeline
_DEFAULT_FACILITY_DESCRIPTION = """
This is a control system with channels organized as flat key-value pairs.

Each channel has a descriptive name and an address used for control system operations.
The channel finder uses semantic matching to find the best channels matching a user's
natural language query.

Channel Naming Patterns:
- Channels may include subsystem prefixes (e.g., "MAG:", "VAC:", "RF:")
- SetPoint/Set channels are for control/command values
- ReadBack/RB channels are for measured/actual values
- Motor channels may indicate movable components
""".strip()

# Default matching rules for in-context pipeline
_DEFAULT_MATCHING_RULES = """
IMPORTANT TERMINOLOGY AND CONVENTIONS:

Channel Naming Patterns:
- "Motor" channels = Control/command channels (for setting positions or states)
- "MotorReadBack" or "ReadBack" channels = Status/measurement channels
- "SetPoint" or "Set" channels = Control values to be commanded
- When query asks for "control" or "motor control", return ONLY Motor/Set channels
- When query asks for "status" or "readback" or "actual", return ONLY ReadBack channels
- When query asks to "check" or is ambiguous, include both Set and ReadBack
""".strip()


class DefaultInContextPromptBuilder(ChannelFinderPromptBuilder):
    """Default prompt builder for in-context channel finder pipeline."""

    def get_facility_description(self) -> str:
        return _DEFAULT_FACILITY_DESCRIPTION

    def get_matching_rules(self) -> str:
        return _DEFAULT_MATCHING_RULES
