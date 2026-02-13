"""Default hierarchical pipeline prompt builder.

Provides generic facility description and matching rules for the hierarchical
channel finder pipeline. Applications override this with facility-specific content.
"""

from .base import ChannelFinderPromptBuilder

# Default facility description for hierarchical pipeline
_DEFAULT_FACILITY_DESCRIPTION = """
This is a control system using hierarchical channel naming.

The control system uses EPICS (Experimental Physics and Industrial Control System) with
hierarchical channel naming for organized access to process variables.

CHANNEL NAMING STRUCTURE:
Channels follow a hierarchical pattern: {SYSTEM}:{FAMILY}[{DEVICE}]:{FIELD}:{SUBFIELD}
- SYSTEM: Top-level subsystem (e.g., MAG, VAC, RF, DIAG)
- FAMILY: Device family within system (e.g., DIPOLE, QUADRUPOLE, BPM)
- DEVICE: Specific device instance (e.g., B05, Q12, BPM08)
- FIELD: Physical quantity being measured/controlled (e.g., CURRENT, POSITION, PRESSURE)
- SUBFIELD: Measurement type or control function (e.g., SP, RB, X, Y)
""".strip()

# Default matching rules for hierarchical pipeline
_DEFAULT_MATCHING_RULES = """
CRITICAL TERMINOLOGY:

Setpoint vs Readback:
- "SP" (Setpoint) = Control/command value to be written
- "RB" (Readback) = Actual measured value (read-only)
- When user asks to "set", "control", "adjust" -> return SP channels
- When user asks to "read", "monitor", "measure" -> return RB channels
- When ambiguous (e.g., "show me", "what is") -> include both SP and RB

Position and Axis:
- "X" or "horizontal" = horizontal plane
- "Y" or "vertical" = vertical plane
- "position" or "orbit" = spatial location of beam

Common Synonyms:
- "bending magnet" = dipole magnet
- "focusing magnet" or "quad" = quadrupole magnet
- "corrector" or "steering" = corrector magnet
- "vacuum level" or "vacuum pressure" = pressure measurement
""".strip()


class DefaultHierarchicalPromptBuilder(ChannelFinderPromptBuilder):
    """Default prompt builder for hierarchical channel finder pipeline."""

    def get_facility_description(self) -> str:
        return _DEFAULT_FACILITY_DESCRIPTION

    def get_matching_rules(self) -> str:
        return _DEFAULT_MATCHING_RULES
