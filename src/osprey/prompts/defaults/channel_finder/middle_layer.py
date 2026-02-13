"""Default middle layer pipeline prompt builder.

Provides generic facility description and matching rules for the middle layer
channel finder pipeline. Applications override this with facility-specific content.
"""

from .base import ChannelFinderPromptBuilder

# Default facility description for middle layer pipeline
_DEFAULT_FACILITY_DESCRIPTION = """
This is a control system using functional (Middle Layer) channel organization.

The control system uses EPICS with a functional hierarchy based on the MATLAB
Middle Layer (MML) pattern.

FUNCTIONAL HIERARCHY STRUCTURE:
Channels are organized by function: System -> Family -> Field -> ChannelNames
- SYSTEM: Major subsystem (e.g., SR, BR, BTS)
- FAMILY: Device family by function (e.g., BPM, HCM, VCM, QF, QD, RF, VAC)
- FIELD: Measurement/control function (e.g., Monitor, Setpoint, X, Y, Pressure)
- ChannelNames: Actual PV addresses stored in database, indexed by device number

CRITICAL DIFFERENCES FROM HIERARCHICAL:
- PV addresses are RETRIEVED from database, NOT built from patterns
- Organization is by FUNCTION (Monitor/Setpoint) not naming convention
- Device filtering uses DeviceList metadata, not name parsing
""".strip()

# Default matching rules for middle layer pipeline
_DEFAULT_MATCHING_RULES = """
CRITICAL TERMINOLOGY:

Monitor vs Setpoint:
- "Monitor" = Readback/measured value (read-only)
- "Setpoint" = Control/command value (writable)
- When user asks to "read", "monitor", "measure" -> return Monitor fields
- When user asks to "set", "control", "adjust" -> return Setpoint fields
- When ambiguous (e.g., "show me", "what is") -> include both

Position and Axis:
- "X" or "horizontal" = horizontal beam position (BPM.X field)
- "Y" or "vertical" = vertical beam position (BPM.Y field)
- "position" or "orbit" = both X and Y coordinates

Common Synonyms:
- "beam position" = BPM family
- "beam current" = DCCT family
- "corrector" or "steering" = HCM and VCM families
- "quadrupole" or "quad" = QF and QD families

Operational Guidelines:
- Use tools to explore database hierarchy systematically
- Start with list_systems() to see available systems
- Use list_families(system) to see device families
- Use inspect_fields(system, family, field) to understand field structure
""".strip()


class DefaultMiddleLayerPromptBuilder(ChannelFinderPromptBuilder):
    """Default prompt builder for middle layer channel finder pipeline."""

    def get_facility_description(self) -> str:
        return _DEFAULT_FACILITY_DESCRIPTION

    def get_matching_rules(self) -> str:
        return _DEFAULT_MATCHING_RULES
