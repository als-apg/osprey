"""Channel matching rules and terminology conventions for in-context pipeline.

This file contains rules for interpreting user queries and matching them
to the correct channels. These rules are largely reusable across facilities,
though some synonyms may need customization.

Sections:
- Channel naming patterns
- Motor vs ReadBack terminology
- Control vs Status interpretation
"""

MATCHING_RULES = """
IMPORTANT TERMINOLOGY AND CONVENTIONS:

Channel Naming Patterns:
- "Motor" channels = Control/command channels (for setting positions or states)
- "MotorReadBack" or "ReadBack" channels = Status/measurement channels (actual positions or states)
- "SetPoint" or "Set" channels = Control values to be commanded
- When query asks for "control" or "motor control", return ONLY Motor/Set channels, NOT readbacks
- When query asks for "status" or "readback" or "actual", return ONLY ReadBack channels
- When query asks to "check" or is ambiguous, include both Set and ReadBack
""".strip()
