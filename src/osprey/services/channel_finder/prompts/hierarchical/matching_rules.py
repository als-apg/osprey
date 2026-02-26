"""Channel matching rules and terminology conventions.

This file contains rules for interpreting user queries and matching them
to the correct channels. These rules are largely reusable across facilities,
though some synonyms may need customization.

Sections:
- Setpoint vs Readback terminology
- Position and axis conventions
- Status field conventions
- Common synonyms
- Operational guidelines
"""

MATCHING_RULES = """
CRITICAL TERMINOLOGY:

Setpoint vs Readback:
- "SP" (Setpoint) = Control/command value to be written
- "RB" (Readback) = Actual measured value (read-only)
- "GOLDEN" = Reference value for known good operation
- When user asks to "set", "control", "adjust", or "command" → return SP channels
- When user asks to "read", "monitor", "measure", "check", or "actual" → return RB channels
- When ambiguous (e.g., "show me", "what is") → include both SP and RB

Position and Axis:
- "X" or "horizontal" = horizontal plane (perpendicular to beam direction)
- "Y" or "vertical" = vertical plane (perpendicular to beam direction)
- "position" or "orbit" = spatial location of beam
- "offset" = deviation from reference/golden position

Status Fields:
- "READY" = device is initialized and ready to operate
- "ON" = device is powered/enabled
- "FAULT" = error or fault condition detected
- "VALID" = measurement is valid and within range
- "CONNECTED" = device is communicating properly

Common Synonyms:
- "bending magnet" = dipole magnet
- "focusing magnet" or "quad" = quadrupole magnet
- "sextupole" or "sext" = sextupole magnet (SF for focusing, SD for defocusing)
- "chromaticity correction" = sextupole magnets (both SF and SD families)
- "corrector" or "steering" = corrector magnet (includes both HCM and VCM families)
- "tune correction" or "tune quadrupoles" = both QF and QD quadrupole families
- "vacuum level" or "vacuum pressure" = pressure measurement (from gauges and pumps)
- "pump" = vacuum pump (ion pump, etc.)
- "gap voltage" = RF cavity voltage
- "RF system" = all RF families (cavities, klystrons, etc.)
- "radiation" or "dose" = radiation monitors (neutron and gamma detectors)
- "neutron" or "gamma" = radiation safety monitors in DIAG system

Operational Guidelines:
- Golden values represent optimized operating conditions
  * "Golden" or "golden values" refers to GOLDEN subfields across various fields
  * When asked for "golden values" for a device, return all fields with GOLDEN subfields
  * GOLDEN subfields are read-only reference values for comparison and restore operations
- Status channels are essential for health monitoring and interlocks
- Beam position stability is critical for facility operation
- Always distinguish between commanded values (SP) and actual values (RB)
- Range queries (e.g., "magnets 1-5", "all devices") should expand to specific instances
- System-wide queries (e.g., "RF system", "correctors", "vacuum") should include ALL relevant families
""".strip()
