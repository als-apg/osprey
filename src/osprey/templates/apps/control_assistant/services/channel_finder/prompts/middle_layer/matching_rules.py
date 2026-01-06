"""Channel matching rules and terminology conventions for Middle Layer pipeline.

This file contains rules for interpreting user queries and matching them
to the correct channels using the MML (MATLAB Middle Layer) pattern.

Sections:
- Monitor vs Setpoint terminology
- Position and axis conventions
- Device selection patterns
- Common synonyms
- Operational guidelines for React agent
"""

MATCHING_RULES = """
CRITICAL TERMINOLOGY:

Monitor vs Setpoint:
- "Monitor" = Readback/measured value (read-only)
- "Setpoint" = Control/command value (writable)
- When user asks to "read", "monitor", "measure", "check" → return Monitor fields
- When user asks to "set", "control", "adjust", "command" → return Setpoint fields
- When ambiguous (e.g., "show me", "what is") → include both Monitor and Setpoint

Position and Axis:
- "X" or "horizontal" = horizontal beam position (BPM.X field)
- "Y" or "vertical" = vertical beam position (BPM.Y field)
- "position" or "orbit" = both X and Y coordinates

Device Selection:
- Devices are numbered (e.g., device 1, device 8, sector 2)
- "sector N" typically refers to devices in that sector/region
- "all devices" = all members of a family
- Specific numbers can be filtered using device parameter

Common Synonyms:
- "beam position" = BPM family (X, Y fields - NOT nested under Monitor)
- "beam current" = DCCT (Monitor field)
- "beam lifetime" = DCCT (Lifetime field)
- "corrector" or "steering" = both HCM and VCM families
- "quadrupole" or "quad" = both QF and QD families
- "sextupole" or "sext" = both SF and SD families
- "horizontal corrector" = HCM family
- "vertical corrector" = VCM family
- "dipole" or "bending magnet" = BEND family
- "vacuum pressure" = SR.VAC.IonPump.Pressure or SR.VAC.Gauge.Pressure
- "ion pump" = SR.VAC.IonPump (nested under VAC family)
- "vacuum gauge" = SR.VAC.Gauge (nested under VAC family)
- "RF frequency" = RF (FrequencyMonitor/FrequencySetpoint fields)
- "RF voltage" or "gap voltage" = RF (VoltageMonitor/VoltageSetpoint fields)
- "RF power" = RF.PowerMonitor (Forward/Reflected subfields)
- "RF phase" = RF (PhaseMonitor/PhaseSetpoint fields)
- "insertion device" or "undulator" or "wiggler" = ID family
- "tune" or "betatron tune" = Tune family
- "temperature" = Thermocouple family
- "scraper" = Scraper family

IMPORTANT FIELD NAMING:
- BPM uses direct field names: X, Y, SumSignal, XError, YError (NOT Monitor.X)
- Correctors (HCM/VCM) use: Monitor, Setpoint, Ready, OnControl
- RF uses compound names: FrequencyMonitor, VoltageSetpoint, etc.
- VAC is a nested family: VAC.IonPump.Pressure, VAC.Gauge.Pressure

Operational Guidelines:
- Use tools to explore database hierarchy systematically
- Start with list_systems() to see available systems (SR, BR, BTS)
- Use list_families(system) to see device families
- Use inspect_fields(system, family, field) to understand field structure
- Use list_channel_names() with appropriate filters to retrieve PVs
- Use get_common_names() to understand device naming/numbering
- Some families are nested (VAC contains IonPump/Gauge) - inspect to discover
- DeviceList metadata enables sector/device filtering

Note: The React agent should explore the database using tools rather than making
assumptions. Always verify available options before making selections.
""".strip()
