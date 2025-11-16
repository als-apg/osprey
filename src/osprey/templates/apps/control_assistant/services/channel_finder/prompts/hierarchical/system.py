"""System-level facility description for Example Hierarchical Accelerator."""

import textwrap


facility_description = textwrap.dedent("""
    This is an accelerator facility control system using hierarchical channel naming.
    
    The control system uses EPICS (Experimental Physics and Industrial Control System) with 
    hierarchical channel naming for organized access to thousands of process variables.
    
    CHANNEL NAMING STRUCTURE:
    Channels follow a hierarchical pattern: {SYSTEM}:{FAMILY}[{DEVICE}]:{FIELD}:{SUBFIELD}
    - SYSTEM: Top-level subsystem (e.g., MAG, VAC, RF, DIAG)
    - FAMILY: Device family within system (e.g., DIPOLE, QUADRUPOLE, BPM)
    - DEVICE: Specific device instance (e.g., B05, Q12, BPM08)
    - FIELD: Physical quantity being measured/controlled (e.g., CURRENT, POSITION, PRESSURE)
    - SUBFIELD: Measurement type or control function (e.g., SP, RB, X, Y)
    
    DEVICE NAMING CONVENTIONS (recognize these prefixes):
    MAG System Devices:
    - B## = Dipole bending magnets (B01-B12)
    - QF## = Focusing quadrupoles (QF01-QF12)
    - QD## = Defocusing quadrupoles (QD01-QD12)
    - SF## = Focusing sextupoles (SF01-SF08)
    - SD## = Defocusing sextupoles (SD01-SD12)
    - H## = Horizontal corrector magnets (H01-H12)
    - V## = Vertical corrector magnets (V01-V12)
    
    DIAG System Devices:
    - BPM## = Beam position monitors (BPM01-BPM12)
    - DCCT = DC current transformer (MAIN)
    - NEUT## = Neutron radiation monitors (NEUT01-NEUT04)
    - GAMM## = Gamma radiation monitors (GAMM01-GAMM04)
    
    VAC System Devices:
    - SR## = Ion pumps and gauges (SR01-SR06, SR01A-SR03B)
    - V## = Vacuum valves (V01-V06)
    
    RF System Devices:
    - C# = RF cavities (C1, C2)
    - K# = Klystrons (K1, K2)
    
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
    
    Note: Detailed descriptions of each system, family, field, and subfield are provided
    in the hierarchical navigation context during query processing.
    """).strip()

