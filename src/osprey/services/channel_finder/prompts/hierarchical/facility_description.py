"""Facility-specific description for Example Hierarchical Accelerator.

This file contains facility-specific information that users should customize
for their own control system. It describes:
- What the facility is
- Channel naming structure
- Device naming conventions

To customize for your facility:
1. Copy this file to your facility's prompts directory
2. Replace the content with your facility's description
3. Update device naming conventions to match your PV naming scheme
"""

FACILITY_DESCRIPTION = """
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
- B## = Dipole bending magnets (B01-B24)
- QF## = Focusing quadrupoles (QF01-QF16)
- QD## = Defocusing quadrupoles (QD01-QD16)
- SF## = Focusing sextupoles (SF01-SF12)
- SD## = Defocusing sextupoles (SD01-SD12)
- H## = Horizontal corrector magnets (H01-H20)
- V## = Vertical corrector magnets (V01-V20)

DIAG System Devices:
- BPM## = Beam position monitors (BPM01-BPM20)
- DCCT = DC current transformer (MAIN)
- NEUT## = Neutron radiation monitors (NEUT01-NEUT04)
- GAMM## = Gamma radiation monitors (GAMM01-GAMM04)

VAC System Devices:
- SR## = Ion pumps and gauges (SR01-SR06, SR01A-SR03B)
- V## = Vacuum valves (V01-V12)

RF System Devices:
- C# = RF cavities (C1, C2)
- K# = Klystrons (K1, K2)

Note: Detailed descriptions of each system, family, field, and subfield are provided
in the hierarchical navigation context during query processing.
""".strip()
