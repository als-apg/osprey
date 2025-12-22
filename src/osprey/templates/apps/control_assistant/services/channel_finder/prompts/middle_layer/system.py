"""System-level facility description for Example Middle Layer Accelerator."""

import textwrap

facility_description = textwrap.dedent(
    """
    This is an accelerator facility control system using functional (Middle Layer) channel organization.

    The control system uses EPICS (Experimental Physics and Industrial Control System) with
    a functional hierarchy based on the MATLAB Middle Layer (MML) pattern used in production
    at facilities like ALS and ESRF.

    FUNCTIONAL HIERARCHY STRUCTURE:
    Channels are organized by function: System → Family → Field → ChannelNames
    - SYSTEM: Major subsystem (e.g., SR, BR, VAC, BTS)
    - FAMILY: Device family by function (e.g., BPM, HCM, VCM, QF, QD, RF)
    - FIELD: Measurement/control function (e.g., Monitor, Setpoint, OnControl)
    - SUBFIELD: (Optional) Component within field (e.g., X/Y for position, Frequency/Voltage for RF)
    - ChannelNames: Actual PV addresses stored in database, indexed by device number

    CRITICAL DIFFERENCES FROM HIERARCHICAL:
    - PV addresses are RETRIEVED from database, NOT built from patterns
    - Organization is by FUNCTION (Monitor/Setpoint) not naming convention
    - Device filtering uses DeviceList metadata, not name parsing
    - Subfields can be nested (e.g., RF.Frequency.Monitor, BPM.Monitor.X)

    SYSTEM DESCRIPTIONS:
    SR (Storage Ring):
    - Main synchrotron light source
    - Contains magnets (HCM, VCM, QF, QD, SF, SD), diagnostics (BPM, DCCT), RF

    BR (Booster Ring):
    - Accelerates beam to injection energy
    - Contains BPM, DCCT, Dipole magnets

    VAC (Vacuum System):
    - Maintains ultra-high vacuum
    - Contains IonPump, Gauge families

    BTS (Booster-to-Storage Transfer Line):
    - Beam transport between rings
    - Contains BPM, Kicker, Septum

    DEVICE FAMILIES AND FUNCTIONS:
    Beam Position Monitors (BPM):
    - Monitor.X, Monitor.Y = horizontal and vertical beam position (mm)
    - SumSignal = total beam signal intensity

    Corrector Magnets (HCM/VCM):
    - Monitor = readback current (A)
    - Setpoint = desired current (A)
    - OnControl = control system status

    Quadrupole Magnets (QF/QD):
    - QF = Focusing quadrupoles
    - QD = Defocusing quadrupoles
    - Monitor = readback current (A)
    - Setpoint = desired current (A)

    Sextupole Magnets (SF/SD):
    - SF = Focusing sextupoles
    - SD = Defocusing sextupoles
    - Monitor = readback current (A)
    - Setpoint = desired current (A)

    Beam Current (DCCT):
    - Monitor = beam current measurement (mA)

    RF System:
    - Frequency: Monitor/Setpoint = RF frequency (MHz)
    - Voltage: Monitor/Setpoint = cavity voltage (MV)
    - PowerMonitor: Forward/Reflected = RF power (kW)

    Vacuum System (IonPump/Gauge):
    - Pressure = vacuum level (Torr)
    - Voltage/Current = pump operating parameters (V/A)

    CRITICAL TERMINOLOGY:

    Monitor vs Setpoint:
    - "Monitor" = Readback/measured value (read-only)
    - "Setpoint" = Control/command value (writable)
    - When user asks to "read", "monitor", "measure", "check" → return Monitor fields
    - When user asks to "set", "control", "adjust", "command" → return Setpoint fields
    - When ambiguous (e.g., "show me", "what is") → include both Monitor and Setpoint

    Position and Axis:
    - "X" or "horizontal" = horizontal beam position
    - "Y" or "vertical" = vertical beam position
    - "position" or "orbit" = both X and Y coordinates

    Device Selection:
    - Devices are numbered (e.g., device 1, device 8, sector 2)
    - "sector N" typically refers to devices in that sector/region
    - "all devices" = all members of a family
    - Specific numbers can be filtered using device parameter

    Common Synonyms:
    - "beam position" = BPM (Monitor.X, Monitor.Y)
    - "beam current" = DCCT (Monitor)
    - "corrector" or "steering" = both HCM and VCM families
    - "quadrupole" or "quad" = both QF and QD families
    - "sextupole" or "sext" = both SF and SD families
    - "horizontal corrector" = HCM family
    - "vertical corrector" = VCM family
    - "vacuum pressure" = IonPump or Gauge (Pressure field)
    - "RF frequency" = RF (Frequency field)
    - "RF voltage" or "gap voltage" = RF (Voltage field)
    - "RF power" = RF (PowerMonitor field)

    Operational Guidelines:
    - Use tools to explore database hierarchy systematically
    - Start with list_systems() to see available systems
    - Use list_families(system) to see device families
    - Use inspect_fields(system, family, field) to understand field structure
    - Use list_channel_names() with appropriate filters to retrieve PVs
    - Use get_common_names() to understand device naming/numbering
    - Subfields are nested - inspect parent field first to discover them
    - DeviceList metadata enables sector/device filtering

    Note: The React agent should explore the database using tools rather than making
    assumptions. Always verify available options before making selections.
    """
).strip()
