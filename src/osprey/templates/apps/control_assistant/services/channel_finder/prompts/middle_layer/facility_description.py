"""Facility-specific description for Example Middle Layer Accelerator.

This file contains facility-specific information that users should customize
for their own control system. It describes:
- What the facility is
- Functional hierarchy structure (Middle Layer pattern)
- System and family descriptions

To customize for your facility:
1. Copy this file to your facility's prompts directory
2. Replace the content with your facility's description
3. Update system/family descriptions to match your MML structure
"""

FACILITY_DESCRIPTION = """
This is an accelerator facility control system using functional (Middle Layer) channel organization.

The control system uses EPICS (Experimental Physics and Industrial Control System) with
a functional hierarchy based on the MATLAB Middle Layer (MML) pattern used in production
at facilities like ALS and ESRF.

FUNCTIONAL HIERARCHY STRUCTURE:
Channels are organized by function: System → Family → Field → ChannelNames
- SYSTEM: Major subsystem (SR, BR, BTS)
- FAMILY: Device family by function (e.g., BPM, HCM, VCM, QF, QD, RF, VAC)
- FIELD: Measurement/control function (e.g., Monitor, Setpoint, X, Y, Pressure)
- SUBFIELD: (Optional) Nested structure (e.g., VAC.IonPump.Pressure, RF.PowerMonitor.Forward)
- ChannelNames: Actual PV addresses stored in database, indexed by device number

CRITICAL DIFFERENCES FROM HIERARCHICAL:
- PV addresses are RETRIEVED from database, NOT built from patterns
- Organization is by FUNCTION (Monitor/Setpoint) not naming convention
- Device filtering uses DeviceList metadata, not name parsing
- Some families have nested structure (e.g., VAC contains IonPump and Gauge subfamilies)

SYSTEM DESCRIPTIONS:
SR (Storage Ring):
- Main synchrotron light source operating at 1.9 GeV, 12-sector ring
- Magnet families: HCM, VCM (correctors), QF, QD (quadrupoles), SF, SD (sextupoles), BEND (dipoles)
- Diagnostics: BPM (beam position), DCCT (beam current), Tune (betatron tune)
- RF system with 2 cavities
- Vacuum system: VAC.IonPump (72 pumps), VAC.Gauge (24 gauges)
- ID (Insertion Devices): undulators/wigglers in straight sections
- Thermocouple temperature monitoring
- Scraper beam collimators

BR (Booster Ring):
- Accelerates beam from 150 MeV to 1.9 GeV before injection
- Contains BPM, DCCT, Dipole, Quadrupole families

BTS (Booster-to-Storage Transfer Line):
- Beam transport between booster and storage ring (~35m)
- Contains BPM, HCM, VCM, Quadrupole, Kicker, Septum, TransferEfficiency

DEVICE FAMILIES AND FUNCTIONS:

Beam Position Monitors (BPM):
- X = horizontal beam position (mm)
- Y = vertical beam position (mm)
- SumSignal = total beam signal intensity
- XError, YError = position measurement uncertainty (μm)

Corrector Magnets (HCM/VCM):
- Monitor = readback current (A)
- Setpoint = desired current (A)
- OnControl = enable/disable control (HCM only)
- Ready = power supply ready status

Quadrupole Magnets (QF/QD):
- QF = Focusing quadrupoles, QD = Defocusing quadrupoles
- Monitor = readback current (A)
- Setpoint = desired current (A)
- Ready = power supply ready status

Sextupole Magnets (SF/SD):
- SF = Focusing sextupoles, SD = Defocusing sextupoles
- Monitor = readback current (A)
- Setpoint = desired current (A)

Bending Dipoles (BEND):
- Monitor = dipole current readback (A) - defines beam energy
- Setpoint = single setpoint for all dipoles (SR:BEND:SetCurrent)
- Ready = main power supply ready status

Beam Current (DCCT):
- Monitor = beam current measurement (mA)
- FastMonitor = fast current readback for transients (10 kHz)
- Lifetime = calculated beam lifetime (hours)

RF System:
- FrequencyMonitor/FrequencySetpoint = RF frequency (MHz)
- VoltageMonitor/VoltageSetpoint = cavity gap voltage (kV)
- PowerMonitor.Forward, PowerMonitor.Reflected = RF power (kW)
- PhaseMonitor/PhaseSetpoint = RF phase (degrees)
- CavityTemperature = cavity body temperature (°C)

Vacuum System (SR.VAC):
- IonPump.Pressure = vacuum level from ion pumps (Torr)
- IonPump.Voltage = pump high voltage (kV)
- IonPump.Current = pump current (μA)
- Gauge.Pressure = direct gauge measurement (Torr)

Insertion Devices (ID):
- GapMonitor/GapSetpoint = undulator/wiggler gap (mm)
- Moving = motion status, PermissionGranted = beamline permission

Tune Measurement:
- HorizontalMonitor/HorizontalSetpoint = horizontal betatron tune
- VerticalMonitor/VerticalSetpoint = vertical betatron tune

Temperature Monitoring (Thermocouple):
- Temperature = chamber/absorber temperature (°C)
- Limit = temperature limit setpoint (°C)
- OverTemp = over-temperature flag

Beam Scrapers (Scraper):
- TopPosition/BottomPosition = scraper position readback (mm)
- TopSetpoint/BottomSetpoint = scraper position setpoint (mm)
- Moving = motion status
""".strip()
