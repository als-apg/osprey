"""Facility-specific description for UCSB Free Electron Laser.

This file contains facility-specific information that users should customize
for their own control system. It describes:
- What the facility is and how it operates
- Major subsystems and their functions
- Beamline classifications

To customize for your facility:
1. Copy this file to your facility's prompts directory
2. Replace the content with your facility's description
3. Update subsystem descriptions to match your accelerator
"""

FACILITY_DESCRIPTION = """
The University of California, Santa Barbara (UCSB) Free Electron Laser (FEL) uses relativistic electrons to generate a powerful terahertz (THz) laser beam. The accelerator and its control system coordinate several key subsystems to create, accelerate, guide, and recover the electron beam.

1. Electron Source (Thermionic Gun):
- Electrons are emitted from a thermionic cathode in short pulses.
- These pulses form the initial beam that enters the accelerator.
- Key parameters: filament current, gun voltage, grid voltage, beam pulse duration

2. Acceleration Section:
- The electrons are accelerated down the main accelerating tube by the high terminal voltage applied to a series of accelerator plates.
- Control parameters include gun voltage, beam pulse timing, and accelerator voltage stability.
- Grading resistors distribute voltage evenly across accelerator plates (diagnostic currents: Acc_I for accelerating tube, I_mtrs for decelerating tube)

3. Terminal Voltage Regulation:
- The terminal voltage is regulated by a corona triode system with adjustable needles.
- RegulatingNeedles control the needle exposure distance for sink current
- CoronaTriodCurrent and CoronaTriodError are measurement/diagnostic values for the regulation loop
- The terminal voltage is measured capacitively by a generating voltmeter (CapacitivePickup for AC component)

4. Pelletron Charging System:
- Pelletron chains carry charge to build up terminal voltage
- ChargingVoltageTop: High voltage power supply in the terminal charging the chains
- ChargingCurrentBottom: Power supply at bottom of tank for chain charging
- Both have Set and ReadBack channels

5. Beam Transport and Steering:
- Steering coils exist in TWO locations with DIFFERENT channel types:
  * IN-TANK steering coils (inside accelerator/decelerator tubes): SX3, SY3, SX40, SY40 (accelerating tube), SX233, SY233 (decelerating tube) - these are standalone channels with Set/RB suffixes
  * BEAMLINE steering coils (outside tank, in beam transport): SteeringCoil01-19 templates with X/Y SetPoint/ReadBack sub-channels
- When query asks for steering coils "inside the tank", return ONLY the SX/SY standalone channels
- When query asks for steering coils in general or by number, use the SteeringCoil templates
- Dipole magnets (DipoleMagnet01-09) bend the beam trajectory along the beam path
- Quadrupole magnets (QuadrupoleMagnet01-17) focus or defocus the beam in the beamlines

6. Focusing Elements:
- Focusing SOLENOIDS are used INSIDE the tubes: L39 (accelerating tube), L234 (decelerating tube)
- These are different from quadrupole magnets which are in the beamlines
- "Focusing elements in tubes" or "focusing solenoids" = L39, L234
- Quadrupole magnets are for beamline focusing, not in-tube focusing

7. Beam Diagnostics:
- Current monitors (CurrentMonitor01-05): Non-destructive beam current measurement waveforms
- Observation screens (ObservationScreen01-23): Beam profile imaging with Motor control and Image channels
- Operators use these diagnostics to align and optimize the beam transport system.

8. Undulator Section (THz Light Generation):
- The accelerated beam is sent through an undulator, where it oscillates transversely in a periodic magnetic field.
- This motion induces the emission of terahertz radiation, forming the FEL output.

9. Energy Recovery (Deceleration Section):
- After the undulator, the electrons travel back up a decelerating tube, returning part of their energy to the accelerator system.
- The recycled electrons are then recaptured at the top of the machine, improving overall efficiency.

10. Terminal Infrastructure:
- ACPowerInTerminal: AC power from rotating generator (terminal power source)
- FansVoltage: Cooling fans for terminal electronics

Beamline Classifications:
- "FEL beamlines" specifically refers to the THREE FEL output beamlines:
  * HFEL (High-power FEL beamline)
  * FIR (Far Infrared beamline)
  * MM (Millimeter wave beamline)
- "Beam transport" or "transport beamlines" refers to:
  * beamline1 (main transport from accelerating tube)
  * beamline2 (return transport to decelerating tube)
  * xray beamline (diagnostic branch)
- If query says "FEL beamlines", return ONLY the three FEL output beamlines
- If query says "all beamlines", include both FEL output and transport beamlines
""".strip()
