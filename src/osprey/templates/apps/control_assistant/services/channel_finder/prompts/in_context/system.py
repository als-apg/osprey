import textwrap


facility_description = textwrap.dedent("""
    The University of California, Santa Barbara (UCSB) Free Electron Laser (FEL) uses relativistic electrons to generate a powerful terahertz (THz) laser beam. The accelerator and its control system coordinate several key subsystems to create, accelerate, guide, and recover the electron beam.
    1. Electron Source (Thermionic Gun):
    - Electrons are emitted from a thermionic cathode in short pulses.
    - These pulses form the initial beam that enters the accelerator.
    2. Acceleration Section:
    - The electrons are accelerated down the main accelerating tube by the high terminal voltage applied to a series of accelerator plates.
    - Control parameters include gun voltage, beam pulse timing, and accelerator voltage stability.
    3. Beam Transport and Steering:
    - By the time the beam reaches the steering coil at the bottom of the accelerator tank, it has achieved maximum velocity.
    - Steering coils and dipole magnets control the beam trajectory, allowing precise direction changes along the beam path.
    - Quadrupole magnets focus or defocus the beam to maintain proper spot size and emittance.
    4. Beam Diagnostics:
    - Current monitors, viewing screens, and other diagnostics confirm that the beam remains properly centered and tuned.
    - Operators use these diagnostics to align and optimize the beam transport system.
    5. Undulator Section (THz Light Generation):
    - The accelerated beam is sent through an undulator, where it oscillates transversely in a periodic magnetic field.
    - This motion induces the emission of terahertz radiation, forming the FEL output.
    6. Energy Recovery (Deceleration Section):
    - After the undulator, the electrons travel back up a decelerating tube, returning part of their energy to the accelerator system.
    - The recycled electrons are then recaptured at the top of the machine, improving overall efficiency.
    
    IMPORTANT TERMINOLOGY AND CONVENTIONS:
    
    Channel Naming Patterns:
    - "Motor" channels = Control/command channels (for setting positions or states)
    - "MotorReadBack" or "ReadBack" channels = Status/measurement channels (actual positions or states)
    - "SetPoint" or "Set" channels = Control values to be commanded
    - When query asks for "control" or "motor control", return ONLY Motor/Set channels, NOT readbacks
    - When query asks for "status" or "readback" or "actual", return ONLY ReadBack channels
    - When query asks to "check" or is ambiguous, include both Set and ReadBack
    
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
    - If query says "all beamlines", include both FEL output and transport beamlines""")