# Corrector magnets

Corrector magnets apply small dipole kicks that steer the stored beam back onto
its reference orbit. Horizontal and vertical correctors are interleaved with the
quadrupoles around the lattice, and each one accepts a setpoint in milliradians
that the power supply converts into a current.

The orbit feedback writes to these setpoints continuously, so a corrector left
at an unusual value after a manual study is visible as a persistent local bump
in the closed orbit. Setpoint limits are enforced in the power supply itself
rather than in the feedback, which is what keeps a runaway correction from
depositing beam in the vacuum chamber.
