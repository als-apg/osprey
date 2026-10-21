pyAML measurement steps and `set_and_wait` wait for each setpoint's readback
through `osprey.runtime.settle()`: within the facility's stated tolerance, else
on the simulator within the readback's motion band.
