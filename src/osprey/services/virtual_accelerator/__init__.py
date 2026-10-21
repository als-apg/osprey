"""PyAT Virtual Accelerator: a CA-indistinguishable soft-IOC substrate.

Serves the simulator view a build writes from one process (see
:mod:`.entrypoint`), through the serving layer in :mod:`.serving`. Served over
Channel Access, this is indistinguishable from real hardware to any OSPREY
connector -- never special-cased.
"""
