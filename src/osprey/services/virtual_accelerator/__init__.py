"""PyAT Virtual Accelerator: a CA-indistinguishable soft-IOC substrate.

Assembles the namespace-union channel manifest (:mod:`.manifest`), the PyAT
ring the deployment's own deck is loaded into (:mod:`.lattice`), and the
EPICS record/physics/engine-source assembly layer (:mod:`.ioc`) into one
soft-IOC process (see :mod:`.entrypoint`). Served over Channel Access, this
is indistinguishable from real hardware to any OSPREY connector -- never
special-cased.
"""
