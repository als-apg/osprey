"""EPICS IOC assembly layer for the PyAT virtual accelerator.

:mod:`ioc.records` is the pure record-construction factory: it consumes the
namespace-union manifest and builds typed pythonSoftIOC records, exposing
callback slots for the physics bridge (partition a) and engine source
(partition c) to plug into without this package depending on either.

:mod:`ioc.physics_bridge` fills the partition (a) callback slot:
`PhysicsBridge.on_setpoint` is passed as `build_records()`'s
`on_pyat_setpoint`, and `PhysicsBridge.bind()` wires the returned
`IOCRecords.pyat_coupled` BPM records to receive recomputed positions.
"""

from .physics_bridge import OrbitSolveError, PhysicsBridge, UnknownDeviceError
from .records import IOCRecords, ManifestContractError, build_records

__all__ = [
    "IOCRecords",
    "ManifestContractError",
    "build_records",
    "PhysicsBridge",
    "OrbitSolveError",
    "UnknownDeviceError",
]
