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

from typing import Any

from .records import IOCRecords, ManifestContractError, build_records

__all__ = [
    "IOCRecords",
    "ManifestContractError",
    "build_records",
    "PhysicsBridge",
    "OrbitSolveError",
    "UnknownDeviceError",
]

# The physics-bridge names are re-exported lazily (PEP 562):
# physics_bridge.py imports PyAT at module level, and importing this package
# must not require PyAT -- the no-lattice entrypoint path (VA_LATTICE=none)
# imports sibling modules (records, engine_source) from a process where PyAT
# may not even be installed. Attribute access is unchanged for callers:
# ``from ...ioc import PhysicsBridge`` still works, it just pays the PyAT
# import only when actually used.
_PHYSICS_BRIDGE_NAMES = frozenset({"PhysicsBridge", "OrbitSolveError", "UnknownDeviceError"})


def __getattr__(name: str) -> Any:
    if name in _PHYSICS_BRIDGE_NAMES:
        from . import physics_bridge

        return getattr(physics_bridge, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
