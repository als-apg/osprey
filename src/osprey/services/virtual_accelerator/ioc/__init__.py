"""Physics and telemetry sources for the virtual accelerator's serving layer.

Record construction lives in :mod:`..serving.pvdb`, which turns the
namespace-union manifest into the served PV database. This package holds the
two value sources that drive those PVs.

:mod:`ioc.physics_bridge` handles partition (a) (pyat-coupled):
`PhysicsBridge.on_setpoint` is the SR magnet setpoint handler the serving
write path calls, and `PhysicsBridge.bind()` wires the pyat-coupled BPM
channels to receive recomputed positions.

:mod:`ioc.engine_source` handles partition (c) (static/noisy), pushing
simulation-engine values onto their channels on a poll tick.

This package's ``__init__`` deliberately imports neither submodule eagerly:
``physics_bridge`` needs PyAT, and the no-lattice entrypoint path
(``VA_LATTICE=none``) imports ``engine_source`` from a process where PyAT may
not be installed.
"""

from typing import Any

__all__ = [
    "PhysicsBridge",
    "OrbitSolveError",
    "UnknownDeviceError",
]

# The physics-bridge names are re-exported lazily (PEP 562); see the module
# docstring for why importing this package must not pull PyAT in. Attribute
# access is unchanged for callers: ``from ...ioc import PhysicsBridge`` still
# works, it just pays the PyAT import only when actually used.
_PHYSICS_BRIDGE_NAMES = frozenset({"PhysicsBridge", "OrbitSolveError", "UnknownDeviceError"})


def __getattr__(name: str) -> Any:
    if name in _PHYSICS_BRIDGE_NAMES:
        from . import physics_bridge

        return getattr(physics_bridge, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
