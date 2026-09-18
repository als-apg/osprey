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

#: Public name -> the submodule of this package that defines it. Entries are
#: resolved on first attribute access, never at import.
_LAZY_EXPORTS: dict[str, str] = {
    "PhysicsBridge": ".physics_bridge",
    "OrbitSolveError": ".physics_bridge",
    "UnknownDeviceError": ".physics_bridge",
}


def __getattr__(name: str) -> Any:
    """Resolve a public name from its defining module on first access."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})
