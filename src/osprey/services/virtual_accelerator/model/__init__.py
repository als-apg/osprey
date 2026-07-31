"""LUME model layer for the virtual accelerator.

The in-tree seed of a standalone `lume-pyat` package: the softioc-free
half of the VA expressed as a LUME model plus its variable catalog. Nothing
here imports EPICS.

Everywhere else in this service a heavy import is deferred plainly, with an
`import` statement inside the function that needs it (see
`entrypoint.py`'s deferred `physics_bridge` import). A package `__init__`
cannot do that and still offer flat names, so this one uses the PEP 562
module-level `__getattr__` instead: `import
osprey.services.virtual_accelerator.model` must stay free of `at` and
`lume` (importing `lume` eagerly drags in h5py, matplotlib, scipy and
numpy), because the `VA_LATTICE=none` boot path imports this package's
siblings and must never pay for -- or hard-depend on -- any of it.
"""

from typing import Any

# name -> submodule it lives in. Extend here when adding an export.
_LAZY_EXPORTS = {
    "build_variable_catalog": ".catalog",
    "PyATRingModel": ".pyat",
    "UnknownDeviceError": ".pyat",
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_name, __name__), name)


def __dir__() -> list[str]:
    return sorted([*globals(), *_LAZY_EXPORTS])
