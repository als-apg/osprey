"""LUME model layer for the virtual accelerator: the ALS-U facility adapter.

The serving-free half of the VA, expressed as a LUME model plus its variable
catalog. Nothing here imports EPICS.

**Four layers, each knowing only the one below it.** ``lume`` states the
generic model contract (``LUMEModel``, ``ScalarVariable``). ``lume_pyat``
implements it over pyAT and is facility-agnostic: one persistent lattice,
atomic multi-variable writes, one solve per batch, rollback on a lost closed
orbit, and no knowledge of ALS-U at all. This package is the ALS-U adapter
-- :class:`~osprey.services.virtual_accelerator.model.pyat.PyATRingModel`,
the lattice bindings, the current->strength transformer variables, and the
catalog that derives them from the facility databases. Above it sits the
serving layer (``ioc.physics_bridge``, ``serving``), which none of this
reaches into.

**The adapter contract**, in three rules. It is prose rather than a
``Protocol`` deliberately: with one backend in tree a formal interface would
only restate ``LUMEModel``'s, and the rules worth pinning are about which
side of the boundary a fact lives on.

1. *Variable names are the facility's control addresses.* Every model
   variable is keyed and named by its full six-level channel address, so
   nothing between the manifest, the IOC and the model translates addresses.
   That grammar stops here: ``bindings`` resolves each address to an element
   locator -- an ``element_name``/``attribute`` pair, or a monitor and a
   transverse axis -- and the backend receives only that plus declarative
   fields in native pyAT units. ``lume_pyat`` parses nothing out of a name.

2. *The serving layer depends on ``LUMEModel`` alone.* ``PhysicsBridge``
   reaches the ring through the model's public ``set()``/``get()`` and
   nothing else, so a different backend (a surrogate, Cheetah, Bmad) is
   injected through ``model=`` without the serving layer changing. Backends
   swap at the adapter layer, which is this package.

3. *Unit conversion is facility work.* ``lume_pyat``'s writable variable
   writes the value it is handed, unconverted. Amps->strength stays on this
   side as
   :class:`~osprey.services.virtual_accelerator.model.variables.CurrentSetpointVariable`,
   a subclass whose ``_set`` calls the unchanged ``StrengthMap.apply``. A
   second backend would re-implement the binding, never the calibration.

**One class per failure, not one per layer.**
``UnknownDeviceError`` *is* ``lume_pyat.exceptions.UnknownElementError`` --
an alias, never a subclass. ``CurrentSetpointVariable._set`` raises the
``lume_pyat`` class directly, because importing ``model.pyat`` there would
close a ``pyat -> bindings -> variables -> pyat`` cycle; aliasing is what
keeps an ``except UnknownDeviceError`` catching every lookup failure the
stack can produce, wherever it was raised. ``OrbitSolveError`` is likewise
one class, canonically ``lume_pyat.exceptions``, re-exported by
``lattice.solve`` and ``ioc.physics_bridge`` rather than restated -- so a
caller catching either re-export catches the model's own failure too.

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
