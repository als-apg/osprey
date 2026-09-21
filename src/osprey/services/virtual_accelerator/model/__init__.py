"""LUME model layer for the virtual accelerator: the facility adapter.

The serving-free half of the VA, expressed as a LUME model plus its variable
catalog. Nothing here imports EPICS.

**Four layers, each knowing only the one below it.** ``lume`` states the
generic model contract (``LUMEModel``, ``ScalarVariable``). ``lume_pyat``
implements it over pyAT and knows only native pyAT quantities: one
persistent lattice, atomic multi-variable writes, one solve per batch,
rollback on a lost closed orbit. This package is the adapter between the two
-- :class:`~osprey.services.virtual_accelerator.model.pyat.PyATRingModel`,
the lattice bindings, the calibrated transformer variables, and the catalog
that derives them from the served tree. Above it sits the serving layer
(``ioc.physics_bridge``, ``serving``), which none of this reaches into.

**One facility is no more special than another.** What this adapter knows
about an accelerator it reads from the tree it is served: the lattice file,
the channel manifest, ``machine.json``, ``channel_limits.json``, and the
``va_bindings.json`` that pairs an address with the element it drives. No
family, element, attribute or axis is named in this package's code.

**The adapter contract**, in three rules. It is prose rather than a
``Protocol`` deliberately: with one backend in tree a formal interface would
only restate ``LUMEModel``'s, and the rules worth pinning are about which
side of the boundary a fact lives on.

1. *Variable names are the facility's control addresses.* Every model
   variable is keyed and named by its full six-level channel address, so
   nothing between the manifest, the IOC and the model translates addresses.
   That grammar stops here: ``bindings`` looks each address up in the served
   bindings document and hands the backend the element locator it finds
   there -- element names, attribute and component, one per slice, or a
   monitor and a transverse axis -- plus declarative fields in native pyAT
   units. Nothing parses an address, here or below.

2. *The serving layer depends on ``LUMEModel`` alone.* ``PhysicsBridge``
   reaches the ring through the model's public ``set()``/``get()`` and
   nothing else, so a different backend (a surrogate, Cheetah, Bmad) is
   injected through ``model=`` without the serving layer changing. Backends
   swap at the adapter layer, which is this package.

3. *Unit conversion is facility work.* ``lume_pyat``'s writable variable
   writes the value it is handed, unconverted. Hardware units to physics
   stays on this side, in
   :mod:`~osprey.services.virtual_accelerator.model.variables` -- one
   subclass per binding kind, each applying the calibration the facility
   exported for that channel. A second backend would re-implement the
   binding, never the calibration.

**One class per failure, not one per layer.**
``UnknownDeviceError`` *is* ``lume_pyat.exceptions.UnknownElementError`` --
an alias, never a subclass. The backend raises that class from every element
lookup it performs, at model construction and on the write path both, and
aliasing is what keeps an ``except UnknownDeviceError`` catching those as
well as any raised on this side of the boundary. ``OrbitSolveError`` is likewise
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

#: Public name -> the submodule of this package that defines it. Entries are
#: resolved on first attribute access, never at import.
_LAZY_EXPORTS: dict[str, str] = {
    "build_action_variables": ".bindings",
    "build_variable_catalog": ".catalog",
    "couple_energy_knob": ".bindings",
    "PyATRingModel": ".pyat",
    "UnknownDeviceError": ".pyat",
}

__all__ = list(_LAZY_EXPORTS)


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
