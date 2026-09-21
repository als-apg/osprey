#!/usr/bin/env python
"""Drive the virtual accelerator's pyAT ring through the LUME model interface.

The pluggability proof: no softioc, no EPICS, no IOC -- just a `LUMEModel`
whose `set()`/`get()`/`reset()` move a real lattice. Any facility that speaks
the LUME contract can consume this model directly, and any model that speaks
it can be served by the VA in place of this one.

Which device is driven and which readings are watched come out of the served
tree's own bindings document: the corrector is a binding the tree states
kicks the beam horizontally, and the readings are the monitors it states read
that same plane. So this runs against whatever tree it is pointed at, and the
ring it happens to serve names nothing here.

Writes 1.0 A to that corrector and shows three things:

1. `get()` of the corrector returns its nominal before the write and the
   written value after -- the model retains its inputs, which the IOC bridge
   alone never did.
2. The monitors move: a corrector kick produces a sign-alternating betatron
   oscillation around the ring, reported in metres.
3. `reset()` puts the inputs back to nominal and the orbit back to the one
   the ring was serving before the write. That orbit is not flat: a ring
   solved with radiation loses energy around the turn, and its monitors read
   the sawtooth that comes with it.

Run it from the worktree with the worktree venv:

    .venv/bin/python scripts/va/pyat_model_demo.py

Exits 0 if every assertion about the round trip holds, non-zero otherwise, so
it doubles as a smoke gate.
"""

from __future__ import annotations

import sys

DRIVE_CURRENT = 1.0
N_READINGS_SHOWN = 6

#: How far a driven corrector has to move the orbit for the write to count as
#: having reached the beam, and how exactly ``reset()`` has to put it back.
MOVED_ORBIT_M = 1e-6
RESTORED_ORBIT_M = 1e-12

#: The transverse plane driven and watched: the kick component the tree binds
#: a horizontal corrector to, and the axis its monitors read on that plane.
KICK_INDEX = 0
MONITOR_AXIS = "x"


def main() -> int:
    # Imported here, not at module scope, so --help-style failures surface
    # before the ~165 ms ring build.
    from osprey.services.virtual_accelerator.bindings import load_bindings
    from osprey.services.virtual_accelerator.manifest.build import build_manifest
    from osprey.services.virtual_accelerator.manifest.paths import PACKAGE_PATHS
    from osprey.services.virtual_accelerator.model.pyat import PyATRingModel

    document = load_bindings(PACKAGE_PATHS.va_bindings)
    corrector = next(
        binding.setpoint_address
        for binding in document.bindings
        if binding.kind == "kick" and binding.index == KICK_INDEX
    )
    readings = sorted(
        binding.setpoint_address
        for binding in document.bindings
        if binding.kind == "monitor" and binding.attribute == MONITOR_AXIS
    )

    print(f"Building PyATRingModel ({document.system} ring, no softioc)...")
    model = PyATRingModel(PACKAGE_PATHS.data_root, build_manifest()["channels"])

    catalog = model.supported_variables
    inputs = [name for name, var in catalog.items() if not var.read_only]
    outputs = [name for name, var in catalog.items() if var.read_only]
    print(f"  supported_variables: {len(inputs)} inputs + {len(outputs)} read-only outputs")

    corrector_var = catalog[corrector]
    print(f"  {corrector}  range={corrector_var.value_range} unit={corrector_var.unit!r}")

    # 1. nominal in, nothing moved yet
    nominal = model.get(corrector)
    orbit_before = model.get(readings)
    print(f"\nBefore write: get({corrector}) = {nominal} A")
    print(f"              max |reading| = {max(abs(v) for v in orbit_before.values()):.3e} m")

    # 2. write, and read the moved orbit back
    model.set({corrector: DRIVE_CURRENT})
    after = model.get(corrector)
    orbit_after = model.get(readings)
    print(f"\nAfter set({DRIVE_CURRENT} A): get({corrector}) = {after} A   <- input retained")
    print(f"              max |reading| = {max(abs(v) for v in orbit_after.values()):.3e} m")
    print(f"\n  first {N_READINGS_SHOWN} readings (metres), showing the alternating kick:")
    for name in readings[:N_READINGS_SHOWN]:
        print(f"    {name}  {orbit_after[name]:+.4e}")

    # 3. reset puts inputs and orbit back
    model.reset()
    restored = model.get(corrector)
    orbit_reset = model.get(readings)
    print(f"\nAfter reset(): get({corrector}) = {restored} A")
    print(f"              max |reading| = {max(abs(v) for v in orbit_reset.values()):.3e} m")

    # Assertions -- this is a gate, not just a printout. The orbit is read
    # against the nominal one the tree serves rather than against zero: a
    # monitor reads where the beam is on that ring, and a ring whose nominal
    # orbit is not flat is a ring, not a fault.
    moved = max(abs(orbit_after[name] - orbit_before[name]) for name in readings)
    restored_error = max(abs(orbit_reset[name] - orbit_before[name]) for name in readings)
    print(f"\n  orbit moved by {moved:.3e} m, restored to {restored_error:.3e} m")

    assert nominal == corrector_var.default_value, "input did not start at its nominal"
    assert after == DRIVE_CURRENT, "input was not retained"
    assert moved > MOVED_ORBIT_M, "corrector write did not move the monitors"
    assert restored == corrector_var.default_value, "reset did not restore the nominal input"
    assert restored_error < RESTORED_ORBIT_M, "reset did not restore the orbit it started from"

    print("\nOK: set/get/reset round trip through the LUME interface, softioc never imported.")
    assert "softioc" not in sys.modules and "cothread" not in sys.modules
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
