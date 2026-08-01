#!/usr/bin/env python
"""Drive the virtual accelerator's pyAT ring through the LUME model interface.

The pluggability proof: no softioc, no EPICS, no IOC -- just a `LUMEModel`
whose `set()`/`get()`/`reset()` move a real ALS-U AR lattice. Any facility that
speaks the LUME contract can consume this model directly, and any model that
speaks it can be served by the VA in place of this one.

Writes 1.0 A to a horizontal corrector and shows three things:

1. `get()` of the corrector returns its machine.json nominal before the write
   and the written value after -- the model retains its inputs, which the IOC
   bridge alone never did.
2. The BPMs move: a corrector kick produces the sign-alternating ~9 um/A
   betatron oscillation around the ring, reported in metres.
3. `reset()` puts the inputs back to nominal and the orbit back to zero.

Run it from the worktree with the worktree venv:

    .venv/bin/python scripts/va/pyat_model_demo.py

Exits 0 if every assertion about the round trip holds, non-zero otherwise, so
it doubles as a smoke gate.
"""

from __future__ import annotations

import sys

CORRECTOR = "SR:MAG:HCM:01:CURRENT:SP"
DRIVE_CURRENT = 1.0
N_BPMS_SHOWN = 6


def main() -> int:
    # Imported here, not at module scope, so --help-style failures surface
    # before the ~165 ms ring build.
    from osprey.services.virtual_accelerator.model import build_variable_catalog  # noqa: F401
    from osprey.services.virtual_accelerator.model.pyat import PyATRingModel

    print("Building PyATRingModel (ALS-U AR ring, no softioc)...")
    model = PyATRingModel()

    catalog = model.supported_variables
    inputs = [name for name, var in catalog.items() if not var.read_only]
    outputs = [name for name, var in catalog.items() if var.read_only]
    print(f"  supported_variables: {len(inputs)} inputs + {len(outputs)} read-only outputs")

    bpm_x = sorted(name for name in outputs if name.endswith(":POSITION:X"))
    corrector_var = catalog[CORRECTOR]
    print(f"  {CORRECTOR}  range={corrector_var.value_range} unit={corrector_var.unit!r}")

    # 1. nominal in, nothing moved yet
    nominal = model.get(CORRECTOR)
    orbit_before = model.get(bpm_x)
    print(f"\nBefore write: get({CORRECTOR}) = {nominal} A")
    print(f"              max |BPM x| = {max(abs(v) for v in orbit_before.values()):.3e} m")

    # 2. write, and read the moved orbit back
    model.set({CORRECTOR: DRIVE_CURRENT})
    after = model.get(CORRECTOR)
    orbit_after = model.get(bpm_x)
    print(f"\nAfter set({DRIVE_CURRENT} A): get({CORRECTOR}) = {after} A   <- input retained")
    print(f"              max |BPM x| = {max(abs(v) for v in orbit_after.values()):.3e} m")
    print(f"\n  first {N_BPMS_SHOWN} BPM x readings (metres), showing the alternating kick:")
    for name in bpm_x[:N_BPMS_SHOWN]:
        print(f"    {name}  {orbit_after[name]:+.4e}")

    # 3. reset puts inputs and orbit back
    model.reset()
    restored = model.get(CORRECTOR)
    orbit_reset = model.get(bpm_x)
    print(f"\nAfter reset(): get({CORRECTOR}) = {restored} A")
    print(f"              max |BPM x| = {max(abs(v) for v in orbit_reset.values()):.3e} m")

    # Assertions -- this is a gate, not just a printout.
    assert nominal == corrector_var.default_value, "input did not start at its nominal"
    assert after == DRIVE_CURRENT, "input was not retained"
    assert max(abs(v) for v in orbit_before.values()) < 1e-12, "nominal orbit was not flat"
    assert max(abs(v) for v in orbit_after.values()) > 1e-6, "corrector write did not move the BPMs"
    assert restored == corrector_var.default_value, "reset did not restore the nominal input"
    assert max(abs(v) for v in orbit_reset.values()) < 1e-12, "reset did not restore the orbit"

    print("\nOK: set/get/reset round trip through the LUME interface, softioc never imported.")
    assert "softioc" not in sys.modules and "cothread" not in sys.modules
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
