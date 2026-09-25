"""Still the declared motion of the monitor readings a test machine serves.

The virtual accelerator serves a lattice-bound monitor as the solved orbit plus
the drift and noise its ``machine.json`` entry declares. A suite whose oracle is
the noiseless model -- a measured response equal to the in-process solve, a
served reading equal to the model's truth -- needs readings that are the orbit
and nothing else, so it stills them in the data tree it deploys, before that
tree is staged or mounted. Every other channel keeps what the file declares.

Shared by the deploy-backed lanes (``tests/e2e``) and the live-container suite
(``tests/va/e2e``). Imports nothing that serves Channel Access.
"""

from __future__ import annotations

import json
from pathlib import Path

from osprey.services.virtual_accelerator.bindings import load_bindings
from osprey.services.virtual_accelerator.manifest.paths import ManifestPaths

#: The machine-file keys that move a reading on its own: slow drift, relative
#: noise, absolute noise.
MOTION_KEYS = ("texture", "noise", "noise_abs")


def still_monitor_motion(data_root: Path) -> frozenset[str]:
    """Remove the declared motion from every monitor reading ``data_root`` serves.

    Args:
        data_root: The facility data root: the directory whose
            ``simulation/`` holds ``machine.json`` and ``va_bindings.json``.

    Returns:
        The monitor addresses whose declared motion was removed.
    """
    paths = ManifestPaths(data_root)
    assert paths.machine_json.is_file(), f"no machine file at {paths.machine_json}"
    assert paths.va_bindings.is_file(), f"no bindings document at {paths.va_bindings}"
    monitors = {
        address
        for binding in load_bindings(paths.va_bindings).bindings
        if binding.kind == "monitor"
        for address in (binding.setpoint_address, binding.readback_address)
        if address is not None
    }
    machine = json.loads(paths.machine_json.read_text(encoding="utf-8"))
    stilled: set[str] = set()
    for address in monitors:
        entry = machine["channels"].get(address)
        if not isinstance(entry, dict):
            continue
        for key in MOTION_KEYS:
            if entry.pop(key, None):
                stilled.add(address)
    paths.machine_json.write_text(json.dumps(machine, indent=2) + "\n", encoding="utf-8")
    return frozenset(stilled)
