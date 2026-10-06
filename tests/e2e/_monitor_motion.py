"""Still the declared motion of the monitor readings a test facility serves.

A served monitor reading is the model's solved orbit plus the noise and drift
its record in the facility's ``seeds.yaml`` declares. A suite whose oracle is
the noiseless model -- a measured response equal to the in-process solve, a
served reading equal to the model's truth -- needs readings that are the orbit
and nothing else, so it stills them in the facility tree it deploys, before
that tree is built or mounted. A monitor reading is an address whose wiring
record in ``models.yaml`` reads an ``axis``; every other seed keeps what the
file declares.

Shared by the deploy-backed lanes (``tests/e2e``) and the live-container suite
(``tests/va/e2e``). Imports nothing that serves Channel Access.
"""

from __future__ import annotations

from pathlib import Path

import yaml

#: The seed keys that move a reading on its own: white noise and slow drift.
MOTION_KEYS = ("noise", "drift")


def still_monitor_motion(data_root: Path) -> frozenset[str]:
    """Remove the declared motion from every monitor reading ``data_root`` serves.

    Args:
        data_root: The data root: the directory whose ``facility/`` holds
            ``seeds.yaml`` and ``models.yaml``.

    Returns:
        The monitor addresses whose declared motion was removed.
    """
    facility = data_root / "facility"
    seeds_yaml = facility / "seeds.yaml"
    models_yaml = facility / "models.yaml"
    assert seeds_yaml.is_file(), f"no seeds file at {seeds_yaml}"
    assert models_yaml.is_file(), f"no models file at {models_yaml}"
    models = yaml.safe_load(models_yaml.read_text(encoding="utf-8")) or []
    monitors = {
        str(record["address"])
        for model in models
        for record in model.get("wiring") or []
        if "axis" in (record.get("engine") or {})
    }
    seeds = yaml.safe_load(seeds_yaml.read_text(encoding="utf-8")) or {}
    stilled: set[str] = set()
    for address in monitors:
        seed = seeds.get(address)
        if not isinstance(seed, dict):
            continue
        for key in MOTION_KEYS:
            if key in seed:
                del seed[key]
                stilled.add(address)
    seeds_yaml.write_text(yaml.safe_dump(seeds, sort_keys=True), encoding="utf-8")
    return frozenset(stilled)
