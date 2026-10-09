"""Still the declared motion of the monitor readings a test facility serves.

A served monitor reading is the model's solved orbit plus the noise and drift
its record in the facility's ``seeds.yaml`` declares. A suite whose oracle is
the noiseless model -- a measured response equal to the in-process solve, a
served reading equal to the model's truth -- needs readings that are the orbit
and nothing else, so it stills them in the facility tree it deploys, before
that tree is built or mounted. A monitor reading is an address whose wiring
record in ``models.yaml`` the model's engine describes as a ``monitor``, or as
an ``output`` of one plane (a tune or a chromaticity); every other seed keeps
what the file declares. A suite that also reads back the devices it drives stills every
reading the model serves instead (:func:`still_model_motion`).

Shared by the deploy-backed lanes (``tests/e2e``) and the live-container suite
(``tests/va/e2e``). Imports nothing that serves Channel Access.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from osprey_connectors.simulation.view import TEXTURE

#: The seed keys that move a reading on its own: white noise and slow drift.
MOTION_KEYS = ("noise", "drift")


@cache
def _describer(engine: str) -> Callable[[Mapping[str, Any]], Mapping[str, Any]]:
    """The ``describe`` of the engine plug-in named ``engine``, through its entry point."""
    from importlib import metadata

    from osprey.simulation.engines import ENTRY_POINT_GROUP

    describe: Callable[[Mapping[str, Any]], Mapping[str, Any]] = (
        metadata.entry_points(group=ENTRY_POINT_GROUP)[engine].load().describe
    )
    return describe


def _is_monitor_reading(engine: str, record: Mapping[str, Any]) -> bool:
    """Whether the source wiring ``record`` reads a monitor or one plane of an optics output.

    The facility source states no direction, so the record is described as a
    reading; a setting record then describes as its readback and is not one.
    """
    described = _describer(engine)({**record, "direction": "read"})
    return described["role"] == "monitor" or (
        described["role"] == "output" and described["plane"] is not None
    )


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
        if model.get("engine") != TEXTURE and model.get("wiring")
        for record in model["wiring"]
        if _is_monitor_reading(str(model["engine"]), record)
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


def still_model_motion(data_root: Path) -> frozenset[str]:
    """Remove the declared motion from every reading the model serves at ``data_root``.

    A suite whose oracle is the noiseless model, and which reads back the
    devices it drives, stills every reading the model serves: every
    ``address`` of every ``wiring`` record in ``models.yaml`` -- monitors,
    readbacks and setpoints alike. Only the motion keys of those seeds are
    removed; every other key, and every seed no model wires, keeps what the
    file declares.

    Args:
        data_root: The data root: the directory whose ``facility/`` holds
            ``seeds.yaml`` and ``models.yaml``.

    Returns:
        The wired addresses whose declared motion was removed.
    """
    facility = data_root / "facility"
    seeds_yaml = facility / "seeds.yaml"
    models_yaml = facility / "models.yaml"
    assert seeds_yaml.is_file(), f"no seeds file at {seeds_yaml}"
    assert models_yaml.is_file(), f"no models file at {models_yaml}"
    models = yaml.safe_load(models_yaml.read_text(encoding="utf-8")) or []
    wired = {str(record["address"]) for model in models for record in model.get("wiring") or []}
    seeds = yaml.safe_load(seeds_yaml.read_text(encoding="utf-8")) or {}
    stilled: set[str] = set()
    for address in wired:
        seed = seeds.get(address)
        if not isinstance(seed, dict):
            continue
        for key in MOTION_KEYS:
            if key in seed:
                del seed[key]
                stilled.add(address)
    seeds_yaml.write_text(yaml.safe_dump(seeds, sort_keys=True), encoding="utf-8")
    return frozenset(stilled)
