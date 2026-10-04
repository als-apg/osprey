#!/usr/bin/env python3
"""The demo's ``scenarios/<name>.yaml``: each simulation bundle, translated.

Every directory under ``data/simulation/scenarios/`` becomes one scenario file
under ``data/facility/scenarios/`` of the control-assistant preset:

* ``description``, ``overrides``, ``archiver``, ``drivers``, ``couple`` and
  ``noise`` are carried as they are;
* the bundle's ``logbook.json`` becomes ``logbook``;
* ``physics`` becomes ``faults {<model>: {<address>: {<fault field>: value}}}``.
  A per-element monitor error names no plane, so ``offset``, ``gain``,
  ``polarity`` and ``noise`` land on both the x and the y reading of the
  element, and ``roll`` on its x reading. A ``corrector_gain`` factor becomes
  the ``cal_factor`` of the setpoint wired to that element. The wiring record
  whose ``element`` is the element's name gives each address.

A bundle's ``_comment`` is not carried. Keys are written in the Scenario
record's slot order; fault addresses are sorted.

Run it with the project interpreter (``uv run python``); it overwrites the
scenario files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

_DATA = REPO_ROOT / "src/osprey/templates/apps/control_assistant/data"
#: The bundles this module translates.
BUNDLES = _DATA / "simulation" / "scenarios"
#: The facility tree the translations are written into.
FACILITY = _DATA / "facility"

#: The Scenario record's slots, in the order a file states them.
SLOTS = ("description", "overrides", "faults", "archiver", "logbook", "drivers", "couple", "noise")

#: Monitor error field -> the readings of the element it lands on.
_MONITOR_FIELDS: dict[str, tuple[str, ...]] = {
    "offset": ("x", "y"),
    "gain": ("x", "y"),
    "polarity": ("x", "y"),
    "roll": ("x",),
    "noise": ("x", "y"),
}

#: The fault field a corrector's gain factor becomes on its setpoint.
_CORRECTOR_FIELD = "cal_factor"


class Wiring:
    """Each element's wired addresses, read from the facility tree."""

    def __init__(self, facility: Path) -> None:
        models = yaml.safe_load((facility / "models.yaml").read_text(encoding="utf-8"))
        channels = yaml.safe_load(
            (facility / "records" / "channels.yaml").read_text(encoding="utf-8")
        )
        setpoints = {row["id"] for row in channels if row.get("role") == "setpoint"}
        #: element -> (model, {axis: address}) for monitor readings.
        self.monitors: dict[str, tuple[str, dict[str, str]]] = {}
        #: element -> (model, setpoint address).
        self.setpoints: dict[str, tuple[str, str]] = {}
        for model in models:
            for record in model.get("wiring") or []:
                element = record.get("element")
                if element is None:
                    continue
                engine = record.get("engine") or {}
                address = record["address"]
                if "axis" in engine and "attribute" not in engine:
                    entry = self.monitors.setdefault(element, (model["name"], {}))
                    entry[1][engine["axis"]] = address
                elif address in setpoints:
                    self.setpoints.setdefault(element, (model["name"], address))

    def monitor(self, element: str) -> tuple[str, dict[str, str]]:
        """The model and the axis -> address readings of one monitor element."""
        if element not in self.monitors:
            raise SystemExit(f"no wiring reads monitor element {element}")
        return self.monitors[element]

    def setpoint(self, element: str) -> tuple[str, str]:
        """The model and the setpoint address wired to one element."""
        if element not in self.setpoints:
            raise SystemExit(f"no setpoint is wired to element {element}")
        return self.setpoints[element]


def faults(physics: dict[str, Any], wiring: Wiring) -> dict[str, dict[str, Any]]:
    """A bundle's ``physics`` block as per-address faults, by model.

    Args:
        physics: The bundle's ``physics`` block.
        wiring: The facility tree's wiring.

    Returns:
        ``{<model>: {<address>: {<fault field>: value}}}``, addresses sorted.
    """
    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for element, errors in (physics.get("bpm_errors") or {}).items():
        model, readings = wiring.monitor(element)
        for field, value in errors.items():
            for axis in _MONITOR_FIELDS[field]:
                fields = by_model.setdefault(model, {}).setdefault(readings[axis], {})
                fields[field] = value
    for element, factor in (physics.get("corrector_gain") or {}).items():
        model, address = wiring.setpoint(element)
        by_model.setdefault(model, {}).setdefault(address, {})[_CORRECTOR_FIELD] = factor
    return {model: dict(sorted(targets.items())) for model, targets in sorted(by_model.items())}


def translate(bundle: Path, wiring: Wiring) -> dict[str, Any]:
    """One bundle directory as a scenario record, without its name.

    Args:
        bundle: The bundle directory, holding ``scenario.json`` and maybe
            ``logbook.json``.
        wiring: The facility tree's wiring.

    Returns:
        The scenario's slots, in :data:`SLOTS` order.
    """
    spec = json.loads((bundle / "scenario.json").read_text(encoding="utf-8"))
    slots: dict[str, Any] = {key: spec[key] for key in SLOTS if key in spec}
    if spec.get("physics"):
        slots["faults"] = faults(spec["physics"], wiring)
    logbook = bundle / "logbook.json"
    if logbook.is_file():
        slots["logbook"] = json.loads(logbook.read_text(encoding="utf-8"))
    return {key: slots[key] for key in SLOTS if key in slots}


def files(bundles: Path = BUNDLES, facility: Path = FACILITY) -> dict[str, str]:
    """Each scenario file's text, keyed by its path under the facility tree."""
    wiring = Wiring(facility)
    return {
        f"scenarios/{bundle.name}.yaml": yaml.safe_dump(
            translate(bundle, wiring), sort_keys=False, allow_unicode=True, width=100
        )
        for bundle in sorted(path for path in bundles.iterdir() if path.is_dir())
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the demo's scenario files.")
    parser.parse_args(argv)
    for rel, text in files().items():
        path = FACILITY / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
