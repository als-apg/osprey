"""The demo's ``models.yaml``, derived from the virtual accelerator's bindings.

The demo has one deck model, the deck machine's. Its wiring holds one record
per address the virtual accelerator's bindings wire, re-expressed in the
record's own words:

* ``element`` -- the binding's element; every binding has one slice of weight
  1 on that element, so no record states ``slices``;
* ``engine`` -- pyAT's words: a magnet or cavity names the element
  ``attribute`` and, for an array attribute, its ``index``; a monitor names
  the ``axis`` of the closed orbit it reads;
* ``calibration`` -- the binding's linear hardware -> physics ``curve`` and
  its ``energy_scaling``; a monitor's stated inverse is the curve's algebraic
  inverse, so no record states ``inverse``.

A readback states the same element, engine words and calibration as the
setpoint it reads back. Beyond the bindings, the model wires the deck
machine's tune and chromaticity readbacks to the optics solve on their axis,
and one RF cavity's frequency pair to the deck's cavity ``Frequency`` in MHz.

No record states a slot the build computes (``default``, ``direction``,
``unit``, ``value_range``). Records are sorted by address.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_records() -> ModuleType:
    """``_records.py`` beside this file, under the name the generator loads it by."""
    name = "facility_demo__records"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("_records.py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_records = _load_records()

#: The deck the model reads, relative to ``data/facility/``.
DECK = f"decks/{_records.DECK_MACHINE}.json"

#: The binding kind of a beam position monitor.
MONITOR_KIND = "monitor"

#: Each optics readback and the solve output it reads.
OPTICS: dict[str, dict[str, str]] = {
    "SR:DIAG:CHROM:X": {"attribute": "chromaticity", "axis": "x"},
    "SR:DIAG:CHROM:Y": {"attribute": "chromaticity", "axis": "y"},
    "SR:DIAG:TUNE:X": {"attribute": "tune", "axis": "x"},
    "SR:DIAG:TUNE:Y": {"attribute": "tune", "axis": "y"},
}

#: The one RF cavity frequency pair the model drives; its channels are in MHz.
CAVITY_PAIR: tuple[str, ...] = (
    "SR:RF:CAVITY:01:FREQUENCY:RB",
    "SR:RF:CAVITY:01:FREQUENCY:SP",
)

#: The deck cavity's frequency attribute, in Hz.
CAVITY_ATTRIBUTE = "Frequency"

#: MHz -> Hz.
MHZ = 1e6


def _engine(binding: dict[str, Any]) -> dict[str, Any]:
    if binding["kind"] == MONITOR_KIND:
        return {"axis": binding["attribute"]}
    engine: dict[str, Any] = {"attribute": binding["attribute"]}
    if binding["index"] is not None:
        engine["index"] = binding["index"]
    return engine


def _calibration(binding: dict[str, Any]) -> dict[str, Any]:
    curve = binding["calibration"]
    where = binding["setpoint_address"]
    if curve["kind"] != "linear":
        raise _records.RecordsError(f"{where}: calibration kind {curve['kind']} has no inverse")
    inverse = binding["monitor_inverse"]
    if inverse is not None and not (
        inverse["kind"] == "linear"
        and math.isclose(inverse["gain"] * curve["gain"], 1.0, rel_tol=1e-12)
        and math.isclose(
            inverse["offset"], -curve["offset"] / curve["gain"], rel_tol=1e-12, abs_tol=1e-12
        )
    ):
        raise _records.RecordsError(f"{where}: the monitor inverse is not the curve's inverse")
    return {
        "curve": {"linear": {"gain": curve["gain"], "offset": curve["offset"]}},
        "energy_scaling": binding["energy_scaling"],
    }


def _bound_wiring() -> list[dict[str, Any]]:
    records = []
    for binding in _records._json(_records.VA_BINDINGS)["bindings"]:
        slices = binding["slices"]
        if [(s["element"], s["weight"]) for s in slices] != [(binding["element"], 1.0)]:
            raise _records.RecordsError(
                f"{binding['setpoint_address']}: expected one slice of weight 1 on its element"
            )
        body = {
            "element": binding["element"],
            "engine": _engine(binding),
            "calibration": _calibration(binding),
        }
        for key in ("setpoint_address", "readback_address"):
            if binding[key]:
                records.append({"address": binding[key], **body})
    return records


def _cavity_element() -> str:
    """The deck's one RF cavity element."""
    found = [
        str(element["FamName"])
        for element in _records._json(_records.SR_DECK)["elements"]
        if element.get("Class") == _records.RF_CAVITY_CLASS
    ]
    if len(found) != 1:
        raise _records.RecordsError(f"{_records.SR_DECK}: expected one RF cavity, found {found}")
    return found[0]


def _extra_wiring() -> list[dict[str, Any]]:
    additions = {row["address"] for row in _records._json(_records.ADDITIONS)["rows"]}
    if additions != set(OPTICS):
        raise _records.RecordsError(
            f"{_records.ADDITIONS}: expected the optics readbacks {sorted(OPTICS)}"
        )
    records: list[dict[str, Any]] = [
        {"address": address, "engine": dict(engine)} for address, engine in OPTICS.items()
    ]
    element = _cavity_element()
    for address in CAVITY_PAIR:
        records.append(
            {
                "address": address,
                "element": element,
                "engine": {"attribute": CAVITY_ATTRIBUTE},
                "calibration": {
                    "curve": {"linear": {"gain": MHZ, "offset": 0.0}},
                    "energy_scaling": "none",
                },
            }
        )
    return records


def build_models() -> list[dict[str, Any]]:
    """``models.yaml``: the deck machine's model and its wiring, sorted by address.

    Returns:
        The one-model list ``models.yaml`` stores.

    Raises:
        RecordsError: A binding is not one weight-1 slice on its element,
            carries a non-linear curve or a monitor inverse that is not the
            curve's inverse, an address is wired twice, or the deck does not
            hold exactly one RF cavity.
    """
    wiring = sorted(_bound_wiring() + _extra_wiring(), key=lambda record: record["address"])
    addresses = [record["address"] for record in wiring]
    if len(set(addresses)) != len(addresses):
        raise _records.RecordsError("an address is wired twice")
    return [
        {
            "name": _records.DECK_MACHINE,
            "engine": "pyat",
            "deck": DECK,
            "settings": {"pyat": {"solve": "periodic"}},
            "wiring": wiring,
        }
    ]
