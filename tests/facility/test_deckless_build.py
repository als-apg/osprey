"""A facility whose physics model has no deck: the build starts it from its engine.

An engine that sets ``deckless_start_values`` is asked for start values with no
deck, and the wiring stage writes each setpoint's value into the record's
``default``; S6's limits check then refuses a start value outside its band. A
``topoff`` seed on a wired channel stops. A deckless model of any other engine
builds exactly as before: its records carry no ``default``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.facility.build import LATER_STAGES, build_facility
from osprey.facility.validate import StageReport, run_stages

MODEL = "CHAIN"

SETTINGS = {
    "injection": {
        "cycle": {"period_s": 2.0, "width_s": 1.0},
        "stages": ["transfer", "injection"],
        "current_per_charge": 3.0,
    }
}

#: The setpoints and their start values; the request starts at zero.
STARTS = {"REQ": 0.0, "SRC": 4.0, "EFF0": 0.5, "EFF1": 0.25, "STORED": 10.0}
SETPOINTS = tuple(STARTS)
READINGS = ("COUNT", "CHARGE0", "EFF0:RD", "CHARGE1", "STORED:RD", "STORED:FAST", "BUSY")

TOPOFF = {"topoff": {"ceiling": 500.0, "lifetime_h": 10.0, "period_s": 30.0}}


def _record(address: str, **block: Any) -> dict[str, Any]:
    return {"address": address, "engine": block}


def _wiring() -> list[dict[str, Any]]:
    return [
        _record("REQ", signal="request"),
        _record("SRC", signal="parameter", quantity="charge", start=4.0),
        _record("EFF0", signal="parameter", quantity="efficiency", stage=0, start=0.5),
        _record("EFF1", signal="parameter", quantity="efficiency", stage=1, start=0.25),
        _record("STORED", signal="parameter", quantity="current", start=10.0),
        _record("COUNT", signal="counter", offset_s=0.125),
        _record("CHARGE0", signal="charge", offset_s=0.25, stage=0),
        _record("EFF0:RD", signal="efficiency", offset_s=0.375, stage=0),
        _record("CHARGE1", signal="charge", offset_s=0.5, stage=1),
        _record("STORED:RD", signal="current", offset_s=0.75),
        _record("STORED:FAST", signal="current_fast"),
        _record("BUSY", signal="busy"),
    ]


def _tree() -> dict[str, Any]:
    """One deckless injection model wiring five setpoints and seven readings."""
    channels = [{"id": address, "role": "setpoint"} for address in SETPOINTS]
    channels += [{"id": address} for address in READINGS]
    return {
        "records/channels.yaml": channels,
        "models.yaml": [
            {
                "name": MODEL,
                "engine": "injection",
                "settings": copy.deepcopy(SETTINGS),
                "wiring": _wiring(),
            }
        ],
        "limits.yaml": {
            "records": [
                {"address": "SRC", "min_value": 0, "max_value": 10},
                {"address": "STORED", "min_value": 5, "max_value": 500},
            ]
        },
    }


def _write(root: Path, tree: dict[str, Any]) -> Path:
    for rel, data in tree.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return root


def _run(tmp_path: Path, tree: dict[str, Any], *, later=LATER_STAGES) -> StageReport:
    return run_stages(_write(tmp_path / "facility", tree), project_name="p", later=later)


def _lines(report: StageReport) -> list[str]:
    return [error.format_message() for error in report.errors]


def _wired(tree: dict[str, Any], address: str) -> dict[str, Any]:
    return next(r for r in tree["models.yaml"][0]["wiring"] if r["address"] == address)


def _records(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {r["address"]: r for m in document["models"] for r in m.get("wiring", [])}


# -- the deckless engine's start values ------------------------------------------------


def test_deckless_engine_model_builds_with_each_setpoint_started(tmp_path: Path) -> None:
    document = build_facility(_write(tmp_path / "facility", _tree()), project_name="p")
    records = _records(document)
    for address, start in STARTS.items():
        assert records[address]["default"] == start, address
        assert "default" in records[address]["provenance"]["defaults"], address
        assert records[address]["direction"] == "write"
        assert records[address]["role"] == "setpoint"
    for address in READINGS:
        assert "default" not in records[address], address
        assert "default" not in records[address]["provenance"]["defaults"], address
        assert records[address]["direction"] == "read"
        assert records[address]["role"] == "output"
    assert records["SRC"]["value_range"] == [0.0, 10.0]


def test_a_start_value_suppresses_seed_missing(tmp_path: Path) -> None:
    """STORED's band [5, 500] excludes 0, yet its start value 10 seeds it."""
    report = _run(tmp_path, _tree())
    assert _lines(report) == []


def test_start_value_above_its_limits_stops(tmp_path: Path) -> None:
    tree = _tree()
    _wired(tree, "SRC")["engine"]["start"] = 12.0
    assert _lines(_run(tmp_path, tree)) == [
        "facility: seed-invalid: channel SRC — nominal 12 lies above `max_value` 10; "
        "fix: move the operating point inside [min_value, max_value], or widen the limits record"
    ]


def test_start_value_below_its_limits_stops(tmp_path: Path) -> None:
    tree = _tree()
    _wired(tree, "STORED")["engine"]["start"] = 1.5
    assert _lines(_run(tmp_path, tree)) == [
        "facility: seed-invalid: channel STORED — nominal 1.5 lies below `min_value` 5; "
        "fix: move the operating point inside [min_value, max_value], or widen the limits record"
    ]


def test_start_value_absent_starts_at_zero_and_zero_outside_band_stops(tmp_path: Path) -> None:
    tree = _tree()
    del _wired(tree, "STORED")["engine"]["start"]
    assert _lines(_run(tmp_path, tree)) == [
        "facility: seed-invalid: channel STORED — nominal 0 lies below `min_value` 5; "
        "fix: move the operating point inside [min_value, max_value], or widen the limits record"
    ]


def test_engine_model_stop_is_the_build_stop(tmp_path: Path) -> None:
    tree = _tree()
    del tree["models.yaml"][0]["settings"]["injection"]["cycle"]
    lines = _lines(_run(tmp_path, tree))
    assert len(lines) == 1
    assert lines[0].startswith(f"facility: engine-invalid: model {MODEL} — ")


def test_engine_record_stop_names_the_wiring_record(tmp_path: Path) -> None:
    tree = _tree()
    _wired(tree, "SRC")["engine"]["quantity"] = "voltage"
    lines = _lines(_run(tmp_path, tree))
    assert lines and all(line.startswith("facility: engine-invalid: wiring ") for line in lines)
    assert any(f"wiring {MODEL}/SRC — " in line for line in lines)


# -- the top-off stop ------------------------------------------------------------------


def test_topoff_seed_on_a_wired_channel_stops(tmp_path: Path) -> None:
    tree = _tree()
    tree["seeds.yaml"] = {"STORED:RD": copy.deepcopy(TOPOFF)}
    assert _lines(_run(tmp_path, tree, later=())) == [
        f"facility: seed-invalid: channel STORED:RD — a `topoff` on a channel model {MODEL} "
        "wires; fix: remove `topoff`; the model computes the channel's value"
    ]


def test_topoff_seed_on_an_unwired_channel_still_builds(tmp_path: Path) -> None:
    tree = _tree()
    tree["records/channels.yaml"].append({"id": "DCCT", "unit": "mA"})
    tree["seeds.yaml"] = {"DCCT": copy.deepcopy(TOPOFF)}
    assert _lines(_run(tmp_path, tree)) == []


def test_nominal_on_a_wired_channel_keeps_its_stop(tmp_path: Path) -> None:
    tree = _tree()
    tree["seeds.yaml"] = {"SRC": {"nominal": 1.0}}
    assert _lines(_run(tmp_path, tree, later=())) == [
        f"facility: seed-invalid: channel SRC — a `nominal` on a channel model {MODEL} wires; "
        "fix: remove `nominal`; the operating point comes from the deck"
    ]


# -- wiring validation is engine-neutral -----------------------------------------------


def test_wiring_rules_do_not_load_the_lattice_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deckless non-lattice model, even with `pyat`-shaped settings, needs no pyat."""
    tree = _tree()
    tree["models.yaml"][0]["settings"]["pyat"] = {"solve": "single_pass"}
    monkeypatch.setitem(sys.modules, "osprey.simulation.engines.pyat", None)
    report = _run(tmp_path, tree, later=())
    assert _lines(report) == []


# -- a lattice-only facility builds as before -------------------------------------------

#: sha256 of the lattice-only tree's facility file, compact sorted-key JSON,
#: frozen from the build before deckless start values existed.
LATTICE_ONLY_DIGEST = "5872a35aba3d7c886c63347da2ed5da0bded3c4844c11cd2f2da3dc5e55fce32"


def _lattice_tree() -> dict[str, Any]:
    """``deck_tree`` plus a deckless ``pyat`` model wiring one more quadrupole."""
    from tests.facility._synthetic_trees import deck_tree

    tree = deck_tree()
    quad = tree["records/devices.yaml"][1]["class"]
    tree["records/devices.yaml"].append({"id": "SR/QP", "class": quad})
    tree["records/channels.yaml"] += [
        {"id": "QP:SP", "role": "setpoint", "pair": "QP:RB", "on": {"device": "SR/QP"}},
        {"id": "QP:RB", "on": {"device": "SR/QP"}},
    ]
    setting = dict(tree["models.yaml"][0]["wiring"][1]["engine"])
    tree["models.yaml"].append(
        {
            "name": "optics",
            "engine": "pyat",
            "wiring": [{"address": "QP:SP", "element": "QP", "engine": setting}],
        }
    )
    tree["limits.yaml"] = {"records": [{"address": "QP:SP", "min_value": -1, "max_value": 1}]}
    return tree


@pytest.mark.filterwarnings("ignore::Warning")
def test_lattice_only_facility_builds_byte_identical(tmp_path: Path) -> None:
    pytest.importorskip("at")
    from tests.facility._synthetic_trees import write_tree

    root = write_tree(tmp_path / "facility", _lattice_tree())
    document = build_facility(root, project_name="p")
    optics = _records(document)["QP:SP"]
    assert "default" not in optics
    assert "default" not in optics["provenance"]["defaults"]
    built = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(built).hexdigest() == LATTICE_ONLY_DIGEST
