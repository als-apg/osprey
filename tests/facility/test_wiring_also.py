"""A wiring record states the further addresses of its variable under ``also``.

The build holds each further address to the record's own: it is a channel, of
the same role, value type, options, shape and unit, with no seed of its own,
and no other wiring record names it. The simulator view then lists it as the
model's channel and carries ``also`` on the record's wiring entry. A scenario
block keyed by a further address is written under the record's address, and a
facility that states no ``also`` is built and written as it was before.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from osprey.apps.sdk.testing import model
from osprey.facility.views.simulator import write_simulator_view
from osprey_connectors.simulation.view import SimulatorView
from osprey_facility.errors import FacilityBuildError
from osprey_facility.records import FacilityRecords
from tests.facility._pyaml_trees import built_document, measured_tree, view_inputs

#: A reading and a setpoint the measured tree wires, and a second address for each.
READING, READING_ALSO = "BPM1:X", "RACK3:UNIT1:X"
SETPOINT, SETPOINT_ALSO = "QD:SP", "RACK3:UNIT2:SP"


def _record(tree: dict[str, Any], address: str) -> dict[str, Any]:
    """The wiring record of model SR whose ``address`` is ``address``."""
    return next(r for r in model(tree, "SR")["wiring"] if r["address"] == address)


def _channel(tree: dict[str, Any], address: str) -> dict[str, Any]:
    return next(c for c in tree["records/channels.yaml"] if c["id"] == address)


def _tree() -> dict[str, Any]:
    """The measured tree, its BPM reading and its QD setpoint each under a second address."""
    tree = measured_tree()
    tree["records/channels.yaml"] += [
        {"id": READING_ALSO, "on": {"device": "SR/BPM1"}},
        {"id": SETPOINT_ALSO, "role": "setpoint", "on": {"device": "SR/QD"}},
    ]
    _record(tree, READING)["also"] = [READING_ALSO]
    _record(tree, SETPOINT)["also"] = [SETPOINT_ALSO]
    return tree


def _variables(root: Path, tree: dict[str, Any]) -> dict[str, Any]:
    """``variables.json`` of the tree's simulator view."""
    view = root / "render" / "data" / "simulator"
    write_simulator_view(view, view_inputs(root, tree))
    return json.loads((view / "variables.json").read_text(encoding="utf-8"))


def _scenarios(root: Path, tree: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The scenarios of the tree's simulator view, by name."""
    view = root / "render" / "data" / "simulator"
    write_simulator_view(view, view_inputs(root, tree))
    listed = json.loads((view / "scenarios.json").read_text(encoding="utf-8"))["scenarios"]
    return {scenario["name"]: scenario for scenario in listed}


def _stop(tmp_path: Path, tree: dict[str, Any]) -> str:
    with pytest.raises(FacilityBuildError) as stopped:
        built_document(tmp_path, tree)
    return stopped.value.format_message()


# --- what the build and the view state ------------------------------------------


def test_the_view_lists_a_further_address_as_the_models_channel(tmp_path: Path) -> None:
    variables = _variables(tmp_path, _tree())

    owners = {channel["address"]: channel["owner"] for channel in variables["channels"]}
    assert owners[READING_ALSO] == owners[READING] == "SR"
    assert owners[SETPOINT_ALSO] == owners[SETPOINT] == "SR"
    (sr,) = [entry for entry in variables["models"] if entry["name"] == "SR"]
    also = {entry["address"]: entry.get("also") for entry in sr["wiring"]}
    assert also[READING] == [READING_ALSO]
    assert also[SETPOINT] == [SETPOINT_ALSO]
    assert also["QF:SP"] is None


def test_the_opened_view_binds_the_record_once_under_every_address(tmp_path: Path) -> None:
    view = tmp_path / "render" / "data" / "simulator"
    write_simulator_view(view, view_inputs(tmp_path, _tree()))
    opened = SimulatorView.open(view)

    bound = {binding.address: binding for binding in opened.bindings(served_only=False)}
    assert READING_ALSO not in bound
    assert bound[READING].also == (READING_ALSO,)
    assert bound[READING].addresses == (READING, READING_ALSO)
    assert bound["QF:SP"].addresses == ("QF:SP",)
    assert opened.record_address(READING_ALSO) == READING
    assert opened.record_address(READING) == READING


def test_the_records_name_the_wiring_record_under_each_address(tmp_path: Path) -> None:
    records = FacilityRecords(built_document(tmp_path, _tree())[0])

    assert records.wired(READING_ALSO) is records.wired(READING)
    assert records.wired(READING)["address"] == READING


def test_the_records_give_each_address_its_records_address(tmp_path: Path) -> None:
    tree = _tree()
    tree["records/channels.yaml"].append({"id": "RACK3:SPARE", "on": {"device": "SR/BPM1"}})
    records = FacilityRecords(built_document(tmp_path, tree)[0])

    assert records.record_address(SETPOINT_ALSO) == SETPOINT
    assert records.record_address(SETPOINT) == SETPOINT
    assert records.wired("RACK3:SPARE") is None
    assert records.record_address("RACK3:SPARE") == "RACK3:SPARE"


def test_a_facility_stating_no_also_is_written_as_before(tmp_path: Path) -> None:
    """Dropping ``also`` changes the further addresses' owner and nothing else."""
    plain = _tree()
    for address in (READING, SETPOINT):
        del _record(plain, address)["also"]

    with_also = _variables(tmp_path / "also", _tree())
    without = _variables(tmp_path / "plain", plain)

    (sr,) = [entry for entry in without["models"] if entry["name"] == "SR"]
    assert all("also" not in entry for entry in sr["wiring"])
    stripped = copy.deepcopy(with_also)
    for entry in stripped["models"]:
        for record in entry["wiring"]:
            record.pop("also", None)
    for channel in stripped["channels"]:
        if channel["address"] in (READING_ALSO, SETPOINT_ALSO):
            channel["owner"] = "texture"
    assert stripped == without


# --- what the build refuses -----------------------------------------------------


def test_a_further_address_that_is_no_channel_stops_the_build(tmp_path: Path) -> None:
    tree = _tree()
    _record(tree, READING)["also"] = ["RACK3:UNIT9:X"]

    assert _stop(tmp_path, tree) == (
        "facility: reference-missing: wiring SR/BPM1:X — models.yaml `also` names channel "
        "RACK3:UNIT9:X, which does not exist; fix: add channel RACK3:UNIT9:X or correct `also`"
    )


@pytest.mark.parametrize(
    ("change", "slots"),
    [
        ({"role": "setpoint"}, "`role`"),
        ({"value_type": "int"}, "`value_type`"),
        ({"value_type": "waveform", "shape": [4]}, "`value_type`, `shape`"),
        ({"value_type": "enum", "options": ["a", "b"]}, "`value_type`, `options`"),
    ],
)
def test_a_further_address_of_another_shape_stops_the_build(
    tmp_path: Path, change: dict[str, Any], slots: str
) -> None:
    tree = _tree()
    _channel(tree, READING_ALSO).update(change)

    assert _stop(tmp_path, tree) == (
        f"facility: pair-invalid: wiring SR/BPM1:X — `also` names RACK3:UNIT1:X, which differs "
        f"from BPM1:X in {slots}; fix: give BPM1:X and RACK3:UNIT1:X the same {slots}, or "
        "remove RACK3:UNIT1:X from `also`"
    )


def test_a_further_address_in_another_unit_stops_the_build(tmp_path: Path) -> None:
    tree = _tree()
    _channel(tree, READING)["unit"] = "mm"
    _channel(tree, READING_ALSO)["unit"] = "um"

    assert _stop(tmp_path, tree) == (
        "facility: pair-invalid: wiring SR/BPM1:X — `also` names RACK3:UNIT1:X, which differs "
        "from BPM1:X in `unit`; fix: give BPM1:X and RACK3:UNIT1:X the same `unit`, or remove "
        "RACK3:UNIT1:X from `also`"
    )


def test_a_further_address_stating_no_unit_takes_no_part_in_the_unit_check(
    tmp_path: Path,
) -> None:
    tree = _tree()
    _channel(tree, READING)["unit"] = "mm"

    built_document(tmp_path, tree)


def test_a_seed_on_a_further_address_stops_the_build(tmp_path: Path) -> None:
    tree = _tree()
    tree["seeds.yaml"] = {READING_ALSO: {"noise": {"absolute": 0.1}}}

    assert _stop(tmp_path, tree) == (
        "facility: seed-invalid: wiring SR/BPM1:X — `also` names RACK3:UNIT1:X, which carries "
        "a seed; fix: remove the seed of RACK3:UNIT1:X; it serves the value of BPM1:X, motion "
        "included"
    )


@pytest.mark.parametrize(
    "unit", [pytest.param("A", id="with-unit"), pytest.param(None, id="no-unit")]
)
def test_a_tolerance_on_a_further_address_stops_the_build(tmp_path: Path, unit: str | None) -> None:
    tree = _tree()
    if unit is not None:
        _channel(tree, SETPOINT)["unit"] = unit
        _channel(tree, SETPOINT_ALSO)["unit"] = unit
    _channel(tree, SETPOINT_ALSO)["tolerance"] = {"absolute": 0.5}

    assert _stop(tmp_path, tree) == (
        "facility: value-invalid: channel RACK3:UNIT2:SP — `tolerance` on RACK3:UNIT2:SP, "
        "which wiring record SR/QD:SP states under `also`; fix: state `tolerance` on QD:SP "
        "only; it holds under each of its addresses"
    )


def test_a_further_address_naming_the_records_own_pair_stops_the_build(tmp_path: Path) -> None:
    """Two setpoints may share a readback; two addresses of one setpoint are not two."""
    tree = _tree()
    _record(tree, "QF:SP")["also"] = _record(tree, SETPOINT).pop("also")
    _channel(tree, SETPOINT_ALSO).update(on={"device": "SR/QF"}, pair="QF:RB")

    assert _stop(tmp_path, tree) == (
        "facility: pair-invalid: wiring SR/QF:SP — `also` names RACK3:UNIT2:SP, whose `pair` "
        "is QF:RB as QF:SP's is, so QF:RB would count as shared by two setpoints and follow "
        "neither; fix: remove `pair` from RACK3:UNIT2:SP; QF:RB follows the record under "
        "each of its addresses"
    )


def test_a_pair_stated_on_a_further_address_alone_stops_the_build(tmp_path: Path) -> None:
    tree = _tree()
    _record(tree, "QF:SP")["also"] = _record(tree, SETPOINT).pop("also")
    del _channel(tree, "QF:SP")["pair"]
    _channel(tree, SETPOINT_ALSO).update(on={"device": "SR/QF"}, pair="QF:RB")

    assert _stop(tmp_path, tree) == (
        "facility: pair-invalid: wiring SR/QF:SP — `also` names RACK3:UNIT2:SP, whose `pair` "
        "is QF:RB, while QF:SP states none; fix: state `pair: QF:RB` on QF:SP and remove "
        "`pair` from RACK3:UNIT2:SP"
    )


def test_a_pair_naming_a_readbacks_further_address_stops_the_build(tmp_path: Path) -> None:
    tree = _tree()
    tree["records/channels.yaml"].append({"id": "RACK3:UNIT2:RB", "on": {"device": "SR/QF"}})
    _record(tree, "QF:RB")["also"] = ["RACK3:UNIT2:RB"]
    _channel(tree, "QF:SP")["pair"] = "RACK3:UNIT2:RB"

    assert _stop(tmp_path, tree) == (
        "facility: pair-invalid: channel QF:SP — `pair` names RACK3:UNIT2:RB, which wiring "
        "record SR/QF:RB states under `also`; fix: name QF:RB as the pair"
    )


def test_an_address_two_records_name_stops_the_build(tmp_path: Path) -> None:
    tree = _tree()
    tree["records/channels.yaml"].append({"id": "RACK3:UNIT3:X", "on": {"device": "SR/BPM1"}})
    _record(tree, READING)["also"] = [READING_ALSO, "RACK3:UNIT3:X"]
    _record(tree, "BPM1:Y")["also"] = ["RACK3:UNIT3:X"]

    assert _stop(tmp_path, tree) == (
        "facility: wiring-conflict: channel RACK3:UNIT3:X — wiring records SR/BPM1:X, "
        "SR/BPM1:Y each name the address; fix: name the address in one wiring record, once, "
        "as its `address` or under `also`"
    )


def test_a_further_address_another_record_wires_stops_the_build(tmp_path: Path) -> None:
    tree = _tree()
    _record(tree, READING)["also"] = ["BPM1:Y"]

    assert _stop(tmp_path, tree) == (
        "facility: wiring-conflict: channel BPM1:Y — wiring records SR/BPM1:X, SR/BPM1:Y each "
        "name the address; fix: name the address in one wiring record, once, as its `address` "
        "or under `also`"
    )


def test_an_address_one_record_names_twice_stops_the_build(tmp_path: Path) -> None:
    tree = _tree()
    _record(tree, READING)["also"] = [READING_ALSO, READING_ALSO]

    assert _stop(tmp_path, tree) == (
        "facility: wiring-conflict: channel RACK3:UNIT1:X — wiring record SR/BPM1:X names the "
        "address more than once; fix: name the address in one wiring record, once, as its "
        "`address` or under `also`"
    )


def test_a_second_record_on_one_monitor_reading_stops_the_build(tmp_path: Path) -> None:
    """Two records on one reading were served wrongly before; the build now names ``also``."""
    tree = _tree()
    del _record(tree, READING)["also"]
    model(tree, "SR")["wiring"].append(
        {"address": READING_ALSO, "element": "BPM1", "engine": {"axis": "x"}}
    )

    assert _stop(tmp_path, tree) == (
        "facility: engine-invalid: model SR — BPM1:X and RACK3:UNIT1:X both read x of monitor "
        "BPM1, which has one reading per axis; state RACK3:UNIT1:X under `also` of BPM1:X's "
        "record; fix: correct the model's settings or wiring so its pyat engine can declare "
        "its faults"
    )


# --- scenarios ------------------------------------------------------------------


def test_a_scenario_block_keyed_by_a_further_address_is_written_under_the_records(
    tmp_path: Path,
) -> None:
    tree = _tree()
    tree["limits.yaml"] = {
        "records": [
            {"address": address, "min_value": -200.0, "max_value": 200.0}
            for address in (SETPOINT, SETPOINT_ALSO)
        ]
    }
    tree["scenarios/shifted.yaml"] = {
        "name": "shifted",
        "overrides": {SETPOINT_ALSO: -90.0},
        "noise": {READING_ALSO: {"absolute": 0.2}},
        "channel_faults": {READING_ALSO: "frozen"},
    }
    tree["scenarios/quiet.yaml"] = {"name": "quiet", "still": [READING_ALSO]}

    scenarios = _scenarios(tmp_path, tree)

    assert scenarios["shifted"]["overrides"] == {SETPOINT: -90.0}
    assert scenarios["shifted"]["noise"] == {READING: {"absolute": 0.2}}
    assert scenarios["shifted"]["channel_faults"] == {READING: "frozen"}
    assert scenarios["quiet"]["still"] == [READING]


def test_a_fault_variable_is_keyed_by_the_records_address(tmp_path: Path) -> None:
    tree = _tree()
    tree["scenarios/offset.yaml"] = {
        "name": "offset",
        "faults": {"SR": {f"{READING}/offset": 0.1}},
    }

    scenarios = _scenarios(tmp_path, tree)

    assert scenarios["offset"]["faults"]["SR"]["writes"] == {f"{READING}/offset": 0.1}


def test_a_fault_variable_keyed_by_a_further_address_stops_the_build(tmp_path: Path) -> None:
    tree = _tree()
    tree["scenarios/offset.yaml"] = {
        "name": "offset",
        "faults": {"SR": {f"{READING_ALSO}/offset": 0.1}},
    }

    message = _stop(tmp_path, tree)

    assert "engine-invalid: scenario offset" in message
    assert f"closest: {READING}/offset" in message
