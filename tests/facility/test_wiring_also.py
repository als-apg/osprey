"""A wiring record states the further addresses of its variable under ``also``.

Each further address is a channel no other wiring record names, and states
only its id, place, names, label and description: the build writes the
record's kind on it, and stating any other slot stops the build. The simulator view then lists it as the
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
import yaml

from osprey.apps.sdk.testing import model
from osprey.facility.views.bluesky import bluesky_document
from osprey.facility.views.simulator import write_simulator_view
from osprey.services.bluesky_bridge.devices._specs_from_file import (
    SETTLE_TOLERANCE_KEY,
    UNIT_KEY,
)
from osprey_connectors.simulation.view import SimulatorView
from osprey_facility import RECORD_FACTS, TEXTURE
from osprey_facility.errors import FacilityBuildError
from osprey_facility.records import FacilityRecords
from tests.facility._injection_fixture import FIXTURE as INJECTION
from tests.facility._injection_fixture import MODEL as INJECTION_MODEL
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
        {"id": SETPOINT_ALSO, "on": {"device": "SR/QD"}},
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


def _row(document: dict[str, Any], address: str) -> dict[str, Any]:
    return next(channel for channel in document["channels"] if channel["id"] == address)


def test_a_further_address_is_built_with_its_records_kind_as_a_default(tmp_path: Path) -> None:
    row = _row(built_document(tmp_path, _tree())[0], SETPOINT_ALSO)

    assert (row["role"], row["value_type"]) == ("setpoint", "float")
    assert {"role", "value_type"} <= set(row["provenance"]["defaults"])
    assert "pair" not in row


def test_a_further_address_of_a_bool_record_is_built_with_its_records_labels(
    tmp_path: Path,
) -> None:
    tree = {
        rel: yaml.safe_load((INJECTION / rel).read_text(encoding="utf-8"))
        for rel in ("models.yaml", "limits.yaml", "records/channels.yaml")
    }
    tree["records/channels.yaml"][0].update(value_type="bool", options=["Idle", "Fire"])
    tree["records/channels.yaml"].append({"id": "INJ:RACK:REQ"})
    model(tree, INJECTION_MODEL)["wiring"][0]["also"] = ["INJ:RACK:REQ"]

    row = _row(built_document(tmp_path, tree)[0], "INJ:RACK:REQ")

    assert (row["role"], row["value_type"], row["options"]) == (
        "setpoint",
        "bool",
        ["Idle", "Fire"],
    )
    assert {"role", "value_type", "options"} <= set(row["provenance"]["defaults"])
    assert "pair" not in row


def test_a_further_address_reads_its_records_unit_in_every_reader(tmp_path: Path) -> None:
    tree = _tree()
    _channel(tree, SETPOINT)["unit"] = "A"
    _channel(tree, SETPOINT)["tolerance"] = {"absolute": 0.5}
    inputs = view_inputs(tmp_path, tree)

    settables = {entry["name"]: entry for entry in bluesky_document(inputs.records)["settables"]}
    view = tmp_path / "render" / "data" / "simulator"
    write_simulator_view(view, inputs)
    variables = json.loads((view / "variables.json").read_text(encoding="utf-8"))
    units = {channel["address"]: channel["unit"] for channel in variables["channels"]}

    assert settables[SETPOINT_ALSO][UNIT_KEY] == settables[SETPOINT][UNIT_KEY] == "A"
    assert settables[SETPOINT_ALSO][SETTLE_TOLERANCE_KEY] == 0.5
    assert units[SETPOINT_ALSO] == units[SETPOINT] == "A"


def test_a_facility_stating_no_also_is_written_as_before(tmp_path: Path) -> None:
    """Dropping ``also`` changes the further addresses' owner and the record facts they read.

    The former further addresses state their kind, which the build wrote on them from their record.
    """
    plain = _tree()
    for address in (READING, SETPOINT):
        del _record(plain, address)["also"]
    _channel(plain, SETPOINT_ALSO)["role"] = "setpoint"

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
    for document in (stripped, without):
        for channel in document["channels"]:
            if channel["address"] in (READING_ALSO, SETPOINT_ALSO):
                for fact in RECORD_FACTS:
                    channel.pop(fact, None)
    assert stripped == without


# --- what the build refuses -----------------------------------------------------


def test_a_further_address_that_is_no_channel_stops_the_build(tmp_path: Path) -> None:
    tree = _tree()
    _record(tree, READING)["also"] = ["RACK3:UNIT9:X"]

    assert _stop(tmp_path, tree) == (
        "facility: reference-missing: wiring SR/BPM1:X — models.yaml `also` names channel "
        "RACK3:UNIT9:X, which does not exist; fix: add channel RACK3:UNIT9:X or correct `also`"
    )


def _further_stop(slot: str, further: str, wid: str, own: str, code: str) -> str:
    """The stop of a record fact stated on a further address."""
    return (
        f"facility: {code}: channel {further} — `{slot}` on {further}, which wiring record "
        f"{wid} states under `also`; fix: state `{slot}` on {own} only; it holds under each "
        "of its addresses"
    )


@pytest.mark.parametrize(
    ("slot", "value"),
    [
        ("role", "readback"),
        ("value_type", "float"),
        ("options", ["Low", "High"]),
        ("shape", [4]),
    ],
)
def test_a_kind_slot_stated_on_a_further_address_stops_the_build(
    tmp_path: Path, slot: str, value: Any
) -> None:
    """Also when it is the record's own: the build writes the kind from the record."""
    tree = _tree()
    _channel(tree, READING_ALSO)[slot] = value

    assert _stop(tmp_path, tree) == _further_stop(
        slot, READING_ALSO, "SR/BPM1:X", READING, "pair-invalid"
    )


@pytest.mark.parametrize("unit", ["mm", "um"])
def test_a_unit_on_a_further_address_stops_the_build(tmp_path: Path, unit: str) -> None:
    """Also when it is the record's own: the record states its unit once."""
    tree = _tree()
    _channel(tree, READING_ALSO)["unit"] = unit

    assert _stop(tmp_path, tree) == _further_stop(
        "unit", READING_ALSO, "SR/BPM1:X", READING, "value-invalid"
    )


def test_a_seed_on_a_further_address_stops_the_build(tmp_path: Path) -> None:
    tree = _tree()
    tree["seeds.yaml"] = {READING_ALSO: {"noise": {"absolute": 0.1}}}

    assert _stop(tmp_path, tree) == (
        "facility: seed-invalid: channel RACK3:UNIT1:X — `simulation` on RACK3:UNIT1:X, which "
        "wiring record SR/BPM1:X states under `also`; fix: state `simulation` on BPM1:X only; "
        "it holds under each of its addresses"
    )


def test_a_tolerance_on_a_further_address_stops_the_build(tmp_path: Path) -> None:
    tree = _tree()
    _channel(tree, SETPOINT)["unit"] = "A"
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

    assert _stop(tmp_path, tree) == _further_stop(
        "pair", SETPOINT_ALSO, "SR/QF:SP", "QF:SP", "pair-invalid"
    )


def test_a_pair_stated_on_a_further_address_alone_stops_the_build(tmp_path: Path) -> None:
    tree = _tree()
    _record(tree, "QF:SP")["also"] = _record(tree, SETPOINT).pop("also")
    del _channel(tree, "QF:SP")["pair"]
    _channel(tree, SETPOINT_ALSO).update(on={"device": "SR/QF"}, pair="QF:RB")

    assert _stop(tmp_path, tree) == _further_stop(
        "pair", SETPOINT_ALSO, "SR/QF:SP", "QF:SP", "pair-invalid"
    )


#: A value of each record fact a source may state, for a setpoint's further address.
_STATED: dict[str, Any] = {
    "role": "setpoint",
    "value_type": "float",
    "options": ["Low", "High"],
    "shape": [4],
    "signal": "current_setpoint",
    "pair": "QF:RB",
    "unit": "A",
    "precision": 3,
    "tolerance": {"absolute": 0.5},
    "tags": ["rack"],
    "attributes": {"slot": 2},
    "simulation": {"nominal": 1.0},
    "timeout_s": 2.0,
    "idle": 0,
    "started_by": "QF:SP",
    "stopped_by": "QF:SP",
}


def test_the_stated_values_cover_every_record_fact_a_source_states() -> None:
    """``command`` is computed by the build and never stated."""
    assert set(_STATED) == set(RECORD_FACTS) - {"command"}


@pytest.mark.parametrize("slot", sorted(_STATED))
def test_every_record_fact_on_a_further_address_stops_the_build(tmp_path: Path, slot: str) -> None:
    tree = _tree()
    _channel(tree, SETPOINT_ALSO)[slot] = _STATED[slot]

    assert _stop(tmp_path, tree) == _further_stop(
        slot, SETPOINT_ALSO, "SR/QD:SP", SETPOINT, RECORD_FACTS[slot]
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


def test_a_further_address_of_an_unserved_model_keeps_its_own_scenario_keys(
    tmp_path: Path,
) -> None:
    """A model the render does not serve leaves its further addresses to the texture model."""
    tree = _tree()
    tree["scenarios/shifted.yaml"] = {
        "name": "shifted",
        "noise": {READING_ALSO: {"absolute": 0.2}},
    }
    view = tmp_path / "render" / "data" / "simulator"
    write_simulator_view(view, view_inputs(tmp_path, tree, served=[TEXTURE]))
    listed = json.loads((view / "scenarios.json").read_text(encoding="utf-8"))["scenarios"]

    (shifted,) = [scenario for scenario in listed if scenario["name"] == "shifted"]
    assert shifted["noise"] == {READING_ALSO: {"absolute": 0.2}}


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
