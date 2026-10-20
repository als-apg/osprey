"""The simulator view's wiring entries for one model.

``simulator_wiring`` projects the wiring records of one model of the in-memory
facility file onto the simulator's wiring entry shape: the authored keys
(``id``, ``address``, ``element`` or ``slices``, ``engine``, ``calibration``)
plus the slots the build fills from the channel and its limits record
(``direction``, ``unit``, ``default``, ``value_range``). A key the record does
not carry is absent from its entry; ``provenance`` never appears.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from osprey.facility.build import FacilityDocument, build_facility
from osprey.facility.views.simulator import simulator_wiring

DEMO_FACILITY = Path(__file__).resolve().parents[2] / "src/osprey/templates/facilities/example"

ENTRY_KEYS = {
    "id",
    "address",
    "element",
    "slices",
    "engine",
    "calibration",
    "direction",
    "unit",
    "default",
    "value_range",
    "role",
    "plane",
    "refresh",
    "command",
    "value_type",
    "shape",
}
ALWAYS = {"id", "address", "engine", "direction", "default", "role", "plane", "refresh"}
UNWIRED_ELEMENT = {"SR:DIAG:CHROM:X", "SR:DIAG:CHROM:Y", "SR:DIAG:TUNE:X", "SR:DIAG:TUNE:Y"}

#: The element-free records of an injected shot: each monitor's arm command
#: and armed flag, and the injection request.
SHOT_FLAGS = {
    f"SR:DIAG:BPM:{n:02d}:TBT:{tail}" for n in range(1, 73) for tail in ("ARM", "ARMED")
} | {"SR:DIAG:INJECTION:REQUEST"}

#: The cleaning request: an element-free command that empties the stored charge.
CLEANING = "SR:DIAG:CLEANING:REQUEST"

#: The model-wide settings with no element and no calibration: turns tracked.
TURNS_TRACKED = {"SR:DIAG:TBT:TURNS:SP", "SR:DIAG:TBT:TURNS:RB"}

#: The RF-on setting and its readback: on the cavity, as a two-label switch
#: with no calibration.
RF_ON = {"SR:RF:ON:SP", "SR:RF:ON:RB"}

#: The stored-current reading: no element, published through a calibration.
STORED_CURRENT = "SR:DIAG:STORED_BEAM:CURRENT:RB"

#: Each monitor's attenuation setting and readback: on its element, through
#: the decibel-to-gain table.
ATTENUATION = {
    f"SR:DIAG:BPM:{n:02d}:ATTENUATION:{tail}" for n in range(1, 73) for tail in ("SP", "RB")
}

#: Each monitor's labelled readout settings and readbacks: on its element, a
#: discrete engine block, and a table from each label's index to the readout
#: attribute's value.
LABELLED_READOUT = {
    f"SR:DIAG:BPM:{n:02d}:{name}:{tail}": attribute
    for n in range(1, 73)
    for name, attribute in (
        ("MODE", "readout_noise_scale"),
        ("PILOT_TONE", "readout_tone_amplitude"),
    )
    for tail in ("SP", "RB")
}


@pytest.fixture(scope="module")
def demo() -> FacilityDocument:
    return build_facility(DEMO_FACILITY, project_name="demo")


@pytest.fixture(scope="module")
def sr(demo: FacilityDocument) -> list[dict[str, Any]]:
    return simulator_wiring(demo, "SR")


def _records(doc: FacilityDocument, model: str) -> list[dict[str, Any]]:
    (found,) = [m for m in doc["models"] if m["name"] == model]
    return list(found.get("wiring", []))


def test_every_sr_record_becomes_one_entry_in_order(
    demo: FacilityDocument, sr: list[dict[str, Any]]
) -> None:
    records = _records(demo, "SR")
    assert len(sr) == len(records) == 1645
    assert [entry["id"] for entry in sr] == [record["id"] for record in records]


def test_every_entry_has_the_wiring_entry_shape(sr: list[dict[str, Any]]) -> None:
    for entry in sr:
        assert set(entry) <= ENTRY_KEYS, entry["id"]
        assert set(entry) >= ALWAYS, entry["id"]
        assert entry["direction"] in {"read", "write"}
    commands = {entry["address"] for entry in sr if entry.get("command")}
    armed = {address for address in SHOT_FLAGS if address.endswith(":ARMED")}
    assert commands == (SHOT_FLAGS - armed) | {CLEANING}


def test_calibration_and_element_appear_exactly_where_the_record_carries_them(
    sr: list[dict[str, Any]],
) -> None:
    bare = {entry["address"] for entry in sr if "calibration" not in entry}
    assert bare == UNWIRED_ELEMENT | SHOT_FLAGS | TURNS_TRACKED | RF_ON | {CLEANING}
    unplaced = {entry["address"] for entry in sr if "element" not in entry}
    assert unplaced == UNWIRED_ELEMENT | SHOT_FLAGS | TURNS_TRACKED | {STORED_CURRENT, CLEANING}
    assert sum("calibration" in entry for entry in sr) == 1491


def test_unit_appears_exactly_where_the_channel_states_one(
    demo: FacilityDocument, sr: list[dict[str, Any]]
) -> None:
    units = {c["id"]: c["unit"] for c in demo["channels"] if "unit" in c}
    for entry in sr:
        if entry["address"] in units:
            assert entry["unit"] == units[entry["address"]]
        else:
            assert "unit" not in entry
    assert sum("unit" in entry for entry in sr) == 1131


def test_value_range_appears_exactly_where_the_limits_record_bounds_the_channel(
    demo: FacilityDocument, sr: list[dict[str, Any]]
) -> None:
    bounded = {
        r["address"]: [float(r["min_value"]), float(r["max_value"])]
        for r in demo["limits"]["records"]
        if "min_value" in r and "max_value" in r
    }
    ranged = {entry["address"]: entry["value_range"] for entry in sr if "value_range" in entry}
    assert ranged == {address: span for address, span in bounded.items() if address in ranged}
    assert len(ranged) == 2
    assert all(entry["address"] not in bounded for entry in sr if "value_range" not in entry)


def test_the_entries_serialise_and_leave_the_facility_file_untouched(
    demo: FacilityDocument, sr: list[dict[str, Any]]
) -> None:
    json.dumps(sr)
    sr[0]["engine"]["probe"] = 1
    assert all("probe" not in r["engine"] for r in _records(demo, "SR"))
    del sr[0]["engine"]["probe"]
    assert all("provenance" in r for r in _records(demo, "SR"))


def test_a_model_without_wiring_has_no_entries(demo: FacilityDocument) -> None:
    assert simulator_wiring(demo, "texture") == []


def test_an_unknown_model_is_refused(demo: FacilityDocument) -> None:
    with pytest.raises(KeyError, match="NOPE"):
        simulator_wiring(demo, "NOPE")


def test_a_waveform_channel_s_entry_carries_its_value_type_and_shape(
    demo: FacilityDocument,
) -> None:
    doc = json.loads(json.dumps(demo))
    (channel,) = [c for c in doc["channels"] if c["id"] == "SR:DIAG:TUNE:X"]
    channel.update(value_type="waveform", shape=[3])
    entries = {entry["address"]: entry for entry in simulator_wiring(doc, "SR")}
    assert (entries["SR:DIAG:TUNE:X"]["value_type"], entries["SR:DIAG:TUNE:X"]["shape"]) == (
        "waveform",
        [3],
    )
    assert "value_type" not in entries["SR:DIAG:TUNE:Y"]


def test_the_trajectory_settings_and_readings_are_wired_onto_their_engine_attributes(
    sr: list[dict[str, Any]],
) -> None:
    entries = {entry["address"]: entry for entry in sr}
    for address in TURNS_TRACKED:
        assert entries[address]["engine"] == {"attribute": "turns_tracked"}
        assert entries[address]["default"] == 100
    assert (entries["SR:DIAG:TBT:TURNS:SP"]["role"], entries["SR:DIAG:TBT:TURNS:RB"]["role"]) == (
        "setpoint",
        "readback",
    )
    for address in RF_ON:
        assert (entries[address]["element"], entries[address]["engine"]) == (
            "CAV",
            {"attribute": "Voltage"},
        )
        assert entries[address]["default"] == 1
    stored = entries[STORED_CURRENT]
    assert stored["engine"]["attribute"] == "stored_current"
    assert 0.0 <= stored["engine"]["surviving_fraction"] <= 1.0
    assert stored["engine"]["charge_floor"] >= 0.0
    assert (stored["direction"], stored["role"], stored["unit"]) == ("read", "output", "mA")
    for address in ATTENUATION:
        entry = entries[address]
        monitor = "BPM" + address.split(":")[3]
        assert (entry["element"], entry["engine"]) == (monitor, {"attribute": "readout_gain"})
        assert (entry["unit"], entry["default"]) == ("dB", 0.0)
        table = entry["calibration"]["curve"]["table"]
        assert table["grid"][0] == 0.0 and table["values"][0] == 1.0
        assert all(a > b for a, b in zip(table["values"], table["values"][1:], strict=False))


def test_the_labelled_readout_settings_and_the_cleaning_request_are_wired(
    sr: list[dict[str, Any]],
) -> None:
    entries = {entry["address"]: entry for entry in sr}
    for address, attribute in LABELLED_READOUT.items():
        entry = entries[address]
        monitor = "BPM" + address.split(":")[3]
        assert (entry["element"], entry["engine"]) == (
            monitor,
            {"attribute": attribute, "discrete": True},
        )
        assert entry["default"] == 0
        assert entry["direction"] == ("write" if address.endswith(":SP") else "read")
        assert "unit" not in entry
        table = entry["calibration"]["curve"]["table"]
        assert table["grid"] == [0.0, 1.0]
        assert all(value >= 0.0 for value in table["values"])
    for monitor in range(1, 73):
        mode = entries[f"SR:DIAG:BPM:{monitor:02d}:MODE:SP"]["calibration"]["curve"]["table"]
        tone = entries[f"SR:DIAG:BPM:{monitor:02d}:PILOT_TONE:SP"]["calibration"]["curve"]["table"]
        assert all(value > 0.0 for value in mode["values"])
        assert tone["values"][0] == 0.0 < tone["values"][1]
    cleaning = entries[CLEANING]
    assert cleaning["engine"] == {"attribute": "clear_stored_charge"}
    assert (cleaning["direction"], cleaning["default"], cleaning["command"]) == ("write", 0, True)


# -- the trigger sample the engine is built with -----------------------------------


def _model_record(**measurement: Any) -> dict[str, Any]:
    return {
        "name": "SR",
        "engine": "pyat",
        "settings": {"pyat": {"solve": "periodic", "injection": {"marker": "INJ"}}},
        "measurement": measurement,
    }


def test_the_declared_trigger_sample_is_written_into_the_engines_injection_block() -> None:
    from osprey.facility.views.simulator import TRIGGER_SAMPLE_KEY, engine_settings

    record = _model_record(trajectory={"trigger_sample": 7, "steps": []})

    settings = engine_settings(record)

    assert settings["pyat"]["injection"] == {"marker": "INJ", TRIGGER_SAMPLE_KEY: 7}
    assert TRIGGER_SAMPLE_KEY not in record["settings"]["pyat"]["injection"]


@pytest.mark.parametrize(
    "record",
    [
        _model_record(),
        {
            "name": "LINE",
            "engine": "pyat",
            "settings": {"pyat": {"solve": "single_pass"}},
            "measurement": {"trajectory": {"trigger_sample": 7, "steps": []}},
        },
    ],
    ids=["no-declaration", "no-injection-block"],
)
def test_a_model_without_a_declaration_or_an_injection_block_keeps_its_settings(
    record: dict[str, Any],
) -> None:
    from osprey.facility.views.simulator import engine_settings

    assert engine_settings(record) == record["settings"]


def test_the_demo_engine_reads_the_measurement_files_trigger_sample(
    demo: FacilityDocument,
) -> None:
    from osprey.facility.views.simulator import TRIGGER_SAMPLE_KEY, engine_settings

    (record,) = [m for m in demo["models"] if m["name"] == "SR"]
    declared = record["measurement"]["trajectory"][TRIGGER_SAMPLE_KEY]

    assert TRIGGER_SAMPLE_KEY not in record["settings"]["pyat"]["injection"]
    assert engine_settings(record)["pyat"]["injection"][TRIGGER_SAMPLE_KEY] == declared


def test_a_trigger_sample_authored_with_the_model_stops_the_build(tmp_path: Path) -> None:
    pytest.importorskip("at")
    from osprey.facility.errors import FacilityBuildError
    from tests.facility._pyaml_trees import measured_tree, with_correctors
    from tests.facility._synthetic_trees import model, with_trajectory, write_tree

    tree = with_trajectory(with_correctors(measured_tree()), "SR", "SR/BPM1", marker="M1")
    model(tree, "SR")["settings"]["pyat"]["injection"]["trigger_sample"] = 4
    write_tree(tmp_path, tree)

    with pytest.raises(FacilityBuildError) as stopped:
        build_facility(tmp_path, project_name="authored")

    message = str(stopped.value)
    assert "pyat.injection.trigger_sample" in message
    assert "measurement file" in message
