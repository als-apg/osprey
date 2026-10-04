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

DEMO_FACILITY = (
    Path(__file__).resolve().parents[2]
    / "src/osprey/templates/apps/control_assistant/data/facility"
)

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
}
ALWAYS = {"id", "address", "engine", "direction", "default"}
UNWIRED_ELEMENT = {"SR:DIAG:CHROM:X", "SR:DIAG:CHROM:Y", "SR:DIAG:TUNE:X", "SR:DIAG:TUNE:Y"}


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
    assert len(sr) == len(records) == 846
    assert [entry["id"] for entry in sr] == [record["id"] for record in records]


def test_every_entry_has_the_wiring_entry_shape(sr: list[dict[str, Any]]) -> None:
    for entry in sr:
        assert set(entry) <= ENTRY_KEYS, entry["id"]
        assert set(entry) >= ALWAYS, entry["id"]
        assert entry["direction"] in {"read", "write"}


def test_calibration_and_element_appear_exactly_where_the_record_carries_them(
    sr: list[dict[str, Any]],
) -> None:
    bare = {entry["address"] for entry in sr if "calibration" not in entry}
    assert bare == UNWIRED_ELEMENT
    assert {entry["address"] for entry in sr if "element" not in entry} == UNWIRED_ELEMENT
    assert sum("calibration" in entry for entry in sr) == 842


def test_unit_appears_exactly_where_the_channel_states_one(
    demo: FacilityDocument, sr: list[dict[str, Any]]
) -> None:
    units = {c["id"]: c["unit"] for c in demo["channels"] if "unit" in c}
    for entry in sr:
        if entry["address"] in units:
            assert entry["unit"] == units[entry["address"]]
        else:
            assert "unit" not in entry
    assert sum("unit" in entry for entry in sr) == 842


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
