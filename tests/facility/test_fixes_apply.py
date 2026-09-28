"""fixes.yaml applied by stage S2: set, add and drop, and every fix stop."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.facility.combine import FIXES_HEADER, CombineResult, combine
from osprey.facility.errors import FacilityBuildError
from osprey.facility.sources import load_sources

#: A small tree with one authored-only device and records two layers state.
_TREE: dict[str, Any] = {
    "records/places.yaml": [{"id": "SR"}, {"id": "SR/S01"}],
    "records/devices.yaml": [{"id": "BPM9", "class": "BPM"}],
    "records/channels.yaml": [{"id": "SR:Q1:SP", "description": "Q1 current"}],
    "imported/mml/devices.yaml": [
        {"id": "Q1", "class": "Quad", "place": "SR/S01"},
        {"id": "Q2", "class": "Quad"},
    ],
    "imported/mml/channels.yaml": [
        {"id": "SR:Q1:SP", "role": "readback", "on": {"device": "Q1"}},
        {"id": "SR:Q2:SP", "role": "setpoint", "former_addresses": ["SR:Q2:OLD"]},
    ],
    "imported/mml/groups.yaml": [{"id": "SR/QUAD", "members": ["Q2"]}],
    "imported/mml/models.yaml": [
        {
            "name": "lattice",
            "settings": {"pyat": {"solve": "closed"}},
            "wiring": [
                {"address": "SR:Q1:SP", "element": "Q1"},
                {"address": "SR:Q2:SP", "element": "Q2"},
            ],
        },
        {"name": "booster", "wiring": [{"address": "BR:Q:SP", "element": "Q"}]},
    ],
}


def _fixes(*entries: dict[str, Any]) -> dict[str, Any]:
    return {"schema": FIXES_HEADER, "fixes": list(entries)}


def _build(tmp_path: Path, fixes: Any = None, **overrides: Any) -> CombineResult:
    files = {**_TREE, **{k.replace("__", "/"): v for k, v in overrides.items()}}
    if fixes is not None:
        files["fixes.yaml"] = fixes
    for rel, data in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    loaded = load_sources(tmp_path)
    assert [e.format_message() for e in loaded.errors] == []
    return combine(loaded.sources)


def _one(errors: list[FacilityBuildError]) -> FacilityBuildError:
    assert len(errors) == 1, [e.format_message() for e in errors]
    return errors[0]


def _ids(result: CombineResult, plural: str) -> list[str]:
    key = "name" if plural == "models" else "id"
    return [record[key] for record in result.document[plural]]


def _record(result: CombineResult, plural: str, rid: str) -> dict[str, Any]:
    key = "name" if plural == "models" else "id"
    (record,) = [r for r in result.document[plural] if r[key] == rid]
    return record


_SET_ROLE = {
    "op": "set",
    "kind": "channel",
    "id": "SR:Q1:SP",
    "fields": {"role": "setpoint"},
    "was": {"role": {"mml": "readback"}},
    "why": "The export lists the setpoint as a monitor.",
}


def test_the_tree_builds_without_fixes(tmp_path: Path) -> None:
    assert _build(tmp_path).errors == []


class TestSet:
    def test_set_replaces_the_field_and_is_recorded(self, tmp_path: Path) -> None:
        result = _build(tmp_path, _fixes(_SET_ROLE))
        assert result.errors == []
        channel = _record(result, "channels", "SR:Q1:SP")
        assert channel["role"] == "setpoint"
        assert channel["pair"] == "SR:Q1:SP"
        assert channel["provenance"]["fixes"] == [
            {"op": "set", "why": "The export lists the setpoint as a monitor."}
        ]

    def test_set_clears_a_layer_conflict(self, tmp_path: Path) -> None:
        conflicting = [{"id": "SR:Q1:SP", "role": "setpoint"}]
        assert _one(_build(tmp_path / "a", **{"imported__csv__channels.yaml": conflicting}).errors)
        fix = {**_SET_ROLE, "was": {"role": {"csv": "setpoint", "mml": "readback"}}}
        result = _build(
            tmp_path / "b", _fixes(fix), **{"imported__csv__channels.yaml": conflicting}
        )
        assert result.errors == []

    def test_map_slot_conflict_is_cleared_by_a_whole_set(self, tmp_path: Path) -> None:
        authored = [{"name": "lattice", "settings": {"pyat": {"twiss_in": [1, 2]}}}]
        err = _one(_build(tmp_path / "a", **{"models.yaml": authored}).errors)
        assert (err.kind, err.record_kind, err.record_id) == ("layer-conflict", "model", "lattice")
        fix = {
            "op": "set",
            "kind": "model",
            "id": "lattice",
            "fields": {"settings": {"pyat": {"solve": "closed", "twiss_in": [1, 2]}}},
            "was": {
                "settings": {
                    "authored": {"pyat": {"twiss_in": [1, 2]}},
                    "mml": {"pyat": {"solve": "closed"}},
                }
            },
            "why": "Both keys are needed.",
        }
        result = _build(tmp_path / "b", _fixes(fix), **{"models.yaml": authored})
        assert result.errors == []
        assert _record(result, "models", "lattice")["settings"] == {
            "pyat": {"solve": "closed", "twiss_in": [1, 2]}
        }

    def test_set_of_a_field_no_layer_states_needs_no_was(self, tmp_path: Path) -> None:
        fix = {
            "op": "set",
            "kind": "device",
            "id": "Q2",
            "fields": {"place": "SR/S01"},
            "why": "The export omits it.",
        }
        result = _build(tmp_path, _fixes(fix))
        assert result.errors == []
        assert _record(result, "devices", "Q2")["place"] == "SR/S01"

    def test_missing_was_is_fix_stale_with_the_block_to_paste(self, tmp_path: Path) -> None:
        fix = {k: v for k, v in _SET_ROLE.items() if k != "was"}
        err = _one(_build(tmp_path, _fixes(fix)).errors)
        assert (err.kind, err.record_kind, err.record_id) == ("fix-stale", "channel", "SR:Q1:SP")
        assert err.detail == "`was` is missing"
        assert "was: {role: {mml: readback}}" in err.remedy

    def test_changed_layer_value_is_fix_stale(self, tmp_path: Path) -> None:
        fix = {**_SET_ROLE, "was": {"role": {"mml": "none"}}}
        err = _one(_build(tmp_path, _fixes(fix)).errors)
        assert err.kind == "fix-stale"
        assert err.detail == "`was` does not match the layers"
        assert "was: {role: {mml: readback}}" in err.remedy

    @pytest.mark.parametrize(
        ("kind", "rid", "fields"),
        [
            ("device", "Q1", {"s": 1.0}),
            ("device", "Q1", {"model": "lattice"}),
            ("device", "Q1", {"groups": ["SR/QUAD"]}),
            ("wiring", "lattice/SR:Q1:SP", {"default": 1.0}),
        ],
    )
    def test_set_of_a_computed_slot_is_fix_computed(
        self, tmp_path: Path, kind: str, rid: str, fields: dict[str, Any]
    ) -> None:
        fix = {"op": "set", "kind": kind, "id": rid, "fields": fields, "why": "x"}
        err = _one(_build(tmp_path, _fixes(fix)).errors)
        assert (err.kind, err.record_kind, err.record_id) == ("fix-computed", kind, rid)

    def test_fix_on_an_authored_only_record_is_fix_authored(self, tmp_path: Path) -> None:
        fix = {"op": "set", "kind": "device", "id": "BPM9", "fields": {"class": "X"}, "why": "x"}
        err = _one(_build(tmp_path, _fixes(fix)).errors)
        assert (err.kind, err.record_id) == ("fix-authored", "BPM9")
        assert err.remedy == "edit data/facility/records/devices.yaml"


class TestAdd:
    def test_add_creates_the_record(self, tmp_path: Path) -> None:
        fix = {
            "op": "add",
            "kind": "device",
            "id": "Q9",
            "record": {"class": "Quad"},
            "why": "Missing from the export.",
        }
        result = _build(tmp_path, _fixes(fix))
        assert result.errors == []
        device = _record(result, "devices", "Q9")
        assert device["class"] == "Quad"
        assert device["provenance"] == {
            "sources": [],
            "fixes": [{"op": "add", "why": "Missing from the export."}],
            "defaults": [],
        }

    def test_add_of_a_model_adds_its_wiring(self, tmp_path: Path) -> None:
        fix = {
            "op": "add",
            "kind": "model",
            "id": "linac",
            "record": {"engine": "pyat", "wiring": [{"address": "LN:Q:SP", "element": "Q"}]},
            "why": "Not exported.",
        }
        result = _build(tmp_path, _fixes(fix))
        assert result.errors == []
        assert [w["id"] for w in _record(result, "models", "linac")["wiring"]] == ["linac/LN:Q:SP"]

    def test_add_of_a_produced_id_is_fix_duplicate(self, tmp_path: Path) -> None:
        fix = {"op": "add", "kind": "device", "id": "Q1", "record": {}, "why": "x"}
        err = _one(_build(tmp_path, _fixes(fix)).errors)
        assert (err.kind, err.record_id) == ("fix-duplicate", "Q1")
        assert "imported/mml/devices.yaml" in err.sources

    def test_add_writing_a_computed_slot_is_fix_computed(self, tmp_path: Path) -> None:
        fix = {"op": "add", "kind": "device", "id": "Q9", "record": {"s": 2.0}, "why": "x"}
        assert _one(_build(tmp_path, _fixes(fix)).errors).kind == "fix-computed"


class TestDrop:
    def test_drop_of_a_channel_removes_its_wiring(self, tmp_path: Path) -> None:
        fix = {"op": "drop", "kind": "channel", "id": "SR:Q2:SP", "why": "Not a real PV."}
        result = _build(tmp_path, _fixes(fix))
        assert result.errors == []
        assert "SR:Q2:SP" not in _ids(result, "channels")
        wiring = _record(result, "models", "lattice")["wiring"]
        assert [w["id"] for w in wiring] == ["lattice/SR:Q1:SP"]
        assert result.dropped[("wiring", "lattice/SR:Q2:SP")]["id"] == "SR:Q2:SP"

    def test_drop_of_a_model_removes_its_wiring(self, tmp_path: Path) -> None:
        fix = {"op": "drop", "kind": "model", "id": "booster", "why": "Not simulated."}
        result = _build(tmp_path, _fixes(fix))
        assert result.errors == []
        assert _ids(result, "models") == ["lattice"]
        assert ("wiring", "booster/BR:Q:SP") in result.dropped

    def test_drop_of_a_referenced_device_is_fix_referenced(self, tmp_path: Path) -> None:
        fix = {"op": "drop", "kind": "device", "id": "Q2", "why": "x"}
        channels = [{"id": "SR:Q2:SP", "on": {"device": "Q2"}}]
        result = _build(tmp_path, _fixes(fix), **{"imported__csv__channels.yaml": channels})
        err = _one(result.errors)
        assert (err.kind, err.record_kind, err.record_id) == ("fix-referenced", "device", "Q2")
        assert err.detail == (
            "`drop` leaves 2 referrer(s): channel SR:Q2:SP (on); group SR/QUAD (members)"
        )

    def test_drop_with_every_referrer_dropped_or_repointed_builds(self, tmp_path: Path) -> None:
        fixes = _fixes(
            {"op": "drop", "kind": "device", "id": "Q1", "why": "x"},
            {"op": "drop", "kind": "channel", "id": "SR:Q1:SP", "why": "x"},
            {"op": "drop", "kind": "device", "id": "Q2", "why": "x"},
            {
                "op": "set",
                "kind": "group",
                "id": "SR/QUAD",
                "fields": {"members": []},
                "was": {"members": {"mml": ["Q2"]}},
                "why": "x",
            },
        )
        assert _build(tmp_path, fixes).errors == []

    def test_drop_of_a_place_with_a_child_is_fix_referenced(self, tmp_path: Path) -> None:
        places = [{"id": "SR"}, {"id": "SR/S01"}]
        fix = {"op": "drop", "kind": "place", "id": "SR", "why": "x"}
        result = _build(tmp_path, _fixes(fix), **{"imported__csv__places.yaml": places})
        err = _one(result.errors)
        assert err.kind == "fix-referenced"
        assert "place SR/S01 (parent)" in err.detail

    def test_drop_of_a_model_named_by_a_span_is_fix_referenced(self, tmp_path: Path) -> None:
        places = [{"id": "BR", "span": {"model": "booster", "from_marker": "START"}}]
        fix = {"op": "drop", "kind": "model", "id": "booster", "why": "x"}
        result = _build(tmp_path, _fixes(fix), **{"records__places.yaml": places})
        err = _one(result.errors)
        assert (err.kind, err.record_kind) == ("fix-referenced", "model")
        assert "place BR (span)" in err.detail

    def test_drop_of_a_group_named_by_a_measurement_is_fix_referenced(self, tmp_path: Path) -> None:
        fix = {"op": "drop", "kind": "group", "id": "SR/QUAD", "why": "x"}
        measurement = {"kinds": ["orm"], "groups": {"quad": "SR/QUAD"}}
        result = _build(tmp_path, _fixes(fix), **{"measurement__lattice.yaml": measurement})
        err = _one(result.errors)
        assert (err.kind, err.record_kind) == ("fix-referenced", "group")
        assert "measurement/lattice.yaml (groups)" in err.detail


class TestFixStops:
    def test_fix_of_a_missing_record_is_fix_missing(self, tmp_path: Path) -> None:
        fix = {"op": "drop", "kind": "device", "id": "Q7", "why": "x"}
        err = _one(_build(tmp_path, _fixes(fix)).errors)
        assert (err.kind, err.record_kind, err.record_id) == ("fix-missing", "device", "Q7")
        assert err.remedy == "remove the fix from fixes.yaml"

    def test_fix_missing_names_the_former_addresses_match(self, tmp_path: Path) -> None:
        fix = {"op": "drop", "kind": "channel", "id": "SR:Q2:OLD", "why": "x"}
        err = _one(_build(tmp_path, _fixes(fix)).errors)
        assert err.kind == "fix-missing"
        assert "channel SR:Q2:SP lists SR:Q2:OLD in former_addresses" in err.detail
        assert err.remedy == "point the fix at channel SR:Q2:SP"

    def test_two_fixes_on_one_record_are_fix_duplicate(self, tmp_path: Path) -> None:
        fixes = _fixes(_SET_ROLE, {"op": "drop", "kind": "channel", "id": "SR:Q1:SP", "why": "x"})
        err = _one(_build(tmp_path, fixes).errors)
        assert (err.kind, err.record_id) == ("fix-duplicate", "SR:Q1:SP")
        assert "`set` and `drop`" in err.detail

    @pytest.mark.parametrize(
        "fixes",
        [
            {"schema": "osprey.facility.fixes/2", "fixes": []},
            _fixes({"op": "move", "kind": "device", "id": "Q1", "why": "x"}),
            _fixes({"op": "drop", "kind": "device", "id": "Q1"}),
            _fixes({"op": "set", "kind": "device", "id": "Q1", "why": "x"}),
            _fixes({"op": "drop", "kind": "device", "id": "Q1", "why": "x", "index": 3}),
        ],
    )
    def test_malformed_fixes_file_is_source_invalid(self, tmp_path: Path, fixes: Any) -> None:
        err = _one(_build(tmp_path, fixes).errors)
        assert (err.kind, err.record_kind) == ("source-invalid", "path")
        assert err.sources == ("fixes.yaml",)


def test_permuted_fixes_give_a_byte_equal_document(tmp_path: Path) -> None:
    entries = [
        _SET_ROLE,
        {"op": "add", "kind": "device", "id": "Q9", "record": {"class": "Quad"}, "why": "a"},
        {"op": "drop", "kind": "model", "id": "booster", "why": "b"},
        {"op": "drop", "kind": "channel", "id": "SR:Q2:SP", "why": "c"},
    ]
    documents = set()
    for index, order in enumerate(itertools.permutations(entries)):
        result = _build(tmp_path / str(index), _fixes(*order))
        assert result.errors == []
        documents.add(json.dumps(result.document, sort_keys=False))
    assert len(documents) == 1


def test_a_wiring_add_before_its_model_add_builds_in_either_order(tmp_path: Path) -> None:
    entries = [
        {
            "op": "add",
            "kind": "wiring",
            "id": "transfer/TL:Q:SP",
            "record": {"element": "Q"},
            "why": "a",
        },
        {"op": "add", "kind": "model", "id": "transfer", "record": {}, "why": "b"},
    ]
    documents = set()
    for index, order in enumerate(itertools.permutations(entries)):
        result = _build(tmp_path / str(index), _fixes(*order))
        assert [e.format_message() for e in result.errors] == []
        documents.add(json.dumps(result.document, sort_keys=False))
    assert len(documents) == 1


def test_a_record_two_drops_remove_names_the_same_fix_in_either_order(tmp_path: Path) -> None:
    entries = [
        {"op": "drop", "kind": "model", "id": "lattice", "why": "a"},
        {"op": "drop", "kind": "channel", "id": "SR:Q2:SP", "why": "b"},
    ]
    dropped = []
    for index, order in enumerate(itertools.permutations(entries)):
        result = _build(tmp_path / str(index), _fixes(*order))
        dropped.append(result.dropped)
    assert dropped[0] == dropped[1]
    assert dropped[0][("wiring", "lattice/SR:Q2:SP")]["id"] == "SR:Q2:SP"
