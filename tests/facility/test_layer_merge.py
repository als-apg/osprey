"""Stage S1 (per-file load) and stage S2's layer merge on synthetic trees."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.facility.combine import CombineResult, combine
from osprey.facility.errors import FacilityBuildError
from osprey.facility.provenance import add_defaults, build_provenance, set_place_from
from osprey.facility.sources import load_sources


def _write(root: Path, files: dict[str, Any]) -> Path:
    for rel, data in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        text = data if isinstance(data, str) else yaml.safe_dump(data, sort_keys=False)
        path.write_text(text, encoding="utf-8")
    return root


def _load_errors(tmp_path: Path, files: dict[str, Any]) -> list[FacilityBuildError]:
    return load_sources(_write(tmp_path, files)).errors


def _build(tmp_path: Path, files: dict[str, Any]) -> CombineResult:
    loaded = load_sources(_write(tmp_path, files))
    assert [e.format_message() for e in loaded.errors] == []
    return combine(loaded.sources)


def _one(errors: list[FacilityBuildError]) -> FacilityBuildError:
    assert len(errors) == 1, [e.format_message() for e in errors]
    return errors[0]


def _record(result: CombineResult, plural: str, rid: str) -> dict[str, Any]:
    key = "name" if plural == "models" else "id"
    (record,) = [r for r in result.document[plural] if r[key] == rid]
    return record


class TestSourcesLoad:
    def test_missing_directory_is_no_sources(self, tmp_path: Path) -> None:
        loaded = load_sources(tmp_path / "absent")
        assert loaded.errors == []
        assert loaded.sources.records == []

    def test_on_key_is_not_read_as_a_boolean(self, tmp_path: Path) -> None:
        text = "- id: SR:BPM1:X\n  on: {device: BPM1}\n"
        loaded = load_sources(_write(tmp_path, {"records/channels.yaml": text}))
        assert loaded.errors == []
        (record,) = loaded.sources.records
        assert record.fields == {"on": {"device": "BPM1"}}
        assert (record.kind, record.layer, record.file) == (
            "channel",
            "authored",
            "records/channels.yaml",
        )

    def test_imported_directory_is_its_own_layer(self, tmp_path: Path) -> None:
        loaded = load_sources(
            _write(tmp_path, {"imported/mml/devices.yaml": [{"id": "Q1", "class": "Quad"}]})
        )
        (record,) = loaded.sources.records
        assert (record.layer, record.file) == ("mml", "imported/mml/devices.yaml")

    def test_wiring_id_is_model_slash_address(self, tmp_path: Path) -> None:
        models = [{"name": "lattice", "engine": "pyat", "wiring": [{"address": "A/B/C"}]}]
        loaded = load_sources(_write(tmp_path, {"models.yaml": models}))
        assert [(r.kind, r.id) for r in loaded.sources.records] == [
            ("model", "lattice"),
            ("wiring", "lattice/A/B/C"),
        ]

    def test_parse_failure_is_source_invalid(self, tmp_path: Path) -> None:
        err = _one(_load_errors(tmp_path, {"records/devices.yaml": "- id: [unclosed\n"}))
        assert err.kind == "source-invalid"
        assert err.record_kind == "path"
        assert err.sources == ("records/devices.yaml",)

    def test_unknown_key_is_source_invalid(self, tmp_path: Path) -> None:
        rows = [{"id": "Q1", "colour": "red"}]
        err = _one(_load_errors(tmp_path, {"records/devices.yaml": rows}))
        assert (err.kind, err.record_kind, err.record_id) == ("source-invalid", "device", "Q1")
        assert "unknown key `colour`" in err.detail

    @pytest.mark.parametrize("slot", ["model", "s", "length", "ordinalInPlace", "groups"])
    def test_computed_device_slot_is_source_invalid(self, tmp_path: Path, slot: str) -> None:
        rows = [{"id": "Q1", slot: 1}]
        err = _one(_load_errors(tmp_path, {"imported/mml/devices.yaml": rows}))
        assert (err.kind, err.record_id) == ("source-invalid", "Q1")
        assert f"`{slot}` is computed by the build" in err.detail

    @pytest.mark.parametrize("slot", ["direction", "unit", "default", "value_range"])
    def test_computed_wiring_slot_is_source_invalid(self, tmp_path: Path, slot: str) -> None:
        models = [{"name": "lattice", "wiring": [{"address": "SR:Q1", slot: 1}]}]
        err = _one(_load_errors(tmp_path, {"models.yaml": models}))
        assert (err.kind, err.record_kind, err.record_id) == (
            "source-invalid",
            "wiring",
            "lattice/SR:Q1",
        )

    def test_on_naming_both_kinds_is_source_invalid(self, tmp_path: Path) -> None:
        rows = [{"id": "SR:X", "on": {"device": "Q1", "place": "SR"}}]
        err = _one(_load_errors(tmp_path, {"imported/csv/channels.yaml": rows}))
        assert (err.kind, err.record_kind, err.record_id) == ("source-invalid", "channel", "SR:X")
        assert "both a device and a place" in err.detail

    def test_model_name_outside_pn_local_is_source_invalid(self, tmp_path: Path) -> None:
        err = _one(_load_errors(tmp_path, {"models.yaml": [{"name": "storage-lattice"}]}))
        assert (err.kind, err.record_kind, err.record_id) == (
            "source-invalid",
            "model",
            "storage-lattice",
        )

    def test_model_names_equal_but_for_case_are_source_invalid(self, tmp_path: Path) -> None:
        err = _one(
            _load_errors(
                tmp_path,
                {
                    "models.yaml": [{"name": "Lattice"}],
                    "imported/mml/models.yaml": [{"name": "lattice"}],
                },
            )
        )
        assert err.kind == "source-invalid"
        assert "Lattice, lattice" in err.detail
        assert err.sources == ("models.yaml", "imported/mml/models.yaml")

    def test_identity_code_outside_pn_local_is_source_invalid(self, tmp_path: Path) -> None:
        err = _one(_load_errors(tmp_path, {"identity.yaml": {"code": "bad-code"}}))
        assert (err.kind, err.record_id) == ("source-invalid", "identity.yaml.code")

    def test_reserved_layer_directory_names_the_directory(self, tmp_path: Path) -> None:
        rows = [{"id": "Q1"}]
        err = _one(_load_errors(tmp_path, {"imported/authored/devices.yaml": rows}))
        assert (err.kind, err.record_id) == ("source-invalid", "imported/authored")
        assert err.sources == ("imported/authored",)

    def test_id_twice_in_one_layer_is_layer_duplicate(self, tmp_path: Path) -> None:
        rows = [{"id": "Q1"}, {"id": "Q1", "class": "Quad"}]
        err = _one(_load_errors(tmp_path, {"imported/mml/devices.yaml": rows}))
        assert (err.kind, err.record_kind, err.record_id) == ("layer-duplicate", "device", "Q1")
        assert "layer mml" in err.detail

    def test_same_id_in_two_layers_is_not_a_duplicate(self, tmp_path: Path) -> None:
        errors = _load_errors(
            tmp_path,
            {
                "records/devices.yaml": [{"id": "Q1"}],
                "imported/mml/devices.yaml": [{"id": "Q1"}],
            },
        )
        assert errors == []

    def test_ids_repeat_across_kinds(self, tmp_path: Path) -> None:
        errors = _load_errors(
            tmp_path,
            {"records/places.yaml": [{"id": "A/B"}], "records/devices.yaml": [{"id": "A/B"}]},
        )
        assert errors == []

    def test_every_stop_is_collected(self, tmp_path: Path) -> None:
        errors = _load_errors(
            tmp_path,
            {
                "records/devices.yaml": [{"id": "Q1", "s": 1.0}],
                "records/channels.yaml": [{"id": "X", "bogus": 1}],
            },
        )
        assert [e.record_id for e in errors] == ["X", "Q1"]


class TestLayerMerge:
    def test_fields_are_collected_per_layer(self, tmp_path: Path) -> None:
        result = _build(
            tmp_path,
            {
                "records/channels.yaml": [{"id": "SR:X", "description": "Horizontal position"}],
                "imported/mml/channels.yaml": [{"id": "SR:X", "unit": "mm"}],
            },
        )
        assert result.errors == []
        channel = _record(result, "channels", "SR:X")
        assert channel["description"] == "Horizontal position"
        assert channel["unit"] == "mm"
        assert channel["provenance"]["sources"] == [
            {"layer": "authored", "file": "records/channels.yaml", "fields": ["description"]},
            {"layer": "mml", "file": "imported/mml/channels.yaml", "fields": ["unit"]},
        ]

    def test_equal_values_do_not_conflict(self, tmp_path: Path) -> None:
        result = _build(
            tmp_path,
            {
                "records/channels.yaml": [{"id": "SR:X", "unit": "mm"}],
                "imported/mml/channels.yaml": [{"id": "SR:X", "unit": "mm"}],
            },
        )
        assert result.errors == []

    def test_unequal_values_are_layer_conflict(self, tmp_path: Path) -> None:
        result = _build(
            tmp_path,
            {
                "records/channels.yaml": [{"id": "SR:X", "unit": "mm"}],
                "imported/mml/channels.yaml": [{"id": "SR:X", "unit": "m"}],
            },
        )
        err = _one(result.errors)
        assert (err.kind, err.record_kind, err.record_id) == ("layer-conflict", "channel", "SR:X")
        assert err.detail == "`unit` differs: authored=mm; mml=m"
        assert err.sources == ("imported/mml/channels.yaml", "records/channels.yaml")

    def test_a_boolean_never_equals_a_number(self, tmp_path: Path) -> None:
        result = _build(
            tmp_path,
            {
                "records/devices.yaml": [{"id": "Q1", "properties": {"on": True}}],
                "imported/mml/devices.yaml": [{"id": "Q1", "properties": {"on": 1}}],
            },
        )
        assert _one(result.errors).kind == "layer-conflict"

    def test_map_slot_compares_whole(self, tmp_path: Path) -> None:
        result = _build(
            tmp_path,
            {
                "models.yaml": [{"name": "lattice", "settings": {"pyat": {"twiss_in": [1, 2]}}}],
                "imported/mml/models.yaml": [
                    {"name": "lattice", "settings": {"pyat": {"solve": "closed"}}}
                ],
            },
        )
        err = _one(result.errors)
        assert (err.kind, err.record_kind, err.record_id) == ("layer-conflict", "model", "lattice")
        assert err.detail.startswith("`settings` differs:")

    def test_differing_on_is_layer_conflict(self, tmp_path: Path) -> None:
        result = _build(
            tmp_path,
            {
                "records/channels.yaml": [{"id": "SR:X", "on": {"device": "BPM1"}}],
                "imported/csv/channels.yaml": [{"id": "SR:X", "on": {"place": "SR/S01"}}],
            },
        )
        err = _one(result.errors)
        assert err.kind == "layer-conflict"
        assert err.detail.startswith("`on` differs:")

    def test_role_less_list_and_mml_import_build(self, tmp_path: Path) -> None:
        result = _build(
            tmp_path,
            {
                "imported/csv/channels.yaml": [{"id": "SR:Q1:SP", "on": {"device": "Q1"}}],
                "imported/mml/channels.yaml": [{"id": "SR:Q1:SP", "role": "setpoint"}],
            },
        )
        assert result.errors == []
        channel = _record(result, "channels", "SR:Q1:SP")
        assert channel["role"] == "setpoint"
        assert channel["pair"] == "SR:Q1:SP"
        assert "role" not in channel["provenance"]["defaults"]
        assert "pair" in channel["provenance"]["defaults"]

    def test_role_less_address_in_two_lists_takes_the_default(self, tmp_path: Path) -> None:
        result = _build(
            tmp_path,
            {
                "imported/csv/channels.yaml": [{"id": "SR:X", "unit": "mm"}],
                "imported/mml/channels.yaml": [{"id": "SR:X", "role": "readback"}],
                "records/channels.yaml": [{"id": "SR:Y"}],
            },
        )
        assert result.errors == []
        assert "role" not in _record(result, "channels", "SR:X")["provenance"]["defaults"]
        other = _record(result, "channels", "SR:Y")
        assert other["role"] == "readback"
        assert other["value_type"] == "float"
        assert "on" not in other
        assert other["provenance"]["defaults"] == ["on", "role", "value_type"]

    def test_slice_weight_default_is_filled_before_comparison(self, tmp_path: Path) -> None:
        result = _build(
            tmp_path,
            {
                "models.yaml": [
                    {
                        "name": "lattice",
                        "wiring": [{"address": "SR:Q1", "slices": [{"element": "Q1"}]}],
                    }
                ],
                "imported/mml/models.yaml": [
                    {
                        "name": "lattice",
                        "wiring": [
                            {"address": "SR:Q1", "slices": [{"element": "Q1", "weight": 1}]}
                        ],
                    }
                ],
            },
        )
        assert result.errors == []
        (wiring,) = _record(result, "models", "lattice")["wiring"]
        assert wiring["slices"] == [{"element": "Q1", "weight": 1}]

    def test_slices_keep_source_order(self, tmp_path: Path) -> None:
        slices = [{"element": "B", "weight": 2}, {"element": "A"}]
        result = _build(
            tmp_path,
            {
                "records/channels.yaml": [{"id": "SR:Q:SP", "on": {"device": "Q"}}],
                "models.yaml": [
                    {"name": "lattice", "wiring": [{"address": "SR:Q:SP", "slices": slices}]}
                ],
            },
        )
        assert result.errors == []
        (wiring,) = _record(result, "models", "lattice")["wiring"]
        assert wiring["slices"] == [
            {"element": "B", "weight": 2, "device": "Q"},
            {"element": "A", "weight": 1, "device": "Q"},
        ]
        assert wiring["provenance"]["defaults"] == ["slices.device", "slices.weight"]

    def test_set_valued_lists_compare_sorted(self, tmp_path: Path) -> None:
        result = _build(
            tmp_path,
            {
                "records/channels.yaml": [{"id": "SR:X", "tags": ["orbit", "bpm"]}],
                "imported/mml/channels.yaml": [{"id": "SR:X", "tags": ["bpm", "orbit"]}],
            },
        )
        assert result.errors == []
        assert _record(result, "channels", "SR:X")["tags"] == ["bpm", "orbit"]

    def test_ordered_lists_compare_in_source_order(self, tmp_path: Path) -> None:
        result = _build(
            tmp_path,
            {
                "records/channels.yaml": [{"id": "SR:X", "names": ["b", "a"]}],
                "imported/mml/channels.yaml": [{"id": "SR:X", "names": ["a", "b"]}],
            },
        )
        assert _one(result.errors).kind == "layer-conflict"

    def test_bool_options_default(self, tmp_path: Path) -> None:
        result = _build(
            tmp_path, {"records/channels.yaml": [{"id": "SR:ON", "value_type": "bool"}]}
        )
        channel = _record(result, "channels", "SR:ON")
        assert channel["options"] == ["FALSE", "TRUE"]
        assert "options" in channel["provenance"]["defaults"]

    def test_seed_merges_as_the_channel_simulation(self, tmp_path: Path) -> None:
        result = _build(
            tmp_path,
            {
                "imported/mml/channels.yaml": [{"id": "SR:X"}],
                "seeds.yaml": {"SR:X": {"nominal": 0.5}},
            },
        )
        channel = _record(result, "channels", "SR:X")
        assert channel["simulation"] == {"nominal": 0.5}
        assert {"layer": "authored", "file": "seeds.yaml", "fields": ["simulation"]} in channel[
            "provenance"
        ]["sources"]

    def test_emission_order(self, tmp_path: Path) -> None:
        result = _build(
            tmp_path,
            {
                "records/devices.yaml": [{"id": "Q2"}, {"id": "Q1"}],
                "models.yaml": [
                    {"name": "texture"},
                    {"name": "lattice", "wiring": [{"address": "Z"}, {"address": "A"}]},
                    {"name": "booster"},
                ],
                "classes.yaml": [{"class": "Quad"}, {"class": "BPM"}],
            },
        )
        document = result.document
        assert [d["id"] for d in document["devices"]] == ["Q1", "Q2"]
        assert [m["name"] for m in document["models"]] == ["booster", "lattice", "texture"]
        lattice = _record(result, "models", "lattice")
        assert [w["id"] for w in lattice["wiring"]] == ["lattice/A", "lattice/Z"]
        assert list(lattice)[-2:] == ["wiring", "provenance"]
        assert [c["class"] for c in document["classes"]] == ["BPM", "Quad"]


class TestProvenance:
    def test_sources_are_joined_and_sorted(self) -> None:
        block = build_provenance(
            sources=[
                ("mml", "b.yaml", ["unit"]),
                ("authored", "a.yaml", ["x"]),
                ("mml", "b.yaml", ["on"]),
            ]
        )
        assert block == {
            "sources": [
                {"layer": "authored", "file": "a.yaml", "fields": ["x"]},
                {"layer": "mml", "file": "b.yaml", "fields": ["on", "unit"]},
            ],
            "fixes": [],
            "defaults": [],
        }

    def test_defaults_and_place_from(self) -> None:
        block = add_defaults(build_provenance(defaults=["role"]), ["on", "role"])
        assert block["defaults"] == ["on", "role"]
        assert set_place_from(block, "span")["place_from"] == "span"
        with pytest.raises(ValueError, match="place_from"):
            set_place_from(block, "guess")
