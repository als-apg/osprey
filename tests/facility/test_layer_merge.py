"""Stage S1 (per-file load) and stage S2's layer merge on synthetic trees."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

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


def _one(errors: list[FacilityBuildError]) -> FacilityBuildError:
    assert len(errors) == 1, [e.format_message() for e in errors]
    return errors[0]


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
