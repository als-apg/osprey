"""Keyed sets: ``scheme_names`` and ``citations`` on places and devices.

An entry's key is ``(name, scheme)`` for an alias and ``(document, version,
locator)`` for a citation; an absent key field is its own value. Identical
entries collapse, the set is ordered by key, and neither entry order nor map
key order changes the built bytes. One key with two contents is refused.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.facility.build import build_facility
from osprey.facility.combine import FIXES_HEADER, CombineResult, combine
from osprey.facility.errors import FacilityBuildError
from osprey.facility.render import facility_bytes
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


def _combine(tmp_path: Path, files: dict[str, Any]) -> CombineResult:
    loaded = load_sources(_write(tmp_path, files))
    assert [e.format_message() for e in loaded.errors] == []
    return combine(loaded.sources)


def _one(errors: list[FacilityBuildError]) -> FacilityBuildError:
    assert len(errors) == 1, [e.format_message() for e in errors]
    return errors[0]


def _device(result: CombineResult, rid: str) -> dict[str, Any]:
    assert [e.format_message() for e in result.errors] == []
    (record,) = [r for r in result.document["devices"] if r["id"] == rid]
    return record


_DOCS = [{"id": "spec-1", "title": "Magnet list"}, {"id": "dwg-2", "title": "Rack layout"}]

_ALIASES = [
    {"name": "QF1", "scheme": "lattice"},
    {"name": "SR01C:QF1", "scheme": "controls", "basis": "stated"},
    {"name": "QF1", "scheme": "cable-schedule"},
]

_CITATIONS = [
    {"document": "spec-1", "version": "B", "locator": "p. 4"},
    {"document": "spec-1"},
    {"document": "dwg-2", "locator": "sheet 3"},
    {"document": "spec-1", "version": "A"},
]


def _tree(aliases: list[Any], citations: list[Any]) -> dict[str, Any]:
    return {
        "documents.yaml": _DOCS,
        "records/places.yaml": [
            {"id": "SR", "scheme_names": aliases, "citations": citations},
        ],
        "records/devices.yaml": [
            {
                "id": "QF1",
                "class": "Quadrupole",
                "place": "SR",
                "scheme_names": aliases,
                "citations": citations,
            },
        ],
    }


def _reversed_keys(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(reversed(list(entry.items()))) for entry in entries]


class TestNormalization:
    def test_permuted_entries_and_map_keys_build_identical_bytes(self, tmp_path: Path) -> None:
        first = build_facility(
            _write(tmp_path / "a", _tree(_ALIASES, _CITATIONS)), project_name="p"
        )
        permuted = _tree(
            _reversed_keys(list(reversed(_ALIASES))),
            _reversed_keys([_CITATIONS[2], _CITATIONS[0], _CITATIONS[3], _CITATIONS[1]]),
        )
        second = build_facility(_write(tmp_path / "b", permuted), project_name="p")
        assert facility_bytes(first) == facility_bytes(second)

    def test_set_is_ordered_by_key_with_absent_fields_first(self, tmp_path: Path) -> None:
        device = _device(_combine(tmp_path, _tree(_ALIASES, _CITATIONS)), "QF1")
        assert device["citations"] == [
            {"document": "dwg-2", "locator": "sheet 3"},
            {"document": "spec-1"},
            {"document": "spec-1", "version": "A"},
            {"document": "spec-1", "version": "B", "locator": "p. 4"},
        ]
        assert [(a["name"], a["scheme"]) for a in device["scheme_names"]] == [
            ("QF1", "cable-schedule"),
            ("QF1", "lattice"),
            ("SR01C:QF1", "controls"),
        ]

    def test_identical_entries_collapse(self, tmp_path: Path) -> None:
        aliases = [*_ALIASES, _reversed_keys(_ALIASES)[1]]
        citations = [*_CITATIONS, dict(_CITATIONS[0]), {"document": "spec-1"}]
        result = _combine(tmp_path, _tree(aliases, citations))
        device = _device(result, "QF1")
        assert len(device["scheme_names"]) == 3
        assert len(device["citations"]) == 4
        (place,) = result.document["places"]
        assert len(place["citations"]) == 4

    def test_a_citation_without_locator_and_one_with_it_are_two_entries(
        self, tmp_path: Path
    ) -> None:
        citations = [{"document": "spec-1", "locator": "p. 4"}, {"document": "spec-1"}]
        device = _device(_combine(tmp_path, _tree([], citations)), "QF1")
        assert device["citations"] == [
            {"document": "spec-1"},
            {"document": "spec-1", "locator": "p. 4"},
        ]

    def test_string_sets_keep_their_order_by_string_form(self, tmp_path: Path) -> None:
        files = {
            "records/devices.yaml": [{"id": "Q1", "class": "Quadrupole", "names": ["b", "a", "b"]}],
        }
        device = _device(_combine(tmp_path, files), "Q1")
        assert device["names"] == ["a", "b", "b"]


class TestLoadRefusal:
    @pytest.mark.parametrize(
        ("slot", "entries", "key"),
        [
            (
                "scheme_names",
                [
                    {"name": "QF1", "scheme": "lattice"},
                    {"name": "QF1", "scheme": "lattice", "basis": "stated"},
                ],
                "name QF1, scheme lattice",
            ),
            (
                "citations",
                [
                    {"document": "spec-1", "state": "released"},
                    {"document": "spec-1", "state": "draft"},
                ],
                "document spec-1, no version, no locator",
            ),
        ],
    )
    def test_one_key_with_two_contents_is_refused(
        self, tmp_path: Path, slot: str, entries: list[Any], key: str
    ) -> None:
        files = {"imported/mml/devices.yaml": [{"id": "QF1", "class": "Quadrupole", slot: entries}]}
        error = _one(_load_errors(tmp_path, files))
        assert error.kind == "source-invalid"
        assert (error.record_kind, error.record_id) == ("device", "QF1")
        assert error.sources == ("imported/mml/devices.yaml",)
        assert error.detail == f"`{slot}` states two contents for key {key}"

    def test_a_place_is_checked_too(self, tmp_path: Path) -> None:
        citations = [
            {"document": "spec-1", "version": 2},
            {"document": "spec-1", "version": 2, "locator": None, "state": "x"},
        ]
        error = _one(
            _load_errors(tmp_path, {"records/places.yaml": [{"id": "SR", "citations": citations}]})
        )
        assert (error.kind, error.record_kind, error.record_id) == ("source-invalid", "place", "SR")

    def test_permuted_map_keys_are_one_content(self, tmp_path: Path) -> None:
        citations = [_CITATIONS[0], _reversed_keys([_CITATIONS[0]])[0]]
        assert _load_errors(tmp_path, _tree(_ALIASES, citations)) == []


_LAYERED: dict[str, Any] = {
    "documents.yaml": _DOCS,
    "records/devices.yaml": [
        {"id": "QF1", "class": "Quadrupole", "citations": [{"document": "spec-1"}]},
    ],
    "imported/mml/devices.yaml": [
        {"id": "QF1", "citations": [{"document": "dwg-2"}, {"document": "spec-1"}]},
    ],
}


def _fixes(*entries: dict[str, Any]) -> dict[str, Any]:
    return {"schema": FIXES_HEADER, "fixes": list(entries)}


class TestAcrossLayers:
    def test_equal_sets_in_any_order_agree(self, tmp_path: Path) -> None:
        files = {
            **_LAYERED,
            "records/devices.yaml": [
                {
                    "id": "QF1",
                    "class": "Quadrupole",
                    "citations": [{"document": "spec-1"}, {"document": "dwg-2"}],
                },
            ],
        }
        device = _device(_combine(tmp_path, files), "QF1")
        assert device["citations"] == [{"document": "dwg-2"}, {"document": "spec-1"}]

    def test_unequal_sets_conflict(self, tmp_path: Path) -> None:
        error = _one(_combine(tmp_path, _LAYERED).errors)
        assert error.kind == "layer-conflict"
        assert (error.record_kind, error.record_id) == ("device", "QF1")
        assert "citations" in error.detail

    def test_a_fix_naming_the_field_resolves_it(self, tmp_path: Path) -> None:
        fix = {
            "op": "set",
            "kind": "device",
            "id": "QF1",
            "fields": {"citations": [{"document": "spec-1"}, {"document": "dwg-2"}]},
            "was": {
                "citations": {
                    "authored": [{"document": "spec-1"}],
                    "mml": [{"document": "spec-1"}, {"document": "dwg-2"}],
                }
            },
            "why": "The drawing states the magnet too.",
        }
        result = _combine(tmp_path, {**_LAYERED, "fixes.yaml": _fixes(fix)})
        device = _device(result, "QF1")
        assert device["citations"] == [{"document": "dwg-2"}, {"document": "spec-1"}]


_NAMED: dict[str, Any] = {
    "documents.yaml": _DOCS,
    "records/places.yaml": [
        {"id": "SR", "scheme_names": [{"name": "Ring", "scheme": "drawings"}]},
    ],
    "records/devices.yaml": [
        {
            "id": "QF1",
            "class": "Quadrupole",
            "scheme_names": [{"name": "QF1", "scheme": "lattice"}],
        },
    ],
}


def _named(devices: list[Any], places: list[Any] | None = None) -> dict[str, Any]:
    files = {**_NAMED, "imported/cables/devices.yaml": devices}
    if places is not None:
        files["imported/cables/places.yaml"] = places
    return files


class TestSchemeNamesAcrossLayers:
    def test_a_layer_adds_a_name_to_a_device_another_source_states(self, tmp_path: Path) -> None:
        added = {"name": "Q-0301", "scheme": "cable-schedule", "source": {"document": "dwg-2"}}
        result = _combine(tmp_path, _named([{"id": "QF1", "scheme_names": [added]}]))
        device = _device(result, "QF1")
        assert device["scheme_names"] == [
            {"name": "Q-0301", "scheme": "cable-schedule", "source": {"document": "dwg-2"}},
            {"name": "QF1", "scheme": "lattice"},
        ]
        assert result.warnings == []
        assert [(s["layer"], s["fields"]) for s in device["provenance"]["sources"]] == [
            ("authored", ["class", "scheme_names"]),
            ("cables", ["scheme_names"]),
        ]

    def test_a_layer_adds_a_name_to_a_place_another_source_states(self, tmp_path: Path) -> None:
        added = {"name": "R", "scheme": "survey"}
        result = _combine(tmp_path, _named([], [{"id": "SR", "scheme_names": [added]}]))
        assert [e.format_message() for e in result.errors] == []
        (place,) = result.document["places"]
        assert [(a["name"], a["scheme"]) for a in place["scheme_names"]] == [
            ("R", "survey"),
            ("Ring", "drawings"),
        ]

    def test_an_entry_two_layers_state_alike_is_one_entry(self, tmp_path: Path) -> None:
        same = {"scheme": "lattice", "name": "QF1"}
        device = _device(_combine(tmp_path, _named([{"id": "QF1", "scheme_names": [same]}])), "QF1")
        assert device["scheme_names"] == [{"name": "QF1", "scheme": "lattice"}]

    def test_one_key_with_two_contents_across_layers_conflicts(self, tmp_path: Path) -> None:
        other = {"name": "QF1", "scheme": "lattice", "basis": "inferred", "rule": "guess"}
        error = _one(_combine(tmp_path, _named([{"id": "QF1", "scheme_names": [other]}])).errors)
        assert error.kind == "layer-conflict"
        assert (error.record_kind, error.record_id) == ("device", "QF1")
        assert error.sources == ("imported/cables/devices.yaml", "records/devices.yaml")
        assert error.detail == (
            "`scheme_names` states two contents for key name QF1, scheme lattice: authored, cables"
        )

    def test_a_fix_naming_the_field_resolves_the_conflict(self, tmp_path: Path) -> None:
        other = {"name": "QF1", "scheme": "lattice", "basis": "inferred", "rule": "guess"}
        fix = {
            "op": "set",
            "kind": "device",
            "id": "QF1",
            "fields": {"scheme_names": [{"name": "QF1", "scheme": "lattice"}]},
            "was": {
                "scheme_names": {
                    "authored": [{"name": "QF1", "scheme": "lattice"}],
                    "cables": [other],
                }
            },
            "why": "The lattice file states the name.",
        }
        files = {**_named([{"id": "QF1", "scheme_names": [other]}]), "fixes.yaml": _fixes(fix)}
        device = _device(_combine(tmp_path, files), "QF1")
        assert device["scheme_names"] == [{"name": "QF1", "scheme": "lattice"}]


class TestFixValues:
    def test_a_set_with_two_contents_under_one_key_is_refused(self, tmp_path: Path) -> None:
        fix = {
            "op": "set",
            "kind": "device",
            "id": "QF1",
            "fields": {
                "citations": [
                    {"document": "spec-1", "state": "released"},
                    {"document": "spec-1", "state": "draft"},
                ]
            },
            "was": {
                "citations": {
                    "authored": [{"document": "spec-1"}],
                    "mml": [{"document": "dwg-2"}, {"document": "spec-1"}],
                }
            },
            "why": "Both states.",
        }
        errors = _combine(tmp_path, {**_LAYERED, "fixes.yaml": _fixes(fix)}).errors
        # The refused fix resolves nothing, so the layers still conflict.
        assert sorted(e.kind for e in errors) == ["layer-conflict", "source-invalid"]
        (error,) = [e for e in errors if e.kind == "source-invalid"]
        assert error.record_id == "fixes.yaml.fixes.0.fields.citations"
        assert error.sources == ("fixes.yaml",)
        assert error.detail == (
            "device QF1 `citations` states two contents for key "
            "document spec-1, no version, no locator"
        )

    def test_an_add_with_two_contents_under_one_key_is_refused(self, tmp_path: Path) -> None:
        fix = {
            "op": "add",
            "kind": "device",
            "id": "QF9",
            "record": {
                "class": "Quadrupole",
                "scheme_names": [
                    {"name": "QF9", "scheme": "lattice"},
                    {"name": "QF9", "scheme": "lattice", "rule": "r"},
                ],
            },
            "why": "Missing from the export.",
        }
        files = {**_LAYERED, "records/devices.yaml": [{"id": "QF1", "class": "Quadrupole"}]}
        error = _one(_combine(tmp_path, {**files, "fixes.yaml": _fixes(fix)}).errors)
        assert error.kind == "source-invalid"
        assert error.record_id == "fixes.yaml.fixes.0.record.scheme_names"
        assert error.detail == (
            "device QF9 `scheme_names` states two contents for key name QF9, scheme lattice"
        )


def _build_error(tmp_path: Path, files: dict[str, Any]) -> FacilityBuildError:
    """The stop a build reports; a crash fails the test instead."""
    with pytest.raises(FacilityBuildError) as raised:
        build_facility(_write(tmp_path, files), project_name="p")
    return raised.value


class TestTotality:
    """A keyed set whose maps carry non-string keys is a build error, never a crash."""

    def _device_yaml(self, citations: str) -> dict[str, Any]:
        return {
            "documents.yaml": _DOCS,
            "records/devices.yaml": ("- id: QF1\n  class: Quadrupole\n  citations:\n" + citations),
        }

    def test_a_map_with_a_non_string_key_loads_and_combines(self, tmp_path: Path) -> None:
        files = self._device_yaml(
            "  - {document: spec-1, 1: x}\n  - {document: spec-1}\n  - {document: dwg-2}\n"
        )
        result = _combine(tmp_path, files)
        (device,) = result.document["devices"]
        assert {"document": "spec-1", 1: "x"} in device["citations"]

    def test_a_top_level_mixed_key_map_is_a_build_error(self, tmp_path: Path) -> None:
        files = self._device_yaml("  - {document: spec-1, 1: x}\n  - {document: dwg-2}\n")
        assert _build_error(tmp_path, files).kind == "source-invalid"

    def test_a_nested_mixed_key_map_is_a_build_error(self, tmp_path: Path) -> None:
        files = self._device_yaml(
            "  - {document: spec-1, locator: {1: a, b: c}}\n"
            "  - {document: spec-1, locator: {1: a, b: c}, state: x}\n"
            "  - {document: dwg-2}\n"
        )
        errors = _load_errors(tmp_path, files)
        assert [e.kind for e in errors] == ["source-invalid"]
        assert _build_error(tmp_path / "b", files).kind == "source-invalid"


class TestNullKeyFields:
    def test_an_explicit_null_and_an_absent_field_are_one_entry(self, tmp_path: Path) -> None:
        citations = [{"document": "spec-1", "version": None}, {"document": "spec-1"}]
        files = {
            "imported/mml/devices.yaml": [
                {"id": "QF1", "class": "Quadrupole", "citations": citations}
            ],
            "documents.yaml": _DOCS,
        }
        assert _load_errors(tmp_path / "a", files) == []
        device = _device(_combine(tmp_path / "b", files), "QF1")
        assert device["citations"] == [{"document": "spec-1"}]

    def test_null_and_absent_build_identical_bytes(self, tmp_path: Path) -> None:
        first = _tree([], [{"document": "spec-1", "version": None}, _CITATIONS[2]])
        second = _tree([], [_CITATIONS[2], {"document": "spec-1"}])
        assert facility_bytes(
            build_facility(_write(tmp_path / "a", first), project_name="p")
        ) == facility_bytes(build_facility(_write(tmp_path / "b", second), project_name="p"))


_DATED = (
    "- id: QF1\n"
    "  class: Quadrupole\n"
    "  citations:\n"
    "  - {document: spec-1, imported: 2026-03-04}\n"
    "  - {document: spec-1, imported: '2026-03-04'}\n"
    "  scheme_names:\n"
    "  - {name: QF1, scheme: lattice, source: {document: spec-1, imported: 2026-03-04}}\n"
)


class TestImportedDates:
    def test_a_date_and_the_equal_string_are_one_citation(self, tmp_path: Path) -> None:
        files = {"documents.yaml": _DOCS, "imported/mml/devices.yaml": _DATED}
        assert _load_errors(tmp_path / "a", files) == []
        device = _device(_combine(tmp_path / "b", files), "QF1")
        assert device["citations"] == [{"document": "spec-1", "imported": "2026-03-04"}]
        assert device["scheme_names"][0]["source"] == {
            "document": "spec-1",
            "imported": "2026-03-04",
        }

    def test_a_place_citation_date_is_text(self, tmp_path: Path) -> None:
        files = {
            "documents.yaml": _DOCS,
            "records/places.yaml": (
                "- id: SR\n  citations:\n  - {document: spec-1, imported: 2026-03-04}\n"
            ),
        }
        (place,) = _combine(tmp_path, files).document["places"]
        assert place["citations"] == [{"document": "spec-1", "imported": "2026-03-04"}]

    def test_an_unquoted_date_builds_facility_bytes(self, tmp_path: Path) -> None:
        dated = build_facility(
            _write(tmp_path / "a", {"documents.yaml": _DOCS, "records/devices.yaml": _DATED}),
            project_name="p",
        )
        quoted = build_facility(
            _write(
                tmp_path / "b",
                {
                    "documents.yaml": _DOCS,
                    "records/devices.yaml": _DATED.replace(
                        "imported: 2026-03-04", "imported: '2026-03-04'"
                    ),
                },
            ),
            project_name="p",
        )
        assert b"2026-03-04" in facility_bytes(dated)
        assert facility_bytes(dated) == facility_bytes(quoted)

    def test_a_set_fix_brings_dates_in_as_text(self, tmp_path: Path) -> None:
        fixes = (
            f"schema: {FIXES_HEADER}\n"
            "fixes:\n"
            "- op: set\n"
            "  kind: device\n"
            "  id: QF1\n"
            "  fields:\n"
            "    citations:\n"
            "    - {document: spec-1, imported: 2026-03-04}\n"
            "    - {document: spec-1, imported: '2026-03-04'}\n"
            "    - {document: dwg-2}\n"
            "    scheme_names:\n"
            "    - {name: QF1, scheme: lattice, source: {document: dwg-2, imported: 2026-01-02}}\n"
            "  was:\n"
            "    citations:\n"
            "      authored: [{document: spec-1}]\n"
            "      mml: [{document: dwg-2}, {document: spec-1}]\n"
            "  why: The import date is known.\n"
        )
        device = _device(_combine(tmp_path, {**_LAYERED, "fixes.yaml": fixes}), "QF1")
        assert device["citations"] == [
            {"document": "dwg-2"},
            {"document": "spec-1", "imported": "2026-03-04"},
        ]
        assert device["scheme_names"][0]["source"]["imported"] == "2026-01-02"

    def test_an_add_fix_brings_dates_in_as_text(self, tmp_path: Path) -> None:
        fixes = (
            f"schema: {FIXES_HEADER}\n"
            "fixes:\n"
            "- op: add\n"
            "  kind: device\n"
            "  id: QF9\n"
            "  record:\n"
            "    class: Quadrupole\n"
            "    citations: [{document: spec-1, imported: 2026-03-04}]\n"
            "  why: Missing from the export.\n"
        )
        files = {
            "documents.yaml": _DOCS,
            "records/devices.yaml": [{"id": "QF1", "class": "Quadrupole"}],
            "fixes.yaml": fixes,
        }
        device = _device(_combine(tmp_path, files), "QF9")
        assert device["citations"] == [{"document": "spec-1", "imported": "2026-03-04"}]
