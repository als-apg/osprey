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
