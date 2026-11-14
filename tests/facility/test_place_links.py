"""A ``facility:place/<id>`` link names a place; every other link behaves as before."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from osprey.facility.knowledge_links import DanglingLink, check_knowledge_links, dangling_links
from osprey.facility.records import LINK_KINDS, FacilityRecord, FacilityRecords
from osprey.facility.references import (
    FacilityReference,
    find_facility_links,
    find_facility_references,
    resolved_references,
)
from osprey.facility.validate import Validated

DOC: dict[str, Any] = {
    "places": [
        {"id": "HALL", "description": "The experimental hall."},
        {"id": "HALL/ROW3"},
        {"id": "HALL/ROW3/RK08", "description": "Rack 8."},
    ],
    "devices": [{"id": "SR/BPM01"}, {"id": "place/HALL"}],
    "channels": [{"id": "SR:BPM01:X", "unit": "mm"}],
}


def _page(facility_dir: Path, relative: str, body: str) -> None:
    path = facility_dir / "knowledge" / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\ntype: place\n---\n\n{body}", encoding="utf-8")


def test_a_place_link_resolves_to_the_place() -> None:
    records = FacilityRecords(DOC)

    assert records.resolve_link("place/HALL/ROW3/RK08") == (
        FacilityRecord("place", "HALL/ROW3/RK08", records.require("place", "HALL/ROW3/RK08")),
    )
    assert records.resolve_link("place/HALL/ROW3") == (
        FacilityRecord("place", "HALL/ROW3", records.require("place", "HALL/ROW3")),
    )


def test_a_plain_link_resolves_as_resolve_does() -> None:
    records = FacilityRecords(DOC)

    for target in ("SR/BPM01", "SR:BPM01:X", "HALL", "GONE", "place/GONE", "place/"):
        assert records.resolve_link(target) == records.resolve(target)
    assert records.resolve_link("HALL") == ()


def test_a_device_literally_named_place_x_keeps_its_link() -> None:
    records = FacilityRecords(DOC)

    assert [(found.kind, found.id) for found in records.resolve_link("place/HALL")] == [
        ("device", "place/HALL")
    ]
    assert find_facility_links("[d](facility:place/HALL)", records)[0].kinds == ("device",)


def test_targets_and_describe_are_unchanged_by_places() -> None:
    without = FacilityRecords({key: DOC[key] for key in ("devices", "channels")})
    with_places = FacilityRecords(DOC)

    assert with_places.targets() == without.targets()
    assert not any(target.startswith("place/HALL/") for target in with_places.targets())
    assert with_places.describe("place/HALL/ROW3/RK08") == []
    assert with_places.resolve("place/HALL/ROW3/RK08") == ()
    assert LINK_KINDS == ("channel", "device")


def test_a_place_link_carries_the_place_kind() -> None:
    records = FacilityRecords(DOC)
    text = (
        "[rack](facility:place/HALL/ROW3/RK08) [gone](facility:place/NOPE) [x](facility:SR:BPM01:X)"
    )

    assert [(found.target, found.kinds) for found in find_facility_links(text, records)] == [
        ("place/HALL/ROW3/RK08", ("place",)),
        ("place/NOPE", ()),
        ("SR:BPM01:X", ("channel",)),
    ]


def test_a_resolved_place_link_keeps_its_target_text_and_lists_the_place() -> None:
    records = FacilityRecords(DOC)
    found = find_facility_links("[rack](facility:place/HALL/ROW3/RK08)", records)

    assert resolved_references(found, records) == [
        {
            "start": 0,
            "end": 37,
            "target": "place/HALL/ROW3/RK08",
            "kinds": ["place"],
            "label": "rack",
            "records": [{"kind": "place", "id": "HALL/ROW3/RK08", "description": "Rack 8."}],
        }
    ]


def test_the_place_kind_belongs_to_the_place_form_only() -> None:
    assert FacilityReference(0, 5, "place/X", ("place",), label="x").kinds == ("place",)
    assert FacilityReference.from_json(
        {"start": 0, "end": 5, "target": "place/X", "kinds": ["place"], "label": "x"}
    ).kinds == ("place",)
    with pytest.raises(ValueError):
        FacilityReference(0, 1, "X", ("place",), label="x")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        FacilityReference(0, 5, "place/X", ("place", "device"), label="x")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        FacilityReference(0, 5, "place/X", ("device", "place"), label="x")  # type: ignore[arg-type]


def test_logbook_prose_never_quotes_a_place_link() -> None:
    records = FacilityRecords(DOC)

    assert [
        (found.target, found.kinds)
        for found in find_facility_references("see place/HALL/ROW3/RK08 now", records)
    ] == [("place/HALL", ("device",))]


def test_a_place_link_to_a_held_place_does_not_dangle(tmp_path: Path) -> None:
    _page(tmp_path, "places/rack.md", "In [rack 8](facility:place/HALL/ROW3/RK08).\n")

    assert dangling_links(tmp_path, DOC) == []


def test_a_place_link_to_a_missing_place_dangles_with_the_place_remedy(tmp_path: Path) -> None:
    _page(tmp_path, "places/rack.md", "In [rack 9](facility:place/HALL/ROW3/RK09).\n")
    validated = Validated(tmp_path, "demo")
    validated.document = {key: [dict(record) for record in DOC[key]] for key in DOC}

    assert dangling_links(tmp_path, DOC) == [
        DanglingLink("knowledge/places/rack.md", "place/HALL/ROW3/RK09", "body")
    ]
    assert check_knowledge_links(validated) == []
    assert [warning.line for warning in validated.warnings] == [
        "facility: link-dangling: page knowledge/places/rack.md — links "
        "facility:place/HALL/ROW3/RK09, which the facility file does not hold; fix: link a "
        "place id of the facility file as facility:place/<id>, or remove the link"
    ]


def test_a_plain_dangling_link_keeps_its_remedy(tmp_path: Path) -> None:
    _page(tmp_path, "places/rack.md", "In [hall](facility:HALL).\n")
    validated = Validated(tmp_path, "demo")
    validated.document = {key: [dict(record) for record in DOC[key]] for key in DOC}

    assert check_knowledge_links(validated) == []
    assert [warning.remedy for warning in validated.warnings] == [
        "link a channel id or a device id of the facility file, or remove the link"
    ]
