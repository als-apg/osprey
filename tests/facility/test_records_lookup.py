"""The read model's one name-aware entry: :meth:`FacilityRecords.lookup`."""

from __future__ import annotations

from osprey.facility.records import FacilityRecord, FacilityRecords

DOC = {
    "places": [
        {"id": "SR"},
        {"id": "SR/S01", "names": ["Sector 1"]},
        {"id": "BR/S01"},
        {"id": "HALL/ROW3/RK08", "names": ["RK8"]},
        {"id": "QF05", "names": ["quad bay"]},
    ],
    "devices": [
        {"id": "SR/BPM01", "names": ["BPM01", "shared"]},
        {
            "id": "SR/QF05-PS",
            "scheme_names": [
                {"name": "PS-Q-0301", "scheme": "cable-schedule"},
                {"name": "QF05 supply", "scheme": "drawings", "basis": "inferred"},
                "not a mapping",
                {"scheme": "nameless"},
            ],
        },
        {"id": "SR/AAA", "names": ["shared"]},
        {"id": "OTHER", "names": ["QF05"]},
    ],
    "channels": [{"id": "BPM01", "on": {"device": "SR/BPM01"}}],
    "groups": [{"id": "shared"}],
}


def _place(id_: str) -> FacilityRecord:
    return FacilityRecord("place", id_, next(p for p in DOC["places"] if p["id"] == id_))


def _device(id_: str) -> FacilityRecord:
    return FacilityRecord("device", id_, next(d for d in DOC["devices"] if d["id"] == id_))


RECORDS = FacilityRecords(DOC)


def test_an_id_hit_comes_first() -> None:
    assert RECORDS.lookup("SR/BPM01") == (("id", _device("SR/BPM01")),)
    assert RECORDS.lookup("SR") == (("id", _place("SR")),)


def test_a_names_entry_finds_its_record() -> None:
    assert RECORDS.lookup("Sector 1") == (("name", _place("SR/S01")),)
    assert RECORDS.lookup("BPM01") == (("name", _device("SR/BPM01")),)


def test_a_scheme_name_finds_its_record() -> None:
    assert RECORDS.lookup("PS-Q-0301") == (("name", _device("SR/QF05-PS")),)
    assert RECORDS.lookup("QF05 supply") == (("name", _device("SR/QF05-PS")),)


def test_a_flat_place_name_finds_the_places_ending_in_it() -> None:
    assert RECORDS.lookup("RK08") == (("segment", _place("HALL/ROW3/RK08")),)
    assert RECORDS.lookup("S01") == (
        ("segment", _place("BR/S01")),
        ("segment", _place("SR/S01")),
    )


def test_a_segment_matches_whole_segments_only() -> None:
    assert RECORDS.lookup("01") == ()
    assert RECORDS.lookup("K08") == ()


def test_a_shared_alias_returns_both_records_in_id_order() -> None:
    assert RECORDS.lookup("shared") == (
        ("name", _device("SR/AAA")),
        ("name", _device("SR/BPM01")),
    )


def test_an_exact_id_orders_before_an_equal_name_on_another_record() -> None:
    assert RECORDS.lookup("QF05") == (
        ("id", _place("QF05")),
        ("name", _device("OTHER")),
    )


def test_a_record_appears_once_at_its_first_match() -> None:
    doc = {
        "places": [{"id": "SR", "names": ["SR"]}, {"id": "LTB/SR"}],
        "devices": [{"id": "D", "names": ["D"], "scheme_names": [{"name": "D", "scheme": "x"}]}],
    }
    records = FacilityRecords(doc)
    assert records.lookup("SR") == (
        ("id", FacilityRecord("place", "SR", doc["places"][0])),
        ("segment", FacilityRecord("place", "LTB/SR", doc["places"][1])),
    )
    assert records.lookup("D") == (("id", FacilityRecord("device", "D", doc["devices"][0])),)


def test_places_order_before_devices_within_a_group() -> None:
    doc = {
        "places": [{"id": "Z", "names": ["alias"]}],
        "devices": [{"id": "A", "names": ["alias"]}],
    }
    records = FacilityRecords(doc)
    assert records.lookup("alias") == (
        ("name", FacilityRecord("place", "Z", doc["places"][0])),
        ("name", FacilityRecord("device", "A", doc["devices"][0])),
    )


def test_lookup_covers_places_and_devices_only() -> None:
    assert RECORDS.lookup("nowhere") == ()
    doc = {"channels": [{"id": "C", "names": ["c"]}], "groups": [{"id": "G"}]}
    records = FacilityRecords(doc)
    assert records.lookup("C") == ()
    assert records.lookup("c") == ()
    assert records.lookup("G") == ()


def test_a_path_name_that_is_no_place_id_finds_nothing() -> None:
    assert RECORDS.lookup("SR/S02") == ()


def test_resolve_and_find_still_ignore_names() -> None:
    assert RECORDS.resolve("PS-Q-0301") == ()
    assert RECORDS.resolve("Sector 1") == ()
    assert RECORDS.find("PS-Q-0301") == ()
    assert "PS-Q-0301" not in RECORDS.targets()
    assert RECORDS.describe("RK08") == []


def test_an_empty_document_finds_nothing() -> None:
    assert FacilityRecords({}).lookup("SR") == ()
