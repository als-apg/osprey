"""The facility records a link target names, looked up by channel id or device id."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osprey.facility.records import FacilityRecord, FacilityRecords
from osprey.utils.facility import FacilityFileError

DOC = {
    "channels": [
        {"id": "SR:DIAG:BPM:01:POSITION:X", "unit": "mm", "description": "Beam X position."},
        {"id": "TWIN", "unit": "A"},
    ],
    "devices": [
        {"id": "SR/BPM01", "names": ["BPM01"], "label": "BPM 01", "description": "A monitor."},
        {"id": "TWIN"},
    ],
    "places": [{"id": "SR"}],
}


def test_a_channel_id_and_a_device_id_resolve_to_their_records() -> None:
    records = FacilityRecords(DOC)

    assert records.resolve("SR:DIAG:BPM:01:POSITION:X") == (
        FacilityRecord("channel", "SR:DIAG:BPM:01:POSITION:X", DOC["channels"][0]),
    )
    assert records.resolve("SR/BPM01") == (FacilityRecord("device", "SR/BPM01", DOC["devices"][0]),)


def test_a_name_or_label_does_not_resolve() -> None:
    records = FacilityRecords(DOC)

    for target in ("BPM01", "BPM 01", "SR", "SR:DIAG:BPM:01:POSITION:Y", ""):
        assert records.resolve(target) == (), target
    assert records.targets() == frozenset({"SR:DIAG:BPM:01:POSITION:X", "SR/BPM01", "TWIN"})


def test_an_id_held_as_channel_and_device_resolves_to_both() -> None:
    records = FacilityRecords(DOC)

    assert [(record.kind, record.id) for record in records.resolve("TWIN")] == [
        ("channel", "TWIN"),
        ("device", "TWIN"),
    ]


def test_describe_lists_every_record_of_a_target() -> None:
    records = FacilityRecords(DOC)

    assert records.describe("TWIN") == [
        {"kind": "channel", "id": "TWIN", "unit": "A"},
        {"kind": "device", "id": "TWIN"},
    ]
    assert records.describe("SR/BPM01") == [
        {"kind": "device", "id": "SR/BPM01", "description": "A monitor."}
    ]
    assert records.describe("NO/SUCH") == []


def test_a_record_kind_outside_channel_and_device_is_refused() -> None:
    with pytest.raises(ValueError, match="place"):
        FacilityRecord("place", "SR", {"id": "SR"})  # type: ignore[arg-type]


def test_from_file_reads_the_facility_file(tmp_path: Path) -> None:
    path = tmp_path / "facility.json"
    path.write_text(json.dumps(DOC), encoding="utf-8")

    records = FacilityRecords.from_file(path)

    assert records is not None
    assert records.targets() == FacilityRecords(DOC).targets()


def test_from_file_without_a_file_is_none(tmp_path: Path) -> None:
    assert FacilityRecords.from_file(tmp_path / "facility.json") is None


@pytest.mark.parametrize("content", [b"{not json", b"\xff\xfe", b"[]"])
def test_from_file_on_an_unreadable_file_raises(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "facility.json"
    path.write_bytes(content)

    with pytest.raises(FacilityFileError, match=str(path)):
        FacilityRecords.from_file(path)
