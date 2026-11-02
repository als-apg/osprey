"""A facility reference names a facility record in text; a link is its markup."""

from __future__ import annotations

import pytest

from osprey.facility.records import FacilityRecords
from osprey.facility.references import (
    LINK_SCHEME,
    FacilityReference,
    find_facility_links,
    find_facility_references,
    resolved_references,
)


def test_the_bare_form_parses_to_label_and_target() -> None:
    text = "Read [the X position](facility:SR:DIAG:BPM:01:POSITION:X) first."

    assert find_facility_links(text) == [
        FacilityReference(5, 57, "SR:DIAG:BPM:01:POSITION:X", label="the X position")
    ]
    assert text[5:57] == "[the X position](facility:SR:DIAG:BPM:01:POSITION:X)"


def test_an_angle_bracket_link_holds_an_address_with_spaces() -> None:
    text = r"[gauge](<facility:SR VAC GAUGE 01>) and [odd \[label\]](<facility:A(1)\<2\>>)"

    assert [(found.label, found.target) for found in find_facility_links(text)] == [
        ("gauge", "SR VAC GAUGE 01"),
        ("odd [label]", "A(1)<2>"),
    ]


def test_a_link_found_with_records_carries_the_kinds_it_resolves_to() -> None:
    records = FacilityRecords(
        {"channels": [{"id": "TWIN"}, {"id": "CH"}], "devices": [{"id": "TWIN"}]}
    )
    text = "[a](facility:TWIN) [b](facility:CH) [c](facility:GONE)"

    assert [(found.target, found.kinds) for found in find_facility_links(text, records)] == [
        ("TWIN", ("channel", "device")),
        ("CH", ("channel",)),
        ("GONE", ()),
    ]


def test_a_reference_breaking_its_invariants_is_refused() -> None:
    with pytest.raises(ValueError):
        FacilityReference(3, 3, "X")
    with pytest.raises(ValueError):
        FacilityReference(0, 1, "")
    with pytest.raises(ValueError):
        FacilityReference(0, 1, "X", ("device", "channel"))
    with pytest.raises(ValueError):
        FacilityReference(0, 1, "X", ("place",))  # type: ignore[arg-type]


def test_links_in_code_are_not_links() -> None:
    text = (
        "Inline `[a](facility:A)` is code.\n"
        "```\n[b](facility:B)\n```\n"
        "~~~text\n[c](facility:C)\n~~~\n"
        "``[d](facility:D)`` too, but [e](facility:E) is a link.\n"
    )

    assert [found.target for found in find_facility_links(text)] == ["E"]


def test_a_non_facility_link_is_not_parsed() -> None:
    text = (
        "[page](/devices/bpm.md) [web](https://example.org/facility:X) "
        "![image](facility:IMG) [spaced](facility:<SR X>) SR:DIAG:BPM:01:POSITION:X"
    )

    assert find_facility_links(text) == []
    assert LINK_SCHEME == "facility"


def test_resolved_references_carry_each_span_with_its_records() -> None:
    records = FacilityRecords(
        {"channels": [{"id": "TWIN", "unit": "A"}], "devices": [{"id": "TWIN"}]}
    )
    text = "[t](facility:TWIN) [g](facility:GONE)"
    found = find_facility_links(text)

    assert resolved_references(found, records) == [
        {
            "start": 0,
            "end": 18,
            "target": "TWIN",
            "kinds": ["channel", "device"],
            "label": "t",
            "records": [
                {"kind": "channel", "id": "TWIN", "unit": "A"},
                {"kind": "device", "id": "TWIN"},
            ],
        },
        {"start": 19, "end": 37, "target": "GONE", "kinds": [], "label": "g", "records": []},
    ]
    assert [text[entry["start"] : entry["end"]] for entry in resolved_references(found, None)] == [
        "[t](facility:TWIN)",
        "[g](facility:GONE)",
    ]
    assert [entry["records"] for entry in resolved_references(found, None)] == [None, None]


def _records() -> FacilityRecords:
    return FacilityRecords(
        {
            "channels": [
                {"id": "SR:BPM:01:X"},
                {"id": "SR:BPM:01:X:RMS"},
                {"id": "TWIN"},
            ],
            "devices": [{"id": "DEV-7", "name": "BPM02"}, {"id": "TWIN"}],
        }
    )


def test_a_quoted_address_becomes_a_reference_and_the_text_is_unchanged() -> None:
    text = "Orbit jumped on SR:BPM:01:X at noon."

    (found,) = find_facility_references(text, _records())

    assert found == FacilityReference(16, 27, "SR:BPM:01:X", ("channel",))
    assert found.label is None
    assert text[found.start : found.end] == found.target


def test_an_unknown_address_is_no_reference() -> None:
    assert find_facility_references("Checked SR:BPM:99:X today.", _records()) == []


def test_a_longer_known_id_wins_over_its_prefix() -> None:
    (found,) = find_facility_references("Saw SR:BPM:01:X:RMS rise.", _records())

    assert found.target == "SR:BPM:01:X:RMS"


def test_a_known_id_inside_a_longer_word_is_no_reference() -> None:
    text = "xTWIN TWINy TWIN_2 2TWIN"

    assert find_facility_references(text, _records()) == []


def test_a_device_id_is_a_reference_to_the_device() -> None:
    (found,) = find_facility_references("Swapped DEV-7.", _records())

    assert (found.target, found.kinds) == ("DEV-7", ("device",))


def test_a_device_name_is_no_reference() -> None:
    assert find_facility_references("BPM02 was swapped.", _records()) == []


def test_an_id_held_as_channel_and_device_lists_channel_then_device() -> None:
    found = find_facility_references("TWIN, then SR:BPM:01:X", _records())

    assert found[0].kinds == ("channel", "device")
    assert all(reference.kinds for reference in found)


def test_offsets_count_code_points_after_a_non_ascii_character() -> None:
    text = "\N{ROCKET} SR:BPM:01:X"

    (found,) = find_facility_references(text, _records())

    assert found.start == 2
    assert text[found.start : found.end] == "SR:BPM:01:X"


def test_a_reference_rejects_offsets_that_do_not_span_its_target() -> None:
    with pytest.raises(ValueError):
        FacilityReference(0, 4, "TWINS", ("channel",))

    assert FacilityReference(0, 4, "TWINS", label="link").label == "link"


def test_a_reference_round_trips_through_json() -> None:
    mention = FacilityReference(1, 5, "TWIN", ("channel", "device"))
    link = FacilityReference(0, 20, "TWIN", ("channel",), label="a link")

    assert FacilityReference.from_json(mention.to_json()) == mention
    assert FacilityReference.from_json(link.to_json()) == link
    assert mention.to_json() == {
        "start": 1,
        "end": 5,
        "target": "TWIN",
        "kinds": ["channel", "device"],
    }
    with pytest.raises(TypeError):
        FacilityReference.from_json({"start": 1, "end": 5})
