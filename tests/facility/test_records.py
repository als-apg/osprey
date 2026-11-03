"""The facility records a link target names, looked up by channel id or device id."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import pytest

from osprey.facility.records import (
    LINK_KINDS,
    RECORD_KINDS,
    ChannelSelection,
    FacilityRecord,
    FacilityRecords,
)
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


def test_a_record_kind_outside_the_five_is_refused() -> None:
    with pytest.raises(ValueError, match="wiring"):
        FacilityRecord("wiring", "SR/X", {"id": "SR/X"})  # type: ignore[arg-type]


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


def test_from_file_names_the_type_a_non_object_file_holds(tmp_path: Path) -> None:
    path = tmp_path / "facility.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(FacilityFileError, match="is not a JSON object: it holds a list"):
        FacilityRecords.from_file(path)


# --- lookup by kind and id ---------------------------------------------------------


def _channels_by(doc: dict[str, Any], keep: Callable[[dict[str, Any]], bool]) -> list[str]:
    """The ids of the document's channels ``keep`` accepts, in document order."""
    return [str(channel["id"]) for channel in doc["channels"] if keep(channel)]


def _ids(records: Iterable[Mapping[str, Any]]) -> list[str]:
    return [str(record["id"]) for record in records]


def test_every_record_is_found_by_its_kind_and_id_slot(built_control_assistant: Any) -> None:
    doc = built_control_assistant.facility
    records = FacilityRecords(doc)

    for kind, key, slot in RECORD_KINDS:
        for record in doc[key]:
            assert records.record(kind, str(record[slot])) == record


def test_a_returned_record_is_the_documents_own_and_read_only() -> None:
    doc = {"channels": [{"id": "CH"}], "devices": [{"id": "CH"}]}
    records = FacilityRecords(doc)

    found = records.record("channel", "CH")
    assert found is not None
    doc["channels"][0]["unit"] = "A"
    assert found["unit"] == "A"
    assert found is records.find("CH")[1].record
    assert found is records.resolve("CH")[0].record
    assert found is records.of_kind("channel")[0]
    with pytest.raises(TypeError):
        found["unit"] = "V"  # type: ignore[index]
    with pytest.raises(TypeError):
        records.document["channels"] = []  # type: ignore[index]
    assert records.document == doc


def test_channels_on_a_device_are_its_on_and_endpoint_channels(
    built_control_assistant: Any,
) -> None:
    doc = built_control_assistant.facility
    records = FacilityRecords(doc)

    for device in doc["devices"]:
        device_id = device["id"]
        assert _ids(records.channels_on(device_id)) == _channels_by(
            doc,
            lambda channel, held=device_id: (
                (channel.get("on") or {}).get("device") == held
                or held in (channel.get("endpoint_of") or [])
            ),
        )
    assert records.channels_on("NO/SUCH") == ()


def test_a_channel_on_and_endpoint_of_one_device_is_listed_once() -> None:
    channel = {"id": "CH", "on": {"device": "D"}, "endpoint_of": ["D", "E"]}
    records = FacilityRecords({"channels": [channel]})

    assert _ids(records.channels_on("D")) == ["CH"]
    assert _ids(records.channels_on("E")) == ["CH"]


def test_one_id_held_by_two_kinds_is_found_as_both() -> None:
    records = FacilityRecords(
        {
            "places": [{"id": "BR"}],
            "devices": [{"id": "BR"}, {"id": "TWIN"}],
            "channels": [{"id": "TWIN"}],
        }
    )

    assert [found.kind for found in records.find("BR")] == ["place", "device"]
    assert [found.kind for found in records.resolve("TWIN")] == ["channel", "device"]
    assert [kind for kind, _key, _slot in RECORD_KINDS if kind in LINK_KINDS] == [
        "device",
        "channel",
    ]


def test_resolve_answers_only_link_kinds() -> None:
    records = FacilityRecords(
        {"places": [{"id": "P"}], "groups": [{"id": "G"}], "models": [{"name": "M"}]}
    )

    for target, kind in (("P", "place"), ("G", "group"), ("M", "model")):
        assert records.resolve(target) == ()
        assert target not in records.targets()
        assert [found.kind for found in records.find(target)] == [kind]


def test_an_absent_kind_has_no_records() -> None:
    records = FacilityRecords({"groups": "not a list"})

    for kind, _key, _slot in RECORD_KINDS:
        assert records.of_kind(kind) == ()
        assert records.ids(kind) == frozenset()
        assert records.record(kind, "X") is None
    assert records.find("X") == ()
    assert records.targets() == frozenset()


def test_an_integer_id_is_indexed_as_its_string() -> None:
    records = FacilityRecords({"channels": [{"id": 7}]})

    assert records.ids("channel") == frozenset({"7"})
    assert records.record("channel", "7") == {"id": 7}


def test_a_record_without_its_id_slot_is_skipped() -> None:
    records = FacilityRecords({"channels": [{"unit": "A"}, {"id": "CH"}]})

    assert records.of_kind("channel") == ({"id": "CH"},)


def test_a_duplicate_id_is_won_by_the_first_record_in_document_order() -> None:
    records = FacilityRecords({"channels": [{"id": "CH", "unit": "A"}, {"id": "CH", "unit": "V"}]})

    assert records.record("channel", "CH") == {"id": "CH", "unit": "A"}
    assert records.of_kind("channel") == ({"id": "CH", "unit": "A"},)


def test_require_names_the_kind_and_the_id() -> None:
    records = FacilityRecords({"devices": [{"id": "D"}]})

    assert records.require("device", "D") == {"id": "D"}
    with pytest.raises(KeyError, match="device.*NO/SUCH"):
        records.require("device", "NO/SUCH")


# --- wiring ------------------------------------------------------------------------


def _wiring(doc: Mapping[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Every wiring record of the document with its model's name."""
    return [
        (model["name"], entry) for model in doc["models"] for entry in model.get("wiring") or []
    ]


def test_wired_is_the_wiring_record_naming_an_address(built_control_assistant: Any) -> None:
    doc = built_control_assistant.facility
    records = FacilityRecords(doc)

    wiring = _wiring(doc)
    assert wiring
    for _model, entry in wiring:
        assert records.wired(entry["address"]) is entry


def test_an_address_no_model_wires_has_no_wiring_record(built_control_assistant: Any) -> None:
    doc = built_control_assistant.facility
    records = FacilityRecords(doc)

    addresses = {entry["address"] for _model, entry in _wiring(doc)}
    unwired = _channels_by(doc, lambda channel: channel["id"] not in addresses)
    assert unwired
    for address in unwired:
        assert records.wired(address) is None
    assert records.wired("NOT-A-CHANNEL") is None


def test_a_model_without_a_wiring_list_has_no_wiring_records() -> None:
    records = FacilityRecords({"models": [{"name": "A"}, {"name": "B", "wiring": None}]})

    assert records.wired("X") is None


# --- selection ---------------------------------------------------------------------


@pytest.mark.parametrize("facets", [{}, {"signal": "x"}, {"group": "G"}])
def test_a_selection_names_a_model_or_a_place(facets: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="model or a place"):
        ChannelSelection(**facets)


def test_a_selection_value_must_be_a_string() -> None:
    with pytest.raises(ValueError, match="strings"):
        ChannelSelection(model=1)  # type: ignore[arg-type]


def test_a_model_selection_is_the_models_wired_channels(built_control_assistant: Any) -> None:
    doc = built_control_assistant.facility
    records = FacilityRecords(doc)
    device_model = {device["id"]: device.get("model") for device in doc["devices"]}

    assert doc["models"]
    for model in doc["models"]:
        name = model["name"]
        wired = {entry["address"] for entry in model.get("wiring") or []}
        assert _ids(records.select_channels(ChannelSelection(model=name))) == _channels_by(
            doc,
            lambda channel, name=name, wired=wired: (
                channel["id"] in wired
                or device_model.get((channel.get("on") or {}).get("device")) == name
            ),
        )


def test_a_place_selection_takes_the_place_and_every_place_under_it(
    built_control_assistant: Any,
) -> None:
    doc = built_control_assistant.facility
    records = FacilityRecords(doc)
    device_place = {device["id"]: device.get("place") for device in doc["devices"]}
    parents = [
        place["id"]
        for place in doc["places"]
        if any(other["id"].startswith(f"{place['id']}/") for other in doc["places"])
    ]

    def place_of(channel: dict[str, Any]) -> str | None:
        on = channel.get("on") or {}
        return device_place.get(on.get("device")) or on.get("place")

    assert parents, "the demo facility has no place with a child place"
    for parent in parents:
        expected = _channels_by(
            doc,
            lambda channel, parent=parent: (
                place_of(channel) == parent or (place_of(channel) or "").startswith(f"{parent}/")
            ),
        )
        assert _ids(records.select_channels(ChannelSelection(place=parent))) == expected


def test_a_place_id_sharing_a_prefix_is_not_under_it() -> None:
    records = FacilityRecords(
        {
            "places": [{"id": "A"}, {"id": "A/B"}, {"id": "AB"}],
            "channels": [
                {"id": "IN", "on": {"place": "A"}},
                {"id": "UNDER", "on": {"place": "A/B"}},
                {"id": "OUT", "on": {"place": "AB"}},
            ],
        }
    )

    assert _ids(records.select_channels(ChannelSelection(place="A"))) == ["IN", "UNDER"]


def test_group_and_signal_facets_and_with_the_scope() -> None:
    records = FacilityRecords(
        {
            "places": [{"id": "P"}, {"id": "Q"}],
            "models": [{"name": "M", "wiring": [{"address": "A"}, {"address": "B"}]}],
            "groups": [{"id": "G", "members": ["D1"]}],
            "devices": [{"id": "D1", "place": "P"}, {"id": "D2", "place": "Q"}],
            "channels": [
                {"id": "A", "on": {"device": "D1"}, "signal": "x"},
                {"id": "B", "on": {"device": "D2"}, "signal": "y"},
                {"id": "C", "on": {"device": "D1"}, "signal": "y"},
            ],
        }
    )

    assert _ids(records.select_channels(ChannelSelection(model="M", group="G"))) == ["A"]
    assert _ids(records.select_channels(ChannelSelection(place="P", signal="y"))) == ["C"]
    assert records.select_channels(ChannelSelection(place="Q", signal="x")) == ()


@pytest.mark.parametrize(
    "selection",
    [
        ChannelSelection(model="GONE"),
        ChannelSelection(place="GONE"),
        ChannelSelection(place="P", group="GONE"),
    ],
)
def test_an_unknown_model_place_or_group_is_refused_by_name(selection: ChannelSelection) -> None:
    records = FacilityRecords({"places": [{"id": "P"}]})

    with pytest.raises(ValueError, match="GONE"):
        records.select_channels(selection)


# --- wiring facets -----------------------------------------------------------------


def test_engine_role_and_plane_are_read_from_the_wiring_records(
    built_control_assistant: Any,
) -> None:
    doc = built_control_assistant.facility
    records = FacilityRecords(doc)
    wiring = _wiring(doc)

    sr_monitors = {
        entry["address"]
        for model, entry in wiring
        if model == "SR" and entry["engine_role"] == "monitor"
    }
    assert sr_monitors
    assert _ids(
        records.select_channels(ChannelSelection(model="SR", engine_role="monitor"))
    ) == _channels_by(doc, lambda channel: channel["id"] in sr_monitors)

    sr_x = {
        entry["address"] for model, entry in wiring if model == "SR" and entry.get("plane") == "x"
    }
    assert sr_x
    assert _ids(records.select_channels(ChannelSelection(model="SR", plane="x"))) == _channels_by(
        doc, lambda channel: channel["id"] in sr_x
    )

    monitors = {entry["address"] for _model, entry in wiring if entry["engine_role"] == "monitor"}
    seen = 0
    for place in doc["places"]:
        here = _ids(records.select_channels(ChannelSelection(place=place["id"])))
        expected = [address for address in here if address in monitors]
        seen += len(expected)
        assert (
            _ids(
                records.select_channels(ChannelSelection(place=place["id"], engine_role="monitor"))
            )
            == expected
        )
    assert seen


WIRED_DOC = {
    "places": [{"id": "P"}],
    "devices": [{"id": "D", "place": "P", "model": "M"}],
    "channels": [
        {"id": "MON", "on": {"device": "D"}, "role": "setpoint"},
        {"id": "OTHER", "on": {"device": "D"}},
        {"id": "PLAIN", "on": {"device": "D"}, "role": "monitor", "plane": "x"},
    ],
    "models": [
        {"name": "M", "wiring": [{"address": "MON", "engine_role": "monitor", "plane": "x"}]},
        {"name": "N", "wiring": [{"address": "OTHER", "engine_role": "monitor", "plane": "y"}]},
    ],
}


def test_a_channels_own_role_slot_is_not_the_engine_role() -> None:
    records = FacilityRecords(WIRED_DOC)

    assert _ids(records.select_channels(ChannelSelection(place="P", engine_role="monitor"))) == [
        "MON",
        "OTHER",
    ]
    assert records.select_channels(ChannelSelection(place="P", engine_role="setpoint")) == ()


def test_an_unwired_address_matches_no_engine_role_or_plane() -> None:
    records = FacilityRecords(WIRED_DOC)

    assert "PLAIN" not in _ids(records.select_channels(ChannelSelection(place="P", plane="x")))
    assert "PLAIN" in _ids(records.select_channels(ChannelSelection(place="P")))


def test_with_a_model_named_only_that_models_record_matches() -> None:
    records = FacilityRecords(WIRED_DOC)

    assert _ids(records.select_channels(ChannelSelection(model="M", engine_role="monitor"))) == [
        "MON"
    ]
    assert _ids(records.select_channels(ChannelSelection(model="M", plane="y"))) == []
    assert _ids(records.select_channels(ChannelSelection(place="P", plane="y"))) == ["OTHER"]


def test_engine_role_and_plane_vocabulary_is_the_schemas() -> None:
    from osprey.facility.schema import core

    for member in core.EngineRoleEnum:
        assert ChannelSelection(place="P", engine_role=member.value).engine_role == member.value
    for member in core.PlaneEnum:
        assert ChannelSelection(place="P", plane=member.value).plane == member.value


@pytest.mark.parametrize("facets", [{"engine_role": "SHOUTED"}, {"plane": "SHOUTED"}])
def test_an_engine_role_or_plane_outside_the_vocabulary_is_refused_by_name(
    facets: dict[str, str],
) -> None:
    with pytest.raises(ValueError, match="SHOUTED"):
        ChannelSelection(place="P", **facets)


def test_an_engine_role_alone_names_no_scope() -> None:
    with pytest.raises(ValueError, match="model or a place"):
        ChannelSelection(engine_role="monitor")
