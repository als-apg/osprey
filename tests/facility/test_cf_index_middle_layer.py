"""The middle-layer channel-finder index: the facility's groups as System -> Family -> Field.

``data/channel_finder/middle_layer.json`` carries
``"schema": "osprey.facility.channel_finder/1"``, which the middle-layer loader
skips, and one System per top place. A Family is a group, named by its id
less a leading ``<System>/``, or the devices of a class no single-class group
holds under the System, named by the class. A Field is a signal, listing one
channel per member, in ``CommonNames`` order, so ``ChannelNames`` aligns with
``DeviceList``. ``_setup`` is derived from the members: ``CommonNames`` from
each label, ``DeviceList`` from each member's place among its sibling places,
``ElementList`` from the member order. The DuckDB copy ``run_sql`` queries is
written beside it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from tests.facility.conftest import BuiltProject

GOLDEN = Path(__file__).resolve().parent / "golden" / "cf_index_pre_line" / "middle_layer.json"


def _inputs(
    doc: dict[str, Any],
    rendered_config: dict[str, Any] | None = None,
    reported: set[str] | None = None,
) -> Any:
    from osprey.facility.views import ViewInputs

    return ViewInputs(
        doc=doc,
        rendered_config=rendered_config or {},
        facility_dir=Path("."),
        served=[],
        reported=reported,
    )


def _document(doc: dict[str, Any]) -> tuple[dict[str, Any], int, int]:
    from osprey.facility.views.channel_finder import middle_layer_document

    return middle_layer_document(doc)


def _fields(family: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {key: value for key, value in family.items() if not key.startswith("_")}


#: Two machines; one group spans both, one lives on the first only.
SYNTHETIC: dict[str, Any] = {
    "places": [
        {"id": "M", "level": "machine", "description": "the machine"},
        {"id": "M/S1", "level": "sector"},
        {"id": "N", "level": "machine", "description": "the other machine"},
    ],
    "devices": [
        {
            "id": "M/Q2",
            "class": "Quadrupole",
            "place": "M/S1",
            "s": 2.0,
            "label": "Quad 2",
        },
        {
            "id": "M/Q1",
            "class": "Quadrupole",
            "place": "M/S1",
            "s": 1.0,
            "label": "Quad 1",
        },
        {
            "id": "N/Q1",
            "class": "Quadrupole",
            "place": "N",
            "label": "Quad N1",
        },
        {"id": "M/G1", "class": "Gauge", "place": "M"},
    ],
    "groups": [
        {
            "id": "M/QUAD",
            "description": "the quadrupoles",
            "members": ["M/Q1", "M/Q2", "N/Q1"],
        },
        {"id": "M/ALL", "description": "everything", "members": ["M/Q1", "M/Q2", "M/G1"]},
    ],
    "channels": [
        {"id": "M:Q1:CURRENT:SP", "on": {"device": "M/Q1"}, "signal": "current_setpoint"},
        {"id": "M:Q2:CURRENT:SP", "on": {"device": "M/Q2"}, "signal": "current_setpoint"},
        {"id": "N:Q1:CURRENT:SP", "on": {"device": "N/Q1"}, "signal": "current_setpoint"},
        {"id": "M:Q1:TEMP", "on": {"device": "M/Q1"}, "signal": "temperature"},
        {"id": "M:Q2:TEMP", "on": {"device": "M/Q2"}, "signal": "temperature"},
        {"id": "M:G1:P", "on": {"device": "M/G1"}, "signal": "pressure"},
        {"id": "M:TUNE", "on": {"place": "M"}},
    ],
}


def test_a_family_is_filed_under_each_system_of_its_members() -> None:
    document, _left_out, _by_address = _document(SYNTHETIC)

    assert sorted(document) == ["M", "N", "schema"]
    assert sorted(document["M"]) == ["ALL", "Gauge", "QUAD", "_description"]
    assert sorted(document["N"]) == ["M/QUAD", "_description"]
    assert _fields(document["N"]["M/QUAD"])["current_setpoint"]["ChannelNames"] == [
        "N:Q1:CURRENT:SP"
    ]


def test_each_class_lists_the_families_the_index_files_its_members_under() -> None:
    from osprey.facility.views.channel_finder import middle_layer_families

    document, _left_out, _by_address = _document(SYNTHETIC)

    assert middle_layer_families(SYNTHETIC) == {
        "Gauge": [("M", "ALL"), ("M", "Gauge")],
        "Quadrupole": [("M", "ALL"), ("M", "QUAD"), ("N", "M/QUAD")],
    }
    for pairs in middle_layer_families(SYNTHETIC).values():
        assert all(family in document[system] for system, family in pairs)


def test_a_system_is_described_by_its_top_place() -> None:
    document, _left_out, _by_address = _document(SYNTHETIC)

    assert document["M"]["_description"] == "the machine"
    assert document["N"]["_description"] == "the other machine"
    assert document["M"]["QUAD"]["_description"] == "the quadrupoles"


def test_a_field_lists_one_channel_per_member_in_common_name_order() -> None:
    document, _left_out, _by_address = _document(SYNTHETIC)
    family = document["M"]["QUAD"]

    assert family["_setup"] == {
        "CommonNames": ["Quad 1", "Quad 2"],
        "DeviceList": [[1, 1], [2, 1]],
        "ElementList": [1, 2],
    }
    assert _fields(family)["current_setpoint"]["ChannelNames"] == [
        "M:Q1:CURRENT:SP",
        "M:Q2:CURRENT:SP",
    ]
    assert _fields(family)["temperature"]["ChannelNames"] == ["M:Q1:TEMP", "M:Q2:TEMP"]


def test_setup_always_lists_all_three() -> None:
    document, _left_out, _by_address = _document(SYNTHETIC)

    assert document["N"]["M/QUAD"]["_setup"] == {
        "CommonNames": ["Quad N1"],
        "DeviceList": [[1, 1]],
        "ElementList": [1],
    }


def test_a_common_name_is_the_label_else_the_id() -> None:
    doc = {
        "places": [{"id": "M", "level": "machine"}],
        "devices": [
            {"id": "M/A", "place": "M", "s": 1.0, "label": "b", "names": ["a", "z"]},
            {"id": "M/C", "place": "M", "s": 2.0, "names": ["c"]},
        ],
        "groups": [{"id": "M/F", "members": ["M/A", "M/C"]}],
        "channels": [],
    }

    document, _left_out, _by_address = _document(doc)

    assert document["M"]["F"]["_setup"]["CommonNames"] == ["b", "M/C"]


def _sectors(*sectors: tuple[str, list[tuple[str, float | None]]]) -> dict[str, Any]:
    """A machine ``M`` whose sectors hold quadrupoles at the given ``s``, all in group ``M/Q``."""
    places = [{"id": "M", "level": "machine"}]
    devices = []
    for sector, members in sectors:
        places.append({"id": f"M/{sector}", "level": "sector"})
        for device_id, position in members:
            device: dict[str, Any] = {
                "id": device_id,
                "class": "Quadrupole",
                "place": f"M/{sector}",
            }
            if position is not None:
                device["s"] = position
            devices.append(device)
    return {
        "places": places,
        "devices": devices,
        "groups": [{"id": "M/Q", "members": [device["id"] for device in devices]}],
        "channels": [],
    }


def test_sibling_sectors_are_ordered_by_their_devices_s_not_by_id() -> None:
    doc = _sectors(("S2", [("M/B", 20.0)]), ("S10", [("M/A", 5.0)]))

    document, _left_out, _by_address = _document(doc)

    assert document["M"]["Q"]["_setup"] == {
        "CommonNames": ["M/A", "M/B"],
        "DeviceList": [[1, 1], [2, 1]],
        "ElementList": [1, 2],
    }


def test_sibling_sectors_with_no_positioned_device_follow_in_natural_id_order() -> None:
    doc = _sectors(("S10", [("M/C", None)]), ("S2", [("M/B", None)]), ("S3", [("M/A", 1.0)]))

    document, _left_out, _by_address = _document(doc)

    setup = document["M"]["Q"]["_setup"]
    assert setup["CommonNames"] == ["M/A", "M/B", "M/C"]
    assert setup["DeviceList"] == [[1, 1], [2, 1], [3, 1]]


def test_a_member_placed_at_its_system_gets_its_position_and_one() -> None:
    doc = _sectors(("S1", [("M/A", 1.0)]), ("S2", [("M/B", 3.0)]))
    doc["devices"].append({"id": "M/C", "class": "Quadrupole", "place": "M", "s": 2.0})
    doc["groups"][0]["members"].append("M/C")

    document, _left_out, _by_address = _document(doc)

    setup = document["M"]["Q"]["_setup"]
    assert setup["CommonNames"] == ["M/A", "M/C", "M/B"]
    assert setup["DeviceList"] == [[1, 1], [2, 1], [2, 1]]


def test_an_unpositioned_member_takes_its_place_index_and_orders_after_positioned_ones() -> None:
    doc = _sectors(("S1", [("M/A", 1.0), ("M/Z", None)]), ("S2", [("M/B", 3.0)]))

    document, _left_out, _by_address = _document(doc)

    setup = document["M"]["Q"]["_setup"]
    assert setup["CommonNames"] == ["M/A", "M/B", "M/Z"]
    assert setup["DeviceList"] == [[1, 1], [2, 1], [1, 2]]


def test_a_mixed_class_family_gets_distinct_sector_ordinals() -> None:
    doc = _sectors(("S1", [("M/A", 1.0), ("M/B", 2.0)]), ("S2", [("M/C", 3.0)]))
    doc["devices"][1]["class"] = "Sextupole"

    document, _left_out, _by_address = _document(doc)

    setup = document["M"]["Q"]["_setup"]
    assert setup["DeviceList"] == [[1, 1], [1, 2], [2, 1]]


def test_a_system_and_a_family_are_described_only_by_description_or_label() -> None:
    doc = _sectors(("S1", [("M/A", 1.0)]))
    doc["places"][0]["names"] = ["the machine"]
    doc["groups"] = [
        {"id": "M/Q", "names": ["quads"], "members": ["M/A"]},
        {"id": "M/L", "label": "the labelled", "members": ["M/A"]},
        {"id": "M/D", "label": "the labelled", "description": "the described", "members": ["M/A"]},
    ]

    document, _left_out, _by_address = _document(doc)

    assert "_description" not in document["M"]
    assert "_description" not in document["M"]["Q"]
    assert document["M"]["L"]["_description"] == "the labelled"
    assert document["M"]["D"]["_description"] == "the described"


def _signal_text(name: str) -> str:
    from osprey.facility.validate import vocabulary

    (row,) = [row for row in vocabulary()["signal_roles"] if row["name"] == name]
    return str(row["description"])


def _class_text(name: str) -> str:
    from osprey.facility.validate import vocabulary

    (row,) = [row for row in vocabulary()["classes"] if row["name"] == name]
    return str(row["description"])


def test_a_field_is_keyed_by_signal_and_described_by_the_vocabulary() -> None:
    document, _left_out, _by_address = _document(SYNTHETIC)
    field = _fields(document["M"]["QUAD"])["current_setpoint"]

    assert field == {
        "ChannelNames": ["M:Q1:CURRENT:SP", "M:Q2:CURRENT:SP"],
        "_description": _signal_text("current_setpoint"),
    }


def test_an_address_keyed_field_takes_its_channel_s_description() -> None:
    doc = {
        "places": [{"id": "M"}],
        "devices": [{"id": "M/A", "place": "M", "class": "Quadrupole"}],
        "groups": [{"id": "M/F", "members": ["M/A"]}],
        "channels": [{"id": "M:A:TUNE", "on": {"device": "M/A"}, "description": "the tune"}],
    }

    document, _left_out, by_address = _document(doc)

    assert _fields(document["M"]["F"]) == {
        "M:A:TUNE": {"ChannelNames": ["M:A:TUNE"], "_description": "the tune"}
    }
    assert by_address == 1


def test_a_role_without_a_vocabulary_sentence_takes_the_members_common_description(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osprey.facility import validate

    table = validate.vocabulary()
    stripped = {
        **table,
        "signal_roles": [
            {key: value for key, value in row.items() if key != "description"}
            if row["name"] == "current_setpoint"
            else row
            for row in table["signal_roles"]
        ],
    }
    monkeypatch.setattr(validate, "vocabulary", lambda: stripped)

    def described(first: str, second: str) -> dict[str, Any]:
        doc = {
            "places": [{"id": "M"}],
            "devices": [
                {"id": "M/A", "place": "M", "s": 1.0},
                {"id": "M/B", "place": "M", "s": 2.0},
            ],
            "groups": [{"id": "M/F", "members": ["M/A", "M/B"]}],
            "channels": [
                {
                    "id": "A:SP",
                    "on": {"device": "M/A"},
                    "signal": "current_setpoint",
                    "description": first,
                },
                {
                    "id": "B:SP",
                    "on": {"device": "M/B"},
                    "signal": "current_setpoint",
                    "description": second,
                },
            ],
        }
        document, _left_out, _by_address = _document(doc)
        return _fields(document["M"]["F"])["current_setpoint"]

    assert described("the current", "the current")["_description"] == "the current"
    assert "_description" not in described("one", "another")


def test_a_classed_device_no_single_class_group_holds_files_under_its_place_and_class() -> None:
    document, _left_out, _by_address = _document(SYNTHETIC)

    gauge = document["M"]["Gauge"]
    assert gauge["_description"] == _class_text("Gauge")
    assert gauge["_setup"]["CommonNames"] == ["M/G1"]
    assert _fields(gauge)["pressure"]["ChannelNames"] == ["M:G1:P"]
    assert "Quadrupole" not in document["M"]
    assert "Quadrupole" not in document["N"]


def test_a_derived_family_name_colliding_with_a_group_keeps_the_group_s_id() -> None:
    doc = {
        "places": [{"id": "M"}],
        "devices": [
            {"id": "M/G1", "place": "M", "class": "Gauge"},
            {"id": "M/Q1", "place": "M", "class": "Quadrupole"},
        ],
        "groups": [{"id": "M/Gauge", "members": ["M/G1", "M/Q1"]}],
        "channels": [],
    }

    document, _left_out, _by_address = _document(doc)

    assert sorted(key for key in document["M"] if not key.startswith("_")) == [
        "Gauge",
        "M/Gauge",
        "Quadrupole",
    ]
    assert document["M"]["Gauge"]["_setup"]["CommonNames"] == ["M/G1"]
    assert document["M"]["M/Gauge"]["_setup"]["CommonNames"] == ["M/G1", "M/Q1"]


def test_a_channel_in_no_family_is_left_out_and_counted() -> None:
    document, left_out, by_address = _document(SYNTHETIC)
    addresses = {
        address
        for system, families in document.items()
        if system != "schema"
        for family in families.values()
        if isinstance(family, dict)
        for field in _fields(family).values()
        for address in field["ChannelNames"]
    }

    assert "M:G1:P" in addresses
    assert "M:TUNE" not in addresses
    assert (left_out, by_address) == (1, 5)


def test_a_field_a_member_lacks_or_repeats_is_keyed_by_address() -> None:
    doc = {
        "devices": [
            {"id": "A", "place": "M", "label": "a"},
            {"id": "B", "place": "M", "label": "b"},
        ],
        "groups": [{"id": "M/F", "members": ["A", "B"]}],
        "channels": [
            {"id": "A:X", "on": {"device": "A"}, "signal": "position_x_readback"},
            {"id": "B:X", "on": {"device": "B"}, "signal": "position_x_readback"},
            {"id": "A:H:CURRENT", "on": {"device": "A"}, "signal": "current"},
            {"id": "A:V:CURRENT", "on": {"device": "A"}, "signal": "current"},
            {"id": "B:H:CURRENT", "on": {"device": "B"}, "signal": "current"},
            {"id": "A:ONLY", "on": {"device": "A"}, "signal": "only"},
            {"id": "B:BARE", "on": {"device": "B"}},
        ],
    }

    document, left_out, by_address = _document(doc)
    fields = _fields(document["M"]["F"])

    assert fields["position_x_readback"] == {
        "ChannelNames": ["A:X", "B:X"],
        "_description": _signal_text("position_x_readback"),
    }
    keyed = ["A:H:CURRENT", "A:V:CURRENT", "B:H:CURRENT", "A:ONLY", "B:BARE"]
    assert {address: fields[address]["ChannelNames"] for address in keyed} == {
        address: [address] for address in keyed
    }
    assert sorted(fields) == sorted(["position_x_readback", *keyed])
    assert (left_out, by_address) == (0, 5)


def test_a_family_is_keyed_by_signal_else_address() -> None:
    doc = {
        "devices": [
            {"id": "A", "place": "M", "label": "a", "s": 1.0},
            {"id": "B", "place": "M", "label": "b", "s": 2.0},
        ],
        "groups": [{"id": "M/F", "description": "the family", "members": ["A", "B"]}],
        "channels": [
            {"id": "A:X", "on": {"device": "A"}, "signal": "current"},
            {"id": "B:X", "on": {"device": "B"}, "signal": "current"},
            {"id": "A:BARE", "on": {"device": "A"}},
        ],
    }

    document, left_out, by_address = _document(doc)
    family = document["M"]["F"]

    assert family["_description"] == "the family"
    assert _fields(family) == {
        "current": {"ChannelNames": ["A:X", "B:X"]},
        "A:BARE": {"ChannelNames": ["A:BARE"]},
    }
    assert (left_out, by_address) == (0, 1)


def test_an_umbrella_group_is_its_own_family_beside_its_members_groups() -> None:
    document, _left_out, _by_address = _document(SYNTHETIC)

    quad = _fields(document["M"]["QUAD"])
    umbrella = _fields(document["M"]["ALL"])
    assert document["M"]["ALL"]["_description"] == "everything"
    assert quad["current_setpoint"]["ChannelNames"] == ["M:Q1:CURRENT:SP", "M:Q2:CURRENT:SP"]
    in_umbrella = {address for field in umbrella.values() for address in field["ChannelNames"]}
    assert {"M:Q1:CURRENT:SP", "M:Q2:CURRENT:SP", "M:G1:P"} <= in_umbrella
    assert all("_description" not in field for field in umbrella.values())


def test_a_shared_endpoint_is_listed_once_per_device_it_ends() -> None:
    doc = {
        "devices": [
            {"id": "A", "place": "M", "label": "a", "s": 1.0},
            {"id": "B", "place": "M", "label": "b", "s": 2.0},
        ],
        "groups": [{"id": "M/F", "members": ["B", "A"]}],
        "channels": [
            {"id": "BUS:SP", "endpoint_of": ["A", "B"], "signal": "current_setpoint"},
        ],
    }

    document, _left_out, _by_address = _document(doc)

    assert document["M"]["F"]["_setup"]["CommonNames"] == ["a", "b"]
    assert _fields(document["M"]["F"])["current_setpoint"]["ChannelNames"] == [
        "BUS:SP",
        "BUS:SP",
    ]


def test_a_member_with_no_place_sits_under_system_none() -> None:
    doc = {
        "devices": [{"id": "A", "label": "a"}],
        "groups": [{"id": "F", "members": ["A"]}],
        "channels": [{"id": "A:X", "on": {"device": "A"}, "signal": "current_setpoint"}],
    }

    document, _left_out, _by_address = _document(doc)

    assert document["-"]["_description"] == "no place"
    assert _fields(document["-"]["F"])["current_setpoint"]["ChannelNames"] == ["A:X"]


def _one_family(place: str, group: str) -> dict[str, Any]:
    return {
        "places": [{"id": place}],
        "devices": [{"id": "A", "place": place, "label": "a"}],
        "groups": [{"id": group, "members": ["A"]}],
        "channels": [{"id": "A:X", "on": {"device": "A"}, "signal": "current_setpoint"}],
    }


def test_a_family_name_beginning_with_an_underscore_stops_with_view_unsupported() -> None:
    from osprey.facility.errors import FacilityBuildError

    with pytest.raises(FacilityBuildError) as caught:
        _document(_one_family("M", "M/_X"))

    assert caught.value.format_message() == (
        "facility: view-unsupported: group M/_X — its tree key `_X` begins with `_`, and a key "
        "beginning with `_` is a meta key of the middle-layer index; fix: give the group a key "
        "that does not begin with `_`, or select another channel_finder_mode"
    )


def test_a_system_beginning_with_an_underscore_stops_with_view_unsupported() -> None:
    from osprey.facility.errors import FacilityBuildError

    with pytest.raises(FacilityBuildError) as caught:
        _document(_one_family("_M", "F"))

    assert caught.value.format_message() == (
        "facility: view-unsupported: place _M — its tree key `_M` begins with `_`, and a key "
        "beginning with `_` is a meta key of the middle-layer index; fix: give the place a key "
        "that does not begin with `_`, or select another channel_finder_mode"
    )


def test_a_system_named_schema_stops_with_view_unsupported() -> None:
    from osprey.facility.errors import FacilityBuildError

    with pytest.raises(FacilityBuildError) as caught:
        _document(_one_family("schema", "F"))

    assert caught.value.format_message() == (
        "facility: view-unsupported: place schema — its System key `schema` is the document key "
        "of the middle-layer index; fix: give the place an id other than `schema`, or select "
        "another channel_finder_mode"
    )


def test_keys_not_beginning_with_an_underscore_are_unchanged() -> None:
    document, _left_out, _by_address = _document(_one_family("M_", "M_/X_"))

    assert sorted(document) == ["M_", "schema"]
    assert sorted(document["M_"]) == ["X_"]


def test_the_loader_reads_the_index_back_and_skips_its_schema(tmp_path: Path) -> None:
    from osprey.services.channel_finder.databases.middle_layer import MiddleLayerDatabase

    (index, _database) = _write(tmp_path, SYNTHETIC)
    loaded = MiddleLayerDatabase(str(index))

    assert sorted(loaded.channel_map) == [
        "M:G1:P",
        "M:Q1:CURRENT:SP",
        "M:Q1:TEMP",
        "M:Q2:CURRENT:SP",
        "M:Q2:TEMP",
        "N:Q1:CURRENT:SP",
    ]
    assert [system["name"] for system in loaded.list_systems()] == ["M", "N"]
    assert loaded.list_channel_names("M", "QUAD", "current_setpoint", sectors=[1, 2]) == [
        "M:Q1:CURRENT:SP",
        "M:Q2:CURRENT:SP",
    ]


def _write(tmp_path: Path, doc: dict[str, Any], **inputs: Any) -> list[Path]:
    from osprey.facility.views.channel_finder import write_middle_layer

    return write_middle_layer(tmp_path / "channel_finder", _inputs(doc, **inputs))


def test_the_writer_writes_the_index_and_its_duckdb_database(tmp_path: Path) -> None:
    import duckdb

    from osprey.facility.views.channel_finder import (
        CHANNEL_FINDER_SCHEMA,
        MIDDLE_LAYER_DUCKDB_FILE,
        MIDDLE_LAYER_FILE,
    )

    root = tmp_path / "channel_finder"
    written = _write(tmp_path, SYNTHETIC)

    assert written == [root / MIDDLE_LAYER_FILE, root / MIDDLE_LAYER_DUCKDB_FILE]
    raw = written[0].read_bytes()
    assert raw.endswith(b"}\n")
    assert json.loads(raw)["schema"] == CHANNEL_FINDER_SCHEMA
    con = duckdb.connect(str(written[1]), read_only=True)
    try:
        assert con.execute(
            "SELECT count(*), count(DISTINCT channel_name) FROM channels"
        ).fetchone() == (11, 6)
        assert con.execute(
            "SELECT common_name, sector, device FROM device_map "
            "WHERE system = 'M' AND family = 'QUAD' ORDER BY device_index"
        ).fetchall() == [("Quad 1", 1, 1), ("Quad 2", 2, 1)]
        assert con.execute(
            "SELECT common_name FROM device_map WHERE system = 'M' AND family = 'ALL' "
            "ORDER BY device_index"
        ).fetchall() == [("Quad 1",), ("Quad 2",), ("M/G1",)]
    finally:
        con.close()


def test_full_text_search_ranks_a_channel_under_each_of_its_families(tmp_path: Path) -> None:
    import duckdb

    (_index, database) = _write(tmp_path, SYNTHETIC)

    con = duckdb.connect(str(database), read_only=True)
    try:
        rows = con.execute(
            "SELECT channel_name, family, "
            "fts_main_channels.match_bm25(row_id, 'TEMP') AS score "
            "FROM channels WHERE score IS NOT NULL ORDER BY score DESC"
        ).fetchall()
    finally:
        con.close()

    assert sorted((name, family) for name, family, _score in rows) == [
        ("M:Q1:TEMP", "ALL"),
        ("M:Q1:TEMP", "QUAD"),
        ("M:Q2:TEMP", "ALL"),
        ("M:Q2:TEMP", "QUAD"),
    ]


def test_a_rewrite_replaces_the_duckdb_database(tmp_path: Path) -> None:
    import duckdb

    _write(tmp_path, SYNTHETIC)
    smaller = {**SYNTHETIC, "channels": SYNTHETIC["channels"][:3]}
    (_index, database) = _write(tmp_path, smaller)

    con = duckdb.connect(str(database), read_only=True)
    try:
        assert con.execute(
            "SELECT count(*), count(DISTINCT channel_name) FROM channels"
        ).fetchone() == (5, 3)
    finally:
        con.close()


def test_the_counts_are_one_note_per_build(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reported: set[str] = set()
    _write(tmp_path / "a", SYNTHETIC, reported=reported)
    _write(tmp_path / "b", SYNTHETIC, reported=reported)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert (
        captured.err
        == "  view middle_layer: 1 channels in no family left out, 5 keyed by address\n"
    )


def test_a_facility_whose_every_channel_is_filed_prints_no_note(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    doc = {
        "devices": [{"id": "A", "place": "M", "label": "a"}],
        "groups": [{"id": "F", "members": ["A"]}],
        "channels": [{"id": "A:X", "on": {"device": "A"}, "signal": "current_setpoint"}],
    }

    _write(tmp_path, doc)

    assert capsys.readouterr() == ("", "")


def test_no_group_and_no_classed_device_stops_with_view_unsupported(tmp_path: Path) -> None:
    from osprey.facility.errors import FacilityBuildError

    devices = [
        {key: value for key, value in device.items() if key != "class"}
        for device in SYNTHETIC["devices"]
    ]
    doc = {**SYNTHETIC, "devices": devices, "groups": []}
    with pytest.raises(FacilityBuildError) as caught:
        _write(tmp_path, doc)

    assert caught.value.format_message() == (
        "facility: view-unsupported: path channel_finder.pipeline_mode — selects middle_layer "
        "and no device of the facility is in a group or has a class; fix: add a group or give "
        "the devices a class, or select another channel_finder_mode"
    )
    assert not (tmp_path / "channel_finder").exists()


def test_classed_devices_without_groups_still_write_the_index(tmp_path: Path) -> None:
    (index, _database) = _write(tmp_path, {**SYNTHETIC, "groups": []})

    document = json.loads(index.read_bytes())
    assert sorted(key for key in document["M"] if not key.startswith("_")) == [
        "Gauge",
        "Quadrupole",
    ]
    assert _fields(document["M"]["Quadrupole"])["current_setpoint"]["ChannelNames"] == [
        "M:Q1:CURRENT:SP",
        "M:Q2:CURRENT:SP",
    ]


def test_a_database_that_cannot_be_written_stops_with_view_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import duckdb

    from osprey.facility.errors import FacilityBuildError
    from osprey.services.channel_finder.databases import duckdb_import

    def offline(_con: Any) -> None:
        raise duckdb.IOException("the fts extension is not installed")

    monkeypatch.setattr(duckdb_import, "ensure_fts", offline)
    with pytest.raises(FacilityBuildError) as caught:
        _write(tmp_path, SYNTHETIC)

    assert caught.value.format_message() == (
        "facility: view-unsupported: path channel_finder.pipeline_mode — selects middle_layer "
        "and its DuckDB database cannot be written (the fts extension is not installed); "
        "fix: install DuckDB's `fts` extension on this host, or select another "
        "channel_finder_mode"
    )
    assert not (tmp_path / "channel_finder" / "middle_layer.duckdb").exists()


# --- the predicate and the registry ------------------------------------------------


@pytest.mark.parametrize(
    ("rendered_config", "selected"),
    [
        ({"channel_finder": {"pipeline_mode": "middle_layer"}}, True),
        ({"channel_finder": {"pipeline_mode": "hierarchical"}}, False),
        ({"channel_finder": None}, False),
        ({}, False),
    ],
)
def test_the_view_is_written_when_middle_layer_is_selected(
    rendered_config: dict[str, Any], selected: bool
) -> None:
    from osprey.facility.views.channel_finder import middle_layer_selected

    assert middle_layer_selected(_inputs({"channels": []}, rendered_config)) is selected


def test_the_view_is_registered_under_channel_finder() -> None:
    from osprey.facility.views import VIEWS

    (view,) = [view for view in VIEWS if view.name == "middle_layer"]
    assert (view.path, view.selected_by) == ("channel_finder", "channel_finder.pipeline_mode")
    assert view.written_when(_inputs({"channels": []}, {})) == (
        False,
        "channel_finder.pipeline_mode",
    )


def test_a_render_that_selects_another_index_names_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from osprey.facility import views
    from osprey.facility.render import FACILITY_FILE, render_facility_outputs

    (view,) = [view for view in views.VIEWS if view.name == "middle_layer"]
    monkeypatch.setattr(views, "VIEWS", (view,))

    written = render_facility_outputs(
        tmp_path, {"channels": []}, {"channel_finder": {"pipeline_mode": "graph"}}, tmp_path
    )

    assert written == [tmp_path / FACILITY_FILE]
    assert capsys.readouterr() == ("", "")


# --- the demo ----------------------------------------------------------------------


def _group(doc: dict[str, Any], group_id: str) -> dict[str, Any]:
    (group,) = [group for group in doc["groups"] if group["id"] == group_id]
    return group


def _families(document: dict[str, Any]) -> dict[str, list[str]]:
    return {
        system: sorted(key for key in families if not key.startswith("_"))
        for system, families in document.items()
        if system != "schema"
    }


#: The demo's groups by System, each a Family named by its short id.
DEMO_GROUPS = {
    "BR": ["BPM", "DIAG", "DIPOLE", "MAG", "QD", "QF"],
    "BTS": ["BPM", "DIAG", "HCM", "MAG", "VCM"],
    "SR": [
        *("BPM", "DIAG", "DIPOLE", "HCM", "MAG", "QD", "QF", "QFA"),
        *("RF", "SD", "SF", "SHD", "SHF", "VAC", "VCM"),
    ],
}

#: The demo's classes no single-class group holds, by System.
DEMO_DERIVED = {
    "BR": ["BeamCurrentMonitor"],
    "BTS": ["Quadrupole"],
    "SR": [
        *("AcceleratingCavity", "BeamCurrentMonitor", "BeamLossMonitor", "Gauge"),
        *("Modulator", "Pump", "Valve"),
    ],
}


@pytest.mark.slow
def test_every_demo_group_and_ungrouped_class_is_a_family(
    built_control_assistant: BuiltProject,
) -> None:
    document, left_out, by_address = _document(built_control_assistant.facility)

    assert _families(document) == {
        system: sorted([*DEMO_GROUPS[system], *DEMO_DERIVED[system]]) for system in DEMO_GROUPS
    }
    assert sum(len(families) for families in _families(document).values()) == 35
    fields = [
        field
        for system, families in document.items()
        if system != "schema"
        for key, family in families.items()
        if not key.startswith("_")
        for field in _fields(family)
    ]
    assert len(fields) == 1530
    assert by_address == 1387
    assert left_out == 4
    assert left_out == sum(
        1
        for channel in built_control_assistant.facility["channels"]
        if "device" not in (channel.get("on") or {})
    )


@pytest.mark.slow
def test_every_sector_device_s_device_list_names_its_sector(
    built_control_assistant: BuiltProject, tmp_path: Path
) -> None:
    import re

    from osprey.facility.views.channel_finder import _families_by_system, middle_layer_document
    from osprey.services.channel_finder.databases.middle_layer import MiddleLayerDatabase

    facility = built_control_assistant.facility
    devices = {str(device["id"]): device for device in facility["devices"]}
    on = {str(c["id"]): (c.get("on") or {}).get("device") for c in facility["channels"]}
    sector = re.compile(r"SR/SECT(\d+)")
    document, _left_out, _by_address = middle_layer_document(facility)

    checked = 0
    for family in _families_by_system(facility)["SR"]:
        rows = [tuple(row) for row in document["SR"][family.name]["_setup"]["DeviceList"]]
        assert len(rows) == len(set(rows)), family.name
        for member, row in zip(family.members, rows, strict=True):
            match = sector.fullmatch(str(member.get("place") or ""))
            if match:
                assert row[0] == int(match.group(1)), (family.name, member["id"], row)
                checked += 1
    assert checked > 800
    assert not [device["id"] for device in facility["devices"] if "attributes" in device]

    index = tmp_path / "middle_layer.json"
    index.write_text(json.dumps(document), encoding="utf-8")
    database = MiddleLayerDatabase(str(index))
    bpm = document["SR"]["BPM"]
    members = len(bpm["_setup"]["CommonNames"])
    field = next(
        key for key, value in _fields(bpm).items() if len(value["ChannelNames"]) == members
    )
    in_sector_3 = [
        address
        for address in bpm[field]["ChannelNames"]
        if devices[str(on[address])].get("place") == "SR/SECT3"
    ]
    assert len(in_sector_3) == 6
    assert database.list_channel_names("SR", "BPM", field, sectors=[3]) == in_sector_3
    assert {str(on[address]) for address in in_sector_3} == {
        f"SR/BPM{number}" for number in range(13, 19)
    }


@pytest.mark.slow
def test_the_demo_bpm_family_keys_each_once_held_signal_as_one_field(
    built_control_assistant: BuiltProject,
) -> None:
    facility = built_control_assistant.facility
    document, _left_out, _by_address = _document(facility)
    bpm = document["SR"]["BPM"]
    members = _group(facility, "SR/BPM")["members"]
    fields = _fields(bpm)

    assert len(members) == 72
    for signal in (
        "position_x_readback",
        "position_y_readback",
        "position_x_golden_readback",
        "position_y_golden_readback",
    ):
        assert len(fields[signal]["ChannelNames"]) == 72, signal
    twice = [
        channel["id"]
        for channel in facility["channels"]
        if (channel.get("on") or {}).get("device") in members
        and channel.get("signal") in ("status", "position_offset")
    ]
    assert len(twice) == 4 * 72
    assert all(fields[address]["ChannelNames"] == [address] for address in twice)
    assert len(bpm["_setup"]["DeviceList"]) == len(bpm["_setup"]["CommonNames"]) == 72


@pytest.mark.slow
def test_a_demo_field_takes_its_signal_s_vocabulary_sentence(
    built_control_assistant: BuiltProject,
) -> None:
    document, _left_out, _by_address = _document(built_control_assistant.facility)

    for system, family in (("SR", "SF"), ("BR", "DIPOLE")):
        field = _fields(document[system][family])["current_setpoint"]
        assert all(address.endswith(":CURRENT:SP") for address in field["ChannelNames"])
        assert field["_description"] == _signal_text("current_setpoint"), (system, family)


@pytest.mark.slow
def test_the_demo_database_holds_a_row_per_channel_and_family(
    built_control_assistant: BuiltProject, tmp_path: Path
) -> None:
    import duckdb

    (_index, database) = _write(tmp_path, built_control_assistant.facility)

    con = duckdb.connect(str(database), read_only=True)
    try:
        counts = con.execute(
            "SELECT count(*), count(DISTINCT channel_name) FROM channels"
        ).fetchone()

        def family(system: str, name: str) -> set[str]:
            rows = con.execute(
                "SELECT channel_name FROM channels WHERE system = ? AND family = ?",
                [system, name],
            ).fetchall()
            return {channel for (channel,) in rows}

        bpm, diag = family("SR", "BPM"), family("SR", "DIAG")
        quads, magnets = family("SR", "QF"), family("SR", "MAG")
    finally:
        con.close()

    assert counts == (5816, 2908)
    assert bpm and bpm < diag
    assert quads and quads < magnets


@pytest.mark.slow
def test_a_middle_layer_build_writes_the_index_and_notes_its_counts_once(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from osprey.cli.init_cmd import init
    from osprey.facility.views.channel_finder import CHANNEL_FINDER_SCHEMA, MIDDLE_LAYER_FILE
    from tests._builds import run_build

    repo = tmp_path / "demo"
    result = CliRunner().invoke(
        init,
        [
            str(repo),
            "--preset",
            "control-assistant",
            "--no-git",
            "--set",
            "channel_finder_mode=middle_layer",
        ],
    )
    assert result.exit_code == 0, result.output

    built = run_build(repo)

    assert built.exit_code == 0, built.output
    index = repo / "build" / "data" / "channel_finder" / MIDDLE_LAYER_FILE
    assert json.loads(index.read_bytes())["schema"] == CHANNEL_FINDER_SCHEMA
    assert "view middle_layer not written" not in built.output
    assert built.output.count("view middle_layer: 4 channels in no family left out") == 1


def test_an_imported_field_keyed_by_its_signal_carries_the_vocabulary_sentence(
    tmp_path: Path,
) -> None:
    pytest.importorskip("at")
    from osprey.facility.build import build_facility
    from osprey.facility.validate import signal_roles
    from tests.facility._mml_built import WIDENED, import_tree, widen

    facility = import_tree(tmp_path, "spear3")
    widen(facility, WIDENED["spear3"])
    doc = build_facility(facility, project_name="demo")
    roles = signal_roles()

    document, _, _ = _document(doc)

    keyed = [
        (family, key, field)
        for system, families in document.items()
        if not system.startswith("_") and isinstance(families, dict)
        for family, body in families.items()
        if isinstance(body, dict)
        for key, field in _fields(body).items()
        if key in roles
    ]
    assert {key for _, key, _ in keyed} >= {"position_x_readback", "current_setpoint"}
    for family, key, field in keyed:
        assert field["_description"] == _signal_text(key), (family, key)
