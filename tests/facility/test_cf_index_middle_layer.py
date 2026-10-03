"""The middle-layer channel-finder index: the facility's groups as System -> Family -> Field.

``data/channel_finder/middle_layer.json`` carries
``"schema": "osprey.facility.channel_finder/1"``, which the middle-layer loader
skips, and one System per top place. A Family is a group, named by its id
less a leading ``<System>/``; its Fields list one channel per
member, in ``CommonNames`` order, so ``ChannelNames`` aligns with
``DeviceList``. The DuckDB copy ``run_sql`` queries is written beside it.
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
        {"id": "N", "level": "machine", "names": ["the other machine"]},
    ],
    "devices": [
        {
            "id": "M/Q2",
            "class": "Quadrupole",
            "place": "M/S1",
            "s": 2.0,
            "names": ["Q2", "Quad 2"],
            "attributes": {"DeviceList": [1, 2], "ElementList": 2},
        },
        {
            "id": "M/Q1",
            "class": "Quadrupole",
            "place": "M/S1",
            "s": 1.0,
            "names": ["Q1", "Quad 1"],
            "attributes": {"DeviceList": [1, 1], "ElementList": 1},
        },
        {
            "id": "N/Q1",
            "class": "Quadrupole",
            "place": "N",
            "names": ["Quad N1"],
            "attributes": {"DeviceList": [1, 1]},
        },
        {"id": "M/G1", "class": "Gauge", "place": "M"},
    ],
    "groups": [
        {
            "id": "M/QUAD",
            "description": "the quadrupoles",
            "members": ["M/Q1", "M/Q2", "N/Q1"],
            "signals": {"CURRENT/SP": "the current setpoint", "CURRENT": "the current"},
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
    assert sorted(document["M"]) == ["ALL", "QUAD", "_description"]
    assert sorted(document["N"]) == ["M/QUAD", "_description"]
    assert _fields(document["N"]["M/QUAD"])["CURRENT/SP"]["ChannelNames"] == ["N:Q1:CURRENT:SP"]


def test_each_class_lists_the_families_the_index_files_its_members_under() -> None:
    from osprey.facility.views.channel_finder import middle_layer_families

    document, _left_out, _by_address = _document(SYNTHETIC)

    assert middle_layer_families(SYNTHETIC) == {
        "Gauge": [("M", "ALL")],
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
        "DeviceList": [[1, 1], [1, 2]],
        "ElementList": [1, 2],
    }
    assert _fields(family)["CURRENT/SP"]["ChannelNames"] == ["M:Q1:CURRENT:SP", "M:Q2:CURRENT:SP"]
    assert _fields(family)["temperature"]["ChannelNames"] == ["M:Q1:TEMP", "M:Q2:TEMP"]


def test_setup_lists_an_attribute_only_when_every_member_states_it() -> None:
    document, _left_out, _by_address = _document(SYNTHETIC)

    assert document["N"]["M/QUAD"]["_setup"] == {
        "CommonNames": ["Quad N1"],
        "DeviceList": [[1, 1]],
    }


def test_a_common_name_is_the_last_names_entry_else_the_id() -> None:
    doc = {
        "places": [{"id": "M", "level": "machine"}],
        "devices": [
            {"id": "M/A", "place": "M", "s": 1.0, "names": ["a", "b"]},
            {"id": "M/C", "place": "M", "s": 2.0},
        ],
        "groups": [{"id": "M/F", "members": ["M/A", "M/C"]}],
        "channels": [],
    }

    document, _left_out, _by_address = _document(doc)

    assert document["M"]["F"]["_setup"]["CommonNames"] == ["b", "M/C"]


def test_a_field_takes_the_sentence_under_the_longest_key_every_address_ends_with() -> None:
    document, _left_out, _by_address = _document(SYNTHETIC)
    fields = _fields(document["M"]["QUAD"])

    assert fields["CURRENT/SP"]["_description"] == "the current setpoint"
    assert "_description" not in fields["temperature"]


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
            {"id": "A", "place": "M", "names": ["a"]},
            {"id": "B", "place": "M", "names": ["b"]},
        ],
        "groups": [{"id": "M/F", "members": ["A", "B"], "signals": {"X": "x"}}],
        "channels": [
            {"id": "A:X", "on": {"device": "A"}},
            {"id": "B:X", "on": {"device": "B"}},
            {"id": "A:H:CURRENT", "on": {"device": "A"}, "signal": "current"},
            {"id": "A:V:CURRENT", "on": {"device": "A"}, "signal": "current"},
            {"id": "B:H:CURRENT", "on": {"device": "B"}, "signal": "current"},
            {"id": "A:ONLY", "on": {"device": "A"}, "signal": "only"},
            {"id": "B:BARE", "on": {"device": "B"}},
        ],
    }

    document, left_out, by_address = _document(doc)
    fields = _fields(document["M"]["F"])

    assert fields["X"] == {"ChannelNames": ["A:X", "B:X"], "_description": "x"}
    keyed = ["A:H:CURRENT", "A:V:CURRENT", "B:H:CURRENT", "A:ONLY", "B:BARE"]
    assert {address: fields[address]["ChannelNames"] for address in keyed} == {
        address: [address] for address in keyed
    }
    assert sorted(fields) == sorted(["X", *keyed])
    assert (left_out, by_address) == (0, 5)


def test_a_group_without_signals_is_a_family_keyed_by_signal_else_address() -> None:
    doc = {
        "devices": [
            {"id": "A", "place": "M", "names": ["a"], "s": 1.0},
            {"id": "B", "place": "M", "names": ["b"], "s": 2.0},
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
    assert quad["CURRENT/SP"]["ChannelNames"] == ["M:Q1:CURRENT:SP", "M:Q2:CURRENT:SP"]
    in_umbrella = {address for field in umbrella.values() for address in field["ChannelNames"]}
    assert {"M:Q1:CURRENT:SP", "M:Q2:CURRENT:SP", "M:G1:P"} <= in_umbrella
    assert all("_description" not in field for field in umbrella.values())


def test_a_shared_endpoint_is_listed_once_per_device_it_ends() -> None:
    doc = {
        "devices": [
            {"id": "A", "place": "M", "names": ["a"], "s": 1.0},
            {"id": "B", "place": "M", "names": ["b"], "s": 2.0},
        ],
        "groups": [{"id": "M/F", "members": ["B", "A"], "signals": {"SP": "setpoint"}}],
        "channels": [{"id": "BUS:SP", "endpoint_of": ["A", "B"]}],
    }

    document, _left_out, _by_address = _document(doc)

    assert document["M"]["F"]["_setup"]["CommonNames"] == ["a", "b"]
    assert _fields(document["M"]["F"])["SP"]["ChannelNames"] == ["BUS:SP", "BUS:SP"]


def test_a_member_with_no_place_sits_under_system_none() -> None:
    doc = {
        "devices": [{"id": "A", "names": ["a"]}],
        "groups": [{"id": "F", "members": ["A"], "signals": {"X": "x"}}],
        "channels": [{"id": "A:X", "on": {"device": "A"}}],
    }

    document, _left_out, _by_address = _document(doc)

    assert document["-"]["_description"] == "no place"
    assert _fields(document["-"]["F"])["X"]["ChannelNames"] == ["A:X"]


def _one_family(place: str, group: str) -> dict[str, Any]:
    return {
        "places": [{"id": place}],
        "devices": [{"id": "A", "place": place, "names": ["a"]}],
        "groups": [{"id": group, "members": ["A"], "signals": {"X": "x"}}],
        "channels": [{"id": "A:X", "on": {"device": "A"}}],
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
    assert loaded.list_channel_names("M", "QUAD", "CURRENT/SP", sectors=[1]) == [
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
        ).fetchone() == (10, 6)
        assert con.execute(
            "SELECT common_name FROM device_map WHERE system = 'M' ORDER BY device_index"
        ).fetchall() == [("Quad 1",), ("Quad 2",)]
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
        "devices": [{"id": "A", "place": "M", "names": ["a"]}],
        "groups": [{"id": "F", "members": ["A"], "signals": {"X": "x"}}],
        "channels": [{"id": "A:X", "on": {"device": "A"}}],
    }

    _write(tmp_path, doc)

    assert capsys.readouterr() == ("", "")


def test_no_group_stops_with_view_unsupported(tmp_path: Path) -> None:
    from osprey.facility.errors import FacilityBuildError

    doc = {**SYNTHETIC, "groups": []}
    with pytest.raises(FacilityBuildError) as caught:
        _write(tmp_path, doc)

    assert caught.value.format_message() == (
        "facility: view-unsupported: path channel_finder.pipeline_mode — selects middle_layer "
        "and the facility has no group; fix: add at least one group, or select another "
        "channel_finder_mode"
    )
    assert not (tmp_path / "channel_finder").exists()


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
    assert (view.path, view.reason, view.selected_by) == (
        "channel_finder",
        "channel_finder.pipeline_mode",
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


@pytest.mark.slow
@pytest.mark.xdist_group("built_control_assistant")
def test_every_demo_group_is_a_family(
    built_control_assistant: BuiltProject,
) -> None:
    document, left_out, by_address = _document(built_control_assistant.facility)
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    umbrellas = {"BR": ["DIAG", "MAG"], "BTS": ["DIAG", "MAG"], "SR": ["DIAG", "MAG", "RF", "VAC"]}

    assert _families(document) == {
        system: sorted([*families, *umbrellas[system]])
        for system, families in _families(golden).items()
    }
    assert sum(len(families) for families in _families(document).values()) == 36
    fields = [
        field
        for system, families in document.items()
        if system != "schema"
        for key, family in families.items()
        if not key.startswith("_")
        for field in _fields(family)
    ]
    assert len(fields) == 1172
    assert by_address == 1011
    assert left_out == 4
    assert left_out == sum(
        1
        for channel in built_control_assistant.facility["channels"]
        if "device" not in (channel.get("on") or {})
    )


@pytest.mark.slow
@pytest.mark.xdist_group("built_control_assistant")
def test_the_demo_bpm_family_has_one_field_per_today_s_leaf(
    built_control_assistant: BuiltProject,
) -> None:
    document, _left_out, _by_address = _document(built_control_assistant.facility)
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    bpm = document["SR"]["BPM"]
    leaves = sorted(
        f"{field}/{leaf}"
        for field, node in _fields(golden["SR"]["BPM"]).items()
        for leaf in _fields(node)
    )
    members = len(_group(built_control_assistant.facility, "SR/BPM")["members"])

    assert sorted(_fields(bpm)) == leaves
    assert all(len(field["ChannelNames"]) == members for field in _fields(bpm).values())
    assert bpm["_setup"] == golden["SR"]["BPM"]["_setup"]


@pytest.mark.slow
@pytest.mark.xdist_group("built_control_assistant")
def test_a_demo_field_takes_its_own_machine_s_family_sentence(
    built_control_assistant: BuiltProject,
) -> None:
    facility = built_control_assistant.facility
    document, _left_out, _by_address = _document(facility)

    sextupole = _fields(document["SR"]["SF"])["CURRENT/SP"]
    assert all(address.endswith(":CURRENT:SP") for address in sextupole["ChannelNames"])
    assert sextupole["_description"] == _group(facility, "SR/SF")["signals"]["CURRENT/SP"]
    for key, field in _fields(document["BR"]["DIPOLE"]).items():
        assert field["_description"] == _group(facility, "BR/DIPOLE")["signals"][key]
    assert (
        _fields(document["BR"]["DIPOLE"])["CURRENT/SP"]["_description"]
        != _group(facility, "SR/DIPOLE")["signals"]["CURRENT/SP"]
    )


@pytest.mark.slow
@pytest.mark.xdist_group("built_control_assistant")
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
