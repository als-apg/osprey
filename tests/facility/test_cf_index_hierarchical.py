"""The hierarchical channel-finder index: the facility file as a tree of tree levels.

``data/channel_finder/hierarchical.json`` carries
``"schema": "osprey.facility.channel_finder/1"``, a ``hierarchy`` whose levels
are the facility's place level words, then ``class``, ``device`` and ``leaf``,
and a ``tree`` in which every channel sits at the same depth: an absent level
is the node ``-``. Each leaf's ``_channel_part`` is the full address, so the
hierarchical loader reads every channel back as exactly its address.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from tests.facility.conftest import BuiltProject


def _load(path: Path) -> Any:
    from osprey.services.channel_finder.databases.hierarchical import (
        HierarchicalChannelDatabase,
    )

    return HierarchicalChannelDatabase(str(path))


def _leaf(address: str) -> dict[str, Any]:
    return {"_channel_part": address}


# --- the loader --------------------------------------------------------------------


def test_a_bare_placeholder_pattern_loads_each_address_byte_for_byte(tmp_path: Path) -> None:
    addresses = (
        "SR04U___GDS1PS_AC00",
        "SR04U___GDS1PS_AM00",
        "SR01C___QF1____AM00",
        "BTS:HCM1:AC",
    )
    document = {
        "hierarchy": {
            "levels": [{"name": name, "type": "tree"} for name in ("class", "device", "leaf")],
            "naming_pattern": "{class}{device}{leaf}",
        },
        "tree": {
            "-": {
                "_channel_part": "",
                "_description": "no class",
                "-": {
                    "_channel_part": "",
                    "_description": "no device",
                    **{address: _leaf(address) for address in addresses},
                },
            }
        },
    }
    path = tmp_path / "hierarchical.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    database = _load(path)

    assert {row["channel"] for row in database.get_all_channels()} == set(addresses)
    for address in addresses:
        assert database.channel_map[address]["path"] == {"class": "", "device": "", "leaf": address}


# --- the document ------------------------------------------------------------------


def _inputs(doc: dict[str, Any], rendered_config: dict[str, Any] | None = None) -> Any:
    from osprey.facility.views import ViewInputs

    return ViewInputs(
        doc=doc, rendered_config=rendered_config or {}, facility_dir=Path("."), served=[]
    )


def _written(tmp_path: Path, doc: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    from osprey.facility.views.channel_finder import write_hierarchical

    (target,) = write_hierarchical(tmp_path / "channel_finder", _inputs(doc))
    return json.loads(target.read_bytes()), _load(target)


def _assert_every_channel_is_its_address(database: Any, levels: int) -> None:
    for name, entry in database.channel_map.items():
        assert entry["channel"] == name
        assert len(entry["path"]) == levels
        assert entry["path"]["leaf"] == name


def test_two_bare_addresses_sit_under_no_class_and_no_device(tmp_path: Path) -> None:
    document, database = _written(
        tmp_path, {"channels": [{"id": "LAB:TEMP:01"}, {"id": "LAB:TEMP:02"}]}
    )

    assert document["hierarchy"] == {
        "levels": [
            {"name": "class", "type": "tree"},
            {"name": "device", "type": "tree"},
            {"name": "leaf", "type": "tree"},
        ],
        "naming_pattern": "{class}{device}{leaf}",
    }
    assert document["tree"] == {
        "-": {
            "_channel_part": "",
            "_description": "no class",
            "-": {
                "_channel_part": "",
                "_description": "no device",
                "LAB:TEMP:01": {"_channel_part": "LAB:TEMP:01"},
                "LAB:TEMP:02": {"_channel_part": "LAB:TEMP:02"},
            },
        }
    }
    assert sorted(database.channel_map) == ["LAB:TEMP:01", "LAB:TEMP:02"]
    _assert_every_channel_is_its_address(database, 3)


#: A synthetic facility: a machine with sectors, a place-less device, a device
#: with no class, and a channel ``on`` a place.
SYNTHETIC: dict[str, Any] = {
    "places": [
        {"id": "M", "level": "machine", "description": "the machine"},
        {"id": "M/S1", "level": "sector", "description": "sector one"},
        {"id": "N", "level": "machine"},
    ],
    "devices": [
        {"id": "M/Q1", "class": "Quadrupole", "place": "M/S1"},
        {"id": "M/Q2", "class": "Quadrupole", "place": "M/S1", "description": "the second"},
        {"id": "FLOAT", "class": "Gauge"},
        {"id": "N/X", "place": "N"},
    ],
    "groups": [
        {
            "id": "M/QUAD",
            "description": "the quadrupoles",
            "members": ["M/Q1", "M/Q2"],
            "signals": {"CURRENT": "a current", "CURRENT/SP": "the current setpoint"},
        },
        {"id": "M/ALL", "description": "everything", "members": ["M/Q1", "M/Q2", "FLOAT"]},
    ],
    "channels": [
        {
            "id": "M:Q1:CURRENT:SP",
            "on": {"device": "M/Q1"},
            "signal": "current_setpoint",
            "role": "setpoint",
            "description": "own",
        },
        {
            "id": "M:Q1:CURRENT:RB",
            "on": {"device": "M/Q1"},
            "signal": "current_readback",
            "role": "readback",
            "description": "q1 readback",
        },
        {"id": "M:Q2:STATUS:A", "on": {"device": "M/Q2"}, "signal": "status", "role": "readback"},
        {"id": "M:Q2:STATUS:B", "on": {"device": "M/Q2"}, "signal": "status", "role": "readback"},
        {
            "id": "M:Q2:CURRENT",
            "on": {"device": "M/Q2"},
            "signal": "current_readback",
            "role": "readback",
        },
        {
            "id": "M:Q2:CURRENT:SET",
            "on": {"device": "M/Q2"},
            "signal": "current_readback",
            "role": "setpoint",
        },
        {
            "id": "FLOAT:P",
            "on": {"device": "FLOAT"},
            "signal": "pressure",
            "description": "float pressure",
        },
        {"id": "M:TUNE", "on": {"place": "M"}, "description": "the tune"},
        {"id": "N:X:V", "on": {"device": "N/X"}, "signal": "voltage"},
    ],
}


def test_every_channel_of_a_synthetic_tree_sits_at_one_depth(tmp_path: Path) -> None:
    document, database = _written(tmp_path, SYNTHETIC)

    assert [level["name"] for level in document["hierarchy"]["levels"]] == [
        "machine",
        "sector",
        "class",
        "device",
        "leaf",
    ]
    assert sorted(database.channel_map) == sorted(c["id"] for c in SYNTHETIC["channels"])
    _assert_every_channel_is_its_address(database, 5)


def test_the_options_at_each_level_build_back_each_address(tmp_path: Path) -> None:
    document, database = _written(tmp_path, SYNTHETIC)
    levels = [level["name"] for level in document["hierarchy"]["levels"]]
    built: list[str] = []

    def descend(selections: dict[str, str]) -> None:
        level = levels[len(selections)]
        for option in database.get_options_at_level(level, selections):
            chosen = {**selections, level: option["name"]}
            if len(chosen) < len(levels):
                descend(chosen)
                continue
            (channel,) = database.build_channels_from_selections(chosen)
            assert database.validate_channel(channel)
            built.append(channel)

    descend({})
    assert sorted(built) == sorted(c["id"] for c in SYNTHETIC["channels"])
    assert database.build_channels_from_selections(
        {
            "machine": "M",
            "sector": "S1",
            "class": "Quadrupole",
            "device": "M/Q1",
            "leaf": "current_setpoint",
        }
    ) == ["M:Q1:CURRENT:SP"]


def test_a_placeless_device_sits_under_no_place_at_every_place_level(tmp_path: Path) -> None:
    document, database = _written(tmp_path, SYNTHETIC)

    no_machine = document["tree"]["-"]
    assert no_machine["_description"] == "no machine"
    assert no_machine["-"]["_description"] == "no sector"
    assert no_machine["-"]["Gauge"]["FLOAT"]["pressure"] == {
        "_channel_part": "FLOAT:P",
        "_description": "float pressure",
    }
    assert database.build_channels_from_selections(
        {"machine": "", "sector": "", "class": "", "device": "", "leaf": "FLOAT:P"}
    ) == ["FLOAT:P"]
    assert [option["name"] for option in database.get_options_at_level("machine", {})] == [
        "-",
        "M",
        "N",
    ]


def test_a_deviceless_channel_on_a_place_sits_under_no_class_and_no_device(
    tmp_path: Path,
) -> None:
    document, _database = _written(tmp_path, SYNTHETIC)

    machine = document["tree"]["M"]
    assert machine["_description"] == "the machine"
    assert machine["-"]["_description"] == "no sector"
    assert machine["-"]["-"]["_description"] == "no class"
    assert machine["-"]["-"]["-"] == {
        "_channel_part": "",
        "_description": "no device",
        "M:TUNE": {"_channel_part": "M:TUNE", "_description": "the tune"},
    }


def test_a_device_with_no_class_sits_under_no_class(tmp_path: Path) -> None:
    document, _database = _written(tmp_path, SYNTHETIC)

    assert document["tree"]["N"]["-"]["-"]["N/X"]["voltage"] == {"_channel_part": "N:X:V"}


def test_the_levels_are_described_by_their_records(tmp_path: Path) -> None:
    document, _database = _written(tmp_path, SYNTHETIC)

    sector = document["tree"]["M"]["S1"]
    assert sector["_description"] == "sector one"
    assert sector["Quadrupole"]["_description"] == "the quadrupoles"
    assert sector["Quadrupole"]["M/Q1"]["_description"] == "the quadrupoles"
    assert sector["Quadrupole"]["M/Q2"]["_description"] == "the second"
    gauge = document["tree"]["-"]["-"]["Gauge"]
    assert gauge["_description"] == "everything"


def test_a_leaf_is_keyed_by_signal_then_signal_and_role_then_address(tmp_path: Path) -> None:
    document, _database = _written(tmp_path, SYNTHETIC)

    q2 = document["tree"]["M"]["S1"]["Quadrupole"]["M/Q2"]
    assert sorted(key for key in q2 if not key.startswith("_")) == [
        "M:Q2:STATUS:A",
        "M:Q2:STATUS:B",
        "current_readback:readback",
        "current_readback:setpoint",
    ]
    q1 = document["tree"]["M"]["S1"]["Quadrupole"]["M/Q1"]
    assert sorted(key for key in q1 if not key.startswith("_")) == [
        "current_readback",
        "current_setpoint",
    ]


def test_a_leaf_takes_the_longest_family_sentence_its_address_ends_with(
    tmp_path: Path,
) -> None:
    document, _database = _written(tmp_path, SYNTHETIC)

    q1 = document["tree"]["M"]["S1"]["Quadrupole"]["M/Q1"]
    assert q1["current_setpoint"]["_description"] == "the current setpoint"
    assert q1["current_readback"]["_description"] == "q1 readback"
    q2 = document["tree"]["M"]["S1"]["Quadrupole"]["M/Q2"]
    assert q2["current_readback:readback"]["_description"] == "a current"
    assert "_description" not in q2["M:Q2:STATUS:A"]


def test_a_signals_key_spelled_with_underscores_matches_its_runs() -> None:
    from osprey.facility.views.channel_finder import hierarchical_document

    document = hierarchical_document(
        {
            "devices": [{"id": "G1", "class": "Gauge"}],
            "groups": [
                {
                    "id": "G",
                    "members": ["G1"],
                    "signals": {"DOSE_RATE": "a rate", "DOSE_RATE/INST": "instantaneous"},
                }
            ],
            "channels": [{"id": "SR:G:01:DOSE_RATE:INST", "on": {"device": "G1"}}],
        }
    )

    leaf = document["tree"]["Gauge"]["G1"]["SR:G:01:DOSE_RATE:INST"]
    assert leaf["_description"] == "instantaneous"


def test_two_level_words_at_one_depth_stop_with_view_unsupported() -> None:
    from osprey.facility.errors import FacilityBuildError
    from osprey.facility.views.channel_finder import hierarchical_document

    with pytest.raises(FacilityBuildError) as caught:
        hierarchical_document(
            {"places": [{"id": "A", "level": "machine"}, {"id": "B", "level": "line"}]}
        )

    assert caught.value.kind == "view-unsupported"
    assert caught.value.format_message().startswith("facility: view-unsupported: place A — ")
    assert "`line` and `machine` both first appear at depth 0" in caught.value.detail


def test_a_level_word_that_is_a_tail_level_stops_with_view_unsupported() -> None:
    from osprey.facility.errors import FacilityBuildError
    from osprey.facility.views.channel_finder import hierarchical_document

    with pytest.raises(FacilityBuildError) as caught:
        hierarchical_document({"places": [{"id": "A", "level": "device"}]})

    assert caught.value.kind == "view-unsupported"


def test_a_place_without_a_level_word_is_no_node(tmp_path: Path) -> None:
    document, database = _written(
        tmp_path,
        {
            "places": [{"id": "M", "level": "machine"}, {"id": "M/X"}],
            "devices": [{"id": "D", "class": "Gauge", "place": "M/X"}],
            "channels": [{"id": "M:X:D:P", "on": {"device": "D"}}],
        },
    )

    assert sorted(document["tree"]["M"]["Gauge"]["D"]) == ["M:X:D:P", "_channel_part"]
    _assert_every_channel_is_its_address(database, 4)


def test_zero_channels_write_an_empty_tree(tmp_path: Path) -> None:
    from osprey.facility.views.channel_finder import write_hierarchical

    (target,) = write_hierarchical(tmp_path, _inputs({"channels": []}))

    assert json.loads(target.read_bytes())["tree"] == {}


@pytest.mark.parametrize(
    ("doc", "record"),
    [
        (
            {
                "places": [{"id": "_M", "level": "machine"}],
                "devices": [{"id": "D", "class": "Gauge", "place": "_M"}],
                "channels": [{"id": "A", "on": {"device": "D"}}],
            },
            "place _M — its tree key `_M`",
        ),
        (
            {
                "devices": [{"id": "_D", "class": "Gauge"}],
                "channels": [{"id": "A", "on": {"device": "_D"}}],
            },
            "device _D — its tree key `_D`",
        ),
        (
            {
                "devices": [{"id": "D", "class": "Gauge"}],
                "channels": [{"id": "A", "on": {"device": "D"}, "signal": "_raw"}],
            },
            "channel A — its tree key `_raw`",
        ),
        (
            {"channels": [{"id": "_A"}]},
            "channel _A — its tree key `_A`",
        ),
    ],
    ids=["place", "device", "signal", "address"],
)
def test_a_tree_key_beginning_with_an_underscore_stops_with_view_unsupported(
    doc: dict[str, Any], record: str
) -> None:
    from osprey.facility.errors import FacilityBuildError
    from osprey.facility.views.channel_finder import hierarchical_document

    with pytest.raises(FacilityBuildError) as caught:
        hierarchical_document(doc)

    assert caught.value.kind == "view-unsupported"
    message = caught.value.format_message()
    assert message.startswith(f"facility: view-unsupported: {record} begins with `_`")
    assert "a key beginning with `_` is a meta key of the hierarchical index" in message


# --- the predicate and the registry ------------------------------------------------


@pytest.mark.parametrize(
    ("rendered_config", "selected"),
    [
        ({"channel_finder": {"pipeline_mode": "hierarchical"}}, True),
        ({"channel_finder": {"pipeline_mode": "in_context"}}, False),
        ({"channel_finder": None}, False),
        ({}, False),
    ],
)
def test_the_view_is_written_when_hierarchical_is_selected(
    rendered_config: dict[str, Any], selected: bool
) -> None:
    from osprey.facility.views.channel_finder import hierarchical_selected

    assert hierarchical_selected(_inputs({"channels": []}, rendered_config)) is selected


def test_the_view_is_registered_under_channel_finder() -> None:
    from osprey.facility.views import VIEWS

    (view,) = [view for view in VIEWS if view.name == "hierarchical"]
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

    (view,) = [view for view in views.VIEWS if view.name == "hierarchical"]
    monkeypatch.setattr(views, "VIEWS", (view,))

    written = render_facility_outputs(
        tmp_path, {"channels": []}, {"channel_finder": {"pipeline_mode": "graph"}}, tmp_path
    )

    assert written == [tmp_path / FACILITY_FILE]
    assert capsys.readouterr() == ("", "")


def test_the_writer_writes_the_index_with_its_header(tmp_path: Path) -> None:
    from osprey.facility.views.channel_finder import (
        CHANNEL_FINDER_SCHEMA,
        HIERARCHICAL_FILE,
        write_hierarchical,
    )

    root = tmp_path / "data" / "channel_finder"
    written = write_hierarchical(root, _inputs({"channels": [{"id": "A:RB"}]}))

    assert written == [root / HIERARCHICAL_FILE]
    raw = written[0].read_bytes()
    assert raw.endswith(b"}\n")
    assert json.loads(raw)["schema"] == CHANNEL_FINDER_SCHEMA


# --- the demo ----------------------------------------------------------------------


def _group(doc: dict[str, Any], group_id: str) -> dict[str, Any]:
    (group,) = [group for group in doc["groups"] if group["id"] == group_id]
    return group


def _leaf_of(document: dict[str, Any], address: str) -> dict[str, Any]:
    found: list[dict[str, Any]] = []

    def walk(node: dict[str, Any]) -> None:
        for key, child in node.items():
            if key.startswith("_"):
                continue
            if child.get("_channel_part") == address:
                found.append(child)
            else:
                walk(child)

    walk(document["tree"])
    (leaf,) = found
    return leaf


@pytest.mark.slow
def test_the_demo_index_holds_every_channel_at_one_depth(
    built_control_assistant: BuiltProject, tmp_path: Path
) -> None:
    facility = built_control_assistant.facility
    document, database = _written(tmp_path, facility)
    levels = [level["name"] for level in document["hierarchy"]["levels"]]

    assert levels == ["machine", "sector", "class", "device", "leaf"]
    assert sorted(database.channel_map) == sorted(c["id"] for c in facility["channels"])
    _assert_every_channel_is_its_address(database, len(levels))
    assert [
        option["name"] for option in database.get_options_at_level("sector", {"machine": "BR"})
    ] == ["-"]


@pytest.mark.slow
def test_a_demo_leaf_takes_its_family_sentence(
    built_control_assistant: BuiltProject, tmp_path: Path
) -> None:
    facility = built_control_assistant.facility
    document, _database = _written(tmp_path, facility)

    def setpoint(group_id: str) -> str:
        member = _group(facility, group_id)["members"][0]
        (address,) = [
            channel["id"]
            for channel in facility["channels"]
            if (channel.get("on") or {}).get("device") == member
            and channel["id"].endswith(":CURRENT:SP")
        ]
        return address

    sextupole = setpoint("SR/SF")
    dipole = setpoint("BR/DIPOLE")
    assert (
        _leaf_of(document, sextupole)["_description"]
        == _group(facility, "SR/SF")["signals"]["CURRENT/SP"]
    )
    assert (
        _leaf_of(document, dipole)["_description"]
        == _group(facility, "BR/DIPOLE")["signals"]["CURRENT/SP"]
    )
    assert (
        _group(facility, "SR/DIPOLE")["signals"]["CURRENT/SP"]
        != _group(facility, "BR/DIPOLE")["signals"]["CURRENT/SP"]
    )


@pytest.mark.slow
def test_a_hierarchical_build_writes_the_index(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from osprey.cli.init_cmd import init
    from osprey.facility.views.channel_finder import CHANNEL_FINDER_SCHEMA, HIERARCHICAL_FILE
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
            "channel_finder_mode=hierarchical",
        ],
    )
    assert result.exit_code == 0, result.output

    built = run_build(repo)

    assert built.exit_code == 0, built.output
    index = repo / "build" / "data" / "channel_finder" / HIERARCHICAL_FILE
    assert json.loads(index.read_bytes())["schema"] == CHANNEL_FINDER_SCHEMA
    assert "view hierarchical not written" not in built.output
