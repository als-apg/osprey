"""Screen classes: the key each written screen states, its standard, and how members differ."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from osprey.facility.panels import load_graph
from osprey.facility.panels.classes import Sample, standard, summary
from osprey.facility.panels.emit import Draw, check, load_draw, write
from osprey.facility.panels.model import PanelGraph
from osprey.facility.panels.view import Target
from tests.facility.panels._fixture import PANELS_DIR, write_fixture_facility

PUMP_CARDS = ("card-xr01-ip2", "card-xr02-ip1", "card-xr02-ip2")
#: Of the four pumps, XR01:IP1 also states a status (XR02:IP2's interlock is excluded).
PUMP_APART = "card-xr01-ip1"


def _graph(tmp_path: Path, change=None, name: str = "f") -> PanelGraph:
    return load_graph(write_fixture_facility(tmp_path / name / "facility.json", change), PANELS_DIR)


def _written(graph: PanelGraph, out: Path, draw: Draw | None = None) -> dict[str, Any]:
    write(graph, out, draw=draw or load_draw(PANELS_DIR / "draw.json"))
    return json.loads((out / "index.json").read_text())


def _entries(index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {e["id"]: e for e in index["screens"]}


@pytest.fixture(scope="module")
def written(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    tmp = tmp_path_factory.mktemp("classes")
    return _written(_graph(tmp), tmp / "out")


def test_every_screen_states_a_class_the_table_counts(written):
    entries = written["screens"]
    table = written["classes"]
    assert all(e["class"] in table for e in entries)
    assert sum(c["members"] for c in table.values()) == len(entries)
    for name, row in table.items():
        members = [e["file"] for e in entries if e["class"] == name]
        assert row["standard"] in members
        assert row["members"] == len(members)


def test_pumps_with_one_set_of_signals_share_a_class(written):
    entries = _entries(written)
    (shared,) = {entries[card]["class"] for card in PUMP_CARDS}
    assert entries[PUMP_APART]["class"] != shared  # another signal set: its own class
    assert written["classes"][shared]["key"] == {
        "view": "card",
        "device_class": "Pump",
        "recipe": "default",
        "roles": ["vacuum_readback"],
    }


def test_a_place_and_a_rollup_are_each_their_own_class(written):
    for entry in written["screens"]:
        if entry["kind"] in ("place", "rack", "rollup", "checkout", "home"):
            assert entry["class"] == entry["id"]
            assert written["classes"][entry["id"]]["members"] == 1


def test_the_class_id_depends_on_the_key_alone(tmp_path, written):
    def drop_one_pump(document: dict[str, Any]) -> None:
        document["devices"] = [d for d in document["devices"] if d["id"] != "XR02:IP2"]
        document["channels"] = [
            c for c in document["channels"] if (c.get("on") or {}).get("device") != "XR02:IP2"
        ]
        for group in document["groups"]:
            group["members"] = [m for m in group.get("members") or () if m != "XR02:IP2"]

    fewer = _entries(_written(_graph(tmp_path, drop_one_pump), tmp_path / "out"))
    before = _entries(written)
    assert fewer["card-xr01-ip2"]["class"] == before["card-xr01-ip2"]["class"]
    assert fewer["card-xr01-ip2"]["class"].startswith("card-class-pump-default-")


def test_a_repeated_role_stays_in_the_class_and_its_count_is_stated(tmp_path, written):
    def second_gauge(document: dict[str, Any]) -> None:
        document["channels"].append(
            {
                "id": "XR02-IP2:Pressure2-RB",
                "on": {"device": "XR02:IP2"},
                "role": "readback",
                "signal": "vacuum_readback",
                "value_type": "float",
            }
        )

    entries = _entries(_written(_graph(tmp_path, second_gauge), tmp_path / "out"))
    assert entries["card-xr02-ip2"]["class"] == _entries(written)["card-xr02-ip2"]["class"]
    # three members of three structures: the standard is the smallest hash, so either side
    words = {w for e in entries.values() for w in e.get("differs", ())}
    assert words & {
        "2 rows of vacuum_readback, the standard has 1",
        "1 row of vacuum_readback, the standard has 2",
    }


def test_a_member_off_its_standard_says_how_in_words(written):
    entries = _entries(written)
    # XR02:IP1 states a pose in the ring frame and cites its source; the standard does not
    assert "foot: 2 lines under where, the standard has 1" in entries["card-xr02-ip1"]["differs"]
    assert "foot: 1 source, the standard has 0" in entries["card-xr02-ip1"]["differs"]
    assert written["classes"][entries["card-xr02-ip1"]["class"]]["off_standard"] == 1
    for card in PUMP_CARDS:
        if entries[card]["file"] == written["classes"][entries[card]["class"]]["standard"]:
            assert "differs" not in entries[card]


def test_a_label_the_layout_cuts_is_stated(written):
    cut = [w for e in written["screens"] for w in e.get("differs", ()) if w.startswith("label cut")]
    assert "label cut: XR02-QD1:Current-RB" in cut
    assert sum(c["cut"] for c in written["classes"].values()) == sum(
        any(w.startswith("label cut") for w in e.get("differs", ())) for e in written["screens"]
    )


def test_the_summary_counts_classes_and_screens_off_their_standard(written):
    table = written["classes"]
    shared = sum(1 for c in table.values() if c["members"] > 1)
    off = sum(c["off_standard"] for c in table.values())
    assert summary(table).startswith(f"{len(table)} classes ({shared} of more than one screen), ")
    assert f"{off} screens off their standard" in summary(table)


def _sample(file: str, structure: str) -> Sample:
    return Sample(file, Target("card", "device", file), "c", {}, structure, 100)


def test_the_standard_is_the_structure_most_members_share_not_the_first_file():
    members = [_sample("a", "f1"), _sample("b", "e2"), _sample("c", "e2")]
    assert standard(members).file == "b"
    assert standard(reversed(members)).file == "b"  # whatever the order
    tie = [_sample("a", "f1"), _sample("b", "e2")]
    assert standard(tie).file == "b"  # the smaller structure hash


def test_the_layout_hash_moves_with_the_drawing_settings_not_with_a_member(tmp_path, written):
    entries = _entries(written)
    pump = entries["card-xr01-ip2"]["class"]
    layout = written["classes"][pump]["layout"]
    other = _written(_graph(tmp_path), tmp_path / "plain", Draw())
    assert other["classes"][pump]["layout"] != layout  # other drawing settings
    again = _written(_graph(tmp_path, name="g"), tmp_path / "again")
    assert again["classes"][pump]["layout"] == layout  # a rebuild of the same facility


def test_check_finds_a_class_table_the_model_does_not_give(tmp_path):
    graph = _graph(tmp_path)
    draw = load_draw(PANELS_DIR / "draw.json")
    out = tmp_path / "out"
    index = _written(graph, out, draw)
    assert not [p for p in check(graph, out, draw=draw) if p.kind == "stale_class"]
    entry = next(e for e in index["screens"] if e["id"] == "card-xr01-ip2")
    entry["class"] = "card-class-somewhere-else-00000000"
    index["classes"]["card-class-somewhere-else-00000000"] = {"members": 1}
    (out / "index.json").write_text(json.dumps(index))
    found = [p for p in check(graph, out, draw=draw) if p.kind == "stale_class"]
    assert {p.file for p in found} == {entry["file"], "index.json"}
    assert all(p.failing for p in found)
