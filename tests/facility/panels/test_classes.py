"""Screen classes: the key each written screen states, its standard, and how members differ."""

from __future__ import annotations

import dataclasses
import json
import types
import typing
from pathlib import Path
from typing import Any

import pytest

from osprey.facility.panels import load_graph
from osprey.facility.panels.classes import (
    FLAGS,
    GAP_PARTS,
    SIZE_PARTS,
    STRUCTURE_FLAGS,
    STRUCTURE_PARTS,
    VALUE_FLAGS,
    Measure,
    Sample,
    class_key,
    classify,
    measure,
    standard,
    summary,
    summary_line,
)
from osprey.facility.panels.describe import targets
from osprey.facility.panels.emit import Draw, check, load_draw, write
from osprey.facility.panels.model import PanelGraph
from osprey.facility.panels.view import (
    ChainBlock,
    Fact,
    Link,
    Section,
    Target,
    ValueRow,
    View,
)
from tests.facility.panels._fixture import PANELS_DIR, write_fixture_facility

PUMP_CARDS = ("card-xr01-ip2", "card-xr02-ip1", "card-xr02-ip2")
#: Of the four pumps, XR01:IP1 also states a status (XR02:IP2's interlock is excluded).
PUMP_APART = "card-xr01-ip1"
#: XR02:IP1 states a pose in the ring frame and cites its source; the other pumps do not.
PUMP_KNOWN_MORE = "card-xr02-ip1"
#: Of the two dipoles, one has a calibration and its card a calibration section.
DIPOLE_OFF = "card-xr02-bend1"


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
        assert row["sample"][0]["file"] == row["standard"]
        assert "standard" in row["sample"][0]["as"]
        assert {stop["file"] for stop in row["sample"]} <= set(members)
        assert all(stop["as"] for stop in row["sample"])  # each says what it stands for


def test_differs_means_off_standard_and_nothing_else(written):
    table = written["classes"]
    assert sum(1 for e in written["screens"] if e.get("differs")) == sum(
        c["off_standard"] for c in table.values()
    )
    for entry in written["screens"]:
        if entry["file"] == table[entry["class"]]["standard"]:
            assert "differs" not in entry
            assert "varies" not in entry


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


def test_places_and_checkouts_built_alike_share_a_class(written):
    entries = _entries(written)
    (checkouts,) = {entries[c]["class"] for c in ("checkout-xr-xr01", "checkout-xr-xr02")}
    key = written["classes"][checkouts]["key"]
    assert key["view"] == "checkout"
    assert key["kind"] == "place"
    assert len(key["structure"]) == 16
    assert entries["place-xr-xr01"]["class"] == entries["place-xr-xr02"]["class"]


def test_a_set_is_classed_by_its_structure_not_by_its_members_device_types(tmp_path, written):
    entries = _entries(written)
    assert entries["set-xr01-vac"]["class"] == entries["set-xr02-vac"]["class"]
    key = written["classes"][entries["set-xr01-vac"]["class"]]["key"]
    assert sorted(key) == ["kind", "structure", "view"]
    # Sets of other members drawn alike are one class: a member type is a name.
    graph = _graph(tmp_path)
    by_id = {t.id: t for t in targets(graph) if t.view == "set"}
    quads, pumps = by_id["XR-QUAD"], by_id["XR01-VAC"]
    assert class_key(graph, quads, "s")[0] == class_key(graph, pumps, "s")[0]
    assert class_key(graph, quads, "s")[0] != class_key(graph, quads, "t")[0]


def test_a_calibration_is_classed_by_its_kind_and_its_structure(tmp_path, written):
    graph = _graph(tmp_path)
    found = {t.system: t for t in targets(graph) if t.view == "calibration"}
    centre, offset = found["centre"], found["current_offset"]
    # Drawn alike, a centre and a current offset are still judged apart.
    assert class_key(graph, centre, "s")[0] != class_key(graph, offset, "s")[0]
    assert class_key(graph, centre, "s")[0] != class_key(graph, centre, "t")[0]
    assert class_key(graph, centre, "s")[1] == {
        "view": "calibration",
        "calibration": "centre",
        "structure": "s",
    }
    entries = _entries(written)
    for entry in entries.values():
        if entry["file"].startswith("calibration/"):
            assert written["classes"][entry["class"]]["key"]["calibration"] == entry["system"]


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


def test_a_repeated_role_keeps_the_class_and_its_tiles_section(tmp_path, written):
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

    index = _written(_graph(tmp_path, second_gauge), tmp_path / "out")
    entries = _entries(index)
    pump = entries["card-xr02-ip2"]
    assert pump["class"] == _entries(written)["card-xr02-ip2"]["class"]
    # two readbacks of the key role are one figure, the highest: the tiles stay
    assert "differs" not in pump
    assert "all channels: rows: 2, the standard 1" in pump["varies"]


def test_what_is_known_varies_and_its_gaps_are_named(written):
    entries = _entries(written)
    known = entries[PUMP_KNOWN_MORE]
    assert "differs" not in known
    assert "foot: where: 4, the standard 3" in known["varies"]
    assert "foot: citations: 1, the standard 0" in known["varies"]
    assert known["gaps"]["where without a link"] == 1  # the pose is plain text, no link
    assert "where without a link" not in entries["card-xr01-ip2"]["gaps"]


def test_a_default_card_with_a_block_more_is_another_class_not_off_standard(written):
    entries = _entries(written)
    more = entries[DIPOLE_OFF]  # its calibration section is a block the other has not
    plain = entries["card-xr01-bend1"]
    assert more["class"] != plain["class"]
    table = written["classes"]
    assert "Calibrations" in table[more["class"]]["key"]["blocks"]
    assert "Calibrations" not in table[plain["class"]]["key"]["blocks"]
    assert "differs" not in more and table[more["class"]]["off_standard"] == 0


def _card(file: str, cls: str, blocks: tuple[str, ...], **measured: Any) -> Sample:
    key = {"view": "card", "device_class": "Pump", "recipe": "default", "roles": []}
    target = Target("card", "device", file)
    return Sample(file, target, cls, key, Measure((), "s", **measured), blocks=blocks)


def test_blocks_split_only_a_class_whose_cards_differ_and_leave_the_rest_alone():
    act = ("CardHead", "Operate with controls", "channel list")
    look = ("CardHead", "Operate without controls", "channel list")
    grid = {"channel list": "ChannelGrid"}
    count = {"channel list": "ChannelCount"}
    samples = [
        _card("a", "split", act),
        _card("b", "split", look),
        _card("c", "kept", act, variants=grid),
        _card("d", "kept", act, variants=count),  # grid or count is one block
    ]
    table, entries = classify(samples, stamp="x")
    assert "kept" in table and table["kept"]["members"] == 2  # its id byte for byte
    assert "blocks" not in table["kept"]["key"]
    assert "split" not in table
    halves = [c for c in table.values() if c["key"].get("blocks")]
    assert sorted(c["key"]["blocks"] for c in halves) == [list(act), list(look)]
    assert all(c["members"] == 1 and c["off_standard"] == 0 for c in halves)


def test_an_unsplit_class_keeps_the_id_its_key_alone_gives(written, tmp_path):
    graph = _graph(tmp_path)
    kept = split = 0
    for entry in written["screens"]:
        if entry["kind"] != "card" or entry.get("thing_kind") != "device":
            continue
        own, _key = class_key(graph, Target("card", "device", entry["thing"]), "")
        if "blocks" in written["classes"][entry["class"]]["key"]:
            split += 1
            assert entry["class"] != own
        else:
            kept += 1
            assert entry["class"] == own  # byte for byte the id before the term
    assert kept and split


def test_a_label_the_layout_cuts_is_stated_apart_from_the_class(written):
    cut = [w for e in written["screens"] for w in e.get("cuts", ())]
    assert "XR02-QD1:Current-RB" in cut
    assert not [w for e in written["screens"] for w in e.get("differs", ()) if "cut" in w]
    assert sum(c["cut"] for c in written["classes"].values()) == sum(
        bool(e.get("cuts")) for e in written["screens"]
    )


def test_the_summary_is_in_the_index_and_counts_single_classes_per_view(written):
    table = written["classes"]
    counts = written["class_summary"]
    assert counts == summary(table)
    assert counts["shared"] == sum(1 for c in table.values() if c["members"] > 1)
    assert sum(counts["single"].values()) == counts["classes"] - counts["shared"]
    line = summary_line(counts)
    assert line.startswith(f"{len(table)} classes ({counts['shared']} of more than one screen), ")
    assert f"{counts['off_standard']} screens off their standard" in line


def _sample(file: str, structure: str, **measured: Any) -> Sample:
    return Sample(file, Target("card", "device", file), "c", {}, Measure((), structure, **measured))


def test_the_standard_is_the_structure_most_members_share_not_the_first_file():
    members = [_sample("a", "f1"), _sample("b", "e2"), _sample("c", "e2")]
    assert standard(members).file == "b"
    assert standard(reversed(members)).file == "b"  # whatever the order
    tie = [_sample("a", "f1"), _sample("b", "e2")]
    assert standard(tie).file == "b"  # the smaller structure hash


def test_the_sample_spans_sizes_labels_and_each_side_of_a_drawing_variant():
    grid = {"channel list": "ChannelGrid"}
    members = [
        _sample("a", "s", sizes={"rows": 3}, label=10, variants=grid),
        _sample("b", "s", sizes={"rows": 3}, label=10, variants=grid),
        _sample("c", "s", sizes={"rows": 1}, label=10, variants=grid),
        _sample("d", "s", sizes={"rows": 9}, label=10, variants=grid),
        _sample("e", "s", sizes={"rows": 3}, label=40, variants=grid),
        _sample("f", "s", sizes={"rows": 3}, label=10, variants={"channel list": "ChannelCount"}),
        _sample("g", "t", sizes={"rows": 3}, label=10, variants=grid),
        _sample("h", "s", sizes={"rows": 3}, label=10, variants=grid, gaps={"x_mm": 1}),
    ]
    table, entries = classify(members, stamp="x")
    stops = {stop["file"]: stop["as"] for stop in table["c"]["sample"]}
    assert list(stops)[:5] == ["a", "c", "d", "e", "g"]
    assert stops["a"][0] == "standard" and stops["c"] == ["smallest"]
    assert stops["d"] == ["largest"] and stops["e"] == ["longest labels"]
    assert stops["g"] == ["off standard"]
    assert stops["f"] == ["channel list drawn as ChannelCount"]  # a count, not a grid
    assert stops["h"] == ["x_mm missing"]  # a part drawn without a stated position
    assert "b" not in stops
    assert entries["f"]["varies"] == [
        "channel list drawn as ChannelCount, the standard as ChannelGrid"
    ]
    assert entries["g"]["differs"]
    assert "differs" not in entries["f"]


def _view(*items: Any) -> View:
    return View(Target("card", "device", "x"), "x", (), (Section("relations", tuple(items)),))


def _block(title: str, *, link: Link | None = None, inferred: str | None = None) -> ChainBlock:
    return ChainBlock(title, rows=(), link=link, inferred=inferred)


def test_counts_order_text_and_flags_are_not_structure():
    one = _view(_block("a"), _block("b", inferred="a rule"))
    linked = Link("z", Target("card", "device", "z"))
    other = _view(_block("z", link=linked), _block("y"), _block("x"))
    assert measure(one).structure == measure(other).structure
    assert measure(one).flags == {"inferred": 1}
    assert measure(other).sizes == {"relations: items": 3}
    assert measure(one).gaps == {"no link": 2}


def test_a_kind_of_item_present_or_missing_is_structure():
    row = ValueRow("Current", "X:I", None, None, False)
    assert measure(_view(_block("a"))).structure != measure(_view(_block("a"), row)).structure
    fact = Fact("Pose", "1 m")
    assert measure(_view(fact)).structure != measure(_view(fact, row)).structure


def _decided_fields() -> list[str]:
    """Every field of a view part that may be absent, of several kinds, or yes/no."""
    found: list[str] = []
    seen: set[type] = set()
    todo: list[type] = [View]

    def parts(tp: Any) -> list[type]:
        out = [tp] if isinstance(tp, type) and dataclasses.is_dataclass(tp) else []
        for arg in typing.get_args(tp):
            out.extend(parts(arg))
        return out

    def open_kind(tp: Any) -> bool:
        if typing.get_origin(tp) in (typing.Union, types.UnionType):
            return bool(parts(tp))
        return any(open_kind(arg) for arg in typing.get_args(tp))

    while todo:
        cls = todo.pop()
        if cls in seen:
            continue
        seen.add(cls)
        hints = typing.get_type_hints(cls)
        for f in dataclasses.fields(cls):
            tp = hints[f.name]
            todo.extend(parts(tp))
            if tp is bool or open_kind(tp):
                found.append(f"{cls.__name__}.{f.name}")
    return sorted(found)


def test_every_optional_typed_part_and_yes_no_field_has_a_decision():
    """A part of a view that may be absent, be of several kinds, or be yes/no is
    placed in one table: structure, gap, size or flag. None is left to a default."""
    tables = [GAP_PARTS, SIZE_PARTS, STRUCTURE_PARTS, FLAGS, STRUCTURE_FLAGS, set(VALUE_FLAGS)]
    decided = set().union(*tables)
    assert [name for name in _decided_fields() if name not in decided] == []
    for i, a in enumerate(tables):
        for b in tables[i + 1 :]:
            assert not a & b  # each field is decided once


def test_the_layout_hash_moves_with_the_drawing_settings_not_with_a_member(tmp_path, written):
    entries = _entries(written)
    pump = entries["card-xr01-ip2"]["class"]
    layout = written["classes"][pump]["layout"]
    other = _written(_graph(tmp_path), tmp_path / "plain", Draw())
    assert other["classes"][pump]["layout"] != layout  # other drawing settings
    again = _written(_graph(tmp_path, name="g"), tmp_path / "again")
    assert again["classes"][pump]["layout"] == layout  # a rebuild of the same facility


def test_check_finds_what_the_index_states_and_the_model_does_not_give(tmp_path):
    graph = _graph(tmp_path)
    draw = load_draw(PANELS_DIR / "draw.json")
    out = tmp_path / "out"
    index = _written(graph, out, draw)
    assert not [p for p in check(graph, out, draw=draw) if p.kind == "stale_class"]
    entry = next(e for e in index["screens"] if e["id"] == "card-xr01-ip2")
    entry["class"] = "card-class-somewhere-else-00000000"
    index["classes"]["card-class-somewhere-else-00000000"] = {"members": 1}
    known = next(e for e in index["screens"] if e["id"] == PUMP_KNOWN_MORE)
    known["gaps"] = {}
    (out / "index.json").write_text(json.dumps(index))
    found = [p for p in check(graph, out, draw=draw) if p.kind == "stale_class"]
    assert {p.file for p in found} == {entry["file"], known["file"], "index.json"}
    assert any("gaps" in p.detail for p in found if p.file == known["file"])
    assert all(p.failing for p in found)


def test_a_device_screen_names_what_it_shows_of_other_things():
    from osprey.facility.panels.describe import others
    from osprey.facility.panels.view import (
        BlockRow,
        CardStrip,
        Chain,
        ChainBlock,
        StripMember,
    )

    here = Target("card", "device", "XR01:BPM1")
    strip = CardStrip(
        (0.0, 10.0),
        (
            StripMember("XR01:BPM0", "BPM0", 1.0, "bpm", "beam", channel="XR01-BPM0:X"),
            StripMember("XR01:BPM1", "BPM1", 5.0, "bpm", "beam", channel="XR01-BPM1:X", focus=True),
            StripMember("XR01:BPM2", "BPM2", 9.0, "bpm", "beam", channel="XR01-BPM2:X"),
            StripMember("XR01:VV1", "VV1", 7.0, "valve", "beam"),  # nothing bound of it
        ),
    )
    chain = Chain(
        (
            ChainBlock("BPM XR01:BPM1", None, (BlockRow("x", "XR01-BPM1:X", shown_as="value"),), focus=True),
            ChainBlock(
                "chassis XR01:E1",
                Link("XR01:E1", Target("card", "device", "XR01:E1")),
                (BlockRow("sum", "XR01-E1:Sum", shown_as="value"), BlockRow("slot", text="3")),
            ),
        )
    )  # fmt: skip
    view = View(here, "XR01:BPM1", (), (Section("strip", (strip,)), Section("chain", (chain,))))
    assert others(view) == (
        {"thing": "XR01:BPM0", "as": "strip member", "channels": ["XR01-BPM0:X"]},
        {"thing": "XR01:BPM2", "as": "strip member", "channels": ["XR01-BPM2:X"]},
        {"thing": "XR01:E1", "as": "chain block", "channels": ["XR01-E1:Sum"]},
    )
    # a screen about a set or a place shows its members as what it is about
    assert others(dataclasses.replace(view, id=Target("set", "group", "XR01-BPM"))) == ()


def test_the_check_holds_what_the_index_states_of_other_things(tmp_path):
    graph = _graph(tmp_path)
    draw = load_draw(PANELS_DIR / "draw.json")
    out = tmp_path / "out"
    index = _written(graph, out, draw)
    entry = next(e for e in index["screens"] if e["id"] == "card-xr01-ip2")
    entry["others"] = [{"thing": "XR99:NONE", "as": "strip member", "channels": ["X"]}]
    (out / "index.json").write_text(json.dumps(index))
    found = [p for p in check(graph, out, draw=draw) if p.kind == "stale_class"]
    assert [p.file for p in found] == [entry["file"]]
    assert "others" in found[0].detail


def test_a_row_that_names_no_thing_is_named_by_its_channels_owner(tmp_path):
    from osprey.facility.panels.describe import others

    graph = _graph(tmp_path)
    here = Target("channels", "device", "XR01:QD1")
    rows = (
        ValueRow("current", "XR01-QD1:Current-RB", None, None, False),  # its own
        ValueRow("family", "XR01-QF1:Current-RB", None, None, False),  # one its roles reach
    )
    view = View(here, "XR01:QD1", (), (Section("current", rows),))
    assert others(view, graph) == (
        {"thing": "XR01:QF1", "as": "value row", "channels": ["XR01-QF1:Current-RB"]},
    )
    assert others(view) == ()  # without the graph, an unnamed row is not placed


def test_a_class_names_its_standard_s_drawing_apart_from_its_layout(tmp_path, monkeypatch):
    index = _written(_graph(tmp_path), tmp_path / "out")
    drawings = {c: r["drawing"] for c, r in index["classes"].items()}
    assert all(len(d) == 16 and int(d, 16) >= 0 for d in drawings.values())
    # a facility file that differs in nothing drawn moves the stamp, not the drawing
    again = _written(
        _graph(tmp_path, lambda d: d.setdefault("note", "x"), name="g"), tmp_path / "o"
    )
    assert {c: r["drawing"] for c, r in again["classes"].items()} == drawings
    # the engine drawing a screen otherwise moves it, and leaves `layout` alone
    from osprey.facility.panels import layout

    monkeypatch.setattr(layout, "PANEL_PAD", layout.PANEL_PAD + 1)
    moved = _written(_graph(tmp_path, name="h"), tmp_path / "p")
    assert all(moved["classes"][c]["drawing"] != d for c, d in drawings.items())
    assert all(moved["classes"][c]["layout"] == r["layout"] for c, r in index["classes"].items())
