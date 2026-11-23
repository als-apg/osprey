"""Writing the screens to disk and checking a written set against the model."""

from __future__ import annotations

import gc
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import pytest

from osprey_panels import emit, load_graph, targets
from osprey_panels.classes import DRAWING_RULE
from osprey_panels.emit import (
    FORMAT_VERSION,
    SCREEN_SUFFIX,
    EmitError,
    Problem,
    check,
    screen_id,
    screen_ids,
    under,
    write,
)
from osprey_panels.model import PanelGraph
from osprey_panels.view import Target
from osprey_panels_tests._fixture import graph_with_part, graph_with_wide, write_fixture_facility
from osprey_panels_tests._harness import REPO_ROOT


def _drop(*device_ids: str):
    """A ``change`` for the fixture build: the devices, and all that names them, are gone."""
    gone = set(device_ids)

    def change(document: dict[str, Any]) -> None:
        document["devices"] = [d for d in document["devices"] if d["id"] not in gone]
        document["channels"] = [
            c
            for c in document["channels"]
            if (c.get("on") or {}).get("device") not in gone
            and not gone & set(c.get("endpoint_of") or ())
        ]
        document["relations"] = [
            r
            for r in document["relations"]
            if (r.get("subject") or {}).get("device") not in gone
            and (r.get("object") or {}).get("device") not in gone
        ]
        for group in document["groups"]:
            group["members"] = [m for m in group.get("members") or () if m not in gone]

    return change


def _graph(tmp_path: Path, panel_tables: Path, change=None, name: str = "f") -> PanelGraph:
    return load_graph(
        write_fixture_facility(tmp_path / name / "facility.json", change), panel_tables
    )


def _files(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _kinds(problems: list[Problem]) -> set[str]:
    return {p.kind for p in problems}


def _walk(node: Any):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


# -- ids ------------------------------------------------------------------------------


def test_a_screen_id_is_a_lowercase_slug_prefixed_by_its_kind():
    assert screen_id("card", "XR01:QF1") == "card-xr01-qf1"
    assert screen_id("rollup", "XR/XR01", "system") == "rollup-system-xr-xr01"


def test_a_long_id_is_cut_and_carries_a_digest():
    sid = screen_id("card", "X" * 300)
    assert len(sid) <= emit.SLUG_MAX + 40
    assert sid != screen_id("card", "X" * 299)
    assert sid.startswith("card-x")


def test_every_id_is_file_name_safe():
    for sid in (screen_id("card", "Ä ü/..\\ :*?"), screen_id("card", "")):
        assert sid and set(sid) <= set("abcdefghijklmnopqrstuvwxyz0123456789-")


def _thing_targets(*ids: str, view: str = "card", kind: str = "device") -> list[Target]:
    return [Target(view, kind, i) for i in ids]  # type: ignore[arg-type]


def test_ids_differing_only_in_case_or_punctuation_get_different_screen_ids():
    found = _thing_targets("XR01:QF1", "xr01:qf1", "XR01/QF1", "XR01_QF1", "XR01-QF1")
    ids = screen_ids(found)
    assert len(set(ids.values())) == len(found)
    assert all(v.startswith("card-xr01-qf1-") for v in ids.values())


def test_an_id_that_sanitises_like_another_target_of_the_kind_gets_a_digest_only_then():
    found = _thing_targets("A:B", "A/B", "C:D")
    ids = screen_ids(found)
    assert ids[found[2]] == "card-c-d"
    assert ids[found[0]] != ids[found[1]]


def test_two_things_of_one_id_and_different_kinds_get_different_ids():
    found = [Target("card", "device", "X1"), Target("card", "group", "X1")]
    assert len(set(screen_ids(found).values())) == 2


def test_ids_do_not_depend_on_the_order_of_the_targets():
    found = _thing_targets("A:B", "a:b", "A/B", "other")
    assert screen_ids(found) == screen_ids(list(reversed(found)))


def test_a_300_character_id_gets_a_short_name_and_stays_apart_from_its_prefix_twin():
    long = "Q" * 300
    found = _thing_targets(long, long + "Z")
    ids = screen_ids(found)
    assert len(set(ids.values())) == 2
    assert all(len(v) + len(SCREEN_SUFFIX) < 150 for v in ids.values())


def test_roll_ups_by_place_and_by_system_have_different_ids(panel_graph):
    ids = screen_ids(targets(panel_graph))
    assert len(set(ids.values())) == len(ids)
    by = {(t.view, t.id, t.by): i for t, i in ids.items()}
    assert by[("rollup", "XR", "place")] != by[("rollup", "XR", "system")]


# -- write ----------------------------------------------------------------------------


def test_write_then_check_is_clean(panel_graph, tmp_path):
    paths = write(panel_graph, tmp_path / "out")
    assert len(paths) == len(targets(panel_graph))
    assert all(p.is_file() and p.name.endswith(SCREEN_SUFFIX) for p in paths)
    assert check(panel_graph, tmp_path / "out") == []


def test_files_sit_under_their_view_kind_with_an_index(panel_graph, tmp_path):
    out = tmp_path / "out"
    write(panel_graph, out)
    index = json.loads((out / "index.json").read_text())
    assert index["format"] == FORMAT_VERSION
    assert index["facility_hash"] == panel_graph.content_hash
    assert index["drawing_rule"] == DRAWING_RULE  # how ``drawing`` is taken
    assert len(index["screens"]) == len(targets(panel_graph))
    entry = next(e for e in index["screens"] if e["kind"] == "rollup" and e["by"] == "system")
    assert set(entry) >= {"id", "kind", "thing_kind", "thing", "by", "title", "file", "bytes"}
    assert (
        entry["file"].startswith("rollup/")
        and (out / entry["file"]).stat().st_size == entry["bytes"]
    )


def test_every_file_carries_the_stamp_in_its_description(panel_graph, tmp_path):
    out = tmp_path / "out"
    write(panel_graph, out)
    for path in out.glob("*/*.json"):
        text = json.loads(path.read_text())["metadata"]["description"]
        assert panel_graph.content_hash in text and f"format {FORMAT_VERSION}" in text


def test_write_is_deterministic_across_directories(panel_graph, tmp_path):
    write(panel_graph, tmp_path / "a")
    write(panel_graph, tmp_path / "b" / "deeper")
    one, two = _files(tmp_path / "a"), _files(tmp_path / "b" / "deeper")
    assert one == two


def test_no_file_holds_a_path_a_host_or_a_time(panel_graph, tmp_path):
    out = tmp_path / "out"
    write(panel_graph, out)
    blob = b"".join(_files(out).values())
    assert str(tmp_path).encode() not in blob
    assert b"/Users/" not in blob and b"/private/" not in blob


def test_the_content_hash_is_the_same_for_a_copy_of_the_inputs(
    panel_graph, panel_facility, panel_tables, tmp_path
):
    copy = tmp_path / "elsewhere"
    copy.mkdir()
    (copy / "facility.json").write_bytes(panel_facility.read_bytes())
    import shutil

    shutil.copytree(panel_tables, copy / "panels")
    assert (
        load_graph(copy / "facility.json", copy / "panels").content_hash == panel_graph.content_hash
    )


def test_every_navigation_target_resolves_to_a_file_of_the_set(panel_graph, tmp_path):
    out = tmp_path / "out"
    write(panel_graph, out)
    seen = 0
    for path in out.glob("*/*.json"):
        for node in _walk(json.loads(path.read_text())):
            ref = node.get("file")
            if isinstance(ref, str):
                seen += 1
                assert (path.parent / ref).resolve().is_file(), (path.name, ref)
    assert seen > 20


def test_links_are_relative_to_the_file_that_holds_them(panel_graph, tmp_path):
    out = tmp_path / "out"
    write(panel_graph, out)
    refs = {
        node["file"]
        for path in out.glob("*/*.json")
        for node in _walk(json.loads(path.read_text()))
        if isinstance(node.get("file"), str)
    }
    assert refs and all(r.startswith("../") and r.endswith(SCREEN_SUFFIX) for r in refs)


def test_document_ids_are_the_paths_below_the_out_directory(panel_graph, tmp_path):
    out = tmp_path / "out"
    write(panel_graph, out)
    for path in out.glob("*/*.json"):
        rel = path.relative_to(out).as_posix()
        assert json.loads(path.read_text())["id"] == rel[: -len(SCREEN_SUFFIX)]


def test_a_second_write_deletes_exactly_the_files_that_are_no_longer_targets(
    panel_graph, panel_tables, tmp_path
):
    out = tmp_path / "out"
    write(panel_graph, out)
    before = set(_files(out))
    smaller = _graph(tmp_path, panel_tables, _drop("XR01:PS1"), "smaller")
    kept_ids = set(screen_ids(targets(smaller)).values())
    write(smaller, out)
    after = set(_files(out))
    gone = before - after
    # its card, and the roll-ups of the one system only it had (PowerSupply)
    assert "card/card-xr01-ps1.screen.json" in gone
    assert all(name.startswith(("card/", "rollup/rollup-system-")) for name in gone)
    assert all("powersupply" in name for name in gone if name.startswith("rollup/"))
    assert after == {"index.json"} | {
        f"{t.view}/{i}{SCREEN_SUFFIX}" for t, i in screen_ids(targets(smaller)).items()
    }
    assert {Path(n).name[: -len(SCREEN_SUFFIX)] for n in gone} - kept_ids == {
        Path(n).name[: -len(SCREEN_SUFFIX)] for n in gone
    }
    assert check(smaller, out) == []


def test_a_foreign_file_in_out_is_left_alone(panel_graph, tmp_path):
    out = tmp_path / "out"
    write(panel_graph, out)
    (out / "card" / "notes.txt").write_text("mine")
    (out / "mine.json").write_text("{}")
    write(panel_graph, out)
    assert (out / "card" / "notes.txt").read_text() == "mine"
    assert (out / "mine.json").read_text() == "{}"
    assert check(panel_graph, out) == []


def test_a_listed_path_that_leaves_out_is_never_deleted(panel_graph, tmp_path):
    out = tmp_path / "out"
    write(panel_graph, out)
    victim = tmp_path / "victim.txt"
    victim.write_text("keep")
    index = json.loads((out / "index.json").read_text())
    index["screens"].append({"id": "x", "kind": "card", "file": "../victim.txt", "bytes": 4})
    (out / "index.json").write_text(json.dumps(index))
    with pytest.warns(UserWarning, match="victim.txt"):
        write(panel_graph, out)
    assert victim.read_text() == "keep"


def test_write_into_a_non_empty_directory_without_an_index_refuses(panel_graph, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "something.txt").write_text("x")
    with pytest.raises(EmitError, match="index.json"):
        write(panel_graph, out)
    assert _files(out) == {"something.txt": b"x"}


def test_write_into_an_empty_or_new_directory_is_fine(panel_graph, tmp_path):
    (tmp_path / "empty").mkdir()
    write(panel_graph, tmp_path / "empty")
    write(panel_graph, tmp_path / "new" / "deep")
    assert _files(tmp_path / "empty") == _files(tmp_path / "new" / "deep") != {}


def test_no_temporary_file_stays_behind(panel_graph, tmp_path):
    write(panel_graph, tmp_path / "out")
    assert not [n for n in _files(tmp_path / "out") if n.endswith(".tmp")]


def test_only_restricts_the_set_and_links_outside_it_are_still_emitted(panel_graph, tmp_path):
    out = tmp_path / "out"
    paths = write(panel_graph, out, only=lambda t: t.view == "card" and t.kind == "device")
    assert paths and all(p.parent.name == "card" for p in paths)
    index = json.loads((out / "index.json").read_text())
    assert index["outside"]
    problems = check(panel_graph, out)
    assert problems and _kinds(problems) == {"outside_set"}
    assert all(not p.failing for p in problems)


def test_under_a_place_writes_what_is_within_it(panel_graph, tmp_path):
    out = tmp_path / "out"
    write(panel_graph, out, only=under(panel_graph, "XR/XR01"))
    written = {
        (e["thing_kind"], e["thing"])
        for e in json.loads((out / "index.json").read_text())["screens"]
    }
    assert ("device", "XR01:QF1") in written and ("place", "XR/XR01") in written
    assert ("device", "XR02:QF1") not in written
    assert _kinds(check(panel_graph, out)) <= {"outside_set"}


def test_write_under_keeps_a_write_to_places_and_check_agrees(tmp_path):
    from osprey_panels.emit import Draw, write_scope

    graph = graph_with_part(tmp_path)  # XR01:CARD1, in no place, part of a racked chassis
    draw = Draw(write_under=("XR/XR01",))
    assert draw.digest == Draw().digest  # the screens do not depend on it
    scope = write_scope(graph, draw.write_under)
    assert "XR01:CARD1" in scope.devices and "XR02:QF1" not in scope.devices
    assert "XR/XR01/RK1" in scope.places and "XR/XR02" not in scope.places
    out = tmp_path / "out"
    write(graph, out, draw=draw)
    index = json.loads((out / "index.json").read_text())
    written = {(e["kind"], e["thing_kind"], e["thing"]) for e in index["screens"]}
    assert ("home", "place", "facility") in written
    assert ("card", "device", "XR01:CARD1") in written
    assert ("card", "device", "XR02:QF1") not in written
    assert any(e["thing"] == "XR02:QF1" for e in index["outside"])
    problems = check(graph, out, draw=draw)
    assert not [p for p in problems if p.failing], problems
    # a screen the scope keeps must be there
    (out / next(e["file"] for e in index["screens"] if e["thing"] == "XR01:QF1")).unlink()
    assert "missing" in _kinds(check(graph, out, draw=draw))


def test_write_under_a_place_that_is_not_there_is_refused(panel_graph, tmp_path):
    from osprey_panels.emit import Draw, load_draw

    path = tmp_path / "draw.json"
    path.write_text(json.dumps({"write_under": ["XR/XR01"]}))
    assert load_draw(path).write_under == ("XR/XR01",)
    path.write_text(json.dumps({"write_under": "XR/XR01"}))
    with pytest.raises(EmitError, match="write_under must be a list"):
        load_draw(path)
    with pytest.raises(EmitError, match="no place of the facility"):
        write(panel_graph, tmp_path / "out", draw=Draw(write_under=("XR/XR99",)))


def test_a_later_subset_write_keeps_the_files_of_the_earlier_one(panel_graph, tmp_path):
    out = tmp_path / "out"
    write(panel_graph, out, only=lambda t: t.view == "card")
    write(panel_graph, out, only=lambda t: t.view == "rollup")
    kinds = {e["kind"] for e in json.loads((out / "index.json").read_text())["screens"]}
    assert {"card", "rollup"} <= kinds


# -- check ----------------------------------------------------------------------------


def test_removing_a_thing_makes_its_screen_an_orphan_and_each_link_to_it_dangle(
    panel_graph, panel_tables, tmp_path
):
    out = tmp_path / "out"
    write(panel_graph, out)
    smaller = _graph(tmp_path, panel_tables, _drop("XR01:PS1"), "smaller")
    problems = check(smaller, out)
    by_kind = {k: [p for p in problems if p.kind == k] for k in _kinds(problems)}
    assert {"orphan", "stale", "dangling_link"} <= set(by_kind)
    assert any("xr01-ps1" in p.file for p in by_kind["orphan"])
    stale = {p.file for p in by_kind["stale"]}
    assert any("xr01-qf1" in f for f in stale) and any("xr01-qd1" in f for f in stale)
    dangling = by_kind["dangling_link"]
    assert any("xr01-qf1" in p.file and "xr01-ps1" in p.detail for p in dangling)
    assert all(p.failing for p in problems)


def test_a_missing_file_is_reported(panel_graph, tmp_path):
    out = tmp_path / "out"
    paths = write(panel_graph, out)
    paths[0].unlink()
    problems = check(panel_graph, out)
    assert [p.kind for p in problems if p.kind == "missing"] == ["missing"]
    assert any(paths[0].name in p.file for p in problems if p.kind == "missing")


def test_a_new_target_without_a_file_is_missing(panel_graph, panel_tables, tmp_path):
    out = tmp_path / "out"
    smaller = _graph(tmp_path, panel_tables, _drop("XR01:PS1"), "smaller")
    write(smaller, out)
    assert "missing" in _kinds(check(panel_graph, out))


def test_a_changed_file_is_stale(panel_graph, tmp_path):
    out = tmp_path / "out"
    paths = write(panel_graph, out)
    doc = json.loads(paths[0].read_text())
    doc["metadata"]["title"] = "changed"
    paths[0].write_text(json.dumps(doc))
    problems = check(panel_graph, out)
    assert [(p.kind, p.file) for p in problems] == [("stale", paths[0].relative_to(out).as_posix())]


def test_a_file_that_is_not_json_is_invalid(panel_graph, tmp_path):
    out = tmp_path / "out"
    paths = write(panel_graph, out)
    paths[1].write_text("{not json")
    assert [p.kind for p in check(panel_graph, out)] == ["invalid"]


def test_a_changed_facility_gives_a_stamp_problem(panel_graph, panel_tables, tmp_path):
    out = tmp_path / "out"
    write(panel_graph, out)
    other = _graph(
        tmp_path, panel_tables, lambda d: d.setdefault("identity", {}).update(name="Other"), "other"
    )
    assert other.content_hash != panel_graph.content_hash
    problems = check(other, out)
    assert "stamp" in _kinds(problems)
    assert any(p.file == "index.json" for p in problems if p.kind == "stamp")


def test_a_bound_channel_that_the_facility_lacks_is_reported(panel_graph, tmp_path):
    out = tmp_path / "out"
    write(panel_graph, out)
    victim = next(p for p in sorted(out.glob("card/*.json")) if '"pv":' in p.read_text())
    text = victim.read_text()
    doc = json.loads(text)
    node = next(n for n in _walk(doc) if isinstance(n.get("pv"), str))
    node["pv"] = "NOPE:Ghost"
    victim.write_text(
        json.dumps(doc, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
    )
    problems = [p for p in check(panel_graph, out) if p.kind == "unknown_channel"]
    assert [(p.file, "NOPE:Ghost" in p.detail) for p in problems] == [
        (victim.relative_to(out).as_posix(), True)
    ]


def test_a_channel_bound_inside_a_rule_is_checked_too(panel_graph, tmp_path):
    out = tmp_path / "out"
    write(panel_graph, out)
    victim = next(p for p in sorted(out.glob("*/*.json")) if '"pvs":' in p.read_text())
    doc = json.loads(victim.read_text())
    node = next(n for n in _walk(doc) if isinstance(n.get("pvs"), list) and n["pvs"])
    node["pvs"][0]["name"] = "NOPE:InRule"
    victim.write_text(json.dumps(doc))
    assert any(
        "NOPE:InRule" in p.detail for p in check(panel_graph, out) if p.kind == "unknown_channel"
    )


def test_every_channel_a_written_screen_binds_is_a_channel_of_the_facility(panel_graph, tmp_path):
    out = tmp_path / "out"
    write(panel_graph, out)
    bound = set()
    for path in out.glob("*/*.json"):
        for node in _walk(json.loads(path.read_text())):
            if isinstance(node.get("pv"), str) and not node["pv"].startswith("formula://"):
                bound.add(node["pv"])
            for pv in node.get("pvs") or ():
                if not pv["name"].startswith("formula://"):
                    bound.add(pv["name"])
            bound.update((node.get("bindings") or {}).values())  # a screen formula's
    assert bound
    locals_ = {name for name in bound if name.startswith("loc://")}  # a typed value before Apply
    assert all(name.endswith('<VString>("")') for name in locals_)
    for name in bound - locals_:
        panel_graph.channel(name)


def _numeric(value) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _shown(expression: str, local) -> bool | None:
    """The viewer's reading of the Apply gate for a local value: ``None`` where it
    gives no output (its evaluator refuses text in arithmetic), else the truth."""
    assert expression == "{0} * 0 == 0"
    if isinstance(local, str):
        return None
    return local * 0 == 0  # an infinity gives not-a-number, equal to nothing


def test_no_written_button_can_write_a_value_no_one_stated_or_typed(panel_graph, tmp_path):
    """Every button of every written screen: a press with a stated number and no rule
    that changes it, or an entry's Apply that is off the canvas, with text as its
    press value and its rule's default, until its local holds a finite number."""
    from osprey_panels import layout

    out = tmp_path / "out"
    write(panel_graph, out)
    applies = 0
    for path in out.glob("*/*.json"):
        doc = json.loads(path.read_text())
        formulas = {"formula://" + f["name"]: f for f in doc.get("formulas", ())}
        for node in _walk(doc):
            if node.get("type") != "pv-button":
                continue
            props, rules = node["props"], node.get("rules") or []
            presses = [r for r in rules if r["targetProperty"] == "pressValue"]
            if not presses:
                assert _numeric(props["pressValue"]), (path.name, props)
                assert all(r["targetProperty"] not in {"x", "pv"} for r in rules)
                continue
            applies += 1
            assert props["pressValue"] == layout.NOTHING_ENTERED
            assert not _numeric(props["pressValue"])
            assert all(not _numeric(r["default"]) for r in presses)
            assert node["geometry"]["x"] + node["geometry"]["width"] <= 0  # nothing to click
            assert {r["targetProperty"] for r in rules} == {"x", "invisible", "pressValue"}
            (moves,) = [r for r in rules if r["targetProperty"] == "x"]
            assert moves["default"] + node["geometry"]["width"] <= 0
            # not in the page (the viewer renders nothing), so Tab cannot reach it
            (hides,) = [r for r in rules if r["targetProperty"] == "invisible"]
            assert props["invisible"] is True and hides["default"] is True
            assert [c["value"] for c in hides["conditions"]] == [False]
            assert hides["conditions"][0]["expression"] == moves["conditions"][0]["expression"]
            (local,) = {pv["name"] for pv in moves["pvs"]}
            assert {pv["name"] for pv in hides["pvs"]} == {local}
            # its words name exactly the channel it writes
            assert props["tooltip"] == f"writes the new value to {props['pv']}"
            lines = [
                n["props"]["text"] for n in _walk(doc) if n.get("type") == "text-label"
                and "Apply writes it (" in str(n.get("props", {}).get("text", ""))
            ]  # fmt: skip
            named = {
                name for line in lines
                for name in line.split("Apply writes it (", 1)[1].rstrip(")").split(", ")
            }  # fmt: skip
            assert any(props["pv"] == name or props["pv"].endswith(":" + name) for name in named), (
                path.name,
                props["pv"],
                named,
            )
            assert local.endswith('<VString>("")')  # starts as nothing entered
            for rule in presses:
                assert set(formulas[rule["formula"]]["bindings"].values()) == {local}
            (condition,) = moves["conditions"]
            assert [_shown(condition["expression"], v) for v in ("", "abc", 2.5, -0.0, float("inf"))] == [
                None, None, True, True, False,
            ]  # fmt: skip
            assert condition["value"] >= 0  # its place on the card, where the hint is
    assert applies


def test_oversize_is_a_problem_only_with_a_limit(panel_graph, tmp_path):
    out = tmp_path / "out"
    write(panel_graph, out)
    assert check(panel_graph, out) == []
    problems = check(panel_graph, out, max_bytes=1)
    assert _kinds(problems) == {"oversize"} and all(p.failing for p in problems)
    assert check(panel_graph, out, max_bytes=10**9) == []


def test_the_index_records_the_size_of_every_screen(panel_graph, tmp_path):
    out = tmp_path / "out"
    write(panel_graph, out)
    for entry in json.loads((out / "index.json").read_text())["screens"]:
        assert entry["bytes"] == (out / entry["file"]).stat().st_size


def test_check_writes_nothing(panel_graph, panel_tables, tmp_path):
    out = tmp_path / "out"
    write(panel_graph, out)
    before = _files(out)
    stats = {n: (out / n).stat().st_mtime_ns for n in before}
    check(_graph(tmp_path, panel_tables, _drop("XR01:PS1"), "smaller"), out)
    assert _files(out) == before
    assert {n: (out / n).stat().st_mtime_ns for n in before} == stats


def test_check_of_a_directory_with_no_index_says_so(panel_graph, tmp_path):
    (tmp_path / "out").mkdir()
    problems = check(panel_graph, tmp_path / "out")
    assert [(p.kind, p.file) for p in problems] == [("invalid", "index.json")]


def test_a_second_write_after_a_change_makes_check_clean_again(panel_graph, panel_tables, tmp_path):
    out = tmp_path / "out"
    write(panel_graph, out)
    other = _graph(
        tmp_path, panel_tables, lambda d: d.setdefault("identity", {}).update(name="Other"), "other"
    )
    assert check(other, out)
    write(other, out)
    assert check(other, out) == []


def test_the_digest_of_a_screen_name_is_eight_hex_digits():
    found = _thing_targets("A:B", "A/B")
    ids = screen_ids(found)
    digest = hashlib.sha256("\0".join(("card", "device", "A:B", "")).encode()).hexdigest()[:8]
    assert ids[found[0]] == f"card-a-b-{digest}"


# -- scale ----------------------------------------------------------------------------


def _big_graph(devices: int) -> PanelGraph:
    from osprey_panels_tests.test_describe import _big_graph as build

    return build(devices)


def test_write_and_check_scale_near_linearly(tmp_path):
    def once(n: int, tag: int) -> float:
        graph = _big_graph(n)
        started = time.perf_counter()
        out = tmp_path / f"{n}-{tag}"
        write(
            graph,
            out,
            only=lambda t: t.view in {"card", "checkout"} and sum(map(ord, t.id)) % 4 == 0,
        )
        check(graph, out)
        return time.perf_counter() - started

    def cost(n: int) -> float:
        gc.collect()
        gc.disable()
        try:
            return min(once(n, k) for k in range(2))
        finally:
            gc.enable()

    small = cost(2_000)
    ratio = cost(8_000) / small
    assert ratio < 8, f"4x the size took {ratio:.1f}x the time"


def test_the_content_hash_moves_with_a_rule_table(panel_facility, panel_graph, tmp_path):
    from osprey_panels_tests._fixture import copy_tables

    tables = copy_tables(tmp_path / "panels")
    with (tables / "health.csv").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    assert load_graph(panel_facility, tables).content_hash != panel_graph.content_hash


def test_a_rule_channel_no_roll_up_reads_is_unrolled(
    panel_facility, panel_tables, tmp_path, monkeypatch
):
    import importlib

    describe_module = importlib.import_module("osprey_panels.describe")

    clean = load_graph(panel_facility, panel_tables)
    emit.write(clean, tmp_path / "clean")
    assert not [p for p in emit.check(clean, tmp_path / "clean") if p.kind == "unrolled"]
    monkeypatch.setattr(describe_module, "unplaced_channels", lambda graph: [])
    blind = load_graph(panel_facility, panel_tables)
    emit.write(blind, tmp_path / "blind")
    unrolled = [p for p in emit.check(blind, tmp_path / "blind") if p.kind == "unrolled"]
    assert [p.detail.split()[0] for p in unrolled] == ["XR09-LOOSE1:Fault-Sts"]
    assert all(p.failing for p in unrolled)


def _strip_rules(path: Path) -> None:
    doc = json.loads(path.read_text())
    for node in _walk(doc):
        node.pop("rules", None)
    path.write_text(json.dumps(doc))


def test_a_roll_up_with_no_live_light_is_static_only_when_a_child_has_no_screen(
    panel_graph, tmp_path
):
    out = tmp_path / "out"
    write(panel_graph, out)
    name = "rollup/rollup-place-xr.screen.json"
    _strip_rules(out / name)

    def kinds() -> set[str]:
        return {p.kind for p in check(panel_graph, out) if p.file == name}

    here = json.loads((out / name).read_text())
    refs = {
        n["props"]["actions"][0]["file"] for n in _walk(here) if n.get("props", {}).get("actions")
    }
    for ref in refs:  # a drill-down: its children open screens, live or not
        _strip_rules((out / name).parent / ref)
    assert "static_rollup" not in kinds()
    child = next(r for r in sorted(refs) if "rollup-place-xr-" in r)
    ((out / name).parent / child).unlink()
    found = [p for p in check(panel_graph, out) if p.file == name]
    static = [p for p in found if p.kind == "static_rollup"]
    assert static and "no screen to open" in static[0].detail


def test_a_roll_up_or_rack_over_its_budget_fails_and_a_card_over_it_is_noted(tmp_path, monkeypatch):
    from osprey_panels import layout

    panel_graph = graph_with_part(tmp_path / "part")  # the rack has a light

    out = tmp_path / "out"
    write(panel_graph, out)
    assert not [p for p in check(panel_graph, out) if p.kind == "over_budget"]
    monkeypatch.setattr(layout, "SCREEN_RULE_BUDGET", 1)
    monkeypatch.setattr(layout, "SCREEN_CHANNEL_BUDGET", 1)
    over = [p for p in check(panel_graph, out) if p.kind == "over_budget"]
    assert {p.file.split("/")[0] for p in over} >= {"card", "rollup", "rack"}
    for problem in over:  # roll-ups and racks must keep to it; cards and checkouts are noted
        assert problem.failing == (problem.file.split("/")[0] not in {"card", "checkout"})
    assert any("rules subscribe" in p.detail for p in over)
    assert any("channels bound" in p.detail for p in over)


def test_no_fixture_roll_up_is_static_even_over_the_rule_limit(tmp_path, monkeypatch):
    from osprey_panels import layout
    from osprey_panels.describe import light

    panel_graph = graph_with_wide(tmp_path / "wide", 8)
    # a long node is open: a drill-down to its own screen, which has the lights
    singles = [
        layout.ladder(panel_graph, [light(panel_graph, r.channel)]) for r in panel_graph.rules()
    ]
    limit = max(layout.engine_length(rule) for rule in singles if not isinstance(rule, str))
    monkeypatch.setattr(layout, "RULE_EXPRESSION_LIMIT", limit)
    out = tmp_path / "out"
    write(panel_graph, out)
    found = check(panel_graph, out)
    assert not [p for p in found if p.kind in ("static_rollup", "unrolled")]


def test_a_rule_the_engine_would_refuse_as_too_long_is_a_long_formula(tmp_path, panel_graph):
    out = tmp_path / "screens"
    emit.write(panel_graph, out)
    assert not [p for p in emit.check(panel_graph, out) if p.kind == "long_formula"]
    victim = next(f for f in sorted(out.rglob("*.screen.json")) if '"conditions"' in f.read_text())
    doc = json.loads(victim.read_text())
    ruled = next(n for n in _walk(doc) if n.get("rules") and "conditions" in n["rules"][0])
    ruled["rules"][0]["conditions"][0]["expression"] = " or ".join(["{0} == 1"] * 300)
    victim.write_text(json.dumps(doc))
    found = [p for p in emit.check(panel_graph, out) if p.kind == "long_formula"]
    assert found and all(p.file.endswith(victim.name) for p in found)


def test_a_part_of_an_excluded_thing_has_no_screen_link_or_bound_channel(tmp_path) -> None:
    from osprey_panels_tests.test_model import _withheld

    graph = _withheld(tmp_path)
    out = tmp_path / "out"
    write(graph, out)
    text = "".join(p.read_text() for p in out.rglob("*.json"))
    assert "SAF1:CARD" not in text and "SAF1-CARD" not in text
    assert not [p for p in check(graph, out) if "SAF1" in p.detail]


def test_a_channel_a_roll_up_row_opens_is_not_unrolled(panel_graph, tmp_path, monkeypatch):
    from osprey_panels import layout

    monkeypatch.setattr(layout, "SCREEN_RULE_BUDGET", 0)  # every row opens its screen
    out = tmp_path / "out"
    write(panel_graph, out)
    found = check(panel_graph, out)
    assert not [p for p in found if p.kind == "unrolled"]
    assert not [p for p in found if p.kind == "static_rollup"]  # every child opens a screen


def test_under_a_place_writes_the_calibration_views_of_sets_with_a_member_there(
    panel_graph, tmp_path
):
    out = tmp_path / "out"
    write(panel_graph, out, only=under(panel_graph, "XR/XR01"))
    kinds = {e["kind"] for e in json.loads((out / "index.json").read_text())["screens"]}
    assert "calibration" in kinds  # XR-QUAD has members in XR/XR01
    test = under(panel_graph, "XR/XR01")
    assert not [t for t in targets(panel_graph) if t.view == "set" and test(t)]
    rack = under(panel_graph, "XR/XR01/RK1")  # no magnet in the rack
    assert not [t for t in targets(panel_graph) if t.view == "calibration" and rack(t)]


def test_a_device_whose_card_lists_counts_has_a_paged_roll_up_that_reads_each_rule_once(
    tmp_path, monkeypatch
):
    import importlib

    from osprey_panels.describe import device_rollup
    from osprey_panels.view import Node

    describe_module = importlib.import_module("osprey_panels.describe")
    monkeypatch.setattr(describe_module, "DEVICE_PAGE", 50)
    graph = graph_with_wide(tmp_path / "wide", 120)  # over the card's 60: counts only
    pages = [t for t in targets(graph) if t.view == "rollup" and t.kind == "device"]
    assert [t.system for t in pages] == [None, "2", "3"]  # 120 rules, 50 a page
    read = [
        channel
        for target in pages
        for section in device_rollup(graph, "XR01:WIDE1", int(target.system or 1)).sections
        for item in section.items
        if isinstance(item, Node)
        for channel in item.channels
    ]
    assert sorted(read) == sorted(f"XR01-WIDE1:Sig{i}-Sts" for i in range(120))  # once each
    emit.write(graph, tmp_path / "out")
    problems = emit.check(graph, tmp_path / "out")
    assert not [p for p in problems if p.kind in ("unrolled", "over_budget", "missing")]


def test_a_scoped_home_lists_only_the_scope_and_says_so(tmp_path):
    from osprey_panels.describe import home
    from osprey_panels.emit import scoped, write_scope
    from osprey_panels.view import Fact, Group

    graph = graph_with_part(tmp_path)
    view = home(scoped(graph, write_scope(graph, ("XR/XR01",))))
    items = [i for s in view.sections for i in s.items]
    (said,) = [i for i in items if isinstance(i, Fact)]
    assert said.text == "keeps to XR/XR01; the screens outside it are not written"
    chips = {n.label for i in items if isinstance(i, Group) for n in i.nodes}
    assert "XR/XR01" in chips and "XR" not in chips  # the kept place stands at the top


def test_a_scope_cuts_a_group_to_its_kept_members_and_writes_a_duplicate_once(tmp_path):
    from osprey_panels.describe import set_overview
    from osprey_panels.emit import scoped, write_scope

    def per_sector(document) -> None:  # the facility's own set of XR01's quadrupoles
        document["groups"].append(
            {"id": "XR01-QUAD", "kind": "family", "members": ["XR01:QF1", "XR01:QD1"]}
        )

    graph = graph_with_part(tmp_path, per_sector)
    scope = write_scope(graph, ("XR/XR01",))
    # XR-QUAD cut to XR01 holds what XR01-QUAD holds: the narrower one is written
    assert "XR01-QUAD" in scope.groups and "XR-QUAD" not in scope.groups
    only = write_scope(graph_with_part(tmp_path / "b"), ("XR/XR01",))
    assert set(only.members["XR-QUAD"]) == {"XR01:QF1", "XR01:QD1"}  # cut, no duplicate
    view = set_overview(scoped(graph_with_part(tmp_path / "c"), only), "XR-QUAD")
    (grid,) = [i for s in view.sections if s.title == "members" for i in s.items]
    assert {c.member.label for row in grid.rows for c in row.cells} == {"XR01:QF1", "XR01:QD1"}


def test_no_written_screen_links_to_a_duplicate_the_scope_writes_once(tmp_path):
    """The dropped duplicate is no set under the scope: every screen that named it
    names the kept one, so no link leaves the write for it."""
    import posixpath

    from osprey_panels.emit import Draw, _links

    def per_sector(document) -> None:  # the facility's own set of XR01's quadrupoles
        document["groups"].append(
            {"id": "XR01-QUAD", "kind": "family", "members": ["XR01:QF1", "XR01:QD1"]}
        )

    graph = graph_with_part(tmp_path, per_sector)
    out = tmp_path / "out"
    draw = Draw(write_under=("XR/XR01",))
    write(graph, out, draw=draw)
    index = json.loads((out / "index.json").read_text())
    files = {e["file"] for e in index["screens"]}
    dropped = {e["file"] for e in index.get("outside", ()) if e["thing"] == "XR-QUAD"}
    linked = set()
    for name in files:
        if not name.endswith(".json"):
            continue
        here = posixpath.dirname(name)
        for ref in _links(json.loads((out / name).read_text())):
            linked.add(posixpath.normpath(posixpath.join(here, ref)))
    assert not linked & dropped
    assert any("xr01-quad" in name for name in linked)  # the kept one is reached


def test_a_scoped_write_names_what_it_does_not_write_and_links_only_to_its_files(tmp_path):
    """A written screen holds no link to a file the same write does not produce: the
    place above the scope is named as text, and check notes each such name."""
    import posixpath

    from osprey_panels.emit import Draw, _links
    from osprey_panels_tests._views import nodes

    graph = graph_with_part(tmp_path)
    out = tmp_path / "out"
    draw = Draw(write_under=("XR/XR01",))
    write(graph, out, draw=draw)
    index = json.loads((out / "index.json").read_text())
    files = {e["file"] for e in index["screens"]}
    for name in files:
        if name.endswith(".json"):
            here = posixpath.dirname(name)
            refs = _links(json.loads((out / name).read_text()))
            assert {posixpath.normpath(posixpath.join(here, r)) for r in refs} <= files, name
    card = json.loads((out / "card/card-xr01-bend1.screen.json").read_text())
    named = [n for n in nodes(card) if n["props"].get("text") == "XR"]
    assert named and not [n for n in nodes(card) if n["props"].get("tooltip") == "XR"]
    notes = [p for p in check(graph, out, draw=draw) if p.kind == "outside_set"]
    assert notes and all(p.detail.startswith("names ") for p in notes)
    assert any(
        p.file == "card/card-xr01-bend1.screen.json" and "place XR" in p.detail for p in notes
    )


def test_a_paged_roll_up_s_header_ends_its_way_out_without_a_mark(tmp_path, monkeypatch):
    """The device names its own roll-up as the last step of the way out, so the
    header does not name it again; the mark after that step and the join before
    the page text go with the name, and a mark stands only between two steps."""
    import importlib

    from osprey_panels import layout
    from osprey_panels.describe import device_rollup
    from osprey_panels_tests._views import stub_render

    describe_module = importlib.import_module("osprey_panels.describe")
    monkeypatch.setattr(describe_module, "DEVICE_PAGE", 50)
    graph = graph_with_wide(tmp_path / "wide", 120)
    view = device_rollup(graph, "XR01:WIDE1", 2)
    doc = stub_render(view, graph)
    header = sorted(
        (
            n
            for n in doc["root"]["children"]
            if n["geometry"]["y"] + n["geometry"]["height"] <= layout.HEADER_HEIGHT
            and n["type"] != "rectangle"
        ),
        key=lambda n: n["geometry"]["x"],
    )
    words = [str(n["props"].get("text") or n["props"].get("label") or "") for n in header]
    words = [w for w in words if w and not w.startswith("model ")]  # the stamp
    assert words[-1] == "page 2 of 3 health", words  # the page text, no join mark before it
    assert words[-2] == "XR01:WIDE1"  # the device, once: its card's step
    marks = [i for i, w in enumerate(words) if w == "›"]
    assert marks and all(0 < i and words[i + 1] not in ("›", "page 2 of 3 health") for i in marks)
    assert not [w for w in words if w.startswith("·")]


# --------------------------------------------------------------------------- theme input

#: The demo theme, read by its path in the checkout (a file, not an import).
DEMO_THEME = REPO_ROOT / "src/osprey/templates/apps/demo/data/canopy/theme/osprey.json"
#: The demo theme's MINOR light colour: far from the fallback MINOR shades.
THEME_ONLY_MINOR = "#b5710d"


def _draw_file(tmp_path: Path, colour: str) -> Path:
    path = tmp_path / "draw.json"
    path.write_text(json.dumps({"colours": {"quadrupole": colour}}))
    return path


def test_load_theme_returns_the_roles_of_the_theme() -> None:
    roles = emit.load_theme(DEMO_THEME)
    assert roles == json.loads(DEMO_THEME.read_text(encoding="utf-8"))["roles"]
    assert roles["MINOR"]["light"] == THEME_ONLY_MINOR


def test_a_colour_near_a_theme_state_colour_is_refused_with_the_theme(tmp_path: Path) -> None:
    path = _draw_file(tmp_path, THEME_ONLY_MINOR)
    with pytest.raises(EmitError, match="too near the state colour MINOR"):
        emit.load_draw(path, theme_roles=emit.load_theme(DEMO_THEME))


def test_without_a_theme_only_the_fallback_state_colours_are_the_reference(
    tmp_path: Path,
) -> None:
    path = _draw_file(tmp_path, THEME_ONLY_MINOR)
    assert emit.load_draw(path).colours == {"quadrupole": THEME_ONLY_MINOR}
    with pytest.raises(EmitError, match="too near the state colour MAJOR"):
        emit.load_draw(_draw_file(tmp_path, "#c4162a"))


def test_the_engine_names_no_theme_file_of_its_own() -> None:
    assert not hasattr(emit, "THEME_FILE")
    assert not hasattr(emit, "_theme_roles")


@pytest.mark.parametrize(
    "content",
    [
        None,  # missing
        "{not json",
        "[]",
        '{"id": "x"}',
        '{"roles": []}',
        '{"roles": {"MINOR": "#b5710d"}}',
        '{"roles": {"MINOR": {"light": 7}}}',
    ],
    ids=["missing", "not-json", "not-object", "no-roles", "roles-list", "role-str", "shade-int"],
)
def test_an_unreadable_or_malformed_theme_raises_naming_the_path(
    tmp_path: Path, content: str | None
) -> None:
    path = tmp_path / "theme.json"
    if content is not None:
        path.write_text(content)
    with pytest.raises(EmitError, match=re.escape(str(path))):
        emit.load_theme(path)
