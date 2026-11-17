"""Writing the screens to disk and checking a written set against the model."""

from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pytest

from osprey.facility.panels import emit, load_graph, targets
from osprey.facility.panels.emit import (
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
from osprey.facility.panels.model import PanelGraph
from osprey.facility.panels.view import Target
from tests.facility.panels._fixture import write_fixture_facility


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
    assert gone and all(name.startswith("card/") for name in gone)
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
            if isinstance(node.get("pv"), str):
                bound.add(node["pv"])
            for pv in node.get("pvs") or ():
                bound.add(pv["name"])
    assert bound
    for name in bound:
        panel_graph.channel(name)


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
    from tests.facility.panels.test_describe import _big_graph as build

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
    from tests.facility.panels._fixture import copy_tables

    tables = copy_tables(tmp_path / "panels")
    with (tables / "health.csv").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    assert load_graph(panel_facility, tables).content_hash != panel_graph.content_hash
