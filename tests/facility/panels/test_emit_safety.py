"""What ``write`` may touch, and what an interrupted run leaves behind."""

from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path

import pytest

from osprey.facility.panels import emit, targets
from osprey.facility.panels.emit import Draw, EmitError, check, screen_ids, write
from tests.facility.panels.test_emit import _drop, _files, _graph, _kinds, _walk


def _smaller(tmp_path: Path, panel_tables: Path):
    return _graph(tmp_path, panel_tables, _drop("XR01:PS1"), "smaller")


# -- links and paths -------------------------------------------------------------------


def test_a_symlinked_view_directory_is_refused_and_nothing_outside_is_touched(
    panel_graph, panel_tables, tmp_path
):
    out, outside = tmp_path / "out", tmp_path / "external"
    write(panel_graph, out)
    shutil.move(out / "card", outside)
    (out / "card").symlink_to(outside, target_is_directory=True)
    before = {n: (outside / n).read_bytes() for n in map(str, (p.name for p in outside.iterdir()))}
    with pytest.raises(EmitError, match="card"):
        write(_smaller(tmp_path, panel_tables), out)
    assert {p.name: p.read_bytes() for p in outside.iterdir()} == before
    assert (outside / "card-xr01-ps1.screen.json").is_file()


def test_a_symlinked_file_is_refused(panel_graph, tmp_path):
    out = tmp_path / "out"
    write(panel_graph, out)
    secret = tmp_path / "secret.txt"
    secret.write_text("keep")
    victim = sorted((out / "card").glob("*.screen.json"))[0]
    victim.unlink()
    victim.symlink_to(secret)
    with pytest.raises(EmitError, match=victim.name):
        write(panel_graph, out)
    assert secret.read_text() == "keep"


def test_out_itself_may_be_a_symlink(panel_graph, tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real, target_is_directory=True)
    write(panel_graph, tmp_path / "link")
    assert check(panel_graph, tmp_path / "link") == []


def test_a_symlinked_view_directory_is_a_problem_for_check(panel_graph, tmp_path):
    out, outside = tmp_path / "out", tmp_path / "external"
    write(panel_graph, out)
    shutil.move(out / "card", outside)
    (out / "card").symlink_to(outside, target_is_directory=True)
    assert "invalid" in _kinds(check(panel_graph, out))


# -- an index is untrusted -----------------------------------------------------------


def test_a_tampered_index_deletes_nothing_of_a_wrong_shape(panel_graph, tmp_path):
    out = tmp_path / "out"
    write(panel_graph, out)
    (out / "mine.txt").write_text("1")
    (out / "card" / "notes.txt").write_text("2")
    (out / "card" / "UPPER.screen.json").write_text("3")
    (out / "other").mkdir()
    (out / "other" / "x.screen.json").write_text("4")
    (tmp_path / "x").write_text("5")
    absolute = tmp_path / "abs.screen.json"
    absolute.write_text("6")
    index = json.loads((out / "index.json").read_text())
    for name in (
        "mine.txt",
        "card/notes.txt",
        "card/UPPER.screen.json",
        "other/x.screen.json",
        "card/../../x",
        str(absolute),
        "card/../mine.txt",
    ):
        index["screens"].append({"file": name})
    (out / "index.json").write_text(json.dumps(index))
    with pytest.warns(UserWarning, match="mine.txt"):
        write(panel_graph, out)
    assert (out / "mine.txt").read_text() == "1"
    assert (out / "card" / "notes.txt").read_text() == "2"
    assert (out / "card" / "UPPER.screen.json").read_text() == "3"
    assert (out / "other" / "x.screen.json").read_text() == "4"
    assert (tmp_path / "x").read_text() == "5" and absolute.read_text() == "6"


@pytest.mark.parametrize(
    "content", ["{not json", "[]", '{"screens": 3}', '{"screens": [1, 2]}'], ids=str
)
def test_a_corrupt_index_is_an_error_for_write_and_invalid_for_check(
    panel_graph, tmp_path, content
):
    out = tmp_path / "out"
    write(panel_graph, out)
    (out / "index.json").write_text(content)
    with pytest.raises(EmitError, match="index.json"):
        write(panel_graph, out)
    assert [(p.kind, p.file) for p in check(panel_graph, out)] == [("invalid", "index.json")]


def test_an_index_that_is_a_directory_is_an_error(panel_graph, tmp_path):
    out = tmp_path / "out"
    (out / "index.json").mkdir(parents=True)
    with pytest.raises(EmitError, match="index.json"):
        write(panel_graph, out)
    assert [p.kind for p in check(panel_graph, out)] == ["invalid"]


# -- an interrupted run converges ------------------------------------------------------


class _Crash(Exception):
    pass


def _crash_after(monkeypatch, name: str, calls: int) -> None:
    real = getattr(emit, name)
    seen = {"n": 0}

    def flaky(*args, **kwargs):
        seen["n"] += 1
        if seen["n"] > calls:
            raise _Crash
        return real(*args, **kwargs)

    monkeypatch.setattr(emit, name, flaky)


def test_a_first_run_killed_after_five_files_converges_on_rerun(panel_graph, tmp_path, monkeypatch):
    out = tmp_path / "out"
    with monkeypatch.context() as patched:
        _crash_after(patched, "_atomic", 5)
        with pytest.raises(_Crash):
            write(panel_graph, out)
    assert "incomplete" in _kinds(check(panel_graph, out))
    write(panel_graph, out)
    assert check(panel_graph, out) == []
    write(panel_graph, tmp_path / "clean")
    assert _files(out) == _files(tmp_path / "clean")


def test_a_run_killed_before_its_deletions_leaks_nothing_after_a_rerun(
    panel_graph, panel_tables, tmp_path, monkeypatch
):
    out = tmp_path / "out"
    write(panel_graph, out)
    smaller = _smaller(tmp_path, panel_tables)
    with monkeypatch.context() as patched:
        _crash_after(patched, "_remove", 0)
        with pytest.raises(_Crash):
            write(smaller, out)
    assert "incomplete" in _kinds(check(smaller, out))
    write(smaller, out)
    assert check(smaller, out) == []
    write(smaller, tmp_path / "clean")
    assert _files(out) == _files(tmp_path / "clean")


def test_a_complete_index_says_so(panel_graph, tmp_path):
    write(panel_graph, tmp_path / "out")
    assert json.loads((tmp_path / "out" / "index.json").read_text())["state"] == "complete"


def test_leftover_temporary_files_of_ours_are_removed_and_others_kept(panel_graph, tmp_path):
    out = tmp_path / "out"
    write(panel_graph, out)
    ours = [
        out / "card" / ".card-xr01-qf1.screen.json.tmp",
        out / "card" / ".card-xr01-qf1.screen.json.k3_x9q2a.tmp",
        out / ".index.json.tmp",
        out / ".index.json.p0_zz81w.tmp",
    ]
    for path in ours:
        path.write_text("half")
    theirs = out / "card" / ".mine.tmp"
    theirs.write_text("x")
    write(panel_graph, out)
    assert not any(path.exists() for path in ours)
    assert theirs.read_text() == "x"


def test_a_file_of_our_shape_the_index_does_not_list_is_an_orphan(panel_graph, tmp_path):
    out = tmp_path / "out"
    write(panel_graph, out)
    (out / "card" / "card-ghost.screen.json").write_text("{}")
    (out / "card" / "notes.txt").write_text("foreign")
    problems = check(panel_graph, out)
    assert [(p.kind, p.file) for p in problems] == [("orphan", "card/card-ghost.screen.json")]


# -- the draw settings -----------------------------------------------------------------


def test_a_change_of_the_draw_settings_is_named_as_a_stamp_problem(panel_graph, tmp_path):
    out = tmp_path / "out"
    write(panel_graph, out, draw=Draw(symbols={"Quadrupole": "quadrupole"}))
    assert check(panel_graph, out, draw=Draw(symbols={"Quadrupole": "quadrupole"})) == []
    problems = check(panel_graph, out)
    assert "stamp" in _kinds(problems)
    assert any(p.file == "index.json" and "draw" in p.detail for p in problems)


def test_the_draw_digest_does_not_depend_on_order():
    one = Draw(symbols={"A": "bend", "B": "bpm"}, above=frozenset({"bend", "bpm"}))
    two = Draw(symbols={"B": "bpm", "A": "bend"}, above=frozenset({"bpm", "bend"}))
    assert one.digest == two.digest != Draw().digest


def _draw_file(tmp_path: Path, content) -> Path:
    path = tmp_path / "draw.json"
    path.write_text(content if isinstance(content, str) else json.dumps(content))
    return path


def test_a_draw_file_gives_the_settings(tmp_path):
    path = _draw_file(
        tmp_path,
        {
            "rack_classes": ["rack"],
            "symbols": {"Quadrupole": "quadrupole", "Pump": "ion_pump"},
            "above": ["Pump"],
        },
    )
    draw = emit.load_draw(path)
    assert draw == Draw(
        rack_classes=frozenset({"rack"}),
        symbols={"Quadrupole": "quadrupole", "Pump": "ion_pump"},
        above=frozenset({"ion_pump"}),
    )


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ("{nope", "draw.json"),
        ("[]", "object"),
        ({"racks": []}, "racks"),
        ({"rack_classes": "rack"}, "rack_classes"),
        ({"rack_classes": [1]}, "rack_classes"),
        ({"symbols": {"A": 3}}, "symbols"),
        ({"symbols": {"A": "no_such_symbol"}}, "no_such_symbol"),
        ({"symbols": {"A": "bend"}, "above": ["B"]}, "B"),
        ({"above": "A"}, "above"),
    ],
    ids=str,
)
def test_a_bad_draw_file_is_a_clear_error(tmp_path, content, match):
    with pytest.raises(EmitError, match=match):
        emit.load_draw(_draw_file(tmp_path, content))


def test_a_missing_draw_file_is_an_error_when_named(tmp_path):
    with pytest.raises(EmitError, match="nowhere"):
        emit.load_draw(tmp_path / "nowhere.json")


# -- bound channels ----------------------------------------------------------------------


def test_a_simulation_only_channel_is_a_failing_problem(panel_graph, tmp_path):
    out = tmp_path / "out"
    write(panel_graph, out)
    victim = next(p for p in sorted(out.glob("card/*.json")) if '"pv":' in p.read_text())
    doc = json.loads(victim.read_text())
    next(n for n in _walk(doc) if isinstance(n.get("pv"), str))["pv"] = "VA:XR:InjectionKnob"
    victim.write_text(json.dumps(doc))
    found = [p for p in check(panel_graph, out) if p.kind == "unknown_channel"]
    assert found and all(p.failing and "simulation" in p.detail for p in found)


# -- the pre-flight pass ---------------------------------------------------------------


def _snapshot(out: Path) -> dict[str, tuple[bytes, int]]:
    return {
        p.relative_to(out).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in sorted(out.rglob("*"))
        if p.is_file() and not p.is_symlink()
    }


@pytest.mark.parametrize("dangling", [False, True], ids=["external-file", "dangling"])
def test_a_linked_file_stops_the_run_before_anything_is_touched(panel_graph, tmp_path, dangling):
    out = tmp_path / "out"
    write(panel_graph, out)
    victim = sorted((out / "rollup").glob("*.screen.json"))[-1]
    outside = tmp_path / "outside.txt"
    if not dangling:
        outside.write_text("keep")
    victim.unlink()
    victim.symlink_to(outside)
    before = _snapshot(out)
    with pytest.raises(EmitError, match=victim.name):
        write(panel_graph, out)
    assert _snapshot(out) == before
    assert json.loads((out / "index.json").read_text())["state"] == "complete"
    assert not outside.exists() or outside.read_text() == "keep"


def test_a_listed_name_too_long_for_the_file_system_is_ignored_not_carried(panel_graph, tmp_path):
    out = tmp_path / "out"
    write(panel_graph, out)
    index = json.loads((out / "index.json").read_text())
    long_name = f"card/{'a' * 300}.screen.json"
    index["screens"].append({"file": long_name})
    (out / "index.json").write_text(json.dumps(index))
    with pytest.warns(UserWarning, match="left alone"):
        write(panel_graph, out)
    assert long_name not in (out / "index.json").read_text()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        write(panel_graph, out)
    assert check(panel_graph, out) == []


def test_no_name_a_run_can_make_is_longer_than_the_shape_allows(panel_graph):
    longest = max(len(i) for i in screen_ids(targets(panel_graph)).values())
    assert longest <= emit.ID_MAX
    big = [emit.Target("rollup", "place", "Q" * 500, "system")]
    assert len(screen_ids(big)[big[0]]) <= emit.ID_MAX
    assert emit._FILE_SHAPE.fullmatch(f"rollup/{'a' * emit.ID_MAX}.screen.json")
    assert not emit._FILE_SHAPE.fullmatch(f"rollup/{'a' * (emit.ID_MAX + 1)}.screen.json")


@pytest.mark.parametrize(
    ("content", "says"),
    [
        ({"colours": {"bend": "blue"}}, "not #rrggbb"),
        ({"colours": {"wiggler": "#4c78d8"}}, "not a symbol of the engine"),
        ({"colours": {"bend": "#e05454"}}, "too near the state colour MAJOR"),
        ({"colours": {"bend": "#f5b73d"}}, "too near the state colour MINOR"),
        ({"colours": {"bend": "#c39bd3"}}, "too near the state colour Disagree"),
        ({"annotate": {"bpm": ["a", "b", "c"]}}, "more than 2"),
        ({"annotate": {"wiggler": ["a"]}}, "not a symbol of the engine"),
    ],
)
def test_draw_colours_and_annotations_are_checked(tmp_path, content, says) -> None:
    path = tmp_path / "draw.json"
    path.write_text(json.dumps(content))
    with pytest.raises(emit.EmitError, match=says):
        emit.load_draw(path)


def test_colours_and_annotations_are_part_of_the_draw_digest(tmp_path) -> None:
    path = tmp_path / "draw.json"
    digests = set()
    for content in (
        {},
        {"colours": {"bend": "#4c78d8"}},
        {"colours": {"bend": "#2f9e52"}},
        {"annotate": {"bpm": ["position_x_readback"]}},
    ):
        path.write_text(json.dumps(content))
        digests.add(emit.load_draw(path).digest)
    assert len(digests) == 4


def test_the_fixture_colours_stand_out_on_both_themes_and_from_every_state() -> None:
    from osprey.facility.panels import symbols
    from tests.facility.panels._fixture import PANELS_DIR
    from tests.facility.panels.test_symbols import _resolved, contrast

    draw = emit.load_draw(PANELS_DIR / "draw.json")
    assert draw.colours
    for colour in draw.colours.values():
        for theme in ("light", "dark"):
            for ground in (symbols.PAGE_BACKGROUND, symbols.PANEL_BACKGROUND):
                assert contrast(colour, _resolved(ground, theme)) >= 3, (colour, theme)


def test_a_ring_setting_is_refused(tmp_path) -> None:
    path = tmp_path / "draw.json"
    path.write_text(json.dumps({"ring": ["XR"]}))  # a ring is read from the facility file
    with pytest.raises(EmitError, match="ring"):
        emit.load_draw(path)


def test_a_ring_roll_up_is_written_and_checks_clean(ring_graph, tmp_path) -> None:
    out = tmp_path / "out"
    write(ring_graph, out)
    assert not [p for p in check(ring_graph, out) if p.failing]
    assert '"XR01"' in (out / "rollup/rollup-place-xr.screen.json").read_text()
