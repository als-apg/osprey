"""Tests for the UserPromptSubmit workspace-delta hook.

``osprey_workspace_delta.py`` diffs the live web workspace against the snapshot
seeded by the SessionStart panels-context hook and injects one line naming what
moved — tiles opened and closed, tabs opened, moved and closed, the tab a tile
shows, the active tab and launcher-rail membership. Its defining property is silence: an unchanged workspace, an unreadable
session id, a missing or corrupt snapshot and a down web terminal must each exit
0 with *byte-empty* stdout, because anything else spends tokens on every turn
for no information.

These tests cover the diff/describe helpers as pure functions and the ``main``
envelope with a stubbed ``urlopen`` (no real HTTP) and a redirected temp dir.
"""

from __future__ import annotations

import io
import json

import pytest

from tests.hooks.conftest import import_hook

delta = import_hook("osprey_workspace_delta")


class _FakeResponse:
    def __init__(self, payload):
        self._data = json.dumps(payload).encode()

    def read(self):
        return self._data


def _stub_urlopen(monkeypatch, *, payload=None, raises=None, bad_body=False):
    """Point the hook's urlopen at a fake response, an error, or garbage body."""

    def _fake(url, timeout=None):  # noqa: ARG001 - urlopen takes the url positionally and timeout by keyword
        if raises is not None:
            raise raises
        if bad_body:

            class _Bad:
                def read(self):
                    return b"not json{{{"

            return _Bad()
        return _FakeResponse(payload or {})

    monkeypatch.setattr(delta.urllib.request, "urlopen", _fake)


def _stub_stdin(monkeypatch, payload):
    """Hand ``main`` a hook stdin payload (a str is written through verbatim)."""
    raw = payload if isinstance(payload, str) else json.dumps(payload)
    monkeypatch.setattr(delta.sys, "stdin", io.StringIO(raw))


def _tile(spec, index):
    """One snapshot tile from a test spec.

    A str is a one-tab tile; a list is a tile of several tabs, the one ending in
    ``*`` showing (else the first); a dict is taken as written. Ids are
    positional (``t1``, ``t2``, ...) unless the dict names one.
    """
    if isinstance(spec, dict):
        return spec
    tabs = [spec] if isinstance(spec, str) else list(spec)
    showing = next((t for t in tabs if t.endswith("*")), tabs[0]).rstrip("*")
    return {"tile": f"t{index + 1}", "tabs": [t.rstrip("*") for t in tabs], "showing": showing}


def _tiles(specs):
    return None if specs is None else [_tile(spec, i) for i, spec in enumerate(specs)]


def _state(tiles=None, active=None, visible=(), popouts=None):
    flat = _tiles(tiles)
    if popouts is None:
        popouts = None if flat is None else []
    return {"tiles": flat, "popouts": popouts, "active": active, "visible": list(visible)}


def _layout(tiles):
    """The dock layout the server publishes for snapshot ``tiles`` (a row of them)."""
    nodes = [dict(tile, size=1 / len(tiles)) for tile in tiles]
    return nodes[0] if len(nodes) == 1 else {"split": "row", "items": nodes}


def _panels_payload(tiles=None, active=None, visible=(), age=0.5):
    """A ``GET /api/panels`` body; ``tiles=None`` means no dock has reported."""
    flat = _tiles(tiles)
    return {
        "layout": _layout(flat) if flat else None,
        "rev": 1 if flat else 0,
        "layout_age_s": age if tiles is not None else None,
        "layout_dock": True if tiles is not None else None,
        "active": active,
        "visible": list(visible),
    }


def _line(before, after):
    return delta._build_delta_line(_state(*before), _state(*after))


@pytest.fixture
def temp_snapshot_dir(tmp_path, monkeypatch):
    """Redirect ``tempfile.gettempdir()`` so snapshots land in ``tmp_path``."""
    monkeypatch.setattr(delta.tempfile, "tempdir", str(tmp_path))
    return tmp_path


@pytest.fixture
def snapshot_file(temp_snapshot_dir):
    """Read/write access to the snapshot ``main`` will use for session ``s1``."""

    class _Snapshot:
        path = temp_snapshot_dir / "osprey-workspace-s1.json"

        def write(self, state):
            self.path.write_text(json.dumps(state))

        def read(self):
            return json.loads(self.path.read_text())

        def exists(self):
            return self.path.exists()

    return _Snapshot()


# ---------------------------------------------------------------------------
# Session id + snapshot IO (pure)
# ---------------------------------------------------------------------------


def test_session_id_read_from_stdin_payload():
    stream = io.StringIO(json.dumps({"session_id": "s1", "prompt": "hi"}))
    assert delta._read_session_id(stream) == "s1"


@pytest.mark.parametrize(
    "raw",
    ["", "{not json", "[]", "{}", '{"session_id": null}', '{"session_id": "../escape"}'],
    ids=["empty", "invalid", "list", "no-key", "null", "traversal"],
)
def test_session_id_absent_or_unsafe_returns_none(raw):
    assert delta._read_session_id(io.StringIO(raw)) is None


def test_snapshot_path_matches_sibling_hook(temp_snapshot_dir):
    """Both hooks must name the same file or the delta line never fires.

    The seeding hook owns the schema; this asserts the two independent copies of
    the path rule have not drifted apart.
    """
    panels = import_hook("osprey_panels_context")

    assert delta._snapshot_path("s1") == str(temp_snapshot_dir / "osprey-workspace-s1.json")
    assert delta._snapshot_path("s1") == panels._snapshot_path("s1")


def test_load_snapshot_reads_the_three_facets(snapshot_file):
    snapshot_file.write(_state(["terminal", ["a", "b*"]], "a", ["a", "b"]))
    assert delta._load_snapshot(str(snapshot_file.path)) == _state(
        ["terminal", ["a", "b*"]], "a", ["a", "b"]
    )


@pytest.mark.parametrize(
    "raw",
    ["", "{half-writ", "[]", '{"tiles": ["a"]}'],
    ids=["empty", "torn", "wrong-type", "missing-keys"],
)
def test_load_snapshot_rejects_untrustworthy_content(snapshot_file, raw):
    snapshot_file.path.write_text(raw)
    assert delta._load_snapshot(str(snapshot_file.path)) is None


def test_load_snapshot_missing_file_is_none(temp_snapshot_dir):
    assert delta._load_snapshot(str(temp_snapshot_dir / "nope.json")) is None


def test_write_snapshot_leaves_no_temp_file(temp_snapshot_dir):
    path = delta._snapshot_path("s1")
    assert delta._write_snapshot(path, _state(["a"], "a", ["a"])) is True
    assert [p.name for p in temp_snapshot_dir.iterdir()] == ["osprey-workspace-s1.json"]


def test_write_snapshot_returns_false_when_unwritable(tmp_path):
    assert delta._write_snapshot(str(tmp_path / "no-such-dir" / "s.json"), {}) is False


def test_write_snapshot_writes_nothing_without_the_file_helper(temp_snapshot_dir, monkeypatch):
    monkeypatch.setattr(delta, "_safety_file", None)
    assert delta._write_snapshot(delta._snapshot_path("s1"), _state(["a"], "a", ["a"])) is False
    assert list(temp_snapshot_dir.iterdir()) == []


def test_workspace_state_null_layout_means_tiles_unknown():
    state = delta._workspace_state(_panels_payload(tiles=None, active="a", visible=["a"]))
    assert state == _state(None, "a", ["a"])


#: The payload shapes that mean "a client reported, but it has no dock": the
#: server records the layout as unknown, and even a stale layout left beside a
#: ``dock: false`` flag describes nothing on screen.
DOCKLESS_PAYLOADS = [
    {"layout": None, "layout_age_s": 0.3, "layout_dock": False},
    {"layout": _layout(_tiles(["a"])), "layout_age_s": 0.3, "layout_dock": False},
]
DOCKLESS_IDS = ["null-layout", "stale-layout"]


@pytest.mark.parametrize("unknown", DOCKLESS_PAYLOADS, ids=DOCKLESS_IDS)
def test_workspace_state_dock_false_means_occupancy_unknown(unknown):
    state = delta._workspace_state(dict(unknown, active="a", visible=["a", "b"]))
    assert state == _state(None, "a", ["a", "b"])


@pytest.mark.parametrize("dock", [True, None], ids=["dock-capable", "capability-unknown"])
def test_workspace_state_trusts_layout_unless_dock_is_explicitly_false(dock):
    data = {"layout": _layout(_tiles(["a"])), "layout_dock": dock}
    assert delta._workspace_state(data)["tiles"] == _tiles(["a"])


def test_workspace_state_reads_tabs_showing_and_terminal():
    payload = _panels_payload(["terminal", ["a", "b*"]], "b", ["a", "b"])
    assert delta._workspace_state(payload) == _state(["terminal", ["a", "b*"]], "b", ["a", "b"])


def test_workspace_state_older_terminal_without_layout_is_unknown():
    """A payload without the layout field carries no tabs, so it cannot be diffed."""
    data = {"layout_age_s": 0.3, "active": "a", "visible": ["a"]}
    assert delta._workspace_state(data) == _state(None, "a", ["a"])


def test_workspace_state_malformed_layout_is_unknown():
    data = {"layout": {"split": "row", "items": ["junk"]}, "layout_dock": True}
    assert delta._workspace_state(data)["tiles"] is None


def test_layout_reader_matches_sibling_hook():
    """Both hooks flatten the same layout the same way, or a no-op reads as a change."""
    import osprey.templates.claude_code.claude.hooks.osprey_panels_context as panels

    layout = {
        "split": "row",
        "items": [
            {"tile": "t1", "tabs": ["terminal"], "showing": "terminal"},
            {
                "split": "column",
                "items": [
                    {"tile": "t2", "tabs": ["a", "b"], "showing": "b"},
                    {"tile": "t3", "tabs": ["c"]},
                ],
            },
        ],
    }
    data = {"layout": layout, "layout_dock": True, "active": "b", "visible": ["a"]}
    assert delta._workspace_state(data) == panels._workspace_state(data)
    assert delta._render_tiles(delta._workspace_state(data)["tiles"]) == panels._render_tiles(
        panels._workspace_state(data)["tiles"]
    )


@pytest.mark.parametrize("unknown", DOCKLESS_PAYLOADS, ids=DOCKLESS_IDS)
def test_main_says_nothing_about_tiles_for_dockless_client(
    monkeypatch, capsys, snapshot_file, unknown
):
    """Occupancy going unknown is not a tile change, and must not be recorded as one."""
    snapshot_file.write(_state(["a", "b"], "a", ["a", "b"]))
    _stub_urlopen(monkeypatch, payload=dict(unknown, active="a", visible=["a", "b"]))
    _stub_stdin(monkeypatch, {"session_id": "s1"})

    assert delta.main() == 0
    assert capsys.readouterr().out == ""
    assert snapshot_file.read()["tiles"] == _tiles(["a", "b"])


@pytest.mark.parametrize("unknown", DOCKLESS_PAYLOADS, ids=DOCKLESS_IDS)
def test_main_dockless_client_still_reports_rail_changes(
    monkeypatch, capsys, snapshot_file, unknown
):
    """A dock-less client knows the rail perfectly well — that half stays honest.

    The tile list it cannot see must survive into the rewritten snapshot, or a
    returning dock client would read as having reopened everything.
    """
    snapshot_file.write(_state(["a", "b"], "a", ["a", "b"]))
    _stub_urlopen(monkeypatch, payload=dict(unknown, active="a", visible=["a", "b", "c"]))
    _stub_stdin(monkeypatch, {"session_id": "s1"})

    assert delta.main() == 0

    context = json.loads(capsys.readouterr().out)["hookSpecificOutput"]["additionalContext"]
    assert context == "Workspace changed: c added to the launcher rail."
    assert snapshot_file.read()["tiles"] == _tiles(["a", "b"])


@pytest.mark.parametrize(
    "tiles",
    [[], [{"tabs": []}], [{"tabs": ["a", 1]}], [{"tile": "t1"}], ["a", {"tabs": ["b"]}]],
    ids=["empty", "empty-tabs", "non-string-tab", "no-tabs", "mixed"],
)
def test_normalize_tiles_wrong_shapes_are_unknown(tiles):
    assert delta._normalize_tiles(tiles) is None


# ---------------------------------------------------------------------------
# Tests of _build_delta_line, pure — tiles
# ---------------------------------------------------------------------------


def test_no_change_produces_no_line():
    before = _state(["terminal", ["a", "b*"]], "b", ["a", "b"])
    assert delta._build_delta_line(before, dict(before)) is None


def test_tile_closed():
    line = _line(
        (["terminal", "lattice", "artifacts"], "artifacts"),
        (["terminal", "artifacts"], "artifacts"),
    )
    assert line == "Workspace changed: lattice tile closed."


def test_tile_opened():
    line = _line(
        (["terminal", "artifacts"], "artifacts"),
        (["terminal", "artifacts", "bluesky"], "artifacts"),
    )
    assert line == "Workspace changed: bluesky tile opened."


def test_several_tiles_closed_share_one_clause():
    line = _line((["a", "b", "c"], "a"), (["a"], "a"))
    assert line == "Workspace changed: b, c tiles closed."


def test_multi_tab_tile_closed_names_its_tabs():
    line = _line((["terminal", ["a", "b*"]], "terminal"), (["terminal"], "terminal"))
    assert line == "Workspace changed: [a, b (showing)] tile closed."


def test_multi_tab_tile_opened_names_its_tabs():
    line = _line((["terminal"], "terminal"), (["terminal", ["a*", "b"]], "terminal"))
    assert line == "Workspace changed: [a (showing), b] tile opened."


def test_tab_closed_out_of_a_tile_that_remains():
    line = _line((["terminal", ["a", "b*"]], "b"), (["terminal", "a"], "b"))
    assert line == "Workspace changed: b tab closed."


def test_tab_opened_into_an_existing_tile():
    line = _line((["terminal", "a"], "a"), (["terminal", ["a", "b*"]], "a"))
    assert line == "Workspace changed: b tab opened in [a, b (showing)]."


def test_tab_moved_into_another_tile():
    """Dragging a one-tab tile onto another: a move, not a close and an open."""
    line = _line(
        (
            [
                {"tile": "t1", "tabs": ["a"], "showing": "a"},
                {"tile": "t2", "tabs": ["b"], "showing": "b"},
            ],
            "a",
        ),
        ([{"tile": "t2", "tabs": ["b", "a"], "showing": "a"}], "a"),
    )
    assert line == "Workspace changed: a tab moved to [b, a (showing)]."


def test_tab_moved_out_to_a_new_tile():
    """The source tile keeps its id, which settles which side moved."""
    line = _line(
        ([{"tile": "t2", "tabs": ["a", "b"], "showing": "b"}], "b"),
        (
            [
                {"tile": "t3", "tabs": ["b"], "showing": "b"},
                {"tile": "t2", "tabs": ["a"], "showing": "a"},
            ],
            "b",
        ),
    )
    assert line == "Workspace changed: b tab moved to a new tile."


def test_tab_moved_between_two_multi_tab_tiles():
    line = _line(
        (["terminal", ["a", "b"], ["c", "d"]], "a"),
        (["terminal", ["a"], ["c", "d", "b*"]], "a"),
    )
    assert line == "Workspace changed: b tab moved to [c, d, b (showing)]."


def test_terminal_moves_are_reported_too():
    """The terminal is a tab like any other in the report."""
    line = _line(
        (["terminal", "a"], "a"),
        ([{"tile": "t2", "tabs": ["a", "terminal"], "showing": "terminal"}], "terminal"),
    )
    assert (
        line
        == "Workspace changed: terminal tab moved to [a, terminal (showing)]; active now terminal."
    )


def test_showing_tab_switched():
    line = _line((["terminal", ["a*", "b"]], "a"), (["terminal", ["a", "b*"]], "b"))
    assert line == "Workspace changed: b tab now showing; active now b."


def test_closing_the_showing_tab_reports_close_and_new_showing():
    line = _line((["terminal", ["a", "b", "c*"]], "c"), (["terminal", ["a*", "b"]], "a"))
    assert line == "Workspace changed: c tab closed; a tab now showing; active now a."


def test_tab_opened_showing_is_not_a_separate_showing_change():
    line = _line((["terminal", ["a"]], "a"), (["terminal", ["a", "b*"]], "a"))
    assert line == "Workspace changed: b tab opened in [a, b (showing)]."


def test_tiles_rearranged_without_membership_change():
    line = _line((["a", "b"], "a"), (["b", "a"], "a"))
    assert line == "Workspace changed: tiles rearranged to [b], [a]."


def test_tabs_reordered_within_a_tile():
    line = _line(([["a", "b"]], "a"), ([["b", "a*"]], "a"))
    assert line == "Workspace changed: tiles rearranged to [b, a (showing)]."


def test_open_move_and_close_in_one_turn_each_get_a_clause():
    line = _line(
        (["terminal", ["a", "b"], "c"], "a"),
        (["terminal", ["a", "d*"], ["b"]], "d"),
    )
    assert line == (
        "Workspace changed: d tab opened in [a, d (showing)]; b tab moved to a new tile; "
        "c tile closed; active now d."
    )


# ---------------------------------------------------------------------------
# Tests of _build_delta_line, pure — active, rail, transitions
# ---------------------------------------------------------------------------


def test_active_change():
    line = _line((["a", "b"], "a", ["a", "b"]), (["a", "b"], "b", ["a", "b"]))
    assert line == "Workspace changed: active now b."


def test_active_cleared():
    line = _line((["a"], "a", ["a"]), (["a"], None, ["a"]))
    assert line == "Workspace changed: no tab active now."


def test_visible_added_and_removed():
    line = _line((["a"], "a", ["a", "gone"]), (["a"], "a", ["a", "new"]))
    assert line == (
        "Workspace changed: new added to the launcher rail; gone removed from the launcher rail."
    )


def test_rail_reorder_is_not_a_change():
    """Rail order carries nothing the agent can act on."""
    assert _line((["a"], "a", ["a", "b", "c"]), (["a"], "a", ["c", "a", "b"])) is None


def test_combined_delta_names_each_facet_once():
    line = _line(
        (["lattice", "artifacts"], "lattice", ["lattice", "artifacts"]),
        (["artifacts"], "artifacts", ["lattice", "artifacts", "bluesky"]),
    )
    assert line == (
        "Workspace changed: lattice tile closed; active now artifacts; "
        "bluesky added to the launcher rail."
    )


def test_first_tile_report_states_rather_than_claims_a_transition():
    """Unknown -> known is new information, but nothing "opened"."""
    line = _line((None, "a", ["a"]), (["terminal", ["a", "b"]], "a", ["a"]))
    assert line == "Workspace changed: tiles now [terminal], [a (showing), b]."


def test_tiles_going_unknown_says_nothing_about_tiles():
    """The last browser closed — the workspace stopped being observed, not changed."""
    assert _line((["a", "b"], "a", ["a"]), (None, "a", ["a"])) is None


def test_merge_state_carries_forward_last_known_tiles():
    merged = delta._merge_state(_state(["a", "b"], "a", ["a"]), _state(None, "b", ["a"]))
    assert merged == _state(["a", "b"], "b", ["a"])


def test_a_pop_out_and_its_return_are_each_one_clause():
    before = _state(["a", "b"], "a", ["a", "b"])
    out = _state(["a"], "a", ["a", "b"], popouts=["b"])
    assert "b tile closed; b popped out to a window" in (delta._build_delta_line(before, out) or "")
    back = _state(["a", "b"], "a", ["a", "b"])
    assert "b back from its window" in (delta._build_delta_line(out, back) or "")


def test_pop_outs_unknown_on_either_side_say_nothing():
    assert delta._build_delta_line(_state(None, "a", ["a"]), _state(None, "a", ["a"])) is None


# ---------------------------------------------------------------------------
# main() — silence
# ---------------------------------------------------------------------------


def test_main_silent_when_nothing_moved(monkeypatch, capsys, snapshot_file):
    """The defining property: byte-empty stdout, not an empty JSON envelope."""
    snapshot_file.write(_state(["a", "b"], "b", ["a", "b"]))
    _stub_urlopen(monkeypatch, payload=_panels_payload(["a", "b"], "b", ["a", "b"]))
    _stub_stdin(monkeypatch, {"session_id": "s1"})

    assert delta.main() == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_main_silent_without_session_id(monkeypatch, capsys, temp_snapshot_dir):
    """No stdin payload: nothing to diff, nothing written."""
    _stub_urlopen(monkeypatch, payload=_panels_payload(["a"], "a", ["a"]))
    _stub_stdin(monkeypatch, "")

    assert delta.main() == 0
    assert capsys.readouterr().out == ""
    assert list(temp_snapshot_dir.iterdir()) == []


def test_main_silent_when_web_terminal_down(monkeypatch, capsys, snapshot_file):
    """A down terminal leaves the snapshot alone — staleness is never asserted."""
    snapshot_file.write(_state(["a"], "a", ["a"]))
    _stub_urlopen(monkeypatch, raises=ConnectionRefusedError("down"))
    _stub_stdin(monkeypatch, {"session_id": "s1"})

    assert delta.main() == 0
    assert capsys.readouterr().out == ""
    assert snapshot_file.read() == _state(["a"], "a", ["a"])


def test_main_silent_on_unparseable_response(monkeypatch, capsys, snapshot_file):
    snapshot_file.write(_state(["a"], "a", ["a"]))
    _stub_urlopen(monkeypatch, bad_body=True)
    _stub_stdin(monkeypatch, {"session_id": "s1"})

    assert delta.main() == 0
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# main() — reseed + emit
# ---------------------------------------------------------------------------


def test_main_reseeds_missing_snapshot_without_emitting(monkeypatch, capsys, snapshot_file):
    _stub_urlopen(monkeypatch, payload=_panels_payload(["a", "b"], "b", ["a", "b"]))
    _stub_stdin(monkeypatch, {"session_id": "s1"})

    assert delta.main() == 0
    assert capsys.readouterr().out == ""
    assert snapshot_file.read() == _state(["a", "b"], "b", ["a", "b"])


def test_main_reseeds_corrupt_snapshot_without_emitting(monkeypatch, capsys, snapshot_file):
    """A torn file is a missing baseline, not a diff against garbage."""
    snapshot_file.path.write_text('{"tiles": ["a"], "acti')
    _stub_urlopen(monkeypatch, payload=_panels_payload(["a", "b"], "b", ["a", "b"]))
    _stub_stdin(monkeypatch, {"session_id": "s1"})

    assert delta.main() == 0
    assert capsys.readouterr().out == ""
    assert snapshot_file.read() == _state(["a", "b"], "b", ["a", "b"])


@pytest.mark.parametrize(
    "live,expected",
    [
        (_panels_payload(["a"], "a", ["a", "b"]), "b tile closed"),
        (_panels_payload([["a", "b*"]], "a", ["a", "b"]), "b tab moved to [a, b (showing)]"),
        (_panels_payload(["a", "b"], "b", ["a", "b"]), "active now b"),
        (_panels_payload(["a", "b"], "a", ["a", "b", "c"]), "c added to the launcher rail"),
    ],
    ids=["tiles", "tabs", "active", "visible"],
)
def test_main_emits_one_line_per_delta_type(monkeypatch, capsys, snapshot_file, live, expected):
    snapshot_file.write(_state(["a", "b"], "a", ["a", "b"]))
    _stub_urlopen(monkeypatch, payload=live)
    _stub_stdin(monkeypatch, {"session_id": "s1"})

    assert delta.main() == 0

    hook_out = json.loads(capsys.readouterr().out)["hookSpecificOutput"]
    assert hook_out["hookEventName"] == "UserPromptSubmit"
    assert hook_out["additionalContext"] == f"Workspace changed: {expected}."


def test_main_rewrites_snapshot_so_a_change_is_reported_once(monkeypatch, capsys, snapshot_file):
    snapshot_file.write(_state(["a", "b"], "a", ["a", "b"]))
    _stub_urlopen(monkeypatch, payload=_panels_payload(["a"], "a", ["a", "b"]))
    _stub_stdin(monkeypatch, {"session_id": "s1"})

    assert delta.main() == 0
    assert "b tile closed" in capsys.readouterr().out
    assert snapshot_file.read() == _state(["a"], "a", ["a", "b"])

    _stub_stdin(monkeypatch, {"session_id": "s1"})
    assert delta.main() == 0
    assert capsys.readouterr().out == ""


def test_main_keeps_last_known_tiles_when_browser_leaves(monkeypatch, capsys, snapshot_file):
    """Rail change with tiles unreported: the tile list must not be forgotten."""
    snapshot_file.write(_state(["a", "b"], "a", ["a", "b"]))
    _stub_urlopen(monkeypatch, payload=_panels_payload(None, "a", ["a", "b", "c"]))
    _stub_stdin(monkeypatch, {"session_id": "s1"})

    assert delta.main() == 0
    assert "c added to the launcher rail" in capsys.readouterr().out
    assert snapshot_file.read()["tiles"] == _tiles(["a", "b"])
