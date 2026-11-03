"""Tests for the SessionStart panels-context hook.

``osprey_panels_context.py`` fetches the web-terminal panel inventory at session
start, injects it as ``additionalContext`` alongside a line naming the tiles
actually on screen (each tile with its tabs and showing tab, the terminal
included), and seeds the workspace snapshot that the UserPromptSubmit
delta hook diffs against. It is a stdlib-only, fail-open hook: any unreachable
web terminal or parse error must leave the session unblocked (silent, exit 0).
These tests cover the string builders and snapshot helpers as pure functions,
and the ``main`` envelope/fail-open behavior with a stubbed ``urlopen`` (no real
HTTP).
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from tests.hooks.conftest import import_hook

panels = import_hook("osprey_panels_context")


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

    monkeypatch.setattr(panels.urllib.request, "urlopen", _fake)


def _stub_stdin(monkeypatch, payload):
    """Hand ``main`` a hook stdin payload (a str is written through verbatim)."""
    raw = payload if isinstance(payload, str) else json.dumps(payload)
    monkeypatch.setattr(panels.sys, "stdin", io.StringIO(raw))


@pytest.fixture
def temp_snapshot_dir(tmp_path, monkeypatch):
    """Redirect ``tempfile.gettempdir()`` so snapshots land in ``tmp_path``."""
    monkeypatch.setattr(panels.tempfile, "tempdir", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# Tests of _build_inventory, pure
# ---------------------------------------------------------------------------


def test_inventory_lists_enabled_panels_with_visibility():
    data = {
        "enabled": ["archiver", "bluesky"],
        "labels": {"archiver": "Archiver", "bluesky": "Bluesky"},
        "visible": ["archiver"],
        "active": "archiver",
    }
    result = panels._build_inventory(data)
    assert "Archiver (id=archiver, on rail)" in result
    assert "Bluesky (id=bluesky, off rail)" in result
    assert "Active tab: archiver." in result


def test_inventory_label_falls_back_to_uppercase_id():
    data = {"enabled": ["tuning"], "labels": {}, "visible": []}
    result = panels._build_inventory(data)
    assert "TUNING (id=tuning, off rail)" in result


def test_inventory_includes_custom_panels():
    data = {
        "enabled": [],
        "labels": {},
        "visible": ["mypanel"],
        "custom": [{"id": "mypanel", "label": "My Panel"}],
    }
    result = panels._build_inventory(data)
    assert "My Panel (id=mypanel, on rail)" in result


def test_inventory_skips_custom_panel_without_id():
    data = {"enabled": [], "custom": [{"label": "No Id"}, {"id": "ok", "label": "OK"}]}
    result = panels._build_inventory(data)
    assert "No Id" not in result
    assert "OK (id=ok, off rail)" in result


def test_inventory_no_active_tab_phrase():
    data = {"enabled": ["bluesky"], "labels": {"bluesky": "Bluesky"}, "visible": ["bluesky"]}
    result = panels._build_inventory(data)
    assert "No active tab." in result


def test_inventory_empty_returns_none():
    assert panels._build_inventory({"enabled": [], "custom": []}) is None
    assert panels._build_inventory({}) is None


def test_inventory_mentions_control_tools():
    data = {"enabled": ["bluesky"], "labels": {"bluesky": "Bluesky"}, "visible": ["bluesky"]}
    result = panels._build_inventory(data)
    assert "open_panel" in result and "close_panel" in result
    assert "add_panel_to_rail" in result and "remove_panel_from_rail" in result


# ---------------------------------------------------------------------------
# Tests of _read_session_id, pure
# ---------------------------------------------------------------------------


def test_session_id_read_from_stdin_payload():
    stream = io.StringIO(json.dumps({"session_id": "abc-123_DEF", "cwd": "/tmp"}))
    assert panels._read_session_id(stream) == "abc-123_DEF"


@pytest.mark.parametrize(
    "raw",
    ["", "{not json", "[]", '"a string"', "{}", '{"session_id": null}', '{"session_id": ""}'],
    ids=["empty", "invalid", "list", "scalar", "no-key", "null", "blank"],
)
def test_session_id_absent_returns_none(raw):
    """Anything the hook cannot read a session id out of yields None, not a raise."""
    assert panels._read_session_id(io.StringIO(raw)) is None


@pytest.mark.parametrize(
    "session_id",
    ["../../etc/passwd", "a/b", "a b", "with.dot", "sess\x00id"],
    ids=["traversal", "slash", "space", "dot", "nul"],
)
def test_session_id_rejects_unsafe_filename_chars(session_id):
    """The id is joined into a path, so anything outside [A-Za-z0-9_-] is dropped."""
    stream = io.StringIO(json.dumps({"session_id": session_id}))
    assert panels._read_session_id(stream) is None


def test_session_id_unreadable_stream_returns_none():
    class _Exploding:
        def read(self):
            raise OSError("stdin closed")

    assert panels._read_session_id(_Exploding()) is None


# ---------------------------------------------------------------------------
# Tests of _workspace_state, _snapshot_path and _write_snapshot, pure
# ---------------------------------------------------------------------------


def _tile(tile_id, *tabs, showing=None):
    """A dock layout tile node, as the server publishes it."""
    return {"tile": tile_id, "tabs": list(tabs), "showing": showing or tabs[0], "size": 0.5}


def _row(*items):
    return {"split": "row", "items": list(items)}


def _flat(tile_id, *tabs, showing=None):
    """The snapshot form of one tile: id, tabs, showing — no size."""
    return {"tile": tile_id, "tabs": list(tabs), "showing": showing or tabs[0]}


#: Terminal on the left, a two-tab tile showing artifacts beside it.
_LAYOUT = _row(_tile("t1", "terminal"), _tile("t2", "lattice", "artifacts", showing="artifacts"))
_TILES = [_flat("t1", "terminal"), _flat("t2", "lattice", "artifacts", showing="artifacts")]


def test_workspace_state_extracts_three_facets():
    data = {
        "layout": _LAYOUT,
        "rev": 4,
        "layout_dock": True,
        "active": "artifacts",
        "visible": ["lattice", "artifacts", "bluesky"],
    }
    assert panels._workspace_state(data) == {
        "tiles": _TILES,
        "popouts": [],
        "active": "artifacts",
        "visible": ["lattice", "artifacts", "bluesky"],
    }


def test_layout_tiles_reads_nested_splits_depth_first():
    """Reading order is the server's: depth-first, each split's items in order."""
    layout = {
        "split": "row",
        "items": [
            _tile("t1", "terminal"),
            {"split": "column", "items": [_tile("t2", "a", "b"), _tile("t3", "c")]},
            _tile("t4", "d"),
        ],
    }
    assert [t["tabs"] for t in panels._layout_tiles(layout)] == [
        ["terminal"],
        ["a", "b"],
        ["c"],
        ["d"],
    ]


def test_layout_tiles_single_tile_root():
    assert panels._layout_tiles(_tile("t1", "terminal")) == [_flat("t1", "terminal")]


def test_layout_tiles_defaults_showing_to_first_tab():
    """A showing that names no tab of its tile falls back to the engine's default."""
    tiles = panels._layout_tiles({"tile": "t1", "tabs": ["a", "b"], "showing": "zzz"})
    assert tiles[0]["showing"] == "a"
    tiles = panels._layout_tiles({"tabs": ["a", "b"]})
    assert tiles == [{"tile": None, "tabs": ["a", "b"], "showing": "a"}]


@pytest.mark.parametrize(
    "layout",
    [
        None,
        "a layout",
        [],
        {},
        {"tile": "t1", "tabs": []},
        {"tile": "t1", "tabs": ["a", 3]},
        {"tile": "t1", "tabs": ["a", ""]},
        {"split": "row", "items": []},
        {"split": "row", "items": [_tile("t1", "a"), "junk"]},
        {"split": "row"},
    ],
    ids=[
        "none",
        "string",
        "list",
        "empty-object",
        "empty-tabs",
        "non-string-tab",
        "blank-tab",
        "empty-split",
        "junk-item",
        "split-no-items",
    ],
)
def test_layout_tiles_malformed_is_unknown(layout):
    """A layout of the wrong shape describes nothing — never an empty screen."""
    assert panels._layout_tiles(layout) is None


def test_layout_tiles_too_deep_is_unknown():
    node = _tile("t1", "a")
    for _ in range(panels._MAX_LAYOUT_DEPTH + 1):
        node = {"split": "row", "items": [node, _tile("tx", "b")]}
    assert panels._layout_tiles(node) is None


def test_workspace_state_null_layout_means_never_reported():
    """No dock client has reported: tiles unknown, not an empty workspace."""
    data = {"layout": None, "rev": 0, "layout_age_s": None, "visible": ["bluesky"]}
    assert panels._workspace_state(data)["tiles"] is None


@pytest.mark.parametrize(
    "payload",
    [
        {"layout": None, "layout_age_s": 0.3, "layout_dock": False},
        {"layout": _LAYOUT, "layout_age_s": 0.3, "layout_dock": False},
    ],
    ids=["null-layout", "stale-layout-from-dockless-client"],
)
def test_workspace_state_dock_false_means_occupancy_unknown(payload):
    """A client without a dock knows the rail but cannot see tiles."""
    state = panels._workspace_state(dict(payload, visible=["bluesky"], active="bluesky"))
    assert state["tiles"] is None
    assert state["visible"] == ["bluesky"]


@pytest.mark.parametrize("dock", [True, None], ids=["dock-capable", "capability-unknown"])
def test_workspace_state_trusts_layout_unless_dock_is_explicitly_false(dock):
    data = {"layout": _LAYOUT, "layout_dock": dock, "visible": ["bluesky"]}
    assert panels._workspace_state(data)["tiles"] == _TILES


def test_workspace_state_older_terminal_without_layout_field():
    """A web terminal answering without the layout field says nothing about tiles."""
    state = panels._workspace_state(
        {
            "layout_age_s": 0.2,
            "visible": ["bluesky"],
            "active": "bluesky",
        }
    )
    assert state == {"tiles": None, "popouts": [], "active": "bluesky", "visible": ["bluesky"]}


def test_workspace_state_ignores_wrong_types():
    state = panels._workspace_state({"layout": "nope", "visible": {"a": 1}, "active": 7})
    assert state == {"tiles": None, "popouts": [], "active": None, "visible": []}


def test_workspace_state_drops_non_string_rail_ids():
    data = {"layout": _LAYOUT, "visible": ["bluesky", {}, 3]}
    assert panels._workspace_state(data)["visible"] == ["bluesky"]


def test_snapshot_path_is_session_scoped(temp_snapshot_dir):
    path = panels._snapshot_path("sess-1")
    assert path == str(temp_snapshot_dir / "osprey-workspace-sess-1.json")


def test_write_snapshot_roundtrips_and_leaves_no_temp_file(temp_snapshot_dir):
    path = panels._snapshot_path("sess-1")
    state = {"tiles": ["bluesky"], "active": "bluesky", "visible": ["bluesky"]}

    assert panels._write_snapshot(path, state) is True
    assert json.loads(Path(path).read_text()) == state
    assert list(temp_snapshot_dir.iterdir()) == [temp_snapshot_dir / "osprey-workspace-sess-1.json"]


@pytest.mark.usefixtures("temp_snapshot_dir")
def test_write_snapshot_overwrites_previous_content():
    path = panels._snapshot_path("sess-1")
    panels._write_snapshot(path, {"tiles": ["old"], "active": "old", "visible": ["old"]})
    panels._write_snapshot(path, {"tiles": ["new"], "active": "new", "visible": ["new"]})
    assert json.loads(Path(path).read_text())["tiles"] == ["new"]


def test_write_snapshot_writes_nothing_without_the_file_helper(temp_snapshot_dir, monkeypatch):
    monkeypatch.setattr(panels, "_safety_file", None)
    assert panels._write_snapshot(panels._snapshot_path("sess-1"), {"tiles": []}) is False
    assert list(temp_snapshot_dir.iterdir()) == []


def test_write_snapshot_returns_false_when_unwritable(tmp_path):
    """An unwritable temp dir is reported, never raised."""
    assert panels._write_snapshot(str(tmp_path / "missing-dir" / "snap.json"), {}) is False


# ---------------------------------------------------------------------------
# Tests of _build_tile_line, pure
# ---------------------------------------------------------------------------


def test_tile_line_names_tiles_tabs_and_showing_with_terminal():
    line = panels._build_tile_line({"tiles": _TILES, "active": "artifacts", "visible": []})
    assert line == "Open tiles: [terminal], [lattice, artifacts (showing)]; active artifacts."


def test_tile_line_one_tab_tiles_carry_no_marker():
    tiles = [_flat("t1", "terminal"), _flat("t2", "lattice"), _flat("t3", "bluesky")]
    line = panels._build_tile_line({"tiles": tiles, "active": "lattice", "visible": []})
    assert line == "Open tiles: [terminal], [lattice], [bluesky]; active lattice."


def test_tile_line_marks_the_showing_tab_wherever_it_sits():
    tiles = [_flat("t1", "terminal", "lattice", "bluesky", showing="lattice")]
    line = panels._build_tile_line({"tiles": tiles, "active": "lattice", "visible": []})
    assert line == "Open tiles: [terminal, lattice (showing), bluesky]; active lattice."


def test_tile_line_without_active_tab():
    line = panels._build_tile_line({"tiles": _TILES, "active": None, "visible": []})
    assert line.endswith("; no active tab.")


def test_tile_line_is_one_line():
    line = panels._build_tile_line({"tiles": _TILES * 3, "active": "a", "visible": []})
    assert "\n" not in line


def test_tile_line_omitted_when_never_reported():
    """No client has reported — say nothing rather than claim an empty screen."""
    assert (
        panels._build_tile_line({"tiles": None, "active": "bluesky", "visible": ["bluesky"]})
        is None
    )


# ---------------------------------------------------------------------------
# main() — envelope + fail-open
# ---------------------------------------------------------------------------


def test_main_emits_additional_context_envelope(monkeypatch, capsys):
    _stub_urlopen(
        monkeypatch,
        payload={
            "enabled": ["bluesky"],
            "labels": {"bluesky": "Bluesky"},
            "visible": ["bluesky"],
            "active": "bluesky",
        },
    )
    rc = panels.main()
    assert rc == 0

    out = capsys.readouterr().out
    envelope = json.loads(out)
    hook_out = envelope["hookSpecificOutput"]
    assert hook_out["hookEventName"] == "SessionStart"
    assert "Bluesky (id=bluesky, on rail)" in hook_out["additionalContext"]


def test_main_empty_inventory_writes_nothing(monkeypatch, capsys):
    _stub_urlopen(monkeypatch, payload={"enabled": [], "custom": []})
    rc = panels.main()
    assert rc == 0
    assert capsys.readouterr().out == ""


def test_main_fails_open_when_web_terminal_down(monkeypatch, capsys):
    """urlopen raising (terminal unreachable) -> silent, exit 0."""
    _stub_urlopen(monkeypatch, raises=ConnectionRefusedError("down"))
    rc = panels.main()
    assert rc == 0
    assert capsys.readouterr().out == ""


def test_main_fails_open_on_unparseable_body(monkeypatch, capsys):
    """A 200 response with a non-JSON body must not crash the session start."""
    _stub_urlopen(monkeypatch, bad_body=True)
    rc = panels.main()
    assert rc == 0
    assert capsys.readouterr().out == ""


def test_main_uses_configured_web_port(monkeypatch):
    """OSPREY_WEB_PORT selects the port the inventory is fetched from."""
    monkeypatch.setenv("OSPREY_WEB_PORT", "9123")
    seen = {}

    def _fake(url, timeout=None):  # noqa: ARG001 - urlopen takes the url positionally and timeout by keyword
        seen["url"] = url
        return _FakeResponse({"enabled": [], "custom": []})

    monkeypatch.setattr(panels.urllib.request, "urlopen", _fake)
    panels.main()
    assert "127.0.0.1:9123/api/panels" in seen["url"]


# ---------------------------------------------------------------------------
# main() — workspace line + snapshot seed
# ---------------------------------------------------------------------------

_TILED_PAYLOAD = {
    "enabled": ["lattice", "artifacts"],
    "labels": {"lattice": "LATTICE", "artifacts": "WORKSPACE"},
    "visible": ["lattice", "artifacts"],
    "active": "artifacts",
    "layout": _LAYOUT,
    "rev": 3,
    "layout_age_s": 0.4,
    "layout_dock": True,
}

_TILE_LINE = "Open tiles: [terminal], [lattice, artifacts (showing)]; active artifacts."


def test_main_appends_tile_state_to_context(monkeypatch, capsys):
    _stub_urlopen(monkeypatch, payload=_TILED_PAYLOAD)
    _stub_stdin(monkeypatch, {"session_id": "sess-1"})

    assert panels.main() == 0

    context = json.loads(capsys.readouterr().out)["hookSpecificOutput"]["additionalContext"]
    assert "WORKSPACE (id=artifacts, on rail)" in context
    assert _TILE_LINE in context


def test_main_omits_tile_state_when_never_reported(monkeypatch, capsys):
    payload = dict(_TILED_PAYLOAD, layout=None, rev=0, layout_age_s=None)
    _stub_urlopen(monkeypatch, payload=payload)
    _stub_stdin(monkeypatch, {"session_id": "sess-1"})

    assert panels.main() == 0
    assert "Open tiles" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "unknown",
    [
        {"layout": None, "layout_age_s": 0.3, "layout_dock": False},
        {"layout_age_s": 0.3, "layout_dock": False},
    ],
    ids=["null-layout", "stale-layout"],
)
def test_main_omits_tile_line_for_dockless_client(monkeypatch, capsys, temp_snapshot_dir, unknown):
    """A dock-less reporter must not turn into a claim about what is on screen."""
    _stub_urlopen(monkeypatch, payload=dict(_TILED_PAYLOAD, **unknown))
    _stub_stdin(monkeypatch, {"session_id": "sess-1"})

    assert panels.main() == 0

    out = capsys.readouterr().out
    assert "Open tiles" not in out
    # The inventory is still known and still worth injecting.
    assert "WORKSPACE (id=artifacts, on rail)" in out
    snapshot = json.loads((temp_snapshot_dir / "osprey-workspace-sess-1.json").read_text())
    assert snapshot["tiles"] is None


def test_main_seeds_snapshot_from_stdin_session_id(monkeypatch, capsys, temp_snapshot_dir):
    _stub_urlopen(monkeypatch, payload=_TILED_PAYLOAD)
    _stub_stdin(monkeypatch, {"session_id": "sess-1"})

    assert panels.main() == 0
    capsys.readouterr()

    snapshot = json.loads((temp_snapshot_dir / "osprey-workspace-sess-1.json").read_text())
    assert snapshot == {
        "tiles": _TILES,
        "popouts": [],
        "active": "artifacts",
        "visible": ["lattice", "artifacts"],
    }


def test_main_overwrites_stale_snapshot_on_resume(monkeypatch, capsys, temp_snapshot_dir):
    """SessionStart re-fires on resume/clear with the same id — always overwrite.

    A snapshot left from before the resume would make the first delta line of
    the resumed session report changes the agent has already been told about.
    """
    stale = temp_snapshot_dir / "osprey-workspace-sess-1.json"
    stale.write_text(json.dumps({"tiles": ["gone"], "active": "gone", "visible": ["gone"]}))

    _stub_urlopen(monkeypatch, payload=_TILED_PAYLOAD)
    _stub_stdin(monkeypatch, {"session_id": "sess-1"})
    assert panels.main() == 0
    capsys.readouterr()

    assert json.loads(stale.read_text())["tiles"] == _TILES


def test_main_without_session_id_still_emits_context(monkeypatch, capsys, temp_snapshot_dir):
    """No usable stdin: skip the snapshot, keep every context line."""
    _stub_urlopen(monkeypatch, payload=_TILED_PAYLOAD)
    _stub_stdin(monkeypatch, "{truncated")

    assert panels.main() == 0

    context = json.loads(capsys.readouterr().out)["hookSpecificOutput"]["additionalContext"]
    assert _TILE_LINE in context
    assert list(temp_snapshot_dir.iterdir()) == []


def test_main_writes_no_snapshot_when_web_terminal_down(monkeypatch, capsys, temp_snapshot_dir):
    """No live state means nothing honest to record — the delta hook reseeds."""
    _stub_urlopen(monkeypatch, raises=ConnectionRefusedError("down"))
    _stub_stdin(monkeypatch, {"session_id": "sess-1"})

    assert panels.main() == 0
    capsys.readouterr()
    assert list(temp_snapshot_dir.iterdir()) == []


def test_main_keeps_surface_line_when_web_terminal_down(monkeypatch, capsys):
    """Reading stdin must not cost the env-derived line when the fetch fails."""
    monkeypatch.setenv("OSPREY_WEB_UX", "terminal")
    _stub_urlopen(monkeypatch, raises=ConnectionRefusedError("down"))
    _stub_stdin(monkeypatch, {"session_id": "sess-1"})

    assert panels.main() == 0

    context = json.loads(capsys.readouterr().out)["hookSpecificOutput"]["additionalContext"]
    assert "terminal on the Agent tile" in context


# ---------------------------------------------------------------------------
# Fail-open on malformed stdin (subprocess)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stdin", ["", "{nope", "[]"], ids=["empty", "invalid-json", "wrong-shape"])
def test_malformed_stdin_fails_open(reserve_port, hook_runner_raw, monkeypatch, stdin):
    """Stdin the hook cannot use still leaves the session start unblocked.

    SessionStart hands the hook a JSON payload it never reads. Run as a real
    subprocess, a closed pipe, a truncated write and a payload of the wrong
    JSON type must each exit 0 with nothing on stdout. ``OSPREY_WEB_PORT``
    points at a port nobody is listening on so the result does not depend on
    whether a web terminal happens to be running on the developer's machine.
    """
    monkeypatch.setenv("OSPREY_WEB_PORT", str(reserve_port()))

    returncode, stdout, stderr = hook_runner_raw(
        "osprey_panels_context.py",
        tool_name=None,
        tool_input=None,
        stdin_override=stdin,
    )

    assert returncode == 0
    assert stdout.strip() == ""
    assert "Traceback" not in stderr


def test_workspace_state_names_the_pop_outs_beside_the_tiles():
    data = {"layout": _LAYOUT, "layout_dock": True, "popouts": ["ariel", "bluesky", "ariel", 3]}
    state = panels._workspace_state(data)
    assert state["popouts"] == ["ariel", "bluesky"]
    assert "pop-outs: ariel, bluesky;" in panels._build_tile_line(state)


def test_tile_line_names_no_pop_outs_when_there_are_none():
    state = panels._workspace_state({"layout": _LAYOUT, "layout_dock": True, "active": "lattice"})
    assert "pop-outs" not in panels._build_tile_line(state)
