"""Tests for the SessionStart panels-context hook.

``osprey_panels_context.py`` fetches the web-terminal panel inventory at session
start, injects it as ``additionalContext`` alongside a line naming the tiles
actually on screen, and seeds the workspace snapshot that the UserPromptSubmit
delta hook diffs against. It is a stdlib-only, fail-open hook: any unreachable
web terminal or parse error must leave the session unblocked (silent, exit 0).
These tests cover the string builders and snapshot helpers as pure functions,
and the ``main`` envelope/fail-open behavior with a stubbed ``urlopen`` (no real
HTTP).
"""

from __future__ import annotations

import io
import json

import pytest

import osprey.templates.claude_code.claude.hooks.osprey_panels_context as panels


class _FakeResponse:
    def __init__(self, payload):
        self._data = json.dumps(payload).encode()

    def read(self):
        return self._data


def _stub_urlopen(monkeypatch, *, payload=None, raises=None, bad_body=False):
    """Point the hook's urlopen at a fake response, an error, or garbage body."""

    def _fake(url, timeout=None):
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
# _build_inventory (pure)
# ---------------------------------------------------------------------------


def test_inventory_lists_enabled_panels_with_visibility():
    data = {
        "enabled": ["archiver", "scan"],
        "labels": {"archiver": "Archiver", "scan": "Scan"},
        "visible": ["archiver"],
        "active": "archiver",
    }
    result = panels._build_inventory(data)
    assert "Archiver (id=archiver, on rail)" in result
    assert "Scan (id=scan, off rail)" in result
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
    data = {"enabled": ["scan"], "labels": {"scan": "Scan"}, "visible": ["scan"]}
    result = panels._build_inventory(data)
    assert "No active tab." in result


def test_inventory_empty_returns_none():
    assert panels._build_inventory({"enabled": [], "custom": []}) is None
    assert panels._build_inventory({}) is None


def test_inventory_mentions_control_tools():
    data = {"enabled": ["scan"], "labels": {"scan": "Scan"}, "visible": ["scan"]}
    result = panels._build_inventory(data)
    assert "open_panel" in result and "close_panel" in result
    assert "add_panel_to_rail" in result and "remove_panel_from_rail" in result


# ---------------------------------------------------------------------------
# _read_session_id (pure)
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
# _workspace_state / _snapshot_path / _write_snapshot (pure)
# ---------------------------------------------------------------------------


def test_workspace_state_extracts_three_facets():
    data = {
        "open_tiles": ["lattice", "artifacts"],
        "open_tiles_age_s": 1.5,
        "active": "artifacts",
        "visible": ["lattice", "artifacts", "scan"],
    }
    assert panels._workspace_state(data) == {
        "tiles": ["lattice", "artifacts"],
        "active": "artifacts",
        "visible": ["lattice", "artifacts", "scan"],
    }


def test_workspace_state_null_age_means_never_reported():
    """A tile list nobody has reported is None, not an empty workspace."""
    data = {"open_tiles": [], "open_tiles_age_s": None, "visible": ["scan"]}
    assert panels._workspace_state(data)["tiles"] is None


@pytest.mark.parametrize(
    "payload",
    [
        {"open_tiles": None, "open_tiles_age_s": 0.3, "open_tiles_dock": False},
        {"open_tiles": [], "open_tiles_age_s": 0.3, "open_tiles_dock": False},
        {"open_tiles": ["scan"], "open_tiles_age_s": 0.3, "open_tiles_dock": False},
    ],
    ids=["null-tiles", "legacy-empty-list", "stale-list-from-dockless-client"],
)
def test_workspace_state_dock_false_means_occupancy_unknown(payload):
    """A client without a dock knows the rail but cannot see tiles.

    Its report carries a numeric age, so the age check alone would read it as a
    trustworthy empty workspace. Both the post-fix shape (null tiles) and the
    legacy shape (empty list) must land on unknown.
    """
    state = panels._workspace_state(dict(payload, visible=["scan"], active="scan"))
    assert state["tiles"] is None
    assert state["visible"] == ["scan"]


@pytest.mark.parametrize("dock", [True, None], ids=["dock-capable", "capability-unknown"])
def test_workspace_state_trusts_tiles_unless_dock_is_explicitly_false(dock):
    data = {
        "open_tiles": ["scan"],
        "open_tiles_age_s": 0.3,
        "open_tiles_dock": dock,
        "visible": ["scan"],
    }
    assert panels._workspace_state(data)["tiles"] == ["scan"]


def test_workspace_state_older_terminal_without_tile_fields():
    """A web terminal that predates open_tiles reports them as never reported."""
    state = panels._workspace_state({"visible": ["scan"], "active": "scan"})
    assert state == {"tiles": None, "active": "scan", "visible": ["scan"]}


def test_workspace_state_ignores_wrong_types():
    state = panels._workspace_state({"open_tiles": "nope", "visible": {"a": 1}, "active": 7})
    assert state == {"tiles": None, "active": None, "visible": []}


def test_workspace_state_drops_non_string_ids():
    data = {"open_tiles": ["scan", 3, None], "open_tiles_age_s": 0.2, "visible": ["scan", {}]}
    state = panels._workspace_state(data)
    assert state["tiles"] == ["scan"]
    assert state["visible"] == ["scan"]


def test_snapshot_path_is_session_scoped(temp_snapshot_dir):
    path = panels._snapshot_path("sess-1")
    assert path == str(temp_snapshot_dir / "osprey-workspace-sess-1.json")


def test_write_snapshot_roundtrips_and_leaves_no_temp_file(temp_snapshot_dir):
    path = panels._snapshot_path("sess-1")
    state = {"tiles": ["scan"], "active": "scan", "visible": ["scan"]}

    assert panels._write_snapshot(path, state) is True
    assert json.loads(open(path).read()) == state
    assert list(temp_snapshot_dir.iterdir()) == [temp_snapshot_dir / "osprey-workspace-sess-1.json"]


def test_write_snapshot_overwrites_previous_content(temp_snapshot_dir):
    path = panels._snapshot_path("sess-1")
    panels._write_snapshot(path, {"tiles": ["old"], "active": "old", "visible": ["old"]})
    panels._write_snapshot(path, {"tiles": ["new"], "active": "new", "visible": ["new"]})
    assert json.loads(open(path).read())["tiles"] == ["new"]


def test_write_snapshot_returns_false_when_unwritable(tmp_path):
    """An unwritable temp dir is reported, never raised."""
    assert panels._write_snapshot(str(tmp_path / "missing-dir" / "snap.json"), {}) is False


# ---------------------------------------------------------------------------
# _build_tile_line (pure)
# ---------------------------------------------------------------------------


def test_tile_line_names_tiles_in_order_and_active():
    line = panels._build_tile_line(
        {"tiles": ["lattice", "artifacts"], "active": "artifacts", "visible": []}
    )
    assert line == "Open tiles: [lattice, artifacts], active artifacts."


def test_tile_line_without_active_tile():
    line = panels._build_tile_line({"tiles": ["lattice"], "active": None, "visible": []})
    assert line == "Open tiles: [lattice], no active tile."


def test_tile_line_reported_empty_workspace():
    line = panels._build_tile_line({"tiles": [], "active": None, "visible": []})
    assert "Open tiles: []" in line


def test_tile_line_omitted_when_never_reported():
    """No client has reported — say nothing rather than claim an empty screen."""
    assert panels._build_tile_line({"tiles": None, "active": "scan", "visible": ["scan"]}) is None


# ---------------------------------------------------------------------------
# main() — envelope + fail-open
# ---------------------------------------------------------------------------


def test_main_emits_additional_context_envelope(monkeypatch, capsys):
    _stub_urlopen(
        monkeypatch,
        payload={
            "enabled": ["scan"],
            "labels": {"scan": "Scan"},
            "visible": ["scan"],
            "active": "scan",
        },
    )
    rc = panels.main()
    assert rc == 0

    out = capsys.readouterr().out
    envelope = json.loads(out)
    hook_out = envelope["hookSpecificOutput"]
    assert hook_out["hookEventName"] == "SessionStart"
    assert "Scan (id=scan, on rail)" in hook_out["additionalContext"]


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

    def _fake(url, timeout=None):
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
    "open_tiles": ["lattice", "artifacts"],
    "open_tiles_age_s": 0.4,
}


def test_main_appends_tile_state_to_context(monkeypatch, capsys):
    _stub_urlopen(monkeypatch, payload=_TILED_PAYLOAD)
    _stub_stdin(monkeypatch, {"session_id": "sess-1"})

    assert panels.main() == 0

    context = json.loads(capsys.readouterr().out)["hookSpecificOutput"]["additionalContext"]
    assert "WORKSPACE (id=artifacts, on rail)" in context
    assert "Open tiles: [lattice, artifacts], active artifacts." in context


def test_main_omits_tile_state_when_never_reported(monkeypatch, capsys):
    payload = dict(_TILED_PAYLOAD, open_tiles=[], open_tiles_age_s=None)
    _stub_urlopen(monkeypatch, payload=payload)
    _stub_stdin(monkeypatch, {"session_id": "sess-1"})

    assert panels.main() == 0
    assert "Open tiles" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "unknown",
    [
        {"open_tiles": None, "open_tiles_age_s": 0.3, "open_tiles_dock": False},
        {"open_tiles": [], "open_tiles_age_s": 0.3, "open_tiles_dock": False},
    ],
    ids=["null-tiles", "legacy-empty-list"],
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
        "tiles": ["lattice", "artifacts"],
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

    assert json.loads(stale.read_text())["tiles"] == ["lattice", "artifacts"]


def test_main_without_session_id_still_emits_context(monkeypatch, capsys, temp_snapshot_dir):
    """No usable stdin: skip the snapshot, keep every context line."""
    _stub_urlopen(monkeypatch, payload=_TILED_PAYLOAD)
    _stub_stdin(monkeypatch, "{truncated")

    assert panels.main() == 0

    context = json.loads(capsys.readouterr().out)["hookSpecificOutput"]["additionalContext"]
    assert "Open tiles: [lattice, artifacts], active artifacts." in context
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
    monkeypatch.setenv("OSPREY_WEB_UX", "expert")
    _stub_urlopen(monkeypatch, raises=ConnectionRefusedError("down"))
    _stub_stdin(monkeypatch, {"session_id": "sess-1"})

    assert panels.main() == 0

    context = json.loads(capsys.readouterr().out)["hookSpecificOutput"]["additionalContext"]
    assert "EXPERT surface" in context


# ---------------------------------------------------------------------------
# Fail-open on malformed stdin (subprocess)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stdin", ["", "{nope", "[]"], ids=["empty", "invalid-json", "wrong-shape"])
def test_malformed_stdin_fails_open(hook_runner_raw, monkeypatch, stdin):
    """Stdin the hook cannot use still leaves the session start unblocked.

    SessionStart hands the hook a JSON payload it never reads. Run as a real
    subprocess, a closed pipe, a truncated write and a payload of the wrong
    JSON type must each exit 0 with nothing on stdout. ``OSPREY_WEB_PORT``
    points at a port nobody is listening on so the result does not depend on
    whether a web terminal happens to be running on the developer's machine.
    """
    from osprey.interfaces._serving import free_port

    monkeypatch.setenv("OSPREY_WEB_PORT", str(free_port()))

    returncode, stdout, stderr = hook_runner_raw(
        "osprey_panels_context.py",
        tool_name=None,
        tool_input=None,
        stdin_override=stdin,
    )

    assert returncode == 0
    assert stdout.strip() == ""
    assert "Traceback" not in stderr
