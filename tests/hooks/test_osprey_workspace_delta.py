"""Tests for the UserPromptSubmit workspace-delta hook.

``osprey_workspace_delta.py`` diffs the live web workspace against the snapshot
seeded by the SessionStart panels-context hook and injects one line naming what
moved. Its defining property is silence: an unchanged workspace, an unreadable
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

import osprey.templates.claude_code.claude.hooks.osprey_workspace_delta as delta


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

    monkeypatch.setattr(delta.urllib.request, "urlopen", _fake)


def _stub_stdin(monkeypatch, payload):
    """Hand ``main`` a hook stdin payload (a str is written through verbatim)."""
    raw = payload if isinstance(payload, str) else json.dumps(payload)
    monkeypatch.setattr(delta.sys, "stdin", io.StringIO(raw))


def _state(tiles=None, active=None, visible=()):
    return {"tiles": tiles, "active": active, "visible": list(visible)}


def _panels_payload(tiles=None, active=None, visible=(), age=0.5):
    """A ``GET /api/panels`` body; ``tiles=None`` means nobody has reported."""
    return {
        "open_tiles": list(tiles) if tiles is not None else [],
        "open_tiles_age_s": age if tiles is not None else None,
        "active": active,
        "visible": list(visible),
    }


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
    import osprey.templates.claude_code.claude.hooks.osprey_panels_context as panels

    assert delta._snapshot_path("s1") == str(temp_snapshot_dir / "osprey-workspace-s1.json")
    assert delta._snapshot_path("s1") == panels._snapshot_path("s1")


def test_load_snapshot_reads_the_three_facets(snapshot_file):
    snapshot_file.write({"tiles": ["a"], "active": "a", "visible": ["a", "b"]})
    assert delta._load_snapshot(str(snapshot_file.path)) == _state(["a"], "a", ["a", "b"])


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


def test_workspace_state_null_age_means_tiles_unknown():
    state = delta._workspace_state(_panels_payload(tiles=None, active="a", visible=["a"]))
    assert state == _state(None, "a", ["a"])


#: The two payload shapes that mean "a client reported, but it has no dock".
#: The first is what the server sends after the occupancy fix; the second is the
#: legacy shape an older server still sends. Both carry a *numeric* age, so an
#: age-only guard would read either as a trustworthy empty workspace.
DOCKLESS_PAYLOADS = [
    {"open_tiles": None, "open_tiles_age_s": 0.3, "open_tiles_dock": False},
    {"open_tiles": [], "open_tiles_age_s": 0.3, "open_tiles_dock": False},
]
DOCKLESS_IDS = ["null-tiles", "legacy-empty-list"]


@pytest.mark.parametrize("unknown", DOCKLESS_PAYLOADS, ids=DOCKLESS_IDS)
def test_workspace_state_dock_false_means_occupancy_unknown(unknown):
    state = delta._workspace_state(dict(unknown, active="a", visible=["a", "b"]))
    assert state == _state(None, "a", ["a", "b"])


@pytest.mark.parametrize("dock", [True, None], ids=["dock-capable", "capability-unknown"])
def test_workspace_state_trusts_tiles_unless_dock_is_explicitly_false(dock):
    data = {"open_tiles": ["a"], "open_tiles_age_s": 0.3, "open_tiles_dock": dock}
    assert delta._workspace_state(data)["tiles"] == ["a"]


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
    assert snapshot_file.read()["tiles"] == ["a", "b"]


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
    assert snapshot_file.read()["tiles"] == ["a", "b"]


def test_workspace_state_reads_reported_tiles():
    state = delta._workspace_state(
        _panels_payload(tiles=["a", "b"], active="b", visible=["a", "b"])
    )
    assert state == _state(["a", "b"], "b", ["a", "b"])


# ---------------------------------------------------------------------------
# _build_delta_line (pure)
# ---------------------------------------------------------------------------


def test_no_change_produces_no_line():
    before = _state(["a", "b"], "b", ["a", "b"])
    assert delta._build_delta_line(before, dict(before)) is None


def test_tile_closed():
    line = delta._build_delta_line(
        _state(["lattice", "artifacts"], "artifacts", ["lattice", "artifacts"]),
        _state(["artifacts"], "artifacts", ["lattice", "artifacts"]),
    )
    assert line == "Workspace changed: lattice tile closed."


def test_tile_opened():
    line = delta._build_delta_line(
        _state(["artifacts"], "artifacts", ["artifacts"]),
        _state(["artifacts", "bluesky"], "artifacts", ["artifacts"]),
    )
    assert line == "Workspace changed: bluesky tile opened."


def test_several_tiles_closed_share_one_clause():
    line = delta._build_delta_line(
        _state(["a", "b", "c"], "a", []),
        _state(["a"], "a", []),
    )
    assert line == "Workspace changed: b, c tiles closed."


def test_tiles_rearranged_without_membership_change():
    line = delta._build_delta_line(
        _state(["a", "b"], "a", []),
        _state(["b", "a"], "a", []),
    )
    assert line == "Workspace changed: tiles rearranged to [b, a]."


def test_active_change():
    line = delta._build_delta_line(
        _state(["a", "b"], "a", ["a", "b"]),
        _state(["a", "b"], "b", ["a", "b"]),
    )
    assert line == "Workspace changed: active now b."


def test_active_cleared():
    line = delta._build_delta_line(_state(["a"], "a", ["a"]), _state(["a"], None, ["a"]))
    assert line == "Workspace changed: no tile active now."


def test_visible_added_and_removed():
    line = delta._build_delta_line(
        _state(["a"], "a", ["a", "gone"]),
        _state(["a"], "a", ["a", "new"]),
    )
    assert line == (
        "Workspace changed: new added to the launcher rail; gone removed from the launcher rail."
    )


def test_rail_reorder_is_not_a_change():
    """Rail order carries nothing the agent can act on."""
    before = _state(["a"], "a", ["a", "b", "c"])
    assert delta._build_delta_line(before, _state(["a"], "a", ["c", "a", "b"])) is None


def test_combined_delta_names_each_facet_once():
    line = delta._build_delta_line(
        _state(["lattice", "artifacts"], "lattice", ["lattice", "artifacts"]),
        _state(["artifacts"], "artifacts", ["lattice", "artifacts", "bluesky"]),
    )
    assert line == (
        "Workspace changed: lattice tile closed; active now artifacts; "
        "bluesky added to the launcher rail."
    )


def test_first_tile_report_states_rather_than_claims_a_transition():
    """Unknown -> known is new information, but nothing "opened"."""
    line = delta._build_delta_line(_state(None, "a", ["a"]), _state(["a", "b"], "a", ["a"]))
    assert line == "Workspace changed: tiles now [a, b]."


def test_tiles_going_unknown_says_nothing_about_tiles():
    """The last browser closed — the workspace stopped being observed, not changed."""
    assert delta._build_delta_line(_state(["a", "b"], "a", ["a"]), _state(None, "a", ["a"])) is None


def test_merge_state_carries_forward_last_known_tiles():
    merged = delta._merge_state(_state(["a", "b"], "a", ["a"]), _state(None, "b", ["a"]))
    assert merged == _state(["a", "b"], "b", ["a"])


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
        (_panels_payload(["a", "b"], "b", ["a", "b"]), "active now b"),
        (_panels_payload(["a", "b"], "a", ["a", "b", "c"]), "c added to the launcher rail"),
    ],
    ids=["tiles", "active", "visible"],
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
    assert snapshot_file.read()["tiles"] == ["a", "b"]
