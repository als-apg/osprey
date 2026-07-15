"""Tests for the terminal PTY provenance-session env injection.

The terminal surface forces its ``claude`` onto a known session UUID
(``--session-id``) and injects that id as ``OSPREY_TELEMETRY_SESSION_ID`` so the
workspace provenance_locator tool can hand it back for a filed issue. Crucially,
the telemetry id must NOT be conflated with ``OSPREY_SESSION_ID`` (which
relocates the artifact store) — a new terminal session gets the telemetry var
but keeps its default artifact root.
"""

from datetime import datetime
from types import SimpleNamespace

from osprey.interfaces.web_terminal.routes.websocket import _build_extra_env


def _ws(hooks_env=None):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(hooks_env=hooks_env or {})))


def test_new_session_injects_telemetry_id_without_relocating_artifacts():
    env = _build_extra_env(_ws(), claude_session_id=None, telemetry_session_id="forced-uuid")
    assert env["OSPREY_TELEMETRY_SESSION_ID"] == "forced-uuid"
    # start stamp is present and ISO-8601 parseable
    datetime.fromisoformat(env["OSPREY_TELEMETRY_SESSION_START"])
    # new session must NOT set the artifact-relocating var
    assert "OSPREY_SESSION_ID" not in env


def test_resume_sets_both_ids():
    env = _build_extra_env(_ws(), claude_session_id="sid", telemetry_session_id="sid")
    assert env["OSPREY_SESSION_ID"] == "sid"
    assert env["OSPREY_TELEMETRY_SESSION_ID"] == "sid"


def test_no_telemetry_id_sets_no_telemetry_vars():
    env = _build_extra_env(_ws(), claude_session_id=None, telemetry_session_id=None)
    assert "OSPREY_TELEMETRY_SESSION_ID" not in env
    assert "OSPREY_TELEMETRY_SESSION_START" not in env


def test_hooks_env_still_merged():
    env = _build_extra_env(_ws(hooks_env={"OSPREY_HOOK_X": "1"}), None, "t")
    assert env["OSPREY_HOOK_X"] == "1"
    assert env["OSPREY_TELEMETRY_SESSION_ID"] == "t"
