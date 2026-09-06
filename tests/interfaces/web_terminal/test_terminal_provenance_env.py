"""Tests for the terminal PTY provenance-session env injection.

The terminal surface forces its ``claude`` onto a known session UUID
(``--session-id``) and injects that id as ``OSPREY_TELEMETRY_SESSION_ID`` so the
workspace provenance_locator tool can hand it back for a filed issue. That id
is also this session's pool key, so it is stamped as ``OSPREY_SESSION_ID`` too
— one key names the conversation on every surface. The two variables stay
separate names because they answer different questions and part company on the
``switch_session`` path, which has a session key and no telemetry id.

``OSPREY_SESSION_ID`` on all three spawn paths is pinned in
``test_child_env_session_id.py``; what is pinned here is the telemetry pair.
"""

from datetime import datetime
from types import SimpleNamespace

from osprey.interfaces.web_terminal.routes.websocket import _build_extra_env


def _ws(hooks_env=None):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(hooks_env=hooks_env or {})))


def test_new_session_injects_the_telemetry_pair():
    env = _build_extra_env(_ws(), claude_session_id=None, telemetry_session_id="forced-uuid")
    assert env["OSPREY_TELEMETRY_SESSION_ID"] == "forced-uuid"
    # start stamp is present and ISO-8601 parseable
    datetime.fromisoformat(env["OSPREY_TELEMETRY_SESSION_START"])
    # the forced id is the pool key, so it is the session id as well
    assert env["OSPREY_SESSION_ID"] == "forced-uuid"


def test_resume_sets_both_ids():
    env = _build_extra_env(_ws(), claude_session_id="sid", telemetry_session_id="sid")
    assert env["OSPREY_SESSION_ID"] == "sid"
    assert env["OSPREY_TELEMETRY_SESSION_ID"] == "sid"


def test_no_telemetry_id_sets_no_telemetry_vars():
    env = _build_extra_env(_ws(), claude_session_id=None, telemetry_session_id=None)
    assert "OSPREY_TELEMETRY_SESSION_ID" not in env
    assert "OSPREY_TELEMETRY_SESSION_START" not in env


def test_switch_session_stamps_the_key_without_a_telemetry_id():
    """The two names part company: a switch names a session, not a new run."""
    env = _build_extra_env(_ws(), claude_session_id="sid", telemetry_session_id=None)
    assert env["OSPREY_SESSION_ID"] == "sid"
    assert "OSPREY_TELEMETRY_SESSION_ID" not in env


def test_hooks_env_still_merged():
    env = _build_extra_env(_ws(hooks_env={"OSPREY_HOOK_X": "1"}), None, "t")
    assert env["OSPREY_HOOK_X"] == "1"
    assert env["OSPREY_TELEMETRY_SESSION_ID"] == "t"


def test_every_pty_session_is_marked_expert_surface():
    """PTY sessions serve the expert web surface — new and resumed alike."""
    assert _build_extra_env(_ws(), None, None)["OSPREY_WEB_UX"] == "expert"
    assert _build_extra_env(_ws(), "sid", "sid")["OSPREY_WEB_UX"] == "expert"
