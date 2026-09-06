"""The turn-state hook, its wiring, and the server's detection of both.

A web terminal may move a conversation between its two surfaces only while the
process holding it is between turns. Nothing in the terminal's output says when
that is, so the process reports its own turn edges: ``osprey_turn_state.py``
POSTs to ``/api/agent-turn`` on the four events that carry one.

Three claims are pinned here, because a break in any of them is silent.

* **The wiring.** ``settings.json.j2`` registers the hook on ``UserPromptSubmit``
  (a turn starts), on ``Stop`` and ``StopFailure`` (a turn ends), and on
  ``SessionStart`` under an explicit ``startup|resume|clear`` matcher.
  ``compact`` must never match: auto-compaction fires *inside* a running turn,
  and its idle report would invite a teardown mid-answer.
* **The hook.** It reports the right state for each event, carries both
  identities (the never-changing pool key and Claude Code's own session id,
  which names the transcript), authenticates with the panel token, and is a
  silent no-op outside a web terminal. Every path exits 0 and writes nothing to
  stdout — this hook runs on the operator's prompt and on the end of their turn,
  and neither may be blocked by it.
* **The detection.** ``app.state.turn_hook_present`` is a permission, not a
  prediction: it says an idle edge can actually arrive, so a deployment whose
  settings carry no turn hook is never left waiting for one.

The hook is exercised in-process against a stub terminal on loopback, so no
test here reaches a real server or a real Claude Code.
"""

from __future__ import annotations

import io
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import osprey.templates.claude_code.claude.hooks.osprey_turn_state as turn_state
from osprey.cli.templates.manager import TemplateManager
from osprey.interfaces.web_terminal.app import (
    TURN_HOOK_SESSION_START_MATCHER,
    TURN_STATE_HOOK,
    _detect_turn_hook,
)

SETTINGS_TEMPLATE = "claude_code/claude/settings.json.j2"

PORT_ENV = "OSPREY_WEB_PORT"
KEY_ENV = "OSPREY_SESSION_ID"
SURFACE_ENV = "OSPREY_WEB_UX"
TOKEN_ENV = "OSPREY_PANEL_TOKEN"

#: A session key and a transcript id that are deliberately different values —
#: the whole point of the two-identity payload is that a ``/clear`` moves one
#: and not the other, and a test reusing one value for both would pass while
#: the hook swapped them.
POOL_KEY = "11111111-2222-3333-4444-555555555555"
TRANSCRIPT_ID = "99999999-8888-7777-6666-555555555555"

#: Context sufficient to render ``settings.json.j2``. Keys the hooks block does
#: not read are supplied only because the permissions block above it would fail
#: on an undefined mapping.
_BASE_CTX: dict[str, Any] = {
    "current_python_env": "/usr/bin/python3",
    "agent_data_root": "/tmp/turn-state-demo/agent_data",
    "servers": [],
    "agents": [],
    "facility_permissions": {},
    "deny_defaults": ["Bash"],
    "declared_hooks": {},
    "declared_extra_events": [],
    "framework_pre_hooks": [],
    "framework_post_hooks": [],
}


def _no_duplicate_keys(pairs):
    """``object_pairs_hook`` that refuses a JSON object with a repeated key.

    ``json.loads`` keeps the last of a repeated key and drops the rest without
    a word, so a template that emitted ``"Stop"`` twice — once from the
    framework, once from a profile's declaration — would parse into settings
    that silently lost the framework's entry. This turns that into a failure.
    """
    seen: dict[str, Any] = {}
    for key, value in pairs:
        assert key not in seen, f"duplicate JSON key: {key}"
        seen[key] = value
    return seen


def render_settings(**overrides: Any) -> dict[str, Any]:
    """Render ``settings.json.j2`` and parse it, refusing duplicate keys."""
    rendered = (
        TemplateManager()
        .jinja_env.get_template(SETTINGS_TEMPLATE)
        .render(**{**_BASE_CTX, **overrides})
    )
    return json.loads(rendered, object_pairs_hook=_no_duplicate_keys)


def turn_hook_entries(settings: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Map each event to the entries whose commands name the turn-state hook."""
    found: dict[str, list[dict[str, Any]]] = {}
    for event, entries in settings["hooks"].items():
        for entry in entries:
            if any(TURN_STATE_HOOK in hook["command"] for hook in entry["hooks"]):
                found.setdefault(event, []).append(entry)
    return found


def declared_rule(hook_file: str, matcher: str | None = None) -> dict[str, Any]:
    """A profile-declared hook rule in the shape the renderer hands the template."""
    rule: dict[str, Any] = {
        "hooks": [
            {
                "type": "command",
                "command": f'python3 "$CLAUDE_PROJECT_DIR/.claude/hooks/{hook_file}"',
                "timeout": 60,
            }
        ]
    }
    if matcher is not None:
        rule["matcher"] = matcher
    return rule


# ---------------------------------------------------------------------------
# The wiring: what settings.json.j2 registers
# ---------------------------------------------------------------------------


def test_settings_register_the_turn_hook_on_all_four_edges():
    """Both edges of a turn, a failed turn, and a session opening are wired."""
    events = turn_hook_entries(render_settings())

    assert set(events) == {"UserPromptSubmit", "Stop", "StopFailure", "SessionStart"}
    for event, entries in events.items():
        assert len(entries) == 1, f"{event} registers the hook more than once"


def test_settings_session_start_matcher_excludes_compaction():
    """The SessionStart registration names the three real starts, never compact."""
    entry = turn_hook_entries(render_settings())["SessionStart"][0]

    assert entry["matcher"] == TURN_HOOK_SESSION_START_MATCHER
    assert "compact" not in entry["matcher"]


def test_settings_turn_hook_command_fails_open():
    """Every registration runs the shipped file and swallows its own failure.

    The suffix is what the other framework entries carry: a hook that cannot be
    found or cannot run must not turn into a non-zero exit that Claude Code
    reports to the operator.
    """
    for entries in turn_hook_entries(render_settings()).values():
        commands = [h["command"] for h in entries[0]["hooks"] if TURN_STATE_HOOK in h["command"]]
        assert len(commands) == 1
        assert commands[0].endswith(f'{TURN_STATE_HOOK}" 2>/dev/null || true')
        assert commands[0].startswith('"/usr/bin/python3"')


def test_settings_keep_the_other_session_start_hooks_unmatched():
    """The framework's existing SessionStart entry keeps running on every source.

    The matcher belongs to the turn hook alone. Attaching it to the entry that
    carries the config-drift and panels-context hooks would quietly stop those
    from running after a compaction.
    """
    entries = render_settings()["hooks"]["SessionStart"]
    others = [e for e in entries if all(TURN_STATE_HOOK not in h["command"] for h in e["hooks"])]

    assert len(others) == 1
    assert "matcher" not in others[0]


def test_settings_stop_absorbs_a_declared_rule_into_one_key():
    """A profile declaring its own Stop hook extends the framework's array.

    ``Stop`` is now framework-wired, so a declared rule must be appended to the
    same key rather than emitted as a second ``"Stop"`` — which would be legal
    JSON whose framework half is dropped on parse. ``render_settings`` refuses
    duplicate keys, so this test fails loudly either way.
    """
    settings = render_settings(
        declared_hooks={"Stop": [declared_rule("facility_audit.py")]},
        declared_extra_events=["Stop"],
    )

    commands = [h["command"] for entry in settings["hooks"]["Stop"] for h in entry["hooks"]]
    assert any(TURN_STATE_HOOK in command for command in commands)
    assert any("facility_audit.py" in command for command in commands)


def test_settings_still_emit_declared_events_the_framework_does_not_wire():
    """Filtering Stop out of the extra-events loop leaves the other events alone."""
    settings = render_settings(
        declared_hooks={"SessionEnd": [declared_rule("facility_farewell.py")]},
        declared_extra_events=["SessionEnd"],
    )

    commands = [h["command"] for entry in settings["hooks"]["SessionEnd"] for h in entry["hooks"]]
    assert any("facility_farewell.py" in command for command in commands)


# ---------------------------------------------------------------------------
# The hook: what it sends, and when it stays quiet
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_ambient_terminal(monkeypatch):
    """No inherited web-terminal environment: each test states what it wants."""
    for name in (PORT_ENV, KEY_ENV, SURFACE_ENV, TOKEN_ENV):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def turn_server():
    """Stub web terminal recording ``POST /api/agent-turn`` calls.

    Yields ``(port, received)``; a call is appended before the response is
    written, so a hook that has returned has necessarily already been recorded.
    """
    received: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - http.server API
            length = int(self.headers.get("Content-Length", 0))
            received.append(
                {
                    "path": self.path,
                    "body": json.loads(self.rfile.read(length) or b"{}"),
                    "content_type": self.headers.get("Content-Type"),
                    "authorization": self.headers.get("Authorization"),
                }
            )
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1], received
    finally:
        server.shutdown()
        server.server_close()


def run_hook(monkeypatch, payload, *, port, surface="expert", key=POOL_KEY, token=None):
    """Drive the hook's ``main`` over *payload*, returning its exit status."""
    if port is not None:
        monkeypatch.setenv(PORT_ENV, str(port))
    if key is not None:
        monkeypatch.setenv(KEY_ENV, key)
    if surface is not None:
        monkeypatch.setenv(SURFACE_ENV, surface)
    if token is not None:
        monkeypatch.setenv(TOKEN_ENV, token)
    monkeypatch.setattr(turn_state.sys, "stdin", io.StringIO(json.dumps(payload)))
    return turn_state.main()


def record_calls(monkeypatch, *, raises=None):
    """Stub the hook's ``urlopen`` and hand back the requests it was given."""
    seen = []

    def _fake(target, timeout=None):
        seen.append(target)
        if raises is not None:
            raise raises
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(turn_state.urllib.request, "urlopen", _fake)
    return seen


def payload_for(event, *, session_id=TRANSCRIPT_ID, **extra):
    """A hook stdin payload for *event*, shaped as Claude Code sends it."""
    return {"hook_event_name": event, "session_id": session_id, **extra}


def test_reports_busy_when_a_prompt_starts_a_turn(monkeypatch, turn_server, capsys):
    port, received = turn_server

    assert run_hook(monkeypatch, payload_for("UserPromptSubmit"), port=port) == 0

    assert len(received) == 1
    call = received[0]
    assert call["path"] == "/api/agent-turn"
    assert call["content_type"] == "application/json"
    assert call["body"]["state"] == "busy"
    assert capsys.readouterr().out == ""


def test_reports_the_two_identities_and_the_surface(monkeypatch, turn_server):
    """The body carries the pool key, the transcript id and the surface apart."""
    port, received = turn_server

    assert run_hook(monkeypatch, payload_for("Stop"), port=port) == 0

    body = received[0]["body"]
    assert body["pool_key"] == POOL_KEY
    assert body["session_id"] == TRANSCRIPT_ID
    assert body["surface"] == "expert"
    assert body["state"] == "idle"
    assert body["source"] == "Stop"
    assert isinstance(body["ts"], float)
    assert set(body) == {"session_id", "pool_key", "state", "surface", "ts", "source"}


@pytest.mark.parametrize(
    ("event", "state"),
    [
        ("UserPromptSubmit", "busy"),
        ("Stop", "idle"),
        ("StopFailure", "idle"),
        ("SessionStart", "idle"),
    ],
)
def test_each_wired_event_reports_its_state(monkeypatch, turn_server, event, state):
    port, received = turn_server

    assert run_hook(monkeypatch, payload_for(event), port=port) == 0

    assert [call["body"]["state"] for call in received] == [state]


def test_a_session_start_after_compaction_reports_nothing(monkeypatch):
    """The matcher keeps compact out; the hook refuses it a second time.

    A hand-edited ``settings.json`` that dropped the matcher would otherwise
    have the hook announce idleness in the middle of the turn the compaction
    happened in.
    """
    seen = record_calls(monkeypatch)

    assert run_hook(monkeypatch, payload_for("SessionStart", source="compact"), port=10100) == 0

    assert seen == []


def test_a_turn_ending_is_reported_whatever_the_payload_calls_its_source(monkeypatch, turn_server):
    """The compaction backstop asks only about SessionStart.

    ``source`` means "which kind of session start"; on any other event the word
    is not that, so a Stop that happened to carry it must still report the turn
    it ended.
    """
    port, received = turn_server

    assert run_hook(monkeypatch, payload_for("Stop", source="compact"), port=port) == 0

    assert [call["body"]["state"] for call in received] == ["idle"]


@pytest.mark.parametrize("source", ["startup", "resume", "clear"])
def test_the_other_session_start_sources_are_reported(monkeypatch, turn_server, source):
    port, received = turn_server

    assert run_hook(monkeypatch, payload_for("SessionStart", source=source), port=port) == 0

    assert len(received) == 1


@pytest.mark.parametrize("event", ["PreToolUse", "SubagentStop", "", None])
def test_an_event_that_is_not_a_turn_edge_reports_nothing(monkeypatch, event):
    seen = record_calls(monkeypatch)

    assert run_hook(monkeypatch, payload_for(event), port=10100) == 0

    assert seen == []


def test_a_payload_without_a_transcript_id_reports_nothing(monkeypatch):
    """A blank id would erase the key's recorded transcript, not update it."""
    seen = record_calls(monkeypatch)

    assert run_hook(monkeypatch, payload_for("Stop", session_id=""), port=10100) == 0

    assert seen == []


def test_unreadable_stdin_reports_nothing(monkeypatch):
    seen = record_calls(monkeypatch)
    monkeypatch.setenv(PORT_ENV, "10100")
    monkeypatch.setenv(KEY_ENV, POOL_KEY)
    monkeypatch.setattr(turn_state.sys, "stdin", io.StringIO("not json"))

    assert turn_state.main() == 0
    assert seen == []


@pytest.mark.parametrize("absent", [PORT_ENV, KEY_ENV])
def test_a_session_outside_the_web_terminal_reports_nothing(monkeypatch, absent, capsys):
    """No port or no session key means there is no server that knows this session."""
    seen = record_calls(monkeypatch)
    monkeypatch.setenv(PORT_ENV, "10100")
    monkeypatch.setenv(KEY_ENV, POOL_KEY)
    monkeypatch.delenv(absent)
    monkeypatch.setattr(turn_state.sys, "stdin", io.StringIO(json.dumps(payload_for("Stop"))))

    assert turn_state.main() == 0
    assert seen == []
    assert capsys.readouterr().out == ""


def test_the_simple_surface_still_reports(monkeypatch, turn_server):
    """The hook does not decide which surface counts — the server drops the rest.

    Keeping one code path here means a surface the server later starts caring
    about needs no change in a file that ships into every deployment.
    """
    port, received = turn_server

    assert run_hook(monkeypatch, payload_for("Stop"), port=port, surface="simple") == 0

    assert received[0]["body"]["surface"] == "simple"


def test_a_missing_surface_variable_reports_an_empty_surface(monkeypatch, turn_server):
    port, received = turn_server

    assert run_hook(monkeypatch, payload_for("Stop"), port=port, surface=None) == 0

    assert received[0]["body"]["surface"] == ""


@pytest.mark.parametrize(("value", "reported"), [("  expert\n", "expert"), ("   ", "")])
def test_the_surface_variable_is_read_stripped(monkeypatch, turn_server, value, reported):
    """The same rule the other env reads apply: padding is not part of the value.

    An uninterpolated compose variable arrives blank and a hand-edited one can
    arrive padded; the server compares the surface for equality, so a padded
    ``expert`` would be dropped as an unrecognised surface.
    """
    port, received = turn_server

    assert run_hook(monkeypatch, payload_for("Stop"), port=port, surface=value) == 0

    assert received[0]["body"]["surface"] == reported


def test_the_panel_token_authenticates_the_report(monkeypatch, turn_server):
    port, received = turn_server

    assert run_hook(monkeypatch, payload_for("Stop"), port=port, token="s3cret-panel-token") == 0

    assert received[0]["authorization"] == "Bearer s3cret-panel-token"


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_a_blank_panel_token_sends_no_bearer(monkeypatch, blank):
    """A blank carrier is absent, not a credential — the same rule the server applies."""
    seen = record_calls(monkeypatch)

    assert run_hook(monkeypatch, payload_for("Stop"), port=10100, token=blank) == 0

    assert seen[0].get_header("Authorization") is None


def test_a_web_terminal_that_refuses_the_report_costs_only_the_report(monkeypatch, capsys):
    """An unreachable or refusing terminal must not fail the turn that reported."""
    record_calls(monkeypatch, raises=OSError("connection refused"))

    assert run_hook(monkeypatch, payload_for("Stop"), port=10100) == 0
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# The detection: what the server concludes from the rendered settings
# ---------------------------------------------------------------------------


def project_with(settings: Any, tmp_path: Path):
    """Write *settings* into a project tree and return an app pointed at it."""
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir(exist_ok=True)
    if settings is not None:
        claude_dir.joinpath("settings.json").write_text(
            settings if isinstance(settings, str) else json.dumps(settings)
        )
    return SimpleNamespace(
        state=SimpleNamespace(project_cwd=str(tmp_path), turn_hook_present=False)
    )


def test_a_rendered_project_is_detected_as_reporting_turns(tmp_path, caplog):
    """The template's own output satisfies the detection it is read by."""
    app = project_with(render_settings(), tmp_path)

    with caplog.at_level(logging.WARNING, logger="osprey.interfaces.web_terminal.app"):
        _detect_turn_hook(app)

    assert app.state.turn_hook_present is True
    assert caplog.records == []


def test_settings_without_a_stop_entry_are_not_turn_reporting(tmp_path):
    settings = render_settings()
    del settings["hooks"]["Stop"]

    app = project_with(settings, tmp_path)
    _detect_turn_hook(app)

    assert app.state.turn_hook_present is False


def test_settings_whose_session_start_lost_its_matcher_are_not_turn_reporting(tmp_path):
    """Without the matcher the hook also runs on compact, so it is not trusted."""
    settings = render_settings()
    for entry in settings["hooks"]["SessionStart"]:
        entry.pop("matcher", None)

    app = project_with(settings, tmp_path)
    _detect_turn_hook(app)

    assert app.state.turn_hook_present is False


def test_settings_without_stop_failure_still_report_turns_but_warn(tmp_path, caplog):
    """A missing failure edge is a gap worth naming, not a reason to distrust the rest."""
    settings = render_settings()
    del settings["hooks"]["StopFailure"]

    app = project_with(settings, tmp_path)
    with caplog.at_level(logging.WARNING, logger="osprey.interfaces.web_terminal.app"):
        _detect_turn_hook(app)

    assert app.state.turn_hook_present is True
    assert "StopFailure" in caplog.text


def test_a_project_without_settings_reports_no_turns(tmp_path):
    app = project_with(None, tmp_path)
    _detect_turn_hook(app)

    assert app.state.turn_hook_present is False


def test_unreadable_settings_report_no_turns(tmp_path):
    """A malformed artifact reads as "no hook" rather than breaking the boot."""
    app = project_with("{ not json", tmp_path)
    _detect_turn_hook(app)

    assert app.state.turn_hook_present is False
