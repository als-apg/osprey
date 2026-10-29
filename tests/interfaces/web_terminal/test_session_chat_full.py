"""Tests for the replay reads of ``GET /api/session-chat``.

The endpoint serves two callers from one route: the diagnostics view, which
wants a capped preview of the conversation, and a view flip, which wants the
whole thing. These cover the second caller — the ``full`` flag, the
session-key-to-transcript mapping the route reads through, and the decision each
replayed call carries (``events=1``).
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from osprey.agent_record.client import ClientAnswer
from osprey.agent_runner.project_paths import CLAUDE_CONFIG_DIR_ENV, claude_project_dir
from osprey.interfaces.web_terminal.operator_session import _EXPIRED_MESSAGES
from osprey.interfaces.web_terminal.routes import session as session_routes
from osprey.interfaces.web_terminal.routes.session import (
    _APPROVED_CALL_NOTE,
    _DECLINED_RESULTS,
    router,
)
from osprey.mcp_server.workspace.transcript_reader import MAX_CHAT_MESSAGE_LENGTH
from tests.agent_record.fakes import FakeRecordClient
from tests.interfaces.web_terminal._fakes import armed_chat
from tests.interfaces.web_terminal._started_app import terminal_client

from .conftest import bare_route_app

SESSION_KEY = "11111111-2222-3333-4444-555555555555"
CLEARED_TRANSCRIPT = "99999999-8888-7777-6666-555555555555"


def _entry(role: str, text: str) -> dict:
    """A transcript entry holding one plain-text message."""
    return {
        "type": role,
        "timestamp": datetime(2026, 2, 19, 12, 0, 0, tzinfo=UTC).isoformat(),
        "sessionId": SESSION_KEY,
        "message": {"role": role, "content": [{"type": "text", "text": text}]},
    }


def _write(transcript_dir: Path, transcript_id: str, entries: list[dict]) -> None:
    """Write *entries* as the JSONL transcript named *transcript_id*."""
    path = transcript_dir / f"{transcript_id}.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")


@pytest.fixture
def transcripts(tmp_path, monkeypatch):
    """A project whose Claude Code transcripts live under ``tmp_path``, and a record serving them.

    The record is a fake answering each session's timeline with the nodes the
    agent record's own parser and timeline builder make of that session's
    transcript, so a test writes what the CLI writes and reads what the
    record would serve.
    """
    monkeypatch.setenv(CLAUDE_CONFIG_DIR_ENV, str(tmp_path / "claude-config"))
    project = tmp_path / "project"
    project.mkdir()
    transcript_dir = claude_project_dir(project)
    transcript_dir.mkdir(parents=True)

    def timeline(**kwargs) -> ClientAnswer:
        path = transcript_dir / f"{kwargs['session_id']}.jsonl"
        nodes = session_routes._transcript_nodes(path) if path.is_file() else []
        return ClientAnswer(data=nodes, complete=True)

    record = FakeRecordClient({"timeline": timeline})
    monkeypatch.setattr(session_routes, "record_client", lambda: record)
    monkeypatch.delenv("OSPREY_DISPATCH_RUN", raising=False)
    return project, transcript_dir


@pytest.fixture
def client(tmp_path, transcripts):
    """A Web Terminal client reading those transcripts, mapping nothing yet."""
    project, _ = transcripts
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    with terminal_client(workspace, project_dir=project) as c:
        c.app.state.transcript_map = {}
        c.app.state.transcript_map_provisional = False
        yield c


class TestSessionChatFull:
    def test_maps_key_to_its_current_transcript(self, client, transcripts):
        """A key that has cleared reads the transcript it moved to, not its own name."""
        _, transcript_dir = transcripts
        _write(transcript_dir, SESSION_KEY, [_entry("user", "before the clear")])
        _write(transcript_dir, CLEARED_TRANSCRIPT, [_entry("user", "after the clear")])
        client.app.state.transcript_map[SESSION_KEY] = CLEARED_TRANSCRIPT

        resp = client.get(f"/api/session-chat?session_id={SESSION_KEY}&full=1")

        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert data["turns"][0]["content"] == "after the clear"

    def test_unmapped_key_reads_its_own_transcript(self, client, transcripts):
        """A key that has never cleared needs no entry — it names its own transcript."""
        _, transcript_dir = transcripts
        _write(transcript_dir, SESSION_KEY, [_entry("user", "still the first conversation")])

        resp = client.get(f"/api/session-chat?session_id={SESSION_KEY}&full=1")

        assert resp.json()["turns"][0]["content"] == "still the first conversation"

    def test_full_leaves_long_messages_whole(self, client, transcripts):
        """``full=1`` returns the message at its real length."""
        _, transcript_dir = transcripts
        long_text = "x" * 5000
        _write(transcript_dir, SESSION_KEY, [_entry("assistant", long_text)])

        resp = client.get(f"/api/session-chat?session_id={SESSION_KEY}&full=1")

        assert resp.json()["turns"][0]["content"] == long_text

    def test_default_keeps_the_cap(self, client, transcripts):
        """Without the flag the per-message cap stands, as the diagnostics view reads it."""
        _, transcript_dir = transcripts
        _write(transcript_dir, SESSION_KEY, [_entry("assistant", "x" * 5000)])

        resp = client.get(f"/api/session-chat?session_id={SESSION_KEY}")

        assert resp.json()["turns"][0]["content"] == "x" * MAX_CHAT_MESSAGE_LENGTH + "..."

    def test_unknown_key_returns_an_empty_list(self, client):
        """A key with no transcript anywhere collapses to an empty conversation."""
        resp = client.get("/api/session-chat?session_id=no-such-key&full=1")

        assert resp.status_code == 200
        assert resp.json() == {"turns": [], "count": 0, "record": "complete"}

    def test_full_accepts_the_other_true_spellings(self, client, transcripts):
        """``true`` reads the same as ``1``; an unrecognized value keeps the cap."""
        _, transcript_dir = transcripts
        _write(transcript_dir, SESSION_KEY, [_entry("assistant", "y" * 5000)])

        whole = client.get(f"/api/session-chat?session_id={SESSION_KEY}&full=true")
        capped = client.get(f"/api/session-chat?session_id={SESSION_KEY}&full=maybe")

        assert len(whole.json()["turns"][0]["content"]) == 5000
        assert len(capped.json()["turns"][0]["content"]) < 5000


#: Every fixture entry carries this timestamp; no test reads a duration off it.
STAMP = "2026-10-23T05:56:12.493Z"
#: The text a Simple-view decline reaches the SDK with, and the CLI's rejection at its prompt.
SIMPLE_DECLINE = "The operator declined this tool call."
CLI_REJECTION = (
    "The user doesn't want to proceed with this tool use. The tool use was rejected "
    "(eg. if it was a file edit, the new_string was NOT written to the file). STOP what "
    "you are doing and wait for the user to tell you how to proceed."
)
#: A result a safety hook refused. The call never reached the operator.
HOOK_REFUSAL = (
    "PreToolUse:mcp__controls__channel_write hook error: WRITES OFF - this deployment "
    "refuses control-system writes."
)
HOOK_PATH = (
    Path(__file__).parents[3] / "src/osprey/templates/claude_code/claude/hooks/osprey_approval.py"
)


def _call(tool_use_id: str) -> dict:
    """An assistant entry calling one MCP tool."""
    block = {
        "type": "tool_use",
        "id": tool_use_id,
        "name": "mcp__python__execute",
        "input": {"code": "print(1)"},
    }
    message = {"role": "assistant", "content": [block]}
    return {"type": "assistant", "timestamp": STAMP, "message": message}


def _result(tool_use_id: str, text: str, *, is_error: bool = False) -> dict:
    """A user entry carrying one call's result, as the CLI writes it."""
    block = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": text,
        "is_error": is_error,
    }
    message = {"role": "user", "content": [block]}
    return {"type": "user", "timestamp": STAMP, "message": message}


def _note(tool_use_id: str, hook_event: str = "PostToolUse") -> dict:
    """The attachment the CLI writes when the approval note reaches a call's outcome hook."""
    attachment = {
        "type": "hook_additional_context",
        "content": [_APPROVED_CALL_NOTE],
        "hookName": f"{hook_event}:mcp__python__execute",
        "toolUseID": tool_use_id,
        "hookEvent": hook_event,
    }
    return {"type": "attachment", "timestamp": STAMP, "attachment": attachment}


@pytest.fixture
def replay_client(transcripts):
    """A client on the session routes alone, reading those transcripts and mapping nothing.

    A replay read needs the route, the project and the transcript map, and nothing the
    app's lifespan starts, so these tests skip booting the whole app.
    """
    project, _ = transcripts
    app = bare_route_app(
        router, project_cwd=str(project), transcript_map={}, transcript_map_provisional=False
    )
    with TestClient(app) as c:
        yield c


def _replayed_results(client, transcripts, entries: list[dict]) -> dict[str, dict]:
    """Write *entries* as the session's transcript; return each replayed result by call id."""
    _, transcript_dir = transcripts
    _write(transcript_dir, SESSION_KEY, entries)
    resp = client.get(f"/api/session-chat?session_id={SESSION_KEY}&events=1")
    assert resp.status_code == 200
    (agent_turn,) = [turn for turn in resp.json()["turns"] if turn["role"] == "assistant"]
    return {e["tool_use_id"]: e for e in agent_turn["events"] if e["type"] == "tool_result"}


class TestReplayedDecisions:
    """A replayed call's result carries the decision its row showed live, when there was one."""

    @pytest.mark.parametrize(
        ("text", "is_error", "note_event", "decision"),
        [
            # The approval note is the same in either view, so one check covers both.
            pytest.param("1", False, "PostToolUse", "allowed", id="approved-and-ran"),
            pytest.param(
                "NameError: name x is not defined",
                True,
                "PostToolUseFailure",
                "allowed",
                id="approved-and-failed",
            ),
            pytest.param(SIMPLE_DECLINE, True, None, "denied", id="simple-view-decline"),
            pytest.param(CLI_REJECTION, True, None, "denied", id="expert-view-rejection"),
            # Nobody answered, so the row shows neither the operator's "No" nor an expiry
            # the transcript does not record.
            pytest.param(_EXPIRED_MESSAGES["tool"], True, None, None, id="expired-ask"),
            # A refusal was never put to the operator, so its row shows no pill.
            pytest.param(HOOK_REFUSAL, True, None, None, id="hook-refusal"),
            pytest.param("1", False, None, None, id="no-note-and-no-decline"),
            # Only an error result can be a decline; a tool that printed the words did not.
            pytest.param(SIMPLE_DECLINE, False, None, None, id="decline-words-printed"),
        ],
    )
    def test_a_replayed_call_shows_the_decision_its_row_showed_live(
        self, replay_client, transcripts, text, is_error, note_event, decision
    ):
        entries = [
            _entry("user", "run it"),
            _call("toolu_1"),
            _result("toolu_1", text, is_error=is_error),
        ]
        if note_event is not None:
            entries.append(_note("toolu_1", hook_event=note_event))

        result = _replayed_results(replay_client, transcripts, entries)["toolu_1"]

        assert result["decision"] == decision
        assert result["is_error"] is is_error

    def test_a_note_marks_only_the_call_it_names(self, replay_client, transcripts):
        results = _replayed_results(
            replay_client,
            transcripts,
            [
                _entry("user", "run both"),
                _call("toolu_first"),
                _call("toolu_second"),
                _result("toolu_first", "1"),
                _result("toolu_second", "2"),
                _note("toolu_first"),
            ],
        )

        assert results["toolu_first"]["decision"] == "allowed"
        assert results["toolu_second"]["decision"] is None


class TestReplayedBookkeeping:
    """What the CLI writes for itself never reaches the replay as the operator's words."""

    @pytest.mark.parametrize(
        "marker",
        ["[Request interrupted by user for tool use]", "[Request interrupted by user]"],
    )
    def test_an_interrupt_marker_is_no_operator_bubble(self, replay_client, transcripts, marker):
        """The marker after an Expert-view rejection stands alone, beside the result, or both."""
        _, transcript_dir = transcripts
        beside = _result("toolu_no", CLI_REJECTION, is_error=True)
        beside["message"]["content"].append({"type": "text", "text": marker})
        _write(
            transcript_dir,
            SESSION_KEY,
            [
                _entry("user", "run it"),
                _call("toolu_no"),
                beside,
                _entry("user", marker),
                _entry("user", "fine, skip it"),
            ],
        )

        resp = replay_client.get(f"/api/session-chat?session_id={SESSION_KEY}&events=1")
        turns = resp.json()["turns"]

        assert [t["content"] for t in turns if t["role"] == "user"] == ["run it", "fine, skip it"]
        (agent_turn,) = [t for t in turns if t["role"] == "assistant"]
        (result,) = [e for e in agent_turn["events"] if e["type"] == "tool_result"]
        assert result["decision"] == "denied"

    def test_an_image_result_replays_as_a_placeholder_without_its_bytes(
        self, replay_client, transcripts
    ):
        """The live chat and the replay read the same text: never the image's base64."""
        image = {"type": "image", "source": {"type": "base64", "data": "QUJD" * 400}}
        results = _replayed_results(
            replay_client,
            transcripts,
            [_entry("user", "look"), _call("toolu_img"), _result("toolu_img", [image])],
        )

        assert results["toolu_img"]["preview"] == "[image]"
        assert "QUJD" not in json.dumps(results)


def test_the_approval_note_is_the_hooks_own_text():
    """The replay matches the note the approval hook writes, so the two copies cannot drift."""
    spec = importlib.util.spec_from_file_location("osprey_approval_pin", HOOK_PATH)
    hook = importlib.util.module_from_spec(spec)
    hooks_dir = str(HOOK_PATH.parent)
    try:
        spec.loader.exec_module(hook)
    finally:
        # The hook puts its own directory on sys.path to import its sibling; take that back.
        sys.path[:] = [entry for entry in sys.path if os.path.abspath(entry) != hooks_dir]

    assert _APPROVED_CALL_NOTE == hook.APPROVED_CALL_NOTE


async def test_the_decline_the_chat_sends_is_one_the_replay_reads(tmp_path):
    """A decline given in the Simple view reaches the SDK as a text the replay reads."""
    async with armed_chat(tmp_path) as chat:
        answer, card = await chat.ask()
        assert await chat.session.resolve_permission(card["request_id"], allow=False)
        declined = await asyncio.wait_for(answer, timeout=2.0)

    assert declined.message in _DECLINED_RESULTS
