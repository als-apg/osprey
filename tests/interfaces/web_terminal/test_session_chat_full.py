"""Tests for the replay reads of ``GET /api/session-chat``.

The endpoint serves two callers from one route: the diagnostics view, which
wants a capped preview of the conversation, and a surface change, which wants the
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
from osprey.interfaces.web_terminal.operator_session import _ASYNC_LAUNCH_MARK, _EXPIRED_MESSAGES
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
        "timestamp": STAMP,
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
#: The time every entry of a test transcript carries unless the test sets one: the
#: record orders a session by time, and lines of one instant in their file order.
STAMP = datetime(2026, 2, 19, 12, 0, 0, tzinfo=UTC).isoformat()
#: The text a chat-view decline reaches the SDK with, and the CLI's rejection at its prompt.
CHAT_DECLINE = "The operator declined this tool call."
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
            pytest.param(CHAT_DECLINE, True, None, "denied", id="chat-decline"),
            pytest.param(CLI_REJECTION, True, None, "denied", id="terminal-rejection"),
            # Nobody answered, so the row shows neither the operator's "No" nor an expiry
            # the transcript does not record.
            pytest.param(_EXPIRED_MESSAGES["tool"], True, None, None, id="expired-ask"),
            # A refusal was never put to the operator, so its row shows no pill.
            pytest.param(HOOK_REFUSAL, True, None, None, id="hook-refusal"),
            pytest.param("1", False, None, None, id="no-note-and-no-decline"),
            # Only an error result can be a decline; a tool that printed the words did not.
            pytest.param(CHAT_DECLINE, False, None, None, id="decline-words-printed"),
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
        """The marker after a terminal-view rejection stands alone, beside the result, or both."""
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


def _agent_call(tool_use_id: str) -> dict:
    """An assistant entry starting one subagent."""
    block = {
        "type": "tool_use",
        "id": tool_use_id,
        "name": "Agent",
        "input": {"subagent_type": "data-visualizer", "description": "Plot the tunes"},
    }
    return {
        "type": "assistant",
        "timestamp": STAMP,
        "message": {"role": "assistant", "content": [block]},
    }


def _write_subagent(transcript_dir: Path, agent_id: str, entries: list[dict]) -> None:
    """Write a subagent's transcript where the CLI writes it, beside its session's."""
    folder = transcript_dir / SESSION_KEY / "subagents"
    folder.mkdir(parents=True, exist_ok=True)
    sidechain = [{**e, "isSidechain": True} for e in entries]
    (folder / f"agent-{agent_id}.jsonl").write_text(
        "\n".join(json.dumps(e) for e in sidechain) + "\n"
    )


class TestReplayedSubagents:
    """A subagent's chip replays with the activity it showed live, from the stream the
    record holds for it: nested under its call once the call's result names the agent,
    and, while it still works, the stream the call without a result waits on."""

    def _agent_turn(self, client, transcripts, result: dict | None) -> dict:
        _, transcript_dir = transcripts
        main = [_entry("user", "plot the tunes"), _agent_call("toolu_agent")]
        if result is not None:
            main += [result, _entry("assistant", "Here is the plot.")]
        _write(transcript_dir, SESSION_KEY, main)
        prompt = {
            "type": "user",
            "timestamp": STAMP,
            "message": {"role": "user", "content": "Plot Qx vs Qy."},
        }
        _write_subagent(
            transcript_dir,
            "a1",
            [
                prompt,
                _call("toolu_plot"),
                _result("toolu_plot", "saved"),
                _entry("assistant", "Plotted."),
            ],
        )
        resp = client.get(f"/api/session-chat?session_id={SESSION_KEY}&events=1")
        (agent_turn,) = [t for t in resp.json()["turns"] if t["role"] == "assistant"]
        return agent_turn

    def test_a_subagents_work_follows_its_call_under_that_call(self, replay_client, transcripts):
        # The CLI names the agent a call started in the call's structured result.
        named = {
            **_result("toolu_agent", "Created the tune diagram plot."),
            "toolUseResult": {"agentId": "a1", "status": "completed"},
        }
        turn = self._agent_turn(replay_client, transcripts, named)

        shape = [
            (e["type"], e.get("tool_use_id"), e.get("parent_tool_use_id")) for e in turn["events"]
        ]
        assert shape[:4] == [
            ("tool_use", "toolu_agent", None),
            ("tool_use", "toolu_plot", "toolu_agent"),
            ("tool_result", "toolu_plot", "toolu_agent"),
            ("text", None, "toolu_agent"),
        ]
        # The prompt it was started with is not the agent speaking.
        assert "Plot Qx vs Qy." not in json.dumps(turn["events"])
        # The subagent's words stay in its chip, out of the main agent's text.
        assert turn["content"] == "Here is the plot."

    def test_a_subagent_started_by_another_call_is_not_attached(self, replay_client, transcripts):
        other = {
            **_result("toolu_agent", "Created the tune diagram plot."),
            "toolUseResult": {"agentId": "elsewhere", "status": "completed"},
        }
        turn = self._agent_turn(replay_client, transcripts, other)

        assert not [e for e in turn["events"] if e.get("parent_tool_use_id")]

    def test_a_subagent_still_at_work_follows_the_call_waiting_on_it(
        self, replay_client, transcripts
    ):
        turn = self._agent_turn(replay_client, transcripts, None)

        shape = [
            (e["type"], e.get("tool_use_id"), e.get("parent_tool_use_id")) for e in turn["events"]
        ]
        assert shape == [
            ("tool_use", "toolu_agent", None),
            ("tool_use", "toolu_plot", "toolu_agent"),
            ("tool_result", "toolu_plot", "toolu_agent"),
            ("text", None, "toolu_agent"),
        ]
        # The call waits on its agent: the turn is still running.
        assert not [e for e in turn["events"] if e["type"] == "result"]


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
    """A decline given in the chat reaches the SDK as a text the replay reads."""
    async with armed_chat(tmp_path) as chat:
        answer, card = await chat.ask()
        assert await chat.session.resolve_permission(card["request_id"], allow=False)
        declined = await asyncio.wait_for(answer, timeout=2.0)

    assert declined.message in _DECLINED_RESULTS


def _at(entry: dict, second: int | None) -> dict:
    """*entry* stamped *second* seconds past noon, or with no timestamp at all."""
    if second is None:
        entry.pop("timestamp", None)
    else:
        entry["timestamp"] = datetime(2026, 2, 19, 12, 0, second, tzinfo=UTC).isoformat()
    return entry


def _thinking(second: int | None) -> dict:
    entry = _entry("assistant", "")
    entry["message"]["content"] = [{"type": "thinking", "thinking": "Let me check."}]
    return _at(entry, second)


@pytest.mark.parametrize(
    ("prompt_at", "duration"),
    [
        pytest.param(0, 3000, id="from-the-entry-before"),
        # The record stamps every entry, so a reasoning lacks a time only when no
        # entry of the conversation came before it.
        pytest.param(None, None, id="nothing-before-it"),
    ],
)
def test_replayed_reasoning_says_how_long_it_took(replay_client, transcripts, prompt_at, duration):
    """The time runs from the entry before the reasoning to the entry that holds it."""
    _, transcript_dir = transcripts
    entries = [
        *([_at(_entry("user", "check the orbit"), prompt_at)] if prompt_at is not None else []),
        _thinking(3),
        _at(_entry("assistant", "It is fine."), 4),
    ]
    _write(transcript_dir, SESSION_KEY, entries)
    resp = replay_client.get(f"/api/session-chat?session_id={SESSION_KEY}&events=1")
    (agent_turn,) = [t for t in resp.json()["turns"] if t["role"] == "assistant"]
    (thought,) = [e for e in agent_turn["events"] if e["type"] == "thinking"]
    assert thought.get("duration_ms") == duration


class TestReplayedSlashCommands:
    """A slash command the operator ran replays as they typed it; a skill also as its row."""

    @staticmethod
    def _echo(name: str, args: str = "") -> dict:
        """The echo CLI 2.1.300 writes for a skill or a command file: string content, no isMeta."""
        entry = _entry("user", "")
        entry["message"]["content"] = (
            f"<command-message>{name}</command-message>\n<command-name>/{name}</command-name>"
            + (f"\n<command-args>{args}</command-args>" if args else "")
        )
        return entry

    @staticmethod
    def _loaded(text: str) -> dict:
        """The meta entry holding the instructions the CLI injected for the command."""
        return {**_entry("user", text), "isMeta": True}

    def _replay(self, client, transcripts, entries: list[dict]) -> list[dict]:
        _, transcript_dir = transcripts
        _write(transcript_dir, SESSION_KEY, entries)
        resp = client.get(f"/api/session-chat?session_id={SESSION_KEY}&events=1")
        assert resp.status_code == 200
        return resp.json()["turns"]

    def test_a_skill_replays_as_typed_then_as_the_skill_row(self, replay_client, transcripts):
        turns = self._replay(
            replay_client,
            transcripts,
            [
                self._echo("diagnose", "SR03 orbit"),
                self._loaded("Base directory for this skill: /x/diagnose\n\n# Diagnose"),
                _entry("assistant", "Looking at SR03."),
            ],
        )
        assert [(t["role"], t["content"]) for t in turns] == [
            ("user", "/diagnose SR03 orbit"),
            ("assistant", "Looking at SR03."),
        ]
        call, result, *_ = turns[1]["events"]
        assert (call["type"], call["tool_name_raw"], call["input"]) == (
            "tool_use",
            "Skill",
            {"skill": "diagnose", "by": "operator", "args": "SR03 orbit"},
        )
        assert (result["type"], result["tool_use_id"], result["preview"]) == (
            "tool_result",
            call["tool_use_id"],
            "Launching skill: diagnose",
        )
        assert "Base directory" not in json.dumps(turns)

    def test_a_skills_row_leaves_out_the_files_attached_to_it(self, replay_client, transcripts):
        """The live row reads its arguments without the attached-files block; so does the replay."""
        block = (
            "\n\nAttached files (read them with the Read tool):\n- /tmp/orbit.png (image/png, 2 KB)"
        )
        turns = self._replay(
            replay_client,
            transcripts,
            [
                self._echo("diagnose", f"SR03 orbit{block}"),
                self._loaded("Base directory for this skill: /x/diagnose\n\n# Diagnose"),
                _entry("assistant", "Looking at SR03."),
            ],
        )
        # The operator's bubble keeps the block: the chat draws it as the file chips.
        assert turns[0]["content"] == f"/diagnose SR03 orbit{block}"
        call, *_ = turns[1]["events"]
        assert call["input"]["args"] == "SR03 orbit"

    def test_a_command_file_replays_as_typed_with_no_skill_row(self, replay_client, transcripts):
        turns = self._replay(
            replay_client,
            transcripts,
            [
                self._echo("check", "SR:DCCT"),
                self._loaded("Check the PV the operator names."),
                _entry("assistant", "Checking."),
            ],
        )
        assert [(t["role"], t["content"]) for t in turns] == [
            ("user", "/check SR:DCCT"),
            ("assistant", "Checking."),
        ]
        assert [e["type"] for e in turns[1]["events"]] == ["text", "result"]

    def test_a_builtins_echo_stays_bookkeeping(self, replay_client, transcripts):
        turns = self._replay(
            replay_client,
            transcripts,
            [_entry("user", "<command-name>/compact</command-name>"), _entry("user", "go on")],
        )
        assert [t["content"] for t in turns if t["role"] == "user"] == ["go on"]


#: The marker the CLI writes when the operator stops a turn.
INTERRUPTED = "[Request interrupted by user]"


def _stopped(entry: dict, stop_reason: str | None) -> dict:
    """*entry* with the stop reason CLI 2.1.300 records on each assistant entry."""
    entry["message"]["stop_reason"] = stop_reason
    return entry


def _report(tool_use_id: str) -> dict:
    """The user entry the CLI injects when a background agent reports."""
    entry = _entry("user", "")
    entry["message"]["content"] = (
        f"<task-notification>\n<tool-use-id>{tool_use_id}</tool-use-id>\n"
        "<status>completed</status>\n<summary>Agent finished</summary>\n</task-notification>"
    )
    return entry


def _launched(tool_use_id: str) -> list[dict]:
    """A subagent call the CLI sent to the background, and the result that says so."""
    return [
        _stopped(_agent_call(tool_use_id), "tool_use"),
        _result(tool_use_id, f"{_ASYNC_LAUNCH_MARK} successfully. agentId: a1"),
    ]


def _api_error(error: str, text: str) -> dict:
    """The assistant entry the CLI writes when the model API fails a request (CLI 2.1.228)."""
    entry = _stopped(_entry("assistant", text), "stop_sequence")
    entry["message"]["model"] = "<synthetic>"
    return {**entry, "error": error, "isApiErrorMessage": True}


class TestReplayedTurnEnds:
    """A replayed turn ends with a result only where the transcript shows that it ended.

    The transcript's last turn can still be running: a reload mid-turn reads it
    before the agent is done. That turn replays open, so the live stream that
    follows continues it rather than drawing a finished turn and a second one.
    """

    def _agent_turns(self, client, transcripts, entries: list[dict]) -> list[dict]:
        _, transcript_dir = transcripts
        _write(transcript_dir, SESSION_KEY, entries)
        resp = client.get(f"/api/session-chat?session_id={SESSION_KEY}&events=1")
        assert resp.status_code == 200
        return [t for t in resp.json()["turns"] if t["role"] == "assistant"]

    @pytest.mark.parametrize(
        "tail",
        [
            pytest.param([_stopped(_call("toolu_1"), "tool_use")], id="waiting-on-its-call"),
            pytest.param(
                [_stopped(_call("toolu_1"), "tool_use"), _result("toolu_1", "1")],
                id="reading-its-calls-result",
            ),
            pytest.param(
                [
                    _stopped(_entry("assistant", "Started the plot."), "end_turn"),
                    _report("toolu_bg"),
                ],
                id="answering-a-background-agents-report",
            ),
            # Its answer ended, but the live turn stays open until the agent reports.
            pytest.param(
                [
                    *_launched("toolu_bg"),
                    _stopped(_entry("assistant", "The plot is on its way."), "end_turn"),
                ],
                id="waiting-on-its-background-agent",
            ),
        ],
    )
    def test_a_turn_still_running_replays_open(self, replay_client, transcripts, tail):
        (turn,) = self._agent_turns(
            replay_client, transcripts, [_entry("user", "read the orbit"), *tail]
        )
        assert "result" not in [e["type"] for e in turn["events"]]

    def test_a_skill_the_operator_just_ran_replays_open(self, replay_client, transcripts):
        """The agent has the skill's instructions and has not answered yet."""
        (turn,) = self._agent_turns(
            replay_client,
            transcripts,
            [
                TestReplayedSlashCommands._echo("diagnose"),
                TestReplayedSlashCommands._loaded("Base directory for this skill: /x/diagnose"),
            ],
        )
        assert [e["type"] for e in turn["events"]] == ["tool_use", "tool_result"]

    @pytest.mark.parametrize(
        "tail",
        [
            pytest.param([_stopped(_entry("assistant", "It is flat."), "end_turn")], id="answered"),
            pytest.param(
                [_stopped(_call("toolu_1"), "tool_use"), _entry("user", INTERRUPTED)],
                id="stopped-during-its-call",
            ),
            pytest.param(
                [_stopped(_entry("assistant", "Reading"), None), _entry("user", INTERRUPTED)],
                id="stopped-while-it-wrote",
            ),
            # An entry that records no stop reason gives no sign the turn went on.
            pytest.param([_entry("assistant", "It is flat.")], id="no-stop-reason"),
            pytest.param(
                [
                    *_launched("toolu_bg"),
                    _stopped(_entry("assistant", "The plot is on its way."), "end_turn"),
                    _report("toolu_bg"),
                    _stopped(_entry("assistant", "Here it is."), "end_turn"),
                ],
                id="its-background-agent-reported",
            ),
        ],
    )
    def test_a_turn_that_ended_replays_with_its_result(self, replay_client, transcripts, tail):
        (turn,) = self._agent_turns(
            replay_client, transcripts, [_entry("user", "read the orbit"), *tail]
        )
        last = turn["events"][-1]
        assert (last["type"], last["is_error"]) == ("result", False)

    def test_a_turn_a_later_prompt_follows_has_ended(self, replay_client, transcripts):
        """However its last entry reads, the operator's next prompt came after it."""
        first, _second = self._agent_turns(
            replay_client,
            transcripts,
            [
                _entry("user", "read the orbit"),
                _stopped(_call("toolu_1"), "tool_use"),
                _entry("user", "never mind, read the tunes"),
                _stopped(_entry("assistant", "Reading."), "end_turn"),
            ],
        )
        assert first["events"][-1]["type"] == "result"

    def test_an_api_error_replays_as_the_live_error_and_a_failed_turn(
        self, replay_client, transcripts
    ):
        from osprey.agent_runner.events import ApiErrorEvent
        from osprey.interfaces.web_terminal.operator_session import _event_to_wire

        failure = "Failed to authenticate: OAuth session expired and could not be refreshed"
        (turn,) = self._agent_turns(
            replay_client,
            transcripts,
            [_entry("user", "read the orbit"), _api_error("authentication_failed", failure)],
        )

        error, text, result = turn["events"]
        assert error == _event_to_wire(ApiErrorEvent(error="authentication_failed"))
        assert text == {"type": "text", "content": failure}
        assert result["type"] == "result"
        assert result["is_error"] is True

    def test_a_turn_that_recovered_from_an_api_error_did_not_fail(self, replay_client, transcripts):
        """The outcome is the turn's last answer: one that came after the error succeeded."""
        (turn,) = self._agent_turns(
            replay_client,
            transcripts,
            [
                _entry("user", "read the orbit"),
                _api_error("rate_limit", "API Error: Rate limited"),
                _report("toolu_bg"),
                _stopped(_entry("assistant", "The plot is done."), "end_turn"),
            ],
        )
        assert turn["events"][-1]["is_error"] is False
