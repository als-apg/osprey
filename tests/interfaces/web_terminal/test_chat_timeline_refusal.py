"""C-FRAME's ``refusal``: a refused write's reason and phrase, live and replayed alike.

Every refusal a write path returns to the agent ends with the trailer
``[osprey-refusal: <reason>]``. The trailer sits at the end of the text, where
the preview's head cut would lose it, so it is read from the full result text by
one helper (``tool_preview.tool_result_refusal``) that both the live stream
(``operator_session._event_to_wire``) and the history replay
(``routes/session._chat_turns``) call. The texts below come from the real
producers: the writes-check hook's deny header, the twin's and the package's
``refusal_text``, the bluesky ``queue_start`` in-tool refusal and the chat's own
tier deny.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastmcp.exceptions import ToolError

import osprey.templates.claude_code.claude.hooks.osprey_target_state as twin
import osprey.templates.claude_code.claude.hooks.osprey_writes_check as writes_check
from osprey.agent_runner import ToolResultEvent
from osprey.interfaces.web_terminal.operator_session import _event_to_wire
from osprey.interfaces.web_terminal.routes.session import _chat_turns, _transcript_nodes
from osprey.interfaces.web_terminal.tool_preview import (
    PLAIN_TEXT_CHARS,
    tool_result_refusal,
)
from osprey_connectors import posture_store
from osprey_connectors.refusals import refusal_phrase, refusal_text

WRITE = "mcp__controls__channel_write"

#: The lead the CLI puts before a PreToolUse hook's deny reason.
HOOK_LEAD = f"PreToolUse:{WRITE} hook error: "


def _queue_start_refusal() -> str:
    """The text of the bluesky ``queue_start`` in-tool refusal, as the tool raises it."""
    from osprey.mcp_server.bluesky.tools.queue import _refuse_writes_disabled

    verdict = posture_store.WriteVerdict(
        allowed=False, reason="switch_off", target="standin", kind="rehearsal"
    )
    with pytest.raises(ToolError) as raised:
        _refuse_writes_disabled("queue_start", verdict, lane="bluesky_standin")
    return str(raised.value)


def _writes_check_deny(reason: str, kind: str | None) -> str:
    """The writes-check PreToolUse deny, as the CLI hands it back: lead, header, text."""
    return HOOK_LEAD + writes_check._DENY_HEADER + twin.refusal_text(reason, kind)


#: (id, result text, is_error, expected refusal).
CASES: list[tuple[str, str, bool, dict[str, str] | None]] = [
    (
        "pre-card-writes-check",
        _writes_check_deny("switch_off", "real"),
        True,
        {
            "reason": "switch_off",
            "phrase": writes_check._DENY_HEADER + refusal_phrase("switch_off", "real"),
        },
    ),
    (
        "hook-refusal",
        f"{HOOK_LEAD}[python3 .claude/hooks/osprey_writes_check.py]: "
        + twin.refusal_text("record_unavailable", "rehearsal"),
        True,
        {
            "reason": "record_unavailable",
            "phrase": refusal_phrase("record_unavailable", "rehearsal"),
        },
    ),
    (
        "app-render-deny",
        refusal_text("tier_never", "simulator"),
        True,
        {"reason": "tier_never", "phrase": refusal_phrase("tier_never", "simulator")},
    ),
    (
        "queue-start-in-tool",
        _queue_start_refusal(),
        True,
        {
            "reason": "switch_off",
            "phrase": refusal_phrase("switch_off", "rehearsal")
            + " queue_start was refused before reaching the bridge.",
        },
    ),
    (
        "long-deny-text",
        refusal_text("launch_pinned", "real", detail="x" * (PLAIN_TEXT_CHARS * 2)),
        True,
        {
            "reason": "launch_pinned",
            "phrase": refusal_phrase("launch_pinned", "real") + " " + "x" * (PLAIN_TEXT_CHARS * 2),
        },
    ),
    ("ordinary-error", "Channel SR:X does not exist.", True, None),
    (
        "quoted-by-a-successful-result",
        refusal_text("switch_off", "real"),
        False,
        None,
    ),
    ("reason-outside-the-closed-set", "Nope.\n[osprey-refusal: because]", True, None),
    ("trailer-not-on-its-own-line", "Nope [osprey-refusal: switch_off]", True, None),
]


def _write_transcript(path: Path, content: object, is_error: bool) -> None:
    """A transcript in which one prompt runs one write, whose result is *content*."""
    stamp = datetime(2026, 10, 24, 12, 0, 0, tzinfo=UTC).isoformat()
    entries = [
        {"type": "user", "timestamp": stamp, "message": {"role": "user", "content": "set it"}},
        {
            "type": "assistant",
            "timestamp": stamp,
            "message": {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": WRITE, "input": {}}],
            },
        },
        {
            "type": "user",
            "timestamp": stamp,
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": content,
                        "is_error": is_error,
                    }
                ],
            },
        },
    ]
    path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n")


def _live(content: object, is_error: bool) -> dict[str, Any]:
    frame = _event_to_wire(
        ToolResultEvent(
            tool_use_id="t1", content=content, is_error=is_error, parent_tool_use_id=None
        )
    )
    assert frame is not None
    return frame


def _replayed(tmp_path: Path, content: object, is_error: bool) -> dict[str, Any]:
    transcript = tmp_path / "session.jsonl"
    _write_transcript(transcript, content, is_error)
    (event,) = [
        event
        for turn in _chat_turns(_transcript_nodes(transcript))
        for event in turn.get("events", [])
        if event["type"] == "tool_result"
    ]
    return event


@pytest.mark.parametrize(
    ("text", "is_error", "expected"),
    [(text, is_error, expected) for _, text, is_error, expected in CASES],
    ids=[case_id for case_id, *_ in CASES],
)
@pytest.mark.parametrize("as_blocks", [False, True], ids=["string", "text-blocks"])
def test_live_and_replayed_carry_the_same_refusal(tmp_path, text, is_error, expected, as_blocks):
    content: object = [{"type": "text", "text": text}] if as_blocks else text

    live = _live(content, is_error)
    replayed = _replayed(tmp_path, content, is_error)

    assert live["refusal"] == expected
    assert replayed["refusal"] == expected
    assert replayed["preview"] == live["preview"]
    assert replayed["truncated"] == live["truncated"]
    if expected is not None:
        assert "[osprey-refusal:" not in live["preview"]


def test_a_long_deny_is_read_whole_though_its_preview_is_cut():
    """The trailer lies past the preview cut; the refusal is read before the cut."""
    text = refusal_text("switch_off", "real", detail="y" * (PLAIN_TEXT_CHARS * 3))

    live = _live(text, True)

    assert live["truncated"] is True
    assert live["refusal"]["reason"] == "switch_off"
    assert live["preview"].startswith(refusal_phrase("switch_off", "real"))


def test_the_in_tool_refusal_keeps_its_envelope_with_the_trailer_stripped():
    """The queue refusal arrives as a JSON envelope; stripped, it is still that envelope."""
    refusal, stripped = tool_result_refusal(_queue_start_refusal(), True)

    assert refusal is not None and refusal["reason"] == "switch_off"
    envelope = json.loads(stripped)
    assert envelope["error_type"] == "writes_disabled"
    assert "[osprey-refusal:" not in envelope["error_message"]
    assert envelope["details"]["reason"] == "switch_off"


def test_an_in_tool_refusal_behind_a_lead_is_read():
    text = "Error executing tool queue_start: " + _queue_start_refusal()

    refusal, stripped = tool_result_refusal(text, True)

    assert refusal is not None and refusal["reason"] == "switch_off"
    assert stripped.startswith("Error executing tool queue_start: {")


def test_a_replayed_refusal_keeps_its_decision_pill_off(tmp_path):
    """A safety hook refused the call; it never reached the operator, so no pill."""
    replayed = _replayed(tmp_path, _writes_check_deny("switch_off", "real"), True)

    assert replayed["decision"] is None
    assert replayed["refusal"]["reason"] == "switch_off"


def test_every_reason_round_trips():
    """Every reason the closed set holds is read back from its own text."""
    from osprey_connectors.refusals import REASONS

    for reason in REASONS:
        refusal, _ = tool_result_refusal(refusal_text(reason, "real"), True)
        assert refusal == {"reason": reason, "phrase": refusal_phrase(reason, "real")}
