"""Write a stated agent conversation as the transcript the agent CLI would have kept.

A scenario's ``sessions`` block (``osprey_connectors.simulation.sessions``) states a
past conversation: prompts, tool calls and their results, answers. A deployment that
lays such a conversation down needs it as the CLI's own JSONL transcript, so that
every surface reading transcripts (the agent record, the chat replay, ``--resume``)
reads it as one it recorded. :func:`author_transcript` turns one part of a session
into those lines.

The output is a pure function of its arguments: line uuids and tool-use ids derive
from the transcript id and the line's index, and timestamps from the turns'
resolved instants, so authoring the same part twice writes the same bytes.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from osprey.agent_runner.tool_names import _MCP_NAME_UNSAFE

if TYPE_CHECKING:
    from osprey_connectors.simulation.sessions import ResolvedPart, ToolStep

__all__ = ["APPROVED_CALL_NOTE", "author_transcript"]

#: The note the approval hook gives the model when a person approved a call and the call
#: then ran, or ran and failed. A copy of ``APPROVED_CALL_NOTE`` in the hook at
#: ``templates/claude_code/claude/hooks/osprey_approval.py``; a test pins the two together.
#: The session-chat replay (``interfaces/web_terminal/routes/session.py``) reads it too.
APPROVED_CALL_NOTE = "The operator approved this call at the approval prompt before it ran."

#: Namespace of the ids this module derives; fixed so ids stay stable across releases.
_AUTHOR_NS = uuid.UUID("5b0e2c1e-8f43-4f8e-9d0a-6c3d2a71f6b4")

#: The longest gap between two lines of one turn; a turn whose next prompt follows
#: sooner spreads its lines evenly before that prompt instead.
_LINE_STEP = timedelta(seconds=2)


def author_transcript(
    part: ResolvedPart, *, session_id: str, cwd: str, cli_version: str
) -> list[dict[str, Any]]:
    """The CLI transcript lines of one session part, in file order.

    Each turn becomes a user line holding the prompt, then per step: for a tool
    call, an assistant line with its ``tool_use`` block (named
    ``mcp__<server>__<tool>`` as the CLI names it), a user line with its
    ``tool_result``, and for an ``approved`` call the ``PostToolUse`` attachment
    line carrying :data:`APPROVED_CALL_NOTE` that the approval hook leaves; for an
    answer, an assistant line with its text. A turn's last answer stops with
    ``end_turn``. Every line's ``parentUuid`` is the uuid of the line before it.

    Args:
        part: One resolved part of a scenario session; its turns carry instants.
        session_id: The transcript id the lines belong to (the file's stem).
        cwd: The project directory the session ran in.
        cli_version: The pinned agent CLI build the lines claim to come from.

    Returns:
        JSON-ready line objects; write each as one line of ``<session_id>.jsonl``.
    """
    from osprey_connectors.simulation.sessions import AnswerStep

    lines: list[dict[str, Any]] = []

    def add(line_type: str, at: datetime, body: dict[str, Any]) -> None:
        index = len(lines)
        line_uuid = str(_derived(session_id, f"line/{index}"))
        lines.append(
            {
                "parentUuid": lines[-1]["uuid"] if lines else None,
                "isSidechain": False,
                "userType": "external",
                "cwd": cwd,
                "sessionId": session_id,
                "version": cli_version,
                "type": line_type,
                **body,
                "uuid": line_uuid,
                "timestamp": _stamp(at),
            }
        )

    turns = part.turns
    calls = 0
    for t, resolved in enumerate(turns):
        turn = resolved.turn
        steps = turn.steps
        n_lines = 1 + sum(
            1 if isinstance(s, AnswerStep) else (3 if s.approved else 2) for s in steps
        )
        gap = (turns[t + 1].at - resolved.at) if t + 1 < len(turns) else None
        step = _LINE_STEP if gap is None else min(_LINE_STEP, gap / (n_lines + 1))
        clock = iter(resolved.at + step * k for k in range(n_lines))

        add("user", next(clock), {"message": {"role": "user", "content": turn.prompt}})
        last_answer = max(
            (i for i, s in enumerate(steps) if isinstance(s, AnswerStep)), default=None
        )
        for i, s in enumerate(steps):
            if isinstance(s, AnswerStep):
                stop = "end_turn" if i == last_answer else None
                add(
                    "assistant",
                    next(clock),
                    {"message": _assistant(session_id, len(lines), [_text(s.answer)], stop)},
                )
                continue
            tool_use_id = "toolu_" + _derived(session_id, f"call/{calls}").hex
            calls += 1
            name = _cli_tool_name(s)
            block = {"type": "tool_use", "id": tool_use_id, "name": name, "input": s.input}
            add(
                "assistant",
                next(clock),
                {"message": _assistant(session_id, len(lines), [block], "tool_use")},
            )
            result = {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": [_text(_result_text(s.result))],
                "is_error": False,
            }
            add("user", next(clock), {"message": {"role": "user", "content": [result]}})
            if s.approved:
                attachment = {
                    "type": "hook_additional_context",
                    "content": [APPROVED_CALL_NOTE],
                    "hookName": f"PostToolUse:{name}",
                    "toolUseID": tool_use_id,
                    "hookEvent": "PostToolUse",
                }
                add("attachment", next(clock), {"attachment": attachment})
    return lines


def _derived(session_id: str, name: str) -> uuid.UUID:
    return uuid.uuid5(_AUTHOR_NS, f"{session_id}/{name}")


def _cli_tool_name(step: ToolStep) -> str:
    """The name the agent CLI gives an MCP server's tool (``mcp__<server>__<tool>``)."""
    return f"mcp__{_MCP_NAME_UNSAFE.sub('_', step.server)}__{step.name}"


def _assistant(
    session_id: str, index: int, content: list[dict[str, Any]], stop: str | None
) -> dict[str, Any]:
    return {
        "id": "msg_" + _derived(session_id, f"message/{index}").hex,
        "type": "message",
        "role": "assistant",
        "content": content,
        "stop_reason": stop,
        "stop_sequence": None,
    }


def _text(text: str) -> dict[str, str]:
    return {"type": "text", "text": text}


def _result_text(result: Any) -> str:
    """A tool's stated result as the text block an MCP tool returns."""
    return result if isinstance(result, str) else json.dumps(result)


def _stamp(at: datetime) -> str:
    """An instant as the CLI writes it: UTC, milliseconds, ``Z``."""
    utc = at.astimezone(UTC)
    return utc.strftime("%Y-%m-%dT%H:%M:%S.") + f"{utc.microsecond // 1000:03d}Z"
