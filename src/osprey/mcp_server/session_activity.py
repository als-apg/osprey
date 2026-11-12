"""A session's record entries joined into one activity reading.

The ``timeline`` read answers a session as nodes (``TimelineNode.to_json``):
a call carries its decision and its result, a subagent's stream is nested
under the call that started it, and audit and usage entries stand loose in the
main stream. :func:`session_activity` turns that into what the Activity page
shows:

* every tool call of every stream, built-in and MCP alike, with its verdict
  (``allowed``, ``approved``, ``asked``, ``denied`` or ``none``), its approval,
  the audit entries filed for it (joined by ``tool_use_id`` at any depth), its
  result and whether it failed;
* the words between the calls — the operator's prompts, a subagent's brief,
  thinking and answers — with what the harness wrote into the stream (a skill's
  body, a subagent's report, a reminder) told apart as one-line notes;
* one ``usage`` node per model request.

``debug`` entries are left out of the nodes and of the counts. An entry the
record could not give whole (``incomplete``, ``body`` ``None``) is kept and
marked, never raised on.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime
from typing import Any

from osprey.agent_record.reader import ACTIVITY_KINDS, INTERRUPT_MARKER
from osprey.mcp_server.session import _body, _mapping, _ts
from osprey.mcp_server.workspace.transcript_reader import _extract_text, _split_mcp_tool_name

#: A result's, a prompt's or a note's text is cut to this many characters; the
#: whole result is one more read away (the entry's key travels with the cut).
RESULT_CAP = 2400

#: How many times a result is decoded: a tool answers JSON, the MCP layer wraps
#: it in ``{"result": ...}`` and the transcript stores that as a string.
_UNWRAP_DEPTH = 3

_AUDIT_TOOL_CALL = "tool_call"
_AUDIT_APPROVAL = "hook_approval"
_MAIN = "main"

_COMMAND_NAME = re.compile(r"<command-name>(.*?)</command-name>", re.DOTALL)
_COMMAND_ARGS = re.compile(r"<command-args>(.*?)</command-args>", re.DOTALL)
_COMMAND_OUTPUT = re.compile(
    r"<local-command-std(?:out|err)>(.*?)</local-command-std(?:out|err)>", re.DOTALL
)
_AGENT_MESSAGE = re.compile(r'<agent-message from="[^"]+"[^>]*>(.*?)</agent-message>', re.DOTALL)
_SKILL_PREFIX = "Base directory for this skill"


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def unwrap_result(content: Any) -> Any:
    """A tool result's content, decoded to what the tool answered.

    Content blocks are joined to text; text that is JSON is decoded, up to
    three levels deep, and an object holding only ``result`` gives that value.
    Anything else — plain text, a ``<persisted-output>`` stub — is returned as
    the text it is.
    """
    if isinstance(content, list):
        content = _extract_text(content)
    if content is None:
        return ""
    for _ in range(_UNWRAP_DEPTH):
        if not isinstance(content, str):
            break
        try:
            decoded = json.loads(content)
        except ValueError:
            break
        wrapped = isinstance(decoded, dict) and set(decoded) == {"result"}
        content = decoded["result"] if wrapped else decoded
    return content


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=1, ensure_ascii=False, default=str)


def _view(value: Any, key: str) -> dict[str, Any]:
    text = _text(value)
    return {"text": text[:RESULT_CAP], "size": len(text), "cut": len(text) > RESULT_CAP, "key": key}


def result_text(content: Any) -> str:
    """A result's whole text: what :func:`result_view` cuts at :data:`RESULT_CAP`."""
    return _text(unwrap_result(content))


def result_view(content: Any, key: str = "") -> dict[str, Any]:
    """A result as the page shows it: ``{text, size, cut, key}``.

    ``text`` is the unwrapped content (an object as indented JSON) cut at
    :data:`RESULT_CAP`, ``size`` its whole length, ``key`` the record entry to
    read for the rest.
    """
    return _view(unwrap_result(content), key)


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------


def _when(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts) if ts else None
    except ValueError:
        return None


def _ms(start: str, end: str) -> int | None:
    """Milliseconds from *start* to *end*, never negative; ``None`` without both.

    The two may come from different clocks (a hook's audit entry, the CLI's
    telemetry), so a small negative difference is a zero.
    """
    a, b = _when(start), _when(end)
    if a is None or b is None:
        return None
    try:
        return max(0, round((b - a).total_seconds() * 1000))
    except TypeError:  # one stamp carries an offset and the other does not
        return None


# ---------------------------------------------------------------------------
# The walk
# ---------------------------------------------------------------------------


def _children(node: Mapping[str, Any]) -> list[Any]:
    timeline = _mapping(node.get("subagent")).get("timeline")
    return timeline if isinstance(timeline, list) else []


def _entries(nodes: Sequence[Any]) -> Iterator[Mapping[str, Any]]:
    """Every record entry under *nodes*: each node's own, its decision, its result."""
    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        if not node.get("orphan"):
            yield node
        for joined in (node.get("decision"), node.get("result")):
            if isinstance(joined, Mapping):
                yield joined
        yield from _entries(_children(node))


def session_activity(nodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The activity reading of a session's timeline *nodes*.

    Returns ``{nodes, entries, kinds, start, end, target, posture}``: the
    nodes the page shows, how many record entries they were read from and of
    which kinds, the transcript's first and last time, and the control target
    and posture of the first call that names one.
    """
    audits: dict[str, list[Mapping[str, Any]]] = {}
    kinds: Counter[str] = Counter()
    span: list[tuple[datetime, str]] = []
    for entry in _entries(nodes):
        kind = entry.get("kind")
        if not isinstance(kind, str) or kind == "debug":
            continue
        kinds[kind] += 1
        if kind == "audit" and entry.get("tool_use_id"):
            audits.setdefault(str(entry["tool_use_id"]), []).append(entry)
        moment = _when(_ts(entry))
        if kind in ACTIVITY_KINDS and moment is not None:
            span.append((moment, _ts(entry)))
    for filed in audits.values():
        filed.sort(key=_ts)
    span.sort(key=lambda pair: pair[0])
    named: dict[str, str | None] = {"target": None, "posture": None}
    return {
        "nodes": _convert(nodes, audits, _MAIN, named),
        "entries": sum(kinds.values()),
        "kinds": dict(kinds),
        "start": span[0][1] if span else None,
        "end": span[-1][1] if span else None,
        **named,
    }


def _convert(
    nodes: Sequence[Any],
    audits: Mapping[str, list[Mapping[str, Any]]],
    agent: str,
    named: dict[str, str | None],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        kind = node.get("kind")
        ts = _ts(node)
        body = _body(node)
        if kind == "tool_call" or node.get("orphan"):
            out.append(_tool(node, audits, agent, named))
        elif node.get("incomplete") and kind in ("prompt", "thinking", "answer"):
            out.append(_note(ts, "alert", f"Incomplete {kind} entry", 0))
        elif kind == "prompt":
            words = _prompt(ts, body, brief=agent != _MAIN)
            if words is not None:
                out.append(words)
        elif kind in ("thinking", "answer"):
            text = body.get("text")
            if isinstance(text, str) and text.strip():
                out.append({"t": kind, "ts": ts, "text": text})
        elif kind == "usage" and body:
            out.append(
                {
                    "t": "usage",
                    "ts": ts,
                    "cost": body.get("cost_usd") or 0,
                    "ms": body.get("duration_ms") or 0,
                }
            )
        elif kind == "error":
            out.append(_request_error(ts, body))
    return out


# ---------------------------------------------------------------------------
# Words
# ---------------------------------------------------------------------------


def _note(ts: str, icon: str, text: str, size: int, body: str | None = None) -> dict[str, Any]:
    note: dict[str, Any] = {"t": "note", "ts": ts, "icon": icon, "text": text, "size": size}
    if body is not None:
        note["body"] = body[:RESULT_CAP]
    return note


def _prompt(ts: str, body: Mapping[str, Any], *, brief: bool) -> dict[str, Any] | None:
    """A prompt entry as what it is: the operator's words, or what the harness wrote."""
    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    stripped = text.strip()
    command = _COMMAND_NAME.search(text)
    if command:
        args = _COMMAND_ARGS.search(text)
        return {
            "t": "prompt",
            "ts": ts,
            "command": command.group(1).strip(),
            "text": args.group(1).strip() if args else "",
        }
    if stripped.startswith(_SKILL_PREFIX):
        name = stripped.splitlines()[0].rstrip("/").rsplit("/", 1)[-1]
        return _note(ts, "book", f"Skill loaded: {name}", len(text))
    report = _AGENT_MESSAGE.search(text)
    if report:
        said = report.group(1).strip()
        return _note(ts, "reply", "Subagent reported back", len(said), said)
    if stripped.startswith(INTERRUPT_MARKER):
        return _note(ts, "stop", "Interrupted by the operator", 0)
    output = _COMMAND_OUTPUT.search(text)
    if output:
        said = output.group(1).strip()
        return _note(ts, "info", "Command output", len(said), said)
    if stripped.startswith("<task-notification>"):
        return _note(ts, "info", "Background task finished", len(text))
    if stripped.startswith("<system-reminder>") or body.get("is_meta"):
        return _note(ts, "info", "Harness instruction", len(text))
    return {"t": "brief" if brief else "prompt", "ts": ts, "text": text[:RESULT_CAP]}


def _request_error(ts: str, body: Mapping[str, Any]) -> dict[str, Any]:
    """A failed model request (the CLI's ``api_error`` event) as a note."""
    text = "Model request failed"
    if body.get("status_code"):
        text += f" ({body['status_code']})"
    if body.get("attempt"):
        text += f", attempt {body['attempt']}"
    error = body.get("error")
    return _note(ts, "alert", text, 0, error if isinstance(error, str) and error else None)


# ---------------------------------------------------------------------------
# Calls
# ---------------------------------------------------------------------------


def _verdict(
    decision: Mapping[str, Any],
    approval: Mapping[str, Any],
    filed: Sequence[Mapping[str, Any]],
) -> str:
    """How a call was let through — or not.

    Without an approval prompt the CLI's decision is the verdict. A prompt
    is ``approved`` or ``denied`` by its answer (the ``tool_call`` entry's
    approval stamp, the approval hook's own answer entry, the CLI's decision)
    and ``asked`` while none of them is in the record.
    """
    accepted = _body(decision).get("decision") == "accept" if decision else None
    answers = {
        _body(a).get("decision") for a in filed if _body(a).get("surface") == _AUDIT_APPROVAL
    }
    outcome = approval.get("outcome")
    if "ask" not in answers and not approval:
        if accepted is None:
            return "none"
        return "allowed" if accepted else "denied"
    if outcome == "approved" or (outcome is None and ("approved" in answers or accepted)):
        return "approved"
    if outcome is not None or "denied" in answers or accepted is False:
        return "denied"
    return "asked"


def _failure(result: Mapping[str, Any], value: Any) -> tuple[bool, str | None]:
    """Whether a call failed, and what it said.

    A failure is the transcript's ``is_error`` or a result object that says so
    (``error: true``, ``status`` ``error`` or ``failed``).
    """
    said = isinstance(value, Mapping) and (
        value.get("error") is True or value.get("status") in ("error", "failed")
    )
    if not (_body(result).get("is_error") or said):
        return False, None
    message = value.get("error_message") if isinstance(value, Mapping) else None
    return True, message if isinstance(message, str) and message else None


def _tool(
    node: Mapping[str, Any],
    audits: Mapping[str, list[Mapping[str, Any]]],
    agent: str,
    named: dict[str, str | None],
) -> dict[str, Any]:
    orphan = bool(node.get("orphan"))
    body = _body(node)
    call_id = str(node.get("tool_use_id") or body.get("id") or "")
    name = "Agent" if orphan else str(body.get("name") or "")
    split = _split_mcp_tool_name(name)
    tool, server = (split[0], split[1]) if split else (name, "")
    args = _mapping(body.get("input"))
    decision = _mapping(node.get("decision"))
    result = _mapping(node.get("result"))
    filed = audits.get(call_id, []) if call_id else []
    stamped = next((_body(a) for a in filed if _body(a).get("surface") == _AUDIT_TOOL_CALL), {})
    gate = next(
        (
            _body(a)
            for a in filed
            if _body(a).get("surface") not in (_AUDIT_TOOL_CALL, _AUDIT_APPROVAL)
        ),
        {},
    )
    asked = next((a for a in filed if _body(a).get("decision") == "ask"), None)
    approval = _mapping(stamped.get("approval"))

    children = _children(node)
    sub_id = str(_mapping(node.get("subagent")).get("agent_id") or node.get("agent_id") or "")
    is_sub = bool(children) or orphan
    kind = str(args.get("subagent_type") or "subagent")
    kids = _convert(children, audits, kind, named) if is_sub else []

    value: Any = None
    view = None
    if result and result.get("body") is not None and not is_sub:
        value = unwrap_result(_body(result).get("content"))
        view = _view(value, str(result.get("key") or ""))
    error, errmsg = _failure(result, value)

    ts = _ts(node) or (kids[0]["ts"] if orphan and kids else "")
    target = stamped.get("target")
    posture = stamped.get("posture") or gate.get("posture")
    for field, found in (("target", target), ("posture", posture)):
        if named[field] is None and isinstance(found, str) and found:
            named[field] = found

    out: dict[str, Any] = {
        "t": "tool",
        "ts": ts,
        "id": call_id,
        "key": str(node.get("key") or ""),
        "name": name,
        "server": server,
        "tool": tool,
        "agent": agent,
        "input": dict(args),
        "verdict": _verdict(decision, approval, filed),
        "source": _body(decision).get("source") if decision else None,
        "duration_ms": stamped.get("duration_ms"),
        "elapsed_ms": _ms(_ts(node), _ts(result)) if result else None,
        "target": target,
        "posture": posture,
        "approval": None,
        "result": view,
        "error": error,
        "errmsg": errmsg,
        "trail": [
            {
                "ts": _ts(a),
                "surface": _body(a).get("surface"),
                "decision": _body(a).get("decision"),
                "reason": _body(a).get("reason"),
                "detail": _body(a).get("detail"),
            }
            for a in filed
            if _body(a).get("surface") != _AUDIT_TOOL_CALL
        ],
        "incomplete": any(
            bool(part.get("incomplete")) for part in (node, decision, result) if part
        ),
        "running": not result and not is_sub,
        "orphan": orphan,
    }
    if asked is not None or approval:
        asked_ts = _ts(asked) if asked is not None else None
        decided_ts = _ts(decision) or None
        out["approval"] = {
            "asked_ts": asked_ts,
            "decided_ts": decided_ts,
            "wait_ms": _ms(asked_ts or "", decided_ts or ""),
            **approval,
        }
    if is_sub:
        out["sub"] = {
            "id": sub_id,
            "type": kind,
            "description": str(args.get("description") or ""),
            "children": kids,
            "ms": (_ms(kids[0]["ts"], kids[-1]["ts"]) or 0) if kids else 0,
        }
    return out
