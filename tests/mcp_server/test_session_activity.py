"""A session's record entries become one activity reading.

Pins :mod:`osprey.mcp_server.session_activity`: every tool call of every
stream is kept (built-in and MCP, main stream and subagents), its decision,
audit entries, approval and result are joined onto it, the prompts the harness
injected are told apart from the operator's, and an entry the record could not
give whole never raises.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

from osprey.mcp_server.session_activity import (
    RESULT_CAP,
    result_text,
    result_view,
    session_activity,
    unwrap_result,
)

_T0 = datetime(2026, 11, 12, 16, 42, 0, tzinfo=UTC)


def _at(seconds: float) -> str:
    return (_T0 + timedelta(seconds=seconds)).isoformat()


def _item(kind: str, ts: float, body: Any, **ids: Any) -> dict[str, Any]:
    item: dict[str, Any] = {
        "key": f"k:{kind}:{ts}",
        "kind": kind,
        "ts": _at(ts),
        "body": body,
        "agent_id": "",
        "tool_use_id": "",
        "incomplete": False,
    }
    item.update(ids)
    return item


def _decision(ts: float, tid: str, decision: str = "accept", source: str = "config"):
    return _item("decision", ts, {"decision": decision, "source": source}, tool_use_id=tid)


def _result(ts: float, tid: str, content: Any, **body: Any):
    return _item(
        "tool_result", ts, {"tool_use_id": tid, "content": content, **body}, tool_use_id=tid
    )


def _call(
    tid: str,
    name: str,
    ts: float,
    *,
    args: dict[str, Any] | None = None,
    decision: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    subagent: dict[str, Any] | None = None,
    **ids: Any,
) -> dict[str, Any]:
    node = _item(
        "tool_call", ts, {"id": tid, "name": name, "input": args or {}}, tool_use_id=tid, **ids
    )
    node.update({"decision": decision, "result": result, "subagent": subagent})
    return node


def _audit(ts: float, tid: str, **body: Any) -> dict[str, Any]:
    return _item("audit", ts, body, tool_use_id=tid)


def _tools(reading: dict[str, Any]) -> list[dict[str, Any]]:
    return [n for n in reading["nodes"] if n["t"] == "tool"]


# --- calls ------------------------------------------------------------------------


def test_builtin_call_in_main_stream_is_kept():
    call = _call(
        "t1",
        "Read",
        1,
        args={"file_path": "/app/notes.md"},
        decision=_decision(1.1, "t1"),
        result=_result(1.5, "t1", "notes"),
    )
    (tool,) = _tools(session_activity([call]))
    assert (tool["name"], tool["server"], tool["tool"]) == ("Read", "", "Read")
    assert tool["agent"] == "main"
    assert tool["id"] == "t1" and tool["key"] == call["key"]
    assert tool["input"] == {"file_path": "/app/notes.md"}
    assert tool["ts"] == _at(1)
    assert tool["elapsed_ms"] == 500
    assert tool["running"] is False and tool["error"] is False


def test_mcp_name_splits_server_and_tool():
    calls = [
        _call("t1", "mcp__controls__channel_read", 1),
        _call("t2", "mcp__channel-finder__find__all", 2),
    ]
    first, second = _tools(session_activity(calls))
    assert (first["server"], first["tool"]) == ("controls", "channel_read")
    assert (second["server"], second["tool"]) == ("channel-finder", "find__all")


def test_verdict_allowed_from_config_decision():
    (tool,) = _tools(session_activity([_call("t1", "Read", 1, decision=_decision(1.1, "t1"))]))
    assert tool["verdict"] == "allowed"
    assert tool["source"] == "config"
    assert tool["approval"] is None


def test_verdict_approved_carries_wait_and_approver():
    approval = {
        "outcome": "approved",
        "approver": "human",
        "permission_mode": "auto",
        "position": "asked",
    }
    nodes = [
        _call(
            "t1",
            "mcp__bluesky__queue_start",
            10,
            decision=_decision(27.4, "t1", source="user_temporary"),
            result=_result(29, "t1", "ok"),
        ),
        _audit(10, "t1", surface="hook_approval", decision="ask", reason="approval_required"),
        _audit(27.5, "t1", surface="hook_approval", decision="approved", reason="executed"),
        _audit(28.9, "t1", surface="tool_call", duration_ms=1400, approval=approval, target="va"),
    ]
    (tool,) = _tools(session_activity(nodes))
    assert tool["verdict"] == "approved"
    assert tool["approval"] == {
        "asked_ts": _at(10),
        "decided_ts": _at(27.4),
        "wait_ms": 17400,
        **approval,
    }
    assert [(t["surface"], t["decision"]) for t in tool["trail"]] == [
        ("hook_approval", "ask"),
        ("hook_approval", "approved"),
    ]
    assert tool["duration_ms"] == 1400


def test_verdict_asked_while_the_prompt_is_open():
    nodes = [
        _call("t1", "mcp__controls__channel_write", 10),
        _audit(10, "t1", surface="hook_approval", decision="ask", reason="approval_required"),
    ]
    (tool,) = _tools(session_activity(nodes))
    assert tool["verdict"] == "asked"
    assert tool["approval"]["asked_ts"] == _at(10)
    assert tool["approval"]["decided_ts"] is None and tool["approval"]["wait_ms"] is None


def test_verdict_denied_when_the_operator_rejects():
    nodes = [
        _call(
            "t1",
            "mcp__controls__channel_write",
            10,
            decision=_decision(14, "t1", decision="reject", source="user_reject"),
        ),
        _audit(10, "t1", surface="hook_approval", decision="ask", reason="approval_required"),
        _audit(14.1, "t1", surface="hook_approval", decision="denied", reason="not_executed"),
    ]
    (tool,) = _tools(session_activity(nodes))
    assert tool["verdict"] == "denied"
    assert tool["approval"]["wait_ms"] == 4000


def test_verdict_none_without_decision():
    (tool,) = _tools(session_activity([_call("t1", "Read", 1, result=_result(2, "t1", "x"))]))
    assert tool["verdict"] == "none"
    assert tool["source"] is None


def test_audit_joins_subagent_call_by_tool_use_id():
    child = _call("t2", "mcp__controls__channel_read", 5, agent_id="a1")
    agent = _call(
        "t1",
        "Agent",
        4,
        args={"subagent_type": "plan-runner", "description": "Run the scan"},
        subagent={"agent_id": "a1", "timeline": [child]},
    )
    nodes = [
        agent,
        _audit(5.1, "t2", surface="controls", decision="allowed", reason="tool_call", posture="rw"),
        _audit(5.2, "t2", surface="tool_call", duration_ms=12, target="va", posture="rw"),
    ]
    reading = session_activity(nodes)
    (inner,) = reading["nodes"][0]["sub"]["children"]
    assert inner["duration_ms"] == 12
    assert inner["target"] == "va" and inner["posture"] == "rw"
    assert [t["surface"] for t in inner["trail"]] == ["controls"]
    assert (reading["target"], reading["posture"]) == ("va", "rw")


def test_posture_falls_back_to_the_gate_audit():
    nodes = [
        _call("t1", "mcp__python__execute", 1),
        _audit(1.1, "t1", surface="python", decision="allowed", reason="tool_call", posture="ro"),
    ]
    (tool,) = _tools(session_activity(nodes))
    assert tool["posture"] == "ro"
    assert tool["duration_ms"] is None and tool["target"] is None


def test_tool_error_audit_sets_error_not_denied():
    content = json.dumps({"error": True, "error_message": "boom"})
    nodes = [
        _call(
            "t1",
            "mcp__bluesky__queue_start",
            1,
            decision=_decision(1.1, "t1"),
            result=_result(2, "t1", content),
        ),
        _audit(1.9, "t1", surface="bluesky", decision="refused", reason="tool_error"),
    ]
    (tool,) = _tools(session_activity(nodes))
    assert tool["verdict"] == "allowed"
    assert tool["error"] is True
    assert tool["errmsg"] == "boom"


def test_status_failed_and_is_error_read_as_errors():
    nodes = [
        _call("t1", "mcp__x__y", 1, result=_result(2, "t1", json.dumps({"status": "failed"}))),
        _call("t2", "Bash", 3, result=_result(4, "t2", "exit 1", is_error=True)),
        _call("t3", "mcp__x__y", 5, result=_result(6, "t3", json.dumps({"status": "ok"}))),
    ]
    first, second, third = _tools(session_activity(nodes))
    assert (first["error"], second["error"], third["error"]) == (True, True, False)
    assert second["errmsg"] is None


def test_running_call_has_no_result():
    (tool,) = _tools(session_activity([_call("t1", "Bash", 1, decision=_decision(1.1, "t1"))]))
    assert tool["running"] is True
    assert tool["result"] is None and tool["elapsed_ms"] is None
    assert tool["error"] is False


# --- results ----------------------------------------------------------------------


def test_result_unwraps_nested_json_and_cuts():
    content = json.dumps({"result": json.dumps({"result": "x" * 3000})})
    result = {**_result(2, "t1", content), "key": "k:result"}
    (tool,) = _tools(session_activity([_call("t1", "mcp__x__y", 1, result=result)]))
    assert tool["result"] == {
        "text": "x" * RESULT_CAP,
        "size": 3000,
        "cut": True,
        "key": "k:result",
    }


def test_result_text_is_the_whole_text_the_view_cuts():
    content = json.dumps({"result": json.dumps({"result": "x" * 3000})})
    assert result_text(content) == "x" * 3000
    assert result_view(content)["text"] == result_text(content)[:RESULT_CAP]
    assert json.loads(result_text(json.dumps({"result": {"n": 3}}))) == {"n": 3}
    assert result_text(None) == ""


def test_plain_text_and_persisted_output_results():
    stub = "<persisted-output>\nOutput too large (48.2KB). Full output saved to: /tmp/x.txt"
    assert result_view("hello") == {"text": "hello", "size": 5, "cut": False, "key": ""}
    assert result_view(stub)["text"] == stub
    assert unwrap_result("{not json") == "{not json"
    assert unwrap_result(None) == ""


def test_content_block_list_result():
    assert unwrap_result([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "a\nb"
    blocks = [{"type": "text", "text": json.dumps({"result": {"ok": True, "n": 3}})}]
    assert unwrap_result(blocks) == {"ok": True, "n": 3}
    view = result_view(blocks, "k1")
    assert json.loads(view["text"]) == {"ok": True, "n": 3}
    assert view["key"] == "k1" and view["cut"] is False
    assert unwrap_result([{"type": "image", "source": {}}, "tail"]) == "tail"


def test_a_result_object_that_only_looks_wrapped_is_kept_whole():
    assert unwrap_result(json.dumps({"result": 1, "status": "ok"})) == {"result": 1, "status": "ok"}
    assert unwrap_result({"result": "inner"}) == {"result": "inner"}


# --- words ------------------------------------------------------------------------


def test_slash_command_prompt_is_a_prompt():
    text = (
        "<command-message>operating-bluesky-plans</command-message>\n"
        "<command-name>/operating-bluesky-plans</command-name>\n"
        "<command-args>scan HCM1 over BPM3</command-args>"
    )
    (node,) = session_activity([_item("prompt", 1, {"text": text})])["nodes"]
    assert node == {
        "t": "prompt",
        "ts": _at(1),
        "command": "/operating-bluesky-plans",
        "text": "scan HCM1 over BPM3",
    }


def test_operator_prompt_is_cut_at_the_cap():
    (node,) = session_activity([_item("prompt", 1, {"text": "p" * 3000})])["nodes"]
    assert node["t"] == "prompt" and "command" not in node
    assert len(node["text"]) == RESULT_CAP


def test_skill_body_and_reminder_become_notes():
    skill = (
        "Base directory for this skill: /app/build/.claude/skills/operating-bluesky-plans\n\n# x"
    )
    nodes = [
        _item("prompt", 1, {"text": skill, "is_meta": True}),
        _item("prompt", 2, {"text": "<system-reminder>\nbe careful\n</system-reminder>"}),
        _item("prompt", 3, {"text": "caveat text", "is_meta": True}),
    ]
    first, second, third = session_activity(nodes)["nodes"]
    assert first == {
        "t": "note",
        "ts": _at(1),
        "icon": "book",
        "text": "Skill loaded: operating-bluesky-plans",
        "size": len(skill),
    }
    assert (second["t"], second["icon"], second["text"]) == ("note", "info", "Harness instruction")
    assert (third["t"], third["icon"]) == ("note", "info")
    assert "be careful" not in json.dumps([first, second, third])


def test_subagent_report_becomes_a_note_with_its_body():
    text = (
        "Another Claude session sent a message:\n"
        '<agent-message from="a6dcc94303454f548">\n[Subagent hand-back] The scan ran.\n'
        "</agent-message>"
    )
    (node,) = session_activity([_item("prompt", 1, {"text": text, "is_meta": True})])["nodes"]
    assert (node["t"], node["icon"], node["text"]) == ("note", "reply", "Subagent reported back")
    assert node["body"] == "[Subagent hand-back] The scan ran."


def test_interrupt_and_command_output_become_notes():
    nodes = [
        _item("prompt", 1, {"text": "[Request interrupted by user for tool use]"}),
        _item(
            "prompt", 2, {"text": "<local-command-stdout>Set model to opus</local-command-stdout>"}
        ),
        _item(
            "prompt",
            3,
            {"text": "<task-notification>\n<task-id>b1</task-id>\n</task-notification>"},
        ),
    ]
    first, second, third = session_activity(nodes)["nodes"]
    assert (first["t"], first["icon"], first["text"]) == (
        "note",
        "stop",
        "Interrupted by the operator",
    )
    assert (second["icon"], second["text"]) == ("info", "Command output")
    assert second["body"] == "Set model to opus"
    assert (third["icon"], third["text"]) == ("info", "Background task finished")


def test_model_request_error_becomes_a_note():
    error = _item(
        "error", 1, {"error": "Overloaded", "status_code": "529", "attempt": "2", "model": "opus"}
    )
    (node,) = session_activity([error])["nodes"]
    assert (node["t"], node["icon"]) == ("note", "alert")
    assert node["text"] == "Model request failed (529), attempt 2"
    assert node["body"] == "Overloaded"


def test_thinking_and_answer_need_words():
    nodes = [
        _item("thinking", 1, {"text": "consider the scan"}),
        _item("answer", 2, {"text": "   "}),
        _item("answer", 3, {"text": "Done."}),
    ]
    assert session_activity(nodes)["nodes"] == [
        {"t": "thinking", "ts": _at(1), "text": "consider the scan"},
        {"t": "answer", "ts": _at(3), "text": "Done."},
    ]


def test_usage_nodes_carry_cost_and_ms():
    usage = _item("usage", 1, {"cost_usd": 0.0123, "duration_ms": 2400, "ttft_ms": 800})
    assert session_activity([usage])["nodes"] == [
        {"t": "usage", "ts": _at(1), "cost": 0.0123, "ms": 2400}
    ]


# --- subagents --------------------------------------------------------------------


def test_subagent_nests_children_and_brief():
    children = [
        _item("prompt", 5, {"text": "Run the plan and report."}, agent_id="a1"),
        _item("thinking", 6, {"text": "start"}, agent_id="a1"),
        _call("t2", "mcp__bluesky__queue_add", 7, agent_id="a1", result=_result(9, "t2", "ok")),
    ]
    agent = _call(
        "t1",
        "Agent",
        4,
        args={"subagent_type": "plan-runner", "description": "Run the scan"},
        decision=_decision(4.1, "t1"),
        result=_item("subagent", 12, {"agent_id": "a1", "content": "done"}, tool_use_id="t1"),
        subagent={"agent_id": "a1", "timeline": children},
    )
    (node,) = session_activity([agent])["nodes"]
    sub = node["sub"]
    assert (sub["id"], sub["type"], sub["description"]) == ("a1", "plan-runner", "Run the scan")
    assert [c["t"] for c in sub["children"]] == ["brief", "thinking", "tool"]
    assert sub["children"][0]["text"] == "Run the plan and report."
    assert sub["children"][2]["agent"] == "plan-runner"
    assert sub["ms"] == 2000
    assert node["agent"] == "main"
    assert node["result"] is None and node["running"] is False
    assert node["elapsed_ms"] == 8000


def _stream(agent_id: str, *children: dict[str, Any]) -> dict[str, Any]:
    """A subagent's stream as the record serves it while no result names the agent."""
    return {
        "kind": "subagent",
        "agent_id": agent_id,
        "orphan": True,
        "decision": None,
        "result": None,
        "subagent": {"agent_id": agent_id, "timeline": list(children)},
    }


def test_subagent_at_work_is_under_the_call_waiting_on_it():
    waiting = _call(
        "t1",
        "Agent",
        4,
        args={"subagent_type": "plan-runner", "description": "Run the scan"},
        decision=_decision(4.1, "t1"),
    )
    child = _call("t2", "Read", 6, agent_id="a1")
    reading = session_activity([waiting, _stream("a1", child)])
    (node,) = reading["nodes"]
    assert (node["id"], node["name"], node["ts"]) == ("t1", "Agent", _at(4))
    assert node["orphan"] is False and node["running"] is True
    assert node["verdict"] == "allowed"
    sub = node["sub"]
    assert (sub["id"], sub["type"], sub["description"]) == ("a1", "plan-runner", "Run the scan")
    assert [(c["id"], c["agent"]) for c in sub["children"]] == [("t2", "plan-runner")]
    assert reading["entries"] == 3
    assert reading["kinds"] == {"tool_call": 2, "decision": 1}


def test_subagents_at_work_take_the_waiting_calls_in_order():
    first = _call("t1", "Agent", 4, args={"subagent_type": "one"})
    second = _call("t2", "Task", 5, args={"subagent_type": "two"})
    nodes = session_activity(
        [
            first,
            second,
            _stream("a1", _call("c1", "Read", 6, agent_id="a1")),
            _stream("a2", _call("c2", "Read", 7, agent_id="a2")),
        ]
    )["nodes"]
    assert [(n["id"], n["sub"]["id"], n["sub"]["type"]) for n in nodes] == [
        ("t1", "a1", "one"),
        ("t2", "a2", "two"),
    ]


def test_subagent_stream_with_no_call_waiting_stays_an_orphan():
    early = _stream("a0", _call("c0", "Read", 1, agent_id="a0"))
    done = _call(
        "t1",
        "Agent",
        4,
        result=_item("subagent", 12, {"agent_id": "a1", "content": "done"}, tool_use_id="t1"),
        subagent={"agent_id": "a1", "timeline": [_call("c1", "Read", 5, agent_id="a1")]},
    )
    plain = _call("t2", "Read", 13)
    late = _stream("a9", _call("c9", "Read", 14, agent_id="a9"))
    nodes = session_activity([early, done, plain, late])["nodes"]
    assert [(n["id"], n["orphan"], n["running"]) for n in nodes] == [
        ("", True, False),
        ("t1", False, False),
        ("t2", False, True),
        ("", True, False),
    ]
    assert [n["sub"]["id"] for n in nodes if "sub" in n] == ["a0", "a1", "a9"]


def test_orphan_subagent_group_is_kept():
    child = _call("t9", "mcp__controls__channel_read", 20, agent_id="a9")
    orphan = _stream("a9", child)
    reading = session_activity([orphan])
    (node,) = reading["nodes"]
    assert (node["t"], node["name"], node["verdict"]) == ("tool", "Agent", "none")
    assert node["orphan"] is True and node["running"] is False
    assert node["ts"] == _at(20)
    assert [c["id"] for c in node["sub"]["children"]] == ["t9"]
    assert node["sub"]["id"] == "a9"
    assert reading["entries"] == 1 and reading["kinds"] == {"tool_call": 1}


# --- what the record could not give whole ------------------------------------------


def test_incomplete_item_with_null_body():
    nodes = [
        _call(
            "t1", "Read", 1, result=_item("tool_result", 2, None, tool_use_id="t1", incomplete=True)
        ),
        _item("tool_call", 3, None, tool_use_id="t2", incomplete=True),
        _item("prompt", 4, None, incomplete=True),
        _item("answer", 5, None, incomplete=True),
        _item("usage", 6, None, incomplete=True),
    ]
    reading = session_activity(nodes)
    first, second = _tools(reading)
    assert first["incomplete"] is True and first["result"] is None
    assert first["running"] is False
    assert second["incomplete"] is True
    assert (second["name"], second["input"], second["id"]) == ("", {}, "t2")
    assert reading["entries"] == 6
    notes = [n for n in reading["nodes"] if n["t"] == "note"]
    assert [n["text"] for n in notes] == ["Incomplete prompt entry", "Incomplete answer entry"]
    assert all(n["icon"] == "alert" for n in notes)


def test_complete_call_is_not_marked_incomplete():
    (tool,) = _tools(session_activity([_call("t1", "Read", 1, result=_result(2, "t1", "x"))]))
    assert tool["incomplete"] is False
    assert "orphan" not in tool or tool["orphan"] is False


def test_malformed_nodes_are_skipped():
    reading = session_activity([None, "x", {"kind": "tool_call"}, {"kind": "prompt", "ts": 7}])  # type: ignore[list-item]
    assert [n["t"] for n in reading["nodes"]] == ["tool"]
    assert reading["nodes"][0]["ts"] == ""


# --- the summary ------------------------------------------------------------------


def test_debug_is_dropped_and_counts():
    nodes = [
        _item("debug", 0, {"event": "api_request"}),
        _item("prompt", 1, {"text": "read the orbit"}),
        _call(
            "t1",
            "mcp__controls__channel_read",
            2,
            decision=_decision(2.1, "t1"),
            result=_result(3, "t1", "1.0"),
        ),
        _audit(2.5, "t1", surface="tool_call", duration_ms=5, target="va"),
        _item("usage", 4, {"cost_usd": 0.01, "duration_ms": 900}),
        _item("debug", 9, {"event": "late"}),
    ]
    reading = session_activity(nodes)
    assert [n["t"] for n in reading["nodes"]] == ["prompt", "tool", "usage"]
    assert reading["entries"] == 6
    assert reading["kinds"] == {
        "prompt": 1,
        "tool_call": 1,
        "decision": 1,
        "tool_result": 1,
        "audit": 1,
        "usage": 1,
    }
    # The span is the transcript's: an audit or usage row carries the session
    # key, which after a /clear names an older transcript.
    assert (reading["start"], reading["end"]) == (_at(1), _at(3))


def test_empty_timeline():
    assert session_activity([]) == {
        "nodes": [],
        "entries": 0,
        "kinds": {},
        "start": None,
        "end": None,
        "target": None,
        "posture": None,
    }
