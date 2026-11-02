"""The transcript parser: Claude Code JSONL lines to record entries.

Lines are built inline with the CLI 2.1.300 field set (``uuid``,
``parentUuid``, ``sessionId``, ``agentId``, ``isSidechain``, ``version``,
``message.usage``); there is no captured transcript fixture yet.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from osprey.agent_record import contract as c
from osprey.agent_record.ingest import transcripts as t

IDENTITY = "alice-claude-projects"
SESSION = "5f0c9a3e-1111-4222-8333-944455556666"
MAIN = f"-data-workspace/{SESSION}.jsonl"
NOW = datetime(2026, 10, 23, 18, 0, tzinfo=UTC)
TS = "2026-10-23T12:00:00.000Z"

USAGE = {
    "input_tokens": 12,
    "output_tokens": 34,
    "cache_creation_input_tokens": 100,
    "cache_read_input_tokens": 2000,
    "cache_creation": {"ephemeral_5m_input_tokens": 100, "ephemeral_1h_input_tokens": 0},
    "service_tier": "standard",
}


def _line(obj: dict) -> bytes:
    return (json.dumps(obj) + "\n").encode("utf-8")


def _assistant(content: list, *, uuid: str = "a-1", agent_id: str | None = None, **extra) -> dict:
    obj = {
        "parentUuid": "u-1",
        "isSidechain": agent_id is not None,
        "userType": "external",
        "cwd": "/data/workspace",
        "sessionId": SESSION,
        "version": "2.1.300",
        "gitBranch": "main",
        "message": {
            "id": "msg_01",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": content,
            "stop_reason": None,
            "usage": USAGE,
        },
        "requestId": "req_01",
        "type": "assistant",
        "uuid": uuid,
        "timestamp": TS,
    }
    if agent_id is not None:
        obj["agentId"] = agent_id
    obj.update(extra)
    return obj


def _user(content, *, uuid: str = "u-1", parent: str | None = None, **extra) -> dict:
    obj = {
        "parentUuid": parent,
        "isSidechain": False,
        "userType": "external",
        "cwd": "/data/workspace",
        "sessionId": SESSION,
        "version": "2.1.300",
        "type": "user",
        "message": {"role": "user", "content": content},
        "uuid": uuid,
        "timestamp": TS,
    }
    obj.update(extra)
    return obj


def _parse(obj_or_bytes, relpath: str = MAIN):
    line = obj_or_bytes if isinstance(obj_or_bytes, bytes) else _line(obj_or_bytes)
    return t.parse_line(IDENTITY, relpath, line, now=NOW)


def _key(uuid: str, block: int, session: str = SESSION) -> str:
    return f"cc:{IDENTITY}:{session}:{uuid}:{block}"


# ---------------------------------------------------------------------------
# prompts and answers
# ---------------------------------------------------------------------------


def test_string_prompt_is_one_prompt_entry():
    (entry,) = _parse(_user("What is the beam current?"))
    assert entry.kind == c.KIND_PROMPT
    assert entry.key == _key("u-1", 0)
    assert entry.body["text"] == "What is the beam current?"
    assert entry.identity == IDENTITY
    assert entry.session_id == SESSION
    assert entry.ts == datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
    assert entry.agent_id == ""
    assert entry.parent_key == ""


def test_answer_carries_usage_token_counts():
    (entry,) = _parse(_assistant([{"type": "text", "text": "It is 500 mA."}]))
    assert entry.kind == c.KIND_ANSWER
    assert entry.key == _key("a-1", 0)
    assert entry.body["text"] == "It is 500 mA."
    assert entry.body["usage"] == USAGE
    assert entry.body["model"] == "claude-opus-5-5"
    assert entry.body["message_id"] == "msg_01"
    assert entry.parent_key == _key("u-1", 0)


def test_usage_goes_only_into_the_answer_body():
    entries = _parse(
        _assistant(
            [
                {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                {"type": "text", "text": "Done."},
                {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}},
            ]
        )
    )
    with_usage = [e.kind for e in entries if "usage" in e.body]
    assert with_usage == [c.KIND_ANSWER]


# ---------------------------------------------------------------------------
# multi-block lines and keys
# ---------------------------------------------------------------------------


def test_multi_block_line_gives_one_entry_per_block_keyed_by_index():
    entries = _parse(
        _assistant(
            [
                {"type": "text", "text": "Running two commands."},
                {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}},
                {"type": "tool_use", "id": "toolu_2", "name": "Read", "input": {"path": "x"}},
            ]
        )
    )
    assert [e.kind for e in entries] == [c.KIND_ANSWER, c.KIND_TOOL_CALL, c.KIND_TOOL_CALL]
    assert [e.key for e in entries] == [_key("a-1", 0), _key("a-1", 1), _key("a-1", 2)]
    assert [e.tool_use_id for e in entries] == ["", "toolu_1", "toolu_2"]
    assert entries[1].body["name"] == "Bash"
    assert entries[1].body["input"] == {"command": "ls"}
    assert len({e.unique_key for e in entries}) == 3


def test_line_without_uuid_is_keyed_by_a_hash_of_the_line():
    obj = _user("hello")
    del obj["uuid"]
    raw = _line(obj)
    (entry,) = _parse(raw)
    expected = c.transcript_key(IDENTITY, SESSION, None, 0, line=raw.rstrip(b"\r\n"))
    assert entry.key == expected
    # The same bytes always give the same key; different bytes a different one.
    assert _parse(raw)[0].key == expected
    other = _line({**obj, "message": {"role": "user", "content": "bye"}})
    assert _parse(other)[0].key != expected


def test_keys_do_not_depend_on_key_order():
    obj = _assistant([{"type": "text", "text": "same"}])
    reordered = dict(reversed(list(obj.items())))
    reordered["message"] = dict(reversed(list(obj["message"].items())))
    first, second = _parse(obj), _parse(reordered)
    assert [e.key for e in first] == [e.key for e in second]
    assert [e.kind for e in first] == [e.kind for e in second]
    assert first[0].body == second[0].body


def test_the_same_line_parses_to_the_same_entries():
    obj = _assistant([{"type": "text", "text": "x"}])
    assert _parse(obj) == _parse(obj)


# ---------------------------------------------------------------------------
# thinking
# ---------------------------------------------------------------------------


def test_thinking_text_is_the_block_thinking_field():
    (entry,) = _parse(
        _assistant([{"type": "thinking", "thinking": "Check the PV first.", "signature": "s"}])
    )
    assert entry.kind == c.KIND_THINKING
    assert entry.body["text"] == "Check the PV first."
    assert "usage" not in entry.body


def test_omitted_empty_thinking_block_is_kept_with_empty_text():
    entries = _parse(
        _assistant(
            [
                {"type": "thinking", "thinking": "", "signature": "EqQBCkYIBRgCKkB"},
                {"type": "text", "text": "Answer."},
            ]
        )
    )
    assert [e.kind for e in entries] == [c.KIND_THINKING, c.KIND_ANSWER]
    assert entries[0].key == _key("a-1", 0)
    assert entries[0].body["text"] == ""
    assert entries[1].key == _key("a-1", 1)


def test_redacted_thinking_becomes_thinking_flagged_redacted():
    (entry,) = _parse(_assistant([{"type": "redacted_thinking", "data": "opaque-ciphertext"}]))
    assert entry.kind == c.KIND_THINKING
    assert entry.body["redacted"] is True
    assert entry.body["text"] == ""


# ---------------------------------------------------------------------------
# tool results and images
# ---------------------------------------------------------------------------


def test_tool_result_keeps_full_content():
    result = {
        "type": "tool_result",
        "tool_use_id": "toolu_1",
        "content": [{"type": "text", "text": "a\nb\n" * 500}],
        "is_error": False,
    }
    (entry,) = _parse(_user([result], uuid="u-2", parent="a-1", toolUseResult={"stdout": "a\nb\n"}))
    assert entry.kind == c.KIND_TOOL_RESULT
    assert entry.tool_use_id == "toolu_1"
    assert entry.body["content"] == result["content"]
    assert entry.body["is_error"] is False
    assert entry.parent_key == _key("a-1", 0)


def test_image_result_is_kept_whole():
    image = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo" * 200},
    }
    result = {
        "type": "tool_result",
        "tool_use_id": "toolu_9",
        "content": [{"type": "text", "text": "screenshot"}, image],
    }
    (entry,) = _parse(_user([result], uuid="u-3"))
    assert entry.kind == c.KIND_TOOL_RESULT
    assert entry.body["content"][1] == image


def test_prompt_with_text_and_image_gives_two_entries():
    image = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": "QQ=="},
    }
    entries = _parse(_user([{"type": "text", "text": "look"}, image]))
    assert [e.kind for e in entries] == [c.KIND_PROMPT, c.KIND_PROMPT]
    assert entries[0].body["text"] == "look"
    assert entries[1].body["image"] == image
    assert [e.key for e in entries] == [_key("u-1", 0), _key("u-1", 1)]


# ---------------------------------------------------------------------------
# subagents
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tool_name", ["Agent", "Task"])
def test_subagent_result_links_by_agent_id_for_either_tool_name(tool_name):
    call = _parse(
        _assistant(
            [
                {
                    "type": "tool_use",
                    "id": "toolu_sa",
                    "name": tool_name,
                    "input": {"description": "scan", "prompt": "find it"},
                }
            ]
        )
    )
    assert [e.kind for e in call] == [c.KIND_TOOL_CALL]
    result = _parse(
        _user(
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_sa",
                    "content": [{"type": "text", "text": "Async agent launched."}],
                }
            ],
            uuid="u-9",
            toolUseResult={"isAsync": True, "status": "async_launched", "agentId": "a3f9c2"},
        )
    )
    (entry,) = result
    assert entry.kind == c.KIND_SUBAGENT
    assert entry.agent_id == "a3f9c2"
    assert entry.tool_use_id == "toolu_sa"
    assert entry.body["transcript"] == f"-data-workspace/{SESSION}/subagents/agent-a3f9c2.jsonl"
    assert entry.body["content"] == [{"type": "text", "text": "Async agent launched."}]
    assert entry.body["status"] == "async_launched"


def test_subagent_agent_id_from_result_text_when_no_structured_result():
    text = "Report done.\nagentId: b7e1 (for resuming to continue this agent's work if needed)"
    (entry,) = _parse(
        _user(
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_x",
                    "content": [{"type": "text", "text": text}],
                }
            ]
        )
    )
    assert entry.kind == c.KIND_SUBAGENT
    assert entry.agent_id == "b7e1"


def test_ordinary_tool_output_mentioning_an_agent_id_is_a_tool_result():
    text = 'transcript_reader.py:778: if parsed.get("agentId"): agentId: deadbeef'
    (entry,) = _parse(
        _user(
            [{"type": "tool_result", "tool_use_id": "toolu_g", "content": text}],
            toolUseResult={"stdout": text, "stderr": "", "interrupted": False},
        )
    )
    assert entry.kind == c.KIND_TOOL_RESULT
    assert entry.agent_id == ""


def test_subagent_file_lines_carry_the_agent_id_from_the_file_stem():
    relpath = f"-data-workspace/{SESSION}/subagents/agent-a3f9c2.jsonl"
    entries = _parse(
        _assistant(
            [
                {"type": "thinking", "thinking": "sub thinks", "signature": "s"},
                {
                    "type": "tool_use",
                    "id": "toolu_h",
                    "name": "SubagentHandback",
                    "input": {"m": "x"},
                },
            ],
            uuid="s-1",
            agent_id="a3f9c2",
        ),
        relpath,
    )
    assert [e.kind for e in entries] == [c.KIND_THINKING, c.KIND_TOOL_CALL]
    assert all(e.agent_id == "a3f9c2" for e in entries)
    assert all(e.session_id == SESSION for e in entries)
    assert entries[0].key == _key("s-1", 0)


def test_subagent_file_line_without_agent_id_uses_the_stem():
    relpath = f"-data-workspace/{SESSION}/subagents/agent-zz9.jsonl"
    (entry,) = _parse(_user("sub prompt", uuid="s-0"), relpath)
    assert entry.agent_id == "zz9"


# ---------------------------------------------------------------------------
# Debug mode drops nothing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "obj",
    [
        {
            "type": "system",
            "subtype": "compact_boundary",
            "uuid": "sy-1",
            "sessionId": SESSION,
            "timestamp": TS,
            "content": "Conversation compacted",
        },
        {"type": "summary", "summary": "Beam work", "leafUuid": "a-1"},
        {"type": "attachment", "attachment": {"kind": "file"}},
        {"type": "brand-new-line-type", "whatever": [1, 2, 3]},
    ],
    ids=["system", "summary", "attachment", "unknown"],
)
def test_unknown_line_types_become_debug_with_the_raw_line(obj):
    raw = _line(obj)
    (entry,) = _parse(raw)
    assert entry.kind == c.KIND_DEBUG
    assert entry.body["raw"] == raw.decode("utf-8").rstrip("\n")
    assert entry.body["type"] == obj["type"]
    assert entry.session_id == SESSION
    assert entry.key.startswith(f"cc:{IDENTITY}:{SESSION}:")


def test_debug_line_without_timestamp_falls_back_to_now():
    (entry,) = _parse({"type": "last-prompt", "lastPrompt": "hi"})
    assert entry.ts == NOW


def test_debug_line_with_uuid_is_keyed_by_uuid():
    (entry,) = _parse({"type": "system", "uuid": "sy-2", "sessionId": SESSION, "timestamp": TS})
    assert entry.key == _key("sy-2", 0)


def test_unparseable_line_becomes_debug():
    raw = b'{"type": "assistant", "trunc'
    (entry,) = _parse(raw)
    assert entry.kind == c.KIND_DEBUG
    assert entry.body["raw"] == raw.decode()
    assert entry.body["type"] is None
    assert entry.ts == NOW


def test_unknown_content_block_becomes_debug_and_neighbours_survive():
    block = {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {}}
    entries = _parse(_assistant([{"type": "text", "text": "Searching."}, block]))
    assert [e.kind for e in entries] == [c.KIND_ANSWER, c.KIND_DEBUG]
    assert entries[1].body["block"] == block
    assert entries[1].key == _key("a-1", 1)


def test_assistant_line_without_content_becomes_debug():
    obj = _assistant([])
    (entry,) = _parse(obj)
    assert entry.kind == c.KIND_DEBUG


def test_blank_line_yields_nothing():
    assert _parse(b"\n") == []
    assert _parse(b"   \r\n") == []


# ---------------------------------------------------------------------------
# contract conformance
# ---------------------------------------------------------------------------


def test_every_kind_is_allowed_on_the_transcript_route():
    lines = [
        _user("p"),
        _assistant(
            [
                {"type": "thinking", "thinking": "t", "signature": "s"},
                {"type": "text", "text": "a"},
                {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {}},
            ]
        ),
        _user([{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}], uuid="u-5"),
        {"type": "system"},
    ]
    kinds = {e.kind for obj in lines for e in _parse(obj)}
    assert kinds <= c.ROUTE_KINDS[c.ROUTE_TRANSCRIPT]


def test_session_id_falls_back_to_the_path_when_the_line_has_none():
    obj = _user("x")
    del obj["sessionId"]
    (entry,) = _parse(obj)
    assert entry.session_id == SESSION


def test_bad_relpath_is_refused():
    with pytest.raises(ValueError):
        t.parse_line(IDENTITY, "/abs/x.jsonl", _line(_user("x")), now=NOW)
    with pytest.raises(ValueError):
        t.parse_line(IDENTITY, "../x.jsonl", _line(_user("x")), now=NOW)


def test_parser_module_does_no_io_imports():
    import ast
    from pathlib import Path

    tree = ast.parse(Path(t.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"os", "io", "pathlib", "shutil", "subprocess", "socket", "duckdb"}
