"""L2a — deterministic unit tests for the Anthropic<->OpenAI translation layer.

These exercise the pure functions in osprey.infrastructure.proxy.translator with
no network, establishing that the Route B mechanism (Claude Code speaks Anthropic
to the local proxy; proxy speaks OpenAI to the upstream) translates text AND tool
calls correctly. This is the core of "CBORG self-hosted models in Claude Code".

Experiment branch: experiment/cborg-claude-code (issue #259).
"""

from __future__ import annotations

import json

from osprey.infrastructure.proxy.translator import (
    anthropic_to_openai_request,
    openai_to_anthropic_response,
)

# ── Request translation: Anthropic → OpenAI ──────────────────────────


def test_system_and_user_text_request():
    body = {
        "model": "cborg-coder",
        "system": "You are a control-system assistant.",
        "messages": [{"role": "user", "content": "What is a PV?"}],
        "max_tokens": 64,
    }
    out = anthropic_to_openai_request(body)
    assert out["model"] == "cborg-coder"
    assert out["max_tokens"] == 64
    assert out["messages"][0] == {
        "role": "system",
        "content": "You are a control-system assistant.",
    }
    assert out["messages"][1] == {"role": "user", "content": "What is a PV?"}


def test_system_as_block_list_is_flattened():
    body = {
        "model": "cborg-coder",
        "system": [
            {"type": "text", "text": "Line one."},
            {"type": "text", "text": "Line two."},
        ],
        "messages": [{"role": "user", "content": "hi"}],
    }
    out = anthropic_to_openai_request(body)
    assert out["messages"][0] == {"role": "system", "content": "Line one.\nLine two."}


def test_embedded_system_message_is_hoisted_not_dropped():
    """A ``role: system`` entry *inside* ``messages`` must be preserved as an
    OpenAI system message, not silently dropped (issue #285).

    The Anthropic Messages API places the system prompt in the top-level
    ``system`` field, so a strict gateway rejects a ``role: system`` entry in
    the array. The proxy used to drop it on the floor, silently losing the
    instruction; it must hoist it into an OpenAI system message instead.
    """
    body = {
        "model": "cborg-coder",
        "messages": [
            {"role": "system", "content": "You are the channel-finder subagent."},
            {"role": "user", "content": "find PV X"},
        ],
    }
    out = anthropic_to_openai_request(body)
    roles = [m["role"] for m in out["messages"]]
    assert roles == ["system", "user"]
    assert out["messages"][0] == {
        "role": "system",
        "content": "You are the channel-finder subagent.",
    }
    assert out["messages"][1] == {"role": "user", "content": "find PV X"}


def test_embedded_system_block_list_is_flattened():
    """A ``role: system`` message whose content is a block list is flattened
    to text, mirroring how the top-level ``system`` field is handled."""
    body = {
        "model": "cborg-coder",
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Line one."},
                    {"type": "text", "text": "Line two."},
                ],
            },
            {"role": "user", "content": "hi"},
        ],
    }
    out = anthropic_to_openai_request(body)
    assert out["messages"][0] == {"role": "system", "content": "Line one.\nLine two."}
    assert out["messages"][1] == {"role": "user", "content": "hi"}


def test_top_level_and_embedded_system_both_preserved():
    """When both a top-level ``system`` and an embedded ``role: system`` message
    are present, both survive, with the top-level prompt first."""
    body = {
        "model": "cborg-coder",
        "system": "Top-level system.",
        "messages": [
            {"role": "system", "content": "Embedded system."},
            {"role": "user", "content": "go"},
        ],
    }
    out = anthropic_to_openai_request(body)
    assert [m["role"] for m in out["messages"]] == ["system", "system", "user"]
    assert out["messages"][0]["content"] == "Top-level system."
    assert out["messages"][1]["content"] == "Embedded system."


def test_tool_definitions_become_openai_functions():
    body = {
        "model": "cborg-coder",
        "messages": [{"role": "user", "content": "read PV X"}],
        "tools": [
            {
                "name": "read_pv",
                "description": "Read a process variable.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            }
        ],
        "tool_choice": {"type": "any"},
    }
    out = anthropic_to_openai_request(body)
    assert out["tool_choice"] == "required"  # any → required
    assert out["tools"][0]["type"] == "function"
    fn = out["tools"][0]["function"]
    assert fn["name"] == "read_pv"
    assert fn["description"] == "Read a process variable."
    assert fn["parameters"]["properties"]["name"]["type"] == "string"


def test_tool_choice_specific_tool_maps_to_function():
    body = {
        "model": "cborg-coder",
        "messages": [{"role": "user", "content": "x"}],
        "tools": [{"name": "read_pv", "input_schema": {}}],
        "tool_choice": {"type": "tool", "name": "read_pv"},
    }
    out = anthropic_to_openai_request(body)
    assert out["tool_choice"] == {"type": "function", "function": {"name": "read_pv"}}


def test_assistant_tool_use_and_user_tool_result_roundtrip():
    """A full tool-call turn must preserve the tool_use_id <-> tool_call_id link."""
    body = {
        "model": "cborg-coder",
        "messages": [
            {"role": "user", "content": "read PV X"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Calling the tool."},
                    {
                        "type": "tool_use",
                        "id": "toolu_abc",
                        "name": "read_pv",
                        "input": {"name": "X"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_abc", "content": "3.14"}],
            },
        ],
    }
    out = anthropic_to_openai_request(body)
    roles = [m["role"] for m in out["messages"]]
    assert roles == ["user", "assistant", "tool"]

    assistant = out["messages"][1]
    assert assistant["content"] == "Calling the tool."
    tc = assistant["tool_calls"][0]
    assert tc["id"] == "toolu_abc"
    assert tc["function"]["name"] == "read_pv"
    assert json.loads(tc["function"]["arguments"]) == {"name": "X"}

    tool_msg = out["messages"][2]
    assert tool_msg["tool_call_id"] == "toolu_abc"  # link preserved
    assert tool_msg["content"] == "3.14"


def test_thinking_blocks_are_stripped():
    """Open models via OpenAI format can't take Anthropic thinking blocks."""
    body = {
        "model": "cborg-coder",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "secret reasoning"},
                    {"type": "text", "text": "answer"},
                ],
            }
        ],
    }
    out = anthropic_to_openai_request(body)
    assert out["messages"][0]["content"] == "answer"
    assert "thinking" not in json.dumps(out)


# ── Response translation: OpenAI → Anthropic ─────────────────────────


def test_text_response_translation():
    openai_resp = {
        "choices": [
            {"message": {"role": "assistant", "content": "A PV is..."}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    out = openai_to_anthropic_response(openai_resp, "cborg-coder")
    assert out["type"] == "message"
    assert out["role"] == "assistant"
    assert out["model"] == "cborg-coder"
    assert out["stop_reason"] == "end_turn"
    assert out["content"] == [{"type": "text", "text": "A PV is..."}]
    assert out["usage"] == {"input_tokens": 10, "output_tokens": 5}


def test_tool_call_response_translation():
    openai_resp = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_xyz",
                            "type": "function",
                            "function": {"name": "read_pv", "arguments": '{"name": "X"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8},
    }
    out = openai_to_anthropic_response(openai_resp, "cborg-coder")
    assert out["stop_reason"] == "tool_use"  # tool_calls → tool_use
    block = out["content"][0]
    assert block["type"] == "tool_use"
    assert block["id"] == "call_xyz"
    assert block["name"] == "read_pv"
    assert block["input"] == {"name": "X"}


def test_malformed_tool_arguments_dont_crash():
    """Open models sometimes emit invalid JSON in tool arguments."""
    openai_resp = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{not json"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }
    out = openai_to_anthropic_response(openai_resp, "cborg-coder")
    block = out["content"][0]
    assert block["type"] == "tool_use"
    assert block["input"] == {"raw": "{not json"}  # falls back, no crash
