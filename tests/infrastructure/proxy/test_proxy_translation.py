"""L2a — deterministic unit tests for the Anthropic<->OpenAI translation layer.

These exercise the pure functions in osprey.infrastructure.proxy.translator with
no network, establishing that the Route B mechanism (Claude Code speaks Anthropic
to the local proxy; proxy speaks OpenAI to the upstream) translates text AND tool
calls correctly. This is the core of "CBORG self-hosted models in Claude Code".

Experiment branch: experiment/cborg-claude-code (issue #259).
"""

from __future__ import annotations

import json

import pytest

from osprey.infrastructure.proxy.translator import (
    _IMAGE_NOT_CARRIED,
    _IMAGE_SOURCE_NOT_CARRIED,
    anthropic_to_openai_request,
    openai_to_anthropic_response,
)


def _png_block(data: str = "iVBORw0KGgo=") -> dict:
    """A base64 ``image/png`` content block."""
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


def _tool_use_turn(*ids: str) -> dict:
    """An assistant turn calling ``read_screen`` once per id."""
    return {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": i, "name": "read_screen", "input": {}} for i in ids],
    }


# ── Request translation: Anthropic → OpenAI ──────────────────────────


def test_system_and_user_text_request():
    body = {
        "model": "cborg-coder",
        "system": "You are a control-system assistant.",
        "messages": [{"role": "user", "content": "What is a PV?"}],
        "max_tokens": 64,
    }
    out = anthropic_to_openai_request(body).body
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
    out = anthropic_to_openai_request(body).body
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
    out = anthropic_to_openai_request(body).body
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
    out = anthropic_to_openai_request(body).body
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
    out = anthropic_to_openai_request(body).body
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
    out = anthropic_to_openai_request(body).body
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
    out = anthropic_to_openai_request(body).body
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
    out = anthropic_to_openai_request(body).body
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
    assert out.body["messages"][0]["content"] == "answer"
    assert "thinking" not in json.dumps(out.body)
    assert out.dropped == {"thinking"}


def test_the_request_keeps_max_tokens_and_temperature_by_default():
    body = {
        "model": "cborg-coder",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 64,
        "temperature": 0.0,
    }
    out = anthropic_to_openai_request(body)
    assert out.body["max_tokens"] == 64
    assert out.body["temperature"] == 0.0
    assert "max_completion_tokens" not in out.body
    assert out.dropped == frozenset()


def test_the_token_cap_goes_out_under_the_declared_parameter():
    """OpenAI's own API refuses max_tokens on its reasoning models."""
    body = {
        "model": "gpt-6-sol",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 64,
    }
    out = anthropic_to_openai_request(body, max_tokens_param="max_completion_tokens").body
    assert out["max_completion_tokens"] == 64
    assert "max_tokens" not in out


def test_no_temperature_goes_out_where_the_upstream_refuses_one():
    body = {
        "model": "gpt-6-sol",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 64,
        "temperature": 0.0,
    }
    out = anthropic_to_openai_request(
        body, max_tokens_param="max_completion_tokens", accepts_temperature=False
    )
    assert "temperature" not in out.body
    assert out.body["max_completion_tokens"] == 64
    assert "temperature" in out.dropped


# ── Images, and what the route does not carry ────────────────────────


def test_a_text_only_request_translates_as_before():
    body = {
        "model": "m",
        "messages": [
            _tool_use_turn("toolu_1"),
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
                    },
                    {"type": "text", "text": "one"},
                    {"type": "text", "text": "two"},
                ],
            },
        ],
    }
    out = anthropic_to_openai_request(body, supports_images=True)
    assert out.body["messages"][1:] == [
        {"role": "tool", "tool_call_id": "toolu_1", "content": "a\nb"},
        {"role": "user", "content": "one\ntwo"},
    ]
    assert out.dropped == frozenset()
    assert out.images_sent == 0


def test_a_user_image_becomes_a_data_url_part_where_the_route_takes_images():
    body = {
        "model": "m",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    _png_block(),
                    {"type": "text", "text": "after"},
                ],
            }
        ],
    }
    out = anthropic_to_openai_request(body, supports_images=True)
    assert out.body["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "before"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
                {"type": "text", "text": "after"},
            ],
        }
    ]
    assert out.images_sent == 1
    assert out.dropped == frozenset()


def test_a_url_image_passes_its_url_through():
    block = {"type": "image", "source": {"type": "url", "url": "https://example.test/a.png"}}
    body = {"model": "m", "messages": [{"role": "user", "content": [block]}]}
    out = anthropic_to_openai_request(body, supports_images=True)
    assert out.body["messages"][0]["content"] == [
        {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}}
    ]
    assert out.images_sent == 1


def test_a_tool_result_image_rides_the_next_user_message():
    body = {
        "model": "m",
        "messages": [
            _tool_use_turn("toolu_1"),
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [{"type": "text", "text": "shot"}, _png_block()],
                    },
                    {"type": "text", "text": "look"},
                ],
            },
        ],
    }
    out = anthropic_to_openai_request(body, supports_images=True)
    messages = out.body["messages"]
    assert [m["role"] for m in messages] == ["assistant", "tool", "user"]
    assert messages[1] == {
        "role": "tool",
        "tool_call_id": "toolu_1",
        "content": "shot\n[image: sent in the next user message]",
    }
    assert messages[2]["content"] == [
        {"type": "text", "text": "Images returned by tool call toolu_1:"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
        {"type": "text", "text": "look"},
    ]
    assert out.images_sent == 1


def test_images_from_two_tool_results_follow_both_tool_messages():
    body = {
        "model": "m",
        "messages": [
            _tool_use_turn("toolu_1", "toolu_2"),
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [_png_block("QQ==")],
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_2",
                        "content": [_png_block("Qg==")],
                    },
                ],
            },
        ],
    }
    out = anthropic_to_openai_request(body, supports_images=True)
    messages = out.body["messages"]
    assert [m["role"] for m in messages] == ["assistant", "tool", "tool", "user"]
    assert messages[3]["content"] == [
        {"type": "text", "text": "Images returned by tool call toolu_1:"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,QQ=="}},
        {"type": "text", "text": "Images returned by tool call toolu_2:"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,Qg=="}},
    ]
    assert out.images_sent == 2


@pytest.mark.parametrize("where", ["user", "tool_result"])
def test_an_image_on_a_route_without_images_is_named_in_the_turn(where):
    if where == "user":
        messages = [{"role": "user", "content": [{"type": "text", "text": "see"}, _png_block()]}]
    else:
        messages = [
            _tool_use_turn("toolu_1"),
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": [_png_block()]}
                ],
            },
        ]
    out = anthropic_to_openai_request({"model": "m", "messages": messages})
    assert _IMAGE_NOT_CARRIED in json.dumps(out.body)
    assert "image_url" not in json.dumps(out.body)
    assert "image" in out.dropped
    assert out.images_sent == 0


def test_an_image_by_file_reference_is_named_as_not_carried():
    block = {"type": "image", "source": {"type": "file", "file_id": "file_1"}}
    body = {"model": "m", "messages": [{"role": "user", "content": [block]}]}
    out = anthropic_to_openai_request(body, supports_images=True)
    assert out.body["messages"] == [{"role": "user", "content": _IMAGE_SOURCE_NOT_CARRIED}]
    assert "image reference" in out.dropped
    assert out.images_sent == 0


def test_a_document_is_named_in_the_turn():
    document = {
        "type": "document",
        "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0="},
    }
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "read"}, document]}],
    }
    out = anthropic_to_openai_request(body, supports_images=True)
    assert out.body["messages"][0]["content"] == (
        "read\n[document not sent: this provider's route does not carry it]"
    )
    assert "document" in out.dropped


def test_a_thinking_request_is_named_as_dropped():
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "enabled", "budget_tokens": 1024},
    }
    out = anthropic_to_openai_request(body)
    assert "thinking" not in out.body
    assert out.dropped == {"thinking"}


def test_disabled_thinking_is_not_a_drop():
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": {"type": "disabled"},
    }
    assert anthropic_to_openai_request(body).dropped == frozenset()


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
