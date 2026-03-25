"""Pure translation functions between Anthropic Messages API and OpenAI Chat Completions.

All functions are stateless and perform no I/O — they transform dicts.
"""

from __future__ import annotations

import json
import random
import string
import time


def _gen_id(prefix: str = "msg_") -> str:
    chars = string.ascii_lowercase + string.digits
    return prefix + "".join(random.choices(chars, k=24))


# ── Request: Anthropic → OpenAI ──────────────────────────────────────


def anthropic_to_openai_request(body: dict) -> dict:
    """Convert an Anthropic Messages API request body to OpenAI Chat Completions."""
    messages = _convert_messages(body.get("messages", []), body.get("system"))
    tools = _convert_tools_to_openai(body.get("tools"))

    openai_body: dict = {
        "model": body.get("model", ""),
        "messages": messages,
        "stream": body.get("stream", False),
    }

    if body.get("max_tokens"):
        openai_body["max_tokens"] = body["max_tokens"]
    if body.get("temperature") is not None:
        openai_body["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        openai_body["top_p"] = body["top_p"]
    if body.get("stop_sequences"):
        openai_body["stop"] = body["stop_sequences"]
    if tools:
        openai_body["tools"] = tools

    # tool_choice mapping
    tc = body.get("tool_choice")
    if tc:
        if isinstance(tc, dict):
            tc_type = tc.get("type")
            if tc_type == "auto":
                openai_body["tool_choice"] = "auto"
            elif tc_type == "any":
                openai_body["tool_choice"] = "required"
            elif tc_type == "tool":
                openai_body["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tc.get("name", "")},
                }

    return openai_body


def _convert_messages(
    anthropic_messages: list[dict],
    system: str | list | None,
) -> list[dict]:
    """Convert Anthropic message array to OpenAI message array."""
    openai_messages: list[dict] = []

    # System prompt
    if system:
        if isinstance(system, str):
            openai_messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            # Anthropic system can be list of content blocks
            text_parts = [
                b["text"] for b in system if isinstance(b, dict) and b.get("type") == "text"
            ]
            if text_parts:
                openai_messages.append({"role": "system", "content": "\n".join(text_parts)})

    for msg in anthropic_messages:
        role = msg.get("role", "user")
        content = msg.get("content")

        if role == "user":
            openai_messages.extend(_convert_user_message(content))
        elif role == "assistant":
            openai_messages.extend(_convert_assistant_message(content))

    return openai_messages


def _convert_user_message(content) -> list[dict]:
    """Convert a single Anthropic user message to OpenAI format."""
    if isinstance(content, str):
        return [{"role": "user", "content": content}]

    if isinstance(content, list):
        # May contain text blocks and tool_result blocks
        text_parts = []
        tool_results = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_result":
                    tool_results.append(block)
                elif btype == "image":
                    # Skip images for now — Phase 3
                    text_parts.append("[image content omitted]")

        messages = []
        # Emit tool results first (OpenAI requires role=tool for each)
        for tr in tool_results:
            tr_content = tr.get("content", "")
            if isinstance(tr_content, list):
                tr_content = "\n".join(
                    b.get("text", "") for b in tr_content if isinstance(b, dict)
                )
            messages.append({
                "role": "tool",
                "tool_call_id": tr.get("tool_use_id", ""),
                "content": str(tr_content),
            })

        if text_parts:
            messages.append({"role": "user", "content": "\n".join(text_parts)})

        return messages if messages else [{"role": "user", "content": ""}]

    return [{"role": "user", "content": str(content)}]


def _convert_assistant_message(content) -> list[dict]:
    """Convert an Anthropic assistant message to OpenAI format."""
    if isinstance(content, str):
        return [{"role": "assistant", "content": content}]

    if isinstance(content, list):
        text_parts = []
        tool_calls = []
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    tool_calls.append({
                        "id": block.get("id", _gen_id("call_")),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    })
                elif btype == "thinking":
                    # Strip thinking blocks — not supported by OpenAI
                    pass

        msg: dict = {"role": "assistant"}
        msg["content"] = "\n".join(text_parts) if text_parts else None
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return [msg]

    return [{"role": "assistant", "content": str(content)}]


def _convert_tools_to_openai(anthropic_tools: list[dict] | None) -> list[dict] | None:
    """Convert Anthropic tool definitions to OpenAI function-calling format."""
    if not anthropic_tools:
        return None

    openai_tools = []
    for tool in anthropic_tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            },
        })
    return openai_tools


# ── Response: OpenAI → Anthropic ─────────────────────────────────────


_FINISH_REASON_MAP = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "end_turn",
}


def openai_to_anthropic_response(openai_resp: dict, model: str) -> dict:
    """Convert an OpenAI Chat Completions response to Anthropic Messages format."""
    choice = (openai_resp.get("choices") or [{}])[0]
    message = choice.get("message", {})
    finish = choice.get("finish_reason", "stop")

    content_blocks: list[dict] = []

    # Text content
    text = message.get("content")
    if text:
        content_blocks.append({"type": "text", "text": text})

    # Tool calls
    for tc in message.get("tool_calls") or []:
        func = tc.get("function", {})
        try:
            input_data = json.loads(func.get("arguments", "{}"))
        except json.JSONDecodeError:
            input_data = {"raw": func.get("arguments", "")}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id", _gen_id("toolu_")),
            "name": func.get("name", ""),
            "input": input_data,
        })

    usage = openai_resp.get("usage", {})

    return {
        "id": _gen_id("msg_"),
        "type": "message",
        "role": "assistant",
        "content": content_blocks or [{"type": "text", "text": ""}],
        "model": model,
        "stop_reason": _FINISH_REASON_MAP.get(finish, "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


# ── Streaming: OpenAI SSE → Anthropic SSE ────────────────────────────


def format_sse(event: str, data: dict) -> str:
    """Format a single Anthropic SSE event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def make_message_start(model: str, msg_id: str) -> str:
    return format_sse("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })


def make_content_block_start(index: int, block_type: str, **kwargs) -> str:
    block: dict = {"type": block_type}
    if block_type == "text":
        block["text"] = ""
    elif block_type == "tool_use":
        block["id"] = kwargs.get("tool_id", _gen_id("toolu_"))
        block["name"] = kwargs.get("tool_name", "")
        block["input"] = {}
    return format_sse("content_block_start", {
        "type": "content_block_start",
        "index": index,
        "content_block": block,
    })


def make_text_delta(index: int, text: str) -> str:
    return format_sse("content_block_delta", {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "text_delta", "text": text},
    })


def make_tool_input_delta(index: int, json_fragment: str) -> str:
    return format_sse("content_block_delta", {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "input_json_delta", "partial_json": json_fragment},
    })


def make_content_block_stop(index: int) -> str:
    return format_sse("content_block_stop", {
        "type": "content_block_stop",
        "index": index,
    })


def make_message_delta(stop_reason: str, output_tokens: int = 0) -> str:
    return format_sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })


def make_message_stop() -> str:
    return format_sse("message_stop", {"type": "message_stop"})
