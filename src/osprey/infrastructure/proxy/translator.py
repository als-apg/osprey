"""Pure translation functions between Anthropic Messages API and OpenAI Chat Completions.

All functions are stateless and perform no I/O — they transform dicts.
"""

from __future__ import annotations

import json
import random
import string
from dataclasses import dataclass, field


def _gen_id(prefix: str = "msg_") -> str:
    chars = string.ascii_lowercase + string.digits
    return prefix + "".join(random.choices(chars, k=24))


# ── Request: Anthropic → OpenAI ──────────────────────────────────────

#: The model reads this in place of an image the route does not carry.
_IMAGE_NOT_CARRIED = "[image not sent: this provider's route does not carry images]"

#: The model reads this in place of an image given by a source the proxy cannot send.
_IMAGE_SOURCE_NOT_CARRIED = "[image not sent: the proxy carries base64 and URL images only]"

#: The model reads this in a tool message in place of an image that rides the
#: user message following the tool messages.
_IMAGE_IN_NEXT_MESSAGE = "[image: sent in the next user message]"


def _not_carried(kind: str) -> str:
    """The note the model reads in place of a *kind* of content the route does not carry."""
    return f"[{kind} not sent: this provider's route does not carry it]"


@dataclass(frozen=True)
class TranslatedRequest:
    """An OpenAI Chat Completions request and what translating it left out.

    Attributes:
        body: The OpenAI request body.
        dropped: The kinds of content this request lost; the proxy reads it to
            name what it left out.
        images_sent: The ``image_url`` parts in ``body``; the proxy reads it to
            name the images in an upstream refusal.
    """

    body: dict
    dropped: frozenset[str] = frozenset()
    images_sent: int = 0


@dataclass
class _Notes:
    """What the converters record while translating one request."""

    supports_images: bool
    dropped: set[str] = field(default_factory=set)
    images_sent: int = 0


def anthropic_to_openai_request(
    body: dict,
    *,
    max_tokens_param: str = "max_tokens",
    accepts_temperature: bool = True,
    supports_images: bool = False,
) -> TranslatedRequest:
    """Convert an Anthropic Messages API request body to OpenAI Chat Completions.

    Args:
        body: The Anthropic Messages request.
        max_tokens_param: The upstream parameter that carries the output-token cap.
        accepts_temperature: Whether the upstream takes a caller-chosen temperature;
            when False the request carries none.
        supports_images: Whether the upstream route takes ``image_url`` parts;
            when False every image is replaced by a note.

    Returns:
        The OpenAI request body, with the kinds of content it left out and the
        number of images it carries.
    """
    notes = _Notes(supports_images=supports_images)
    messages = _convert_messages(body.get("messages", []), body.get("system"), notes)
    tools = _convert_tools_to_openai(body.get("tools"))

    openai_body: dict = {
        "model": body.get("model", ""),
        "messages": messages,
        "stream": body.get("stream", False),
    }

    if body.get("max_tokens"):
        openai_body[max_tokens_param] = body["max_tokens"]
    if body.get("temperature") is not None:
        if accepts_temperature:
            openai_body["temperature"] = body["temperature"]
        else:
            notes.dropped.add("temperature")
    thinking = body.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") != "disabled":
        notes.dropped.add("thinking")
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

    return TranslatedRequest(openai_body, frozenset(notes.dropped), notes.images_sent)


def _system_text(content: str | list | None) -> str:
    """Flatten an Anthropic system value (str or content-block list) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"]
        return "\n".join(text_parts)
    return ""


def _convert_messages(
    anthropic_messages: list[dict],
    system: str | list | None,
    notes: _Notes,
) -> list[dict]:
    """Convert Anthropic message array to OpenAI message array."""
    openai_messages: list[dict] = []

    # Top-level system prompt
    if system:
        text = _system_text(system)
        if text:
            openai_messages.append({"role": "system", "content": text})

    for msg in anthropic_messages:
        role = msg.get("role", "user")
        content = msg.get("content")

        if role == "user":
            openai_messages.extend(_convert_user_message(content, notes))
        elif role == "assistant":
            openai_messages.extend(_convert_assistant_message(content, notes))
        elif role == "system":
            # The Anthropic API carries the system prompt in the top-level
            # ``system`` field, but some clients put a ``role: system`` entry in
            # the array. Hoist it into an OpenAI system message rather than
            # dropping it on the floor and silently losing the instruction.
            text = _system_text(content)
            if text:
                openai_messages.append({"role": "system", "content": text})

    return openai_messages


def _image_part(block: dict, notes: _Notes) -> dict | str:
    """An Anthropic image block as an OpenAI ``image_url`` part, or the note in its place."""
    if not notes.supports_images:
        notes.dropped.add("image")
        return _IMAGE_NOT_CARRIED
    source = block.get("source")
    source = source if isinstance(source, dict) else {}
    url = None
    if source.get("type") == "base64" and source.get("data"):
        url = f"data:{source.get('media_type') or 'image/png'};base64,{source['data']}"
    elif source.get("type") == "url" and source.get("url"):
        url = source["url"]
    if url is None:
        notes.dropped.add("image reference")
        return _IMAGE_SOURCE_NOT_CARRIED
    notes.images_sent += 1
    return {"type": "image_url", "image_url": {"url": url}}


def _tool_result_text(content, images: list[dict], notes: _Notes) -> str:
    """A tool result's content as tool-message text; carried images go to *images*."""
    if not isinstance(content, list):
        return str(content)
    lines = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            lines.append(block.get("text", ""))
        elif btype == "image":
            part = _image_part(block, notes)
            if isinstance(part, dict):
                images.append(part)
                lines.append(_IMAGE_IN_NEXT_MESSAGE)
            else:
                lines.append(part)
        else:
            notes.dropped.add(str(btype))
            lines.append(_not_carried(str(btype)))
    return "\n".join(lines)


def _convert_user_message(content, notes: _Notes) -> list[dict]:
    """Convert a single Anthropic user message to OpenAI format."""
    if isinstance(content, str):
        return [{"role": "user", "content": content}]

    if isinstance(content, list):
        # The turn's own parts, in block order: text strings and image_url parts.
        parts: list[str | dict] = []
        tool_messages: list[dict] = []
        tool_images: list[tuple[str, list[dict]]] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                btype = block.get("type")
                if btype == "text":
                    parts.append(block.get("text", ""))
                elif btype == "image":
                    parts.append(_image_part(block, notes))
                elif btype == "tool_result":
                    tool_use_id = block.get("tool_use_id", "")
                    images: list[dict] = []
                    text = _tool_result_text(block.get("content", ""), images, notes)
                    tool_messages.append(
                        {"role": "tool", "tool_call_id": tool_use_id, "content": text}
                    )
                    if images:
                        tool_images.append((tool_use_id, images))
                else:
                    notes.dropped.add(str(btype))
                    parts.append(_not_carried(str(btype)))

        # Chat Completions tool messages carry text only, and they must directly
        # follow the assistant turn that called them, so a tool's images ride
        # the one user message emitted after every tool message.
        messages = list(tool_messages)
        if tool_images or any(isinstance(p, dict) for p in parts):
            content_parts: list[dict] = []
            for tool_use_id, images in tool_images:
                content_parts.append(
                    {"type": "text", "text": f"Images returned by tool call {tool_use_id}:"}
                )
                content_parts.extend(images)
            content_parts.extend(
                p if isinstance(p, dict) else {"type": "text", "text": p} for p in parts
            )
            messages.append({"role": "user", "content": content_parts})
        elif parts:
            messages.append({"role": "user", "content": "\n".join(str(p) for p in parts)})

        return messages if messages else [{"role": "user", "content": ""}]

    return [{"role": "user", "content": str(content)}]


def _convert_assistant_message(content, notes: _Notes) -> list[dict]:
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
                    tool_calls.append(
                        {
                            "id": block.get("id", _gen_id("call_")),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        }
                    )
                elif btype in ("thinking", "redacted_thinking"):
                    notes.dropped.add("thinking")
                else:
                    # The model's own history: what the route cannot carry is
                    # named as dropped and leaves no text in the assistant turn.
                    notes.dropped.add(str(btype))

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
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            }
        )
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
        content_blocks.append(
            {
                "type": "tool_use",
                "id": tc.get("id", _gen_id("toolu_")),
                "name": func.get("name", ""),
                "input": input_data,
            }
        )

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
    return format_sse(
        "message_start",
        {
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
        },
    )


def make_content_block_start(index: int, block_type: str, **kwargs) -> str:
    block: dict = {"type": block_type}
    if block_type == "text":
        block["text"] = ""
    elif block_type == "tool_use":
        block["id"] = kwargs.get("tool_id", _gen_id("toolu_"))
        block["name"] = kwargs.get("tool_name", "")
        block["input"] = {}
    return format_sse(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": index,
            "content_block": block,
        },
    )


def make_text_delta(index: int, text: str) -> str:
    return format_sse(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "text_delta", "text": text},
        },
    )


def make_tool_input_delta(index: int, json_fragment: str) -> str:
    return format_sse(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": json_fragment},
        },
    )


def make_content_block_stop(index: int) -> str:
    return format_sse(
        "content_block_stop",
        {
            "type": "content_block_stop",
            "index": index,
        },
    )


def make_message_delta(stop_reason: str, output_tokens: int = 0) -> str:
    return format_sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        },
    )


def make_message_stop() -> str:
    return format_sse("message_stop", {"type": "message_stop"})
