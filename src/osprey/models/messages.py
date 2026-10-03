"""Chat message and completion request types for structured LLM conversations.

Provides dataclasses for building multi-turn conversations that enable
provider-side prompt caching. ``ChatCompletionRequest`` converts to
LiteLLM message format and optionally adds Anthropic ``cache_control``
markers for 90% cost reduction on cached tokens.

.. seealso::
   :func:`osprey.models.completion.get_chat_completion` : Accepts ``chat_request``
   :func:`osprey.models.providers.litellm_adapter.execute_litellm_completion` : Consumes messages
"""

from __future__ import annotations

from dataclasses import dataclass, field

_CACHE_CONTROL = {"type": "ephemeral"}


def parse_data_url(url: str) -> tuple[str, int, str]:
    """Split a base64 ``data:`` URL into its MIME type, decoded size and payload.

    The canonical image content part is LiteLLM's
    ``{"type": "image_url", "image_url": {"url": "data:<mime>;base64,<b64>"}}``;
    this reads the ``url`` of such a part without decoding the payload.

    :param url: A ``data:<mime>;base64,<b64>`` URL
    :return: ``(mime, n_bytes, b64)`` where ``n_bytes`` is the decoded length
    :raises ValueError: If ``url`` is not a base64 data URL with a MIME type
    """
    if not url.startswith("data:") or "," not in url:
        raise ValueError("not a data URL")
    header, b64 = url[len("data:") :].split(",", 1)
    mime, sep, encoding = header.partition(";")
    if not sep or encoding != "base64" or not mime:
        raise ValueError("not a base64 data URL with a MIME type")
    padding = len(b64) - len(b64.rstrip("="))
    n_bytes = len(b64) * 3 // 4 - padding
    return mime, n_bytes, b64


def _image_url(part: dict) -> str | None:
    image_url = part.get("image_url")
    if isinstance(image_url, dict):
        return image_url.get("url")
    if isinstance(image_url, str):
        return image_url
    return None


def _render_part(part: dict) -> str:
    """Render one content part for a log line; image payloads become a size note."""
    part_type = part.get("type")
    if part_type == "text":
        return str(part.get("text", ""))
    if part_type == "image_url":
        url = _image_url(part)
        try:
            mime, n_bytes, _ = parse_data_url(url or "")
        except ValueError:
            return "[image]"
        return f"[image {mime}, {n_bytes} bytes]"
    return f"[{part_type}]"


def _with_cache_marker(content: list[dict]) -> list[dict]:
    """Return a copy of *content* with a cache marker on its last text part."""
    parts = [dict(p) for p in content]
    for part in reversed(parts):
        if part.get("type") == "text":
            part["cache_control"] = dict(_CACHE_CONTROL)
            break
    return parts


@dataclass
class ChatMessage:
    """A single message in a chat conversation.

    :param role: Message role — ``"system"``, ``"user"``, ``"assistant"``, or ``"tool"``
    :param content: Text content, a list of LiteLLM content parts (``text`` and
        ``image_url``), or None for assistant messages with tool_calls
    :param tool_calls: List of tool call dicts from assistant (OpenAI format)
    :param tool_call_id: ID of the tool call this message responds to (role="tool")
    :param name: Tool function name (role="tool")
    """

    role: str
    content: str | list[dict] | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    def to_dict(self) -> dict:
        """Convert to a plain dict suitable for LiteLLM.

        Omits ``content`` key when None (assistant tool-call messages).
        Omits ``tool_calls``, ``tool_call_id``, ``name`` when not set.
        List content is copied part by part, so callers that edit the returned
        dict in place never reach this message.

        :return: Message dict for LiteLLM
        """
        d: dict = {"role": self.role}
        if isinstance(self.content, list):
            d["content"] = [dict(p) for p in self.content]
        elif self.content is not None:
            d["content"] = self.content
        if self.tool_calls is not None:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            d["name"] = self.name
        return d


@dataclass
class ChatCompletionRequest:
    """A structured multi-turn conversation for LLM completion.

    Wraps a list of :class:`ChatMessage` objects and provides conversion
    to LiteLLM message format with optional Anthropic prompt-cache markers.

    :param messages: Ordered list of chat messages
    """

    messages: list[ChatMessage] = field(default_factory=list)

    def to_litellm_messages(self, provider: str | None = None) -> list[dict]:
        """Convert to LiteLLM ``messages`` format.

        For Anthropic, adds ``cache_control`` markers to enable prompt caching:

        * **System message**: content wrapped as a content block with
          ``cache_control: {"type": "ephemeral"}`` (list content: the marker goes
          on its last text part)
        * **Second-to-last user message** (when 2+ user msgs): gets
          ``cache_control`` (stable conversation prefix)
        * **Last user message**: no cache marker (changes every turn)

        For all other providers the output is plain ``{"role", "content"}`` dicts.

        Tool messages (role="tool") pass through unchanged — cache markers
        are only applied to system and user messages.

        :param provider: Provider name (``"anthropic"`` triggers cache markers)
        :return: List of message dicts for ``litellm.completion()``
        """
        if not self.messages:
            return []

        result = [msg.to_dict() for msg in self.messages]

        if provider != "anthropic":
            return result

        # --- Anthropic cache markers ---
        # 1. System message gets cache_control on its content block
        for msg in result:
            if msg["role"] == "system":
                content = msg.get("content")
                if isinstance(content, str):
                    msg["content"] = _with_cache_marker([{"type": "text", "text": content}])
                elif isinstance(content, list):
                    msg["content"] = _with_cache_marker(content)

        # 2. Find user messages and mark the second-to-last for caching
        user_indices = [i for i, msg in enumerate(result) if msg["role"] == "user"]
        if len(user_indices) >= 2:
            cache_idx = user_indices[-2]
            content = result[cache_idx].get("content")
            if isinstance(content, str):
                result[cache_idx]["content"] = _with_cache_marker(
                    [{"type": "text", "text": content}]
                )
            elif isinstance(content, list):
                result[cache_idx]["content"] = _with_cache_marker(content)

        return result

    def to_single_string(self) -> str:
        """Flatten all messages into a single string for logging/fallback.

        Skips messages with None content (e.g. assistant tool-call messages).
        List content renders its text parts joined by newlines and each image
        part as ``[image <mime>, <n> bytes]`` — never the base64 payload.

        :return: All message contents joined with double newlines
        """
        if not self.messages:
            return ""
        rendered: list[str] = []
        for msg in self.messages:
            if isinstance(msg.content, list):
                rendered.append("\n".join(_render_part(p) for p in msg.content))
            elif msg.content is not None:
                rendered.append(msg.content)
        return "\n\n".join(rendered)
