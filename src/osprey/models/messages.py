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


@dataclass
class ChatMessage:
    """A single message in a chat conversation.

    :param role: Message role — ``"system"``, ``"user"``, ``"assistant"``, or ``"tool"``
    :param content: Text content of the message (None for assistant messages with tool_calls)
    :param tool_calls: List of tool call dicts from assistant (OpenAI format)
    :param tool_call_id: ID of the tool call this message responds to (role="tool")
    :param name: Tool function name (role="tool")
    """

    role: str
    content: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    def to_dict(self) -> dict:
        """Convert to a plain dict suitable for LiteLLM.

        Omits ``content`` key when None (assistant tool-call messages).
        Omits ``tool_calls``, ``tool_call_id``, ``name`` when not set.

        :return: Message dict for LiteLLM
        """
        d: dict = {"role": self.role}
        if self.content is not None:
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
          ``cache_control: {"type": "ephemeral"}``
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
            if msg["role"] == "system" and isinstance(msg.get("content"), str):
                msg["content"] = [
                    {
                        "type": "text",
                        "text": msg["content"],
                        "cache_control": {"type": "ephemeral"},
                    }
                ]

        # 2. Find user messages and mark the second-to-last for caching
        user_indices = [i for i, msg in enumerate(result) if msg["role"] == "user"]
        if len(user_indices) >= 2:
            cache_idx = user_indices[-2]
            content = result[cache_idx]["content"]
            # Content is still a plain string for user messages
            if isinstance(content, str):
                result[cache_idx]["content"] = [
                    {
                        "type": "text",
                        "text": content,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]

        return result

    def to_single_string(self) -> str:
        """Flatten all messages into a single string for logging/fallback.

        Skips messages with None content (e.g. assistant tool-call messages).

        :return: All message contents joined with double newlines
        """
        if not self.messages:
            return ""
        return "\n\n".join(msg.content for msg in self.messages if msg.content is not None)
