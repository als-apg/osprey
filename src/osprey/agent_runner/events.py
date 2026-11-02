"""Plain event records for agent output.

The agent SDK reports a run as a stream of message objects whose content is a
list of typed blocks. :func:`translate_message` is the one place in the package
that reads those classes: it turns each message into zero or more of the records
below, in block order. Every record field holds a plain value — ``str``,
``int``, ``float``, ``bool``, ``None``, or a ``dict``/``list`` of those as the
CLI decoded them from JSON — so a caller that reads records never handles an
SDK object and never imports the SDK.

The module stays importable without ``claude_agent_sdk``; :func:`translate_message`
then returns ``[]`` for every input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# SDK imports — keep module importable even when SDK is absent.
try:
    from claude_agent_sdk import (
        AssistantMessage,
        ResultMessage,
        StreamEvent,
        SystemMessage,
        TextBlock,
        ThinkingBlock,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
    )

    _HAS_SDK = True
except ImportError:
    _HAS_SDK = False


@dataclass(frozen=True, slots=True, kw_only=True)
class TextEvent:
    """Assistant text."""

    text: str
    parent_tool_use_id: str | None


@dataclass(frozen=True, slots=True, kw_only=True)
class ThinkingEvent:
    """Assistant reasoning text."""

    text: str
    parent_tool_use_id: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolUseEvent:
    """A tool call the agent issued."""

    tool_use_id: str
    name: str
    input: dict[str, Any]
    parent_tool_use_id: str | None


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolResultEvent:
    """The result returned for the tool call named by ``tool_use_id``."""

    tool_use_id: str
    content: str | list[dict[str, Any]] | None
    is_error: bool
    parent_tool_use_id: str | None

    @property
    def text(self) -> str | None:
        """``content`` as text: a string as is, a list's text items joined by newlines
        (the list stringified when it has none), ``None`` when there is no content."""
        content = self.content
        if content is None:
            return None
        if isinstance(content, str):
            return content
        texts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "\n".join(texts) if texts else str(content)


@dataclass(frozen=True, slots=True, kw_only=True)
class ApiErrorEvent:
    """The model API refused or failed the assistant message that follows it."""

    error: str


@dataclass(frozen=True, slots=True, kw_only=True)
class SystemEvent:
    """A system notice from the CLI (initialisation, task progress, …)."""

    subtype: str
    data: dict[str, Any]


@dataclass(frozen=True, slots=True, kw_only=True)
class ResultEvent:
    """The end of a response: outcome, turn count, timing and cost."""

    subtype: str
    is_error: bool
    num_turns: int
    duration_ms: int
    session_id: str
    total_cost_usd: float | None
    usage: dict[str, Any] | None
    result: str | None
    api_error_status: int | None


@dataclass(frozen=True, slots=True, kw_only=True)
class DeltaEvent:
    """A piece of assistant text or reasoning as it streams.

    Produced only for a run opened with ``include_partial_messages``. The whole
    block still arrives afterwards as a :class:`TextEvent` or
    :class:`ThinkingEvent`, which is the authoritative copy.

    A ``thinking`` delta with empty ``text`` marks a reasoning block opening. A
    model that hides its reasoning streams only empty pieces, so the block's
    start is the one sign, while it thinks, that it is thinking.
    """

    kind: str  # "text" | "thinking"
    text: str
    parent_tool_use_id: str | None


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolStartEvent:
    """The agent began writing a tool call; its input is still streaming.

    Produced only for a run opened with ``include_partial_messages``, from the
    call's block start. The whole call arrives later as a :class:`ToolUseEvent`
    with the same ``tool_use_id``; until then this is the one sign of what the
    agent is about to do.
    """

    name: str
    tool_use_id: str
    parent_tool_use_id: str | None


AgentEvent = (
    TextEvent
    | ThinkingEvent
    | ToolUseEvent
    | ToolResultEvent
    | ApiErrorEvent
    | SystemEvent
    | ResultEvent
    | DeltaEvent
    | ToolStartEvent
)

#: Stream-event delta types the chat shows as they arrive, and the record kind
#: and payload field each maps to.
_DELTA_FIELDS = {"text_delta": ("text", "text"), "thinking_delta": ("thinking", "thinking")}

#: Content blocks whose start is announced as an empty ``thinking`` delta.
_REASONING_BLOCKS = frozenset({"thinking", "redacted_thinking"})

#: Content blocks whose start is announced as a :class:`ToolStartEvent`.
_TOOL_BLOCKS = frozenset({"tool_use", "server_tool_use"})


def _tool_block_event(block: object, parent_tool_use_id: str | None) -> AgentEvent | None:
    """The record for a tool-use or tool-result block, ``None`` for any other block."""
    if isinstance(block, ToolUseBlock):
        return ToolUseEvent(
            tool_use_id=block.id,
            name=block.name,
            input=dict(block.input),
            parent_tool_use_id=parent_tool_use_id,
        )
    if isinstance(block, ToolResultBlock):
        return ToolResultEvent(
            tool_use_id=block.tool_use_id,
            content=block.content,
            is_error=bool(block.is_error),
            parent_tool_use_id=parent_tool_use_id,
        )
    return None


def translate_message(message: object) -> list[AgentEvent]:
    """The event records one agent message carries, in block order.

    An assistant message yields an :class:`ApiErrorEvent` first when it carries
    an API error, then one record per text, thinking, tool-use and tool-result
    block. A user message with list content yields its tool-use and tool-result
    blocks; its text and thinking blocks, and a user message whose content is a
    plain string, yield nothing. Every system message, subclasses included,
    yields one :class:`SystemEvent`; the result message yields one
    :class:`ResultEvent`. A partial stream event yields one :class:`DeltaEvent`
    when it is a text or thinking delta with text, or the start of a reasoning
    block, and one :class:`ToolStartEvent` at the start of a tool call. Anything
    else — other stream events, rate-limit notices, an unknown type — yields
    nothing.

    Args:
        message: One message from the SDK's response stream.

    Returns:
        The records, empty when the message carries none or the SDK is absent.
    """
    if not _HAS_SDK:
        return []

    events: list[AgentEvent] = []
    if isinstance(message, AssistantMessage):
        parent = message.parent_tool_use_id
        if message.error is not None:
            events.append(ApiErrorEvent(error=str(message.error)))
        for block in message.content:
            if isinstance(block, TextBlock):
                events.append(TextEvent(text=block.text, parent_tool_use_id=parent))
            elif isinstance(block, ThinkingBlock):
                events.append(ThinkingEvent(text=block.thinking, parent_tool_use_id=parent))
            else:
                tool_event = _tool_block_event(block, parent)
                if tool_event is not None:
                    events.append(tool_event)
    elif isinstance(message, UserMessage):
        if isinstance(message.content, list):
            for block in message.content:
                tool_event = _tool_block_event(block, message.parent_tool_use_id)
                if tool_event is not None:
                    events.append(tool_event)
    elif isinstance(message, SystemMessage):
        events.append(SystemEvent(subtype=message.subtype, data=dict(message.data)))
    elif isinstance(message, ResultMessage):
        events.append(
            ResultEvent(
                subtype=message.subtype,
                is_error=message.is_error,
                num_turns=message.num_turns,
                duration_ms=message.duration_ms,
                session_id=message.session_id,
                total_cost_usd=message.total_cost_usd,
                usage=message.usage,
                result=message.result,
                api_error_status=message.api_error_status,
            )
        )
    elif isinstance(message, StreamEvent):
        raw = message.event if isinstance(message.event, dict) else {}
        parent = message.parent_tool_use_id
        opened = raw.get("content_block") if raw.get("type") == "content_block_start" else None
        if isinstance(opened, dict) and opened.get("type") in _REASONING_BLOCKS:
            events.append(DeltaEvent(kind="thinking", text="", parent_tool_use_id=parent))
        elif (
            isinstance(opened, dict)
            and opened.get("type") in _TOOL_BLOCKS
            and isinstance(opened.get("name"), str)
        ):
            events.append(
                ToolStartEvent(
                    name=opened["name"],
                    tool_use_id=str(opened.get("id", "")),
                    parent_tool_use_id=parent,
                )
            )
        delta = raw.get("delta") if raw.get("type") == "content_block_delta" else None
        if isinstance(delta, dict):
            mapping = _DELTA_FIELDS.get(delta.get("type", ""))
            piece = delta.get(mapping[1]) if mapping is not None else None
            if mapping is not None and isinstance(piece, str) and piece:
                events.append(DeltaEvent(kind=mapping[0], text=piece, parent_tool_use_id=parent))
    return events
