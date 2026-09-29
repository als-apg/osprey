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


AgentEvent = (
    TextEvent
    | ThinkingEvent
    | ToolUseEvent
    | ToolResultEvent
    | ApiErrorEvent
    | SystemEvent
    | ResultEvent
)


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
    :class:`ResultEvent`. Anything else — partial stream events, rate-limit
    notices, an unknown type — yields nothing.

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
                events.append(ThinkingEvent(text=block.thinking))
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
    return events
