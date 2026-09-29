"""Unit tests for osprey.agent_runner.events (agent message → plain event records).

Real SDK message objects are built (the SDK is a dev dependency) and run through
``translate_message``; every assertion is on the plain records that come out.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from typing import Any

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    RateLimitEvent,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TaskStartedMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from claude_agent_sdk.types import RateLimitInfo

from osprey.agent_runner.events import (
    ApiErrorEvent,
    ResultEvent,
    SystemEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolUseEvent,
    translate_message,
)


def _result_message(**overrides: Any) -> ResultMessage:
    fields: dict[str, Any] = {
        "subtype": "success",
        "duration_ms": 120,
        "duration_api_ms": 100,
        "is_error": False,
        "num_turns": 3,
        "session_id": "abc-123",
        "total_cost_usd": 0.0042,
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "result": "done",
        "api_error_status": None,
        "stop_reason": "end_turn",
        "model_usage": {"m": {"inputTokens": 10}},
        "permission_denials": [],
    }
    fields.update(overrides)
    return ResultMessage(**fields)


def test_text_and_thinking_blocks_become_text_and_thinking_events() -> None:
    message = AssistantMessage(
        content=[
            ThinkingBlock(thinking="considering", signature="sig"),
            TextBlock(text="hello"),
        ],
        model="m",
        parent_tool_use_id="parent-1",
    )

    assert translate_message(message) == [
        ThinkingEvent(text="considering"),
        TextEvent(text="hello", parent_tool_use_id="parent-1"),
    ]


def test_tool_use_carries_its_id_name_input_and_parent() -> None:
    message = AssistantMessage(
        content=[ToolUseBlock(id="tu-1", name="mcp__controls__channel_read", input={"ch": "X"})],
        model="m",
        parent_tool_use_id="parent-2",
    )

    assert translate_message(message) == [
        ToolUseEvent(
            tool_use_id="tu-1",
            name="mcp__controls__channel_read",
            input={"ch": "X"},
            parent_tool_use_id="parent-2",
        )
    ]


def test_tool_results_are_read_from_user_and_assistant_messages() -> None:
    from_user = UserMessage(
        content=[ToolResultBlock(tool_use_id="tu-1", content="42", is_error=None)],
        parent_tool_use_id="p-u",
    )
    from_assistant = AssistantMessage(
        content=[
            ToolResultBlock(
                tool_use_id="tu-2", content=[{"type": "text", "text": "off"}], is_error=True
            )
        ],
        model="m",
    )

    assert translate_message(from_user) == [
        ToolResultEvent(tool_use_id="tu-1", content="42", is_error=False, parent_tool_use_id="p-u")
    ]
    assert translate_message(from_assistant) == [
        ToolResultEvent(
            tool_use_id="tu-2",
            content=[{"type": "text", "text": "off"}],
            is_error=True,
            parent_tool_use_id=None,
        )
    ]


def test_tool_use_in_a_user_message_is_read() -> None:
    message = UserMessage(content=[ToolUseBlock(id="tu-9", name="Read", input={"path": "a"})])

    assert translate_message(message) == [
        ToolUseEvent(tool_use_id="tu-9", name="Read", input={"path": "a"}, parent_tool_use_id=None)
    ]


class TestToolResultText:
    def _event(self, content: Any) -> ToolResultEvent:
        return ToolResultEvent(
            tool_use_id="t", content=content, is_error=False, parent_tool_use_id=None
        )

    def test_string_content_is_returned_as_is(self) -> None:
        assert self._event("plain").text == "plain"

    def test_list_content_joins_its_text_items(self) -> None:
        content = [
            {"type": "text", "text": "a"},
            {"type": "image", "source": {}},
            {"type": "text", "text": "b"},
        ]
        assert self._event(content).text == "a\nb"

    def test_list_without_text_items_is_stringified(self) -> None:
        content = [{"type": "image", "source": {}}]
        assert self._event(content).text == str(content)

    def test_absent_content_is_none(self) -> None:
        assert self._event(None).text is None


def test_an_api_error_precedes_the_message_blocks() -> None:
    message = AssistantMessage(content=[TextBlock(text="partial")], model="m", error="rate_limit")

    events = translate_message(message)

    assert events == [
        ApiErrorEvent(error="rate_limit"),
        TextEvent(text="partial", parent_tool_use_id=None),
    ]
    assert isinstance(events[0], ApiErrorEvent)
    assert events[0].error == "rate_limit"


def test_a_result_message_becomes_a_result_event_with_its_plain_fields() -> None:
    message = _result_message(is_error=True, api_error_status=529)

    assert translate_message(message) == [
        ResultEvent(
            subtype="success",
            is_error=True,
            num_turns=3,
            duration_ms=120,
            session_id="abc-123",
            total_cost_usd=0.0042,
            usage={"input_tokens": 10, "output_tokens": 5},
            result="done",
            api_error_status=529,
        )
    ]


def test_system_messages_and_their_subclasses_keep_subtype_and_data() -> None:
    plain = SystemMessage(subtype="init", data={"tools": ["Read"], "model": "m"})
    task = TaskStartedMessage(
        subtype="task_started",
        data={"task_id": "t1", "description": "d"},
        task_id="t1",
        description="d",
        uuid="u",
        session_id="s",
    )

    [plain_event] = translate_message(plain)
    [task_event] = translate_message(task)

    assert plain_event == SystemEvent(subtype="init", data={"tools": ["Read"], "model": "m"})
    assert plain_event.data is not plain.data
    assert task_event == SystemEvent(
        subtype="task_started", data={"task_id": "t1", "description": "d"}
    )


def test_stream_rate_limit_and_plain_user_text_translate_to_nothing() -> None:
    stream = StreamEvent(uuid="u", session_id="s", event={"type": "content_block_delta"})
    rate = RateLimitEvent(rate_limit_info=RateLimitInfo(status="allowed"), uuid="u", session_id="s")
    user_text = UserMessage(content="just text")
    user_blocks = UserMessage(
        content=[TextBlock(text="t"), ThinkingBlock(thinking="x", signature="s")]
    )

    for message in (stream, rate, user_text, user_blocks):
        assert translate_message(message) == []


def _walk(value: Any) -> list[Any]:
    """Every value reachable from *value* through dataclass fields, dicts and lists."""
    out = [value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        for f in dataclasses.fields(value):
            out.extend(_walk(getattr(value, f.name)))
    elif isinstance(value, dict):
        for k, v in value.items():
            out.extend(_walk(k))
            out.extend(_walk(v))
    elif isinstance(value, (list, tuple)):
        for v in value:
            out.extend(_walk(v))
    return out


def _assert_no_sdk_object(root: Any) -> None:
    for value in _walk(root):
        if value is root:
            continue
        assert not type(value).__module__.startswith("claude_agent_sdk"), (
            f"{type(value).__module__}.{type(value).__name__} leaked into an event record"
        )


def test_no_event_holds_an_agent_sdk_object() -> None:
    messages = [
        AssistantMessage(
            content=[
                TextBlock(text="t"),
                ThinkingBlock(thinking="th", signature="s"),
                ToolUseBlock(id="a", name="n", input={"k": [1, {"x": "y"}]}),
                ToolResultBlock(tool_use_id="a", content=[{"type": "text", "text": "r"}]),
            ],
            model="m",
            error="server_error",
            parent_tool_use_id="p",
        ),
        UserMessage(
            content=[
                ToolResultBlock(tool_use_id="a", content="r", is_error=True),
                ToolUseBlock(id="b", name="n", input={}),
            ]
        ),
        SystemMessage(subtype="init", data={"mcp_servers": [{"name": "x"}]}),
        TaskStartedMessage(
            subtype="task_started",
            data={"task_id": "t"},
            task_id="t",
            description="d",
            uuid="u",
            session_id="s",
        ),
        _result_message(),
    ]

    events = [event for message in messages for event in translate_message(message)]

    assert {type(e) for e in events} == {
        TextEvent,
        ThinkingEvent,
        ToolUseEvent,
        ToolResultEvent,
        ApiErrorEvent,
        SystemEvent,
        ResultEvent,
    }
    for event in events:
        _assert_no_sdk_object(event)


def test_events_are_immutable() -> None:
    event = TextEvent(text="t", parent_tool_use_id=None)

    with pytest.raises(dataclasses.FrozenInstanceError):
        event.text = "changed"  # type: ignore[misc]


def test_the_events_module_imports_without_the_agent_sdk() -> None:
    script = (
        "import sys\n"
        "sys.modules['claude_agent_sdk'] = None\n"
        "from osprey.agent_runner.events import TextEvent, translate_message\n"
        "TextEvent(text='t', parent_tool_use_id=None)\n"
        "assert translate_message(object()) == []\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )

    assert completed.returncode == 0, completed.stderr
