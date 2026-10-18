"""A headless run of the pinned Claude Code can call the task-list tools.

The CLI offers ``TaskCreate``, ``TaskGet``, ``TaskList`` and ``TaskUpdate`` by
default only to the models it recognises by name. OSPREY turns them on for
every process it starts, and the dispatch allow-list passes them through. That
they are *listed* is recorded in the tool inventory; this module proves they
*work*: the real pinned binary, started through the options every SDK run is
built from, creates a task and then lists it — under the CLI's own default
model, which is not offered the tools unless they are turned on, and under a
model name of the kind a gateway hands out.

No model is involved. The run talks to a local stand-in for the Messages API
that answers with a fixed script — call ``TaskCreate``, call ``TaskList``, say
so — and the tools themselves execute inside the real CLI.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from osprey.agent_runner.clean_env import build_base_child_env
from osprey.agent_runner.primitives import build_agent_options
from osprey.cli.templates.scaffolding import _DEFAULT_CLAUDE_CLI_VERSION

pytestmark = [pytest.mark.slow, pytest.mark.usefixtures("real_agent_cli")]

#: A model name of the kind a gateway hands out: nothing the CLI recognises.
_GATEWAY_MODEL = "facility-gateway/large-model"

_SUBJECT = "Check the injection efficiency"

#: Upper bound on the whole scripted run; it normally takes a few seconds.
_RUN_TIMEOUT_S = 120.0

#: The scripted assistant turns, chosen by how many tool results the request carries.
_SCRIPT: list[dict[str, Any]] = [
    {
        "type": "tool_use",
        "id": "toolu_create",
        "name": "TaskCreate",
        "input": {"subject": _SUBJECT, "description": "Read it back from the archiver."},
    },
    {"type": "tool_use", "id": "toolu_list", "name": "TaskList", "input": {}},
    {"type": "text", "text": "The task is on the list."},
]


def _tool_results(body: dict[str, Any]) -> int:
    count = 0
    for message in body.get("messages", []):
        content = message.get("content")
        if isinstance(content, list):
            count += sum(1 for block in content if block.get("type") == "tool_result")
    return count


def _events(block: dict[str, Any], model: str) -> list[tuple[str, dict[str, Any]]]:
    """The server-sent events of one assistant message holding *block*."""
    is_tool = block["type"] == "tool_use"
    start: dict[str, Any] = (
        {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}
        if is_tool
        else {"type": "text", "text": ""}
    )
    delta: dict[str, Any] = (
        {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
        if is_tool
        else {"type": "text_delta", "text": block["text"]}
    )
    message = {
        "id": "msg_scripted",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 1},
    }
    return [
        ("message_start", {"type": "message_start", "message": message}),
        (
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": start},
        ),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": delta}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": "tool_use" if is_tool else "end_turn",
                    "stop_sequence": None,
                },
                "usage": {"output_tokens": 10},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]


class _ScriptedMessagesApi(BaseHTTPRequestHandler):
    """Answers every Messages API request from :data:`_SCRIPT`."""

    protocol_version = "HTTP/1.1"
    #: The tool names each agent-turn request offered, in arrival order.
    offered: list[set[str]] = []

    def log_message(self, *_args: Any) -> None:
        pass

    def _send(self, status: int, content_type: str, payload: bytes) -> None:
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        self._send(404, "application/json", b"{}")

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        if "count_tokens" in self.path:
            self._send(200, "application/json", b'{"input_tokens": 10}')
            return
        tools = {tool.get("name", "") for tool in body.get("tools") or []}
        if tools:
            type(self).offered.append(tools)
            block = _SCRIPT[min(_tool_results(body), len(_SCRIPT) - 1)]
        else:
            # A side request the CLI makes for itself (a title, a summary).
            block = {"type": "text", "text": "ok"}
        model = body.get("model", _GATEWAY_MODEL)
        if body.get("stream"):
            stream = "".join(
                f"event: {name}\ndata: {json.dumps(data)}\n\n"
                for name, data in _events(block, model)
            )
            self._send(200, "text/event-stream", stream.encode())
            return
        content = (
            {k: block[k] for k in ("type", "id", "name", "input")}
            if block["type"] == "tool_use"
            else block
        )
        message = {
            "id": "msg_scripted",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [content],
            "stop_reason": "tool_use" if block["type"] == "tool_use" else "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 10},
        }
        self._send(200, "application/json", json.dumps(message).encode())


@pytest.fixture
def messages_api():
    _ScriptedMessagesApi.offered = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ScriptedMessagesApi)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(10)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A render pinned to the build under test, with no MCP server."""
    root = tmp_path / "project"
    (root / ".claude").mkdir(parents=True)
    (root / ".mcp.json").write_text('{"mcpServers": {}}', encoding="utf-8")
    (root / "config.yml").write_text(
        f"claude_code:\n  cli_version: '{_DEFAULT_CLAUDE_CLI_VERSION}'\n", encoding="utf-8"
    )
    return root


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content
        )
    return "" if content is None else str(content)


@pytest.mark.parametrize("model", [None, _GATEWAY_MODEL], ids=["cli-default-model", "gateway"])
def test_a_headless_run_of_the_pinned_build_creates_and_lists_a_task(
    project: Path, messages_api: str, tmp_path: Path, model: str | None
) -> None:
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeSDKClient,
        ResultMessage,
        SystemMessage,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
    )

    from osprey.agent_runner.errors import AgentCliUnavailableError

    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    # The probe runs the pinned build under the environment a headless run gets:
    # the product's base child env, with only the isolation keys laid over it.
    env = {
        **build_base_child_env(),
        "HOME": str(home),
        "CLAUDE_CONFIG_DIR": str(home / ".claude"),
        "ANTHROPIC_API_KEY": "scripted",
        "ANTHROPIC_BASE_URL": messages_api,
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    }
    try:
        options = build_agent_options(
            project, disallowed_tools=[], env=env, model=model, max_turns=6
        )
    except AgentCliUnavailableError as exc:
        pytest.fail(f"this check needs the pinned Claude Code and must not skip:\n{exc}")

    listed: set[str] = set()
    version: list[str] = []
    calls: dict[str, str] = {}
    results: dict[str, tuple[bool, str]] = {}

    async def run() -> None:
        async with ClaudeSDKClient(options=options) as client:
            await client.query("Track the work with your task list.")
            async for message in client.receive_response():
                if isinstance(message, SystemMessage) and message.subtype == "init":
                    listed.update(message.data.get("tools", []))
                    version.append(str(message.data.get("claude_code_version")))
                elif isinstance(message, AssistantMessage | UserMessage):
                    blocks = message.content if isinstance(message.content, list) else []
                    for block in blocks:
                        if isinstance(block, ToolUseBlock):
                            calls[block.id] = block.name
                        elif isinstance(block, ToolResultBlock):
                            results[calls.get(block.tool_use_id, block.tool_use_id)] = (
                                bool(block.is_error),
                                _text_of(block.content),
                            )
                elif isinstance(message, ResultMessage):
                    return

    asyncio.run(asyncio.wait_for(run(), _RUN_TIMEOUT_S))

    assert version == [_DEFAULT_CLAUDE_CLI_VERSION], "the run executed the pinned build"
    assert {"TaskCreate", "TaskGet", "TaskList", "TaskUpdate"} <= listed
    assert all({"TaskCreate", "TaskList"} <= offered for offered in _ScriptedMessagesApi.offered), (
        "the tools were offered to the model on every turn"
    )
    assert set(results) == {"TaskCreate", "TaskList"}, f"tool results seen: {results}"
    created_failed, created = results["TaskCreate"]
    assert not created_failed, f"TaskCreate failed: {created}"
    listing_failed, listing = results["TaskList"]
    assert not listing_failed, f"TaskList failed: {listing}"
    assert _SUBJECT in listing, f"the created task is not on the list: {listing}"
