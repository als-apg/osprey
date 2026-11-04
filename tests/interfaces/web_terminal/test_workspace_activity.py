"""Each agent ``set_workspace`` call on CANOPY yields one activity-history row.

The call reaches CANOPY through the terminal's panel proxy (the CANOPY agent
hop) and nowhere else of OSPREY's, so the hop observes it: the request names
the commands, the reply names the batch status the tool returned. One row per
call goes into the history buffer, with ``done``, ``pending`` or ``refused``
(and its reason) as returned, whether the reply came as one JSON body or as an
SSE stream. Every other request on the hop, and every other panel's hop, adds
nothing, and the reply is relayed byte for byte either way.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.common_middleware import WebAuthMiddleware
from osprey.interfaces.web_auth import WebCredentials, reset_web_credentials
from osprey.interfaces.web_terminal.routes import proxy, workspace_activity
from osprey.interfaces.web_terminal.routes.agent_activity import ACTIVITY_BUFFER_MAX
from osprey.interfaces.web_terminal.routes.workspace_activity import (
    MAX_ROW_COMMANDS,
    SetWorkspaceObserver,
    batch_outcome,
    command_rows,
    set_workspace_calls,
)
from osprey.profiles.sidecar_panels import CANOPY_PANEL_ID
from osprey.registry.web import panel_url_state_attr

from ._proxy_fakes import _FakeStreamResponse

BACKEND_URL = "http://127.0.0.1:9600"
HOP_ROUTE = f"/panel/{CANOPY_PANEL_ID}/mcp"
PANEL_TOKEN = "panel-token-value"
BEARER = {"authorization": f"Bearer {PANEL_TOKEN}", "content-type": "application/json"}
MCP_ACCEPT = "application/json, text/event-stream"

COMMANDS = [
    {"cmd": "open", "args": {"ref": "library/sr-orbit", "mode": "polite"}},
    {"cmd": "close_others", "args": {"key": "s:library/sr-orbit"}},
    {"cmd": "close_all"},
]
ROW_COMMANDS = [
    {"cmd": "open", "key": "library/sr-orbit"},
    {"cmd": "close_others", "key": "s:library/sr-orbit"},
    {"cmd": "close_all"},
]


def _call(call_id=7, *, name="set_workspace", commands=COMMANDS) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": call_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": {"commands": commands, "wait_s": 5}},
    }


def _batch(status: str, reason: str | None = None) -> dict:
    batch: dict = {"id": "b-1", "status": status, "results": []}
    if reason is not None:
        batch["reason"] = reason
    return batch


def _answer(call_id=7, *, batch: dict, structured: bool = True) -> dict:
    """A FastMCP tool answer: the document as text, and as structured content."""
    document = {"protocol": 1, "seq": 4, "batch": batch}
    result: dict = {"content": [{"type": "text", "text": json.dumps(document)}], "isError": False}
    if structured:
        result["structuredContent"] = document
    return {"jsonrpc": "2.0", "id": call_id, "result": result}


def _sse(*messages: dict) -> bytes:
    return b"".join(
        b"event: message\r\ndata: " + json.dumps(m).encode() + b"\r\n\r\n" for m in messages
    )


@pytest.fixture
def hop():
    """The proxy behind the real gate, with the CANOPY sidecar running."""
    reset_web_credentials()
    app = FastAPI()
    app.include_router(proxy.router)
    app.state.custom_panels = []
    setattr(app.state, panel_url_state_attr(CANOPY_PANEL_ID), BACKEND_URL)
    app.state.panel_auth_headers = {}
    app.state.proxy_client = httpx.AsyncClient()
    app.state.agent_activity_buffer = deque(maxlen=ACTIVITY_BUFFER_MAX)
    app.state.web_credentials = WebCredentials(
        operator_secret="operator-secret-value", panel_token=PANEL_TOKEN
    )
    app.add_middleware(WebAuthMiddleware, cookie_name="osprey_terminal_session_8080")
    with TestClient(app, client=("127.0.0.1", 54321)) as client:
        yield app, client
    asyncio.run(app.state.proxy_client.aclose())
    reset_web_credentials()


def _reply_json(app, body: bytes, content_type: str = "application/json") -> None:
    async def fake_request(**_kwargs):
        return httpx.Response(200, content=body, headers={"content-type": content_type})

    app.state.proxy_client.request = AsyncMock(side_effect=fake_request)
    app.state.proxy_client.send = AsyncMock(side_effect=AssertionError("no stream expected"))


def _reply_stream(app, chunks: tuple[bytes, ...], content_type="text/event-stream") -> None:
    app.state.proxy_client.send = AsyncMock(
        return_value=_FakeStreamResponse(headers={"content-type": content_type}, chunks=chunks)
    )
    app.state.proxy_client.request = AsyncMock(side_effect=AssertionError("no buffered expected"))


def _post(client, body: bytes | None = None, *, stream: bool = False) -> httpx.Response:
    """POST *body* (one ``set_workspace`` call by default) on the hop; *stream* accepts SSE."""
    headers = {**BEARER, "accept": MCP_ACCEPT} if stream else BEARER
    if body is None:
        body = json.dumps(_call()).encode()
    return client.post(HOP_ROUTE, headers=headers, content=body)


def _rows(app) -> list[dict]:
    return list(app.state.agent_activity_buffer)


def _workspace(row: dict) -> dict:
    assert row["type"] == "agent_activity"
    assert row["tool"] == "set_workspace"
    assert row["target"]["kind"] == "panel"
    assert row["target"]["panel"] == CANOPY_PANEL_ID
    return row["target"]["workspace"]


RETURNS = [
    pytest.param(_batch("done"), {"status": "done"}, id="done"),
    pytest.param(_batch("pending"), {"status": "pending"}, id="pending"),
    pytest.param(
        _batch("refused", "no_window"), {"status": "refused", "reason": "no_window"}, id="refused"
    ),
]


@pytest.mark.no_auth_seam
class TestEachCallYieldsOneRow:
    @pytest.mark.parametrize(("batch", "outcome"), RETURNS)
    def test_set_workspace_json_reply_yields_one_row_with_the_returned_status(
        self, hop, batch, outcome
    ):
        app, client = hop
        reply = json.dumps(_answer(batch=batch)).encode()
        _reply_json(app, reply)

        resp = _post(client)

        assert resp.status_code == 200
        assert resp.content == reply
        [row] = _rows(app)
        assert _workspace(row) == {"commands": ROW_COMMANDS, **outcome}

    @pytest.mark.parametrize(("batch", "outcome"), RETURNS)
    def test_set_workspace_streamed_reply_yields_one_row_with_the_returned_status(
        self, hop, batch, outcome
    ):
        app, client = hop
        progress = {"jsonrpc": "2.0", "method": "notifications/progress", "params": {}}
        stream = _sse(progress, _answer(batch=batch))
        # Split mid-event and mid-line ending, the way a network delivers it.
        chunks = (stream[:40], stream[40:-3], stream[-3:])
        _reply_stream(app, chunks)

        resp = _post(client, stream=True)

        assert resp.status_code == 200
        assert resp.content == stream
        [row] = _rows(app)
        assert _workspace(row) == {"commands": ROW_COMMANDS, **outcome}

    def test_set_workspace_json_reply_on_the_streaming_branch_is_read_too(self, hop):
        app, client = hop
        reply = json.dumps(_answer(batch=_batch("done"))).encode()
        _reply_stream(app, (reply[:10], reply[10:]), content_type="application/json")

        _post(client, stream=True)

        [row] = _rows(app)
        assert _workspace(row)["status"] == "done"

    def test_set_workspace_answer_read_from_text_content_alone(self, hop):
        app, client = hop
        _reply_json(app, json.dumps(_answer(batch=_batch("done"), structured=False)).encode())

        _post(client)

        [row] = _rows(app)
        assert _workspace(row)["status"] == "done"

    def test_set_workspace_tool_error_reads_error(self, hop):
        app, client = hop
        error = {"jsonrpc": "2.0", "id": 7, "result": {"content": [], "isError": True}}
        _reply_json(app, json.dumps(error).encode())

        _post(client)

        [row] = _rows(app)
        assert _workspace(row)["status"] == "error"

    def test_set_workspace_unreachable_upstream_reads_unanswered(self, hop):
        app, client = hop
        app.state.proxy_client.request = AsyncMock(side_effect=httpx.ConnectError("down"))

        resp = _post(client)

        assert resp.status_code == 502
        [row] = _rows(app)
        assert _workspace(row) == {"commands": ROW_COMMANDS, "status": "unanswered"}

    def test_set_workspace_stream_ending_before_the_answer_reads_unanswered(self, hop):
        app, client = hop
        _reply_stream(app, (b'event: message\ndata: {"jsonrpc":"2.0",',))

        _post(client, stream=True)

        [row] = _rows(app)
        assert _workspace(row)["status"] == "unanswered"

    def test_set_workspace_batch_of_two_calls_yields_two_rows(self, hop):
        app, client = hop
        body = json.dumps([_call(1), _call("2", commands=[{"cmd": "reopen"}])]).encode()
        reply = json.dumps(
            [_answer("2", batch=_batch("refused", "queue_full")), _answer(1, batch=_batch("done"))]
        ).encode()
        _reply_json(app, reply)

        _post(client, body)

        statuses = sorted(_workspace(row)["status"] for row in _rows(app))
        assert statuses == ["done", "refused"]


@pytest.mark.no_auth_seam
class TestNothingElseIsRecorded:
    @pytest.mark.parametrize(
        "body",
        [
            json.dumps(_call(name="get_workspace")).encode(),
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).encode(),
            b"not json",
            b"",
        ],
        ids=["other-tool", "tools-list", "not-json", "empty"],
    )
    def test_no_set_workspace_activity_for_other_traffic(self, hop, body):
        app, client = hop
        _reply_json(app, json.dumps(_answer(batch=_batch("done"))).encode())

        _post(client, body)

        assert _rows(app) == []

    def test_no_set_workspace_activity_for_another_panels_hop(self):
        """Only CANOPY's calls are observed."""
        request = object()
        assert (
            workspace_activity.observe_set_workspace(
                request, "agent-hop", json.dumps(_call()).encode()
            )
            is None
        )
        assert (
            workspace_activity.observe_set_workspace(
                request, CANOPY_PANEL_ID, json.dumps(_call()).encode()
            )
            is not None
        )


class TestRowContent:
    def test_set_workspace_activity_command_rows_name_verb_and_key(self):
        rows, more = command_rows(
            [
                {"cmd": "open_tool", "args": {"tool": "trace", "query": "pv=A"}},
                {"cmd": "reopen", "args": {"id": "g-2"}},
                {"cmd": 5},
                "junk",
                {"cmd": "focus", "args": {"key": "x" * 1000}},
            ]
        )
        assert more == 0
        assert rows[:4] == [
            {"cmd": "open_tool", "key": "trace"},
            {"cmd": "reopen", "key": "g-2"},
            {"cmd": "?"},
            {"cmd": "?"},
        ]
        assert len(rows[4]["key"]) == workspace_activity.MAX_ROW_TEXT

    def test_set_workspace_activity_drop_and_close_tile_name_their_subject(self):
        """A drop names what its nested source moves; close_tile names its tile."""
        target = {"tile": "w2", "zone": "centre"}
        rows, _ = command_rows(
            [
                {"cmd": "drop", "args": {"source": {"tab": "@trace/1"}, "target": target}},
                {"cmd": "drop", "args": {"source": {"tile": "w3"}, "target": target}},
                {"cmd": "drop", "args": {"source": {"new": {"ref": "s/GTL"}}, "target": target}},
                {
                    "cmd": "drop",
                    "args": {"source": {"new": {"tool": "trace", "query": "pv=A"}}, "target": {}},
                },
                {"cmd": "drop", "args": {"source": {"new": {"ref": 5}}, "target": target}},
                {"cmd": "drop", "args": {"source": "junk", "target": target}},
                {"cmd": "close_tile", "args": {"tile": "w4"}},
                {"cmd": "drop", "args": {"source": {"tab": "y" * 1000}, "target": target}},
            ]
        )
        assert rows[:7] == [
            {"cmd": "drop", "key": "@trace/1"},
            {"cmd": "drop", "key": "w3"},
            {"cmd": "drop", "key": "s/GTL"},
            {"cmd": "drop", "key": "trace"},
            {"cmd": "drop"},
            {"cmd": "drop"},
            {"cmd": "close_tile", "key": "w4"},
        ]
        assert len(rows[7]["key"]) == workspace_activity.MAX_ROW_TEXT

    def test_set_workspace_activity_long_batch_counts_the_rest(self):
        rows, more = command_rows([{"cmd": "close_all"}] * (MAX_ROW_COMMANDS + 3))
        assert len(rows) == MAX_ROW_COMMANDS
        assert more == 3

    def test_set_workspace_activity_unknown_status_or_odd_reason_is_not_trusted(self):
        assert batch_outcome(_answer(batch=_batch("exploded"))) == {"status": "error"}
        assert batch_outcome(_answer(batch=_batch("refused", "<b>x</b>"))) == {"status": "refused"}
        assert batch_outcome({"jsonrpc": "2.0", "id": 7, "error": {"code": -1}}) == {
            "status": "error"
        }

    def test_set_workspace_activity_ids_keep_number_and_string_apart(self):
        calls = set_workspace_calls(json.dumps(_call(1)).encode())
        assert list(calls) == ["1"]
        assert set_workspace_calls(json.dumps(_call("1")).encode()).keys() != calls.keys()

    def test_set_workspace_activity_observer_records_once_and_never_raises(self):
        class _State:
            agent_activity_buffer = deque()

        class _App:
            state = _State()

        class _Request:
            app = _App()

        observer = SetWorkspaceObserver(
            _Request(), set_workspace_calls(json.dumps(_call()).encode())
        )
        observer.begin("text/event-stream")
        observer.feed(b"data: \xff\xfe not json\n\n")
        observer.feed(_sse(_answer(batch=_batch("done"))))
        observer.close()
        observer.close()
        observer.feed(_sse(_answer(batch=_batch("done"))))

        [row] = _Request.app.state.agent_activity_buffer
        assert row["target"]["workspace"]["status"] == "done"
