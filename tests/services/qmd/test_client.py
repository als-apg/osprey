"""Tests for :mod:`osprey.services.qmd.client`.

Every test drives the client through a stubbed transport that emulates the real
daemon's MCP Streamable-HTTP behaviour: the mandatory ``initialize`` handshake,
the ``Mcp-Session-Id`` header echo, the ``HTTP 400 Missing session ID`` refusal
for an unauthenticated call, and the ``HTTP 404 Session not found`` reply after
a restart. The shapes the stub returns were captured from a live qmd 2.5.3
daemon over a 134,996-document index.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

import pytest

from osprey.deployment.qmd_service import QMDServiceConfig
from osprey.services.qmd import client as client_module
from osprey.services.qmd.client import (
    QMDClient,
    QMDClientError,
    QMDResponse,
    QMDUnavailableError,
)

SESSION_ID = "1f6f4a3c-0000-4000-8000-abcdefabcdef"

#: A ``query`` result exactly as the live daemon returns it: hits under
#: ``structuredContent.results``, paths prefixed with the collection name,
#: line-numbered snippets, and a human rendering under ``content`` that the
#: client must ignore.
LIVE_QUERY_RESULT: dict[str, Any] = {
    "content": [{"type": "text", "text": '3 results for "magnet water leak"'}],
    "structuredContent": {
        "results": [
            {
                "docid": "#de06c5",
                "file": "ariel/2016/03/109769.md",
                "title": "1400: Magnet Water Leak Repaired",
                "score": 1,
                "context": None,
                "line": 1,
                "snippet": "1: @@ -1,3 @@\n2: # 1400: Magnet Water Leak Repaired",
            },
            {
                "docid": "#7bfd04",
                "file": "ariel/2009/06/48914.md",
                "title": "Water LEAK from Magnet !!",
                "score": 0.5,
                "context": "surrounding section",
                "line": 6,
                "snippet": "6: @@ -5,2 @@\n7: split water hose",
            },
        ]
    },
}


class StubTransport:
    """An in-process qmd daemon that speaks just enough of the protocol.

    Args:
        healthy: Whether ``GET /health`` reports ``status: ok``.
        tool_result: ``tools/call`` result payload to return.
        unreachable: When true, every request raises as if the host were down.
        expire_session_once: When true, the first ``tools/call`` carrying a
            session id is refused with the daemon's post-restart 404, as if the
            sidecar had been replaced between requests.
        sse: When true, JSON-RPC replies are wrapped in SSE ``data:`` framing.
    """

    def __init__(
        self,
        *,
        healthy: bool = True,
        tool_result: Mapping[str, Any] | None = None,
        unreachable: bool = False,
        expire_session_once: bool = False,
        sse: bool = False,
    ) -> None:
        self.healthy = healthy
        self.tool_result: Mapping[str, Any] = (
            tool_result if tool_result is not None else LIVE_QUERY_RESULT
        )
        self.unreachable = unreachable
        self.expire_session_once = expire_session_once
        self.sse = sse
        self.health_calls: list[str] = []
        self.posts: list[tuple[dict[str, Any], dict[str, str]]] = []
        self.handshakes = 0

    # -- transport surface -------------------------------------------------

    def get(self, url: str, timeout: float) -> QMDResponse:
        """Answer ``GET /health``."""
        if self.unreachable:
            raise QMDUnavailableError(f"qmd sidecar unreachable at {url}")
        self.health_calls.append(url)
        if not self.healthy:
            return QMDResponse(status=503, body='{"status":"degraded"}')
        return QMDResponse(status=200, body=json.dumps({"status": "ok", "uptime": 1355}))

    def post(
        self, url: str, body: bytes, headers: Mapping[str, str], timeout: float
    ) -> QMDResponse:
        """Answer one JSON-RPC message on ``POST /mcp``."""
        if self.unreachable:
            raise QMDUnavailableError(f"qmd sidecar unreachable at {url}")
        payload = json.loads(body.decode("utf-8"))
        self.posts.append((payload, dict(headers)))
        method = payload.get("method")

        if method == "initialize":
            self.handshakes += 1
            return self._reply(
                payload,
                {
                    "protocolVersion": "2025-06-18",
                    "serverInfo": {"name": "qmd", "version": "2.5.3"},
                },
                headers={"mcp-session-id": SESSION_ID},
            )
        if method == "notifications/initialized":
            return QMDResponse(status=202, body="")

        if not headers.get("Mcp-Session-Id"):
            return QMDResponse(
                status=400,
                body='{"error":{"code":-32000,"message":"Bad Request: Missing session ID"}}',
            )
        if self.expire_session_once:
            self.expire_session_once = False
            return QMDResponse(
                status=404,
                body='{"jsonrpc":"2.0","error":{"code":-32001,"message":"Session not found"}}',
            )
        return self._reply(payload, self.tool_result)

    # -- helpers -----------------------------------------------------------

    def _reply(
        self,
        payload: Mapping[str, Any],
        result: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
    ) -> QMDResponse:
        """Frame ``result`` as a JSON-RPC reply to ``payload``."""
        message = json.dumps({"jsonrpc": "2.0", "id": payload.get("id"), "result": result})
        body = f"event: message\ndata: {message}\n\n" if self.sse else message
        return QMDResponse(status=200, body=body, headers=dict(headers or {}))

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        """Arguments of every ``tools/call`` the client made."""
        return [
            payload["params"] for payload, _ in self.posts if payload.get("method") == "tools/call"
        ]


@pytest.fixture
def config() -> QMDServiceConfig:
    """A resolved sidecar config pointing at a non-default port."""
    return QMDServiceConfig(port=8180)


def make_client(config: QMDServiceConfig | None, **kwargs: Any) -> tuple[QMDClient, StubTransport]:
    """Build a client wired to a fresh stub transport."""
    transport = StubTransport(**kwargs)
    return QMDClient(config, transport=transport), transport


# -- unconfigured ---------------------------------------------------------


def test_unconfigured_client_is_unavailable_without_any_request() -> None:
    """No sidecar configured means no socket, not a failed one."""
    client, transport = make_client(None)

    assert client.is_configured is False
    assert client.base_url is None
    assert client.is_available() is False
    assert client.is_available() is False
    assert transport.health_calls == []
    assert transport.posts == []


def test_unconfigured_client_is_silent(caplog: pytest.LogCaptureFixture) -> None:
    """Unconfigured is a state, so it never logs a degradation."""
    client, _ = make_client(None)
    with caplog.at_level(logging.DEBUG):
        client.is_available()
    assert caplog.records == []


def test_unconfigured_query_raises_unavailable() -> None:
    """A query against an absent sidecar fails loudly rather than silently."""
    client, transport = make_client(None)
    with pytest.raises(QMDUnavailableError):
        client.query("okf", "quadrupole")
    assert transport.posts == []


# -- availability ---------------------------------------------------------


def test_is_available_reads_health_endpoint(config: QMDServiceConfig) -> None:
    """A healthy daemon reports available, from ``GET /health``."""
    client, transport = make_client(config)
    assert client.is_available() is True
    assert transport.health_calls == ["http://127.0.0.1:8180/health"]


def test_health_verdict_is_cached_for_the_ttl(config: QMDServiceConfig) -> None:
    """Repeated availability checks inside the TTL cost one request."""
    client, transport = make_client(config)
    for _ in range(5):
        assert client.is_available() is True
    assert len(transport.health_calls) == 1


def test_health_is_reprobed_once_the_ttl_expires(
    config: QMDServiceConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An expired verdict is re-derived rather than reused."""
    monkeypatch.setattr(client_module, "HEALTH_TTL_SECONDS", 0.0)
    client, transport = make_client(config)
    client.is_available()
    client.is_available()
    assert len(transport.health_calls) == 2


def test_unhealthy_status_code_is_unavailable(config: QMDServiceConfig) -> None:
    """A reachable but unhealthy daemon is not available."""
    client, _ = make_client(config, healthy=False)
    assert client.is_available() is False


def test_unreachable_daemon_is_unavailable(config: QMDServiceConfig) -> None:
    """A refused connection is an availability verdict, not an exception."""
    client, _ = make_client(config, unreachable=True)
    assert client.is_available() is False


def test_down_sidecar_warns_once_per_state_transition(
    config: QMDServiceConfig, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Polling a stopped sidecar logs one warning, not one per call."""
    monkeypatch.setattr(client_module, "HEALTH_TTL_SECONDS", 0.0)
    transport = StubTransport(unreachable=True)
    client = QMDClient(config, transport=transport)

    with caplog.at_level(logging.WARNING):
        for _ in range(4):
            assert client.is_available() is False

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "unavailable" in warnings[0].getMessage()


def test_recovery_logs_a_state_change(
    config: QMDServiceConfig, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Coming back up is also a transition, and is reported once."""
    monkeypatch.setattr(client_module, "HEALTH_TTL_SECONDS", 0.0)
    transport = StubTransport(unreachable=True)
    client = QMDClient(config, transport=transport)

    with caplog.at_level(logging.INFO):
        assert client.is_available() is False
        transport.unreachable = False
        assert client.is_available() is True
        assert client.is_available() is True

    messages = [r.getMessage() for r in caplog.records]
    assert sum("is available" in m for m in messages) == 1
    assert sum("unavailable" in m for m in messages) == 1


# -- protocol -------------------------------------------------------------


def test_query_performs_the_handshake_then_echoes_the_session(config: QMDServiceConfig) -> None:
    """The mandatory handshake happens once; later calls carry its session id."""
    client, transport = make_client(config)
    client.query("ariel", "magnet water leak")
    client.query("ariel", "orbit correction")

    assert transport.handshakes == 1
    methods = [payload["method"] for payload, _ in transport.posts]
    assert methods == [
        "initialize",
        "notifications/initialized",
        "tools/call",
        "tools/call",
    ]
    tool_headers = [h for p, h in transport.posts if p.get("method") == "tools/call"]
    assert all(h["Mcp-Session-Id"] == SESSION_ID for h in tool_headers)


def test_expired_session_is_reestablished_and_the_call_retried(
    config: QMDServiceConfig,
) -> None:
    """A restarted daemon costs one extra handshake, not a failed query."""
    client, transport = make_client(config, expire_session_once=True)
    results = client.query("ariel", "magnet water leak")

    assert transport.handshakes == 2
    assert len(results) == 2


def test_persistently_rejected_session_raises(config: QMDServiceConfig) -> None:
    """A daemon that refuses every session is a fault, not an infinite retry."""

    class AlwaysExpired(StubTransport):
        def post(
            self, url: str, body: bytes, headers: Mapping[str, str], timeout: float
        ) -> QMDResponse:
            response = super().post(url, body, headers, timeout)
            if json.loads(body.decode("utf-8")).get("method") == "tools/call":
                return QMDResponse(status=404, body='{"error":{"message":"Session not found"}}')
            return response

    client = QMDClient(config, transport=AlwaysExpired())
    with pytest.raises(QMDClientError, match="twice"):
        client.query("ariel", "magnet")


def test_sse_framed_replies_are_decoded(config: QMDServiceConfig) -> None:
    """The daemon may frame replies as server-sent events; both shapes parse."""
    client, _ = make_client(config, sse=True)
    results = client.query("ariel", "magnet water leak")
    assert [hit.docid for hit in results] == ["#de06c5", "#7bfd04"]


def test_unreachable_daemon_makes_query_raise(config: QMDServiceConfig) -> None:
    """A query cannot degrade silently — the caller decides what to do."""
    client, _ = make_client(config, unreachable=True)
    with pytest.raises(QMDUnavailableError):
        client.query("ariel", "magnet")


def test_query_failure_invalidates_the_cached_health_verdict(
    config: QMDServiceConfig,
) -> None:
    """A query that proves the daemon is gone must not leave "available" cached."""
    client, transport = make_client(config)
    assert client.is_available() is True

    transport.unreachable = True
    with pytest.raises(QMDUnavailableError):
        client.query("ariel", "magnet")
    assert client.is_available() is False


# -- query arguments ------------------------------------------------------


def test_query_sends_the_documented_tool_arguments(config: QMDServiceConfig) -> None:
    """Every knob maps onto the qmd ``query`` tool's own parameter names."""
    client, transport = make_client(config)
    client.query(
        "okf",
        "quadrupole calibration",
        limit=5,
        min_score=0.2,
        candidate_limit=20,
        rerank=False,
        intent="magnet setup procedures",
    )

    (call,) = transport.tool_calls
    assert call["name"] == "query"
    assert call["arguments"] == {
        "searches": [
            {"type": "lex", "query": "quadrupole calibration"},
            {"type": "vec", "query": "quadrupole calibration"},
        ],
        "limit": 5,
        "minScore": 0.2,
        "collections": ["okf"],
        "candidateLimit": 20,
        "rerank": False,
        "intent": "magnet setup procedures",
    }


def test_rerank_and_candidate_limit_default_to_the_daemons_own(
    config: QMDServiceConfig,
) -> None:
    """Unset knobs are omitted, so qmd's defaults stand and nothing is hardcoded."""
    client, transport = make_client(config)
    client.query("okf", "quadrupole")

    arguments = transport.tool_calls[0]["arguments"]
    assert "rerank" not in arguments
    assert "candidateLimit" not in arguments


def test_rerank_true_is_forwarded(config: QMDServiceConfig) -> None:
    """An agent-facing caller can opt into the slow, higher-quality path."""
    client, transport = make_client(config)
    client.query("ariel", "magnet", rerank=True)
    assert transport.tool_calls[0]["arguments"]["rerank"] is True


def test_no_collection_means_no_filter(config: QMDServiceConfig) -> None:
    """``None`` searches every indexed collection."""
    client, transport = make_client(config)
    client.query(None, "magnet")
    assert "collections" not in transport.tool_calls[0]["arguments"]


def test_search_types_order_is_preserved(config: QMDServiceConfig) -> None:
    """qmd weights the first sub-query 2x, so the caller's order is the contract."""
    client, transport = make_client(config)
    client.query("ariel", "magnet", search_types=("vec", "lex"))
    assert [s["type"] for s in transport.tool_calls[0]["arguments"]["searches"]] == ["vec", "lex"]


def test_empty_search_types_is_rejected(config: QMDServiceConfig) -> None:
    """A search with no sub-queries is a caller bug, caught before the wire."""
    client, transport = make_client(config)
    with pytest.raises(ValueError, match="search_types"):
        client.query("ariel", "magnet", search_types=())
    assert transport.posts == []


# -- result parsing -------------------------------------------------------


def test_results_are_typed_and_collection_prefix_is_stripped(
    config: QMDServiceConfig,
) -> None:
    """Callers get a path they can decode without knowing about the prefix."""
    client, _ = make_client(config)
    first, second = client.query("ariel", "magnet water leak")

    assert first.docid == "#de06c5"
    assert first.file == "2016/03/109769.md"
    assert first.collection == "ariel"
    assert first.title == "1400: Magnet Water Leak Repaired"
    assert isinstance(first.score, float)
    assert first.score == 1.0
    assert first.line == 1
    assert first.snippet.startswith("1: @@")
    assert first.context is None

    assert second.file == "2009/06/48914.md"
    assert second.score == 0.5
    assert second.line == 6
    assert second.context == "surrounding section"


def test_unprefixed_path_is_left_alone(config: QMDServiceConfig) -> None:
    """A path with no collection segment is reported as-is, not truncated."""
    result = {"structuredContent": {"results": [{"docid": "#1", "file": "notes.md", "score": 1}]}}
    client, _ = make_client(config, tool_result=result)
    (hit,) = client.query(None, "notes")
    assert hit.file == "notes.md"
    assert hit.collection == ""


def test_no_matches_returns_an_empty_list(config: QMDServiceConfig) -> None:
    """An unknown collection or a miss is emptiness, not an error."""
    result = {
        "content": [{"type": "text", "text": 'No results found for "magnet"'}],
        "structuredContent": {"results": []},
    }
    client, _ = make_client(config, tool_result=result)
    assert client.query("nope", "magnet") == []


def test_human_rendering_is_ignored(config: QMDServiceConfig) -> None:
    """``content`` is a rendering of the same data; only structured data counts."""
    result = {
        "content": [{"type": "text", "text": "1. some other document"}],
        "structuredContent": {"results": [{"docid": "#z", "file": "c/a.md", "score": 0.7}]},
    }
    client, _ = make_client(config, tool_result=result)
    (hit,) = client.query("c", "anything")
    assert hit.docid == "#z"


def test_missing_structured_content_raises(config: QMDServiceConfig) -> None:
    """A reply the client cannot trust is an error, not an empty result set."""
    client, _ = make_client(config, tool_result={"content": [{"type": "text", "text": "hi"}]})
    with pytest.raises(QMDClientError, match="structuredContent"):
        client.query("ariel", "magnet")


def test_tool_error_result_raises_with_the_daemons_message(config: QMDServiceConfig) -> None:
    """``isError`` results surface the daemon's own explanation."""
    result = {"isError": True, "content": [{"type": "text", "text": "index is rebuilding"}]}
    client, _ = make_client(config, tool_result=result)
    with pytest.raises(QMDClientError, match="index is rebuilding"):
        client.query("ariel", "magnet")


def test_malformed_result_row_raises(config: QMDServiceConfig) -> None:
    """A non-object hit is a protocol break, reported rather than skipped."""
    client, _ = make_client(config, tool_result={"structuredContent": {"results": ["nope"]}})
    with pytest.raises(QMDClientError, match="non-object"):
        client.query("ariel", "magnet")
