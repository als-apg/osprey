"""Unit tests for the containerized Python execution engine.

This module is part of the sandboxed-execution safety surface: it drives code
into a Jupyter container over HTTP + WebSocket and collects results back through
a shared filesystem. The tests below pin the *behavioural contract* of that
plumbing without touching a real runtime:

* The container runtime boundary is entirely mocked. ``requests`` (session /
  kernel HTTP) and ``websocket`` (kernel channel) are patched at the module
  seam; no HTTP, WebSocket, container, or network call is ever made.
* The file-based result collector is exercised against real ``tmp_path``
  directories only.

Contracts covered: endpoint URL/protocol derivation, session-manager health
and creation error-translation, WebSocket execute request construction,
completion/timeout/error classification, file-based result assembly and its
error paths, orchestration re-raise-vs-wrap behaviour, and cleanup guarantees.

There are no container-runtime CLI invocations anywhere in this engine, so no
image/prune/volume command strings are constructed or asserted.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import requests
import websocket

from osprey.services.python_executor.exceptions import (
    CodeRuntimeError,
    ContainerConnectivityError,
    ExecutionTimeoutError,
)
from osprey.services.python_executor.execution import container_engine as ce
from osprey.services.python_executor.models import PythonExecutionEngineResult

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeWS:
    """Minimal stand-in for a ``websocket.WebSocket`` connection.

    ``recv`` replays a queued list of raw JSON strings; once drained it raises
    ``TimeoutError`` to mimic an idle channel (which the engine treats as a
    recv timeout to keep waiting).
    """

    def __init__(self, messages: list[str] | None = None):
        self._messages = list(messages or [])
        self.sent: list[str] = []
        self.closed = False
        self.recv_timeout: float | None = None

    def settimeout(self, value: float) -> None:
        self.recv_timeout = value

    def send(self, data: str) -> None:
        self.sent.append(data)

    def recv(self) -> str:
        if self._messages:
            return self._messages.pop(0)
        raise TimeoutError("no more messages")

    def close(self) -> None:
        self.closed = True


class FakeResponse:
    """Stand-in for a ``requests`` response object."""

    def __init__(self, status_code=200, json_data=None, text="", raise_http=False):
        self.status_code = status_code
        self._json = json_data or {}
        self.text = text
        self._raise_http = raise_http

    def json(self):
        return self._json

    def raise_for_status(self):
        if self._raise_http:
            raise requests.exceptions.HTTPError("http error")


def _execute_reply(msg_id: str, status: str = "ok", content: dict | None = None) -> str:
    body = {"status": status}
    if content:
        body.update(content)
    return json.dumps(
        {
            "header": {"msg_type": "execute_reply"},
            "parent_header": {"msg_id": msg_id},
            "content": body,
        }
    )


# ---------------------------------------------------------------------------
# ContainerEndpoint
# ---------------------------------------------------------------------------


class TestContainerEndpoint:
    def test_http_base_url_and_ws_protocol(self):
        ep = ce.ContainerEndpoint(host="localhost", port=8888, kernel_name="python3")
        assert ep.base_url == "http://localhost:8888"
        assert ep.ws_protocol == "ws"

    def test_https_base_url_and_wss_protocol(self):
        ep = ce.ContainerEndpoint(
            host="jupyter.example.com", port=443, kernel_name="python3", use_https=True
        )
        assert ep.base_url == "https://jupyter.example.com:443"
        assert ep.ws_protocol == "wss"

    def test_use_https_defaults_false(self):
        ep = ce.ContainerEndpoint(host="h", port=1, kernel_name="k")
        assert ep.use_https is False


class TestSessionInfo:
    def test_valid_when_both_ids_present(self):
        assert ce.SessionInfo(session_id="s", kernel_id="k").is_valid is True

    @pytest.mark.parametrize(
        "session_id,kernel_id",
        [("", "k"), ("s", ""), ("", "")],
    )
    def test_invalid_when_any_id_missing(self, session_id, kernel_id):
        assert ce.SessionInfo(session_id=session_id, kernel_id=kernel_id).is_valid is False


# ---------------------------------------------------------------------------
# JupyterSessionManager
# ---------------------------------------------------------------------------


class TestSessionHealth:
    def _manager(self):
        ep = ce.ContainerEndpoint("h", 8888, "python3")
        return ce.JupyterSessionManager(ep)

    async def test_health_true_on_http_200(self, monkeypatch):
        monkeypatch.setattr(ce.requests, "get", lambda *a, **k: FakeResponse(status_code=200))
        mgr = self._manager()
        assert await mgr.check_session_health(ce.SessionInfo("s", "k")) is True

    async def test_health_false_on_non_200(self, monkeypatch):
        monkeypatch.setattr(ce.requests, "get", lambda *a, **k: FakeResponse(status_code=404))
        mgr = self._manager()
        assert await mgr.check_session_health(ce.SessionInfo("s", "k")) is False

    async def test_health_false_on_request_exception(self, monkeypatch):
        def _boom(*a, **k):
            raise requests.exceptions.ConnectionError("down")

        monkeypatch.setattr(ce.requests, "get", _boom)
        mgr = self._manager()
        assert await mgr.check_session_health(ce.SessionInfo("s", "k")) is False


class TestSessionCreation:
    def _manager(self):
        ep = ce.ContainerEndpoint("myhost", 8888, "python3")
        return ce.JupyterSessionManager(ep)

    async def test_creates_session_from_response(self, monkeypatch):
        captured = {}

        def _post(url, json=None, timeout=None, proxies=None):
            captured["url"] = url
            captured["json"] = json
            return FakeResponse(json_data={"id": "sess-1", "kernel": {"id": "kern-1"}})

        monkeypatch.setattr(ce.requests, "post", _post)
        mgr = self._manager()
        session = await mgr._create_new_session()

        assert session.session_id == "sess-1"
        assert session.kernel_id == "kern-1"
        assert session.is_valid
        # The kernel name from the endpoint is propagated into the request body.
        assert captured["json"]["kernel"]["name"] == "python3"
        assert captured["url"].endswith("/api/sessions")

    async def test_connect_timeout_becomes_connectivity_error(self, monkeypatch):
        def _post(*a, **k):
            raise requests.exceptions.ConnectTimeout("slow")

        monkeypatch.setattr(ce.requests, "post", _post)
        mgr = self._manager()
        with pytest.raises(ContainerConnectivityError) as exc:
            await mgr._create_new_session()
        assert exc.value.host == "myhost"
        assert exc.value.port == 8888

    async def test_connection_error_becomes_connectivity_error(self, monkeypatch):
        def _post(*a, **k):
            raise requests.exceptions.ConnectionError("refused")

        monkeypatch.setattr(ce.requests, "post", _post)
        mgr = self._manager()
        with pytest.raises(ContainerConnectivityError):
            await mgr._create_new_session()

    async def test_http_error_becomes_connectivity_error(self, monkeypatch):
        def _post(*a, **k):
            return FakeResponse(status_code=500, text="boom", raise_http=True)

        monkeypatch.setattr(ce.requests, "post", _post)
        mgr = self._manager()
        with pytest.raises(ContainerConnectivityError) as exc:
            await mgr._create_new_session()
        assert exc.value.technical_details["http_error"] == 500

    async def test_generic_request_exception_becomes_connectivity_error(self, monkeypatch):
        def _post(*a, **k):
            raise requests.exceptions.RequestException("weird")

        monkeypatch.setattr(ce.requests, "post", _post)
        mgr = self._manager()
        with pytest.raises(ContainerConnectivityError):
            await mgr._create_new_session()

    async def test_unexpected_exception_becomes_connectivity_error(self, monkeypatch):
        def _post(*a, **k):
            raise ValueError("not a requests error")

        monkeypatch.setattr(ce.requests, "post", _post)
        mgr = self._manager()
        with pytest.raises(ContainerConnectivityError) as exc:
            await mgr._create_new_session()
        assert "unexpected_error" in exc.value.technical_details


class TestEnsureSession:
    def _manager(self):
        ep = ce.ContainerEndpoint("h", 8888, "python3")
        return ce.JupyterSessionManager(ep)

    async def test_reuses_healthy_existing_session(self):
        mgr = self._manager()
        existing = ce.SessionInfo("s0", "k0")
        mgr._current_session = existing
        mgr.check_session_health = AsyncMock(return_value=True)
        mgr._create_new_session = AsyncMock()

        result = await mgr.ensure_session()

        assert result is existing
        mgr._create_new_session.assert_not_called()

    async def test_creates_when_no_session(self):
        mgr = self._manager()
        fresh = ce.SessionInfo("s1", "k1")
        mgr._create_new_session = AsyncMock(return_value=fresh)
        mgr._wait_for_kernel_ready = AsyncMock()

        result = await mgr.ensure_session()

        assert result is fresh
        assert mgr._current_session is fresh
        mgr._wait_for_kernel_ready.assert_awaited_once()

    async def test_recreates_when_existing_unhealthy(self):
        mgr = self._manager()
        mgr._current_session = ce.SessionInfo("stale", "stale")
        mgr.check_session_health = AsyncMock(return_value=False)
        fresh = ce.SessionInfo("s2", "k2")
        mgr._create_new_session = AsyncMock(return_value=fresh)
        mgr._wait_for_kernel_ready = AsyncMock()

        result = await mgr.ensure_session()
        assert result is fresh

    async def test_cleanup_session_is_noop(self):
        # Documented behaviour: sessions are kept alive for reuse.
        mgr = self._manager()
        assert await mgr.cleanup_session() is None


class TestWaitForKernelReady:
    def _manager(self):
        ep = ce.ContainerEndpoint("h", 8888, "python3")
        return ce.JupyterSessionManager(ep)

    async def test_returns_when_kernel_idle(self, monkeypatch):
        monkeypatch.setattr(
            ce.requests,
            "get",
            lambda *a, **k: FakeResponse(json_data={"execution_state": "idle"}),
        )
        mgr = self._manager()
        # Idle on first probe -> returns without sleeping.
        assert await mgr._wait_for_kernel_ready(ce.SessionInfo("s", "k")) is None

    async def test_gives_up_after_max_attempts(self, monkeypatch):
        monkeypatch.setattr(
            ce.requests,
            "get",
            lambda *a, **k: FakeResponse(json_data={"execution_state": "starting"}),
        )
        # Avoid ten real one-second sleeps.
        monkeypatch.setattr(ce.asyncio, "sleep", AsyncMock())
        mgr = self._manager()
        # Never reaches idle; the method logs a warning and returns (best-effort).
        assert await mgr._wait_for_kernel_ready(ce.SessionInfo("s", "k")) is None


# ---------------------------------------------------------------------------
# CodeExecutionEngine
# ---------------------------------------------------------------------------


class TestSendExecuteRequest:
    def test_builds_execute_request_message(self):
        ep = ce.ContainerEndpoint("h", 8888, "python3")
        engine = ce.CodeExecutionEngine(ep)
        ws = FakeWS()
        session = ce.SessionInfo("sess", "kern")

        msg_id = engine._send_execute_request(ws, "print('hi')", session)

        assert isinstance(msg_id, str) and msg_id
        assert len(ws.sent) == 1
        payload = json.loads(ws.sent[0])
        assert payload["header"]["msg_type"] == "execute_request"
        assert payload["header"]["msg_id"] == msg_id
        assert payload["header"]["session"] == "sess"
        assert payload["content"]["code"] == "print('hi')"
        # Stdin is disabled inside the sandbox.
        assert payload["content"]["allow_stdin"] is False


class TestWaitForCompletion:
    def _engine(self, timeout=300):
        ep = ce.ContainerEndpoint("h", 8888, "python3")
        return ce.CodeExecutionEngine(ep, timeout=timeout)

    def test_completes_on_ok_execute_reply(self):
        engine = self._engine()
        ws = FakeWS([_execute_reply("mid", status="ok")])
        # No raise == success.
        engine._wait_for_completion(ws, "mid")
        assert ws.recv_timeout == 5.0

    def test_error_status_raises_code_runtime_error(self):
        engine = self._engine()
        ws = FakeWS(
            [
                _execute_reply(
                    "mid",
                    status="error",
                    content={
                        "ename": "ValueError",
                        "evalue": "bad",
                        "traceback": ["line1", "line2"],
                    },
                )
            ]
        )
        with pytest.raises(CodeRuntimeError) as exc:
            engine._wait_for_completion(ws, "mid")
        assert "ValueError" in str(exc.value)
        assert exc.value.traceback_info == "line1\nline2"

    def test_reply_for_other_msg_id_is_ignored_until_timeout(self):
        # A reply whose parent msg_id does not match ours must not complete us;
        # with a zero timeout the loop exits immediately as a timeout.
        engine = self._engine(timeout=0)
        ws = FakeWS([_execute_reply("someone-else")])
        with pytest.raises(ExecutionTimeoutError) as exc:
            engine._wait_for_completion(ws, "mid")
        assert exc.value.timeout_seconds == 0

    def test_timeout_raises_execution_timeout_error(self):
        engine = self._engine(timeout=0)
        ws = FakeWS([])  # recv would TimeoutError, but loop never runs (timeout=0)
        with pytest.raises(ExecutionTimeoutError):
            engine._wait_for_completion(ws, "mid")

    def test_unexpected_recv_error_raises_connectivity_error(self):
        engine = self._engine(timeout=300)

        class BadWS(FakeWS):
            def recv(self):
                raise ValueError("socket exploded")

        with pytest.raises(ContainerConnectivityError) as exc:
            engine._wait_for_completion(BadWS(), "mid")
        assert "websocket_error" in exc.value.technical_details


class TestExecuteCode:
    def _engine(self):
        ep = ce.ContainerEndpoint("myhost", 8888, "python3")
        return ce.CodeExecutionEngine(ep, timeout=300)

    async def test_success_builds_ws_url_and_closes(self, monkeypatch):
        ws = FakeWS()
        captured = {}

        def _create_connection(url, **kwargs):
            captured["url"] = url
            captured["kwargs"] = kwargs
            return ws

        monkeypatch.setattr(ce.websocket, "create_connection", _create_connection)
        engine = self._engine()
        engine._send_execute_request = MagicMock(return_value="mid")
        engine._wait_for_completion = MagicMock()

        session = ce.SessionInfo("sess", "kern")
        await engine.execute_code("print(1)", session)

        assert "ws://myhost:8888/api/kernels/kern/channels" in captured["url"]
        assert "session_id=sess" in captured["url"]
        engine._send_execute_request.assert_called_once()
        engine._wait_for_completion.assert_called_once_with(ws, "mid")
        # Connection is always closed via the finally block.
        assert ws.closed is True

    async def test_ws_timeout_becomes_connectivity_error(self, monkeypatch):
        def _create_connection(*a, **k):
            raise websocket.WebSocketTimeoutException("slow")

        monkeypatch.setattr(ce.websocket, "create_connection", _create_connection)
        engine = self._engine()
        with pytest.raises(ContainerConnectivityError) as exc:
            await engine.execute_code("code", ce.SessionInfo("s", "k"))
        assert exc.value.technical_details.get("websocket_timeout") is True

    async def test_ws_exception_becomes_connectivity_error(self, monkeypatch):
        def _create_connection(*a, **k):
            raise websocket.WebSocketException("handshake failed")

        monkeypatch.setattr(ce.websocket, "create_connection", _create_connection)
        engine = self._engine()
        with pytest.raises(ContainerConnectivityError) as exc:
            await engine.execute_code("code", ce.SessionInfo("s", "k"))
        assert "websocket_error" in exc.value.technical_details

    async def test_generic_connect_failure_becomes_connectivity_error(self, monkeypatch):
        def _create_connection(*a, **k):
            raise OSError("no route to host")

        monkeypatch.setattr(ce.websocket, "create_connection", _create_connection)
        engine = self._engine()
        with pytest.raises(ContainerConnectivityError) as exc:
            await engine.execute_code("code", ce.SessionInfo("s", "k"))
        assert "connection_failed" in exc.value.technical_details

    async def test_closes_connection_even_when_wait_raises(self, monkeypatch):
        ws = FakeWS()
        monkeypatch.setattr(ce.websocket, "create_connection", lambda *a, **k: ws)
        engine = self._engine()
        engine._send_execute_request = MagicMock(return_value="mid")
        engine._wait_for_completion = MagicMock(side_effect=CodeRuntimeError("x", "", 1))

        with pytest.raises(CodeRuntimeError):
            await engine.execute_code("code", ce.SessionInfo("s", "k"))
        assert ws.closed is True


# ---------------------------------------------------------------------------
# FileBasedResultCollector
# ---------------------------------------------------------------------------


class TestReadJsonFile:
    async def test_no_folder_returns_none(self):
        collector = ce.FileBasedResultCollector(None)
        assert await collector._read_json_file("x.json") is None

    async def test_missing_file_returns_none(self, tmp_path):
        collector = ce.FileBasedResultCollector(tmp_path)
        assert await collector._read_json_file("absent.json") is None

    async def test_valid_json_returned(self, tmp_path):
        (tmp_path / "data.json").write_text(json.dumps({"a": 1}))
        collector = ce.FileBasedResultCollector(tmp_path)
        assert await collector._read_json_file("data.json") == {"a": 1}

    async def test_invalid_json_returns_none(self, tmp_path):
        (tmp_path / "bad.json").write_text("{not valid json")
        collector = ce.FileBasedResultCollector(tmp_path)
        assert await collector._read_json_file("bad.json") is None


class TestCollectFigureFiles:
    async def test_no_folder_returns_empty(self):
        collector = ce.FileBasedResultCollector(None)
        assert await collector._collect_figure_files() == []

    async def test_collects_from_root_and_subdirs_excluding_attempts(self, tmp_path):
        (tmp_path / "root.png").write_bytes(b"\x89PNG")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.png").write_bytes(b"\x89PNG")
        attempts = tmp_path / "attempts"
        attempts.mkdir()
        (attempts / "ignored.png").write_bytes(b"\x89PNG")

        collector = ce.FileBasedResultCollector(tmp_path)
        figures = await collector._collect_figure_files()
        names = {p.name for p in figures}

        assert "root.png" in names
        assert "nested.png" in names
        # Figures under the 'attempts' directory are deliberately excluded.
        assert "ignored.png" not in names


class TestCollectResults:
    def _write_metadata(self, folder: Path, data) -> None:
        (folder / "execution_metadata.json").write_text(json.dumps(data))

    async def test_missing_metadata_raises_with_folder_listing(self, tmp_path):
        (tmp_path / "debug_something.log").write_text("hi")
        collector = ce.FileBasedResultCollector(tmp_path)
        with pytest.raises(CodeRuntimeError) as exc:
            await collector.collect_results(start_time=0.0)
        assert "Failed to read execution metadata" in str(exc.value)
        assert exc.value.technical_details.get("metadata_missing") is True

    async def test_missing_metadata_no_folder_configured(self):
        collector = ce.FileBasedResultCollector(None)
        with pytest.raises(CodeRuntimeError) as exc:
            await collector.collect_results(start_time=0.0)
        assert "not configured" in str(exc.value) or "does not exist" in str(exc.value)

    async def test_success_with_results_dict(self, tmp_path):
        self._write_metadata(
            tmp_path,
            {"success": True, "stdout": "hello", "results_saved": True},
        )
        (tmp_path / "results.json").write_text(json.dumps({"mean": 3.0}))
        collector = ce.FileBasedResultCollector(tmp_path)

        result = await collector.collect_results(start_time=0.0)

        assert isinstance(result, PythonExecutionEngineResult)
        assert result.success is True
        assert result.result_dict == {"mean": 3.0}
        assert result.stdout == "hello"

    async def test_success_combines_stdout_and_stderr(self, tmp_path):
        self._write_metadata(
            tmp_path,
            {"success": True, "stdout": "out", "stderr": "warn", "results_saved": False},
        )
        collector = ce.FileBasedResultCollector(tmp_path)
        result = await collector.collect_results(start_time=0.0)
        assert "out" in result.stdout
        assert "--- STDERR ---" in result.stdout
        assert "warn" in result.stdout

    async def test_success_but_results_missing_raises(self, tmp_path):
        self._write_metadata(
            tmp_path,
            {"success": True, "stdout": "", "results_missing": True},
        )
        collector = ce.FileBasedResultCollector(tmp_path)
        with pytest.raises(CodeRuntimeError) as exc:
            await collector.collect_results(start_time=0.0)
        assert "results" in str(exc.value).lower()

    async def test_code_error_appends_traceback(self, tmp_path):
        self._write_metadata(
            tmp_path,
            {
                "success": False,
                "error": "boom",
                "traceback": "Traceback line",
                "error_type": "CODE_ERROR",
            },
        )
        collector = ce.FileBasedResultCollector(tmp_path)
        with pytest.raises(CodeRuntimeError) as exc:
            await collector.collect_results(start_time=0.0)
        assert "boom" in str(exc.value)
        assert "Traceback line" in str(exc.value)

    async def test_infrastructure_error_stays_connectivity_error(self, tmp_path):
        # INFRASTRUCTURE_ERROR must surface as ContainerConnectivityError so the
        # retry strategy re-executes the same code instead of regenerating it.
        self._write_metadata(
            tmp_path,
            {
                "success": False,
                "error": "ctx load failed",
                "error_type": "INFRASTRUCTURE_ERROR",
                "infrastructure_error": "context.json missing",
            },
        )
        collector = ce.FileBasedResultCollector(tmp_path)
        with pytest.raises(ContainerConnectivityError) as exc:
            await collector.collect_results(start_time=0.0)
        assert "context.json missing" in str(exc.value)

    async def test_malformed_metadata_wrapped_as_code_runtime_error(self, tmp_path):
        # A truthy-but-non-dict metadata payload trips ``.get`` and must be
        # surfaced as a CodeRuntimeError, not leak the raw AttributeError.
        (tmp_path / "execution_metadata.json").write_text(json.dumps([1, 2, 3]))
        collector = ce.FileBasedResultCollector(tmp_path)
        with pytest.raises(CodeRuntimeError) as exc:
            await collector.collect_results(start_time=0.0)
        assert "Failed to collect execution results" in str(exc.value)


# ---------------------------------------------------------------------------
# ContainerExecutor orchestration
# ---------------------------------------------------------------------------


class TestContainerExecutor:
    def _executor(self):
        ep = ce.ContainerEndpoint("myhost", 8888, "python3")
        return ce.ContainerExecutor(endpoint=ep, execution_folder=None, timeout=300)

    def test_initializes_components(self):
        executor = self._executor()
        assert isinstance(executor.session_manager, ce.JupyterSessionManager)
        assert isinstance(executor.execution_engine, ce.CodeExecutionEngine)
        assert isinstance(executor.result_collector, ce.FileBasedResultCollector)
        assert executor.execution_engine.timeout == 300

    async def test_success_returns_collected_result(self):
        executor = self._executor()
        sentinel = PythonExecutionEngineResult(success=True, stdout="ok")
        executor.session_manager.ensure_session = AsyncMock(return_value=ce.SessionInfo("s", "k"))
        executor.session_manager.cleanup_session = AsyncMock()
        executor.execution_engine.execute_code = AsyncMock()
        executor.result_collector.collect_results = AsyncMock(return_value=sentinel)

        result = await executor.execute_code("print(1)")

        assert result is sentinel
        executor.execution_engine.execute_code.assert_awaited_once()
        executor.session_manager.cleanup_session.assert_awaited_once()

    async def test_known_exception_is_reraised_unchanged(self):
        executor = self._executor()
        original = CodeRuntimeError("bad code", "tb", 1)
        executor.session_manager.ensure_session = AsyncMock(side_effect=original)
        executor.session_manager.cleanup_session = AsyncMock()

        with pytest.raises(CodeRuntimeError) as exc:
            await executor.execute_code("print(1)")
        assert exc.value is original
        executor.session_manager.cleanup_session.assert_awaited_once()

    async def test_unexpected_exception_wrapped_as_connectivity_error(self):
        executor = self._executor()
        executor.session_manager.ensure_session = AsyncMock(side_effect=RuntimeError("kaboom"))
        executor.session_manager.cleanup_session = AsyncMock()

        with pytest.raises(ContainerConnectivityError) as exc:
            await executor.execute_code("print(1)")
        assert exc.value.host == "myhost"
        assert exc.value.port == 8888
        assert "unexpected_error" in exc.value.technical_details
        # Cleanup still runs on the failure path.
        executor.session_manager.cleanup_session.assert_awaited_once()


class TestExecutePublicApi:
    async def test_delegates_to_container_executor(self, monkeypatch):
        captured = {}
        sentinel = PythonExecutionEngineResult(success=True, stdout="ok")

        class FakeExecutor:
            def __init__(self, **kwargs):
                captured["init"] = kwargs

            async def execute_code(self, code):
                captured["code"] = code
                return sentinel

        monkeypatch.setattr(ce, "ContainerExecutor", FakeExecutor)
        ep = ce.ContainerEndpoint("h", 8888, "python3")
        result = await ce.execute_python_code_in_container(
            code="print(1)", endpoint=ep, timeout=42, execution_folder=Path("/tmp/exec")
        )

        assert result is sentinel
        assert captured["code"] == "print(1)"
        assert captured["init"]["timeout"] == 42
        assert captured["init"]["execution_folder"] == Path("/tmp/exec")
        assert captured["init"]["endpoint"] is ep
