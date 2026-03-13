"""Tests for TuningScriptsClient and the execution node.

Uses mocked aiohttp responses — no running API required.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.services.xopt_optimizer.execution.api_client import (
    TuningScriptsAPIError,
    TuningScriptsClient,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_response(status: int = 200, json_data: dict | None = None, text: str = ""):
    """Create a mock aiohttp response."""
    resp = MagicMock()
    resp.status = status
    resp.method = "GET"
    resp.url = "http://test/mock"
    resp.json = AsyncMock(return_value=json_data or {})
    resp.text = AsyncMock(return_value=text)
    return resp


def _make_client(**kwargs) -> TuningScriptsClient:
    """Create a client with config loading disabled."""
    with patch.object(TuningScriptsClient, "_load_api_config", return_value={}):
        return TuningScriptsClient(
            base_url="http://test-api:8001",
            poll_interval_seconds=0.01,  # Fast polling for tests
            timeout_seconds=1.0,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Client method tests
# ---------------------------------------------------------------------------


def _mock_session_for(mock_resp, method="get"):
    """Build a mock aiohttp.ClientSession whose `method` returns mock_resp."""
    # The response is returned from an async context manager (session.get/post)
    resp_ctx = AsyncMock()
    resp_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
    resp_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()
    setattr(mock_session, method, MagicMock(return_value=resp_ctx))
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    return mock_session


class TestTuningScriptsClient:
    """Unit tests for TuningScriptsClient HTTP methods."""

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        client = _make_client()
        mock_resp = _make_mock_response(200, {"status": "ok"})
        mock_session = _mock_session_for(mock_resp, "get")

        with patch(
            "osprey.services.xopt_optimizer.execution.api_client.aiohttp.ClientSession",
            return_value=mock_session,
        ):
            result = await client.health_check()

        assert result == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        client = _make_client()
        mock_resp = _make_mock_response(503, {"detail": "unhealthy"})
        mock_session = _mock_session_for(mock_resp, "get")

        with patch(
            "osprey.services.xopt_optimizer.execution.api_client.aiohttp.ClientSession",
            return_value=mock_session,
        ):
            with pytest.raises(TuningScriptsAPIError) as exc_info:
                await client.health_check()

        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    async def test_submit_config_returns_job_id(self):
        client = _make_client()
        mock_resp = _make_mock_response(200, {"job_id": "abc-123", "status": "submitted"})
        mock_session = _mock_session_for(mock_resp, "post")

        with patch(
            "osprey.services.xopt_optimizer.execution.api_client.aiohttp.ClientSession",
            return_value=mock_session,
        ):
            job_id = await client.submit_config({"algorithm": "random", "n_iterations": 10})

        assert job_id == "abc-123"

    @pytest.mark.asyncio
    async def test_submit_yaml_returns_job_id(self):
        client = _make_client()
        mock_resp = _make_mock_response(200, {"job_id": "abc-123", "status": "submitted"})
        mock_session = _mock_session_for(mock_resp, "post")

        with patch(
            "osprey.services.xopt_optimizer.execution.api_client.aiohttp.ClientSession",
            return_value=mock_session,
        ):
            job_id = await client.submit_yaml("generator:\n  name: random\n")

        assert job_id == "abc-123"

    @pytest.mark.asyncio
    async def test_submit_yaml_with_iterations(self):
        client = _make_client()
        mock_resp = _make_mock_response(200, {"job_id": "abc-123", "status": "submitted"})
        mock_session = _mock_session_for(mock_resp, "post")

        with patch(
            "osprey.services.xopt_optimizer.execution.api_client.aiohttp.ClientSession",
            return_value=mock_session,
        ):
            job_id = await client.submit_yaml("yaml: data\n", n_iterations=10)

        assert job_id == "abc-123"

    @pytest.mark.asyncio
    async def test_get_status(self):
        client = _make_client()
        mock_resp = _make_mock_response(200, {"job_id": "abc", "status": "running"})
        mock_session = _mock_session_for(mock_resp, "get")

        with patch(
            "osprey.services.xopt_optimizer.execution.api_client.aiohttp.ClientSession",
            return_value=mock_session,
        ):
            result = await client.get_status("abc")

        assert result["status"] == "running"

    @pytest.mark.asyncio
    async def test_get_full_state(self):
        client = _make_client()
        full_data = {
            "job_id": "abc",
            "status": "completed",
            "data": [{"x": 1.0, "f": 0.5}],
            "variable_names": ["x"],
            "objective_name": "f",
        }
        mock_resp = _make_mock_response(200, full_data)
        mock_session = _mock_session_for(mock_resp, "get")

        with patch(
            "osprey.services.xopt_optimizer.execution.api_client.aiohttp.ClientSession",
            return_value=mock_session,
        ):
            result = await client.get_full_state("abc")

        assert result["data"] == [{"x": 1.0, "f": 0.5}]

    @pytest.mark.asyncio
    async def test_cancel(self):
        client = _make_client()
        mock_resp = _make_mock_response(200, {"status": "cancelled", "job_id": "abc"})
        mock_session = _mock_session_for(mock_resp, "post")

        with patch(
            "osprey.services.xopt_optimizer.execution.api_client.aiohttp.ClientSession",
            return_value=mock_session,
        ):
            result = await client.cancel("abc")

        assert result["status"] == "cancelled"


# ---------------------------------------------------------------------------
# Polling tests
# ---------------------------------------------------------------------------


class TestPolling:
    """Tests for poll_until_complete."""

    @pytest.mark.asyncio
    async def test_poll_completes_on_success(self):
        client = _make_client()

        # First call returns running, second returns completed
        status_responses = [
            {"job_id": "j1", "status": "running"},
            {"job_id": "j1", "status": "completed"},
        ]
        full_state = {
            "job_id": "j1",
            "status": "completed",
            "data": [{"x": 1.0}],
        }

        call_count = 0

        async def mock_get_status(job_id):
            nonlocal call_count
            resp = status_responses[min(call_count, len(status_responses) - 1)]
            call_count += 1
            return resp

        client.get_status = mock_get_status
        client.get_full_state = AsyncMock(return_value=full_state)

        result = await client.poll_until_complete("j1")
        assert result["status"] == "completed"
        assert result["data"] == [{"x": 1.0}]

    @pytest.mark.asyncio
    async def test_poll_raises_on_error_status(self):
        client = _make_client()

        client.get_status = AsyncMock(return_value={"job_id": "j1", "status": "error"})
        client.get_full_state = AsyncMock(
            return_value={"job_id": "j1", "status": "error", "message": "Divergence detected"}
        )

        with pytest.raises(TuningScriptsAPIError, match="Divergence detected"):
            await client.poll_until_complete("j1")

    @pytest.mark.asyncio
    async def test_poll_timeout(self):
        client = _make_client()
        client.timeout = 0.05  # Very short timeout

        client.get_status = AsyncMock(return_value={"job_id": "j1", "status": "running"})

        with pytest.raises(TuningScriptsAPIError, match="Timeout"):
            await client.poll_until_complete("j1")


# ---------------------------------------------------------------------------
# Executor node tests
# ---------------------------------------------------------------------------


class TestExecutorNode:
    """Tests for the execution node with mocked client."""

    @pytest.mark.asyncio
    async def test_executor_with_api_success(self):
        from osprey.services.xopt_optimizer.execution.node import create_executor_node

        full_state = {
            "job_id": "test-job",
            "status": "completed",
            "data": [{"x": 1.0, "f": 0.5}],
            "environment_name": "test_env",
            "objective_name": "f",
            "variable_names": ["x"],
            "results_path": "2024/test",
            "logs": "some logs",
        }

        mock_client = MagicMock(spec=TuningScriptsClient)
        mock_client.health_check = AsyncMock(return_value={"status": "ok"})
        mock_client.submit_config = AsyncMock(return_value="test-job")
        mock_client.poll_until_complete = AsyncMock(return_value=full_state)

        node = create_executor_node()

        state = {
            "optimization_config": {"algorithm": "random", "n_iterations": 10},
            "request": MagicMock(),
        }

        with patch(
            "osprey.services.xopt_optimizer.execution.node.TuningScriptsClient",
            return_value=mock_client,
        ):
            result = await node(state)

        assert result["execution_failed"] is False
        assert result["run_artifact"]["job_id"] == "test-job"
        assert result["run_artifact"]["data"] == [{"x": 1.0, "f": 0.5}]

    @pytest.mark.asyncio
    async def test_executor_api_error(self):
        from osprey.services.xopt_optimizer.execution.node import create_executor_node

        mock_client = MagicMock(spec=TuningScriptsClient)
        mock_client.health_check = AsyncMock(return_value={"status": "ok"})
        mock_client.submit_config = AsyncMock(
            side_effect=TuningScriptsAPIError("conflict", status_code=409, detail="already running")
        )

        node = create_executor_node()
        state = {"optimization_config": {"algorithm": "random"}, "request": MagicMock()}

        with patch(
            "osprey.services.xopt_optimizer.execution.node.TuningScriptsClient",
            return_value=mock_client,
        ):
            result = await node(state)

        assert result["execution_failed"] is True
        assert result["is_failed"] is True

    @pytest.mark.asyncio
    async def test_executor_falls_back_on_connection_error(self):
        from osprey.services.xopt_optimizer.execution.node import create_executor_node

        mock_client = MagicMock(spec=TuningScriptsClient)
        mock_client.health_check = AsyncMock(
            side_effect=TuningScriptsAPIError("Connection refused")
        )

        node = create_executor_node()
        state = {"optimization_config": {"algorithm": "random"}, "request": MagicMock()}

        with patch(
            "osprey.services.xopt_optimizer.execution.node.TuningScriptsClient",
            return_value=mock_client,
        ):
            result = await node(state)

        # Should fall back to placeholder, not fail
        assert result["execution_failed"] is False
        assert "placeholder" in result["run_artifact"].get("note", "").lower()

    @pytest.mark.asyncio
    async def test_executor_missing_config(self):
        from osprey.services.xopt_optimizer.execution.node import create_executor_node

        node = create_executor_node()
        state = {"optimization_config": None, "request": MagicMock()}

        result = await node(state)

        assert result["execution_failed"] is True
        assert "Missing optimization config" in result["failure_reason"]
