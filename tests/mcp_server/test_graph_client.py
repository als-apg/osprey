"""Tests for the DePlot HTTP client."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from osprey.mcp_server.workspace.tools.graph_client import DePlotClient, get_deplot_url


def _mock_response(status_code: int = 200, json_data: dict | None = None) -> httpx.Response:
    """Build a mock httpx.Response with a request instance set."""
    request = httpx.Request("GET", "http://test")
    resp = httpx.Response(status_code, json=json_data, request=request)
    return resp


class TestGetDeplotUrl:
    """Tests for get_deplot_url config resolution."""

    def test_default_url(self, monkeypatch, tmp_path):
        from osprey.mcp_server.common import reset_config_cache

        monkeypatch.chdir(tmp_path)
        reset_config_cache()
        url = get_deplot_url()
        assert url == "http://127.0.0.1:8095"

    def test_custom_url_from_config(self, monkeypatch, tmp_path):
        import yaml

        from osprey.mcp_server.common import reset_config_cache

        config = tmp_path / "config.yml"
        config.write_text(yaml.dump({"deplot": {"host": "10.0.0.5", "port": 9000}}))
        monkeypatch.chdir(tmp_path)
        reset_config_cache()

        url = get_deplot_url()
        assert url == "http://10.0.0.5:9000"


class TestDePlotClient:
    """Tests for DePlotClient."""

    @pytest.fixture
    def client(self):
        return DePlotClient(base_url="http://localhost:8095")

    async def test_health_success(self, client):
        mock_resp = _mock_response(200, {"status": "ok", "service": "deplot"})
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
            result = await client.health()
        assert result["status"] == "ok"

    async def test_is_available_true(self, client):
        mock_resp = _mock_response(200, {"status": "ok"})
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_resp):
            assert await client.is_available() is True

    async def test_is_available_false_on_connect_error(self, client):
        with patch(
            "httpx.AsyncClient.get",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("refused"),
        ):
            assert await client.is_available() is False

    async def test_extract_file_not_found(self, client):
        with pytest.raises(FileNotFoundError):
            await client.extract("/nonexistent/image.png")

    async def test_extract_success(self, client, tmp_path):
        img_file = tmp_path / "chart.png"
        img_file.write_bytes(b"fake-png-data")

        mock_result = {
            "columns": ["x", "y"],
            "data": [[1.0, 2.0]],
            "raw_table": "x | y\n1 | 2",
            "title": "",
        }
        mock_resp = _mock_response(200, mock_result)

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await client.extract(str(img_file), preprocess=True)

        assert result["columns"] == ["x", "y"]
        assert result["data"] == [[1.0, 2.0]]
