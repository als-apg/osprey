"""Tests for the DePlot FastAPI server endpoints."""

from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from osprey.services.deplot.server import create_app


@pytest.fixture
def app():
    """Create a test FastAPI app."""
    return create_app()


@pytest.fixture
async def client(app):
    """Create an async httpx test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestHealthEndpoint:
    """Tests for GET /health."""

    async def test_health_returns_ok(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "deplot"


class TestExtractEndpoint:
    """Tests for POST /extract."""

    async def test_extract_without_preprocess(self, client):
        """Test extraction with preprocessing disabled (mocked model)."""
        mock_result = {
            "title": "Test Chart",
            "columns": ["x", "y"],
            "data": [[1.0, 2.0], [3.0, 4.0]],
            "raw_table": "x | y\n1 | 2\n3 | 4",
        }

        with (
            patch(
                "osprey.services.deplot.model.extract_table",
                return_value="x | y\n1 | 2\n3 | 4",
            ),
            patch(
                "osprey.services.deplot.model.parse_table",
                return_value=mock_result,
            ),
        ):
            import io

            from PIL import Image

            buf = io.BytesIO()
            Image.new("RGB", (10, 10), "white").save(buf, format="PNG")
            buf.seek(0)

            resp = await client.post(
                "/extract",
                files={"image": ("test.png", buf, "image/png")},
                params={"preprocess": "false"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["columns"] == ["x", "y"]
        assert len(data["data"]) == 2

    async def test_extract_with_preprocess(self, client):
        """Test extraction with preprocessing enabled (mocked model + CV)."""
        pytest.importorskip("cv2", reason="OpenCV not installed")

        mock_result = {
            "title": "",
            "columns": ["time", "value"],
            "data": [[0.0, 1.5]],
            "raw_table": "time | value\n0 | 1.5",
        }

        with (
            patch(
                "osprey.services.deplot.preprocessing.preprocess_chart",
                side_effect=lambda img, **kw: img,  # passthrough
            ),
            patch(
                "osprey.services.deplot.model.extract_table",
                return_value="time | value\n0 | 1.5",
            ),
            patch(
                "osprey.services.deplot.model.parse_table",
                return_value=mock_result,
            ),
        ):
            import io

            from PIL import Image

            buf = io.BytesIO()
            Image.new("RGB", (10, 10), "white").save(buf, format="PNG")
            buf.seek(0)

            resp = await client.post(
                "/extract",
                files={"image": ("chart.png", buf, "image/png")},
                params={"preprocess": "true"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "columns" in data
