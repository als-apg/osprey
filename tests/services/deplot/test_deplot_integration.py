"""Integration tests for DePlot service with a real Pix2Struct model.

These tests create actual matplotlib charts, pass them through the full
DePlot pipeline (preprocessing → model → parsing), and verify the
extracted data is reasonable. They are NOT meant for CI — they require
~1 GB of model weights and take 30–60 s on first run.

Run with:
    uv run pytest tests/services/deplot/test_deplot_integration.py -v
"""

import io

pytest = __import__("pytest")
torch = pytest.importorskip("torch", reason="PyTorch required for DePlot model")
pytest.importorskip("transformers", reason="transformers required for DePlot model")
pytest.importorskip("cv2", reason="OpenCV required for preprocessing")

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402

from osprey.services.deplot.server import create_app  # noqa: E402

matplotlib.use("Agg")

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------


def _make_line_chart(xs: list[float], ys: list[float], title: str) -> bytes:
    """Render a simple line chart and return PNG bytes."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, marker="o")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _make_bar_chart(labels: list[str], values: list[float], title: str) -> bytes:
    """Render a simple bar chart and return PNG bytes."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel("Value")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def app():
    """Create a test FastAPI app."""
    return create_app()


@pytest.fixture
async def client(app):
    """Create an async httpx test client (in-memory, no socket)."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDePlotIntegration:
    """End-to-end tests that run the real DePlot model."""

    async def test_extract_real_model_no_preprocess(self, client):
        """Line chart extraction without preprocessing returns valid data."""
        chart_bytes = _make_line_chart(
            xs=[1, 2, 3, 4, 5],
            ys=[2, 4, 6, 8, 10],
            title="y = 2x",
        )

        resp = await client.post(
            "/extract",
            files={"image": ("chart.png", chart_bytes, "image/png")},
            params={"preprocess": "false"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "columns" in data
        assert "data" in data
        assert "raw_table" in data
        assert len(data["data"]) >= 3, f"Expected ≥3 rows, got {len(data['data'])}"

    async def test_extract_real_model_with_preprocess(self, client):
        """Line chart extraction with preprocessing returns valid data."""
        chart_bytes = _make_line_chart(
            xs=[1, 2, 3, 4, 5],
            ys=[2, 4, 6, 8, 10],
            title="y = 2x",
        )

        resp = await client.post(
            "/extract",
            files={"image": ("chart.png", chart_bytes, "image/png")},
            params={"preprocess": "true"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "columns" in data
        assert "data" in data
        assert len(data["data"]) >= 3, f"Expected ≥3 rows, got {len(data['data'])}"

    async def test_extracted_values_monotonically_increase(self, client):
        """DePlot should capture the upward trend of a y=2x chart."""
        chart_bytes = _make_line_chart(
            xs=[1, 2, 3, 4, 5],
            ys=[2, 4, 6, 8, 10],
            title="y = 2x",
        )

        resp = await client.post(
            "/extract",
            files={"image": ("chart.png", chart_bytes, "image/png")},
            params={"preprocess": "false"},
        )

        assert resp.status_code == 200
        rows = resp.json()["data"]

        # Extract y-values (last column) that are numeric
        y_values = []
        for row in rows:
            val = row[-1] if row else None
            if isinstance(val, (int, float)):
                y_values.append(val)

        assert len(y_values) >= 3, f"Need ≥3 numeric y-values, got {y_values}"

        # Soft monotonicity: at least 2/3 of consecutive pairs should increase
        increases = sum(1 for a, b in zip(y_values[:-1], y_values[1:], strict=True) if b > a)
        total_pairs = len(y_values) - 1
        assert increases >= total_pairs * 2 / 3, (
            f"Expected mostly increasing y-values, got {y_values}"
        )

    async def test_bar_chart_extraction(self, client):
        """Bar chart extraction returns rows with numeric values."""
        chart_bytes = _make_bar_chart(
            labels=["A", "B", "C"],
            values=[10, 20, 30],
            title="Simple Bar Chart",
        )

        resp = await client.post(
            "/extract",
            files={"image": ("chart.png", chart_bytes, "image/png")},
            params={"preprocess": "false"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data
        assert len(data["data"]) >= 2, f"Expected ≥2 rows, got {len(data['data'])}"

        # At least some rows should contain numeric values
        has_numeric = any(isinstance(val, (int, float)) for row in data["data"] for val in row)
        assert has_numeric, f"Expected numeric values in data: {data['data']}"
