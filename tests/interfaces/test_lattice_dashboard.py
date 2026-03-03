"""Tests for lattice dashboard infrastructure.

Covers:
- Plotly.js CDN version compatibility
- Raw data serialization (no bdata, no numpy types)
- figure_to_dict round-trip safety
- Worker raw data schema spot-checks
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest

from osprey.interfaces.lattice_dashboard.workers._base import (
    _numpy_default,
    figure_to_dict,
    save_data,
)

STATIC_DIR = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "osprey"
    / "interfaces"
    / "lattice_dashboard"
    / "static"
)


class TestPlotlyCDNVersion:
    """Verify that the Plotly.js CDN version supports bdata serialization."""

    @pytest.fixture
    def index_html(self) -> str:
        return (STATIC_DIR / "index.html").read_text()

    def test_plotly_cdn_version_supports_bdata(self, index_html: str) -> None:
        """Plotly.js must be >= 3.0 to decode bdata from Plotly.py 6.x."""
        match = re.search(r"plotly-(\d+)\.(\d+)\.\d+", index_html)
        assert match, "No Plotly CDN URL found in index.html"
        major = int(match.group(1))
        assert major >= 3, (
            f"Plotly.js {match.group(0)} does not support bdata; "
            "upgrade to >= 3.0 for Plotly.py 6.x compatibility"
        )


class TestNumpyDefault:
    """_numpy_default converts numpy types to JSON-safe primitives."""

    def test_ndarray_to_list(self) -> None:
        arr = np.array([1.0, 2.0, 3.0])
        result = _numpy_default(arr)
        assert result == [1.0, 2.0, 3.0]
        assert isinstance(result, list)

    def test_2d_ndarray_to_nested_list(self) -> None:
        arr = np.array([[1, 2], [3, 4]])
        result = _numpy_default(arr)
        assert result == [[1, 2], [3, 4]]

    def test_numpy_float_to_float(self) -> None:
        val = np.float64(3.14)
        result = _numpy_default(val)
        assert result == 3.14
        assert isinstance(result, float)

    def test_numpy_int_to_float(self) -> None:
        val = np.int64(42)
        result = _numpy_default(val)
        assert result == 42.0
        assert isinstance(result, float)

    def test_unsupported_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Not JSON serializable"):
            _numpy_default(object())


class TestSaveData:
    """save_data() produces plain JSON with no numpy types or bdata."""

    def test_plain_json_output(self, tmp_path: Path) -> None:
        data = {
            "x": np.array([1.0, 2.0, 3.0]),
            "y": np.array([4.0, 5.0, 6.0]),
            "scalar": np.float64(7.7),
            "baseline": None,
        }
        out = tmp_path / "test.json"
        save_data(data, out)

        raw_text = out.read_text()
        parsed = json.loads(raw_text)

        assert parsed["x"] == [1.0, 2.0, 3.0]
        assert parsed["y"] == [4.0, 5.0, 6.0]
        assert parsed["scalar"] == 7.7
        assert parsed["baseline"] is None

    def test_no_bdata_in_output(self, tmp_path: Path) -> None:
        data = {"arr": np.linspace(0, 1, 100)}
        out = tmp_path / "test.json"
        save_data(data, out)

        raw_text = out.read_text()
        assert "bdata" not in raw_text
        assert "dtype" not in raw_text

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        out = tmp_path / "sub" / "dir" / "test.json"
        save_data({"a": 1}, out)
        assert out.exists()

    def test_nested_numpy_arrays(self, tmp_path: Path) -> None:
        data = {
            "map": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "info": {"val": np.float64(0.5)},
        }
        out = tmp_path / "test.json"
        save_data(data, out)

        parsed = json.loads(out.read_text())
        assert parsed["map"] == [[1.0, 2.0], [3.0, 4.0]]
        assert parsed["info"]["val"] == 0.5


class TestFigureToDict:
    """figure_to_dict() converts Plotly figures to JSON-safe dicts."""

    def test_simple_figure(self) -> None:
        fig = go.Figure(data=[go.Scatter(x=np.array([1, 2, 3]), y=np.array([4, 5, 6]))])
        result = figure_to_dict(fig)

        assert isinstance(result, dict)
        assert "data" in result
        assert "layout" in result

        # Verify the result is fully JSON-serializable
        roundtrip = json.loads(json.dumps(result))
        assert roundtrip == result

    def test_json_serializable_output(self) -> None:
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=np.linspace(0, 1, 50),
                    y=np.random.randn(50),
                )
            ]
        )
        result = figure_to_dict(fig)

        # Ensure full JSON round-trip works (would fail with raw numpy types)
        text = json.dumps(result)
        parsed = json.loads(text)
        assert "data" in parsed
        assert len(parsed["data"]) == 1


class TestRawDataSchemas:
    """Spot-check that worker raw data schemas match expected structure."""

    def test_optics_schema(self, tmp_path: Path) -> None:
        raw = {
            "s_pos": [0.0, 1.0, 2.0],
            "beta_x": [10.0, 12.0, 11.0],
            "beta_y": [5.0, 6.0, 5.5],
            "eta_x": [0.1, 0.15, 0.12],
            "baseline": None,
        }
        out = tmp_path / "optics.json"
        save_data(raw, out)
        parsed = json.loads(out.read_text())
        assert set(parsed.keys()) == {"s_pos", "beta_x", "beta_y", "eta_x", "baseline"}

    def test_optics_schema_with_baseline(self, tmp_path: Path) -> None:
        raw = {
            "s_pos": [0.0, 1.0],
            "beta_x": [10.0, 12.0],
            "beta_y": [5.0, 6.0],
            "eta_x": [0.1, 0.15],
            "baseline": {
                "s_pos": [0.0, 1.0],
                "beta_x": [9.0, 11.0],
                "beta_y": [4.5, 5.5],
                "eta_x": [0.09, 0.14],
            },
        }
        out = tmp_path / "optics.json"
        save_data(raw, out)
        parsed = json.loads(out.read_text())
        bl = parsed["baseline"]
        assert set(bl.keys()) == {"s_pos", "beta_x", "beta_y", "eta_x"}

    def test_resonance_schema(self, tmp_path: Path) -> None:
        raw = {
            "nux": 0.34,
            "nuy": 0.22,
            "baseline_nux": None,
            "baseline_nuy": None,
        }
        out = tmp_path / "resonance.json"
        save_data(raw, out)
        parsed = json.loads(out.read_text())
        assert set(parsed.keys()) == {"nux", "nuy", "baseline_nux", "baseline_nuy"}
        assert isinstance(parsed["nux"], float)

    def test_da_schema(self, tmp_path: Path) -> None:
        raw = {
            "da_x": [1.0, 2.0, 3.0],
            "da_y": [0.5, 1.0, 1.5],
            "area_mm2": 45.2,
            "nturns": 512,
            "baseline": None,
        }
        out = tmp_path / "da.json"
        save_data(raw, out)
        parsed = json.loads(out.read_text())
        assert set(parsed.keys()) == {"da_x", "da_y", "area_mm2", "nturns", "baseline"}

    def test_fma_schema(self, tmp_path: Path) -> None:
        raw = {
            "nux_map": [[0.34, 0.35], [0.33, 0.34]],
            "nuy_map": [[0.22, 0.23], [0.21, 0.22]],
            "diffusion": [[-5.0, -6.0], [-4.0, -5.0]],
            "design_tune": [0.34, 0.22],
            "baseline_tune": None,
        }
        out = tmp_path / "fma.json"
        save_data(raw, out)
        parsed = json.loads(out.read_text())
        assert set(parsed.keys()) == {
            "nux_map",
            "nuy_map",
            "diffusion",
            "design_tune",
            "baseline_tune",
        }
        assert len(parsed["nux_map"]) == 2
        assert len(parsed["nux_map"][0]) == 2
