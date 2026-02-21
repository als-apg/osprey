"""Tests for _viz_common data reader code generation."""

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest


def _exec_data_reader(data_source: str) -> object:
    """Generate and execute the data reader code, returning the `data` variable."""
    from osprey.mcp_server.workspace.tools._viz_common import build_data_reader

    code = build_data_reader(data_source)
    ns = {"pd": pd}
    exec(code, ns)  # noqa: S102
    return ns["data"]


class TestBuildDataReaderJSON:
    """Test JSON branch of build_data_reader()."""

    def test_archiver_format_produces_timeseries_dataframe(self, tmp_path: Path):
        """Archiver JSON: {_osprey_metadata, data: {query, dataframe: {split}}}."""
        archiver_json = {
            "_osprey_metadata": {"tool": "archiver_read"},
            "data": {
                "query": {"channels": ["SR:C01-MG:G01"], "start": "2025-01-01"},
                "dataframe": {
                    "columns": ["SR:C01-MG:G01"],
                    "index": [1000, 2000, 3000],
                    "data": [[1.1], [2.2], [3.3]],
                },
            },
        }
        fp = tmp_path / "archiver.json"
        fp.write_text(json.dumps(archiver_json))

        data = _exec_data_reader(str(fp))

        assert isinstance(data, pd.DataFrame)
        assert list(data.columns) == ["SR:C01-MG:G01"]
        assert list(data.index) == [1000, 2000, 3000]
        assert data["SR:C01-MG:G01"].tolist() == pytest.approx([1.1, 2.2, 3.3])

    def test_flat_split_orient_json(self, tmp_path: Path):
        """Flat split-orient: {_osprey_metadata, data: {columns, index, data}}."""
        flat_json = {
            "_osprey_metadata": {"tool": "execute"},
            "data": {
                "columns": ["A", "B"],
                "index": [0, 1],
                "data": [[10, 20], [30, 40]],
            },
        }
        fp = tmp_path / "flat.json"
        fp.write_text(json.dumps(flat_json))

        data = _exec_data_reader(str(fp))

        assert isinstance(data, pd.DataFrame)
        assert list(data.columns) == ["A", "B"]
        assert data["A"].tolist() == [10, 30]
        assert data["B"].tolist() == [20, 40]

    def test_list_of_dicts_json(self, tmp_path: Path):
        """Simple list-of-dicts JSON (existing behavior)."""
        list_json = [
            {"x": 1, "y": 10},
            {"x": 2, "y": 20},
        ]
        fp = tmp_path / "simple.json"
        fp.write_text(json.dumps(list_json))

        data = _exec_data_reader(str(fp))

        assert isinstance(data, pd.DataFrame)
        assert list(data.columns) == ["x", "y"]
        assert data["x"].tolist() == [1, 2]

    def test_plain_dict_json(self, tmp_path: Path):
        """Plain dict without OSPREY envelope or split keys."""
        plain = {"col_a": [1, 2, 3], "col_b": [4, 5, 6]}
        fp = tmp_path / "plain.json"
        fp.write_text(json.dumps(plain))

        data = _exec_data_reader(str(fp))

        assert isinstance(data, pd.DataFrame)
        assert list(data.columns) == ["col_a", "col_b"]
        assert len(data) == 3


class TestBuildDataReaderCSV:
    """Test CSV branch of build_data_reader()."""

    def test_csv_loads_dataframe(self, tmp_path: Path):
        fp = tmp_path / "data.csv"
        fp.write_text("a,b\n1,2\n3,4\n")

        data = _exec_data_reader(str(fp))

        assert isinstance(data, pd.DataFrame)
        assert list(data.columns) == ["a", "b"]
        assert len(data) == 2
