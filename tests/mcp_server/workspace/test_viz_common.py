"""Tests for _viz_common data reader code generation."""

import json
from pathlib import Path
from types import SimpleNamespace

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

    def test_archiver_series_format_produces_wide_dataframe(self, tmp_path: Path):
        """Current archiver_read on-disk shape: bare {query, series} -- no envelope."""
        series_json = {
            "query": {"channels": ["A", "B"], "start_time": "2026-07-30T00:00:00-07:00"},
            "series": {
                "A": {
                    "timestamps": [
                        "2026-07-30T12:00:00+00:00",
                        "2026-07-30T12:01:00+00:00",
                    ],
                    "values": [1.1, 2.2],
                },
                "B": {
                    "timestamps": ["2026-07-30T12:00:30+00:00"],
                    "values": [9.9],
                },
            },
        }
        fp = tmp_path / "series.json"
        fp.write_text(json.dumps(series_json))

        data = _exec_data_reader(str(fp))

        assert isinstance(data, pd.DataFrame)
        assert list(data.columns) == ["A", "B"]
        assert isinstance(data.index, pd.DatetimeIndex)
        # Union of both channels' own real timestamps -- three distinct instants.
        assert len(data) == 3
        assert data["A"].dropna().tolist() == pytest.approx([1.1, 2.2])
        assert data["B"].dropna().tolist() == pytest.approx([9.9])
        # A has no sample at B's timestamp and vice versa -- NaN, not fabricated.
        assert data["A"].isna().sum() == 1
        assert data["B"].isna().sum() == 2

    def test_archiver_series_non_numeric_channel_does_not_crash(self, tmp_path: Path):
        """An enum/status channel (archived as strings) must round-trip, not crash."""
        series_json = {
            "query": {"channels": ["SR:MODE", "SR:DCCT"]},
            "series": {
                "SR:MODE": {
                    "timestamps": ["2026-07-30T12:00:00+00:00"],
                    "values": ["DECAY"],
                },
                "SR:DCCT": {
                    "timestamps": ["2026-07-30T12:00:00+00:00"],
                    "values": [500.0],
                },
            },
        }
        fp = tmp_path / "series_enum.json"
        fp.write_text(json.dumps(series_json))

        data = _exec_data_reader(str(fp))

        assert isinstance(data, pd.DataFrame)
        assert data["SR:MODE"].tolist() == ["DECAY"]
        assert data["SR:DCCT"].tolist() == pytest.approx([500.0])

    def test_archiver_series_empty_yields_empty_dataframe(self, tmp_path: Path):
        """A query that matched zero channels/samples must not crash the pivot."""
        series_json = {"query": {"channels": []}, "series": {}}
        fp = tmp_path / "series_empty.json"
        fp.write_text(json.dumps(series_json))

        data = _exec_data_reader(str(fp))

        assert isinstance(data, pd.DataFrame)
        assert data.empty

    @pytest.mark.parametrize("order", ["populated_first", "empty_first"])
    def test_archiver_series_populated_and_empty_channel_concat(self, tmp_path: Path, order: str):
        """A channel with zero samples in range must not crash the pivot's concat.

        ``pd.to_datetime([])`` yields a tz-naive empty index while a populated
        channel's is tz-aware, so ``pd.concat`` raises unless the empty index
        is built with ``utc=True`` too. Parametrized on dict order since either
        channel could be the one ``pd.concat`` sees first.
        """
        populated = {
            "A": {
                "timestamps": ["2026-07-30T12:00:00+00:00"],
                "values": [1.1],
            }
        }
        empty = {"B": {"timestamps": [], "values": []}}
        series = {**populated, **empty} if order == "populated_first" else {**empty, **populated}
        series_json = {"query": {"channels": ["A", "B"]}, "series": series}
        fp = tmp_path / f"series_mixed_{order}.json"
        fp.write_text(json.dumps(series_json))

        data = _exec_data_reader(str(fp))

        assert isinstance(data, pd.DataFrame)
        assert list(data.columns) == (["A", "B"] if order == "populated_first" else ["B", "A"])
        assert data["A"].dropna().tolist() == pytest.approx([1.1])
        assert data["B"].isna().all()
        # An empty channel left to pandas' inference lands on object dtype,
        # which breaks Plotly Express one layer later -- see the px.line test.
        assert data["A"].dtype == "float64"
        assert data["B"].dtype == "float64"

    @pytest.mark.parametrize("order", ["populated_first", "empty_first"])
    def test_archiver_series_populated_and_empty_channel_plots(self, tmp_path: Path, order: str):
        """The mixed populated/empty frame must survive Plotly Express, not just concat.

        Plotly Express rejects a wide frame "with columns of different type",
        so an object-dtype empty channel beside a float64 populated one raises
        even though the concat succeeded.
        """
        px = pytest.importorskip("plotly.express")

        populated = {"A": {"timestamps": ["2026-07-30T12:00:00+00:00"], "values": [1.1]}}
        empty = {"B": {"timestamps": [], "values": []}}
        series = {**populated, **empty} if order == "populated_first" else {**empty, **populated}
        fp = tmp_path / f"series_plot_{order}.json"
        fp.write_text(json.dumps({"query": {"channels": ["A", "B"]}, "series": series}))

        data = _exec_data_reader(str(fp))

        assert isinstance(data, pd.DataFrame)
        # Both the bare-frame and explicit-y forms.
        px.line(data)
        px.line(data, y=list(data.columns))

    def test_archiver_series_ragged_lengths_tolerated(self, tmp_path: Path):
        """A channel whose timestamps/values differ in length must not raise.

        Matches the sibling implementations' ``zip(..., strict=False)`` tolerance.
        """
        series_json = {
            "query": {"channels": ["A"]},
            "series": {
                "A": {
                    "timestamps": [
                        "2026-07-30T12:00:00+00:00",
                        "2026-07-30T12:01:00+00:00",
                    ],
                    "values": [1.1],  # one shorter than timestamps
                }
            },
        }
        fp = tmp_path / "series_ragged.json"
        fp.write_text(json.dumps(series_json))

        data = _exec_data_reader(str(fp))

        assert isinstance(data, pd.DataFrame)
        assert data["A"].dropna().tolist() == pytest.approx([1.1])

    @pytest.mark.parametrize(
        ("payload", "expected_values"),
        [
            # Dict of dicts, no 'timestamps' anywhere. Pins the
            # `'timestamps' in _v` requirement.
            pytest.param(
                {"series": {"a": {"x": 1}, "b": {"x": 2}}},
                None,
                id="dict_of_dicts_without_timestamps",
            ),
            # One entry carries 'timestamps', the other does not. Pins
            # `all(...)` against `any(...)`.
            pytest.param(
                {
                    "series": {
                        "a": {"timestamps": ["2026-07-30T12:00:00+00:00"], "values": [1.0]},
                        "b": {"x": 2},
                    }
                },
                None,
                id="partial_timestamps",
            ),
            # 'series' is a list, not a dict. Pins the outer
            # `isinstance(_series, dict)`.
            pytest.param({"series": [1, 2, 3]}, [1, 2, 3], id="list"),
            # 'series' entries are bare lists rather than per-channel dicts.
            # Pins the `isinstance(_v, dict)` half of the entry predicate.
            pytest.param({"series": {"a": [1, 2]}}, None, id="bare_lists"),
        ],
    )
    def test_non_archiver_series_key_falls_through_to_generic_dict(
        self, tmp_path: Path, payload: dict, expected_values: list | None
    ):
        """A top-level ``series`` key that is not an archiver envelope must not
        be pivoted -- it falls through to the generic dict/list branch.
        """
        fp = tmp_path / "series_collision.json"
        fp.write_text(json.dumps(payload))

        data = _exec_data_reader(str(fp))

        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert list(data.columns) == ["series"]
        if expected_values is not None:
            assert data["series"].tolist() == expected_values

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


class TestBuildDataReaderSandboxSafety:
    """The generated reader code must clear the viz sandbox's import whitelist.

    The AST-level whitelist does not include ``osprey`` itself, so any runtime
    ``import osprey...`` would fail every JSON ``data_source`` load.
    """

    def test_generated_code_has_no_disallowed_imports(self):
        from osprey.mcp_server.workspace.execution.sandbox_executor import (
            validate_sandbox_code,
        )
        from osprey.mcp_server.workspace.tools._viz_common import build_data_reader

        code = build_data_reader("/tmp/whatever.json")
        is_safe, violations = validate_sandbox_code(code)

        assert is_safe, violations


class TestBuildDataReaderCSV:
    """Test CSV branch of build_data_reader()."""

    def test_csv_loads_dataframe(self, tmp_path: Path):
        fp = tmp_path / "data.csv"
        fp.write_text("a,b\n1,2\n3,4\n")

        data = _exec_data_reader(str(fp))

        assert isinstance(data, pd.DataFrame)
        assert list(data.columns) == ["a", "b"]
        assert len(data) == 2


class TestCollectAndRegisterArtifacts:
    """Artifact-store registration shared by all three visualization tools."""

    @pytest.fixture
    def art_store(self, tmp_path: Path):
        from osprey.stores.artifact_store import initialize_artifact_store

        return initialize_artifact_store(workspace_root=tmp_path)

    @staticmethod
    def _exec_result(*paths: Path):
        """Fake SandboxExecutionResult carrying one artifact dict per path."""
        return SimpleNamespace(
            artifacts=[
                {
                    "path": p,
                    "title": p.stem,
                    "description": f"plot {p.stem}",
                    "artifact_type": "plot_png",
                    "mime_type": "image/png",
                }
                for p in paths
            ]
        )

    def test_unreadable_artifact_skipped_and_data_source_recorded(self, art_store, tmp_path: Path):
        """One artifact failing to save must not lose the ones after it, and
        data_source alone still lands in viz metadata (empty code/stdout add no keys).
        """
        from osprey.mcp_server.workspace.tools._viz_common import collect_and_register_artifacts

        ghost = tmp_path / "ghost.png"  # never written -> read_bytes raises
        real = tmp_path / "real.png"
        real.write_bytes(b"\x89PNG fake")

        ids = collect_and_register_artifacts(
            self._exec_result(ghost, real),
            title="t",
            description="d",
            tool_source="create_static_plot",
            data_source="abcdefabcdef",
        )

        assert len(ids) == 1
        entry = art_store.get_entry(ids[0])
        assert entry.filename.endswith("real.png")
        assert entry.metadata == {"data_source": "abcdefabcdef"}
        # No category given: the entry is neither categorized nor agent-tagged.
        assert entry.category == ""
        assert entry.source_agent == ""


class TestBuildVizResponse:
    """The response dict assembled after artifact registration."""

    def test_gallery_url_failure_is_swallowed(self, monkeypatch):
        """An unresolvable gallery URL never fails the tool — the key is just omitted."""
        import osprey.mcp_server.http as http_mod
        from osprey.mcp_server.workspace.tools._viz_common import build_viz_response

        def _boom():
            raise RuntimeError("no web config")

        monkeypatch.setattr(http_mod, "gallery_url", _boom)

        response = build_viz_response(["abc123def456"], title="t")

        assert response["status"] == "success"
        assert response["artifact_id"] == "abc123def456"
        assert "gallery_url" not in response
