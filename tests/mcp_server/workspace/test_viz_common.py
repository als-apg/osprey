"""Tests for _viz_common data reader code generation."""

import json
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

    def test_archiver_series_format_produces_wide_dataframe(self, tmp_path: Path):
        """Current archiver_read on-disk shape: bare {query, series} -- no envelope.

        ``ArtifactStore.save_data`` writes raw JSON with no ``_osprey_metadata``
        wrapper (see ``test_archiver_read_tool.py``'s own assertion of that), so
        this -- not the enveloped legacy fixture above -- is what a real
        ``archiver_read`` artifact file looks like on disk.
        """
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

        ``archiver_read`` always emits every *requested* channel, even one
        with no data in range, as ``{"timestamps": [], "values": []}``.
        ``pd.to_datetime([])`` yields a tz-naive empty index while a
        populated channel's index is tz-aware, so ``pd.concat`` raises
        "Cannot join tz-naive with tz-aware DatetimeIndex" unless the empty
        channel's index is built with ``utc=True`` too. Parametrized on
        dict order since Python dicts preserve insertion order and either
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
        # Both columns must be float64. An empty channel left to pandas'
        # inference lands on object dtype, which survives concat but breaks
        # Plotly Express one layer later -- see the px.line test below.
        assert data["A"].dtype == "float64"
        assert data["B"].dtype == "float64"

    @pytest.mark.parametrize("order", ["populated_first", "empty_first"])
    def test_archiver_series_populated_and_empty_channel_plots(self, tmp_path: Path, order: str):
        """The mixed populated/empty frame must survive Plotly Express, not just concat.

        ``create_interactive_plot`` runs operator-written Plotly code against
        the reader's ``data``; ``px.line(data)`` is the idiomatic call and the
        one the data-visualizer agent template teaches. Plotly Express rejects
        a wide frame "with columns of different type", so an object-dtype empty
        channel beside a float64 populated one raises there even though the
        concat in the reader succeeded. This is the operator scenario "plot
        beam current alongside a channel with no data in this window".
        """
        px = pytest.importorskip("plotly.express")

        populated = {"A": {"timestamps": ["2026-07-30T12:00:00+00:00"], "values": [1.1]}}
        empty = {"B": {"timestamps": [], "values": []}}
        series = {**populated, **empty} if order == "populated_first" else {**empty, **populated}
        fp = tmp_path / f"series_plot_{order}.json"
        fp.write_text(json.dumps({"query": {"channels": ["A", "B"]}, "series": series}))

        data = _exec_data_reader(str(fp))

        assert isinstance(data, pd.DataFrame)
        # Both the bare-frame and explicit-y forms; both raised before the fix.
        px.line(data)
        px.line(data, y=list(data.columns))

    def test_archiver_series_ragged_lengths_tolerated(self, tmp_path: Path):
        """A channel whose timestamps/values differ in length must not raise.

        The sibling implementations (``lttb_downsample_channel`` and
        ``_pivot_channel_series_to_table``) both tolerate this with
        ``zip(..., strict=False)``; the viz reader's pivot must match rather
        than raising an uncaught ``ValueError``.
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

    def test_series_key_collision_list_falls_through_to_generic_dict(self, tmp_path: Path):
        """A non-archiver file with a top-level 'series' key must still load.

        Before this branch existed, ``{"series": [1, 2, 3]}`` loaded fine via
        the generic ``pd.DataFrame(data)`` branch. The archiver-pivot branch
        must not hijack it just because the key name matches.
        """
        fp = tmp_path / "series_list.json"
        fp.write_text(json.dumps({"series": [1, 2, 3]}))

        data = _exec_data_reader(str(fp))

        assert isinstance(data, pd.DataFrame)
        assert list(data.columns) == ["series"]
        assert data["series"].tolist() == [1, 2, 3]

    def test_series_key_collision_bare_lists_falls_through_to_generic_dict(self, tmp_path: Path):
        """A 'series' dict whose entries are bare lists (not per-channel dicts) still loads."""
        fp = tmp_path / "series_bare_lists.json"
        fp.write_text(json.dumps({"series": {"a": [1, 2]}}))

        data = _exec_data_reader(str(fp))

        assert isinstance(data, pd.DataFrame)
        assert list(data.columns) == ["series"]

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

    Regression guard: the series-format branch must be hand-rolled, not an
    ``import osprey...`` at runtime -- the sandbox's AST-level whitelist
    (``osprey.mcp_server.workspace.execution.sandbox_executor``) does not
    include ``osprey`` itself, so any such import would fail *every* JSON
    ``data_source`` load (not just archiver ones) with "Import not allowed".
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
