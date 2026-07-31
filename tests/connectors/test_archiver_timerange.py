"""Tests for the shared archiver time-range and processing helpers."""

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from osprey.connectors.archiver._timerange import (
    PROCESSING_MODES,
    align_to_grid,
    resolve_processing,
    to_utc,
)


class TestToUtc:
    def test_aware_non_utc_converts(self):
        dt = datetime(2026, 7, 30, 11, 0, 0, tzinfo=ZoneInfo("America/Los_Angeles"))
        assert to_utc(dt) == datetime(2026, 7, 30, 18, 0, 0, tzinfo=UTC)

    def test_aware_utc_is_unchanged(self):
        dt = datetime(2026, 7, 30, 18, 0, 0, tzinfo=UTC)
        assert to_utc(dt) == dt

    def test_naive_reads_as_facility_zone(self, monkeypatch):
        monkeypatch.setattr(
            "osprey.connectors.archiver._timerange.get_facility_timezone",
            lambda: ZoneInfo("America/Los_Angeles"),
        )
        assert to_utc(datetime(2026, 7, 30, 11, 0, 0)) == datetime(
            2026, 7, 30, 18, 0, 0, tzinfo=UTC
        )

    def test_naive_with_no_configured_zone_is_utc(self):
        # tests/conftest.py clears CONFIG_FILE, so get_facility_timezone() -> UTC
        assert to_utc(datetime(2026, 7, 30, 18, 0, 0)) == datetime(
            2026, 7, 30, 18, 0, 0, tzinfo=UTC
        )


class TestResolveProcessing:
    @pytest.mark.parametrize(
        "mode,operator,agg",
        [
            ("raw", "lastSample_60", "last"),
            ("mean", "mean_60", "mean"),
            ("min", "min_60", "min"),
            ("max", "max_60", "max"),
            ("median", "median_60", "median"),
            ("std", "std_60", "std"),
            ("count", "count_60", "count"),
        ],
    )
    def test_each_mode_renders_for_both_backends(self, mode, operator, agg):
        p = resolve_processing(mode, 60_000)
        assert p.epics_operator == operator
        assert p.pandas_agg == agg

    def test_every_declared_mode_resolves(self):
        for mode in PROCESSING_MODES:
            assert resolve_processing(mode, 1000).pandas_agg

    def test_raw_at_full_resolution_has_no_operator(self):
        assert resolve_processing("raw", 0).epics_operator is None

    def test_unknown_mode_lists_the_valid_set(self):
        with pytest.raises(ValueError, match="Unknown processing mode"):
            resolve_processing("p99", 1000)

    def test_aggregate_without_bin_width_is_rejected(self):
        with pytest.raises(ValueError, match="requires precision_ms > 0"):
            resolve_processing("mean", 0)


class TestAlignToGrid:
    def test_utc_series_aligns_onto_the_grid(self):
        idx = pd.to_datetime([0, 60], unit="s", utc=True)
        series = {"PV": pd.Series([1.0, 2.0], index=idx, name="PV")}
        frame = align_to_grid(
            series,
            datetime(1970, 1, 1, tzinfo=UTC),
            datetime(1970, 1, 1, 0, 2, tzinfo=UTC),
            60_000,
        )
        assert list(frame.columns) == ["PV"]
        assert frame["PV"].tolist() == [1.0, 2.0, 2.0]

    def test_facility_local_bounds_align_against_utc_samples(self):
        idx = pd.to_datetime([0, 60], unit="s", utc=True)
        series = {"PV": pd.Series([1.0, 2.0], index=idx, name="PV")}
        la = ZoneInfo("America/Los_Angeles")
        frame = align_to_grid(
            series,
            datetime(1970, 1, 1, tzinfo=UTC).astimezone(la),
            datetime(1970, 1, 1, 0, 2, tzinfo=UTC).astimezone(la),
            60_000,
        )
        assert frame["PV"].tolist() == [1.0, 2.0, 2.0]

    def test_empty_series_becomes_an_all_nan_column(self):
        series = {"PV": pd.Series(dtype=float, name="PV")}
        frame = align_to_grid(
            series,
            datetime(1970, 1, 1, tzinfo=UTC),
            datetime(1970, 1, 1, 0, 2, tzinfo=UTC),
            60_000,
        )
        assert frame["PV"].isna().all()
        assert len(frame) == 3
