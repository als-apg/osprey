"""Tests for mock connector."""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from osprey.connectors.archiver.mock_archiver_connector import MockArchiverConnector
from osprey.connectors.control_system.mock_connector import MockConnector


def _config_with_writes_enabled(key, default=None):
    """Mock get_config_value that enables writes but returns sane defaults otherwise."""
    if key == "control_system.writes_enabled":
        return True
    return default


class TestMockConnector:
    """Test MockConnector functionality."""

    @pytest.mark.asyncio
    async def test_connect_disconnect(self):
        """Test connector connection and disconnection."""
        connector = MockConnector()
        config = {"response_delay_ms": 0, "noise_level": 0.01}

        await connector.connect(config)
        assert connector._connected is True

        await connector.disconnect()
        assert connector._connected is False

    @pytest.mark.asyncio
    async def test_read_pv_accepts_any_name(self):
        """Test that mock connector accepts any PV name."""
        with patch("osprey.utils.config.get_config_value", return_value=True):
            connector = MockConnector()
            await connector.connect({"response_delay_ms": 0})

            # Test with arbitrary PV names
            result1 = await connector.read_channel("MADE:UP:CHANNEL")
            assert result1.value is not None
            assert isinstance(result1.value, float)

            result2 = await connector.read_channel("ANY:RANDOM:NAME")
            assert result2.value is not None

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_read_returns_tz_aware_timestamps(self):
        """Live-read timestamps carry an explicit offset (facility zone), not a
        naive datetime — guards the connector render sites against silent
        reversion to ``datetime.now()``."""
        with patch("osprey.utils.config.get_config_value", return_value=True):
            connector = MockConnector()
            await connector.connect({"response_delay_ms": 0})

            result = await connector.read_channel("ANY:CHANNEL")
            assert result.timestamp.tzinfo is not None
            assert result.timestamp.utcoffset() is not None
            assert result.metadata.timestamp.tzinfo is not None

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_read_pv_infers_units(self):
        """Test that connector infers units from PV names."""
        with patch("osprey.utils.config.get_config_value", return_value=True):
            connector = MockConnector()
            await connector.connect({"response_delay_ms": 0})

            # Test beam current units
            beam_result = await connector.read_channel("BEAM:CURRENT")
            assert "mA" in beam_result.metadata.units or "A" in beam_result.metadata.units

            # Test voltage units
            voltage_result = await connector.read_channel("MAGNET:VOLTAGE")
            assert "V" in voltage_result.metadata.units

            # Test pressure units
            pressure_result = await connector.read_channel("VACUUM:PRESSURE")
            assert "Torr" in pressure_result.metadata.units

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_write_and_read_maintains_state(self):
        """Test that mock connector maintains state between writes and reads."""
        connector = MockConnector()
        with patch(
            "osprey.utils.config.get_config_value",
            side_effect=_config_with_writes_enabled,
        ):
            await connector.connect(
                {
                    "response_delay_ms": 0,
                    "noise_level": 0.0,  # No noise for exact comparison
                }
            )

            # Write a value
            pv_name = "TEST:SETPOINT:SP"
            test_value = 123.45
            result = await connector.write_channel(pv_name, test_value)
            assert result.success is True

            # Read it back
            result = await connector.read_channel(pv_name)
            assert abs(result.value - test_value) < 0.1  # Allow tiny variance

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_write_creates_readback(self):
        """Test that writing to :SP creates corresponding :RB."""
        connector = MockConnector()
        with patch(
            "osprey.utils.config.get_config_value",
            side_effect=_config_with_writes_enabled,
        ):
            await connector.connect({"response_delay_ms": 0, "noise_level": 0.001})

            # Write to setpoint
            sp_name = "MAGNET:CURRENT:SP"
            rb_name = "MAGNET:CURRENT:RB"
            test_value = 100.0

            await connector.write_channel(sp_name, test_value)

            # Check that readback exists and is close
            rb_result = await connector.read_channel(rb_name)
            assert abs(rb_result.value - test_value) < 1.0

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_write_disabled(self):
        """Test that writes are blocked via base class when config says false."""
        connector = MockConnector()
        with patch("osprey.utils.config.get_config_value", return_value=False):
            await connector.connect({"response_delay_ms": 0})

            result = await connector.write_channel("TEST:PV", 100.0)
            assert result.success is False

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_read_multiple_channels(self):
        """Test reading multiple PVs concurrently."""
        with patch("osprey.utils.config.get_config_value", return_value=True):
            connector = MockConnector()
            await connector.connect({"response_delay_ms": 0})

            pv_names = ["PV:1", "PV:2", "PV:3", "PV:4"]
            results = await connector.read_multiple_channels(pv_names)

            assert len(results) == len(pv_names)
            for pv_name in pv_names:
                assert pv_name in results
                assert results[pv_name].value is not None

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_validate_pv_always_true(self):
        """Test that all PV names are valid in mock mode."""
        with patch("osprey.utils.config.get_config_value", return_value=True):
            connector = MockConnector()
            await connector.connect({"response_delay_ms": 0})

            assert await connector.validate_channel("ANY:PV:NAME") is True
            assert await connector.validate_channel("RANDOM:CHANNEL") is True

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_metadata(self):
        """Test getting PV metadata."""
        with patch("osprey.utils.config.get_config_value", return_value=True):
            connector = MockConnector()
            await connector.connect({"response_delay_ms": 0})

            metadata = await connector.get_metadata("BEAM:CURRENT")
            assert metadata.units is not None
            assert metadata.description is not None
            assert "Mock" in metadata.description

            await connector.disconnect()


class TestMockArchiverConnector:
    """Test MockArchiverConnector functionality."""

    @pytest.mark.asyncio
    async def test_connect_disconnect(self):
        """Test archiver connection and disconnection."""
        connector = MockArchiverConnector()
        config = {"sample_rate_hz": 1.0, "noise_level": 0.01}

        await connector.connect(config)
        assert connector._connected is True

        await connector.disconnect()
        assert connector._connected is False

    @pytest.mark.asyncio
    async def test_get_data_accepts_any_pvs(self):
        """Test that mock archiver accepts any PV names."""
        connector = MockArchiverConnector()
        await connector.connect({"noise_level": 0.01})

        start_date = datetime(2024, 1, 1, 0, 0, 0)
        end_date = datetime(2024, 1, 1, 1, 0, 0)
        pv_list = ["FAKE:PV:1", "RANDOM:PV:2", "ANY:NAME:3"]

        df = await connector.get_data(pv_list=pv_list, start_date=start_date, end_date=end_date)

        assert df is not None
        assert len(df) > 0
        assert set(df["channel"]) == set(pv_list)

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_get_data_returns_dataframe(self):
        """Test that get_data returns the canonical long-format DataFrame."""
        connector = MockArchiverConnector()
        await connector.connect({"noise_level": 0.01})

        start_date = datetime(2024, 1, 1, 0, 0, 0)
        end_date = datetime(2024, 1, 1, 0, 10, 0)

        df = await connector.get_data(
            pv_list=["BEAM:CURRENT"], start_date=start_date, end_date=end_date, precision_ms=1000
        )

        import pandas as pd

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["timestamp", "channel", "value"]
        assert df["timestamp"].dtype == "datetime64[ns, UTC]"
        assert df["value"].dtype == "float64"
        assert (df["channel"] == "BEAM:CURRENT").all()

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_get_metadata(self):
        """Test getting archiver metadata."""
        connector = MockArchiverConnector()
        await connector.connect({})

        metadata = await connector.get_metadata("BEAM:CURRENT")
        assert metadata.pv_name == "BEAM:CURRENT"
        assert metadata.is_archived is True
        assert metadata.archival_start is not None

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_check_availability_all_true(self):
        """Test that all PVs are available in mock archiver."""
        connector = MockArchiverConnector()
        await connector.connect({})

        pv_names = ["PV:1", "PV:2", "PV:3"]
        availability = await connector.check_availability(pv_names)

        assert len(availability) == len(pv_names)
        for pv in pv_names:
            assert availability[pv] is True

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_generated_time_series_has_variation(self):
        """Test that generated time series have realistic variation."""
        connector = MockArchiverConnector()
        await connector.connect({"noise_level": 0.1})

        start_date = datetime(2024, 1, 1, 0, 0, 0)
        end_date = datetime(2024, 1, 1, 1, 0, 0)

        df = await connector.get_data(
            pv_list=["BEAM:CURRENT"], start_date=start_date, end_date=end_date
        )

        # Check that values vary (not all the same)
        values = df.loc[df["channel"] == "BEAM:CURRENT", "value"].to_numpy()
        assert len(set(values)) > 1
        assert values.std() > 0

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_multi_pv_returns_independent_rows_per_channel(self):
        """Each channel contributes its own rows to the long frame; nothing is
        collapsed, dropped, or cross-mixed between PVs the way a shared-index
        wide frame would force."""
        connector = MockArchiverConnector()
        await connector.connect({"noise_level": 0.01})

        start_date = datetime(2024, 1, 1, 0, 0, 0)
        end_date = datetime(2024, 1, 1, 0, 1, 0)

        df = await connector.get_data(
            pv_list=["BEAM:CURRENT", "MAGNET:VOLTAGE"],
            start_date=start_date,
            end_date=end_date,
            precision_ms=1000,
        )

        assert list(df.columns) == ["timestamp", "channel", "value"]
        current_rows = df[df["channel"] == "BEAM:CURRENT"]
        voltage_rows = df[df["channel"] == "MAGNET:VOLTAGE"]

        assert len(current_rows) > 0
        assert len(voltage_rows) > 0
        # Every row belongs to exactly one of the two requested channels: no
        # row is dropped or double-counted when the per-channel series are
        # assembled into one frame.
        assert len(current_rows) + len(voltage_rows) == len(df)

        await connector.disconnect()


class TestMockArchiverProcessing:
    """The mock connector must genuinely aggregate non-raw processing modes.

    Regression coverage for a resample that used to be bound by the
    *requested* precision_ms instead of the grid actually generated: on a
    long window the 10,000-point cap makes the real sample spacing far wider
    than precision_ms, and even without the cap, the natural spacing
    (duration / (num_points - 1)) is marginally wider than precision_ms.
    Resampling at the requested precision_ms used to ask pandas to fill a much
    finer grid than the data had, inflating the frame with mostly-NaN rows.
    That is now structurally impossible: aggregate_series drops any bin with
    no samples instead of filling it, so no bin-width floor is needed to
    prevent the inflation.
    """

    @pytest.mark.asyncio
    async def test_processing_mean_aggregates_multiple_raw_samples(self):
        """A bin much wider than the data's spacing must average, not pass through."""
        connector = MockArchiverConnector()
        # noise_level=0 makes the generated series deterministic (pure trend +
        # wave) so the raw and mean calls — two independent get_data() calls,
        # each drawing its own noise — are directly comparable.
        await connector.connect({"noise_level": 0.0})

        # Both calls generate the same 10 points (the generator's forced
        # minimum) over this 10s window regardless of precision_ms, since
        # num_points is floored at 10 either way. The raw fetch uses a 1s bin
        # -- finer than the ~1.11s natural sample spacing, so every point
        # keeps its own bin (decimate_raw is a no-op here) -- to get the true
        # per-sample ground truth. The mean fetch uses a 60s bin, wide enough
        # to force every sample into a single aggregation bin, so a real mean
        # is distinguishable from a relabeled pass-through.
        start_date = datetime(2024, 1, 1, 0, 0, 0)
        end_date = datetime(2024, 1, 1, 0, 0, 10)
        pv = "BEAM:CURRENT"

        raw_df = await connector.get_data(
            pv_list=[pv],
            start_date=start_date,
            end_date=end_date,
            precision_ms=1_000,
            processing="raw",
        )
        mean_df = await connector.get_data(
            pv_list=[pv],
            start_date=start_date,
            end_date=end_date,
            precision_ms=60_000,
            processing="mean",
        )

        assert len(raw_df) > 1
        assert len(mean_df) == 1
        assert mean_df["value"].iloc[0] == pytest.approx(raw_df["value"].mean())

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_processing_mean_bounded_when_point_cap_binds(self):
        """A window wide enough to hit the 10,000-point cap must not blow up on resample."""
        connector = MockArchiverConnector()
        await connector.connect({"noise_level": 0.01})

        start_date = datetime(2024, 1, 1)
        end_date = start_date + timedelta(days=7)

        df = await connector.get_data(
            pv_list=["BEAM:CURRENT"],
            start_date=start_date,
            end_date=end_date,
            precision_ms=1000,
            processing="mean",
        )

        # Before the fix, resampling at the raw precision_ms (1000ms) against
        # data actually spaced ~60s apart inflated this to ~604,801 rows,
        # almost entirely NaN. aggregate_series now drops empty bins outright
        # instead of filling them, so the row count stays near the generator's
        # 10,000-point cap and no NaN values can appear at all.
        assert len(df) <= 10_001
        assert not df["value"].isna().any()

        await connector.disconnect()
