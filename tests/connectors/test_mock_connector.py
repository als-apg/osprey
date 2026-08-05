"""Tests for mock connector."""

import json
import os
import subprocess
import sys
import textwrap
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
        """Each channel contributes its own rows to the long frame."""
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
        # Every row belongs to exactly one of the two requested channels.
        assert len(current_rows) + len(voltage_rows) == len(df)

        await connector.disconnect()


class TestMockArchiverProcessing:
    """The mock connector must genuinely aggregate non-raw processing modes.

    Regression: resampling at the requested precision_ms against data spaced
    far wider (the 10,000-point cap) inflated the frame with mostly-NaN rows.
    """

    @pytest.mark.asyncio
    async def test_processing_mean_aggregates_multiple_raw_samples(self):
        """A bin much wider than the data's spacing must average, not pass through."""
        connector = MockArchiverConnector()
        # noise_level=0 makes the two independent get_data() calls comparable.
        await connector.connect({"noise_level": 0.0})

        # Both calls generate the same 10 points (the generator's forced
        # minimum) over this 10s window; the 60s mean bin forces every sample
        # into a single aggregation bin.
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

        # Regression: resampling at 1000ms against ~60s-spaced data inflated
        # this to ~604,801 mostly-NaN rows.
        assert len(df) <= 10_001
        assert not df["value"].isna().any()

        await connector.disconnect()


class TestMockArchiverReproducibility:
    """The mock's synthetic data must be reproducible, which it advertises but did not do.

    Regression: non-BPM noise came from the global ``np.random``, and the
    per-PV seed came from salted ``hash(pv_name)``, so results differed both
    within and across processes.
    """

    _WINDOW = (datetime(2024, 1, 15, 10, 0, 0), datetime(2024, 1, 15, 10, 5, 0))

    async def _values(self, pv: str) -> list[float]:
        connector = MockArchiverConnector()
        await connector.connect({})
        start, end = self._WINDOW
        df = await connector.get_data(pv_list=[pv], start_date=start, end_date=end)
        await connector.disconnect()
        return df["value"].tolist()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("pv", ["SR:UNKNOWN:PRESSURE", "SR:BPM01:POSITION"])
    async def test_same_pv_and_window_repeats_within_a_process(self, pv):
        assert await self._values(pv) == await self._values(pv)

    @pytest.mark.asyncio
    async def test_distinct_pvs_do_not_collide(self):
        """Reproducible must not mean identical across channels."""
        assert await self._values("SR:UNKNOWN:PRESSURE") != await self._values("SR:OTHER:PRESSURE")

    def test_same_pv_and_window_repeats_across_processes(self):
        """The point of the fix — run it in fresh interpreters with different hash seeds.

        ``hash()`` is salted per process, so only fresh interpreters can catch
        a seed derived from it.
        """
        script = textwrap.dedent("""
            import asyncio, json
            from datetime import datetime
            from osprey.connectors.archiver.mock_archiver_connector import (
                MockArchiverConnector,
            )

            async def main():
                c = MockArchiverConnector()
                await c.connect({})
                df = await c.get_data(
                    pv_list=["SR:UNKNOWN:PRESSURE"],
                    start_date=datetime(2024, 1, 15, 10, 0, 0),
                    end_date=datetime(2024, 1, 15, 10, 5, 0),
                )
                await c.disconnect()
                print(json.dumps(df["value"].tolist()))

            asyncio.run(main())
        """)

        runs = []
        for seed in ("0", "1", "12345"):
            env = {**os.environ, "PYTHONHASHSEED": seed}
            proc = subprocess.run(  # noqa: S603
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                env=env,
                check=True,
            )
            runs.append(json.loads(proc.stdout.strip().splitlines()[-1]))

        assert runs[0] == runs[1] == runs[2]
        assert len(runs[0]) > 0
