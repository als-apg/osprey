"""Tests for mock connector."""

import asyncio
import json
import os
import subprocess
import sys
import textwrap
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from osprey.connectors.archiver.mock_archiver_connector import MockArchiverConnector
from osprey.connectors.control_system.base import WriteOutcome
from osprey.connectors.control_system.mock_connector import NO_VIEW_MESSAGE, MockConnector
from tests.facility.served_tree import mock_config, served_tree


def _config_with_writes_enabled(key, default=None):
    """Mock get_config_value that enables writes but returns sane defaults otherwise."""
    if key == "control_system.writes_enabled":
        return True
    return default


class TestMockConnector:
    """Test MockConnector functionality."""

    @pytest.mark.asyncio
    async def test_connect_disconnect(self, tmp_path):
        """Test connector connection and disconnection."""
        connector = MockConnector()
        config = mock_config(served_tree(tmp_path), response_delay_ms=0)

        await connector.connect(config)
        assert connector._connected is True

        await connector.disconnect()
        assert connector._connected is False

    @pytest.mark.asyncio
    async def test_connect_without_a_built_view_is_refused(self, tmp_path, monkeypatch):
        """No view named and none beside a loaded config: connect says to build."""
        monkeypatch.chdir(tmp_path)
        connector = MockConnector()

        with pytest.raises(RuntimeError) as refusal:
            await connector.connect({"response_delay_ms": 0})

        assert str(refusal.value) == NO_VIEW_MESSAGE
        assert connector._connected is False

    @pytest.mark.asyncio
    async def test_connect_reads_the_view_beside_the_loaded_config(self, tmp_path, monkeypatch):
        """With no view named, the view beside the loaded config is served."""
        view = served_tree(tmp_path, readings=["BEAM:CURRENT"])
        render = view.parent.parent
        (render / "config.yml").write_text("control_system:\n  type: mock\n")
        monkeypatch.chdir(render)
        connector = MockConnector()
        await connector.connect({"response_delay_ms": 0})

        assert await connector.validate_channel("BEAM:CURRENT") is True

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_read_refuses_an_address_outside_the_facility_file(self, tmp_path):
        """An address the built facility file does not hold is refused, by name."""
        view = served_tree(tmp_path, readings=["MADE:UP:CHANNEL"])
        with patch("osprey.utils.config.get_config_value", return_value=True):
            connector = MockConnector()
            await connector.connect(mock_config(view, response_delay_ms=0))

            result = await connector.read_channel("MADE:UP:CHANNEL")
            assert isinstance(result.value, float)

            with pytest.raises(ValueError) as refusal:
                await connector.read_channel("ANY:RANDOM:NAME")
            assert str(refusal.value) == "ANY:RANDOM:NAME is not in build/facility.json"

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_read_returns_tz_aware_timestamps(self, tmp_path):
        """Live-read timestamps carry an explicit offset (facility zone), not a
        naive datetime — guards the connector render sites against silent
        reversion to ``datetime.now()``."""
        view = served_tree(tmp_path, readings=["ANY:CHANNEL"])
        with patch("osprey.utils.config.get_config_value", return_value=True):
            connector = MockConnector()
            await connector.connect(mock_config(view, response_delay_ms=0))

            result = await connector.read_channel("ANY:CHANNEL")
            assert result.timestamp.tzinfo is not None
            assert result.timestamp.utcoffset() is not None
            assert result.metadata.timestamp.tzinfo is not None

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_read_carries_the_unit_of_the_channel_record(self, tmp_path):
        """A read's unit is the one the channel record states."""
        view = served_tree(
            tmp_path,
            readings=["BEAM:CURRENT", "MAGNET:VOLTAGE", "VACUUM:PRESSURE"],
            channels={
                "BEAM:CURRENT": {"unit": "mA"},
                "MAGNET:VOLTAGE": {"unit": "V"},
                "VACUUM:PRESSURE": {"unit": "Torr"},
            },
        )
        with patch("osprey.utils.config.get_config_value", return_value=True):
            connector = MockConnector()
            await connector.connect(mock_config(view, response_delay_ms=0))

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
    async def test_write_and_read_maintains_state(self, tmp_path):
        """Test that mock connector maintains state between writes and reads."""
        view = served_tree(tmp_path, ["TEST:SETPOINT:SP"])
        connector = MockConnector()
        with patch(
            "osprey.utils.config.get_config_value",
            side_effect=_config_with_writes_enabled,
        ):
            await connector.connect(mock_config(view, response_delay_ms=0))

            # Write a value
            channel = "TEST:SETPOINT:SP"
            test_value = 123.45
            result = await connector.write_channel(channel, test_value)
            assert result.outcome is WriteOutcome.CONFIRMED

            # Read it back
            result = await connector.read_channel(channel)
            assert abs(result.value - test_value) < 0.1  # Allow tiny variance

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_write_echoes_into_the_paired_readback(self, tmp_path):
        """A setpoint's write is echoed into the readback its pair names."""
        view = served_tree(tmp_path, {"MAGNET:CURRENT:SP": "MAGNET:CURRENT:RB"})
        connector = MockConnector()
        with patch(
            "osprey.utils.config.get_config_value",
            side_effect=_config_with_writes_enabled,
        ):
            await connector.connect(mock_config(view, response_delay_ms=0))

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
    async def test_write_to_a_readback_is_refused_before_any_put(self, tmp_path, monkeypatch):
        """A channel that is not a writable setpoint is refused, and nothing is put."""
        monkeypatch.setattr("osprey.utils.config.get_config_value", _config_with_writes_enabled)
        view = served_tree(tmp_path, {"MAGNET:CURRENT:SP": "MAGNET:CURRENT:RB"})
        connector = MockConnector()
        await connector.connect(mock_config(view, response_delay_ms=0))
        puts = []
        monkeypatch.setattr(connector, "_put", lambda *args: puts.append(args))

        result = await connector.write_channel("MAGNET:CURRENT:RB", 5.0)

        assert result.outcome is WriteOutcome.REFUSED
        assert "not a writable setpoint" in result.error_message
        assert puts == []

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_write_outside_the_facility_file_is_refused(self, tmp_path, monkeypatch):
        """A write to an address the facility file does not hold is refused by name."""
        monkeypatch.setattr("osprey.utils.config.get_config_value", _config_with_writes_enabled)
        connector = MockConnector()
        await connector.connect(mock_config(served_tree(tmp_path, ["A:SP"]), response_delay_ms=0))

        result = await connector.write_channel("B:SP", 1.0)

        assert result.outcome is WriteOutcome.REFUSED
        assert result.error_message == "B:SP is not in build/facility.json"

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_a_write_the_composite_rejects_is_refused_with_its_text(
        self, tmp_path, monkeypatch
    ):
        """The composite's refusal text is the refusal's message, verbatim."""
        monkeypatch.setattr("osprey.utils.config.get_config_value", _config_with_writes_enabled)
        connector = MockConnector()
        await connector.connect(mock_config(served_tree(tmp_path, ["A:SP"]), response_delay_ms=0))

        def rejecting_set(_values):
            raise ValueError("orbit does not close")

        monkeypatch.setattr(connector._composite, "set", rejecting_set)

        result = await connector.write_channel("A:SP", 1.0)

        assert result.outcome is WriteOutcome.REFUSED
        assert result.error_message == "orbit does not close"

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_write_disabled(self, tmp_path):
        """Test that writes are blocked via base class when config says false."""
        view = served_tree(tmp_path, ["TEST:PV"])
        connector = MockConnector()
        with patch("osprey.utils.config.get_config_value", return_value=False):
            await connector.connect(mock_config(view, response_delay_ms=0))

            result = await connector.write_channel("TEST:PV", 100.0)
            assert result.outcome is WriteOutcome.REFUSED

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_an_enum_channel_reads_its_option_index(self, tmp_path, monkeypatch):
        """A bool or enum channel reads back as its option index, its label beside it."""
        monkeypatch.setattr("osprey.utils.config.get_config_value", _config_with_writes_enabled)
        view = served_tree(
            tmp_path,
            ["MODE:SP"],
            channels={"MODE:SP": {"value_type": "enum", "options": ["OFF", "CW", "PULSED"]}},
        )
        connector = MockConnector()
        await connector.connect(mock_config(view, response_delay_ms=0))

        result = await connector.write_channel("MODE:SP", 2)
        reading = await connector.read_channel("MODE:SP")

        assert result.outcome is WriteOutcome.CONFIRMED
        assert reading.value == 2
        assert reading.metadata.enum_label == "PULSED"
        assert reading.metadata.enum_labels == ["OFF", "CW", "PULSED"]

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_a_subscription_fires_after_a_write(self, tmp_path, monkeypatch):
        """A write that changes a subscribed channel's held value fires its callback."""
        monkeypatch.setattr("osprey.utils.config.get_config_value", _config_with_writes_enabled)
        view = served_tree(tmp_path, {"MAGNET:CURRENT:SP": "MAGNET:CURRENT:RB"})
        connector = MockConnector()
        await connector.connect(mock_config(view, response_delay_ms=0))
        seen = []
        await connector.subscribe("MAGNET:CURRENT:RB", seen.append)

        await connector.write_channel("MAGNET:CURRENT:SP", 7.0)

        assert [reading.value for reading in seen] == [pytest.approx(7.0)]

        await connector.disconnect()

    @staticmethod
    def _ticking_tree(tmp_path, tick_s=None):
        """A tree with one noisy and one quiet reading, rendered with ``tick_s``."""
        view = served_tree(
            tmp_path,
            readings=["BEAM:NOISY", "BEAM:QUIET"],
            channels={
                "BEAM:NOISY": {"simulation": {"nominal": 1.0, "noise": 0.1}},
                "BEAM:QUIET": {"simulation": {"nominal": 2.0}},
            },
        )
        if tick_s is not None:
            (view.parent.parent / "config.yml").write_text(f"simulation:\n  tick_s: {tick_s}\n")
        return view

    @staticmethod
    async def _wait_for(predicate, timeout_s):
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while not predicate() and loop.time() < deadline:
            await asyncio.sleep(0.01)
        return predicate()

    @pytest.mark.asyncio
    async def test_a_tick_fires_a_moving_channel_with_no_write(self, tmp_path):
        """After a tick of ``simulation.tick_s`` a channel declaring motion fires."""
        connector = MockConnector()
        await connector.connect(
            mock_config(self._ticking_tree(tmp_path, 0.05), response_delay_ms=0)
        )
        seen = []
        await connector.subscribe("BEAM:NOISY", seen.append)

        # Two ticks of the default period would take 2 s; at 0.05 s they come well inside 1.5 s.
        assert await self._wait_for(lambda: len(seen) >= 2, timeout_s=1.5)
        assert all(reading.value is not None for reading in seen)

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_a_tick_leaves_a_quiet_channel_silent(self, tmp_path):
        """A channel with no motion and an unchanged held value never fires on a tick."""
        connector = MockConnector()
        await connector.connect(
            mock_config(self._ticking_tree(tmp_path, 0.05), response_delay_ms=0)
        )
        noisy, quiet = [], []
        await connector.subscribe("BEAM:NOISY", noisy.append)
        await connector.subscribe("BEAM:QUIET", quiet.append)

        assert await self._wait_for(lambda: len(noisy) >= 3, timeout_s=5.0)
        # The first tick records the quiet channel's held value; it fires only
        # when that value has no predecessor, never again after.
        fired_once = len(quiet)
        assert await self._wait_for(lambda: len(noisy) >= 6, timeout_s=5.0)
        assert len(quiet) == fired_once <= 1

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_the_tick_period_comes_from_the_rendered_config(self, tmp_path):
        """With no ``simulation.tick_s`` the default period applies: no tick in 0.3 s."""
        from osprey_connectors.simulation import DEFAULT_TICK_S

        assert DEFAULT_TICK_S >= 0.6
        connector = MockConnector()
        await connector.connect(mock_config(self._ticking_tree(tmp_path), response_delay_ms=0))
        seen = []
        await connector.subscribe("BEAM:NOISY", seen.append)

        await asyncio.sleep(0.3)

        assert seen == []

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect_stops_the_tick(self, tmp_path):
        """No callback fires after disconnect."""
        connector = MockConnector()
        await connector.connect(
            mock_config(self._ticking_tree(tmp_path, 0.05), response_delay_ms=0)
        )
        seen = []
        await connector.subscribe("BEAM:NOISY", seen.append)
        assert await self._wait_for(lambda: len(seen) >= 1, timeout_s=5.0)

        await connector.disconnect()
        fired = len(seen)
        await asyncio.sleep(0.3)

        assert len(seen) == fired

    @pytest.mark.asyncio
    async def test_read_multiple_channels(self, tmp_path):
        """Test reading multiple PVs concurrently."""
        view = served_tree(tmp_path, readings=["PV:1", "PV:2", "PV:3", "PV:4"])
        with patch("osprey.utils.config.get_config_value", return_value=True):
            connector = MockConnector()
            await connector.connect(mock_config(view, response_delay_ms=0))

            channels = ["PV:1", "PV:2", "PV:3", "PV:4"]
            results = await connector.read_multiple_channels(channels)

            assert len(results) == len(channels)
            for channel in channels:
                assert channel in results
                assert results[channel].value is not None

            await connector.disconnect()

    @staticmethod
    def _fail_read_of(connector, channel, error):
        real_read = connector.read_channel

        async def read_channel(channel_address, timeout=None):
            if channel_address == channel:
                raise error
            return await real_read(channel_address, timeout)

        connector.read_channel = read_channel

    @pytest.mark.asyncio
    async def test_read_multiple_channels_omits_a_failed_read(self, tmp_path):
        """A read that fails with an ordinary error is left out of the result."""
        view = served_tree(tmp_path, readings=["PV:1", "PV:2", "PV:3"])
        with patch("osprey.utils.config.get_config_value", return_value=True):
            connector = MockConnector()
            await connector.connect(mock_config(view, response_delay_ms=0))
            self._fail_read_of(connector, "PV:2", RuntimeError("read failed"))

            results = await connector.read_multiple_channels(["PV:1", "PV:2", "PV:3"])

            assert set(results) == {"PV:1", "PV:3"}
            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_read_multiple_channels_propagates_a_cancelled_read(self, tmp_path):
        """A cancelled read raises the cancellation instead of returning it as a value."""
        view = served_tree(tmp_path, readings=["PV:1", "PV:2", "PV:3"])
        with patch("osprey.utils.config.get_config_value", return_value=True):
            connector = MockConnector()
            await connector.connect(mock_config(view, response_delay_ms=0))
            self._fail_read_of(connector, "PV:2", asyncio.CancelledError())

            with pytest.raises(asyncio.CancelledError):
                await connector.read_multiple_channels(["PV:1", "PV:2", "PV:3"])
            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_validate_channel_is_membership(self, tmp_path):
        """A channel is valid exactly when the built facility file holds it."""
        view = served_tree(tmp_path, readings=["ANY:PV:NAME"])
        with patch("osprey.utils.config.get_config_value", return_value=True):
            connector = MockConnector()
            await connector.connect(mock_config(view, response_delay_ms=0))

            assert await connector.validate_channel("ANY:PV:NAME") is True
            assert await connector.validate_channel("RANDOM:CHANNEL") is False

            await connector.disconnect()

    @pytest.mark.asyncio
    async def test_metadata(self, tmp_path):
        """Metadata carries the channel record's unit and description."""
        view = served_tree(
            tmp_path,
            readings=["BEAM:CURRENT"],
            channels={"BEAM:CURRENT": {"unit": "mA", "description": "Stored beam current"}},
        )
        with patch("osprey.utils.config.get_config_value", return_value=True):
            connector = MockConnector()
            await connector.connect(mock_config(view, response_delay_ms=0))

            metadata = await connector.get_metadata("BEAM:CURRENT")
            assert metadata.units == "mA"
            assert metadata.description == "Stored beam current"

            await connector.disconnect()


#: Readbacks whose seeds move, so a window of their history varies.
_MOVING = {"simulation": {"nominal": 500.0, "noise": 1.0}}


def _archived_tree(tmp_path, *readings):
    """A served tree whose readings each carry a noisy seed."""
    return served_tree(tmp_path, readings=readings, channels=dict.fromkeys(readings, _MOVING))


class TestMockArchiverConnector:
    """Test MockArchiverConnector functionality."""

    @pytest.mark.asyncio
    async def test_connect_disconnect(self, tmp_path):
        """Test archiver connection and disconnection."""
        connector = MockArchiverConnector()
        config = mock_config(served_tree(tmp_path), sample_rate_hz=1.0)

        await connector.connect(config)
        assert connector._connected is True

        await connector.disconnect()
        assert connector._connected is False

    @pytest.mark.asyncio
    async def test_connect_without_a_built_view_is_refused(self, tmp_path, monkeypatch):
        """No view named and none beside a loaded config: connect says to build."""
        monkeypatch.chdir(tmp_path)
        connector = MockArchiverConnector()

        with pytest.raises(RuntimeError) as refusal:
            await connector.connect({})

        assert str(refusal.value) == NO_VIEW_MESSAGE

    @pytest.mark.asyncio
    @pytest.mark.parametrize("sample_rate_hz", [0, -1.0])
    async def test_connect_rejects_non_positive_sample_rate(self, sample_rate_hz):
        """A zero or negative rate would divide by zero later; connect refuses it."""
        connector = MockArchiverConnector()

        with pytest.raises(ValueError, match="sample_rate_hz must be > 0"):
            await connector.connect({"sample_rate_hz": sample_rate_hz})

    @pytest.mark.asyncio
    async def test_get_data_serves_the_view_and_refuses_any_other_name(self, tmp_path):
        """The view's channels are served; a name outside it is refused, by name."""
        channels = ["FAKE:PV:1", "RANDOM:PV:2", "ANY:NAME:3"]
        connector = MockArchiverConnector()
        await connector.connect(mock_config(_archived_tree(tmp_path, *channels)))

        start_date = datetime(2024, 1, 1, 0, 0, 0)
        end_date = datetime(2024, 1, 1, 1, 0, 0)

        df = await connector.get_data(channels=channels, start_date=start_date, end_date=end_date)

        assert df is not None
        assert len(df) > 0
        assert set(df["channel"]) == set(channels)

        with pytest.raises(ValueError) as refusal:
            await connector.get_data(
                channels=["NOT:SERVED"], start_date=start_date, end_date=end_date
            )
        assert str(refusal.value) == "NOT:SERVED is not in build/facility.json"

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_get_data_returns_dataframe(self, tmp_path):
        """Test that get_data returns the canonical long-format DataFrame."""
        connector = MockArchiverConnector()
        await connector.connect(mock_config(_archived_tree(tmp_path, "BEAM:CURRENT")))

        start_date = datetime(2024, 1, 1, 0, 0, 0)
        end_date = datetime(2024, 1, 1, 0, 10, 0)

        df = await connector.get_data(
            channels=["BEAM:CURRENT"], start_date=start_date, end_date=end_date, precision_ms=1000
        )

        import pandas as pd

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["timestamp", "channel", "value"]
        assert df["timestamp"].dtype == "datetime64[ns, UTC]"
        assert df["value"].dtype == "float64"
        assert (df["channel"] == "BEAM:CURRENT").all()

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_get_metadata(self, tmp_path):
        """Test getting archiver metadata."""
        connector = MockArchiverConnector()
        await connector.connect(mock_config(_archived_tree(tmp_path, "BEAM:CURRENT")))

        metadata = await connector.get_metadata("BEAM:CURRENT")
        assert metadata.channel == "BEAM:CURRENT"
        assert metadata.is_archived is True
        assert metadata.archival_start is not None

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_check_availability_is_membership(self, tmp_path):
        """A channel is available exactly when the simulator view holds it."""
        channels = ["PV:1", "PV:2", "PV:3"]
        connector = MockArchiverConnector()
        await connector.connect(mock_config(_archived_tree(tmp_path, *channels)))

        availability = await connector.check_availability([*channels, "PV:4"])

        assert len(availability) == len(channels) + 1
        for pv in channels:
            assert availability[pv] is True
        assert availability["PV:4"] is False

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_generated_time_series_has_variation(self, tmp_path):
        """Test that generated time series have realistic variation."""
        connector = MockArchiverConnector()
        await connector.connect(mock_config(_archived_tree(tmp_path, "BEAM:CURRENT")))

        start_date = datetime(2024, 1, 1, 0, 0, 0)
        end_date = datetime(2024, 1, 1, 1, 0, 0)

        df = await connector.get_data(
            channels=["BEAM:CURRENT"], start_date=start_date, end_date=end_date
        )

        # Check that values vary (not all the same)
        values = df.loc[df["channel"] == "BEAM:CURRENT", "value"].to_numpy()
        assert len(set(values)) > 1
        assert values.std() > 0

        await connector.disconnect()

    @pytest.mark.asyncio
    async def test_multi_pv_returns_independent_rows_per_channel(self, tmp_path):
        """Each channel contributes its own rows to the long frame."""
        connector = MockArchiverConnector()
        await connector.connect(
            mock_config(_archived_tree(tmp_path, "BEAM:CURRENT", "MAGNET:VOLTAGE"))
        )

        start_date = datetime(2024, 1, 1, 0, 0, 0)
        end_date = datetime(2024, 1, 1, 0, 1, 0)

        df = await connector.get_data(
            channels=["BEAM:CURRENT", "MAGNET:VOLTAGE"],
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
    async def test_processing_mean_aggregates_multiple_raw_samples(self, tmp_path):
        """A bin much wider than the data's spacing must average, not pass through."""
        connector = MockArchiverConnector()
        # Samples are a pure function of channel and timestamp, so the two
        # independent get_data() calls are comparable.
        await connector.connect(mock_config(_archived_tree(tmp_path, "BEAM:CURRENT")))

        # Both calls generate the same 10 points (the generator's forced
        # minimum) over this 10s window; the 60s mean bin forces every sample
        # into a single aggregation bin.
        start_date = datetime(2024, 1, 1, 0, 0, 0)
        end_date = datetime(2024, 1, 1, 0, 0, 10)
        pv = "BEAM:CURRENT"

        raw_df = await connector.get_data(
            channels=[pv],
            start_date=start_date,
            end_date=end_date,
            precision_ms=1_000,
            processing="raw",
        )
        mean_df = await connector.get_data(
            channels=[pv],
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
    async def test_processing_mean_bounded_when_point_cap_binds(self, tmp_path):
        """A window wide enough to hit the 10,000-point cap must not blow up on resample."""
        connector = MockArchiverConnector()
        await connector.connect(mock_config(_archived_tree(tmp_path, "BEAM:CURRENT")))

        start_date = datetime(2024, 1, 1)
        end_date = start_date + timedelta(days=7)

        df = await connector.get_data(
            channels=["BEAM:CURRENT"],
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
    """The mock's synthetic data must be reproducible within and across processes.

    Every sample is keyed by the channel's address and the epoch time, never by
    a global random state or a salted ``hash()``.
    """

    _WINDOW = (datetime(2024, 1, 15, 10, 0, 0), datetime(2024, 1, 15, 10, 5, 0))
    _CHANNELS = ("SR:UNKNOWN:PRESSURE", "SR:OTHER:PRESSURE", "SR:BPM01:POSITION")

    @pytest.fixture
    def view(self, tmp_path):
        return _archived_tree(tmp_path, *self._CHANNELS)

    async def _values(self, view, pv: str) -> list[float]:
        connector = MockArchiverConnector()
        await connector.connect(mock_config(view))
        start, end = self._WINDOW
        df = await connector.get_data(channels=[pv], start_date=start, end_date=end)
        await connector.disconnect()
        return df["value"].tolist()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("pv", ["SR:UNKNOWN:PRESSURE", "SR:BPM01:POSITION"])
    async def test_same_pv_and_window_repeats_within_a_process(self, view, pv):
        assert await self._values(view, pv) == await self._values(view, pv)

    @pytest.mark.asyncio
    async def test_distinct_pvs_do_not_collide(self, view):
        """Reproducible must not mean identical across channels."""
        assert await self._values(view, "SR:UNKNOWN:PRESSURE") != await self._values(
            view, "SR:OTHER:PRESSURE"
        )

    def test_same_pv_and_window_repeats_across_processes(self, view):
        """Run it in fresh interpreters with different hash seeds.

        ``hash()`` is salted per process, so only fresh interpreters can catch
        a seed derived from it.
        """
        script = textwrap.dedent("""
            import asyncio, json, sys
            from datetime import datetime
            from osprey.connectors.archiver.mock_archiver_connector import (
                MockArchiverConnector,
            )

            async def main():
                c = MockArchiverConnector()
                await c.connect({"simulator_view": sys.argv[1]})
                df = await c.get_data(
                    channels=["SR:UNKNOWN:PRESSURE"],
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
            proc = subprocess.run(
                [sys.executable, "-c", script, str(view)],
                capture_output=True,
                text=True,
                env=env,
                check=True,
            )
            runs.append(json.loads(proc.stdout.strip().splitlines()[-1]))

        assert runs[0] == runs[1] == runs[2]
        assert len(runs[0]) > 0


class TestMockWriteConfirmationContract:
    """Mock write results carry one outcome word and the value observed.

    Consumers decide what happened from ``outcome`` and ``observed_value``,
    never by parsing the display-only ``notes``. The confirming re-read is
    ``_confirming_read``, not ``read_channel``: confirmation reports what the
    simulated control system holds, so it is the seam these tests patch to
    reach the failure paths a mock cannot produce on its own.
    """

    @staticmethod
    async def _connected_mock(monkeypatch, tmp_path, channels=None, **settings):
        """A connected mock with writes enabled for the whole test.

        The writes_enabled gate is re-read on every write, so the config patch
        has to outlive connect(). The tree serves the setpoints these tests
        write, and the readback one of them is paired with.
        """
        monkeypatch.setattr("osprey.utils.config.get_config_value", _config_with_writes_enabled)
        view = served_tree(
            tmp_path,
            {"MAGNET:CURRENT:SP": "MAGNET:CURRENT:RB", "TEST:CHANNEL:SP": None},
            channels=channels,
        )
        connector = MockConnector()
        await connector.connect(mock_config(view, response_delay_ms=0, **settings))
        return connector

    @staticmethod
    def _raising_read(message):
        async def _read(_channel_address):
            raise RuntimeError(message)

        return _read

    async def test_a_write_confirms_against_what_the_store_holds(self, monkeypatch, tmp_path):
        """A re-read holding the value sent is ``confirmed``, with no message."""
        connector = await self._connected_mock(monkeypatch, tmp_path)

        result = await connector.write_channel("TEST:CHANNEL:SP", 42.0)

        assert result.outcome is WriteOutcome.CONFIRMED
        assert result.observed_value == pytest.approx(42.0)
        assert result.error_message is None
        assert result.refusal_reason is None
        # Mock has no alarm metadata to report; "not reported" stays None.
        assert result.alarm_status is None
        assert result.alarm_severity is None

        await connector.disconnect()

    async def test_read_noise_does_not_manufacture_a_mismatch(self, monkeypatch, tmp_path):
        """A write confirms against what is held while the readback it echoes into moves.

        Noise is measurement, not storage: the readback's seed declares noise,
        so its ordinary read moves, and a write to its setpoint still confirms
        on every attempt.
        """
        connector = await self._connected_mock(
            monkeypatch, tmp_path, channels={"MAGNET:CURRENT:RB": {"simulation": {"noise": 0.5}}}
        )

        for _ in range(5):
            result = await connector.write_channel("MAGNET:CURRENT:SP", 42.0)
            assert result.outcome is WriteOutcome.CONFIRMED
            assert result.observed_value == pytest.approx(42.0)

        # The ordinary read path is untouched and still noisy.
        noisy = await connector.read_channel("MAGNET:CURRENT:RB")
        assert noisy.value != pytest.approx(42.0, abs=1e-9)

        await connector.disconnect()

    async def test_a_perturbed_store_value_is_a_mismatch_without_an_error_message(
        self, monkeypatch, tmp_path
    ):
        """A setpoint the machine did not keep is reported, not tolerated.

        Both numbers are already on the result, so ``error_message`` stays None
        — it is reserved for the outcomes that carry something the numbers
        cannot say.
        """
        connector = await self._connected_mock(monkeypatch, tmp_path)

        def _clamping_put(channel_address, _value):
            connector._composite.set({channel_address: 10.0})

        monkeypatch.setattr(connector, "_put", _clamping_put)

        result = await connector.write_channel("TEST:CHANNEL:SP", 42.0)

        assert result.outcome is WriteOutcome.MISMATCH
        assert result.observed_value == pytest.approx(10.0)
        assert result.value_written == pytest.approx(42.0)
        assert result.error_message is None

        await connector.disconnect()

    async def test_confirming_read_that_raises_is_unconfirmed(self, monkeypatch, tmp_path):
        """The value went out but what the channel holds is unknown."""
        connector = await self._connected_mock(monkeypatch, tmp_path)
        monkeypatch.setattr(connector, "_confirming_read", self._raising_read("CA disconnected"))

        result = await connector.write_channel("TEST:CHANNEL:SP", 42.0)

        assert result.outcome is WriteOutcome.UNCONFIRMED
        assert result.observed_value is None
        assert "CA disconnected" in result.error_message
        assert result.alarm_status is None
        assert result.alarm_severity is None

        await connector.disconnect()

    async def test_confirm_false_does_not_read(self, monkeypatch, tmp_path):
        """``unrequested`` is the fast path: a read that would raise is never issued."""
        connector = await self._connected_mock(monkeypatch, tmp_path)
        monkeypatch.setattr(connector, "_confirming_read", self._raising_read("must not be called"))

        result = await connector.write_channel("TEST:CHANNEL:SP", 42.0, confirm=False)

        assert result.outcome is WriteOutcome.UNREQUESTED
        assert result.observed_value is None
        assert result.error_message is None

        await connector.disconnect()

    async def test_a_value_the_store_cannot_hold_is_a_failed_write(self, monkeypatch, tmp_path):
        """The put itself failing is ``failed``: the control system did not take it."""
        connector = await self._connected_mock(monkeypatch, tmp_path)
        monkeypatch.setattr(connector, "_confirming_read", self._raising_read("must not be called"))

        result = await connector.write_channel("TEST:CHANNEL:SP", "not-a-number")

        assert result.outcome is WriteOutcome.FAILED
        assert result.observed_value is None
        assert result.error_message is not None

        await connector.disconnect()

    async def test_notes_text_does_not_change_the_outcome(self, monkeypatch, tmp_path):
        """Two confirming reads failing differently classify identically.

        The exception text flows into ``notes`` and ``error_message`` and
        nowhere else — the machine-readable verdict must be identical.
        """
        connector = await self._connected_mock(monkeypatch, tmp_path)

        monkeypatch.setattr(connector, "_confirming_read", self._raising_read("timeout after 3s"))
        first = await connector.write_channel("TEST:CHANNEL:SP", 42.0)

        monkeypatch.setattr(connector, "_confirming_read", self._raising_read("channel not found"))
        second = await connector.write_channel("TEST:CHANNEL:SP", 42.0)

        def structured(result):
            return (
                result.outcome,
                result.refusal_reason,
                result.observed_value,
                result.alarm_status,
                result.alarm_severity,
            )

        assert first.notes != second.notes, "notes should differ"
        assert structured(first) == structured(second)

        await connector.disconnect()

    async def test_write_is_echoed_into_the_readback_its_pair_names(self, monkeypatch, tmp_path):
        """A setpoint's write is echoed into the readback its pair names."""
        connector = await self._connected_mock(monkeypatch, tmp_path)

        await connector.write_channel("MAGNET:CURRENT:SP", 100.0)

        readback = await connector.read_channel("MAGNET:CURRENT:RB")
        assert readback.value == pytest.approx(100.0, abs=1.0)

        await connector.disconnect()
