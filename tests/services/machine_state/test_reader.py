"""Unit tests for MachineStateReader."""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from osprey.connectors.control_system.base import ChannelMetadata, ChannelValue
from osprey.services.machine_state.models import (
    ChannelDefinition,
    ChannelResult,
    MachineStateSnapshot,
)
from osprey.services.machine_state.reader import MachineStateReader

# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def mock_connector():
    """Create a mock ControlSystemConnector."""
    connector = AsyncMock()
    return connector


@pytest.fixture
def sample_channels_file(tmp_path):
    """Write a valid channel definitions file and return its path."""
    data = {
        "_comment": "Test channels",
        "_version": "1.0",
        "SR:CURRENT:RB": {"label": "Beam current", "group": "beam"},
        "VA:PRESSURE:01": {"label": "Vacuum sector 1", "group": "vacuum"},
        "RF:POWER:01": {},
    }
    path = tmp_path / "channels.json"
    path.write_text(json.dumps(data))
    return str(path)


# =========================================================================
# Channel definition loading
# =========================================================================


class TestLoadChannelDefs:
    """Tests for _load_channel_defs."""

    def test_load_valid_file(self, sample_channels_file, mock_connector):
        reader = MachineStateReader(sample_channels_file, mock_connector)
        defs, warnings = reader._load_channel_defs()

        assert len(defs) == 3
        assert "SR:CURRENT:RB" in defs
        assert defs["SR:CURRENT:RB"].label == "Beam current"
        assert defs["SR:CURRENT:RB"].group == "beam"
        assert defs["RF:POWER:01"].label == ""
        assert warnings == []

    def test_skip_metadata_keys(self, tmp_path, mock_connector):
        data = {
            "_comment": "skip me",
            "_version": "1.0",
            "_custom_meta": "also skip",
            "REAL:CHANNEL": {"label": "real"},
        }
        path = tmp_path / "ch.json"
        path.write_text(json.dumps(data))

        reader = MachineStateReader(str(path), mock_connector)
        defs, warnings = reader._load_channel_defs()

        assert list(defs.keys()) == ["REAL:CHANNEL"]
        assert warnings == []

    def test_missing_file_raises(self, mock_connector):
        reader = MachineStateReader("/nonexistent/path.json", mock_connector)
        with pytest.raises(FileNotFoundError):
            reader._load_channel_defs()

    def test_invalid_json_raises(self, tmp_path, mock_connector):
        path = tmp_path / "bad.json"
        path.write_text("{not valid json")

        reader = MachineStateReader(str(path), mock_connector)
        with pytest.raises(json.JSONDecodeError):
            reader._load_channel_defs()

    def test_non_dict_entry_produces_warning(self, tmp_path, mock_connector):
        data = {
            "GOOD:CH": {"label": "ok"},
            "BAD:CH": "this is not a dict",
        }
        path = tmp_path / "ch.json"
        path.write_text(json.dumps(data))

        reader = MachineStateReader(str(path), mock_connector)
        defs, warnings = reader._load_channel_defs()

        assert "GOOD:CH" in defs
        assert "BAD:CH" not in defs
        assert len(warnings) == 1
        assert "BAD:CH" in warnings[0]

    def test_empty_file(self, tmp_path, mock_connector):
        path = tmp_path / "empty.json"
        path.write_text("{}")

        reader = MachineStateReader(str(path), mock_connector)
        defs, warnings = reader._load_channel_defs()

        assert defs == {}
        assert warnings == []

    def test_non_dict_root_raises(self, tmp_path, mock_connector):
        path = tmp_path / "array.json"
        path.write_text('[{"address": "CH:01"}]')

        reader = MachineStateReader(str(path), mock_connector)
        with pytest.raises(ValueError, match="JSON object"):
            reader._load_channel_defs()


# =========================================================================
# Reading channels
# =========================================================================


class TestRead:
    """Tests for the async read() method."""

    @pytest.mark.asyncio
    async def test_all_channels_succeed(self, sample_channels_file, mock_connector):
        now = datetime.now(tz=UTC)
        mock_connector.read_multiple_channels.return_value = {
            "SR:CURRENT:RB": ChannelValue(
                value=500.0, timestamp=now, metadata=ChannelMetadata(units="mA")
            ),
            "VA:PRESSURE:01": ChannelValue(
                value=1e-9, timestamp=now, metadata=ChannelMetadata(units="Torr")
            ),
            "RF:POWER:01": ChannelValue(
                value=42.0, timestamp=now, metadata=ChannelMetadata(units="kW")
            ),
        }

        reader = MachineStateReader(sample_channels_file, mock_connector)
        snapshot = await reader.read()

        assert snapshot.channels_read == 3
        assert snapshot.channels_failed == 0
        assert snapshot.success_rate == 1.0
        assert snapshot.failed_channels() == []

        result = snapshot.channels["SR:CURRENT:RB"]
        assert result.ok is True
        assert result.value == 500.0
        assert result.units == "mA"
        assert result.label == "Beam current"

    @pytest.mark.asyncio
    async def test_partial_failure(self, sample_channels_file, mock_connector):
        now = datetime.now(tz=UTC)
        # Connector only returns 2 of 3 channels
        mock_connector.read_multiple_channels.return_value = {
            "SR:CURRENT:RB": ChannelValue(
                value=500.0, timestamp=now, metadata=ChannelMetadata(units="mA")
            ),
            "VA:PRESSURE:01": ChannelValue(
                value=1e-9, timestamp=now, metadata=ChannelMetadata(units="Torr")
            ),
        }

        reader = MachineStateReader(sample_channels_file, mock_connector)
        snapshot = await reader.read()

        assert snapshot.channels_read == 2
        assert snapshot.channels_failed == 1
        assert snapshot.channels["RF:POWER:01"].ok is False
        assert snapshot.channels["RF:POWER:01"].error is not None
        assert "RF:POWER:01" in snapshot.failed_channels()

    @pytest.mark.asyncio
    async def test_numpy_array_normalized(self, tmp_path, mock_connector):
        """Numpy-like arrays are converted via .tolist()."""
        data = {"NP:CH": {"label": "numpy channel"}}
        path = tmp_path / "ch.json"
        path.write_text(json.dumps(data))

        class FakeArray:
            def tolist(self):
                return [1.0, 2.0, 3.0]

        now = datetime.now(tz=UTC)
        mock_connector.read_multiple_channels.return_value = {
            "NP:CH": ChannelValue(value=FakeArray(), timestamp=now, metadata=ChannelMetadata()),
        }

        reader = MachineStateReader(str(path), mock_connector)
        snapshot = await reader.read()

        assert snapshot.channels["NP:CH"].value == [1.0, 2.0, 3.0]

    @pytest.mark.asyncio
    async def test_empty_channel_file_returns_empty_snapshot(self, tmp_path, mock_connector):
        path = tmp_path / "empty.json"
        path.write_text("{}")

        reader = MachineStateReader(str(path), mock_connector)
        snapshot = await reader.read()

        assert snapshot.channels == {}
        assert snapshot.channels_read == 0
        assert snapshot.channels_failed == 0
        mock_connector.read_multiple_channels.assert_not_called()

    @pytest.mark.asyncio
    async def test_connector_exception_marks_all_failed(self, sample_channels_file, mock_connector):
        mock_connector.read_multiple_channels.side_effect = RuntimeError("connection lost")

        reader = MachineStateReader(sample_channels_file, mock_connector)
        snapshot = await reader.read()

        assert snapshot.channels_read == 0
        assert snapshot.channels_failed == 3
        for result in snapshot.channels.values():
            assert result.ok is False
            assert "connection lost" in result.error


# =========================================================================
# from_config
# =========================================================================


class TestFromConfig:
    """Tests for the from_config classmethod."""

    def test_returns_none_when_not_configured(self, mock_connector):
        with patch("osprey.utils.config.get_config_value", side_effect=KeyError):
            # KeyError is caught — returns None
            reader = MachineStateReader.from_config(mock_connector)
            assert reader is None

    def test_returns_none_when_channels_file_is_none(self, mock_connector):
        with patch(
            "osprey.utils.config.get_config_value",
            return_value=None,
        ):
            reader = MachineStateReader.from_config(mock_connector)
            assert reader is None

    def test_resolves_relative_path(self, mock_connector):
        def fake_config(key, default=None):
            return {
                "machine_state.channels_file": "data/channels.json",
                "project_root": "/opt/osprey",
            }.get(key, default)

        with patch(
            "osprey.utils.config.get_config_value",
            side_effect=fake_config,
        ):
            reader = MachineStateReader.from_config(mock_connector)
            assert reader is not None
            assert reader._channels_path == "/opt/osprey/data/channels.json"


# =========================================================================
# Model tests
# =========================================================================


class TestModels:
    """Tests for dataclass models."""

    def test_channel_definition_defaults(self):
        defn = ChannelDefinition(address="CH:01")
        assert defn.label == ""
        assert defn.group == ""

    def test_channel_result_ok(self):
        result = ChannelResult(address="CH:01", ok=True, value=42.0, units="mA")
        assert result.error is None

    def test_snapshot_success_rate_all_ok(self):
        snap = MachineStateSnapshot(
            snapshot_time=datetime.now(tz=UTC),
            channels_read=10,
            channels_failed=0,
        )
        assert snap.success_rate == 1.0

    def test_snapshot_success_rate_partial(self):
        snap = MachineStateSnapshot(
            snapshot_time=datetime.now(tz=UTC),
            channels_read=7,
            channels_failed=3,
        )
        assert snap.success_rate == pytest.approx(0.7)

    def test_snapshot_success_rate_empty(self):
        snap = MachineStateSnapshot(snapshot_time=datetime.now(tz=UTC))
        assert snap.success_rate == 1.0

    def test_snapshot_failed_channels(self):
        snap = MachineStateSnapshot(
            snapshot_time=datetime.now(tz=UTC),
            channels={
                "OK:CH": ChannelResult(address="OK:CH", ok=True, value=1.0),
                "FAIL:CH": ChannelResult(address="FAIL:CH", ok=False, value=None, error="timeout"),
            },
            channels_read=1,
            channels_failed=1,
        )
        assert snap.failed_channels() == ["FAIL:CH"]
