"""
Unit tests for DOOCSConnector.

All tests mock doocs4py so no installed DOOCS environment is required.
"""

import sys
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from osprey.connectors.control_system.base import ChannelValue, ChannelWriteResult

# --------------------------------------------------------------------------------------
# Helpers to build mock doocs4py objects
# --------------------------------------------------------------------------------------

_EPOCH_S = 1_700_000_000  # arbitrary fixed timestamp
_EPOCH_US = 500_000

# Patch targets used in multiple test classes
_LIMITS_PATCH = "osprey.connectors.control_system.doocs_connector.LimitsValidator.from_config"
_TZ_PATCH = "osprey.connectors.control_system.doocs_connector.get_facility_timezone"


def _make_eq_data(value=42.0, macropulse=12345):
    """Return a mock EqData object as returned by doocs4py.get()."""
    ts = MagicMock()
    ts.get_seconds_and_microseconds_since_epoch.return_value = (_EPOCH_S, _EPOCH_US)

    eq = MagicMock()
    eq.get_data.return_value = value
    eq.macropulse = macropulse
    eq.timestamp = ts
    return eq


def _make_doocs4py(names_result=None, get_data_value=42.0):
    """Return a mock doocs4py module."""
    d = MagicMock()
    d.__version__ = "2.0.0"
    d.names.return_value = names_result or [("FACILITY", "XFEL")]
    d.get.return_value = _make_eq_data(get_data_value)
    d.set.return_value = None
    return d


# --------------------------------------------------------------------------------------
# Fixture: a fully connected DOOCSConnector with mocked dependencies
# --------------------------------------------------------------------------------------


def _writes_enabled(key, default=None):
    if key == "control_system.writes_enabled":
        return True
    if key == "control_system.write_verification.default_level":
        return "none"
    return default


@pytest.fixture
async def connector():
    """DOOCSConnector wired with a mock doocs4py, limits disabled, writes on."""
    mock_d4py = _make_doocs4py()

    with (
        patch.dict(sys.modules, {"doocs4py": mock_d4py}),
        patch(_LIMITS_PATCH, return_value=None),
        patch(_TZ_PATCH, return_value=UTC),
        patch("osprey.utils.config.get_config_value", side_effect=_writes_enabled),
    ):
        from osprey.connectors.control_system.doocs_connector import DOOCSConnector

        conn = DOOCSConnector()
        await conn.connect({})
        yield conn, mock_d4py
        await conn.disconnect()


# --------------------------------------------------------------------------------------
# connect / disconnect
# --------------------------------------------------------------------------------------


class TestConnect:
    async def test_connect_sets_connected(self):
        mock_d4py = _make_doocs4py()
        with (
            patch.dict(sys.modules, {"doocs4py": mock_d4py}),
            patch(_LIMITS_PATCH, return_value=None),
            patch(_TZ_PATCH, return_value=UTC),
            patch("osprey.utils.config.get_config_value", return_value=False),
        ):
            from osprey.connectors.control_system.doocs_connector import DOOCSConnector

            conn = DOOCSConnector()
            assert conn._connected is False
            await conn.connect({})
            assert conn._connected is True
            await conn.disconnect()

    async def test_connect_raises_import_error_without_doocs4py(self):
        with patch.dict(sys.modules, {"doocs4py": None}):
            from osprey.connectors.control_system.doocs_connector import DOOCSConnector

            conn = DOOCSConnector()
            with pytest.raises(ImportError, match="doocs4py"):
                await conn.connect({})

    async def test_connect_raises_on_ens_failure(self):
        mock_d4py = _make_doocs4py()
        mock_d4py.names.side_effect = RuntimeError("ENS unreachable")
        with (
            patch.dict(sys.modules, {"doocs4py": mock_d4py}),
            patch(_LIMITS_PATCH, return_value=None),
            patch("osprey.utils.config.get_config_value", return_value=False),
        ):
            from osprey.connectors.control_system.doocs_connector import DOOCSConnector

            conn = DOOCSConnector()
            with pytest.raises(Exception, match="ENS"):
                await conn.connect({})

    async def test_disconnect_clears_connected(self, connector):
        conn, _ = connector
        assert conn._connected is True
        await conn.disconnect()
        assert conn._connected is False


# --------------------------------------------------------------------------------------
# read_channel / _read_channel_sync
# --------------------------------------------------------------------------------------


class TestReadChannel:
    async def test_read_returns_channel_value(self, connector):
        conn, _ = connector
        result = await conn.read_channel("FAC/DEV/LOC/PROP")

        assert isinstance(result, ChannelValue)
        assert result.value == 42.0

    async def test_read_timestamp_is_datetime(self, connector):
        conn, _ = connector
        result = await conn.read_channel("FAC/DEV/LOC/PROP")

        assert isinstance(result.timestamp, datetime)
        expected_ts = _EPOCH_S + _EPOCH_US / 1e6
        assert result.timestamp == datetime.fromtimestamp(expected_ts, UTC)

    async def test_read_metadata_contains_macropulse(self, connector):
        conn, _ = connector
        result = await conn.read_channel("FAC/DEV/LOC/PROP")

        assert result.metadata.raw_metadata["macropulse"] == 12345

    async def test_read_calls_doocs_get(self, connector):
        conn, mock_d4py = connector
        await conn.read_channel("FAC/DEV/LOC/PROP")

        mock_d4py.get.assert_called_once_with("FAC/DEV/LOC/PROP")

    async def test_read_propagates_exception(self, connector):
        conn, mock_d4py = connector
        mock_d4py.get.side_effect = RuntimeError("channel not found")

        with pytest.raises(RuntimeError, match="channel not found"):
            await conn.read_channel("INVALID/ADDR")


# --------------------------------------------------------------------------------------
# write_channel
# --------------------------------------------------------------------------------------


class TestWriteChannel:
    async def test_write_none_verification_success(self, connector):
        conn, mock_d4py = connector
        result = await conn.write_channel("FAC/DEV/LOC/PROP", 10.0, verification_level="none")

        assert isinstance(result, ChannelWriteResult)
        assert result.success is True
        assert result.value_written == 10.0
        assert result.verification.level == "none"
        assert result.verification.verified is False
        mock_d4py.set.assert_called_once_with("FAC/DEV/LOC/PROP", 10.0)

    async def test_write_none_verification_set_failure(self, connector):
        conn, mock_d4py = connector
        mock_d4py.set.side_effect = RuntimeError("write failed")

        result = await conn.write_channel("FAC/DEV/LOC/PROP", 5.0, verification_level="none")

        assert result.success is False
        assert "write failed" in result.error_message or "FAC/DEV/LOC/PROP" in result.error_message

    async def test_write_readback_verified(self, connector):
        conn, mock_d4py = connector
        mock_d4py.set.return_value = None
        mock_d4py.get.return_value = _make_eq_data(value=10.0)

        result = await conn.write_channel(
            "FAC/DEV/LOC/PROP", 10.0, verification_level="readback", tolerance=0.1
        )

        assert result.success is True
        assert result.verification.level == "readback"
        assert result.verification.verified is True
        assert result.verification.readback_value == pytest.approx(10.0)
        assert result.verification.tolerance_used == 0.1

    async def test_write_readback_mismatch(self, connector):
        conn, mock_d4py = connector
        mock_d4py.set.return_value = None
        mock_d4py.get.return_value = _make_eq_data(value=99.0)

        result = await conn.write_channel(
            "FAC/DEV/LOC/PROP", 10.0, verification_level="readback", tolerance=0.1
        )

        assert result.success is True
        assert result.verification.verified is False

    async def test_write_callback_treated_as_readback(self, connector):
        conn, mock_d4py = connector
        mock_d4py.get.return_value = _make_eq_data(value=7.0)

        result = await conn.write_channel(
            "FAC/DEV/LOC/PROP", 7.0, verification_level="callback", tolerance=0.5
        )

        assert result.success is True
        assert result.verification.level == "readback"

    async def test_write_invalid_verification_level_raises(self, connector):
        conn, _ = connector
        with pytest.raises(ValueError, match="Invalid verification_level"):
            await conn.write_channel("FAC/DEV/LOC/PROP", 1.0, verification_level="bad")

    async def test_write_blocked_when_writes_disabled(self):
        mock_d4py = _make_doocs4py()
        with (
            patch.dict(sys.modules, {"doocs4py": mock_d4py}),
            patch(_LIMITS_PATCH, return_value=None),
            patch(_TZ_PATCH, return_value=UTC),
            patch("osprey.utils.config.get_config_value", return_value=False),
        ):
            from osprey.connectors.control_system.doocs_connector import DOOCSConnector

            conn = DOOCSConnector()
            await conn.connect({})
            result = await conn.write_channel("FAC/DEV/LOC/PROP", 1.0)
            await conn.disconnect()

        assert result.success is False
        assert "disabled" in result.error_message.lower()

    async def test_write_readback_failure_returns_success_with_unverified(self, connector):
        """Write succeeds but readback throws — success=True, verified=False."""
        conn, mock_d4py = connector
        mock_d4py.set.return_value = None
        mock_d4py.get.side_effect = RuntimeError("readback error")

        result = await conn.write_channel(
            "FAC/DEV/LOC/PROP", 10.0, verification_level="readback", tolerance=0.1
        )

        assert result.success is True
        assert result.verification.verified is False
        assert "Readback failed" in result.verification.notes


# --------------------------------------------------------------------------------------
# read_multiple_channels
# --------------------------------------------------------------------------------------


class TestReadMultipleChannels:
    async def test_reads_all_channels(self, connector):
        conn, _ = connector
        addresses = ["FAC/DEV/LOC/A", "FAC/DEV/LOC/B"]

        results = await conn.read_multiple_channels(addresses)

        assert set(results.keys()) == set(addresses)
        for v in results.values():
            assert isinstance(v, ChannelValue)

    async def test_failed_channels_excluded(self, connector):
        conn, mock_d4py = connector

        def _side_effect(address):
            if "BAD" in address:
                raise RuntimeError("bad channel")
            return _make_eq_data()

        mock_d4py.get.side_effect = _side_effect

        results = await conn.read_multiple_channels(["FAC/DEV/LOC/OK", "FAC/DEV/LOC/BAD"])

        assert "FAC/DEV/LOC/OK" in results
        assert "FAC/DEV/LOC/BAD" not in results


# --------------------------------------------------------------------------------------
# subscribe / unsubscribe
# --------------------------------------------------------------------------------------


class TestSubscribe:
    async def test_subscribe_returns_subscription_id(self, connector):
        conn, _ = connector
        cb = MagicMock()
        sub_id = await conn.subscribe("FAC/DEV/LOC/PROP", cb)

        assert isinstance(sub_id, str)
        assert "FAC/DEV/LOC/PROP" in sub_id

    async def test_subscribe_adds_to_subscriptions(self, connector):
        conn, _ = connector
        cb = MagicMock()
        sub_id = await conn.subscribe("FAC/DEV/LOC/PROP", cb)

        assert sub_id in conn._subscriptions

    async def test_unsubscribe_removes_subscription(self, connector):
        conn, mock_d4py = connector
        cb = MagicMock()
        sub_id = await conn.subscribe("FAC/DEV/LOC/PROP", cb)
        await conn.unsubscribe(sub_id)

        assert sub_id not in conn._subscriptions
        mock_d4py.unsubscribe.assert_called_once()

    async def test_unsubscribe_unknown_id_is_noop(self, connector):
        conn, mock_d4py = connector
        await conn.unsubscribe("nonexistent_id")
        mock_d4py.unsubscribe.assert_not_called()

    async def test_disconnect_unsubscribes_all(self, connector):
        conn, _ = connector
        cb = MagicMock()
        await conn.subscribe("FAC/DEV/LOC/A", cb)
        await conn.subscribe("FAC/DEV/LOC/B", cb)
        assert len(conn._subscriptions) == 2

        await conn.disconnect()

        assert len(conn._subscriptions) == 0


# --------------------------------------------------------------------------------------
# get_metadata / validate_channel
# --------------------------------------------------------------------------------------


class TestMetadataAndValidation:
    async def test_get_metadata_delegates_to_read(self, connector):
        conn, _ = connector
        meta = await conn.get_metadata("FAC/DEV/LOC/PROP")

        assert meta.raw_metadata["macropulse"] == 12345

    async def test_validate_channel_true_on_success(self, connector):
        conn, _ = connector
        assert await conn.validate_channel("FAC/DEV/LOC/PROP") is True

    async def test_validate_channel_false_on_error(self, connector):
        conn, mock_d4py = connector
        mock_d4py.get.side_effect = RuntimeError("no such channel")
        assert await conn.validate_channel("BAD/ADDR") is False
