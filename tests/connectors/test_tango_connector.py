"""
Unit tests for TangoConnector.

All tests mock PyTango (the ``tango`` module) so no installed TANGO
environment is required — the same seam the cross-connector parity suite
drives, and the same reason registration stays safe on machines without
PyTango.
"""

import asyncio
import sys
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from osprey.connectors.control_system.base import (
    ChannelValue,
    ChannelWriteResult,
    WriteOutcome,
)
from osprey.connectors.control_system.tango_connector import (
    _quality_fields,
    _split_address,
)

# --------------------------------------------------------------------------------------
# Helpers to build mock PyTango objects
# --------------------------------------------------------------------------------------

_EPOCH_S = 1_700_000_000  # arbitrary fixed timestamp
_EPOCH_US = 500_000

# Patch targets used in multiple test classes
_LIMITS_PATCH = "osprey.connectors.control_system.tango_connector.LimitsValidator.from_config"
_TZ_PATCH = "osprey.connectors.control_system.tango_connector.get_facility_timezone"

_ADDRESS = "sr/power_supply/ps01/Current"


class _Quality:
    """Stand-in for ``tango.AttrQuality`` members — an object with a name."""

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name


def _make_attribute(value=42.0, quality="ATTR_VALID", attr_type="DevDouble"):
    """Return a mock DeviceAttribute as returned by ``read_attribute()``."""
    time_val = MagicMock()
    time_val.tv_sec = _EPOCH_S
    time_val.tv_usec = _EPOCH_US

    attr = MagicMock()
    attr.value = value
    attr.quality = _Quality(quality) if quality is not None else None
    attr.time = time_val
    attr.type = attr_type
    return attr


def _make_proxy(read_value=42.0, quality="ATTR_VALID", attr_type="DevDouble"):
    """Return a mock DeviceProxy."""
    proxy = MagicMock()
    proxy.read_attribute.return_value = _make_attribute(read_value, quality, attr_type)
    proxy.write_attribute.return_value = None
    return proxy


def _make_tango(proxy=None):
    """Return a mock ``tango`` module serving one DeviceProxy."""
    t = MagicMock()
    t.__version__ = "10.0.0"
    t.DeviceProxy.return_value = proxy if proxy is not None else _make_proxy()
    database = MagicMock()
    database.get_info.return_value = "TANGO Database sys/database/2"
    t.Database.return_value = database
    return t


def _structured_write_facts(result):
    """The machine-readable half of a write result — the free text left out."""
    return (
        result.outcome,
        result.observed_value,
        result.refusal_reason,
        result.error_message is not None,
        result.alarm_status,
        result.alarm_severity,
    )


def _make_limits_validator(confirm=True):
    """A limits validator that passes validation and reports a confirm policy."""
    validator = MagicMock()
    validator.validate.return_value = None
    validator.resolve_confirm.return_value = confirm
    return validator


def _writes_enabled(key, default=None):
    if key == "control_system.writes_enabled":
        return True
    return default


@pytest.fixture
async def connector():
    """TangoConnector wired with a mock tango module, limits disabled, writes on."""
    proxy = _make_proxy()
    mock_tango = _make_tango(proxy)

    with (
        patch.dict(sys.modules, {"tango": mock_tango}),
        patch(_LIMITS_PATCH, return_value=None),
        patch(_TZ_PATCH, return_value=UTC),
        patch("osprey.utils.config.get_config_value", side_effect=_writes_enabled),
    ):
        from osprey.connectors.control_system.tango_connector import TangoConnector

        conn = TangoConnector()
        await conn.connect({})
        yield conn, proxy
        await conn.disconnect()


# --------------------------------------------------------------------------------------
# Address parsing and quality mapping — the two pure seams
# --------------------------------------------------------------------------------------


class TestSplitAddress:
    def test_plain_address_splits_device_and_attribute(self):
        assert _split_address(_ADDRESS) == ("sr/power_supply/ps01", "Current")

    def test_database_prefix_stays_on_the_device(self):
        device, attribute = _split_address("tango://db:10000/sr/power_supply/ps01/Current")
        assert device == "tango://db:10000/sr/power_supply/ps01"
        assert attribute == "Current"

    @pytest.mark.parametrize(
        "address",
        [
            "sr/power_supply/ps01",  # no attribute
            "sr/power_supply/ps01/Current/extra",  # too many segments
            "sr//ps01/Current",  # empty segment
            "tango://db:10000",  # prefix with no device
            "",
        ],
    )
    def test_malformed_addresses_raise_value_error(self, address):
        with pytest.raises(ValueError, match="TANGO channel address"):
            _split_address(address)


class TestQualityFields:
    @pytest.mark.parametrize(
        ("quality", "expected"),
        [
            ("ATTR_VALID", ("NO_ALARM", 0)),
            ("ATTR_CHANGING", ("CHANGING", 0)),
            ("ATTR_WARNING", ("WARNING", 1)),
            ("ATTR_ALARM", ("ALARM", 2)),
            ("ATTR_INVALID", ("INVALID", 3)),
        ],
    )
    def test_known_qualities_map_to_alarm_fields(self, quality, expected):
        assert _quality_fields(_Quality(quality)) == expected

    def test_none_is_not_reported(self):
        """``None`` means "not reported" — distinct from a healthy reading."""
        assert _quality_fields(None) == (None, None)

    def test_unknown_quality_is_named_unknown_with_no_severity(self):
        assert _quality_fields(_Quality("ATTR_SOMETHING_NEW")) == ("UNKNOWN", None)


# --------------------------------------------------------------------------------------
# connect / disconnect
# --------------------------------------------------------------------------------------


class TestConnect:
    async def test_connect_sets_connected(self):
        mock_tango = _make_tango()
        with (
            patch.dict(sys.modules, {"tango": mock_tango}),
            patch(_LIMITS_PATCH, return_value=None),
            patch(_TZ_PATCH, return_value=UTC),
            patch("osprey.utils.config.get_config_value", return_value=False),
        ):
            from osprey.connectors.control_system.tango_connector import TangoConnector

            conn = TangoConnector()
            await conn.connect({})
            assert conn._connected is True
            await conn.disconnect()

    async def test_connect_raises_import_error_without_pytango(self):
        with patch.dict(sys.modules, {"tango": None}):
            from osprey.connectors.control_system.tango_connector import TangoConnector

            conn = TangoConnector()
            with pytest.raises(ImportError, match="PyTango"):
                await conn.connect({})

    async def test_connect_raises_on_unreachable_database(self):
        mock_tango = _make_tango()
        mock_tango.Database.side_effect = RuntimeError("no database")
        with (
            patch.dict(sys.modules, {"tango": mock_tango}),
            patch(_LIMITS_PATCH, return_value=None),
            patch("osprey.utils.config.get_config_value", return_value=False),
        ):
            from osprey.connectors.control_system.tango_connector import TangoConnector

            conn = TangoConnector()
            with pytest.raises(ConnectionError, match="TANGO database"):
                await conn.connect({})

    async def test_explicit_tango_host_reaches_that_database(self):
        mock_tango = _make_tango()
        with (
            patch.dict(sys.modules, {"tango": mock_tango}),
            patch(_LIMITS_PATCH, return_value=None),
            patch("osprey.utils.config.get_config_value", return_value=False),
        ):
            from osprey.connectors.control_system.tango_connector import TangoConnector

            conn = TangoConnector()
            await conn.connect({"tango_host": "db.example.org:10000"})
            mock_tango.Database.assert_called_once_with("db.example.org", "10000")
            # ...and unprefixed device names are qualified with it.
            conn._get_proxy("sr/power_supply/ps01")
            mock_tango.DeviceProxy.assert_called_once_with(
                "tango://db.example.org:10000/sr/power_supply/ps01"
            )
            await conn.disconnect()


class TestDisconnect:
    async def test_disconnect_clears_connected_and_proxies(self, connector):
        conn, _proxy = connector
        await conn.read_channel(_ADDRESS)
        assert conn._proxies
        await conn.disconnect()
        assert conn._connected is False
        assert not conn._proxies


# --------------------------------------------------------------------------------------
# read_channel
# --------------------------------------------------------------------------------------


class TestReadChannel:
    async def test_read_returns_channel_value(self, connector):
        conn, _ = connector
        result = await conn.read_channel(_ADDRESS)
        assert isinstance(result, ChannelValue)
        assert result.value == 42.0

    async def test_read_timestamp_is_the_readings_own(self, connector):
        conn, _ = connector
        result = await conn.read_channel(_ADDRESS)
        expected = datetime.fromtimestamp(_EPOCH_S + _EPOCH_US / 1e6, UTC)
        assert result.timestamp == expected

    async def test_read_reports_alarm_state_from_quality(self, connector):
        conn, proxy = connector
        proxy.read_attribute.return_value = _make_attribute(quality="ATTR_ALARM")
        result = await conn.read_channel(_ADDRESS)
        assert result.metadata.alarm_status == "ALARM"
        assert result.metadata.raw_metadata["severity"] == 2

    async def test_healthy_reading_reports_no_alarm_not_nothing(self, connector):
        """A reported healthy reading stays distinct from "not reported"."""
        conn, _ = connector
        result = await conn.read_channel(_ADDRESS)
        assert result.metadata.alarm_status == "NO_ALARM"
        assert result.metadata.raw_metadata["severity"] == 0

    async def test_a_first_class_severity_field_is_populated_when_it_exists(self, connector):
        """``ChannelMetadata.alarm_severity`` is feature-detected, not assumed.

        The contract may grow the field independently of this connector; a
        metadata type that carries it gets the severity first-class, and one
        that does not still carries it in ``raw_metadata``.
        """
        from dataclasses import dataclass

        from osprey.connectors.control_system import tango_connector as mod
        from osprey.connectors.control_system.base import ChannelMetadata

        @dataclass
        class _MetadataWithSeverity(ChannelMetadata):
            alarm_severity: int | None = None

        conn, proxy = connector
        proxy.read_attribute.return_value = _make_attribute(quality="ATTR_ALARM")
        with patch.object(mod, "ChannelMetadata", _MetadataWithSeverity):
            result = await conn.read_channel(_ADDRESS)
        assert result.metadata.alarm_severity == 2
        assert result.metadata.raw_metadata["severity"] == 2

    async def test_read_reads_the_named_attribute_on_the_named_device(self, connector):
        conn, proxy = connector
        await conn.read_channel(_ADDRESS)
        proxy.read_attribute.assert_called_once_with("Current")

    async def test_the_device_proxy_is_cached_across_reads(self, connector):
        conn, _ = connector
        await conn.read_channel(_ADDRESS)
        await conn.read_channel("sr/power_supply/ps01/Voltage")
        assert list(conn._proxies) == ["sr/power_supply/ps01"]

    async def test_read_propagates_exception(self, connector):
        conn, proxy = connector
        proxy.read_attribute.side_effect = RuntimeError("device down")
        with pytest.raises(RuntimeError, match="device down"):
            await conn.read_channel(_ADDRESS)

    async def test_malformed_address_raises_before_any_device_call(self, connector):
        conn, proxy = connector
        with pytest.raises(ValueError):
            await conn.read_channel("not-an-address")
        proxy.read_attribute.assert_not_called()


class TestEnumReadings:
    async def test_enum_reading_resolves_its_label(self, connector):
        conn, proxy = connector
        proxy.read_attribute.return_value = _make_attribute(value=1, attr_type="DevEnum")
        info = MagicMock()
        info.enum_labels = ["Closed", "Open"]
        proxy.get_attribute_config.return_value = info

        result = await conn.read_channel(_ADDRESS)
        assert result.value == 1
        assert result.metadata.enum_labels == ["Closed", "Open"]
        assert result.metadata.enum_label == "Open"

    async def test_enum_labels_are_cached_per_attribute(self, connector):
        conn, proxy = connector
        proxy.read_attribute.return_value = _make_attribute(value=0, attr_type="DevEnum")
        info = MagicMock()
        info.enum_labels = ["Closed", "Open"]
        proxy.get_attribute_config.return_value = info

        await conn.read_channel(_ADDRESS)
        await conn.read_channel(_ADDRESS)
        proxy.get_attribute_config.assert_called_once_with("Current")

    async def test_label_fetch_failure_never_loses_the_reading(self, connector):
        conn, proxy = connector
        proxy.read_attribute.return_value = _make_attribute(value=1, attr_type="DevEnum")
        proxy.get_attribute_config.side_effect = RuntimeError("ctrl fetch failed")

        result = await conn.read_channel(_ADDRESS)
        assert result.value == 1
        assert result.metadata.enum_labels is None
        assert result.metadata.enum_label is None

    async def test_non_enum_reading_carries_no_labels(self, connector):
        conn, proxy = connector
        result = await conn.read_channel(_ADDRESS)
        assert result.metadata.enum_labels is None
        proxy.get_attribute_config.assert_not_called()


# --------------------------------------------------------------------------------------
# write_channel
# --------------------------------------------------------------------------------------


async def _write_with_validator(validator, value=10.0, readback=10.0, **kwargs):
    """Drive one write through a connector wired with the given validator."""
    proxy = _make_proxy(read_value=readback)
    mock_tango = _make_tango(proxy)
    with (
        patch.dict(sys.modules, {"tango": mock_tango}),
        patch(_LIMITS_PATCH, return_value=validator),
        patch(_TZ_PATCH, return_value=UTC),
        patch("osprey.utils.config.get_config_value", side_effect=_writes_enabled),
    ):
        from osprey.connectors.control_system.tango_connector import TangoConnector

        conn = TangoConnector()
        await conn.connect({})
        result = await conn.write_channel(_ADDRESS, value, **kwargs)
        await conn.disconnect()
        return result, proxy


class TestWriteChannel:
    async def test_confirmed_write_reports_what_the_attribute_holds(self, connector):
        conn, proxy = connector
        proxy.read_attribute.return_value = _make_attribute(value=10.0)
        result = await conn.write_channel(_ADDRESS, 10.0)
        assert isinstance(result, ChannelWriteResult)
        assert result.outcome is WriteOutcome.CONFIRMED
        assert result.observed_value == 10.0
        proxy.write_attribute.assert_called_once_with("Current", 10.0)

    async def test_confirm_false_is_unrequested_and_reads_nothing(self, connector):
        conn, proxy = connector
        result = await conn.write_channel(_ADDRESS, 10.0, confirm=False)
        assert result.outcome is WriteOutcome.UNREQUESTED
        assert result.observed_value is None
        proxy.read_attribute.assert_not_called()

    async def test_failed_put_is_failed_and_never_reads_back(self, connector):
        conn, proxy = connector
        proxy.write_attribute.side_effect = RuntimeError("write refused by device")
        result = await conn.write_channel(_ADDRESS, 10.0)
        assert result.outcome is WriteOutcome.FAILED
        assert result.error_message is not None
        proxy.read_attribute.assert_not_called()

    async def test_read_that_raises_is_unconfirmed(self, connector):
        conn, proxy = connector
        proxy.read_attribute.side_effect = RuntimeError("read exploded")
        result = await conn.write_channel(_ADDRESS, 10.0)
        assert result.outcome is WriteOutcome.UNCONFIRMED
        assert result.error_message is not None

    async def test_mismatch_carries_both_values_and_no_message(self, connector):
        conn, proxy = connector
        proxy.read_attribute.return_value = _make_attribute(value=9.7)
        result = await conn.write_channel(_ADDRESS, 10.0)
        assert result.outcome is WriteOutcome.MISMATCH
        assert result.value_written == 10.0
        assert result.observed_value == 9.7
        assert result.error_message is None

    async def test_a_ramping_readback_is_a_mismatch_not_a_softer_word(self, connector):
        """``ATTR_CHANGING`` reports beside the verdict, never instead of it.

        The write contract has no settling concept: a readback that does not
        yet hold the value sent is a ``mismatch``, and the quality that says
        the value is in motion rides on the alarm fields for the operator to
        interpret.
        """
        conn, proxy = connector
        proxy.read_attribute.return_value = _make_attribute(value=3.2, quality="ATTR_CHANGING")
        result = await conn.write_channel(_ADDRESS, 10.0)
        assert result.outcome is WriteOutcome.MISMATCH
        assert result.alarm_status == "CHANGING"
        assert result.alarm_severity == 0

    async def test_confirmed_write_in_alarm_stays_confirmed(self, connector):
        """Alarm state is reported with the write, never raised on."""
        conn, proxy = connector
        proxy.read_attribute.return_value = _make_attribute(value=10.0, quality="ATTR_ALARM")
        result = await conn.write_channel(_ADDRESS, 10.0)
        assert result.outcome is WriteOutcome.CONFIRMED
        assert result.alarm_status == "ALARM"
        assert result.alarm_severity == 2

    async def test_write_refused_when_writes_disabled(self):
        mock_tango = _make_tango()
        with (
            patch.dict(sys.modules, {"tango": mock_tango}),
            patch(_LIMITS_PATCH, return_value=None),
            patch(_TZ_PATCH, return_value=UTC),
            patch("osprey.utils.config.get_config_value", return_value=False),
        ):
            from osprey.connectors.control_system.tango_connector import TangoConnector

            conn = TangoConnector()
            await conn.connect({})
            result = await conn.write_channel(_ADDRESS, 10.0)
            assert result.outcome is WriteOutcome.REFUSED
            assert result.refusal_reason == "WRITES_DISABLED"
            mock_tango.DeviceProxy.return_value.write_attribute.assert_not_called()
            await conn.disconnect()

    async def test_enum_setpoint_written_as_text_confirms_by_its_label(self, connector):
        conn, proxy = connector
        proxy.read_attribute.return_value = _make_attribute(value=1, attr_type="DevEnum")
        info = MagicMock()
        info.enum_labels = ["Closed", "Open"]
        proxy.get_attribute_config.return_value = info

        result = await conn.write_channel(_ADDRESS, "Open")
        assert result.outcome is WriteOutcome.CONFIRMED


class TestConfirmResolution:
    """``confirm=None`` resolves from the limits database; an answer is kept."""

    async def test_omitted_confirm_follows_the_channel_policy_when_true(self):
        result, proxy = await _write_with_validator(_make_limits_validator(confirm=True))
        assert result.outcome is WriteOutcome.CONFIRMED
        assert proxy.read_attribute.call_count == 1

    async def test_omitted_confirm_follows_the_channel_policy_when_false(self):
        result, proxy = await _write_with_validator(_make_limits_validator(confirm=False))
        assert result.outcome is WriteOutcome.UNREQUESTED
        assert proxy.read_attribute.call_count == 0

    async def test_explicit_confirm_false_is_not_resolved_away(self):
        result, proxy = await _write_with_validator(
            _make_limits_validator(confirm=True), confirm=False
        )
        assert result.outcome is WriteOutcome.UNREQUESTED
        assert proxy.read_attribute.call_count == 0

    async def test_explicit_confirm_true_is_not_resolved_away(self):
        result, _ = await _write_with_validator(_make_limits_validator(confirm=False), confirm=True)
        assert result.outcome is WriteOutcome.CONFIRMED

    async def test_no_limits_validator_confirms_by_default(self, connector):
        conn, proxy = connector
        proxy.read_attribute.return_value = _make_attribute(value=10.0)
        result = await conn.write_channel(_ADDRESS, 10.0)
        assert result.outcome is WriteOutcome.CONFIRMED
        assert proxy.read_attribute.call_count == 1

    async def test_limits_violation_propagates(self):
        from osprey_connectors.errors import ChannelLimitsViolationError

        validator = MagicMock()
        validator.validate.side_effect = ChannelLimitsViolationError(
            _ADDRESS, 1e9, "range", "too big", max_value=200.0
        )
        with pytest.raises(ChannelLimitsViolationError):
            await _write_with_validator(validator, value=1e9)


class TestWriteTextIsDisplayOnly:
    async def test_message_text_does_not_change_the_structured_facts(self, connector):
        conn, proxy = connector
        proxy.read_attribute.return_value = _make_attribute(value=10.0)
        result = await conn.write_channel(_ADDRESS, 10.0)
        assert _structured_write_facts(result) == (
            WriteOutcome.CONFIRMED,
            10.0,
            None,
            False,
            "NO_ALARM",
            0,
        )


# --------------------------------------------------------------------------------------
# read_multiple_channels
# --------------------------------------------------------------------------------------


class TestReadMultipleChannels:
    async def test_reads_all_channels(self, connector):
        conn, _ = connector
        addresses = [_ADDRESS, "sr/power_supply/ps01/Voltage"]
        results = await conn.read_multiple_channels(addresses)
        assert set(results) == set(addresses)
        assert all(isinstance(v, ChannelValue) for v in results.values())

    async def test_a_failing_channel_is_dropped_not_fatal(self, connector):
        conn, proxy = connector

        def fail_voltage(attribute):
            if attribute == "Voltage":
                raise RuntimeError("bad attribute")
            return _make_attribute()

        proxy.read_attribute.side_effect = fail_voltage
        results = await conn.read_multiple_channels([_ADDRESS, "sr/power_supply/ps01/Voltage"])
        assert list(results) == [_ADDRESS]


# --------------------------------------------------------------------------------------
# subscribe / unsubscribe
# --------------------------------------------------------------------------------------


class TestSubscriptions:
    async def test_subscribe_registers_a_change_event(self, connector):
        conn, proxy = connector
        proxy.subscribe_event.return_value = 7
        sub_id = await conn.subscribe(_ADDRESS, lambda cv: None)
        assert sub_id in conn._subscriptions
        assert proxy.subscribe_event.call_args[0][0] == "Current"

    async def test_unsubscribe_releases_the_event(self, connector):
        conn, proxy = connector
        proxy.subscribe_event.return_value = 7
        sub_id = await conn.subscribe(_ADDRESS, lambda cv: None)
        await conn.unsubscribe(sub_id)
        proxy.unsubscribe_event.assert_called_once_with(7)
        assert sub_id not in conn._subscriptions

    async def test_event_callback_delivers_a_channel_value(self, connector):
        conn, proxy = connector
        proxy.subscribe_event.return_value = 7
        received: list[ChannelValue] = []
        await conn.subscribe(_ADDRESS, received.append)

        tango_callback = proxy.subscribe_event.call_args[0][2]
        event = MagicMock()
        event.err = False
        event.attr_value = _make_attribute(value=3.14, quality="ATTR_WARNING")
        tango_callback(event)
        # call_soon_threadsafe needs the loop to turn once
        await asyncio.sleep(0)

        assert len(received) == 1
        assert received[0].value == 3.14
        assert received[0].metadata.alarm_status == "WARNING"

    async def test_error_events_are_dropped(self, connector):
        conn, proxy = connector
        proxy.subscribe_event.return_value = 7
        received: list[ChannelValue] = []
        await conn.subscribe(_ADDRESS, received.append)

        event = MagicMock()
        event.err = True
        proxy.subscribe_event.call_args[0][2](event)
        await asyncio.sleep(0)
        assert received == []


# --------------------------------------------------------------------------------------
# get_metadata / validate_channel
# --------------------------------------------------------------------------------------


class TestGetMetadata:
    async def test_metadata_is_enriched_from_the_attribute_config(self, connector):
        conn, proxy = connector
        info = MagicMock()
        info.unit = "mA"
        info.description = "Main coil current"
        info.min_value = "0.0"
        info.max_value = "120.0"
        info.format = "%6.2f"
        info.writable = "READ_WRITE"
        proxy.get_attribute_config.return_value = info

        metadata = await conn.get_metadata(_ADDRESS)
        assert metadata.units == "mA"
        assert metadata.description == "Main coil current"
        # TANGO's min/max are write limits, not a display range: raw only.
        assert metadata.display_low is None
        assert metadata.display_high is None
        assert metadata.raw_metadata["min_value"] == "0.0"
        assert metadata.raw_metadata["max_value"] == "120.0"

    async def test_tangos_unitless_spellings_read_as_no_unit(self, connector):
        conn, proxy = connector
        info = MagicMock()
        info.unit = "No unit"
        info.description = "No description"
        proxy.get_attribute_config.return_value = info

        metadata = await conn.get_metadata(_ADDRESS)
        assert metadata.units == ""
        assert metadata.description is None

    async def test_config_fetch_failure_keeps_the_readings_metadata(self, connector):
        conn, proxy = connector
        proxy.get_attribute_config.side_effect = RuntimeError("config fetch failed")
        metadata = await conn.get_metadata(_ADDRESS)
        assert metadata.alarm_status == "NO_ALARM"


class TestValidateChannel:
    async def test_readable_channel_is_valid(self, connector):
        conn, _ = connector
        assert await conn.validate_channel(_ADDRESS) is True

    async def test_unreadable_channel_is_invalid(self, connector):
        conn, proxy = connector
        proxy.read_attribute.side_effect = RuntimeError("no such attribute")
        assert await conn.validate_channel(_ADDRESS) is False

    async def test_malformed_address_is_invalid_not_fatal(self, connector):
        conn, _ = connector
        assert await conn.validate_channel("not-an-address") is False
