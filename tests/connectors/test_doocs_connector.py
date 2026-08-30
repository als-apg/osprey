"""
Unit tests for DOOCSConnector.

All tests mock doocs4py so no installed DOOCS environment is required.
"""

import sys
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from osprey.connectors.control_system.base import (
    ChannelValue,
    ChannelWriteResult,
    WriteOutcome,
)

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


def _structured_write_facts(result):
    """The machine-readable half of a write result — the free text left out.

    ``notes`` and the wording of ``error_message`` are display text. What a
    consumer branches on is the outcome, what the property was seen to hold,
    the alarm fields, and whether a message is carried at all — the
    ``error_message`` iff-rule, not its sentence.
    """
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


async def _write_with_validator(validator, value=10.0, readback=10.0, **kwargs):
    """Run one write against a connector whose limits validator is ``validator``."""
    mock_d4py = _make_doocs4py()
    mock_d4py.get.return_value = _make_eq_data(value=readback)

    with (
        patch.dict(sys.modules, {"doocs4py": mock_d4py}),
        patch(_LIMITS_PATCH, return_value=validator),
        patch(_TZ_PATCH, return_value=UTC),
        patch("osprey.utils.config.get_config_value", side_effect=_writes_enabled),
    ):
        from osprey.connectors.control_system.doocs_connector import DOOCSConnector

        conn = DOOCSConnector()
        await conn.connect({})
        result = await conn.write_channel("FAC/DEV/LOC/PROP", value, **kwargs)
        await conn.disconnect()

    return result


class TestWriteChannel:
    """One confirm flow: send the value, then re-read it unless asked not to."""

    async def test_confirmed_write_reports_what_the_property_holds(self, connector):
        conn, mock_d4py = connector
        mock_d4py.get.return_value = _make_eq_data(value=10.0)

        result = await conn.write_channel("FAC/DEV/LOC/PROP", 10.0, confirm=True)

        assert isinstance(result, ChannelWriteResult)
        assert result.outcome is WriteOutcome.CONFIRMED
        assert result.value_written == 10.0
        assert result.observed_value == pytest.approx(10.0)
        assert result.error_message is None
        mock_d4py.set.assert_called_once_with("FAC/DEV/LOC/PROP", 10.0)

    async def test_confirm_false_is_unrequested_and_reads_nothing(self, connector):
        conn, mock_d4py = connector

        result = await conn.write_channel("FAC/DEV/LOC/PROP", 10.0, confirm=False)

        assert result.outcome is WriteOutcome.UNREQUESTED
        assert result.observed_value is None
        assert result.error_message is None
        mock_d4py.set.assert_called_once_with("FAC/DEV/LOC/PROP", 10.0)
        mock_d4py.get.assert_not_called()

    async def test_failed_set_is_failed_and_never_reads_back(self, connector):
        conn, mock_d4py = connector
        mock_d4py.set.side_effect = RuntimeError("write failed")

        result = await conn.write_channel("FAC/DEV/LOC/PROP", 5.0, confirm=True)

        assert result.outcome is WriteOutcome.FAILED
        assert "write failed" in result.error_message
        assert "FAC/DEV/LOC/PROP" in result.error_message
        assert result.observed_value is None
        # Nothing was taken, so there is nothing to confirm.
        mock_d4py.get.assert_not_called()

    async def test_read_that_raises_is_unconfirmed(self, connector):
        conn, mock_d4py = connector
        mock_d4py.get.side_effect = RuntimeError("readback error")

        result = await conn.write_channel("FAC/DEV/LOC/PROP", 10.0, confirm=True)

        assert result.outcome is WriteOutcome.UNCONFIRMED
        assert "readback error" in result.error_message
        assert result.observed_value is None

    async def test_mismatch_carries_both_values_and_no_message(self, connector):
        conn, mock_d4py = connector
        mock_d4py.get.return_value = _make_eq_data(value=99.0)

        result = await conn.write_channel("FAC/DEV/LOC/PROP", 10.0, confirm=True)

        assert result.outcome is WriteOutcome.MISMATCH
        assert result.value_written == 10.0
        assert result.observed_value == pytest.approx(99.0)
        # Both numbers are on the result; there is nothing left to say.
        assert result.error_message is None

    async def test_a_rounded_setpoint_is_a_mismatch_not_a_tolerated_write(self, connector):
        """There is no configurable tolerance: a nudged setpoint is reported."""
        conn, mock_d4py = connector
        mock_d4py.get.return_value = _make_eq_data(value=10.05)

        result = await conn.write_channel("FAC/DEV/LOC/PROP", 10.0, confirm=True)

        assert result.outcome is WriteOutcome.MISMATCH
        assert result.observed_value == pytest.approx(10.05)

    async def test_write_refused_when_writes_disabled(self):
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

        assert result.outcome is WriteOutcome.REFUSED
        assert result.refusal_reason == "WRITES_DISABLED"
        assert "disabled" in result.error_message.lower()
        mock_d4py.set.assert_not_called()

    async def test_doocs_never_reports_alarm_state(self, connector):
        """DOOCS reads carry no alarm metadata, so the fields stay unset."""
        conn, mock_d4py = connector
        mock_d4py.get.return_value = _make_eq_data(value=10.0)

        result = await conn.write_channel("FAC/DEV/LOC/PROP", 10.0, confirm=True)

        assert result.alarm_status is None
        assert result.alarm_severity is None


class TestConfirmResolution:
    """An omitted ``confirm`` is policy; an explicit one is an answer."""

    async def test_omitted_confirm_follows_the_channel_policy_when_true(self):
        validator = _make_limits_validator(confirm=True)

        result = await _write_with_validator(validator)

        assert result.outcome is WriteOutcome.CONFIRMED
        validator.resolve_confirm.assert_called_once_with("FAC/DEV/LOC/PROP")

    async def test_omitted_confirm_follows_the_channel_policy_when_false(self):
        validator = _make_limits_validator(confirm=False)

        result = await _write_with_validator(validator)

        assert result.outcome is WriteOutcome.UNREQUESTED
        validator.resolve_confirm.assert_called_once_with("FAC/DEV/LOC/PROP")

    async def test_explicit_confirm_false_is_not_resolved_away(self):
        """``confirm=False`` is an answer — the policy must not overrule it."""
        validator = _make_limits_validator(confirm=True)

        result = await _write_with_validator(validator, confirm=False)

        assert result.outcome is WriteOutcome.UNREQUESTED
        validator.resolve_confirm.assert_not_called()

    async def test_explicit_confirm_true_is_not_resolved_away(self):
        validator = _make_limits_validator(confirm=False)

        result = await _write_with_validator(validator, confirm=True)

        assert result.outcome is WriteOutcome.CONFIRMED
        validator.resolve_confirm.assert_not_called()

    async def test_no_limits_validator_confirms_by_default(self, connector):
        """Limits checking off means no policy to read — the fleet default confirms."""
        conn, mock_d4py = connector
        mock_d4py.get.return_value = _make_eq_data(value=10.0)

        result = await conn.write_channel("FAC/DEV/LOC/PROP", 10.0)

        assert result.outcome is WriteOutcome.CONFIRMED
        mock_d4py.get.assert_called_once_with("FAC/DEV/LOC/PROP")


class _Incomparable:
    """A readback whose equality test raises — nothing sensible to compare."""

    def __eq__(self, other):
        raise TypeError("no comparison defined")

    __hash__ = object.__hash__


class TestNonNumericReadback:
    """A non-numeric property confirms by equality, and never by fabrication.

    ``observed_value`` holds whatever the property reads back — a string, a
    sequence, an object — in the type the channel holds; ``observed_number``
    narrows it to a float only where that means something.
    """

    async def test_matching_string_readback_confirms(self, connector):
        conn, mock_d4py = connector
        mock_d4py.get.return_value = _make_eq_data(value="DESIRED")

        result = await conn.write_channel("FAC/DEV/LOC/PROP", "DESIRED", confirm=True)

        assert result.outcome is WriteOutcome.CONFIRMED
        assert result.observed_value == "DESIRED"
        assert result.observed_number is None
        assert result.error_message is None

    async def test_differing_string_readback_is_a_mismatch(self, connector):
        """The read worked and disagreed — that is a mismatch, not an unknown."""
        conn, mock_d4py = connector
        mock_d4py.get.return_value = _make_eq_data(value="OTHER")

        result = await conn.write_channel("FAC/DEV/LOC/PROP", "DESIRED", confirm=True)

        assert result.outcome is WriteOutcome.MISMATCH
        assert result.observed_value == "OTHER"
        assert result.error_message is None

    async def test_sequence_readback_confirms_elementwise(self, connector):
        conn, mock_d4py = connector
        mock_d4py.get.return_value = _make_eq_data(value=[1, 2, 3])

        result = await conn.write_channel("FAC/DEV/LOC/PROP", [1, 2, 3], confirm=True)

        assert result.outcome is WriteOutcome.CONFIRMED
        assert result.observed_value == [1, 2, 3]

    async def test_sequence_mismatch_is_a_mismatch(self, connector):
        conn, mock_d4py = connector
        mock_d4py.get.return_value = _make_eq_data(value=[1, 2, 4])

        result = await conn.write_channel("FAC/DEV/LOC/PROP", [1, 2, 3], confirm=True)

        assert result.outcome is WriteOutcome.MISMATCH
        assert result.observed_value == [1, 2, 4]
        assert result.error_message is None

    async def test_array_readback_confirms_elementwise(self, connector):
        np = pytest.importorskip("numpy")
        conn, mock_d4py = connector
        mock_d4py.get.return_value = _make_eq_data(value=np.array([1.0, 2.0]))

        result = await conn.write_channel("FAC/DEV/LOC/PROP", np.array([1.0, 2.0]), confirm=True)

        assert result.outcome is WriteOutcome.CONFIRMED

    async def test_incomparable_readback_is_a_mismatch(self, connector):
        """A comparison that raises is not a match, and not a failed read."""
        conn, mock_d4py = connector
        observed = _Incomparable()
        mock_d4py.get.return_value = _make_eq_data(value=observed)

        result = await conn.write_channel("FAC/DEV/LOC/PROP", 10.0, confirm=True)

        assert result.outcome is WriteOutcome.MISMATCH
        assert result.observed_value is observed
        # The read itself worked; only the comparison has no meaning.
        assert result.error_message is None

    async def test_numeric_readback_for_a_non_numeric_setpoint_is_a_mismatch(self, connector):
        """A string setpoint read back as a number disagrees — it is not unknown."""
        conn, mock_d4py = connector
        mock_d4py.get.return_value = _make_eq_data(value=1.0)

        result = await conn.write_channel("FAC/DEV/LOC/PROP", "ON", confirm=True)

        assert result.outcome is WriteOutcome.MISMATCH
        assert result.observed_value == pytest.approx(1.0)
        assert result.error_message is None


class TestWriteTextIsDisplayOnly:
    """``notes`` and the message wording never carry the classification."""

    async def test_message_text_does_not_change_the_structured_facts(self, connector):
        conn, mock_d4py = connector

        results = []
        for message in ("readback error", "an entirely different failure text"):
            mock_d4py.get.side_effect = RuntimeError(message)
            results.append(await conn.write_channel("FAC/DEV/LOC/PROP", 10.0, confirm=True))

        first, second = results
        assert first.error_message != second.error_message
        assert _structured_write_facts(first) == _structured_write_facts(second)

    async def test_a_mismatch_says_both_values_without_an_error_message(self, connector):
        conn, mock_d4py = connector
        mock_d4py.get.return_value = _make_eq_data(value=99.0)

        result = await conn.write_channel("FAC/DEV/LOC/PROP", 10.0, confirm=True)

        assert "99.0" in result.notes
        assert "10.0" in result.notes
        assert _structured_write_facts(result) == (
            WriteOutcome.MISMATCH,
            99.0,
            None,
            False,
            None,
            None,
        )


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
