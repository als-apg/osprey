"""
Unit tests for DOOCSConnector.

All tests mock doocs4py so no installed DOOCS environment is required.
"""

import asyncio
import sys
import threading
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

# How long an offload test waits on a threading.Event before giving up. It only
# elapses when the code under test is broken; a passing run never waits on it.
_OFFLOAD_CEILING_S = 5.0


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

    @pytest.mark.parametrize("readback", [99.0, 10.05], ids=["far", "rounded"])
    async def test_a_readback_that_differs_is_a_mismatch_with_no_message(self, connector, readback):
        """Both values are on the result, so there is nothing left to say.

        The ``rounded`` case pins that there is no configurable tolerance: a
        nudged setpoint is reported, not tolerated. DOOCS reads carry no alarm
        metadata, so the alarm fields stay unset.
        """
        conn, mock_d4py = connector
        mock_d4py.get.return_value = _make_eq_data(value=readback)

        result = await conn.write_channel("FAC/DEV/LOC/PROP", 10.0, confirm=True)

        assert result.value_written == 10.0
        assert _structured_write_facts(result) == (
            WriteOutcome.MISMATCH,
            readback,
            None,
            False,
            None,
            None,
        )
        assert str(readback) in result.notes
        assert "10.0" in result.notes

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


class TestNonBlockingOffload:
    """Limits validation and the send share ONE thread offload.

    A max_step check reads the property's present value with a blocking
    ``doocs4py.get()``, so running validation on the event loop would stall
    every other coroutine in the process for the length of that read.
    """

    async def test_validate_and_send_run_off_the_event_loop(self):
        """Validation and the send run on one thread, and it is not the loop's.

        A stand-in for ``max_step``'s blocking fresh read parks inside
        ``validate()`` until this test releases it. Regaining the loop while
        that call is still parked is possible only if it never ran there.
        The thread each fake records says the same thing a second way:
        ``validate()`` and ``doocs4py.set`` report one thread, and it is not
        the one the loop runs on.
        """
        loop_thread = threading.get_ident()
        entered = threading.Event()  # validate() has begun
        release = threading.Event()  # the test lets it finish
        finished = threading.Event()  # validate() has returned
        threads: dict[str, int] = {}

        def blocking_validate(_address, _value, *, read_current=None):  # noqa: ARG001 - the limits-validator interface names read_current
            threads["validate"] = threading.get_ident()
            entered.set()
            release.wait(_OFFLOAD_CEILING_S)
            finished.set()

        validator = _make_limits_validator(confirm=False)
        validator.validate = MagicMock(side_effect=blocking_validate)

        mock_d4py = _make_doocs4py()

        def record_set(_address, _value):
            threads["set"] = threading.get_ident()

        mock_d4py.set.side_effect = record_set

        with (
            patch.dict(sys.modules, {"doocs4py": mock_d4py}),
            patch(_LIMITS_PATCH, return_value=validator),
            patch(_TZ_PATCH, return_value=UTC),
            patch("osprey.utils.config.get_config_value", side_effect=_writes_enabled),
        ):
            from osprey.connectors.control_system.doocs_connector import DOOCSConnector

            conn = DOOCSConnector()
            await conn.connect({})

            write_task = asyncio.create_task(
                conn.write_channel("FAC/DEV/LOC/PROP", 10.0, confirm=False)
            )
            try:
                began = await asyncio.to_thread(entered.wait, _OFFLOAD_CEILING_S)
                still_parked = not finished.is_set()
            finally:
                release.set()
            result = await write_task
            await conn.disconnect()

        assert began, "validate() was never called"
        assert still_parked, "the loop was regained only after validate() returned"
        assert threads.keys() == {"validate", "set"}
        assert loop_thread not in threads.values()
        assert len(set(threads.values())) == 1, f"the offload was split: {threads}"
        # The check was still made, and the value still went out.
        validator.validate.assert_called_once()
        assert result.outcome is WriteOutcome.UNREQUESTED
        mock_d4py.set.assert_called_once_with("FAC/DEV/LOC/PROP", 10.0)

    async def test_a_limits_refusal_raised_in_the_offload_sends_nothing(self):
        """A refusal from the worker thread reaches the caller as itself.

        Validation and the send share one offload, so the refusal is raised off
        the loop and has a thread boundary to cross. It must arrive carrying its
        LIMITS meaning rather than flattened into a failed write, and the value
        it refused must never reach ``doocs4py.set``.
        """
        from osprey_connectors.errors import ChannelLimitsViolationError

        refusal = ChannelLimitsViolationError(
            "FAC/DEV/LOC/PROP", 10.0, "max_value", "above the configured ceiling"
        )
        validator = _make_limits_validator()
        validator.validate = MagicMock(side_effect=refusal)

        mock_d4py = _make_doocs4py()

        with (
            patch.dict(sys.modules, {"doocs4py": mock_d4py}),
            patch(_LIMITS_PATCH, return_value=validator),
            patch(_TZ_PATCH, return_value=UTC),
            patch("osprey.utils.config.get_config_value", side_effect=_writes_enabled),
        ):
            from osprey.connectors.control_system.doocs_connector import DOOCSConnector

            conn = DOOCSConnector()
            await conn.connect({})

            with pytest.raises(ChannelLimitsViolationError) as raised:
                await conn.write_channel("FAC/DEV/LOC/PROP", 10.0, confirm=False)

            await conn.disconnect()

        assert raised.value is refusal
        mock_d4py.set.assert_not_called()


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
    async def test_subscribe_returns_an_id_that_unsubscribe_releases(self, connector):
        conn, mock_d4py = connector
        mock_d4py.Address.side_effect = lambda address: f"<Address {address}>"
        cb = MagicMock()
        sub_id = await conn.subscribe("FAC/DEV/LOC/PROP", cb)

        assert isinstance(sub_id, str)
        assert "FAC/DEV/LOC/PROP" in sub_id
        # The id is registered: releasing it reaches the driver for that address.
        await conn.unsubscribe(sub_id)
        mock_d4py.unsubscribe.assert_called_once_with("<Address FAC/DEV/LOC/PROP>")

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
        conn, mock_d4py = connector
        mock_d4py.Address.side_effect = lambda address: f"<Address {address}>"
        cb = MagicMock()
        await conn.subscribe("FAC/DEV/LOC/A", cb)
        await conn.subscribe("FAC/DEV/LOC/B", cb)

        await conn.disconnect()

        assert mock_d4py.unsubscribe.call_count == 2
        assert {c.args[0] for c in mock_d4py.unsubscribe.call_args_list} == {
            "<Address FAC/DEV/LOC/A>",
            "<Address FAC/DEV/LOC/B>",
        }
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
