"""
Unit tests for DOOCSConnector.

All tests mock doocs4py so no installed DOOCS environment is required.
"""

import asyncio
import sys
import threading
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from osprey_connectors.connector_types import DOOCS
from osprey_connectors.control_system._base import (
    ChannelValue,
    ChannelWriteResult,
    WriteOutcome,
)
from osprey_connectors.errors import ChannelUnreachableError, ConnectorUnavailableError
from osprey_connectors.ipc import _frames as frames
from osprey_connectors.refusals import refusal_phrase
from osprey_connectors_tests._config_trees import control_system_config, installed
from osprey_connectors_tests._stall import STALL_CEILING_S, Stall
from osprey_connectors_tests._write_fakes import (
    ARMED_SECTION,
    arm_switch,
    armed_context,
)

# --------------------------------------------------------------------------------------
# Helpers to build mock doocs4py objects
# --------------------------------------------------------------------------------------

_EPOCH_S = 1_700_000_000  # arbitrary fixed timestamp
_EPOCH_US = 500_000

# Patch targets used in multiple test classes
_LIMITS_PATCH = "osprey_connectors.control_system._limits_validator.LimitsValidator.from_config"
_TZ_PATCH = "osprey_connectors.config.facility_timezone"

# How long an offload test waits on a threading.Event before giving up. It only
# elapses when the code under test is broken; a passing run never waits on it.
_OFFLOAD_CEILING_S = 5.0


@pytest.fixture(autouse=True)
def _switch_on(monkeypatch, tmp_path):
    """The login's switch on for the simulator target, connectors stamped with it.

    With :func:`serve_writes_enabled` serving a write tier that allows, that is
    everything the reference monitor's verdict asks before a write reaches the
    transport; a test that disarms serves a tier that does not.
    """
    return arm_switch(monkeypatch, tmp_path / "armed-data")


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
    observed = result.observed
    alarm = None if observed is None else observed.alarm
    return (
        result.outcome,
        None if observed is None else observed.value,
        result.refusal_reason,
        result.error_message is not None,
        None if alarm is None else alarm.status,
        None if alarm is None else alarm.severity,
    )


def _make_limits_validator(confirm=True):
    """A limits validator that passes validation and reports a confirm policy."""
    validator = MagicMock()
    validator.validate.return_value = None
    validator.resolve_confirm.return_value = confirm
    return validator


@pytest.fixture
async def connector():
    """DOOCSConnector wired with a mock doocs4py, limits disabled, writes on."""
    mock_d4py = _make_doocs4py()

    with (
        patch.dict(sys.modules, {"doocs4py": mock_d4py}),
        patch(_LIMITS_PATCH, return_value=None),
        patch(_TZ_PATCH, return_value=UTC),
        installed(control_system_config(ARMED_SECTION)),
    ):
        from osprey_connectors.control_system._doocs_connector import DOOCSConnector

        conn = DOOCSConnector()
        await conn.connect(armed_context())
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
            installed(control_system_config({})),
        ):
            from osprey_connectors.control_system._doocs_connector import DOOCSConnector

            conn = DOOCSConnector()
            assert conn._connected is False
            await conn.connect(armed_context())
            assert conn._connected is True
            await conn.disconnect()

    async def test_connect_raises_import_error_without_doocs4py(self):
        with patch.dict(sys.modules, {"doocs4py": None}):
            from osprey_connectors.control_system._doocs_connector import DOOCSConnector

            conn = DOOCSConnector()
            with pytest.raises(ImportError, match="doocs4py"):
                await conn.connect(armed_context())

    async def test_connect_raises_on_ens_failure(self):
        mock_d4py = _make_doocs4py()
        mock_d4py.names.side_effect = RuntimeError("ENS unreachable")
        with (
            patch.dict(sys.modules, {"doocs4py": mock_d4py}),
            patch(_LIMITS_PATCH, return_value=None),
            installed(control_system_config({})),
        ):
            from osprey_connectors.control_system._doocs_connector import DOOCSConnector

            conn = DOOCSConnector()
            with pytest.raises(ConnectorUnavailableError, match="ENS") as raised:
                await conn.connect(armed_context())

        assert isinstance(raised.value.__cause__, RuntimeError)

    async def test_an_ens_that_does_not_answer_fails_connect_within_timeout_s(self):
        mock_d4py = _make_doocs4py()
        with (
            Stall() as names,
            patch.dict(sys.modules, {"doocs4py": mock_d4py}),
            patch(_LIMITS_PATCH, return_value=None),
            installed(control_system_config({})),
        ):
            mock_d4py.names.side_effect = names
            from osprey_connectors.control_system._doocs_connector import DOOCSConnector

            conn = DOOCSConnector()
            with pytest.raises(ConnectorUnavailableError, match=r"within 0\.2s") as raised:
                await conn.connect(armed_context(settings={"timeout_s": 0.2}))
            answered_first = not names.returned

        assert "ENS" in str(raised.value)
        assert answered_first, "connect answered only once the ENS lookup returned"
        assert conn._connected is False

    async def test_the_ens_probe_runs_off_the_event_loop(self):
        """Regaining the loop while the ENS lookup is still parked is possible only off the loop."""
        mock_d4py = _make_doocs4py()
        with (
            Stall(answer=[("FACILITY", "XFEL")]) as names,
            patch.dict(sys.modules, {"doocs4py": mock_d4py}),
            patch(_LIMITS_PATCH, return_value=None),
            installed(control_system_config({})),
        ):
            mock_d4py.names.side_effect = names
            from osprey_connectors.control_system._doocs_connector import DOOCSConnector

            conn = DOOCSConnector()
            connect_task = asyncio.create_task(conn.connect(armed_context()))
            began = await asyncio.to_thread(names.entered.wait, STALL_CEILING_S)
            still_parked = not names.returned
            names.release()
            await connect_task
            await conn.disconnect()

        assert began, "the ENS was never asked"
        assert still_parked, "the loop was regained only after the ENS lookup returned"

    async def test_timeout_s_defaults_to_five_seconds(self, connector):
        conn, _ = connector
        assert conn._timeout_s == 5.0

    @pytest.mark.parametrize("bad", [0, -1, "five", True, float("nan"), float("inf")])
    async def test_a_timeout_s_that_is_not_a_positive_number_is_refused(self, bad):
        mock_d4py = _make_doocs4py()
        with (
            patch.dict(sys.modules, {"doocs4py": mock_d4py}),
            patch(_LIMITS_PATCH, return_value=None),
            installed(control_system_config({})),
        ):
            from osprey_connectors.control_system._doocs_connector import DOOCSConnector

            conn = DOOCSConnector()
            with pytest.raises(ValueError, match="control_system.connector.doocs.timeout_s"):
                await conn.connect(armed_context(connector_type=DOOCS, settings={"timeout_s": bad}))

        assert conn._connected is False

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
        assert result.address == "FAC/DEV/LOC/PROP"
        assert result.value == 42.0

    async def test_a_reading_reports_no_alarm_and_no_label(self, connector):
        """DOOCS reports no alarm state: the reading's alarm is "not reported", never healthy."""
        conn, _ = connector
        result = await conn.read_channel("FAC/DEV/LOC/PROP")

        assert result.alarm is None
        assert result.enum_label is None

    async def test_read_sampled_at_is_the_readings_own_in_utc(self, connector):
        conn, _ = connector
        result = await conn.read_channel("FAC/DEV/LOC/PROP")

        expected_ts = _EPOCH_S + _EPOCH_US / 1e6
        assert result.sampled_at == datetime.fromtimestamp(expected_ts, UTC)
        assert result.sampled_at.utcoffset().total_seconds() == 0

    async def test_read_metadata_contains_macropulse(self, connector):
        conn, _ = connector
        result = await conn.read_channel("FAC/DEV/LOC/PROP")

        assert result.metadata.raw_metadata["macropulse"] == 12345

    async def test_read_calls_doocs_get(self, connector):
        conn, mock_d4py = connector
        await conn.read_channel("FAC/DEV/LOC/PROP")

        mock_d4py.get.assert_called_once_with("FAC/DEV/LOC/PROP")

    async def test_a_failed_get_is_an_unreachable_channel(self, connector):
        conn, mock_d4py = connector
        mock_d4py.get.side_effect = RuntimeError("channel not found")

        with pytest.raises(ChannelUnreachableError, match="channel not found") as raised:
            await conn.read_channel("INVALID/ADDR")

        assert raised.value.address == "INVALID/ADDR"
        assert "INVALID/ADDR" in str(raised.value)
        assert isinstance(raised.value.__cause__, RuntimeError)

    async def test_read_metadata_names_the_value_type(self, connector):
        conn, mock_d4py = connector

        scalar = await conn.read_channel("FAC/DEV/LOC/PROP")
        mock_d4py.get.return_value = _make_eq_data(value=np.array([1.0, 2.0]))
        array = await conn.read_channel("FAC/DEV/LOC/PROP")

        assert scalar.metadata.raw_metadata["type"] == "float"
        assert array.metadata.raw_metadata["type"] == "ndarray"

    async def test_a_reading_round_trips_through_an_ipc_frame(self, connector):
        conn, _ = connector
        value = await conn.read_channel("FAC/DEV/LOC/PROP")

        reader = frames.FrameReader()
        decoded = reader.feed(frames.encode_result("req", value))

        assert len(decoded) == 1
        assert isinstance(decoded[0], frames.ResultFrame)
        assert decoded[0].value == value


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
        installed(control_system_config(ARMED_SECTION)),
    ):
        from osprey_connectors.control_system._doocs_connector import DOOCSConnector

        conn = DOOCSConnector()
        await conn.connect(armed_context())
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
        assert result.observed.value == pytest.approx(10.0)
        assert result.error_message is None
        mock_d4py.set.assert_called_once_with("FAC/DEV/LOC/PROP", 10.0)

    async def test_confirm_false_is_unrequested_and_reads_nothing(self, connector):
        conn, mock_d4py = connector

        result = await conn.write_channel("FAC/DEV/LOC/PROP", 10.0, confirm=False)

        assert result.outcome is WriteOutcome.UNREQUESTED
        assert result.observed is None
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
        assert result.observed is None
        # Nothing was taken, so there is nothing to confirm.
        mock_d4py.get.assert_not_called()

    async def test_read_that_raises_is_unconfirmed(self, connector):
        conn, mock_d4py = connector
        mock_d4py.get.side_effect = RuntimeError("readback error")

        result = await conn.write_channel("FAC/DEV/LOC/PROP", 10.0, confirm=True)

        assert result.outcome is WriteOutcome.UNCONFIRMED
        assert "readback error" in result.error_message
        assert result.observed is None

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
            installed(control_system_config({})),
        ):
            from osprey_connectors.control_system._doocs_connector import DOOCSConnector

            conn = DOOCSConnector()
            await conn.connect(armed_context())
            result = await conn.write_channel("FAC/DEV/LOC/PROP", 1.0)
            await conn.disconnect()

        assert result.outcome is WriteOutcome.REFUSED
        assert result.refusal_reason == "WRITES_DISABLED"
        assert refusal_phrase("tier_never") in result.error_message
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
        """Validation and the send each run off the loop, as two separate steps.

        A stand-in for ``max_step``'s blocking fresh read parks inside
        ``validate()`` until this test releases it. Regaining the loop while
        that call is still parked is possible only if it never ran there.
        The thread each fake records says the same thing a second way:
        neither ``validate()`` nor ``doocs4py.set`` reports the thread the loop
        runs on. The check finishes before the set begins: they are two
        offloads, so a check that outlives its bound sends nothing.
        """
        loop_thread = threading.get_ident()
        entered = threading.Event()  # validate() has begun
        release = threading.Event()  # the test lets it finish
        finished = threading.Event()  # validate() has returned
        threads: dict[str, int] = {}

        def blocking_validate(address, value, *, read_current=None):  # noqa: ARG001 - stands in for LimitsValidator.validate, whose signature this mirrors
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
            installed(control_system_config(ARMED_SECTION)),
        ):
            from osprey_connectors.control_system._doocs_connector import DOOCSConnector

            conn = DOOCSConnector()
            await conn.connect(armed_context())

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
        # The check was still made, and the value still went out.
        validator.validate.assert_called_once()
        assert result.outcome is WriteOutcome.UNREQUESTED
        mock_d4py.set.assert_called_once_with("FAC/DEV/LOC/PROP", 10.0)

    async def test_a_limits_refusal_raised_in_the_check_sends_nothing(self):
        """A refusal from the worker thread reaches the caller as a LIMITS refusal.

        The limits check runs off the loop, so the refusal has a thread
        boundary to cross. It must arrive carrying its LIMITS meaning rather
        than flattened into a failed write, and the value it refused must never
        reach ``doocs4py.set``.
        """
        from osprey_connectors.errors import ChannelLimitsViolationError, LimitsViolation
        from osprey_connectors.refusals import RefusalReason

        refusal = ChannelLimitsViolationError(
            "FAC/DEV/LOC/PROP", 10.0, "MAX_EXCEEDED", "above the configured ceiling"
        )
        validator = _make_limits_validator()
        validator.validate = MagicMock(side_effect=refusal)

        mock_d4py = _make_doocs4py()

        with (
            patch.dict(sys.modules, {"doocs4py": mock_d4py}),
            patch(_LIMITS_PATCH, return_value=validator),
            patch(_TZ_PATCH, return_value=UTC),
            installed(control_system_config(ARMED_SECTION)),
        ):
            from osprey_connectors.control_system._doocs_connector import DOOCSConnector

            conn = DOOCSConnector()
            await conn.connect(armed_context())

            result = await conn.write_channel("FAC/DEV/LOC/PROP", 10.0, confirm=False)

            await conn.disconnect()

        assert result.refusal_reason is RefusalReason.LIMITS
        assert result.limits_violation == LimitsViolation.from_error(refusal)
        mock_d4py.set.assert_not_called()


async def _bounded_connector(timeout_s):
    """A connected DOOCSConnector, patched by the caller, with the given ``timeout_s``."""
    from osprey_connectors.control_system._doocs_connector import DOOCSConnector

    conn = DOOCSConnector()
    await conn.connect(armed_context(settings={"timeout_s": timeout_s}))
    return conn


class TestBoundedCalls:
    """No call outlives its bound, however long the transport stalls.

    Each stalled fake is a :class:`Stall`. The connector answering while the
    stall has not yet returned is what proves the bound fired; the message
    names the bound that did. No test reads a clock.
    """

    async def test_a_stalled_read_raises_channel_unreachable_within_timeout_s(self):
        mock_d4py = _make_doocs4py()
        with (
            Stall() as get,
            patch.dict(sys.modules, {"doocs4py": mock_d4py}),
            patch(_LIMITS_PATCH, return_value=None),
            patch(_TZ_PATCH, return_value=UTC),
            installed(control_system_config(ARMED_SECTION)),
        ):
            mock_d4py.get.side_effect = get
            conn = await _bounded_connector(0.2)
            with pytest.raises(ChannelUnreachableError) as raised:
                await conn.read_channel("FAC/DEV/LOC/PROP")
            answered_first = not get.returned

        assert answered_first
        assert raised.value.address == "FAC/DEV/LOC/PROP"
        assert "FAC/DEV/LOC/PROP" in str(raised.value)
        assert "within 0.2s" in str(raised.value)
        assert not isinstance(raised.value, ConnectionError)
        assert not isinstance(raised.value, TimeoutError)

    async def test_a_per_call_timeout_wins_over_timeout_s(self):
        mock_d4py = _make_doocs4py()
        with (
            Stall() as get,
            patch.dict(sys.modules, {"doocs4py": mock_d4py}),
            patch(_LIMITS_PATCH, return_value=None),
            patch(_TZ_PATCH, return_value=UTC),
            installed(control_system_config(ARMED_SECTION)),
        ):
            mock_d4py.get.side_effect = get
            conn = await _bounded_connector(30)
            with pytest.raises(ChannelUnreachableError) as raised:
                await conn.read_channel("FAC/DEV/LOC/PROP", timeout_s=0.2)
            answered_first = not get.returned

        assert answered_first
        assert "within 0.2s" in str(raised.value)

    async def test_a_stalled_set_is_unconfirmed_not_failed(self):
        mock_d4py = _make_doocs4py()
        with (
            Stall() as send,
            patch.dict(sys.modules, {"doocs4py": mock_d4py}),
            patch(_LIMITS_PATCH, return_value=None),
            patch(_TZ_PATCH, return_value=UTC),
            installed(control_system_config(ARMED_SECTION)),
        ):
            mock_d4py.set.side_effect = send
            conn = await _bounded_connector(0.2)
            result = await conn.write_channel("FAC/DEV/LOC/PROP", 10.0, confirm=True)
            answered_first = not send.returned

        assert answered_first
        assert result.outcome is WriteOutcome.UNCONFIRMED
        assert "within 0.2s" in result.error_message
        assert result.observed is None
        assert mock_d4py.get.call_count == 0

    async def test_a_stalled_limits_check_is_refused_and_sends_nothing(self):
        """A check that outlives its bound refuses the write: nothing is sent.

        The check and the set are separate steps, so a check that does not
        answer in time leaves no set in flight that could still land.
        """
        validator = _make_limits_validator(confirm=True)
        mock_d4py = _make_doocs4py()
        with (
            Stall() as check,
            patch.dict(sys.modules, {"doocs4py": mock_d4py}),
            patch(_LIMITS_PATCH, return_value=validator),
            patch(_TZ_PATCH, return_value=UTC),
            installed(control_system_config(ARMED_SECTION)),
        ):
            validator.validate = MagicMock(side_effect=check)
            conn = await _bounded_connector(0.2)
            result = await conn.write_channel("FAC/DEV/LOC/PROP", 10.0, confirm=True)
            answered_first = not check.returned
            set_calls_at_return = mock_d4py.set.call_count

        assert answered_first
        assert result.outcome is WriteOutcome.REFUSED
        assert result.refusal_reason == "VALIDATION_ERROR"
        assert "within 0.2s" in result.error_message
        assert set_calls_at_return == 0
        mock_d4py.set.assert_not_called()

    async def test_a_stalled_confirming_read_is_unconfirmed(self):
        mock_d4py = _make_doocs4py()
        with (
            Stall() as get,
            patch.dict(sys.modules, {"doocs4py": mock_d4py}),
            patch(_LIMITS_PATCH, return_value=None),
            patch(_TZ_PATCH, return_value=UTC),
            installed(control_system_config(ARMED_SECTION)),
        ):
            mock_d4py.get.side_effect = get
            conn = await _bounded_connector(0.2)
            result = await conn.write_channel("FAC/DEV/LOC/PROP", 10.0, confirm=True)
            answered_first = not get.returned

        assert answered_first
        assert result.outcome is WriteOutcome.UNCONFIRMED
        assert "within 0.2s" in result.error_message
        assert result.observed is None
        mock_d4py.set.assert_called_once_with("FAC/DEV/LOC/PROP", 10.0)


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
        assert result.observed.value == "DESIRED"
        assert result.observed_number is None
        assert result.error_message is None

    async def test_differing_string_readback_is_a_mismatch(self, connector):
        """The read worked and disagreed — that is a mismatch, not an unknown."""
        conn, mock_d4py = connector
        mock_d4py.get.return_value = _make_eq_data(value="OTHER")

        result = await conn.write_channel("FAC/DEV/LOC/PROP", "DESIRED", confirm=True)

        assert result.outcome is WriteOutcome.MISMATCH
        assert result.observed.value == "OTHER"
        assert result.error_message is None

    async def test_sequence_readback_confirms_elementwise(self, connector):
        conn, mock_d4py = connector
        mock_d4py.get.return_value = _make_eq_data(value=[1, 2, 3])

        result = await conn.write_channel("FAC/DEV/LOC/PROP", [1, 2, 3], confirm=True)

        assert result.outcome is WriteOutcome.CONFIRMED
        assert result.observed.value == [1, 2, 3]

    async def test_sequence_mismatch_is_a_mismatch(self, connector):
        conn, mock_d4py = connector
        mock_d4py.get.return_value = _make_eq_data(value=[1, 2, 4])

        result = await conn.write_channel("FAC/DEV/LOC/PROP", [1, 2, 3], confirm=True)

        assert result.outcome is WriteOutcome.MISMATCH
        assert result.observed.value == [1, 2, 4]
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
        assert result.observed.value is observed
        # The read itself worked; only the comparison has no meaning.
        assert result.error_message is None

    async def test_numeric_readback_for_a_non_numeric_setpoint_is_a_mismatch(self, connector):
        """A string setpoint read back as a number disagrees — it is not unknown."""
        conn, mock_d4py = connector
        mock_d4py.get.return_value = _make_eq_data(value=1.0)

        result = await conn.write_channel("FAC/DEV/LOC/PROP", "ON", confirm=True)

        assert result.outcome is WriteOutcome.MISMATCH
        assert result.observed.value == pytest.approx(1.0)
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

        assert set(results.values) == set(addresses)
        assert results.failures == {}
        for v in results.values.values():
            assert isinstance(v, ChannelValue)

    async def test_a_failed_channel_is_reported_with_its_own_error(self, connector):
        conn, mock_d4py = connector

        def _side_effect(address):
            if "BAD" in address:
                raise RuntimeError("bad channel")
            return _make_eq_data()

        mock_d4py.get.side_effect = _side_effect

        results = await conn.read_multiple_channels(["FAC/DEV/LOC/OK", "FAC/DEV/LOC/BAD"])

        assert list(results.values) == ["FAC/DEV/LOC/OK"]
        assert list(results.failures) == ["FAC/DEV/LOC/BAD"]
        failure = results.failures["FAC/DEV/LOC/BAD"]
        assert isinstance(failure, ChannelUnreachableError)
        assert failure.address == "FAC/DEV/LOC/BAD"


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

    async def test_a_subscribed_reading_names_the_value_type(self, connector):
        conn, mock_d4py = connector
        cb = MagicMock()
        await conn.subscribe("FAC/DEV/LOC/PROP", cb)

        doocs_callback = mock_d4py.subscribe.call_args.args[1]
        doocs_callback(_make_eq_data(3.5))
        await asyncio.sleep(0)

        cb.assert_called_once()
        reading = cb.call_args.args[0]
        assert isinstance(reading, ChannelValue)
        assert reading.address == "FAC/DEV/LOC/PROP"
        assert reading.alarm is None
        assert reading.metadata.raw_metadata["type"] == "float"

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
