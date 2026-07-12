"""Fail-closed validation tests for EPICSConnector.write_channel.

Task 1.3: a validation error other than a limits violation must REFUSE the
write (blocked=True, refusal_reason="VALIDATION_ERROR") and never issue a
caput. A ChannelLimitsViolationError still propagates unchanged, and an error
raised by the caput itself (e.g. ConnectionError) propagates untouched — it is
never reclassified as a refusal.
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from osprey.connectors.control_system.base import ChannelWriteResult
from osprey.connectors.control_system.epics_connector import EPICSConnector
from osprey.errors import ChannelLimitsViolationError


def _writes_enabled_config(key, default=None):
    """Config stub: writes enabled so the base wrapper reaches write_channel."""
    if key == "control_system.writes_enabled":
        return True
    return default


def _make_connector(validate_side_effect=None, caput_side_effect=None, caput_return=True):
    """Build an EPICSConnector wired with mock epics + limits validator.

    Bypasses connect() (which imports pyepics) by setting the attributes the
    write path depends on directly.
    """
    connector = EPICSConnector()
    connector._epics = MagicMock()
    connector._epics.caput = MagicMock(side_effect=caput_side_effect, return_value=caput_return)
    connector._limits_validator = MagicMock()
    connector._limits_validator.validate = MagicMock(side_effect=validate_side_effect)
    connector._timeout = 5.0
    connector._connected = True
    return connector


class TestFailClosedValidation:
    @pytest.mark.asyncio
    async def test_non_limits_validation_error_refuses_write(self):
        """A non-limits exception from validate() refuses the write; caput never runs."""
        connector = _make_connector(validate_side_effect=RuntimeError("boom"))

        with patch(
            "osprey.utils.config.get_config_value",
            side_effect=_writes_enabled_config,
        ):
            result = await connector.write_channel("TEST:PV", 42.0, verification_level="none")

        assert isinstance(result, ChannelWriteResult)
        assert result.success is False
        assert result.blocked is True
        assert result.refusal_reason == "VALIDATION_ERROR"
        assert "TEST:PV" in result.error_message
        # The write must NEVER have been issued.
        connector._epics.caput.assert_not_called()

    @pytest.mark.asyncio
    async def test_limits_violation_propagates(self):
        """A ChannelLimitsViolationError still propagates; caput never runs."""
        violation = ChannelLimitsViolationError(
            channel_address="TEST:PV",
            value=999.0,
            violation_type="max_value",
            violation_reason="above max",
        )
        connector = _make_connector(validate_side_effect=violation)

        with patch(
            "osprey.utils.config.get_config_value",
            side_effect=_writes_enabled_config,
        ):
            with pytest.raises(ChannelLimitsViolationError):
                await connector.write_channel("TEST:PV", 999.0, verification_level="none")

        connector._epics.caput.assert_not_called()

    @pytest.mark.asyncio
    async def test_caput_connection_error_propagates_not_refused(self):
        """validate passes but caput raises ConnectionError → it propagates.

        Regression guard: a caput-raised error must NOT be swallowed into a
        refusal ChannelWriteResult. It is a genuine write failure and must
        surface as the raised exception.
        """
        connector = _make_connector(caput_side_effect=ConnectionError("gateway down"))

        with patch(
            "osprey.utils.config.get_config_value",
            side_effect=_writes_enabled_config,
        ):
            with pytest.raises(ConnectionError):
                await connector.write_channel("TEST:PV", 42.0, verification_level="none")

        # validate passed and the caput was actually attempted.
        connector._limits_validator.validate.assert_called_once()
        connector._epics.caput.assert_called_once()


class TestNonBlockingOffload:
    """Task 2.1: validate()+caput run in ONE thread offload so a caller on the
    event loop is never stalled by the blocking caget that max_step performs.
    """

    @pytest.mark.asyncio
    async def test_validate_runs_off_the_event_loop(self):
        """A slow (blocking) validate() must NOT starve the event loop.

        Simulates max_step's blocking caget with a 0.3s time.sleep inside
        validate(). While write_channel is in flight, a concurrently-awaited
        0.05s asyncio.sleep must complete promptly. If validate ran ON the loop,
        the loop would be blocked for the full 0.3s and the concurrent sleep
        could not finish until then — proving validate ran OFF the loop.
        """
        connector = _make_connector()

        def slow_validate(_addr, _val):
            time.sleep(0.3)  # stand-in for max_step's blocking caget

        connector._limits_validator.validate = MagicMock(side_effect=slow_validate)

        # A concurrent ticker that can only advance if the loop is being serviced.
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.005)
                ticks += 1

        with patch(
            "osprey.utils.config.get_config_value",
            side_effect=_writes_enabled_config,
        ):
            ticker_task = asyncio.create_task(ticker())
            write_task = asyncio.create_task(
                connector.write_channel("TEST:PV", 42.0, verification_level="none")
            )

            # Give the write time to reach its to_thread offload, then measure how
            # long a short concurrent sleep takes while validate() blocks a thread.
            start = time.monotonic()
            await asyncio.sleep(0.05)
            elapsed = time.monotonic() - start

            result = await write_task
            ticker_task.cancel()

        # The loop stayed responsive: the 0.05s sleep did NOT get stretched to the
        # 0.3s validate duration, and the ticker advanced during the window.
        assert elapsed < 0.2, f"event loop was blocked for {elapsed:.3f}s"
        assert ticks > 0, "concurrent coroutine was starved (loop blocked)"
        # max_step-style validation was still evaluated, and the write succeeded.
        connector._limits_validator.validate.assert_called_once()
        assert result.success is True

    @pytest.mark.asyncio
    async def test_max_step_violation_still_raised_through_offload(self):
        """max_step is NOT skipped by the offload: a limits violation raised
        inside the thread propagates out of write_channel unchanged, and no
        caput is issued.
        """
        violation = ChannelLimitsViolationError(
            channel_address="TEST:PV",
            value=5.0,
            violation_type="max_step",
            violation_reason="step 5.0 exceeds max_step 1.0",
        )
        connector = _make_connector(validate_side_effect=violation)

        with patch(
            "osprey.utils.config.get_config_value",
            side_effect=_writes_enabled_config,
        ):
            with pytest.raises(ChannelLimitsViolationError):
                await connector.write_channel("TEST:PV", 5.0, verification_level="none")

        connector._limits_validator.validate.assert_called_once()
        connector._epics.caput.assert_not_called()
