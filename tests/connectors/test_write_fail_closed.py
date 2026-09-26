"""Fail-closed validation tests for EPICSConnector.write_channel.

Task 1.3: a validation error other than a limits violation must REFUSE the
write (outcome=WriteOutcome.REFUSED, refusal_reason="VALIDATION_ERROR") and
never issue a caput. A ChannelLimitsViolationError still propagates unchanged,
and an error raised by the caput itself (e.g. ConnectionError) propagates
untouched — it is a genuine failure, not a refusal. The one caput error that
IS a refusal is the control system's own denial; see
test_control_system_refused.py.
"""

import threading
from unittest.mock import MagicMock, patch

import pytest

from osprey.connectors.control_system.base import ChannelWriteResult, WriteOutcome
from osprey.errors import ChannelLimitsViolationError
from tests.connectors._write_fakes import make_mock_epics_connector as _make_connector
from tests.connectors._write_fakes import writes_enabled_config as _writes_enabled_config


class TestFailClosedValidation:
    @pytest.mark.asyncio
    async def test_non_limits_validation_error_refuses_write(self):
        """A non-limits exception from validate() refuses the write; caput never runs."""
        connector = _make_connector(validate_side_effect=RuntimeError("boom"))

        with patch(
            "osprey.utils.config.get_config_value",
            side_effect=_writes_enabled_config,
        ):
            result = await connector.write_channel("TEST:PV", 42.0, confirm=False)

        assert isinstance(result, ChannelWriteResult)
        assert result.outcome is WriteOutcome.REFUSED
        assert result.refusal_reason == "VALIDATION_ERROR"
        assert "TEST:PV" in result.error_message
        # The write must NEVER have been issued.
        connector._epics.caput.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("violation_type", "value", "reason"),
        [
            ("max_value", 999.0, "above max"),
            ("max_step", 5.0, "step 5.0 exceeds max_step 1.0"),
        ],
        ids=["max_value", "max_step"],
    )
    async def test_limits_violation_propagates_without_caput(self, violation_type, value, reason):
        """A ChannelLimitsViolationError raised inside the validate+caput offload
        propagates out of write_channel unchanged, and caput never runs.

        ``max_step`` is not skipped by the offload: it raises from the same
        thread hop a ``max_value`` violation does.
        """
        violation = ChannelLimitsViolationError(
            channel_address="TEST:PV",
            value=value,
            violation_type=violation_type,
            violation_reason=reason,
        )
        connector = _make_connector(validate_side_effect=violation)

        with patch(
            "osprey.utils.config.get_config_value",
            side_effect=_writes_enabled_config,
        ):
            with pytest.raises(ChannelLimitsViolationError) as raised:
                await connector.write_channel("TEST:PV", value, confirm=False)

        assert raised.value is violation
        connector._limits_validator.validate.assert_called_once()
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
                await connector.write_channel("TEST:PV", 42.0, confirm=False)

        # validate passed and the caput was actually attempted.
        connector._limits_validator.validate.assert_called_once()
        connector._epics.caput.assert_called_once()


class TestNonBlockingOffload:
    """Task 2.1: validate()+caput run in ONE thread offload so a caller on the
    event loop is never stalled by the blocking caget that max_step performs.
    """

    @pytest.mark.asyncio
    async def test_validate_runs_off_the_event_loop(self):
        """A blocking validate() runs on a worker thread, not the event loop.

        max_step's fresh read is a blocking caget; if validate() ran on the
        loop thread it would stall every other coroutine for its duration.
        Recording the thread validate() runs on pins the offload directly.
        """
        connector = _make_connector()
        validate_threads: list[int] = []

        def recording_validate(channel_address, value, *, read_current=None):  # noqa: ARG001 - stands in for LimitsValidator.validate, whose signature this mirrors
            validate_threads.append(threading.get_ident())

        connector._limits_validator.validate = MagicMock(side_effect=recording_validate)

        with patch(
            "osprey.utils.config.get_config_value",
            side_effect=_writes_enabled_config,
        ):
            result = await connector.write_channel("TEST:PV", 42.0, confirm=False)

        loop_thread = threading.get_ident()
        assert validate_threads, "validate() was never called"
        assert validate_threads[0] != loop_thread, "validate() ran on the event-loop thread"
        # max_step-style validation was still evaluated, and the write succeeded
        # (unconfirmed, since confirm=False).
        connector._limits_validator.validate.assert_called_once()
        assert result.outcome is WriteOutcome.UNREQUESTED
