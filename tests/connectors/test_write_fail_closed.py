"""Fail-closed validation tests for EPICSConnector.write_channel.

Task 1.3: a validation error other than a limits violation must REFUSE the
write (outcome=WriteOutcome.REFUSED, refusal_reason="VALIDATION_ERROR") and
never issue a put. A ChannelLimitsViolationError still propagates unchanged,
and an error raised by the put itself (an unreachable channel, or any pvapy
failure the connector does not recognize) propagates — it is a genuine
failure whose outcome is unknown, not a refusal. The one put error that IS a
refusal is the control system's own denial; see
test_control_system_refused.py.

The connector holds a fake ``pvaccess`` module (``tests/connectors/_write_fakes.py``);
``connector._pvaccess.calls("put")`` is every put it issued.
"""

import threading
from unittest.mock import MagicMock, patch

import pytest

from osprey.connectors.control_system.base import ChannelWriteResult, WriteOutcome
from osprey.errors import ChannelLimitsViolationError
from tests.connectors._epics_fakes import FakePvaException, timed_out
from tests.connectors._write_fakes import make_mock_epics_connector as _make_connector
from tests.connectors._write_fakes import writes_enabled_config as _writes_enabled_config


class TestFailClosedValidation:
    @pytest.mark.asyncio
    async def test_non_limits_validation_error_refuses_write(self):
        """A non-limits exception from validate() refuses the write; no put is issued."""
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
        assert connector._pvaccess.calls("put") == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("violation_type", "value", "reason"),
        [
            ("max_value", 999.0, "above max"),
            ("max_step", 5.0, "step 5.0 exceeds max_step 1.0"),
        ],
        ids=["max_value", "max_step"],
    )
    async def test_limits_violation_propagates_without_a_put(self, violation_type, value, reason):
        """A ChannelLimitsViolationError raised inside the validation offload
        propagates out of write_channel unchanged, and no put is issued.

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
        assert connector._pvaccess.calls("put") == []

    @pytest.mark.asyncio
    async def test_an_unreachable_channel_on_put_propagates_not_refused(self):
        """validate passes but the put times out → ConnectionError propagates.

        Regression guard: a put-raised error must NOT be swallowed into a
        refusal ChannelWriteResult. It is a genuine write failure and must
        surface as the raised exception — as the stdlib ConnectionError, the
        word the MCP error envelope and connector invalidation key on.
        """
        connector = _make_connector(put_error=timed_out("TEST:PV"))

        with patch(
            "osprey.utils.config.get_config_value",
            side_effect=_writes_enabled_config,
        ):
            with pytest.raises(ConnectionError, match="TEST:PV"):
                await connector.write_channel("TEST:PV", 42.0, confirm=False)

        # validate passed and the put was actually attempted.
        connector._limits_validator.validate.assert_called_once()
        assert len(connector._pvaccess.calls("put")) == 1

    @pytest.mark.asyncio
    async def test_an_unrecognized_put_error_propagates_unchanged(self):
        """A pvapy failure the classifier does not name is neither a refusal nor
        a connection error: the outcome is unknown, so it is raised as it came."""
        error = FakePvaException("channel TEST:PV PvaClientPut::put something else")
        connector = _make_connector(put_error=error)

        with patch(
            "osprey.utils.config.get_config_value",
            side_effect=_writes_enabled_config,
        ):
            with pytest.raises(FakePvaException) as raised:
                await connector.write_channel("TEST:PV", 42.0, confirm=False)

        assert raised.value is error


class TestNonBlockingOffload:
    """Task 2.1: validate() runs off the event loop, so a caller on the loop is
    never stalled by the blocking read that max_step performs.
    """

    @pytest.mark.asyncio
    async def test_validate_runs_off_the_event_loop(self):
        """A blocking validate() runs on a worker thread, not the event loop.

        max_step's fresh read is a blocking pvapy get; if validate() ran on the
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
