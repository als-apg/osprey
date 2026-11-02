"""Fail-closed validation tests for EPICSConnector.write_channel.

Task 1.3: a validation error other than a limits violation must REFUSE the
write (outcome=WriteOutcome.REFUSED, refusal_reason="VALIDATION_ERROR") and
never issue a caput. A ChannelLimitsViolationError still propagates unchanged,
and a put that fails for any other reason (a lost circuit) propagates as a
ChannelUnreachableError — it is a genuine failure, not a refusal. The one caput error that
IS a refusal is the control system's own denial; see
test_control_system_refused.py.
"""

import threading
from unittest.mock import MagicMock, patch

import pytest

from osprey_connectors.control_system._base import ChannelWriteResult, WriteOutcome
from osprey_connectors.errors import ChannelLimitsViolationError, ChannelUnreachableError
from osprey_connectors_tests._epics_fakes import ECA_DISCONN
from osprey_connectors_tests._write_fakes import arm_switch
from osprey_connectors_tests._write_fakes import make_mock_epics_connector as _make_connector
from osprey_connectors_tests._write_fakes import writes_enabled_config as _writes_enabled_config


@pytest.fixture(autouse=True)
def _switch_on(monkeypatch, tmp_path):
    """The login's switch on for the simulator target, connectors stamped with it.

    With :func:`writes_enabled_config` serving a write tier that allows, that is
    everything the reference monitor's verdict asks before a write reaches the
    transport; a test that disarms serves a tier that does not.
    """
    return arm_switch(monkeypatch, tmp_path / "armed-data")


class TestOneNamePerWrite:
    """``X``, ``ca://X`` and a padded spelling are one record, judged and written as one."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("spelling", ["ca://TEST:PV", "  TEST:PV ", " ca://TEST:PV\t"])
    async def test_limits_are_looked_up_under_the_canonical_name(self, spelling):
        """Every spelling validates against the limits entry of ``TEST:PV`` and is
        refused by that entry's limit; the raw spelling never reaches the validator."""
        violation = ChannelLimitsViolationError(
            channel_address="TEST:PV",
            value=999.0,
            violation_type="max_value",
            violation_reason="above max",
        )
        connector = _make_connector(validate_side_effect=violation)

        with patch(
            "osprey_connectors.config.get_config_value",
            side_effect=_writes_enabled_config,
        ):
            with pytest.raises(ChannelLimitsViolationError) as raised:
                await connector.write_channel(spelling, 999.0, confirm=False)

        assert raised.value is violation
        connector._limits_validator.validate.assert_called_once()
        (name, value), _kwargs = connector._limits_validator.validate.call_args
        assert (name, value) == ("TEST:PV", 999.0)
        assert connector._aioca.puts == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("spelling", ["ca://TEST:PV", "  TEST:PV "])
    async def test_the_put_and_the_result_name_the_canonical_record(self, spelling):
        """A write that passes validation puts to ``TEST:PV`` and reports that name."""
        connector = _make_connector()

        with patch(
            "osprey_connectors.config.get_config_value",
            side_effect=_writes_enabled_config,
        ):
            result = await connector.write_channel(spelling, 1.0, confirm=False)

        (name, _value), _kwargs = connector._limits_validator.validate.call_args
        assert name == "TEST:PV"
        assert [(n, v) for n, v, _wait in connector._aioca.puts] == [("TEST:PV", 1.0)]
        assert result.channel_address == "TEST:PV"


class TestFailClosedValidation:
    @pytest.mark.asyncio
    async def test_non_limits_validation_error_refuses_write(self):
        """A non-limits exception from validate() refuses the write; caput never runs."""
        connector = _make_connector(validate_side_effect=RuntimeError("boom"))

        with patch(
            "osprey_connectors.config.get_config_value",
            side_effect=_writes_enabled_config,
        ):
            result = await connector.write_channel("TEST:PV", 42.0, confirm=False)

        assert isinstance(result, ChannelWriteResult)
        assert result.outcome is WriteOutcome.REFUSED
        assert result.refusal_reason == "VALIDATION_ERROR"
        assert "TEST:PV" in result.error_message
        # The write must NEVER have been issued.
        assert connector._aioca.puts == []

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
            "osprey_connectors.config.get_config_value",
            side_effect=_writes_enabled_config,
        ):
            with pytest.raises(ChannelLimitsViolationError) as raised:
                await connector.write_channel("TEST:PV", value, confirm=False)

        assert raised.value is violation
        connector._limits_validator.validate.assert_called_once()
        assert connector._aioca.puts == []

    @pytest.mark.asyncio
    async def test_caput_connection_failure_propagates_not_refused(self):
        """validate passes but the put loses its circuit → ChannelUnreachableError propagates.

        Regression guard: a put failure must NOT be swallowed into a refusal
        ChannelWriteResult. It is a genuine write failure and must surface as
        the raised exception.
        """
        connector = _make_connector(put=ECA_DISCONN)

        with patch(
            "osprey_connectors.config.get_config_value",
            side_effect=_writes_enabled_config,
        ):
            with pytest.raises(ChannelUnreachableError):
                await connector.write_channel("TEST:PV", 42.0, confirm=False)

        # validate passed and the caput was actually attempted.
        connector._limits_validator.validate.assert_called_once()
        assert [name for name, _value, _wait in connector._aioca.puts] == ["TEST:PV"]


class TestNonBlockingOffload:
    """Task 2.1: validate() runs in a thread offload so a caller on the event
    loop is never stalled by the blocking read that max_step performs.
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
            "osprey_connectors.config.get_config_value",
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
