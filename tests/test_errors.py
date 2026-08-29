"""Unit tests for typed exceptions in osprey.errors."""

import pytest

from osprey.connectors.control_system import WriteOutcome
from osprey.errors import ChannelWriteBlockedError, ChannelWriteFailedError


class TestChannelWriteBlockedError:
    def test_default_message(self):
        err = ChannelWriteBlockedError("RING:MAG:PS:SP", "WRITES_DISABLED")
        assert err.channel_address == "RING:MAG:PS:SP"
        assert err.reason == "WRITES_DISABLED"
        assert str(err) == (
            "Write to 'RING:MAG:PS:SP' refused by reference monitor (WRITES_DISABLED)"
        )

    def test_custom_message(self):
        err = ChannelWriteBlockedError("RING:MAG:PS:SP", "LIMITS", "value out of range")
        assert err.channel_address == "RING:MAG:PS:SP"
        assert err.reason == "LIMITS"
        assert str(err) == "value out of range"

    def test_is_exception(self):
        with pytest.raises(ChannelWriteBlockedError):
            raise ChannelWriteBlockedError("chan", "VALIDATION_ERROR")

    def test_valid_reasons_reference(self):
        assert ChannelWriteBlockedError._VALID_REASONS == (
            "WRITES_DISABLED",
            "LIMITS",
            "VALIDATION_ERROR",
            "CONTROL_SYSTEM_REFUSED",
        )

    def test_unknown_reason_permitted(self):
        # Constructor does not validate against _VALID_REASONS (permissive by design).
        err = ChannelWriteBlockedError("chan", "SOMETHING_ELSE")
        assert err.reason == "SOMETHING_ELSE"


class TestChannelWriteFailedError:
    def test_default_message(self):
        err = ChannelWriteFailedError("RING:MAG:PS:SP", "FAILED")
        assert err.channel_address == "RING:MAG:PS:SP"
        assert err.reason == "FAILED"
        assert str(err) == "Write to 'RING:MAG:PS:SP' failed (FAILED)"

    def test_custom_message(self):
        err = ChannelWriteFailedError("RING:MAG:PS:SP", "UNCONFIRMED", "re-read raised")
        assert err.channel_address == "RING:MAG:PS:SP"
        assert err.reason == "UNCONFIRMED"
        assert str(err) == "re-read raised"

    def test_is_exception(self):
        with pytest.raises(ChannelWriteFailedError):
            raise ChannelWriteFailedError("chan", "UNCONFIRMED")

    def test_valid_reasons_reference(self):
        assert ChannelWriteFailedError._VALID_REASONS == (
            "FAILED",
            "MISMATCH",
            "UNCONFIRMED",
        )

    def test_unknown_reason_permitted(self):
        err = ChannelWriteFailedError("chan", "WHATEVER")
        assert err.reason == "WHATEVER"

    def test_write_detail_attributes_default_to_none(self):
        """Positional construction stays valid; the new detail is optional."""
        err = ChannelWriteFailedError("chan", "FAILED")
        assert err.outcome is None
        assert err.value_written is None
        assert err.observed_value is None

    def test_mismatch_carries_outcome_and_both_values(self):
        """A mismatch must say what was sent and what the channel holds."""
        err = ChannelWriteFailedError(
            "RING:MAG:PS:SP",
            "MISMATCH",
            outcome=WriteOutcome.MISMATCH,
            value_written=1.7,
            observed_value=1.5,
        )
        assert err.outcome == WriteOutcome.MISMATCH
        assert err.value_written == 1.7
        assert err.observed_value == 1.5
        assert "1.7" in str(err)
        assert "1.5" in str(err)
        assert "RING:MAG:PS:SP" in str(err)

    def test_mismatch_message_can_still_be_overridden(self):
        err = ChannelWriteFailedError(
            "chan", "MISMATCH", "clamped by the power supply", value_written=1.7, observed_value=1.5
        )
        assert str(err) == "clamped by the power supply"

    def test_mismatch_without_detail_does_not_report_none_as_a_reading(self):
        """A detail-less mismatch falls back rather than saying "sent None"."""
        err = ChannelWriteFailedError("chan", "MISMATCH")
        assert str(err) == "Write to 'chan' failed (MISMATCH)"

    def test_non_mismatch_reason_keeps_the_generic_message(self):
        err = ChannelWriteFailedError("chan", "UNCONFIRMED", value_written=1.7)
        assert str(err) == "Write to 'chan' failed (UNCONFIRMED)"
