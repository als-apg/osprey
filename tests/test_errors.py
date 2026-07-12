"""Unit tests for typed exceptions in osprey.errors."""

import pytest

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
        )

    def test_unknown_reason_permitted(self):
        # Constructor does not validate against _VALID_REASONS (permissive by design).
        err = ChannelWriteBlockedError("chan", "SOMETHING_ELSE")
        assert err.reason == "SOMETHING_ELSE"


class TestChannelWriteFailedError:
    def test_default_message(self):
        err = ChannelWriteFailedError("RING:MAG:PS:SP", "CAPUT_FAILED")
        assert err.channel_address == "RING:MAG:PS:SP"
        assert err.reason == "CAPUT_FAILED"
        assert str(err) == "Write to 'RING:MAG:PS:SP' failed (CAPUT_FAILED)"

    def test_custom_message(self):
        err = ChannelWriteFailedError(
            "RING:MAG:PS:SP", "READBACK_UNVERIFIED", "readback mismatch"
        )
        assert err.channel_address == "RING:MAG:PS:SP"
        assert err.reason == "READBACK_UNVERIFIED"
        assert str(err) == "readback mismatch"

    def test_is_exception(self):
        with pytest.raises(ChannelWriteFailedError):
            raise ChannelWriteFailedError("chan", "READBACK_UNVERIFIED")

    def test_valid_reasons_reference(self):
        assert ChannelWriteFailedError._VALID_REASONS == (
            "CAPUT_FAILED",
            "READBACK_UNVERIFIED",
        )

    def test_unknown_reason_permitted(self):
        err = ChannelWriteFailedError("chan", "WHATEVER")
        assert err.reason == "WHATEVER"
