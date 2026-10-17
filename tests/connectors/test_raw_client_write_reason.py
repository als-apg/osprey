"""The raw-client-write refusal reason, its marker, and its message."""

import pytest

from osprey_connectors.errors import (
    RAW_CLIENT_WRITE_MARKER,
    ChannelWriteBlockedError,
    raw_client_write_message,
)


def test_reason_is_a_valid_blocked_reason():
    assert "RAW_CLIENT_WRITE" in ChannelWriteBlockedError._VALID_REASONS


def test_marker_value_is_pinned():
    # Consumers match this substring in subprocess stderr; changing it breaks them.
    assert RAW_CLIENT_WRITE_MARKER == "raw client write refused"


@pytest.mark.parametrize("address", ["SR:C01:MAG:PS:SP", "<unknown>"])
def test_message_embeds_marker_and_address(address):
    text = raw_client_write_message(address)
    assert isinstance(text, str)
    assert RAW_CLIENT_WRITE_MARKER in text
    assert address in text


def test_message_names_the_sanctioned_write_route():
    text = raw_client_write_message("<unknown>")
    assert "osprey.runtime.write_channel" in text
    assert "write_channels" in text


def test_message_usable_as_blocked_error_text():
    err = ChannelWriteBlockedError("PV:X", "RAW_CLIENT_WRITE", raw_client_write_message("PV:X"))
    assert err.reason == "RAW_CLIENT_WRITE"
    assert RAW_CLIENT_WRITE_MARKER in str(err)
    assert "PV:X" in str(err)
