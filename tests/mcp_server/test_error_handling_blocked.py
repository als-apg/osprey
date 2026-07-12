"""Tests for reference-monitor write-refusal classification in the error handler.

A ``ChannelWriteBlockedError`` raised inside the ``connector_error_handler``
context manager must classify as a dedicated ``write_refused`` error type
(carrying the refusal reason and channel), NOT fall through to the generic
``internal_error`` clause.
"""

from __future__ import annotations

from osprey.errors import ChannelWriteBlockedError
from osprey.mcp_server.control_system.error_handling import connector_error_handler
from tests.mcp_server.conftest import assert_raises_error


async def test_blocked_write_classifies_as_write_refused():
    """A refusal raised in the body surfaces as a ``write_refused`` envelope."""
    with assert_raises_error(error_type="write_refused") as ctx:
        async with connector_error_handler("channel_write"):
            raise ChannelWriteBlockedError("X:Y:SP", "VALIDATION_ERROR")

    envelope = ctx["envelope"]
    assert envelope["details"] == {"channel": "X:Y:SP", "reason": "VALIDATION_ERROR"}
    assert "channel_write" in envelope["error_message"]


async def test_blocked_write_is_not_internal_error():
    """The refusal must NOT be swallowed by the generic ``internal_error`` clause."""
    with assert_raises_error() as ctx:
        async with connector_error_handler("channel_write"):
            raise ChannelWriteBlockedError("A:B:SP", "WRITES_DISABLED")

    assert ctx["envelope"]["error_type"] != "internal_error"
    assert ctx["envelope"]["error_type"] == "write_refused"
    assert ctx["envelope"]["details"]["reason"] == "WRITES_DISABLED"
