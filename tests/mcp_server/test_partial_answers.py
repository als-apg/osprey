"""A request answered in part, or not at all, has one shape.

A result names what was not answered under ``missing``; a request nothing
answered is refused with the same map under ``details.missing``, as its one
shared cause's own error when every item failed alike and as
``nothing_answered`` otherwise.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from osprey.errors import ChannelNotFoundError
from osprey.mcp_server.control_system.error_handling import (
    ToolError,
    connector_error_handler,
    shared_typed_cause,
)
from osprey.mcp_server.errors import (
    MISSING_KEY,
    NOTHING_ANSWERED,
    NothingAnsweredError,
    refuse_nothing_answered,
)
from osprey_connectors.ipc.proxy import ChildUnresponsiveError
from tests.mcp_server.test_channel_read_size_rule import _read


def _envelope(excinfo: pytest.ExceptionInfo[ToolError]) -> dict:
    return json.loads(str(excinfo.value))


# ---------------------------------------------------------------------------
# The envelope
# ---------------------------------------------------------------------------


def test_the_contract_names():
    assert MISSING_KEY == "missing"
    assert NOTHING_ANSWERED == "nothing_answered"


def test_nothing_answered_carries_the_map():
    with pytest.raises(ToolError) as excinfo:
        refuse_nothing_answered("a_tool", "things", {"x": "Gone: no x", "y": None})

    assert _envelope(excinfo) == {
        "error": True,
        "error_type": "nothing_answered",
        "error_message": "a_tool: none of the 2 requested things answered",
        "suggestions": ["Name each entry of details.missing and its text to the operator."],
        "details": {"missing": {"x": "Gone: no x", "y": None}},
    }


def test_nothing_answered_keeps_further_details():
    with pytest.raises(ToolError) as excinfo:
        refuse_nothing_answered("a_tool", "things", {"x": None}, details={"model": "m"})

    assert _envelope(excinfo)["details"] == {"model": "m", "missing": {"x": None}}


# ---------------------------------------------------------------------------
# One shared cause
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "errors",
    [
        [TimeoutError("a"), TimeoutError("b")],
        [ConnectionError("a"), ConnectionRefusedError("b")],
        [ChannelNotFoundError("A"), ChannelNotFoundError("B")],
        [ChildUnresponsiveError("a"), ChildUnresponsiveError("b")],
    ],
    ids=["timeout", "connection", "not-a-channel", "unresponsive-child"],
)
def test_one_typed_cause_is_shared(errors):
    assert shared_typed_cause(errors) is errors[0]


@pytest.mark.parametrize(
    "errors",
    [
        [],
        [TimeoutError("a"), ChannelNotFoundError("B")],
        [ChildUnresponsiveError("a"), TimeoutError("b")],
        [RuntimeError("a"), RuntimeError("b")],
    ],
    ids=["none", "mixed", "child-and-plain-timeout", "untyped"],
)
def test_no_shared_cause(errors):
    assert shared_typed_cause(errors) is None


# ---------------------------------------------------------------------------
# The handler
# ---------------------------------------------------------------------------


async def test_a_shared_cause_keeps_its_own_envelope_with_the_map():
    missing = {"A": "TimeoutError: a", "B": "TimeoutError: b"}
    with pytest.raises(ToolError) as excinfo:
        async with connector_error_handler("channel_read", "archiver"):
            raise NothingAnsweredError(TimeoutError("a"), missing=missing)

    assert _envelope(excinfo) == {
        "error": True,
        "error_type": "timeout_error",
        "error_message": "channel_read timed out: a",
        "suggestions": ["Check network connectivity.", "Try a smaller request."],
        "details": {"missing": missing},
    }


async def test_no_cause_is_nothing_answered_with_the_map():
    missing = {"A": "RuntimeError: a", "B": "RuntimeError: b"}
    with pytest.raises(ToolError) as excinfo:
        async with connector_error_handler("channel_read", "archiver"):
            raise NothingAnsweredError(None, missing=missing)

    envelope = _envelope(excinfo)
    assert envelope["error_type"] == "nothing_answered"
    assert envelope["error_message"] == "channel_read: none of the 2 requested channels answered"
    assert envelope["details"] == {"missing": missing}


async def test_a_shared_connection_error_drops_the_connector_once():
    invalidate = AsyncMock()
    with (
        patch(
            "osprey.mcp_server.control_system.error_handling.invalidate_active_connector",
            invalidate,
        ),
        pytest.raises(ToolError) as excinfo,
    ):
        async with connector_error_handler("channel_read", "archiver"):
            raise NothingAnsweredError(
                ConnectionError("down"), missing={"A": "x", "B": "y", "C": "z"}
            )

    assert _envelope(excinfo)["error_type"] == "connection_error"
    invalidate.assert_awaited_once_with("archiver")


async def test_a_wedged_child_is_pinged_once():
    ping = AsyncMock(return_value="")
    with (
        patch(
            "osprey.mcp_server.control_system.error_handling._answer_unresponsive_child",
            ping,
        ),
        pytest.raises(ToolError) as excinfo,
    ):
        async with connector_error_handler("channel_read"):
            raise NothingAnsweredError(
                ChildUnresponsiveError("deadline"), missing={"A": "x", "B": "y"}
            )

    assert _envelope(excinfo)["error_type"] == "timeout_error"
    ping.assert_awaited_once()


async def test_mixed_causes_still_recover_what_failed():
    invalidate = AsyncMock()
    ping = AsyncMock(return_value="")
    errors = [
        ConnectionError("down"),
        ConnectionRefusedError("refused"),
        ChildUnresponsiveError("deadline"),
        ChildUnresponsiveError("deadline again"),
        TimeoutError("late"),
    ]
    with (
        patch(
            "osprey.mcp_server.control_system.error_handling.invalidate_active_connector",
            invalidate,
        ),
        patch(
            "osprey.mcp_server.control_system.error_handling._answer_unresponsive_child",
            ping,
        ),
        pytest.raises(ToolError) as excinfo,
    ):
        async with connector_error_handler("channel_read"):
            raise NothingAnsweredError(
                None, missing={str(i): str(e) for i, e in enumerate(errors)}, errors=errors
            )

    assert _envelope(excinfo)["error_type"] == "nothing_answered"
    invalidate.assert_awaited_once_with("control_system")
    ping.assert_awaited_once_with("control_system")


async def test_mixed_causes_without_a_dropped_link_recover_nothing():
    invalidate = AsyncMock()
    ping = AsyncMock(return_value="")
    errors = [TimeoutError("late"), ChannelNotFoundError("B")]
    with (
        patch(
            "osprey.mcp_server.control_system.error_handling.invalidate_active_connector",
            invalidate,
        ),
        patch(
            "osprey.mcp_server.control_system.error_handling._answer_unresponsive_child",
            ping,
        ),
        pytest.raises(ToolError) as excinfo,
    ):
        async with connector_error_handler("channel_read"):
            raise NothingAnsweredError(None, missing={"A": "x", "B": "y"}, errors=errors)

    assert _envelope(excinfo)["error_type"] == "nothing_answered"
    invalidate.assert_not_awaited()
    ping.assert_not_awaited()


# ---------------------------------------------------------------------------
# channel_read, the reference tool
# ---------------------------------------------------------------------------


async def test_a_read_nothing_answered_for_one_cause_is_that_cause(tmp_path, monkeypatch):
    values = {"SR:A": TimeoutError("late"), "SR:B": TimeoutError("later")}
    with pytest.raises(ToolError) as excinfo:
        await _read(tmp_path, monkeypatch, values)

    envelope = _envelope(excinfo)
    assert envelope["error_type"] == "timeout_error"
    assert envelope["details"]["missing"] == {
        "SR:A": "TimeoutError: late",
        "SR:B": "TimeoutError: later",
    }


async def test_a_read_nothing_answered_for_mixed_causes_is_nothing_answered(tmp_path, monkeypatch):
    values = {"SR:A": TimeoutError("late"), "SR:B": ChannelNotFoundError("SR:B")}
    with pytest.raises(ToolError) as excinfo:
        await _read(tmp_path, monkeypatch, values)

    envelope = _envelope(excinfo)
    assert envelope["error_type"] == "nothing_answered"
    assert envelope["details"]["missing"] == {
        "SR:A": "TimeoutError: late",
        "SR:B": "ChannelNotFoundError: SR:B is not a channel of this facility",
    }


async def test_a_read_nothing_answered_for_an_untyped_cause_is_nothing_answered(
    tmp_path, monkeypatch
):
    values = {"SR:A": RuntimeError("odd"), "SR:B": RuntimeError("odder")}
    with pytest.raises(ToolError) as excinfo:
        await _read(tmp_path, monkeypatch, values)

    envelope = _envelope(excinfo)
    assert envelope["error_type"] == "nothing_answered"
    assert envelope["details"]["missing"] == {
        "SR:A": "RuntimeError: odd",
        "SR:B": "RuntimeError: odder",
    }


async def test_a_read_that_timed_out_and_lost_its_link_drops_the_connector(tmp_path, monkeypatch):
    invalidate = AsyncMock()
    values = {"SR:A": TimeoutError("late"), "SR:B": ConnectionError("down")}
    with (
        patch(
            "osprey.mcp_server.control_system.error_handling.invalidate_active_connector",
            invalidate,
        ),
        pytest.raises(ToolError) as excinfo,
    ):
        await _read(tmp_path, monkeypatch, values)

    envelope = _envelope(excinfo)
    assert envelope["error_type"] == "nothing_answered"
    assert envelope["details"]["missing"] == {
        "SR:A": "TimeoutError: late",
        "SR:B": "ConnectionError: down",
    }
    invalidate.assert_awaited_once_with("control_system")
