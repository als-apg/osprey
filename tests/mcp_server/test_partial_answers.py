"""A request answered in part, or not at all, has one shape.

A result names what was not answered under ``missing``; a request nothing
answered is refused with the same map under ``details.missing``, as its one
shared cause's own error when every item failed alike and as
``nothing_answered`` otherwise. A write's map is its rows, under ``results``;
which key a tool states its map under is declared once, in ``MAP_KEYS``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from osprey.mcp_server.control_system.error_handling import (
    ToolError,
    connector_error_handler,
    shared_typed_cause,
)
from osprey.mcp_server.errors import (
    MAP_KEYS,
    MISSING_KEY,
    NOTHING_ANSWERED,
    RECOVERY_KEY,
    RESULTS_KEY,
    NothingAnsweredError,
    answer_map,
    refuse_nothing_answered,
)
from osprey_connectors.errors import ChannelNotFoundError, ChannelWriteBlockedError
from osprey_connectors.ipc.proxy import ChildUnresponsiveError
from tests.mcp_server.test_channel_read_size_rule import _read


def _envelope(excinfo: pytest.ExceptionInfo[ToolError]) -> dict:
    return json.loads(str(excinfo.value))


# ---------------------------------------------------------------------------
# The envelope
# ---------------------------------------------------------------------------


def test_the_contract_names():
    assert MISSING_KEY == "missing"
    assert RESULTS_KEY == "results"
    assert RECOVERY_KEY == "recovery"
    assert NOTHING_ANSWERED == "nothing_answered"


def test_nothing_answered_carries_the_map():
    with pytest.raises(ToolError) as excinfo:
        refuse_nothing_answered("channel_read", "things", {"x": "Gone: no x", "y": None})

    assert _envelope(excinfo) == {
        "error": True,
        "error_type": "nothing_answered",
        "error_message": "channel_read: none of the 2 requested things answered",
        "suggestions": ["Name each entry of details.missing and its text to the operator."],
        "details": {"missing": {"x": "Gone: no x", "y": None}},
    }


def test_nothing_answered_carries_the_recovery():
    recovery = [
        {"action": "connector_dropped", "text": "The archiver connector was dropped."},
        {"action": "child_kept", "text": "The child was kept."},
    ]
    with pytest.raises(ToolError) as excinfo:
        refuse_nothing_answered("channel_read", "things", {"x": None}, recovery=recovery)

    envelope = _envelope(excinfo)
    assert envelope["error_message"] == (
        "channel_read: none of the 1 requested things answered. "
        "The archiver connector was dropped. The child was kept."
    )
    assert envelope["details"] == {"missing": {"x": None}, RECOVERY_KEY: recovery}


def test_nothing_answered_keeps_further_details():
    with pytest.raises(ToolError) as excinfo:
        refuse_nothing_answered("channel_read", "things", {"x": None}, details={"model": "m"})

    assert _envelope(excinfo)["details"] == {"model": "m", "missing": {"x": None}}


ROWS = [
    {"channel": "PV:A", "outcome": "refused", "error": "Write to 'PV:A' refused"},
    {"channel": "PV:B", "outcome": "failed", "error": "caput failed: timeout"},
]


def test_a_write_nothing_answered_carries_its_rows():
    with pytest.raises(ToolError) as excinfo:
        refuse_nothing_answered("channel_write", "channels", results=ROWS)

    assert _envelope(excinfo) == {
        "error": True,
        "error_type": "nothing_answered",
        "error_message": "channel_write: none of the 2 requested channels answered",
        "suggestions": ["Name each entry of details.results and its text to the operator."],
        "details": {"results": ROWS},
    }


@pytest.mark.parametrize(
    ("tool", "given"),
    [("channel_write", {"missing": {"PV:A": None}}), ("channel_read", {"results": ROWS})],
    ids=["write-given-missing", "read-given-rows"],
)
def test_a_tool_states_only_the_map_it_declares(tool, given):
    with pytest.raises(ValueError, match=MAP_KEYS[tool]):
        answer_map(tool, **given)


def test_a_tool_with_no_declared_map_key_states_no_map():
    with pytest.raises(KeyError):
        answer_map("archiver_read", missing={})


def test_the_carrier_holds_one_map():
    with pytest.raises(ValueError, match="not both"):
        NothingAnsweredError(None, missing={"PV:A": None}, results=ROWS)


def test_every_declared_map_key_is_on_the_contract_page():
    """The page's tools table lists every tool whose map is ``missing``; the rest
    each have their own entry naming the key they state."""
    page = (
        Path(__file__).parents[2] / "docs/source/reference/contracts/partial-answers.rst"
    ).read_text()
    tools_table = page.split("The tools\n=========\n", 1)[1].split("Two exceptions", 1)[0]
    listed = set(re.findall(r"^   \* - ``(\w+)``$", tools_table, flags=re.MULTILINE))
    declared_missing = {tool for tool, key in MAP_KEYS.items() if key == MISSING_KEY}
    assert listed == declared_missing

    for tool, key in MAP_KEYS.items():
        if key == MISSING_KEY:
            continue
        entry = page.split(f"\n``{tool}``\n", 1)[1].split("\n\n``", 1)[0]
        assert f"``details.{key}``" in entry


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


async def test_a_shared_cause_keeps_its_own_envelope_with_the_rows():
    cause = ChannelWriteBlockedError("PV:A", "WRITES_DISABLED")
    with pytest.raises(ToolError) as excinfo:
        async with connector_error_handler("channel_write", "archiver"):
            raise NothingAnsweredError(cause, results=ROWS)

    envelope = _envelope(excinfo)
    assert envelope["error_type"] == "write_refused"
    assert envelope["details"] == {"channel": "PV:A", "reason": "WRITES_DISABLED", "results": ROWS}


async def test_no_cause_is_nothing_answered_with_the_rows():
    with pytest.raises(ToolError) as excinfo:
        async with connector_error_handler("channel_write", "archiver"):
            raise NothingAnsweredError(None, results=ROWS)

    envelope = _envelope(excinfo)
    assert envelope["error_type"] == "nothing_answered"
    assert envelope["error_message"] == "channel_write: none of the 2 requested channels answered"
    assert envelope["details"] == {"results": ROWS}


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

    envelope = _envelope(excinfo)
    assert envelope["error_type"] == "connection_error"
    assert envelope["details"][RECOVERY_KEY][0]["action"] == "connector_dropped"
    assert envelope["error_message"].endswith(envelope["details"][RECOVERY_KEY][0]["text"])
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
    invalidate = AsyncMock(return_value=True)
    ping = AsyncMock(return_value={"action": "child_kept", "text": "The child was kept."})
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
    invalidate = AsyncMock(return_value=True)
    ping = AsyncMock(return_value=None)
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

    envelope = _envelope(excinfo)
    assert envelope["error_type"] == "nothing_answered"
    assert RECOVERY_KEY not in envelope["details"]
    invalidate.assert_not_awaited()
    ping.assert_not_awaited()


async def test_mixed_causes_report_the_recovery_they_ran(monkeypatch):
    from osprey.mcp_server.control_system import server_context

    monkeypatch.setattr(server_context, "_registry", None)
    invalidate = AsyncMock(return_value=True)
    errors = [ConnectionError("down"), TimeoutError("late")]
    with (
        patch(
            "osprey.mcp_server.control_system.error_handling.invalidate_active_connector",
            invalidate,
        ),
        pytest.raises(ToolError) as excinfo,
    ):
        async with connector_error_handler("channel_read"):
            raise NothingAnsweredError(None, missing={"A": "x", "B": "y"}, errors=errors)

    envelope = _envelope(excinfo)
    assert envelope["error_type"] == "nothing_answered"
    recovery = envelope["details"][RECOVERY_KEY]
    assert [r["action"] for r in recovery] == ["connector_dropped"]
    assert envelope["error_message"].endswith(recovery[0]["text"])
    invalidate.assert_awaited_once_with("control_system")


async def test_mixed_causes_report_a_replaced_child():
    replaced = {
        "action": "child_replaced",
        "text": "The connector-host child did not answer a ping and was replaced.",
    }
    ping = AsyncMock(return_value=replaced)
    errors = [ChildUnresponsiveError("deadline"), TimeoutError("late")]
    with (
        patch(
            "osprey.mcp_server.control_system.error_handling._answer_unresponsive_child",
            ping,
        ),
        pytest.raises(ToolError) as excinfo,
    ):
        async with connector_error_handler("channel_read"):
            raise NothingAnsweredError(None, missing={"A": "x", "B": "y"}, errors=errors)

    envelope = _envelope(excinfo)
    assert envelope["error_type"] == "nothing_answered"
    assert envelope["details"][RECOVERY_KEY] == [replaced]
    assert replaced["text"] in envelope["error_message"]


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
