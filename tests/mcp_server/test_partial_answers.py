"""A request answered in part, or not at all, has one shape.

A result names what was not answered under ``missing``; a request nothing
answered is refused with the same map under ``details.missing``, as its one
shared cause's own error when every item failed alike and as
``nothing_answered`` otherwise. A write's map is its rows, under ``results``;
which key a tool states its map under is declared once, in ``MAP_KEYS``.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from osprey.mcp_server.control_system.error_handling import (
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
from osprey_connectors.errors import (
    ChannelNotFoundError,
    ChannelUnreachableError,
    ChannelWriteBlockedError,
    ConnectorUnavailableError,
)
from osprey_connectors.ipc import ChildUnresponsiveError
from tests.mcp_server.conftest import assert_raises_error
from tests.mcp_server.test_channel_read_size_rule import _read

# ---------------------------------------------------------------------------
# The envelope
# ---------------------------------------------------------------------------


def test_the_contract_names():
    assert MISSING_KEY == "missing"
    assert RESULTS_KEY == "results"
    assert RECOVERY_KEY == "recovery"
    assert NOTHING_ANSWERED == "nothing_answered"


def test_nothing_answered_carries_the_map():
    with assert_raises_error() as raised:
        refuse_nothing_answered("channel_read", "things", {"x": "Gone: no x", "y": None})

    assert raised["envelope"] == {
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
    with assert_raises_error() as raised:
        refuse_nothing_answered("channel_read", "things", {"x": None}, recovery=recovery)

    envelope = raised["envelope"]
    assert envelope["error_message"] == (
        "channel_read: none of the 1 requested things answered. "
        "The archiver connector was dropped. The child was kept."
    )
    assert envelope["details"] == {"missing": {"x": None}, RECOVERY_KEY: recovery}


def test_nothing_answered_keeps_further_details():
    with assert_raises_error() as raised:
        refuse_nothing_answered("channel_read", "things", {"x": None}, details={"model": "m"})

    assert raised["envelope"]["details"] == {"model": "m", "missing": {"x": None}}


ROWS = [
    {"channel": "PV:A", "outcome": "refused", "error": "Write to 'PV:A' refused"},
    {"channel": "PV:B", "outcome": "failed", "error": "caput failed: timeout"},
]


def test_a_write_nothing_answered_carries_its_rows():
    with assert_raises_error() as raised:
        refuse_nothing_answered("channel_write", "channels", results=ROWS)

    assert raised["envelope"] == {
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
    with assert_raises_error() as raised:
        async with connector_error_handler("channel_read", "archiver"):
            raise NothingAnsweredError(TimeoutError("a"), missing=missing)

    assert raised["envelope"] == {
        "error": True,
        "error_type": "timeout_error",
        "error_message": "channel_read timed out: a",
        "suggestions": ["Check network connectivity.", "Try a smaller request."],
        "details": {"missing": missing},
    }


async def test_no_cause_is_nothing_answered_with_the_map():
    missing = {"A": "RuntimeError: a", "B": "RuntimeError: b"}
    with assert_raises_error() as raised:
        async with connector_error_handler("channel_read", "archiver"):
            raise NothingAnsweredError(None, missing=missing)

    envelope = raised["envelope"]
    assert envelope["error_type"] == "nothing_answered"
    assert envelope["error_message"] == "channel_read: none of the 2 requested channels answered"
    assert envelope["details"] == {"missing": missing}


async def test_a_shared_cause_keeps_its_own_envelope_with_the_rows():
    cause = ChannelWriteBlockedError("PV:A", "WRITES_DISABLED")
    with assert_raises_error() as raised:
        async with connector_error_handler("channel_write", "archiver"):
            raise NothingAnsweredError(cause, results=ROWS)

    envelope = raised["envelope"]
    assert envelope["error_type"] == "write_refused"
    assert envelope["details"] == {"channel": "PV:A", "reason": "WRITES_DISABLED", "results": ROWS}


async def test_no_cause_is_nothing_answered_with_the_rows():
    with assert_raises_error() as raised:
        async with connector_error_handler("channel_write", "archiver"):
            raise NothingAnsweredError(None, results=ROWS)

    envelope = raised["envelope"]
    assert envelope["error_type"] == "nothing_answered"
    assert envelope["error_message"] == "channel_write: none of the 2 requested channels answered"
    assert envelope["details"] == {"results": ROWS}


#: What the supervisor attaches to a failure it recovered from.
DROPPED = {
    "action": "connector_dropped",
    "text": "The archiver connector was dropped; the next call builds a new one.",
}
KEPT = {"action": "child_kept", "text": "The connector-host child was kept."}
REPLACED = {
    "action": "child_replaced",
    "text": "The connector-host child did not answer a ping and was replaced.",
}


def _unresponsive(message: str, record: dict) -> ChildUnresponsiveError:
    error = ChildUnresponsiveError(message)
    error.recovery = (record,)
    return error


async def test_the_envelope_renders_the_supervisors_recovery():
    cause = ConnectorUnavailableError("down", recovery=(DROPPED,))
    with assert_raises_error() as raised:
        async with connector_error_handler("channel_read", "archiver"):
            raise NothingAnsweredError(cause, missing={"A": "x", "B": "y", "C": "z"})

    envelope = raised["envelope"]
    assert envelope["error_type"] == "connection_error"
    assert envelope["details"][RECOVERY_KEY] == [DROPPED]
    assert envelope["error_message"].endswith(DROPPED["text"])


async def test_an_unreachable_channel_recovers_nothing():
    cause = ChannelUnreachableError("SR:DEAD", "SR:DEAD did not answer")
    with assert_raises_error() as raised:
        async with connector_error_handler("channel_read", "archiver"):
            raise NothingAnsweredError(cause, missing={"SR:DEAD": "x"})

    envelope = raised["envelope"]
    assert envelope["error_type"] == "channel_unreachable"
    assert envelope["details"]["channel"] == "SR:DEAD"
    assert RECOVERY_KEY not in envelope["details"]


class _UnresponsiveProxy:
    """A connector-host proxy whose every read misses its deadline."""

    async def read_channel(self, address: str, timeout: float | None = None):  # noqa: ARG002 - the connector's read signature
        raise ChildUnresponsiveError(f"no answer for {address}")


class _OneChild:
    """A started manager with one live child, the proxy above."""

    def __init__(self) -> None:
        self.proxy = _UnresponsiveProxy()

    def is_started(self) -> bool:
        return True

    async def ensure_started(self, target=None) -> bool:  # noqa: ARG002 - the manager's signature
        return True

    async def serving_proxy(self):
        return self.proxy

    def active_proxy(self):
        return self.proxy

    def active_binding(self):
        return ("live", 0)


async def test_a_wedged_child_is_pinged_once(monkeypatch):
    """A batch the child did not answer runs one ping, and every failure reports it."""
    from osprey.mcp_server.control_system import server_context
    from osprey.mcp_server.control_system.tools.channel_read import _gather_readings

    context = server_context.ControlSystemContext()
    context._config = server_context.MCPServerConfig(raw={})
    context._switch_capable = True
    context._connector_hosts = _OneChild()
    context._connectors["control_system"] = server_context.ConnectorEntry(
        config={}, connector_type="control_system"
    )
    monkeypatch.setattr(server_context, "_registry", context)
    ping = AsyncMock(return_value=dict(KEPT))
    monkeypatch.setattr(context, "_answer_unresponsive", ping)

    with assert_raises_error() as raised:
        async with connector_error_handler("channel_read"):
            connector = await context.control_system()
            await _gather_readings(connector, ["A", "B"])

    envelope = raised["envelope"]
    assert envelope["error_type"] == "timeout_error"
    assert envelope["details"][RECOVERY_KEY] == [KEPT]
    ping.assert_awaited_once()


async def test_mixed_causes_report_every_recovery_once_in_first_seen_order():
    errors = [
        ConnectorUnavailableError("down", recovery=(DROPPED,)),
        ConnectorUnavailableError("refused", recovery=(DROPPED,)),
        _unresponsive("deadline", KEPT),
        _unresponsive("deadline again", KEPT),
        TimeoutError("late"),
    ]
    with assert_raises_error() as raised:
        async with connector_error_handler("channel_read"):
            raise NothingAnsweredError(
                None, missing={str(i): str(e) for i, e in enumerate(errors)}, errors=errors
            )

    envelope = raised["envelope"]
    assert envelope["error_type"] == "nothing_answered"
    assert envelope["details"][RECOVERY_KEY] == [DROPPED, KEPT]


async def test_mixed_causes_without_a_recovery_report_none():
    errors = [TimeoutError("late"), ChannelNotFoundError("B")]
    with assert_raises_error() as raised:
        async with connector_error_handler("channel_read"):
            raise NothingAnsweredError(None, missing={"A": "x", "B": "y"}, errors=errors)

    envelope = raised["envelope"]
    assert envelope["error_type"] == "nothing_answered"
    assert RECOVERY_KEY not in envelope["details"]


async def test_mixed_causes_report_the_recovery_they_carry(monkeypatch):
    from osprey.mcp_server.control_system import server_context

    monkeypatch.setattr(server_context, "_registry", None)
    errors = [ConnectorUnavailableError("down", recovery=(DROPPED,)), TimeoutError("late")]
    with assert_raises_error() as raised:
        async with connector_error_handler("channel_read"):
            raise NothingAnsweredError(None, missing={"A": "x", "B": "y"}, errors=errors)

    envelope = raised["envelope"]
    assert envelope["error_type"] == "nothing_answered"
    recovery = envelope["details"][RECOVERY_KEY]
    assert [r["action"] for r in recovery] == ["connector_dropped"]
    assert envelope["error_message"].endswith(recovery[0]["text"])


async def test_mixed_causes_report_a_replaced_child():
    errors = [_unresponsive("deadline", REPLACED), TimeoutError("late")]
    with assert_raises_error() as raised:
        async with connector_error_handler("channel_read"):
            raise NothingAnsweredError(None, missing={"A": "x", "B": "y"}, errors=errors)

    envelope = raised["envelope"]
    assert envelope["error_type"] == "nothing_answered"
    assert envelope["details"][RECOVERY_KEY] == [REPLACED]
    assert REPLACED["text"] in envelope["error_message"]


# ---------------------------------------------------------------------------
# channel_read, the reference tool
# ---------------------------------------------------------------------------


async def test_a_read_nothing_answered_for_one_cause_is_that_cause(tmp_path, monkeypatch):
    values = {"SR:A": TimeoutError("late"), "SR:B": TimeoutError("later")}
    with assert_raises_error() as raised:
        await _read(tmp_path, monkeypatch, values)

    envelope = raised["envelope"]
    assert envelope["error_type"] == "timeout_error"
    assert envelope["details"]["missing"] == {
        "SR:A": "TimeoutError: late",
        "SR:B": "TimeoutError: later",
    }


async def test_a_read_nothing_answered_for_mixed_causes_is_nothing_answered(tmp_path, monkeypatch):
    values = {"SR:A": TimeoutError("late"), "SR:B": ChannelNotFoundError("SR:B")}
    with assert_raises_error() as raised:
        await _read(tmp_path, monkeypatch, values)

    envelope = raised["envelope"]
    assert envelope["error_type"] == "nothing_answered"
    assert envelope["details"]["missing"] == {
        "SR:A": "TimeoutError: late",
        "SR:B": "ChannelNotFoundError: SR:B is not a channel of this facility",
    }


async def test_a_read_nothing_answered_for_an_untyped_cause_is_nothing_answered(
    tmp_path, monkeypatch
):
    values = {"SR:A": RuntimeError("odd"), "SR:B": RuntimeError("odder")}
    with assert_raises_error() as raised:
        await _read(tmp_path, monkeypatch, values)

    envelope = raised["envelope"]
    assert envelope["error_type"] == "nothing_answered"
    assert envelope["details"]["missing"] == {
        "SR:A": "RuntimeError: odd",
        "SR:B": "RuntimeError: odder",
    }


async def test_a_read_that_timed_out_and_lost_its_link_drops_the_connector(tmp_path, monkeypatch):
    values = {"SR:A": TimeoutError("late"), "SR:B": ConnectionError("down")}
    with assert_raises_error() as raised:
        await _read(tmp_path, monkeypatch, values)

    envelope = raised["envelope"]
    assert envelope["error_type"] == "nothing_answered"
    assert envelope["details"]["missing"] == {
        "SR:A": "TimeoutError: late",
        "SR:B": "ConnectorUnavailableError: down",
    }
    assert [r["action"] for r in envelope["details"][RECOVERY_KEY]] == ["connector_dropped"]
