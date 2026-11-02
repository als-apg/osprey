"""Tests for reference-monitor write-refusal classification in the error handler.

A ``ChannelWriteBlockedError`` raised inside the ``connector_error_handler``
context manager must classify as a dedicated ``write_refused`` error type
(carrying the refusal reason and channel), NOT fall through to the generic
``internal_error`` clause.
"""

from __future__ import annotations

import pytest

from osprey.mcp_server.control_system import server_context
from osprey.mcp_server.control_system.connector_host_manager import target_display_metadata
from osprey.mcp_server.control_system.error_handling import (
    _target_clause,
    connector_error_handler,
    describe_active_target,
)
from osprey_connectors.errors import ChannelWriteBlockedError
from tests.mcp_server._switch_harness import state_root  # noqa: F401 - fixture
from tests.mcp_server.conftest import assert_raises_error
from tests.mcp_server.test_control_target_roster import (
    REAL_GATEWAY_HOST,
    REHEARSAL_LABEL,
    TUNNELLED_PORT,
    standin_config,
)


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


# ------------------------------------------- the published label in tool errors
#
# A connector tool error names the machine it failed on through
# ``_target_clause``, which renders the label the controls server published.
# These cases drive the real :func:`target_display_metadata` derivation, so the
# words in the error are the words the chip, the roster and the approval prompt
# read: the deployment's own stand-in is the rehearsal, and a ``standin`` that
# points anywhere else reads as the real machine.


class _DisplayOnly:
    """A supervisor answering the two reads the resolver makes, from a real render."""

    def __init__(self, raw: dict, target: str) -> None:
        self._display = target_display_metadata(raw)
        self._target = target

    def active_target(self) -> str:
        return self._target

    def display_metadata(self) -> dict[str, dict]:
        return self._display


class _Context:
    def __init__(self, raw: dict, target: str = "standin") -> None:
        self.connector_hosts = _DisplayOnly(raw, target)


@pytest.fixture
def on_target(monkeypatch, state_root):  # noqa: ARG001, F811 - isolates the verdict env
    """Point the resolver at a supervisor rendered from *raw*, on *target*."""

    def install(raw: dict, target: str = "standin") -> None:
        context = _Context(raw, target)
        monkeypatch.setattr(server_context, "get_server_context", lambda: context)

    return install


async def _timeout_message() -> str:
    with assert_raises_error(error_type="timeout_error") as ctx:
        async with connector_error_handler("channel_write"):
            raise TimeoutError("no answer")
    return ctx["envelope"]["error_message"]


async def test_the_standin_reads_as_the_rehearsal_in_a_tool_error(on_target):
    """The deployment's own stand-in: the error says the rehearsal words."""
    on_target(standin_config())

    identity = describe_active_target()
    assert identity is not None
    assert identity["label"] == REHEARSAL_LABEL
    assert f"(active target: {REHEARSAL_LABEL} at " in _target_clause(identity)

    message = await _timeout_message()
    assert REHEARSAL_LABEL in message
    assert "Live — writes move hardware" not in message


@pytest.mark.parametrize(
    "config_kwargs",
    [
        {"standin_port": None},
        {"gateway_port": TUNNELLED_PORT},
        {"gateway_host": REAL_GATEWAY_HOST},
    ],
    ids=["no-standin-deployed", "moved-endpoint", "off-loopback"],
)
async def test_a_standin_elsewhere_reads_as_the_real_machine(on_target, config_kwargs):
    """A ``standin`` that is not this deployment's stand-in never says "nothing moves"."""
    on_target(standin_config(**config_kwargs))

    message = await _timeout_message()
    assert "(active target: Live — writes move hardware" in message
    assert "nothing moves" not in message.lower()
    assert "Rehearsal" not in message


async def test_the_facility_machine_is_never_the_rehearsal_in_a_tool_error(on_target):
    on_target(standin_config(), target="live")

    message = await _timeout_message()
    assert "(active target: Live — writes move hardware" in message
    assert "nothing moves" not in message.lower()


def test_the_clause_is_total():
    """No identity, or no endpoint, still renders — an envelope never raises."""
    assert _target_clause(None) == ""
    assert _target_clause({"name": "standin", "label": REHEARSAL_LABEL}) == (
        f" (active target: {REHEARSAL_LABEL})"
    )
