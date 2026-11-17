"""A slow child is answered as slow and a lost pipe as lost, wherever the framework classifies.

The link to a connector-host child raises three classes of its own, all
:class:`~osprey_connectors.errors.ChildLinkError`: a missed proxy deadline
(:class:`~osprey_connectors.ipc.ChildUnresponsiveError`) and a missed call on a
child kept because it still answers a ping
(:class:`~osprey_connectors.ipc.ConnectorHostSlowCallError`), both
timeouts under no contract root, and a lost pipe
(:class:`~osprey_connectors.ipc.ChildLostError`), a
``ConnectorUnavailableError``. The link base is no root, so each is read by
the base its meaning names. Three orders read them, and each is pinned here
against the real classes: the error handler's ``TYPED_CAUSES`` grouping, its
envelope, and the ``except`` order of the server's supervised call.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from osprey.mcp_server.control_system.error_handling import (
    TYPED_CAUSES,
    connector_error_handler,
    shared_typed_cause,
)
from osprey.mcp_server.errors import RECOVERY_KEY
from osprey_connectors.errors import (
    ChildLinkError,
    ConnectorUnavailableError,
    RecoveryAction,
    RecoveryRecord,
)
from osprey_connectors.ipc import (
    ChildLostError,
    ChildUnresponsiveError,
    ConnectorHostSlowCallError,
)
from tests.mcp_server.conftest import assert_raises_error

KEPT = RecoveryRecord(action=RecoveryAction.CHILD_KEPT, text="The connector-host child was kept.")
REPLACED = RecoveryRecord(
    action=RecoveryAction.CHILD_REPLACED,
    text="The connector-host child that served this call has been replaced.",
)


def _slow() -> ConnectorHostSlowCallError:
    return ConnectorHostSlowCallError("the child answered a ping but not the call", (KEPT,))


#: Each link failure, the typed cause it groups under, and the envelope it renders.
_LINK_FAILURES = [
    pytest.param(
        lambda: ChildUnresponsiveError("the child did not answer"),
        ChildUnresponsiveError,
        "timeout_error",
        id="proxy-deadline",
    ),
    pytest.param(_slow, TimeoutError, "timeout_error", id="slow-call-kept"),
    pytest.param(
        lambda: ChildLostError("the connector-host child closed its output stream"),
        ConnectionError,
        "connection_error",
        id="lost-pipe",
    ),
]


def _typed(exc: BaseException) -> type[BaseException] | None:
    return next((kind for kind in TYPED_CAUSES if isinstance(exc, kind)), None)


@pytest.mark.parametrize(("make", "cause", "envelope"), _LINK_FAILURES)
async def test_a_link_failure_is_answered_as_its_meaning_names(make, cause, envelope):
    exc = make()

    assert isinstance(exc, ChildLinkError)
    assert _typed(exc) is cause
    assert shared_typed_cause([exc, make()]) is exc
    with assert_raises_error() as raised:
        async with connector_error_handler("channel_read"):
            raise exc
    assert raised["envelope"]["error_type"] == envelope


def test_a_slow_call_never_groups_with_a_lost_pipe():
    slow = [ChildUnresponsiveError("late"), _slow()]
    lost = ChildLostError("gone")

    for exc in slow:
        assert _typed(exc) is not ConnectionError
        assert shared_typed_cause([exc, lost]) is None


class _FailingProxy:
    """A connector-host proxy whose every read fails with the one error a case names."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    async def read_channel(self, address: str, *, timeout_s: float | None = None):  # noqa: ARG002 - the connector's read signature
        raise self.exc


class _OneChild:
    """A started manager with one live child, the proxy above."""

    def __init__(self, proxy: _FailingProxy) -> None:
        self.proxy = proxy

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


@pytest.fixture
def served(monkeypatch):
    """A control-system context served from one child, its two recoveries recorded."""
    from osprey.mcp_server.control_system import server_context

    def serve(exc: BaseException):
        context = server_context.ControlSystemContext()
        context._config = server_context.MCPServerConfig(raw={})
        context._switch_capable = True
        context._connector_hosts = _OneChild(_FailingProxy(exc))
        context._connectors["control_system"] = server_context.ConnectorEntry(
            config={}, connector_type="control_system"
        )
        monkeypatch.setattr(server_context, "_registry", context)
        probe = AsyncMock(return_value=KEPT)
        replace = AsyncMock(return_value=REPLACED)
        monkeypatch.setattr(context, "_answer_unresponsive", probe)
        monkeypatch.setattr(context, "_replace", replace)
        return context, probe, replace

    return serve


async def test_a_missed_deadline_is_probed_and_never_replaced(served):
    context, probe, replace = served(ChildUnresponsiveError("no answer for SR:A"))

    connector = await context.control_system()
    with pytest.raises(ChildUnresponsiveError) as caught:
        await connector.read_channel("SR:A")

    probe.assert_awaited_once()
    replace.assert_not_awaited()
    assert caught.value.recovery == (KEPT,)


async def test_a_lost_pipe_is_replaced_and_served_as_the_connectors_own_loss(served):
    lost = ChildLostError("the connector-host child closed its output stream")
    context, probe, replace = served(lost)

    connector = await context.control_system()
    with pytest.raises(ConnectorUnavailableError) as caught:
        await connector.read_channel("SR:A")

    replace.assert_awaited_once()
    probe.assert_not_awaited()
    # The class a tool names a failure by is the one a pipe end always served.
    assert type(caught.value) is ConnectorUnavailableError
    assert str(caught.value) == str(lost)
    assert caught.value.recovery == (REPLACED,)
    assert caught.value.__cause__ is lost


async def test_a_slow_call_on_a_kept_child_is_neither_probed_again_nor_replaced(served):
    context, probe, replace = served(_slow())

    connector = await context.control_system()
    with assert_raises_error() as raised:
        async with connector_error_handler("channel_read"):
            await connector.read_channel("SR:A")

    probe.assert_not_awaited()
    replace.assert_not_awaited()
    assert raised["envelope"]["error_type"] == "timeout_error"
    assert raised["envelope"]["details"][RECOVERY_KEY] == [KEPT.model_dump(mode="json")]
