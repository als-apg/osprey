"""A CANOPY process whose transport goes stale leaves with 76, and only then.

The backend runs a stale task beside its move watch. When the session says the
transport has gone stale the process ends with
:data:`~osprey.runtime.session.EXIT_TRANSPORT_STALE` at once — no session
teardown, because closing a stale EPICS client blocks the loop — and the
supervisor replaces it. Where staleness cannot occur (the fake until it is told,
an unservable session) the task waits for good, and closing the backend stops it.

Everything here runs in process with the exit replaced.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import pytest
from canopy.epics.config import EpicsConfig
from canopy.epics.gateway import build_connector

from osprey.integrations.canopy import backend as backend_module
from osprey.integrations.canopy.backend import RENDERED_FOR_ENV
from osprey.interfaces.web_terminal.supervised_child import PREVIOUS_EXIT_ENV
from osprey.runtime import session as session_module
from osprey.runtime.session import (
    EXIT_TRANSPORT_STALE,
    STATE_BOUND,
    STATE_UNSERVABLE,
)
from osprey_connectors import control_context, posture_store, types
from osprey_connectors.factory import ConnectorFactory
from osprey_connectors.identity import AUDIT_IDENTITY_ENV, TERMINAL_USER_ENV
from osprey_connectors_tests._fake_connector import FakeConnector
from osprey_connectors_tests._harness import isolated_connector_registries

# start() arms the real raw-put block in this process; put back what it patched.
pytestmark = pytest.mark.usefixtures("raw_put_block_restored")

BACKEND_PATH = "osprey.integrations.canopy.backend:OspreyPVConnector"

#: How long a process that cannot go stale is watched for an exit.
QUIET_S = 3.0

STALE_LINE = "stale transport — exit 76"

VA_SECTION = {
    "type": types.VIRTUAL_ACCELERATOR,
    "connector": {types.VIRTUAL_ACCELERATOR: {"response_delay_ms": 0}},
}


class _Exited(BaseException):
    """Stands in for the process ending."""

    def __init__(self, status: int) -> None:
        super().__init__(status)
        self.status = status


class _TrackedFake(FakeConnector):
    """A fake that counts its connected instances."""

    connected = 0

    async def connect(self, config):
        await super().connect(config)
        type(self).connected += 1

    async def disconnect(self):
        type(self).connected -= 1
        await super().disconnect()


class _BrokenStaleFake(_TrackedFake):
    """A fake whose stale probe fails."""

    async def wait_stale(self) -> None:
        raise RuntimeError("the stale probe broke")


class _RefusingConnector(FakeConnector):
    """A connector whose control system is not there: the session is unservable."""

    async def connect(self, config):  # noqa: ARG002 - the connector connect signature
        raise ConnectionError("no gateway answers")


def _write_record(state_dir: Path) -> None:
    control_context.write_record(
        control_context.ControlContext(target="va", generation=3, posture={}),
        path=state_dir / control_context.RECORD_FILENAME,
    )
    control_context.invalidate_cache()


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    """A session environment bound to ``va@3``, with the exit replaced.

    Yields the list of exit statuses and a function registering the connector
    class the virtual accelerator is served by.
    """
    directory = tmp_path / "control_target"
    directory.mkdir()
    config = tmp_path / "config.yml"
    config.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv(posture_store.CONTROL_CONTEXT_DIR_ENV_VAR, str(directory))
    monkeypatch.setenv(AUDIT_IDENTITY_ENV, "test-operator")
    monkeypatch.delenv(TERMINAL_USER_ENV, raising=False)
    monkeypatch.setenv("OSPREY_CONFIG", str(config))
    monkeypatch.setenv("CONFIG_FILE", str(config))
    for name in list(os.environ):
        if name.startswith(("OSPREY_CONTROL_TARGET", "EPICS_", "CANOPY_EPICS__")):
            monkeypatch.delenv(name)
    for name in (
        posture_store.CONTROL_CONTEXT_TREE_ENV_VAR,
        posture_store.CONTROL_OWNER_ENV_VAR,
        posture_store.LAUNCH_POSTURE_ENV_VAR,
        "OSPREY_EXECUTION_MODE",
        PREVIOUS_EXIT_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(session_module, "MOVE_POLL_S", 0.01)
    monkeypatch.chdir(tmp_path)

    def get_config_value(path, default=None, _config_path=None):
        return VA_SECTION if path == "control_system" else default

    monkeypatch.setattr("osprey_connectors.config.get_config_value", get_config_value)

    exits: list[int] = []

    def fake_exit(status):
        exits.append(status)
        raise _Exited(status)

    monkeypatch.setattr(backend_module.os, "_exit", fake_exit)
    _TrackedFake.connected = 0
    _write_record(directory)
    monkeypatch.setenv(RENDERED_FOR_ENV, "va@3")
    with isolated_connector_registries():

        def serve_with(connector_class) -> None:
            ConnectorFactory.register_control_system(types.VIRTUAL_ACCELERATOR, connector_class)

        serve_with(_TrackedFake)
        yield exits, serve_with
    control_context.invalidate_cache()


def _backend():
    return build_connector(EpicsConfig(backend=BACKEND_PATH, connect_report_s=5.0))


async def _quiet(task: asyncio.Task) -> None:
    """Assert *task* neither ends nor ends the process for :data:`QUIET_S`."""
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(task), QUIET_S)
    assert not task.done()


def test_the_stale_status_is_76():
    """The supervisor matches on 76."""
    assert EXIT_TRANSPORT_STALE == 76


class TestStaleExit:
    """When the stale task ends the process, and how."""

    async def test_going_stale_exits_76_once_and_tears_nothing_down(self, deployment, caplog):
        """``go_stale()``: exactly one exit 76; the session stays open, the move watch runs."""
        exits, _ = deployment
        backend = _backend()
        await backend.start()
        try:
            assert backend._session.binding.state == STATE_BOUND
            stale, watcher = backend._stale_watcher, backend._watcher
            assert stale is not None and not stale.done()
            caplog.set_level(logging.DEBUG)
            backend._session._connector.go_stale()
            with pytest.raises(_Exited):
                await asyncio.wait_for(stale, 5.0)
            assert exits == [EXIT_TRANSPORT_STALE]
            assert STALE_LINE in caplog.text
            # No teardown: the session and its connector are as they were.
            assert backend._session is not None
            assert _TrackedFake.connected == 1
            assert not watcher.done()
            await asyncio.sleep(0.2)
            assert exits == [EXIT_TRANSPORT_STALE]
        finally:
            await backend.close()
        assert exits == [EXIT_TRANSPORT_STALE]

    async def test_the_exit_goes_through_the_flushing_exit(self, deployment, monkeypatch):
        """The status leaves through ``_exit_process``, which flushes the log first."""
        exits, _ = deployment
        called: list[int] = []
        real = backend_module._exit_process

        def spy(status):
            called.append(status)
            real(status)

        monkeypatch.setattr(backend_module, "_exit_process", spy)
        backend = _backend()
        await backend.start()
        try:
            backend._session._connector.go_stale()
            with pytest.raises(_Exited):
                await asyncio.wait_for(backend._stale_watcher, 5.0)
            assert called == exits == [EXIT_TRANSPORT_STALE]
        finally:
            await backend.close()

    async def test_a_fake_that_is_not_told_never_exits(self, deployment):
        """Without ``go_stale()`` the task waits; closing cancels it with no exit."""
        exits, _ = deployment
        backend = _backend()
        await backend.start()
        stale = backend._stale_watcher
        try:
            await _quiet(stale)
            assert exits == []
        finally:
            await backend.close()
        assert stale.cancelled()
        assert backend._stale_watcher is None
        assert exits == []
        assert _TrackedFake.connected == 0

    async def test_an_unservable_session_never_exits(self, deployment):
        """No connector, nothing that could go stale: no exit within the watch."""
        exits, serve_with = deployment
        serve_with(_RefusingConnector)
        backend = _backend()
        await backend.start()
        stale = backend._stale_watcher
        try:
            assert backend._session.binding.state == STATE_UNSERVABLE
            await _quiet(stale)
            assert exits == []
        finally:
            await backend.close()
        assert stale.cancelled()
        assert exits == []

    async def test_a_failing_stale_probe_is_logged_and_does_not_exit(self, deployment, caplog):
        """An exception from ``wait_stale`` ends the task, not the process."""
        exits, serve_with = deployment
        serve_with(_BrokenStaleFake)
        caplog.set_level(logging.DEBUG)
        backend = _backend()
        await backend.start()
        try:
            stale = backend._stale_watcher
            await asyncio.wait_for(stale, 5.0)
            assert stale.exception() is None
            assert "the stale probe broke" in caplog.text
            assert exits == []
            assert not backend._watcher.done()
            assert backend._session is not None
        finally:
            await backend.close()
        assert exits == []

    async def test_closing_first_stops_the_task_and_going_stale_later_is_ignored(self, deployment):
        """An ordinary shutdown cancels the stale task; nothing is left to exit."""
        exits, _ = deployment
        backend = _backend()
        await backend.start()
        connector = backend._session._connector
        stale = backend._stale_watcher
        await backend.close()
        assert stale.cancelled()
        connector.go_stale()
        await asyncio.sleep(0.1)
        assert exits == []

    async def test_close_is_idempotent_with_the_stale_task(self, deployment):
        """A second close finds nothing to cancel."""
        exits, _ = deployment
        backend = _backend()
        await backend.start()
        await backend.close()
        await backend.close()
        assert backend._stale_watcher is None
        assert exits == []
