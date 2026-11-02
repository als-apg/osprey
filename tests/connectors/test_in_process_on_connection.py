"""The in-process simulator's ``subscribe(..., on_connection=...)`` report.

The connectors package pins ``on_connection`` for EPICS, DOOCS, TANGO and the
IPC proxy in its own suite. The in-process simulator needs a served tree to
connect to, which only the framework's test helpers build, so its two cases
live here.
"""

from __future__ import annotations

from osprey_connectors.control_system._va_in_process_connector import VAInProcessConnector
from tests.facility.served_tree import in_process_config, served_tree


async def _connected(tmp_path) -> VAInProcessConnector:
    view = served_tree(tmp_path, {"SR:CH:SP": None})
    connector = VAInProcessConnector()
    await connector.connect(in_process_config(view, response_delay_ms=0))
    return connector


async def test_true_once_before_subscribe_returns(tmp_path):
    connector = await _connected(tmp_path)
    states: list = []

    await connector.subscribe("SR:CH:SP", lambda v: None, on_connection=states.append)

    assert states == [True]
    await connector.disconnect()


async def test_a_raising_connection_callback_does_not_fail_subscribe(tmp_path):
    connector = await _connected(tmp_path)

    def boom(up):  # noqa: ARG001 - the on_connection signature; it raises regardless
        raise RuntimeError("bad subscriber")

    assert await connector.subscribe("SR:CH:SP", lambda v: None, on_connection=boom)
    await connector.disconnect()
