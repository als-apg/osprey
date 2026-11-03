"""``archiver_read`` serves a session only its own target's archive.

One ``archiver:`` block serves the deployment, and it holds the past of some
targets and not others (:mod:`osprey_connectors.archive_targets`): the
facility's appliance for the live machine and the stand-in that rehearses it,
the deployment's own store for the virtual accelerator. The agent's reader of
that block asks :func:`~osprey_connectors.archive_targets.archive_withheld`
before it opens the connector, so a session on ``live`` or ``standin`` is never
shown the simulator's store and a session on ``va`` is never shown the
appliance.

The connector factory is patched out: what is under test is the gate in front
of it, and the served cases must not reach a real store.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
import yaml

from osprey.mcp_server.control_system.server_context import initialize_server_context
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict, get_tool_fn
from tests.mcp_server.test_archiver_read_tool import _make_archiver_df

_READ = {
    "channels": ["SR:DCCT"],
    "start_time": "2026-04-03T10:00:00",
    "end_time": "2026-04-03T11:00:00",
    "bin_size": 60,
}

STORE_BLOCK = {
    "type": "mongodb_archiver",
    "settings": {
        "host": "localhost",
        "port": 27017,  # osprey:not-a-port
        "name": "history",
        "collection": "samples",
        "auth": {"username": "reader", "source": "admin", "password_env": "STORE_PASSWORD"},
    },
}

APPLIANCE_BLOCK = {"type": "epics_archiver", "settings": {"url": "https://archiver.example.org"}}


def _project(tmp_path, monkeypatch, *, archiver: dict, baseline: str, simulator: bool) -> None:
    """A rendered project on *baseline*, with or without a built simulator."""
    monkeypatch.chdir(tmp_path)
    config: dict = {"archiver": archiver, "control_system": {"type": baseline}}
    if simulator:
        config["services"] = {"virtual_accelerator": {"port": 5064}}
    (tmp_path / "config.yml").write_text(yaml.dump(config))
    initialize_server_context()


@pytest.fixture
def archiver_read():
    from osprey.mcp_server.control_system.tools.archiver_read import archiver_read

    connector = AsyncMock()
    connector.get_data.return_value = _make_archiver_df({"SR:DCCT": [500.0, 500.2]})
    with patch(
        "osprey_connectors.factory.ConnectorFactory.create_archiver_connector",
        new_callable=AsyncMock,
        return_value=connector,
    ):
        yield get_tool_fn(archiver_read), connector


async def test_a_stand_in_session_is_not_shown_the_simulators_store(
    tmp_path, monkeypatch, archiver_read, control_context_root, write_control_context
):
    """The owner's shape: baselined on the stand-in, simulator built, mongodb store."""
    _project(tmp_path, monkeypatch, archiver=STORE_BLOCK, baseline="live_standin", simulator=True)
    write_control_context(control_context_root, target="standin")
    fn, connector = archiver_read

    with assert_raises_error(error_type="no_archive") as ctx:
        await fn(**_READ)

    envelope = ctx["envelope"]
    assert envelope["details"] == {"target": "standin", "archive_targets": ["va"]}
    # No controls server has vouched for the stand-in, so it is named the live
    # machine, the cautious words; the store it is refused is the simulator's.
    assert "No archive for the live machine here" in envelope["error_message"]
    assert "holds the past of the simulator" in envelope["error_message"]
    connector.get_data.assert_not_called()


async def test_a_live_session_is_not_shown_the_simulators_store(
    tmp_path, monkeypatch, archiver_read, control_context_root, write_control_context
):
    _project(tmp_path, monkeypatch, archiver=STORE_BLOCK, baseline="live_standin", simulator=True)
    write_control_context(control_context_root, target="live")
    fn, connector = archiver_read

    with assert_raises_error(error_type="no_archive"):
        await fn(**_READ)
    connector.get_data.assert_not_called()


async def test_a_simulator_session_is_shown_its_store_whatever_the_baseline(
    tmp_path, monkeypatch, archiver_read, control_context_root, write_control_context
):
    """Switched to ``va`` from a stand-in baseline: the store is the simulator's."""
    _project(tmp_path, monkeypatch, archiver=STORE_BLOCK, baseline="live_standin", simulator=True)
    write_control_context(control_context_root, target="va")
    fn, connector = archiver_read

    response = extract_response_dict(await fn(**_READ))

    assert response["status"] == "success"
    connector.get_data.assert_awaited_once()


async def test_a_simulator_session_is_not_shown_the_appliance(
    tmp_path, monkeypatch, archiver_read, control_context_root, write_control_context
):
    _project(tmp_path, monkeypatch, archiver=APPLIANCE_BLOCK, baseline="epics", simulator=True)
    write_control_context(control_context_root, target="va")
    fn, connector = archiver_read

    with assert_raises_error(error_type="no_archive") as ctx:
        await fn(**_READ)

    assert ctx["envelope"]["details"] == {"target": "va", "archive_targets": ["live", "standin"]}
    connector.get_data.assert_not_called()


@pytest.mark.parametrize("target", ["live", "standin"])
async def test_the_real_machine_and_its_rehearsal_are_shown_the_appliance(
    tmp_path, monkeypatch, archiver_read, control_context_root, write_control_context, target
):
    _project(tmp_path, monkeypatch, archiver=APPLIANCE_BLOCK, baseline="epics", simulator=True)
    write_control_context(control_context_root, target=target)
    fn, connector = archiver_read

    response = extract_response_dict(await fn(**_READ))

    assert response["status"] == "success"
    connector.get_data.assert_awaited_once()


async def test_a_store_on_a_deployment_that_built_no_simulator_is_nobodys(
    tmp_path, monkeypatch, archiver_read, control_context_root, write_control_context
):
    """A ``va`` block dialling a simulator another host runs: nothing this
    deployment seeded describes it, so even a ``va`` session is told no archive."""
    _project(tmp_path, monkeypatch, archiver=STORE_BLOCK, baseline="epics", simulator=False)
    write_control_context(control_context_root, target="va")
    fn, connector = archiver_read

    with assert_raises_error(error_type="no_archive") as ctx:
        await fn(**_READ)

    assert ctx["envelope"]["details"]["archive_targets"] == []
    connector.get_data.assert_not_called()


async def test_the_mock_archiver_is_left_to_the_pairing(
    tmp_path, monkeypatch, archiver_read, control_context_root, write_control_context
):
    """An unattributed type is withheld from nobody here: whether the mock may
    pair with the session's machine is the honesty rule's question, asked at
    build and at the target switch, not this gate's."""
    _project(
        tmp_path,
        monkeypatch,
        archiver={"type": "mock_archiver"},
        baseline="epics",
        simulator=False,
    )
    write_control_context(control_context_root, target="live")
    fn, connector = archiver_read

    response = extract_response_dict(await fn(**_READ))

    assert response["status"] == "success"
    connector.get_data.assert_awaited_once()
