"""The IOC health addresses of a demo virtual accelerator build, through the ordinary tools.

A deployment on the demo preset's simulator view, served in process, reads the
lane model's ``HEARTBEAT``, ``UPTIME`` and ``ENGINEER`` with ``channel_read``
like any other channel, and the two counters advance between reads. With writes
armed on the ``va`` target a write to a writable setpoint is confirmed, so the
refusals that follow are not the posture's: ``channel_write`` refuses each health
address with the refusal the read-only status address gets.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.audit.posture import OSPREY_AGENT_DATA_ROOT
from osprey.mcp_server.control_system.server_context import (
    initialize_server_context,
    reset_server_context,
)
from osprey.stores.artifact_store import reset_artifact_store
from osprey_connectors.simulation.view import IOC_HEALTH_SIGNALS
from osprey_connectors.workspace import reset_config_cache
from osprey_connectors_tests._harness import write_control_context
from tests._approval_stamp_fixture import ApprovalStamps
from tests.facility.served_tree import in_process_config
from tests.mcp_server.conftest import assert_raises_error
from tests.va.e2e import conftest as e2e_conftest

VIEW = e2e_conftest.PRESET_VIEW
LANE = e2e_conftest.LANE
HEALTH = tuple(VIEW.ioc_health_by_model()[LANE.model].values())
HEARTBEAT, UPTIME, ENGINEER = (
    VIEW.ioc_health_by_model()[LANE.model][signal] for signal in IOC_HEALTH_SIGNALS
)

pytestmark = pytest.mark.usefixtures("deployment")

#: Bound on waiting for the counters to advance; they count whole seconds.
ADVANCE_TIMEOUT_S = 5.0


@pytest.fixture
def deployment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A demo deployment on the in-process simulator, writes armed on ``va``.

    The view is a copy inside ``tmp_path``, so the simulation state the writes
    journal lands there too.
    """
    view = tmp_path / "build" / "data" / "simulator"
    shutil.copytree(VIEW.path, view)
    config = {
        "control_system": {
            "type": "virtual_accelerator",
            "connector": {"virtual_accelerator": in_process_config(view)},
            "writes": "allowed",
            "limits_checking": {
                "enabled": True,
                "database_path": str(e2e_conftest.LIMITS_DB_PATH),
                "mode": "optional",
                "on_violation": "error",
            },
        },
        "archiver": {"type": "mock_archiver"},
        "approval": {
            "enabled": True,
            "default_policy": "always",
            "tools": {"channel_write": "always", "channel_read": "skip"},
        },
    }
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    agent_data = tmp_path / "_agent_data"
    for subdir in ("channel_results", "archiver_data", "python_outputs"):
        (agent_data / subdir).mkdir(parents=True)
    monkeypatch.setenv(OSPREY_AGENT_DATA_ROOT, str(agent_data))
    write_control_context(agent_data, target="va", generation=1, switch={"va": "on"})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OSPREY_CONFIG", str(config_path))
    initialize_server_context()
    yield
    reset_server_context()
    reset_artifact_store()
    reset_config_cache()


def _tool(name: str) -> Any:
    from osprey.mcp_server.control_system.tools import channel_read, channel_write

    tool = {"channel_read": channel_read.channel_read, "channel_write": channel_write.channel_write}
    return getattr(tool[name], "fn", tool[name])


async def _read(*channels: str) -> dict[str, Any]:
    data = json.loads(await _tool("channel_read")(channels=list(channels)))
    assert data["status"] == "success", data
    return data["summary"]["readings"]


async def _write(channel: str, value: Any) -> dict[str, Any]:
    """One approved ``channel_write`` of *value* to *channel*; its reply."""
    from osprey.mcp_server.control_system import target_state
    from osprey.mcp_server.control_system.tools import channel_write as cw

    operations = [{"channel": channel, "value": value}]
    state = target_state.state_dir()
    state.mkdir(parents=True, exist_ok=True)
    binding = cw._read_target_binding()
    target, generation = binding if binding is not None else (None, None)
    with ApprovalStamps(state).approved(
        "channel_write", operations=operations, target=target, generation=generation
    ):
        return json.loads(await _tool("channel_write")(operations=operations))


async def _refusal(channel: str) -> dict[str, Any]:
    """The envelope ``channel_write`` refuses a write of 1 to *channel* with, its name taken out."""
    with assert_raises_error(error_type="write_refused") as refused:
        await _write(channel, 1)
    return json.loads(json.dumps(refused["envelope"]).replace(channel, "<channel>"))


async def test_health_addresses_read_through_channel_read_and_the_counters_advance() -> None:
    first = await _read(*HEALTH)
    deadline = time.monotonic() + ADVANCE_TIMEOUT_S
    while True:
        later = await _read(HEARTBEAT, UPTIME)
        if later[HEARTBEAT]["value"] > first[HEARTBEAT]["value"]:
            break
        assert time.monotonic() < deadline, f"{HEARTBEAT} stayed at {first[HEARTBEAT]}"
        await asyncio.sleep(0.2)

    assert later[UPTIME]["value"] > first[UPTIME]["value"]
    assert isinstance(first[ENGINEER]["value"], str) and first[ENGINEER]["value"]


async def test_armed_writes_reach_a_setpoint_and_are_refused_on_each_health_address() -> None:
    low, high = LANE.band
    written = await _write(LANE.setpoint, (low + high) / 2)
    assert written["status"] == "success", written
    assert written["summary"]["outcomes"] == {"confirmed": 1}

    status = await _refusal(LANE.status_address)
    assert status["details"]["results"][0]["outcome"] == "refused"

    for address in HEALTH:
        assert await _refusal(address) == status, address
