"""The EPICS dispatch trigger source only reads: it never calls a connector write.

The source builds its own Channel Access connector outside the connector pool
(:func:`osprey.dispatch.sources.epics_ca._connect`), so neither the pool's stamp
nor its reconnect on a target switch reaches it. That is safe only while the
source cannot write. Three claims pin it:

- no name in the module's source is one of the connector's write methods;
- a full start, crossing and stop cycle calls no write method on the connector;
- the connector ``_connect`` builds carries no control target, so the write
  verdict refuses any write through it as ``target_unknown`` and nothing reaches
  Channel Access, even with the tier allowing writes and every switch on.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from osprey.dispatch.sources import epics_ca
from osprey.dispatch.sources.epics_ca import EpicsCaSource
from osprey.dispatch.trigger_config import TriggerConfig
from osprey_connectors.control_system._base import (
    ChannelValue,
    ControlSystemConnector,
    WriteOutcome,
)
from osprey_connectors.posture_store import write_verdict
from osprey_connectors.refusals import refusal_phrase
from osprey_connectors_tests._epics_fakes import install_fake_aioca
from osprey_connectors_tests._harness import isolate_verdict_environment, write_control_context
from osprey_connectors_tests._write_fakes import config_reader

#: Every method or property on the connector interface that writes, or decides a write.
WRITE_NAMES = frozenset(
    name for name, _ in inspect.getmembers(ControlSystemConnector) if "write" in name.lower()
)

#: A section whose tier allows writes, so only the missing target can refuse.
WRITES_ALLOWED = {"type": "epics", "writes": "allowed"}


def test_the_write_names_are_the_connectors_write_surface():
    """The set the other tests check against is the interface's, not empty by accident."""
    assert {"write_channel", "write_multiple_channels", "write_channel_checked"} <= WRITE_NAMES


def test_the_source_names_no_connector_write_method():
    names: set[str] = set()
    for node in ast.walk(ast.parse(inspect.getsource(epics_ca))):
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)

    assert names & WRITE_NAMES == set()


def _trigger(name: str, pv: str) -> TriggerConfig:
    return TriggerConfig(
        name=name,
        source="epics_ca",
        source_config={"pv": pv, "threshold": 5.0, "edge": "both", "cool_down_sec": 0},
        action={"prompt": "go"},
    )


def _update(value: float) -> ChannelValue:
    return ChannelValue(value=value, timestamp=datetime.now(tz=UTC))


def _write_calls(connector: MagicMock) -> list[str]:
    return [call[0] for call in connector.mock_calls if call[0].split(".")[0] in WRITE_NAMES]


async def test_a_start_fire_and_stop_cycle_calls_no_write_method():
    connector = MagicMock(spec=ControlSystemConnector)
    callbacks: dict[str, Any] = {}

    async def _subscribe(pv, callback):
        callbacks[pv] = callback
        return f"sub-{pv}"

    connector.subscribe = AsyncMock(side_effect=_subscribe)
    fired: list[tuple[str, float]] = []

    async def _fire(trigger, payload):
        fired.append((trigger.name, payload["value"]))

    source = EpicsCaSource()
    with patch.object(epics_ca, "_connect", AsyncMock(return_value=connector)):
        await source.start([_trigger("a", "SR:A"), _trigger("b", "SR:B")], _fire)
    for pv in ("SR:A", "SR:B"):
        for value in (1.0, 9.0, 1.0):  # first read, then a rising and a falling crossing
            callbacks[pv](_update(value))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await source.stop()

    assert sorted(fired) == [("a", 1.0), ("a", 9.0), ("b", 1.0), ("b", 9.0)]
    assert _write_calls(connector) == []
    connector.unsubscribe.assert_awaited()
    connector.disconnect.assert_awaited_once()


async def test_the_sources_connector_carries_no_target_and_refuses_a_write(tmp_path, monkeypatch):
    aioca = install_fake_aioca(monkeypatch)
    root = tmp_path / "agent-data"
    isolate_verdict_environment(monkeypatch, root)
    write_control_context(root, owned_by=None, switch={"live": "on", "va": "on"})
    monkeypatch.setattr(
        "osprey_connectors.config.get_config_value", config_reader(dict(WRITES_ALLOWED))
    )

    connector = await epics_ca._connect()
    result = await connector.write_channel("SR:A", 1.0)

    assert connector._control_target is None
    # Armed for every target the connector could have been stamped with, so the
    # missing stamp is the only thing that refuses.
    assert write_verdict(dict(WRITES_ALLOWED), "live").allowed
    assert write_verdict(dict(WRITES_ALLOWED), None).reason == "target_unknown"
    assert result.outcome is WriteOutcome.REFUSED
    assert refusal_phrase("target_unknown") in (result.error_message or "")
    assert aioca.puts == []
