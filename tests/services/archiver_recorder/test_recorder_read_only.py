"""The archiver recorder only reads: it never calls a connector write.

The recorder builds its own Channel Access connector outside the connector pool
(:func:`osprey.services.archiver_recorder.recorder._connect`), so neither the
pool's stamp nor its reconnect on a target switch reaches it. That is safe only
while the recorder cannot write. Three claims pin it:

- no name in any of the recorder package's modules is one of the connector's
  write methods;
- sampling through the production reader calls no write method on the connector;
- the connector ``_connect`` builds carries no control target, so the write
  verdict refuses any write through it as ``target_unknown`` and nothing reaches
  Channel Access, even with the tier allowing writes and every switch on.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import pkgutil
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import osprey.services.archiver_recorder as recorder_package
from osprey.services.archiver_recorder import recorder
from osprey.services.archiver_recorder.config import RecorderSettings
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

#: Every module of the recorder service, ``__main__`` included.
MODULES = sorted(
    f"{recorder_package.__name__}.{info.name}"
    for info in pkgutil.iter_modules(recorder_package.__path__)
)


def _settings() -> RecorderSettings:
    return RecorderSettings(
        host="archiver-mongodb",
        port=27017,
        database="osprey_archiver",
        collection="pv_history",
        auth_source="admin",
        username="osprey",
        password_env="MONGO_ROOT_PASSWORD",
        timeout_s=5,
        cadence_sec=1,
        tail_cadence_sec=60,
        poll_sec=30,
        hot_span_hours=48,
        retention_days=30,
    )


def test_the_write_names_are_the_connectors_write_surface():
    """The set the other tests check against is the interface's, not empty by accident."""
    assert {"write_channel", "write_multiple_channels", "write_channel_checked"} <= WRITE_NAMES


def test_the_package_is_the_recorder_and_its_entry_point():
    assert f"{recorder_package.__name__}.recorder" in MODULES
    assert f"{recorder_package.__name__}.__main__" in MODULES


@pytest.mark.parametrize("module_name", MODULES)
def test_no_recorder_module_names_a_connector_write_method(module_name):
    spec = importlib.util.find_spec(module_name)
    assert spec is not None and spec.origin is not None
    tree = ast.parse(Path(spec.origin).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)

    assert names & WRITE_NAMES == set()


def _write_calls(connector: MagicMock) -> list[str]:
    return [call[0] for call in connector.mock_calls if call[0].split(".")[0] in WRITE_NAMES]


async def test_sampling_through_the_reader_calls_no_write_method(monkeypatch):
    connector = MagicMock(spec=ControlSystemConnector)
    connector.read_channel = AsyncMock(
        return_value=ChannelValue(value=1.0, timestamp=datetime.now(tz=UTC))
    )
    monkeypatch.setattr(recorder, "_connect", AsyncMock(return_value=connector))

    read = recorder.channel_access_reader(_settings())
    for _ in range(3):
        assert await read(["SR:BPM01:X", "SR:BPM02:X"]) == {"SR:BPM01:X": 1.0, "SR:BPM02:X": 1.0}

    assert connector.read_channel.await_count == 6
    assert _write_calls(connector) == []


async def test_the_recorders_connector_carries_no_target_and_refuses_a_write(tmp_path, monkeypatch):
    aioca = install_fake_aioca(monkeypatch)
    root = tmp_path / "agent-data"
    isolate_verdict_environment(monkeypatch, root)
    write_control_context(root, owned_by=None, switch={"live": "on", "va": "on"})
    monkeypatch.setattr(
        "osprey_connectors.config.get_config_value", config_reader(dict(WRITES_ALLOWED))
    )

    connector = await recorder._connect()
    result = await connector.write_channel("SR:BPM01:X", 1.0)

    assert connector._control_target is None
    # Armed for every target the connector could have been stamped with, so the
    # missing stamp is the only thing that refuses.
    assert write_verdict(dict(WRITES_ALLOWED), "live").allowed
    assert write_verdict(dict(WRITES_ALLOWED), None).reason == "target_unknown"
    assert result.outcome is WriteOutcome.REFUSED
    assert refusal_phrase("target_unknown") in (result.error_message or "")
    assert aioca.puts == []
