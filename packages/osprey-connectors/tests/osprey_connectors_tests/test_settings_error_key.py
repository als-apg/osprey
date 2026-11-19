"""A settings error names the config key the operator wrote.

``service_targets`` reads the section under ``channel_directory``, or under
``pv_directory`` from a config that still spells it that way, and every backend
built from its specs names that spelling in a settings error: the key that is
in the operator's file, not one it has never held.
"""

from __future__ import annotations

import sys
import types

import pytest

from osprey_connectors.alarm_log._phoebus import PhoebusAlarmLogger
from osprey_connectors.alarm_state._phoebus_kafka import PhoebusAlarmState
from osprey_connectors.alarm_state._relay import RelayAlarmState
from osprey_connectors.directory._channelfinder import ChannelFinderDirectory
from osprey_connectors.put_log._caputlog import CaPutLog
from osprey_connectors.service_answers import Unavailable
from osprey_connectors.service_targets import service_targets
from osprey_connectors.types import TARGET_LIVE

DEADLINE = 1.0
KEYS = ("channel_directory", "pv_directory")

#: A url no backend sends: the transport refuses its scheme, naming the key.
_NOT_HTTP = {"url": "ftp://host"}


def _specs(key: str, **blocks: dict[str, object]):
    section = {"type": "channelfinder", "settings": _NOT_HTTP, **blocks}
    config = {"control_system": {"type": "epics"}, key: section}
    return {kind: row[TARGET_LIVE] for kind, row in service_targets(config).items()}


@pytest.mark.parametrize("key", KEYS)
async def test_the_directory_names_its_settings_by_the_section_key(key: str) -> None:
    answer = await ChannelFinderDirectory(_specs(key)["directory"]).probe(DEADLINE)

    assert isinstance(answer, Unavailable)
    assert f"`{key}.settings.url`" in answer.reason


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize(
    ("kind", "service", "backend"),
    [
        ("alarm_log", "phoebus_alarm_logger", PhoebusAlarmLogger),
        ("alarm_state", "osprey_alarm_relay", RelayAlarmState),
        ("put_log", "caputlog", CaPutLog),
    ],
)
async def test_a_kind_beneath_the_section_names_its_own_block(
    key: str, kind: str, service: str, backend: type
) -> None:
    spec = _specs(key, **{kind: {"type": service, "settings": _NOT_HTTP}})[kind]

    answer = await backend(spec).probe(DEADLINE)

    assert isinstance(answer, Unavailable)
    assert f"`{key}.{kind}.settings.url`" in answer.reason


@pytest.mark.parametrize("key", KEYS)
async def test_the_kafka_feed_names_its_login_by_the_section_key(
    monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    monkeypatch.setitem(sys.modules, "aiokafka", types.ModuleType("aiokafka"))
    settings = {
        "bootstrap_servers": "broker:9093",
        "topic": "Demo",
        "auth": {"username": "reader", "password_env": "FEED_PASSWORD", "mechanism": "GSSAPI"},
    }
    block = {"type": "phoebus_alarm_server", "settings": settings}
    backend = PhoebusAlarmState(_specs(key, alarm_state=block)["alarm_state"])
    await backend.start()

    answer = await backend.probe(DEADLINE)

    assert isinstance(answer, Unavailable)
    assert f"`{key}.alarm_state.settings.auth.mechanism`" in answer.reason
