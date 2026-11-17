"""The ``pv_directory`` section at build validation.

The validator sees the profile's overlay: it rebuilds the section from the
dotted or nested ``config:`` keys, fills the simulator port from the
``virtual_accelerator:`` block, reports a malformed section beside every other
problem, and prints what the section cannot serve once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli import build_profile_model, output
from osprey.cli.build_profile import _parse_profile, load_profile
from osprey.services.pv_directory import resolve
from osprey.services.pv_directory._connectors import DIRECTORY_CONNECTORS_AVAILABLE
from osprey_connectors.errors import BuildProfileError

pytestmark = pytest.mark.usefixtures("facility_data_tree")

needs_directory = pytest.mark.skipif(
    not DIRECTORY_CONNECTORS_AVAILABLE,
    reason="the installed connectors have no PV directory modules",
)

_URL = {"url": "https://directory.example/api"}
_KAFKA = {"type": "phoebus_alarm_server", "settings": {"bootstrap_servers": "kafka:9092"}}


@pytest.fixture(autouse=True)
def printed(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The warnings the validator prints, in order; nothing printed before this test."""
    lines: list[str] = []
    monkeypatch.setattr(build_profile_model, "_WARNED_PV_DIRECTORY", set())
    monkeypatch.setattr(output, "warn_fact", lambda _logger, summary, *rest: lines.append(summary))
    return lines


def _profile(config: dict[str, Any], **keys: Any) -> Any:
    return _parse_profile({"name": "x", "data": "data", "config": config, **keys})


def _problems(profile: Any, base: Path) -> str:
    with pytest.raises(BuildProfileError) as caught:
        profile.validate(base)
    return str(caught.value)


@pytest.mark.parametrize(
    ("config", "key"),
    [
        ({"pv_directory": {"type": "channelfinder"}}, "pv_directory.settings.url"),
        ({"pv_directory.type": "channelfinder"}, "pv_directory.settings.url"),
        (
            {"pv_directory.alarm_log": {"type": "phoebus_alarm_logger"}},
            "pv_directory.alarm_log.settings.url",
        ),
        ({"pv_directory.alarm_state": _KAFKA}, "pv_directory.alarm_state.settings.topic"),
        (
            {"pv_directory.alarm_state.replay_file": "demo:no-such-replay"},
            "pv_directory.alarm_state.replay_file",
        ),
        ({"pv_directory.type": "simulator_view"}, "pv_directory.type"),
        ({"pv_directory": ["channelfinder"]}, "pv_directory"),
    ],
    ids=["no-url", "dotted-no-url", "alarm-log-url", "kafka-topic", "replay", "internal", "list"],
)
def test_each_refusal_reaches_the_build(config: dict[str, Any], key: str, tmp_path: Path) -> None:
    assert f"{key}:" in _problems(_profile(config), tmp_path)


def test_a_refusal_reaches_a_profile_loaded_from_disk(tmp_path: Path) -> None:
    path = tmp_path / "profile.yml"
    path.write_text(
        yaml.safe_dump(
            {"name": "x", "data": "data", "config": {"pv_directory.put_log.type": "caputlog"}}
        ),
        encoding="utf-8",
    )

    with pytest.raises(BuildProfileError, match=r"pv_directory\.put_log\.settings\.url"):
        load_profile(path)


def test_a_section_error_and_an_unrelated_error_share_one_message(tmp_path: Path) -> None:
    profile = _profile({"pv_directory.type": "channelfinder"}, deploy_services="yes")

    message = _problems(profile, tmp_path)

    assert "pv_directory.settings.url" in message
    assert "deploy_services must be a boolean" in message


@needs_directory
def test_the_kafka_warning_is_printed_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, printed: list[str]
) -> None:
    monkeypatch.setattr(resolve, "_kafka_client_installed", lambda: False)
    feed = {**_KAFKA, "settings": {**_KAFKA["settings"], "topic": "Accelerator"}}
    profile = _profile({"pv_directory.alarm_state": feed})

    profile.validate(tmp_path)
    profile.validate(tmp_path)

    assert printed == [f"pv_directory.alarm_state: {resolve.KAFKA_CLIENT_MISSING}"]


@needs_directory
def test_a_replay_prints_no_warning(tmp_path: Path, printed: list[str]) -> None:
    _profile({"pv_directory.alarm_state.replay_file": "replays/alarms.ndjson"}).validate(tmp_path)

    assert printed == []


@needs_directory
def test_a_typeless_section_on_epics_warns_only_without_a_simulator(
    tmp_path: Path, printed: list[str]
) -> None:
    _profile({"pv_directory": {}}, connector="epics", virtual_accelerator={}).validate(tmp_path)
    assert printed == []

    _profile({"pv_directory": {}}, connector="epics").validate(tmp_path)
    assert printed == [resolve.NO_SOURCE_WARNING]
