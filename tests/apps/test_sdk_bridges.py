"""``osprey.apps.sdk.bridges_for_target``: the Bluesky bridges a deployment runs for a target.

An application addresses the bridges a target has through the SDK, never by
reading the rendered ``services.bluesky*`` blocks itself. Each test renders a
``config.yml`` under ``tmp_path`` and points the config loader at it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.apps.sdk import bridges_for_target
from osprey.bluesky_bridge_connection import LANE_KEYS, lane_env_prefix
from osprey_connectors.workspace import reset_config_cache

LIVE_PORT = 9311
VA_PORT = 9312


def _control_system(kind: str = "epics") -> dict[str, Any]:
    return {"type": kind, "connector": {"epics": {"prefix": "RING:"}}}


def _render(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: dict[str, Any]) -> None:
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv("OSPREY_CONFIG", str(path))
    reset_config_cache()


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch):
    for lane in LANE_KEYS:
        monkeypatch.delenv(f"{lane_env_prefix(lane)}_BRIDGE_URL", raising=False)
    yield
    reset_config_cache()


def _two_lanes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _render(
        tmp_path,
        monkeypatch,
        {
            "control_system": _control_system(),
            "services": {
                "bluesky": {"port": LIVE_PORT, "target": "live"},
                "bluesky_va": {"port": VA_PORT, "target": "va"},
            },
        },
    )


def test_each_target_gets_the_bridge_its_lane_serves(tmp_path, monkeypatch):
    _two_lanes(tmp_path, monkeypatch)
    assert dict(bridges_for_target("live")) == {"bluesky": f"http://127.0.0.1:{LIVE_PORT}"}
    assert dict(bridges_for_target("va")) == {"bluesky_va": f"http://127.0.0.1:{VA_PORT}"}


def test_a_target_no_lane_serves_has_no_bridge(tmp_path, monkeypatch):
    _two_lanes(tmp_path, monkeypatch)
    assert dict(bridges_for_target("standin")) == {}


def test_a_lane_url_from_its_environment_wins(tmp_path, monkeypatch):
    _two_lanes(tmp_path, monkeypatch)
    monkeypatch.setenv(f"{lane_env_prefix('bluesky_va')}_BRIDGE_URL", "http://bridge-va:8000/")
    assert dict(bridges_for_target("va")) == {"bluesky_va": "http://bridge-va:8000"}


def test_a_single_lane_serves_the_deployment_baseline(tmp_path, monkeypatch):
    _render(
        tmp_path,
        monkeypatch,
        {"control_system": _control_system(), "services": {"bluesky": {"port": LIVE_PORT}}},
    )
    assert dict(bridges_for_target("live")) == {"bluesky": f"http://127.0.0.1:{LIVE_PORT}"}
    assert dict(bridges_for_target("va")) == {}


def test_a_deployment_without_a_bridge_has_none(tmp_path, monkeypatch):
    _render(tmp_path, monkeypatch, {"control_system": _control_system()})
    assert dict(bridges_for_target("live")) == {}
