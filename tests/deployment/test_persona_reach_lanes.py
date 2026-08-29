"""A persona of a two-lane deployment is told both plan lanes.

``bluesky.second_lane`` renders a second bridge, and the lane resolver picks
the lane a session is on by matching each lane's declared ``target`` against
the session target. These tests drive the REAL injector's output through the
Reach Contract's projection and then ask the REAL lane resolver what it makes
of the persona's config — so a projection that copied the port but not the
target, or lane 1 but not lane 2, fails here rather than in a container.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from osprey.bluesky_bridge_connection import (
    LANE_KEYS,
    discover_lane_keys,
    lane_declared_target,
    lane_env_prefix,
)
from osprey.cli.build_injectors import _inject_bluesky
from osprey.cli.build_profile_schema import BlueskyConfig, VAConfig
from osprey.deployment.reach import project_attached_overrides, reach_dials, reach_errors
from tests.cli.test_bluesky_lane_config import STANDIN_PORT, _read_config, _write_config

#: Each deployment baseline, as ``control_system.type``, with the lane pair the
#: build derives from it: lane 1 serves the baseline, lane 2 the target it is
#: paired with. Every one of the three control targets appears as somebody's
#: lane, which is the point — a persona attached to any of these deployments
#: has to be told both of its lanes by name.
BASELINES = [
    ("virtual_accelerator", ("bluesky", "va"), ("bluesky_live", "live")),
    ("epics", ("bluesky", "live"), ("bluesky_va", "va")),
    ("live_standin", ("bluesky", "standin"), ("bluesky_va", "va")),
]


def _host_render(
    tmp_path: Path, *, second_lane: bool, cs_type: str = "virtual_accelerator"
) -> dict[str, Any]:
    """The deploying project's config after the bluesky injector ran on it."""
    project = tmp_path / "host"
    _write_config(project, cs_type=cs_type)
    _inject_bluesky(
        BlueskyConfig(second_lane=second_lane), project, VAConfig(live_standin=STANDIN_PORT)
    )
    return _read_config(project)


def _persona(host: dict[str, Any], cs_type: str = "virtual_accelerator") -> dict[str, Any]:
    """An attached render that runs the bluesky server, told the host's facts."""
    config: dict[str, Any] = {
        "claude_code": {"servers": {"bluesky": {"enabled": True}}},
        "control_system": {"type": cs_type, "writes_enabled": True},
    }
    for dotted, value in project_attached_overrides(host, config).items():
        node = config
        *parents, leaf = dotted.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    return config


@pytest.fixture(autouse=True)
def _no_bridge_overrides(monkeypatch):
    for lane in LANE_KEYS:
        monkeypatch.delenv(f"{lane_env_prefix(lane)}_BRIDGE_URL", raising=False)


def test_the_persona_renders_both_lanes_with_their_targets(tmp_path):
    host = _host_render(tmp_path, second_lane=True)
    persona = _persona(host)

    assert discover_lane_keys(persona) == ("bluesky", "bluesky_live")
    assert lane_declared_target("bluesky", persona) == "va"
    assert lane_declared_target("bluesky_live", persona) == "live"
    assert persona["services"]["bluesky_live"]["port"] == host["services"]["bluesky_live"]["port"]


@pytest.mark.parametrize(("cs_type", "lane_one", "lane_two"), BASELINES)
def test_every_baseline_projects_its_own_lane_pair(tmp_path, cs_type, lane_one, lane_two):
    """The lane pair follows the BASELINE, and the persona is told whichever it is.

    The stand-in row is the one that would not have existed before: a project
    baselined on its own soft IOC renders a ``standin`` lane, not a ``live``
    one, and a persona attached to it must be handed that lane under that name
    — a projection that only knew ``live``/``va`` would leave such a session
    with a lane whose target it cannot match, which the resolver reads as a
    single-lane deployment.
    """
    lane_one_key, lane_one_target = lane_one
    lane_two_key, lane_two_target = lane_two
    host = _host_render(tmp_path, second_lane=True, cs_type=cs_type)
    persona = _persona(host, cs_type=cs_type)

    assert set(discover_lane_keys(persona)) == {lane_one_key, lane_two_key}
    assert lane_declared_target(lane_one_key, persona) == lane_one_target
    assert lane_declared_target(lane_two_key, persona) == lane_two_target

    # Both are dialable, and neither carries the host-side bridge settings.
    dials = {contract.service: dial for contract, _, dial in reach_dials(persona)}
    for key in (lane_one_key, lane_two_key):
        assert dials[key] == ("127.0.0.1", host["services"][key]["port"])
        assert set(persona["services"][key]) == {"port", "target"}
    assert reach_errors(persona) == []


def test_the_standin_lane_has_a_reach_contract_of_its_own(tmp_path):
    """``bluesky_standin`` is refused by name, like every other second lane.

    The stand-in lane is lane 1 on the deployment that derives it, so the only
    way to reach it as a SECOND lane is to declare it — which a profile may.
    What matters is that the Reach Contract knows the key at all: an unknown
    lane block is not projected and not checked, so a persona would be handed
    a lane it cannot dial and told nothing about it.
    """
    persona = _persona(_host_render(tmp_path, second_lane=True))
    persona["services"]["bluesky_standin"] = {"target": "standin"}

    (error,) = reach_errors(persona)
    assert "bluesky_standin lane" in error
    assert "services.bluesky_standin.port" in error


def test_the_persona_carries_nothing_of_the_lane_but_its_reach(tmp_path):
    """The bridge's own settings — its plan directory, its CA name servers —
    describe the service the HOST runs; a client needs none of them."""
    persona = _persona(_host_render(tmp_path, second_lane=True))
    assert set(persona["services"]["bluesky_live"]) == {"port", "target"}
    assert set(persona["services"]["bluesky"]) == {"port", "target"}


def test_both_lanes_are_live_consumers_the_persona_can_dial(tmp_path):
    host = _host_render(tmp_path, second_lane=True)
    dials = {contract.service: dial for contract, _, dial in reach_dials(_persona(host))}
    assert dials["bluesky"] == ("127.0.0.1", host["services"]["bluesky"]["port"])
    assert dials["bluesky_live"] == ("127.0.0.1", host["services"]["bluesky_live"]["port"])
    assert reach_errors(_persona(host)) == []


def test_a_lane_told_without_its_port_is_refused_by_name(tmp_path):
    persona = _persona(_host_render(tmp_path, second_lane=True))
    del persona["services"]["bluesky_live"]["port"]
    (error,) = reach_errors(persona)
    assert "bluesky_live lane" in error
    assert "services.bluesky_live.port" in error


def test_a_single_lane_host_leaves_the_persona_single_lane(tmp_path):
    """No second lane on the host means no second lane here — and no
    ``target`` on lane 1 either, which is the pre-lane render exactly."""
    persona = _persona(_host_render(tmp_path, second_lane=False))
    assert discover_lane_keys(persona) == ("bluesky",)
    assert set(persona["services"]) == {"bluesky"}
    assert "target" not in persona["services"]["bluesky"]
    lanes = [c.service for c, _, _ in reach_dials(persona) if c.service.startswith("bluesky")]
    assert lanes == ["bluesky"]
