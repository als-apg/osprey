"""The bluesky plan-lane axis: one lane per control-system target, opt-in.

``bluesky.second_lane`` turns the single bluesky stack every project has today
into one stack per switchable target, so a session switched away from the
deployment baseline still has a lane to queue plans on. The properties pinned
here are the ones a rendered deployment depends on and a reader cannot check by
eye:

* default OFF renders exactly what it rendered before the field existed
  (the regression pin — every existing project is this case);
* a lane is named for the TARGET it serves, not for its index, in either
  baseline direction;
* the two lanes never share a port, and tiled — the one shared component —
  stays on lane 1;
* the LIVE lane's gateway address is a required compose variable, verbatim, and
  the VA lane has no such requirement because its gateway is co-deployed.

Harness follows ``test_build_injectors_comment_anchoring.py``: a literal
config.yml written into ``tmp_path``, the injector called directly, then
assertions on the parsed result.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml as pyyaml

from osprey.cli.build_injectors import (
    _LIVE_LANE_CA_NAME_SERVERS,
    _inject_bluesky,
)
from osprey.cli.build_profile import _parse_profile
from osprey.cli.build_profile_schema import (
    SECOND_LANE_PORT_STRIDE,
    BlueskyConfig,
    VAConfig,
)
from osprey.errors import BuildProfileError

CONFIG_TEMPLATE = """\
control_system:
  type: "{cs_type}"

services:
  postgresql:
    path: ./services/postgresql

# Services to deploy with `osprey up`
deployed_services:
  - postgresql

# ============================================================
# SAFETY CONTROLS
# ============================================================

# Approval workflow for sensitive operations
approval:
  enabled: true
"""


def _write_config(project_path: Path, cs_type: str = "epics") -> None:
    project_path.mkdir(parents=True, exist_ok=True)
    (project_path / "config.yml").write_text(
        CONFIG_TEMPLATE.format(cs_type=cs_type), encoding="utf-8"
    )


def _read_config(project_path: Path) -> dict:
    return pyyaml.safe_load((project_path / "config.yml").read_text(encoding="utf-8"))


def _line_no(text: str, needle: str) -> int:
    for i, line in enumerate(text.splitlines()):
        if needle in line:
            return i
    raise AssertionError(f"{needle!r} not found in:\n{text}")


# ---------------------------------------------------------------------------
# Default: single lane, unchanged
# ---------------------------------------------------------------------------


def test_schema_default_is_single_lane() -> None:
    """The lane axis is opt-in — nothing about an existing profile changes."""
    assert BlueskyConfig().second_lane is False


@pytest.mark.parametrize("cs_type", ["epics", "virtual_accelerator", "mock"])
def test_single_lane_block_is_unchanged(tmp_path: Path, cs_type: str) -> None:
    """Default config renders exactly today's block, on any baseline.

    The regression pin for every project built before the lane axis existed:
    the keys, their values, and the absence of every lane key. A ``mock``
    baseline is included because a single-lane deploy needs no switchable
    target at all — only the second lane does.
    """
    project = tmp_path / "project"
    _write_config(project, cs_type=cs_type)

    _inject_bluesky(BlueskyConfig(), project)

    config = _read_config(project)
    assert config["services"]["bluesky"] == {
        "path": "./services/bluesky",
        "port": 8090,
        "tiled_enabled": False,
        "tiled_port": 8091,
        "devices_file": "data/bluesky_devices.yml",
    }
    assert config["deployed_services"] == ["postgresql", "bluesky"]
    assert [key for key in config["services"] if key.startswith("bluesky_")] == []


def test_single_lane_render_is_byte_identical_to_pre_lane_shape(tmp_path: Path) -> None:
    """Two injections that differ only in an untouched knob render the same text.

    Byte equality, not parsed equality: the lane axis writes its keys inside the
    same ``anchored_put`` the block already used, and a stray blank line or
    re-anchored comment would be a rendered-config regression the parsed form
    cannot see.
    """
    first = tmp_path / "a"
    second = tmp_path / "b"
    _write_config(first)
    _write_config(second)

    _inject_bluesky(BlueskyConfig(plan_dir="/facility/plans"), first)
    _inject_bluesky(BlueskyConfig(plan_dir="/facility/plans", second_lane=False), second)

    assert (first / "config.yml").read_text(encoding="utf-8") == (second / "config.yml").read_text(
        encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Two lanes
# ---------------------------------------------------------------------------


def test_va_baseline_renders_a_live_second_lane(tmp_path: Path) -> None:
    """VA baseline: lane 1 = va (keys unchanged), lane 2 = live, named for it."""
    project = tmp_path / "project"
    _write_config(project, cs_type="virtual_accelerator")

    _inject_bluesky(BlueskyConfig(second_lane=True), project)

    config = _read_config(project)
    lane1 = config["services"]["bluesky"]
    lane2 = config["services"]["bluesky_live"]

    assert lane1["target"] == "va"
    assert lane1["port"] == 8090
    assert lane2["target"] == "live"
    assert lane2["path"] == "./services/bluesky"
    assert lane2["port"] == 8090 + SECOND_LANE_PORT_STRIDE

    # The live lane refuses to come up on an unset gateway; the VA lane's
    # gateway is co-deployed, so it carries no such requirement.
    assert lane2["ca_name_servers"] == _LIVE_LANE_CA_NAME_SERVERS
    assert lane2["ca_name_servers"].startswith("${EPICS_CA_NAME_SERVERS:?")
    assert "ca_name_servers" not in lane1

    # tiled is the one shared component: lane 1 only.
    assert lane1["tiled_enabled"] is False
    assert lane1["tiled_port"] == 8091
    assert "tiled_enabled" not in lane2
    assert "tiled_port" not in lane2

    assert config["deployed_services"] == ["postgresql", "bluesky", "bluesky_live"]


def test_live_baseline_renders_a_va_second_lane(tmp_path: Path) -> None:
    """Live baseline: the mirror image — lane 1 = live, lane 2 = va.

    The requirement follows the TARGET, not the lane index: here it is lane 1
    that talks to the live machine, so lane 1 is the block that carries it.
    """
    project = tmp_path / "project"
    _write_config(project, cs_type="epics")

    _inject_bluesky(BlueskyConfig(second_lane=True), project, VAConfig())

    config = _read_config(project)
    lane1 = config["services"]["bluesky"]
    lane2 = config["services"]["bluesky_va"]

    assert lane1["target"] == "live"
    assert lane2["target"] == "va"
    assert "bluesky_live" not in config["services"]

    assert lane1["ca_name_servers"] == _LIVE_LANE_CA_NAME_SERVERS
    assert "ca_name_servers" not in lane2

    assert lane1["port"] != lane2["port"]
    assert lane2["port"] == 8090 + SECOND_LANE_PORT_STRIDE
    assert "tiled_port" not in lane2

    assert config["deployed_services"] == ["postgresql", "bluesky", "bluesky_va"]


def test_second_lane_carries_facility_plan_keys(tmp_path: Path) -> None:
    """Plans and devices belong to the facility, not to a target — both lanes
    carry them, including the always-written ``devices_file``."""
    project = tmp_path / "project"
    _write_config(project, cs_type="epics")

    _inject_bluesky(
        BlueskyConfig(
            second_lane=True,
            plan_dir="/facility/plans",
            excluded_plans=["scan_a", "scan_b"],
            devices_file="/facility/devices.yml",
        ),
        project,
        VAConfig(),
    )

    config = _read_config(project)
    for lane_key in ("bluesky", "bluesky_va"):
        lane = config["services"][lane_key]
        assert lane["plan_dir"] == "/facility/plans"
        assert lane["excluded_plans"] == os.pathsep.join(["scan_a", "scan_b"])
        assert lane["devices_file"] == "/facility/devices.yml"


def test_second_lane_carries_the_default_devices_file(tmp_path: Path) -> None:
    """``devices_file`` is always-written, so an unconfigured two-lane deploy
    still lands the default path on BOTH lanes — the staging step never has to
    re-derive it for a lane that said nothing."""
    project = tmp_path / "project"
    _write_config(project, cs_type="epics")

    _inject_bluesky(BlueskyConfig(second_lane=True), project, VAConfig())

    config = _read_config(project)
    for lane_key in ("bluesky", "bluesky_va"):
        assert config["services"][lane_key]["devices_file"] == "data/bluesky_devices.yml"


def test_second_lane_keeps_section_banner_and_list_intact(tmp_path: Path) -> None:
    """Both lanes land inside their sections, ahead of the SAFETY banner."""
    project = tmp_path / "project"
    _write_config(project, cs_type="epics")

    _inject_bluesky(BlueskyConfig(second_lane=True), project, VAConfig())

    text = (project / "config.yml").read_text(encoding="utf-8")
    assert _line_no(text, "- bluesky_va") < _line_no(text, "# SAFETY CONTROLS")
    assert _line_no(text, "  bluesky_va:") < _line_no(text, "# Services to deploy")
    assert _line_no(text, "# SAFETY CONTROLS") < _line_no(text, "approval:")


def test_second_lane_rerun_is_idempotent(tmp_path: Path) -> None:
    """A second build re-renders both lanes without duplicating either."""
    project = tmp_path / "project"
    _write_config(project, cs_type="epics")

    _inject_bluesky(BlueskyConfig(second_lane=True), project, VAConfig())
    _inject_bluesky(BlueskyConfig(second_lane=True), project, VAConfig())

    deployed = _read_config(project)["deployed_services"]
    assert deployed.count("bluesky") == 1
    assert deployed.count("bluesky_va") == 1


def test_turning_the_second_lane_off_again_leaves_no_lane_keys(tmp_path: Path) -> None:
    """Lane 1 is regenerated whole, so its lane keys go when the axis does.

    The stale-key case: a deploy that tried two lanes and went back to one must
    not keep a ``target``/``ca_name_servers`` pair that no longer describes it.
    (The lane-2 BLOCK is a separate service key and is left where it is — the
    author drops it from ``deployed_services``, as with any other service.)
    """
    project = tmp_path / "project"
    _write_config(project, cs_type="epics")

    _inject_bluesky(BlueskyConfig(second_lane=True), project, VAConfig())
    _inject_bluesky(BlueskyConfig(second_lane=False), project)

    lane1 = _read_config(project)["services"]["bluesky"]
    assert "target" not in lane1
    assert "ca_name_servers" not in lane1


def test_authored_env_is_carried_on_both_lanes(tmp_path: Path) -> None:
    """``env:`` belongs to the author — the whole-block rewrite keeps it, per lane."""
    project = tmp_path / "project"
    _write_config(project, cs_type="epics")

    # Both lanes pre-declare an env passthrough, as `_inject_profile_services`
    # or a dotted `config:` override would have left them.
    text = (project / "config.yml").read_text(encoding="utf-8")
    text = text.replace(
        "services:\n",
        "services:\n"
        "  bluesky:\n"
        "    env:\n"
        "      - HTTPS_PROXY\n"
        "  bluesky_va:\n"
        "    env:\n"
        "      - NO_PROXY\n",
        1,
    )
    (project / "config.yml").write_text(text, encoding="utf-8")

    _inject_bluesky(BlueskyConfig(second_lane=True), project, VAConfig())

    services = _read_config(project)["services"]
    assert services["bluesky"]["env"] == ["HTTPS_PROXY"]
    assert services["bluesky_va"]["env"] == ["NO_PROXY"]


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cs_type", ["mock", "doocs"])
def test_second_lane_refuses_an_unswitchable_baseline(tmp_path: Path, cs_type: str) -> None:
    """A ``mock``/``doocs`` deployment has no second target to serve."""
    project = tmp_path / "project"
    _write_config(project, cs_type=cs_type)

    with pytest.raises(BuildProfileError, match="switchable deployment baseline"):
        _inject_bluesky(BlueskyConfig(second_lane=True), project, VAConfig())


def test_live_baseline_second_lane_refuses_without_a_va_service(tmp_path: Path) -> None:
    """A VA lane with no virtual accelerator to address is refused at build time.

    The lane would render, deploy, and connect to nothing — a plan queued on it
    would sit there looking merely slow. The build can see this one coming (the
    VA soft-IOC is the deployment's own service), so it says so instead.
    """
    project = tmp_path / "project"
    _write_config(project, cs_type="epics")

    with pytest.raises(BuildProfileError, match="deploys none"):
        _inject_bluesky(BlueskyConfig(second_lane=True), project, None)

    # Refused before anything was written: no half-rendered pair left behind.
    assert "bluesky_va" not in (_read_config(project)["services"] or {})


def test_live_baseline_second_lane_is_allowed_with_a_va_service(tmp_path: Path) -> None:
    """The same profile with a ``virtual_accelerator:`` block renders both lanes."""
    project = tmp_path / "project"
    _write_config(project, cs_type="epics")

    _inject_bluesky(BlueskyConfig(second_lane=True), project, VAConfig(port=5064))

    services = _read_config(project)["services"]
    assert services["bluesky"]["target"] == "live"
    assert services["bluesky_va"]["target"] == "va"


def test_va_baseline_second_lane_needs_no_va_block(tmp_path: Path) -> None:
    """No mirror check on a VA baseline — the live lane fails loudly at up-time.

    Its second lane is the LIVE lane, whose gateway is a facility address the
    build cannot verify; that is what ``${EPICS_CA_NAME_SERVERS:?}`` is for.
    Refusing here would only refuse the deployments that are correct.
    """
    project = tmp_path / "project"
    _write_config(project, cs_type="virtual_accelerator")

    _inject_bluesky(BlueskyConfig(second_lane=True), project, None)

    services = _read_config(project)["services"]
    assert services["bluesky_live"]["ca_name_servers"] == _LIVE_LANE_CA_NAME_SERVERS


def test_derived_lane_port_refuses_to_collide_with_tiled() -> None:
    """The derivation is re-checked against the ports the author may have moved."""
    config = BlueskyConfig(second_lane=True, tiled_enabled=True, tiled_port=8190)
    with pytest.raises(ValueError, match="tiled_port"):
        config.second_lane_port()


def test_derived_lane_port_refuses_to_leave_the_port_range() -> None:
    config = BlueskyConfig(second_lane=True, port=65500)
    with pytest.raises(ValueError, match="1\\.\\.65535"):
        config.second_lane_port()


def test_derived_lane_port_ignores_a_disabled_tiled_port() -> None:
    """A tiled port nothing publishes cannot collide with anything."""
    config = BlueskyConfig(second_lane=True, tiled_enabled=False, tiled_port=8190)
    assert config.second_lane_port() == 8190


# ---------------------------------------------------------------------------
# Profile round-trip
# ---------------------------------------------------------------------------


def test_profile_round_trip() -> None:
    """``bluesky.second_lane`` survives the profile parser, and defaults off."""
    profile = _parse_profile(pyyaml.safe_load("name: lanes\nbluesky:\n  second_lane: true\n"))
    assert profile.bluesky is not None
    assert profile.bluesky.second_lane is True

    profile = _parse_profile(pyyaml.safe_load("name: lanes\nbluesky:\n  port: 8090\n"))
    assert profile.bluesky is not None
    assert profile.bluesky.second_lane is False
