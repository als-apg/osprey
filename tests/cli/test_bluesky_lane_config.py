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
  the VA lane has no such requirement because its gateway is co-deployed;
* a deployment whose ``live`` target is a co-deployed stand-in
  (``virtual_accelerator.live_standin``) dials that container instead, because
  there is then nothing for an operator to supply and nothing to refuse over.

Harness follows ``test_build_injectors_comment_anchoring.py``: a literal
config.yml written into ``tmp_path``, the injector called directly, then
assertions on the parsed result.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest
import yaml as pyyaml

from osprey.cli.build_injectors import (
    _LIVE_LANE_CA_NAME_SERVERS,
    _LIVE_STANDIN_COMPOSE_SERVICE,
    _inject_bluesky,
    _live_lane_ca_name_servers,
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


def _declare_lane_target(project_path: Path, lane_key: str, target: str) -> None:
    """Put a hand-written ``target`` on a lane block.

    The shape a profile's ``config:`` overlay leaves behind: it is merged into
    config.yml before the injectors run, so this is what ``_inject_bluesky``
    finds when it loads the file.
    """
    config = _read_config(project_path)
    block = config["services"].get(lane_key) or {}
    config["services"][lane_key] = {**block, "target": target}
    (project_path / "config.yml").write_text(pyyaml.safe_dump(config), encoding="utf-8")


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
# The live lane on a deployment whose `live` target is a stand-in
# ---------------------------------------------------------------------------

#: The stand-in port the preset ships, reused here so the rendered dial in
#: these assertions is the one an operator actually gets.
STANDIN_PORT = 5074
STANDIN_DIAL = f"live-standin:{STANDIN_PORT}"


def test_the_stand_in_dial_names_the_compose_service_not_the_config_key() -> None:
    """The lane dials a CONTAINER, so the name is the hyphenated compose key.

    Spelled out here rather than derived, because the whole value of the
    derivation is that it agrees with the VA compose template's
    ``instance_key | replace('_', '-')`` — a test that recomputed it the same
    way would agree with a typo just as happily.
    """
    assert _LIVE_STANDIN_COMPOSE_SERVICE == "live-standin"
    assert _live_lane_ca_name_servers(VAConfig(live_standin=STANDIN_PORT)) == STANDIN_DIAL


@pytest.mark.parametrize("virtual_accelerator", [None, VAConfig(), VAConfig(port=5065)])
def test_without_a_stand_in_the_derivation_is_the_required_variable(
    virtual_accelerator: VAConfig | None,
) -> None:
    """No stand-in, no co-deployed gateway: the refusing form, byte for byte."""
    assert _live_lane_ca_name_servers(virtual_accelerator) == _LIVE_LANE_CA_NAME_SERVERS


def test_a_va_baseline_live_lane_dials_a_deployed_stand_in(tmp_path: Path) -> None:
    """The shipped shape: VA baseline, live second lane, stand-in as `live`.

    Nothing for the operator to supply means nothing for `osprey up` to refuse
    over — which is what makes `osprey build && osprey up` a no-edit story on a
    profile that sets both the stand-in and `bluesky.second_lane`.
    """
    project = tmp_path / "project"
    _write_config(project, cs_type="virtual_accelerator")

    _inject_bluesky(BlueskyConfig(second_lane=True), project, VAConfig(live_standin=STANDIN_PORT))

    services = _read_config(project)["services"]
    assert services["bluesky_live"]["target"] == "live"
    assert services["bluesky_live"]["ca_name_servers"] == STANDIN_DIAL

    # The VA lane is untouched: its gateway was always co-deployed, and the
    # stand-in is a second machine rather than a change to that one.
    assert services["bluesky"]["target"] == "va"
    assert "ca_name_servers" not in services["bluesky"]


def test_a_live_baseline_lane_one_dials_the_stand_in_too(tmp_path: Path) -> None:
    """The requirement follows the TARGET, and so does the stand-in that fills it.

    Here the stand-in IS the baseline — `control_system.type` is `epics`, so
    lane 1 serves `live` — and lane 2 is the VA lane. The dial has to land on
    lane 1, because a lane's addressing is decided by what it talks to and not
    by its index.
    """
    project = tmp_path / "project"
    _write_config(project, cs_type="epics")

    _inject_bluesky(BlueskyConfig(second_lane=True), project, VAConfig(live_standin=STANDIN_PORT))

    services = _read_config(project)["services"]
    assert services["bluesky"]["target"] == "live"
    assert services["bluesky"]["ca_name_servers"] == STANDIN_DIAL
    assert services["bluesky_va"]["target"] == "va"
    assert "ca_name_servers" not in services["bluesky_va"]


@pytest.mark.parametrize(
    ("cs_type", "live_lane_key"),
    [("virtual_accelerator", "bluesky_live"), ("epics", "bluesky")],
)
def test_without_a_stand_in_the_live_lane_block_is_unchanged(
    tmp_path: Path, cs_type: str, live_lane_key: str
) -> None:
    """The regression pin: no `live_standin`, and the idiom is byte-identical.

    Both baseline directions, because the stand-in fork is applied to whichever
    lane serves `live` and a fork that leaked into the other direction would
    show up here first.
    """
    project = tmp_path / "project"
    _write_config(project, cs_type=cs_type)

    _inject_bluesky(BlueskyConfig(second_lane=True), project, VAConfig())

    services = _read_config(project)["services"]
    addressing = services[live_lane_key]["ca_name_servers"]
    assert addressing == _LIVE_LANE_CA_NAME_SERVERS
    assert addressing.startswith("${EPICS_CA_NAME_SERVERS:?")
    assert "live-standin" not in addressing


def test_the_stand_in_moves_the_gateway_and_nothing_else(tmp_path: Path) -> None:
    """Two renders that differ only in `live_standin` differ only at that key.

    Byte-level, because the claim is about blast radius: the stand-in changes
    which machine the live lane dials, and a port, a lane name, a
    deployed_services entry or a re-anchored comment moving with it would be a
    rendered-config regression the parsed form cannot see. `ca_name_servers` is
    the last key of its block in both renders, so everything before it and
    everything after its value must be identical text.
    """
    without = tmp_path / "without"
    with_standin = tmp_path / "with"
    for project in (without, with_standin):
        _write_config(project, cs_type="virtual_accelerator")

    _inject_bluesky(BlueskyConfig(second_lane=True), without, VAConfig())
    _inject_bluesky(
        BlueskyConfig(second_lane=True), with_standin, VAConfig(live_standin=STANDIN_PORT)
    )

    def split(project: Path) -> tuple[str, str, str]:
        head, key, rest = (
            (project / "config.yml").read_text(encoding="utf-8").partition("ca_name_servers:")
        )
        assert key, "the live lane wrote no gateway address"
        # The value runs to the blank line that ends the services block; the
        # required-variable form is long enough that the emitter folds it over
        # several lines, so the value is compared as one string, not by lines.
        value, blank, tail = rest.partition("\n\n")
        return head, value.strip(), blank + tail

    before_head, before_value, before_tail = split(without)
    after_head, after_value, after_tail = split(with_standin)

    assert before_head == after_head
    assert before_tail == after_tail
    assert before_value.startswith("${EPICS_CA_NAME_SERVERS:?")
    assert after_value == STANDIN_DIAL


def test_a_stand_in_dial_survives_a_rebuild_unquoted(tmp_path: Path) -> None:
    """`live-standin:5074` is a plain YAML scalar, and stays one on re-injection.

    The dial carries a colon, which is the character that decides whether the
    emitter wrote a scalar or something the next build reads back as a mapping.
    A second injection reads the file it wrote, so this is the round trip that
    would catch it.
    """
    project = tmp_path / "project"
    _write_config(project, cs_type="virtual_accelerator")
    va = VAConfig(live_standin=STANDIN_PORT)

    _inject_bluesky(BlueskyConfig(second_lane=True), project, va)
    first = (project / "config.yml").read_text(encoding="utf-8")
    _inject_bluesky(BlueskyConfig(second_lane=True), project, va)

    assert (project / "config.yml").read_text(encoding="utf-8") == first
    assert _read_config(project)["services"]["bluesky_live"]["ca_name_servers"] == STANDIN_DIAL


def test_a_single_lane_deploy_carries_no_gateway_key_even_with_a_stand_in(
    tmp_path: Path,
) -> None:
    """`ca_name_servers` stays LANE-SCOPED. The stand-in does not widen it.

    A one-lane deployment serves the baseline and nothing else, so it has no
    live lane to address — writing the dial anyway would hand the single lane
    an addressing key it never had, on every stand-in project.
    """
    project = tmp_path / "project"
    _write_config(project, cs_type="virtual_accelerator")

    _inject_bluesky(BlueskyConfig(), project, VAConfig(live_standin=STANDIN_PORT))

    services = _read_config(project)["services"]
    assert "ca_name_servers" not in services["bluesky"]
    assert "target" not in services["bluesky"]


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


# ---------------------------------------------------------------------------
# The declared lane target
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lane_key", ["bluesky", "bluesky_va", "bluesky_live"])
def test_a_lane_target_that_names_no_control_target_is_refused(
    tmp_path: Path, lane_key: str
) -> None:
    """The one lane-target mistake no runtime signal can repair.

    A target that does not RESOLVE is a deployment that has not described its
    machine yet, and the bridge falls back to the baseline for it. A target
    that is not spelled ``live`` or ``va`` is a typo, and it would fall back
    forever while the author went on believing the lane served what they wrote.
    Every lane key is swept, not just the ones this build renders: a block left
    behind by a profile that once set ``second_lane`` keeps its target, and the
    bridge reads it whether or not this build wrote it.
    """
    project = tmp_path / "project"
    _write_config(project, cs_type="epics")
    _declare_lane_target(project, lane_key, "prod")

    with pytest.raises(BuildProfileError) as excinfo:
        _inject_bluesky(BlueskyConfig(), project, VAConfig())

    message = str(excinfo.value)
    assert f"services.{lane_key}.target" in message
    assert "'prod'" in message
    assert "'live'" in message and "'va'" in message


@pytest.mark.parametrize("target", ["live", "va"])
def test_a_lane_target_the_build_derives_is_accepted(tmp_path: Path, target: str) -> None:
    """Both spellings are targets, so neither is the typo the refusal is for."""
    project = tmp_path / "project"
    _write_config(project, cs_type="epics")
    _declare_lane_target(project, "bluesky", target)

    _inject_bluesky(BlueskyConfig(), project, VAConfig())

    # Derived, not carried: the injector owns this key on the lanes it renders,
    # and a single-lane block has never had one.
    assert "target" not in _read_config(project)["services"]["bluesky"]


def test_a_lane_whose_target_resolves_to_nothing_is_named_at_build_time(
    tmp_path: Path, caplog
) -> None:
    """A VA baseline's live lane has no connector block, and should be told so.

    Not a refusal — that lane is the shipped, correct case, and its gateway
    arrives at ``osprey up`` as EPICS_CA_NAME_SERVERS. What it does not have is
    a block of its own, so it inherits the deployment-wide write posture, and
    the build is where an author can still do something about that.
    """
    project = tmp_path / "project"
    _write_config(project, cs_type="virtual_accelerator")

    with caplog.at_level(logging.WARNING):
        _inject_bluesky(BlueskyConfig(second_lane=True), project, None)

    assert "bluesky_live" in caplog.text
    assert "control_system.connector.epics" in caplog.text


def test_a_live_baseline_pair_names_no_lane_at_build_time(tmp_path: Path, caplog) -> None:
    """Both of its targets resolve, so neither lane is short of anything."""
    project = tmp_path / "project"
    _write_config(project, cs_type="epics")

    with caplog.at_level(logging.WARNING):
        _inject_bluesky(BlueskyConfig(second_lane=True), project, VAConfig(port=5064))

    assert "control_system.connector" not in caplog.text
