"""Tests for the ``_inject_bluesky`` build step in ``osprey.cli.build_cmd``.

Covers the responsibilities of the bluesky-injection step: copying the bundled
bluesky compose template, writing the ``services.bluesky`` config +
registering it in ``deployed_services`` (additively) so
``find_service_config`` resolves it, and confirming the feature stays
opt-in — a profile with no ``bluesky:`` key injects nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ruamel.yaml import YAML

from osprey.cli.build_cmd import _inject_bluesky
from osprey.cli.build_profile import BlueskyConfig, _parse_profile
from osprey.deployment.compose_generator import find_service_config
from osprey.errors import BuildProfileError
from osprey.port_layout import DEFAULT_PORT_BASE, layout_ports


def _write_config(project_path: Path) -> None:
    """Write a minimal config.yml with a pre-existing deployed service."""
    yaml = YAML()
    config = {
        "deployed_services": ["postgresql"],
        "services": {"postgresql": {}},
    }
    with open(project_path / "config.yml", "w") as fh:
        yaml.dump(config, fh)


def _read_config(project_path: Path) -> dict:
    """Reload config.yml as a plain dict."""
    yaml = YAML()
    with open(project_path / "config.yml") as fh:
        return yaml.load(fh)


def test_inject_bluesky_default_config(tmp_path: Path) -> None:
    """Default BlueskyConfig copies the template and registers config additively."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_bluesky(BlueskyConfig(), project_path)

    # Compose template + Dockerfile copied.
    assert (project_path / "services" / "bluesky" / "docker-compose.yml.j2").is_file()
    assert (project_path / "services" / "bluesky" / "Dockerfile").is_file()

    config = _read_config(project_path)
    svc = config["services"]["bluesky"]
    assert svc["path"] == "./services/bluesky"
    assert svc["port"] == 10080
    assert svc["tiled_enabled"] is False
    assert svc["tiled_port"] == 10070
    assert svc["devices_file"] == "data/bluesky_devices.yml"
    # No pinned image: the service builds the project's local image (compose
    # template defaults to <project>-bluesky-bridge:local + a build: section).
    assert "image" not in svc

    # deployed_services is additive — keeps postgresql, adds bluesky.
    deployed = [str(s) for s in config["deployed_services"]]
    assert "postgresql" in deployed
    assert "bluesky" in deployed


def test_inject_bluesky_custom_ports_and_tiled(tmp_path: Path) -> None:
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_bluesky(BlueskyConfig(port=9500, tiled_enabled=True, tiled_port=9501), project_path)

    config = _read_config(project_path)
    svc = config["services"]["bluesky"]
    assert svc["port"] == 9500
    assert svc["tiled_enabled"] is True
    assert svc["tiled_port"] == 9501


def test_inject_bluesky_idempotent_rerun(tmp_path: Path) -> None:
    """Re-running the injection (e.g. a second build) doesn't duplicate deployed_services."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_bluesky(BlueskyConfig(), project_path)
    _inject_bluesky(BlueskyConfig(port=9999), project_path)

    config = _read_config(project_path)
    deployed = [str(s) for s in config["deployed_services"]]
    assert deployed.count("bluesky") == 1
    # Second call's config wins (last-write, matching _inject_dispatch's contract).
    assert config["services"]["bluesky"]["port"] == 9999


def test_inject_bluesky_missing_config_yml_is_a_noop(tmp_path: Path) -> None:
    """No config.yml (unusual, but shouldn't crash the build) — just warns and returns."""
    project_path = tmp_path / "project"
    project_path.mkdir()

    _inject_bluesky(BlueskyConfig(), project_path)  # must not raise

    # Template is still copied before the config.yml check.
    assert (project_path / "services" / "bluesky" / "docker-compose.yml.j2").is_file()
    assert not (project_path / "config.yml").exists()


def test_find_service_config_resolves_bluesky(tmp_path: Path) -> None:
    """After injection, find_service_config('bluesky') resolves path + template."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_bluesky(BlueskyConfig(), project_path)

    config = _read_config(project_path)
    service_config, template_path = find_service_config(config, "bluesky")
    assert service_config is not None
    assert service_config["path"] == "./services/bluesky"
    assert template_path == "./services/bluesky/docker-compose.yml.j2"


def _image_defaults(project_name: str) -> dict[str, str]:
    """The image map ``_inject_project_metadata`` injects, for hand-built ctx.

    The bridge's image line renders its innermost fallback from this mapping,
    so a context assembled by hand still has to carry it. Taken from the
    production helper rather than restated, so these renders follow the
    registry and tag axes instead of pinning a name the generator may not
    produce any more.
    """
    from osprey.deployment.compose_generator import resolve_image_defaults

    return resolve_image_defaults({"project_name": project_name})


def _render_copied_compose(project_path: Path, config: dict) -> dict:
    """Render the compose template `_inject_bluesky` copied, using the same
    context-key contract `compose_generator.render_template` uses, and parse
    the result. Closes the loop end-to-end: config.yml written by build_cmd.py
    -> env var read by docker-compose.yml.j2 -> app.py's guarded startup hook
    (2.14a) reads it back out of the container environment.
    """
    import yaml as pyyaml
    from jinja2 import Environment, FileSystemLoader

    # Both roots: the template imports the shared axis macros as
    # "services/_*.j2" (resolved from the project root, as at deploy time),
    # while the template itself is addressed by bare name from its own dir.
    env = Environment(
        loader=FileSystemLoader([str(project_path), str(project_path / "services" / "bluesky")])
    )
    tmpl = env.get_template("docker-compose.yml.j2")
    ctx = {
        "osprey_labels": {
            "project_name": "p",
            "project_root": str(project_path),
        },
        "osprey_images": _image_defaults("p"),
        "osprey_version": "",
        "system": {"timezone": "UTC"},
        "deployment": {},
        "services": config["services"],
        # The layout at the base this context implies — ``deployment`` is empty
        # here, so the default one. The template's ports read as
        # ``<key> | default(osprey_ports.<slot>, true)``.
        "osprey_ports": layout_ports(DEFAULT_PORT_BASE),
    }
    return pyyaml.safe_load(tmpl.render(ctx))


def test_the_retired_demo_runner_knob_reaches_nothing(tmp_path: Path) -> None:
    """The bridge does not run plans — the queueserver worker does — so there
    is no in-process demo runner and no `bluesky.demo_runner` knob to arm one.

    This is a standing guard, not a regression test for a bug: such a knob would
    render `BLUESKY_DEMO_RUNNER` into compose and print a console line promising
    a demo, so introducing either half would ship a config key that lies about
    what the deployment does. A stale `demo_runner:` left in a profile is
    ignored by the loader (the `bluesky` block ignores every unknown key), which
    is why the guard is on the OUTPUT rather than on the parse.
    """
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_bluesky(BlueskyConfig(), project_path)

    config = _read_config(project_path)
    assert "demo_runner" not in config["services"]["bluesky"]
    rendered = _render_copied_compose(project_path, config)
    assert "BLUESKY_DEMO_RUNNER" not in rendered["services"]["bluesky-bridge"]["environment"]


# ---------------------------------------------------------------------------
# Facility plan-directory deploy wiring (Task 1.4).
# ---------------------------------------------------------------------------


def test_inject_bluesky_plan_dir_written_to_config(tmp_path: Path) -> None:
    """A configured plan_dir is emitted into services.bluesky config."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_bluesky(BlueskyConfig(plan_dir="/opt/facility/plans"), project_path)

    config = _read_config(project_path)
    assert config["services"]["bluesky"]["plan_dir"] == "/opt/facility/plans"


def test_inject_bluesky_no_plan_dir_omits_key(tmp_path: Path) -> None:
    """No plan_dir configured -> no plan_dir key at all (bridge-only deploy unchanged)."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_bluesky(BlueskyConfig(), project_path)

    config = _read_config(project_path)
    assert "plan_dir" not in config["services"]["bluesky"]


def test_inject_bluesky_excluded_plans_written_to_config(tmp_path: Path) -> None:
    """Configured excluded_plans are emitted as an os.pathsep-joined string."""
    import os

    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_bluesky(BlueskyConfig(excluded_plans=["orm", "tune"]), project_path)

    config = _read_config(project_path)
    assert config["services"]["bluesky"]["excluded_plans"] == os.pathsep.join(["orm", "tune"])


def test_inject_bluesky_no_excluded_plans_omits_key(tmp_path: Path) -> None:
    """Empty excluded_plans -> no excluded_plans key at all (omit-when-empty)."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_bluesky(BlueskyConfig(), project_path)

    config = _read_config(project_path)
    assert "excluded_plans" not in config["services"]["bluesky"]


def test_inject_bluesky_devices_file_written_to_config(tmp_path: Path) -> None:
    """A configured devices_file is emitted into services.bluesky config."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_bluesky(BlueskyConfig(devices_file="/opt/facility/devices.yml"), project_path)

    config = _read_config(project_path)
    assert config["services"]["bluesky"]["devices_file"] == "/opt/facility/devices.yml"


def test_inject_bluesky_devices_file_default_written_when_unset(tmp_path: Path) -> None:
    """Unset devices_file -> the DEFAULT path is written anyway, not omitted.

    The one facility plan key that breaks the omit-when-unset pattern its two
    neighbours follow: a deployment always addresses devices, so the staging
    step reads a key that is always present instead of re-deriving the default
    from a key that is missing.
    """
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_bluesky(BlueskyConfig(), project_path)

    config = _read_config(project_path)
    assert config["services"]["bluesky"]["devices_file"] == "data/bluesky_devices.yml"


# ---------------------------------------------------------------------------
# Device-listing page size: the omit-when-EQUALS-DEFAULT facility plan key.
#
# The third contract in `_facility_plan_keys`, and the only one whose key is
# never "unset": `device_page_size` is an int with a dataclass default, so a
# profile that says nothing and a profile that authors 500 arrive identical.
# Both must render NO line, so that every project built before the key existed
# keeps its exact config.yml (and its exact compose render).
# ---------------------------------------------------------------------------


def test_inject_bluesky_device_page_size_written_to_config(tmp_path: Path) -> None:
    """A non-default device_page_size is emitted into services.bluesky config."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_bluesky(BlueskyConfig(device_page_size=200), project_path)

    config = _read_config(project_path)
    assert config["services"]["bluesky"]["device_page_size"] == 200


def test_inject_bluesky_default_device_page_size_omits_key(tmp_path: Path) -> None:
    """An unstated device_page_size -> no key at all (omit-when-default).

    The regression pin: the bridge falls back to the same default when the env
    var is absent, so an unauthored page size must leave the rendered block
    exactly as it was before this key existed.
    """
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_bluesky(BlueskyConfig(), project_path)

    config = _read_config(project_path)
    assert "device_page_size" not in config["services"]["bluesky"]


def test_inject_bluesky_device_page_size_authored_at_default_omits_key(
    tmp_path: Path,
) -> None:
    """Authoring the default EXPLICITLY renders no line either.

    Where this key parts company with `devices_file`: spelling the default out
    is not a request for a line in config.yml, because the value it would carry
    is the one the bridge already falls back to. Same deployed behaviour, same
    rendered artifact — asserted against the dataclass default rather than a
    literal, so the test cannot drift from the schema.
    """
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_bluesky(BlueskyConfig(device_page_size=BlueskyConfig.device_page_size), project_path)

    config = _read_config(project_path)
    assert "device_page_size" not in config["services"]["bluesky"]


def test_inject_bluesky_device_page_size_written_to_both_lanes(tmp_path: Path) -> None:
    """On a two-lane deploy BOTH lane blocks carry the authored page size.

    The page size describes the facility's device file, not a control-system
    target, so it belongs to the shared facility plan keys: two lanes reading
    one devices file must page it the same way.
    """
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)
    # `second_lane` needs a switchable baseline; a VA baseline pairs lane 1 (va)
    # with a `live` second lane and needs no VA block passed in.
    yaml = YAML()
    config = _read_config(project_path)
    config["control_system"] = {"type": "virtual_accelerator"}
    with open(project_path / "config.yml", "w") as fh:
        yaml.dump(config, fh)

    _inject_bluesky(BlueskyConfig(second_lane=True, device_page_size=250), project_path)

    config = _read_config(project_path)
    assert config["services"]["bluesky"]["device_page_size"] == 250
    assert config["services"]["bluesky_live"]["device_page_size"] == 250


def test_inject_bluesky_devices_file_survives_the_omit_when_unset_neighbours(
    tmp_path: Path,
) -> None:
    """A deploy that configures neither plan_dir nor excluded_plans still
    carries devices_file — the mixed contract in one assertion."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_bluesky(BlueskyConfig(devices_file="devices/beamline.yml"), project_path)

    svc = _read_config(project_path)["services"]["bluesky"]
    assert svc["devices_file"] == "devices/beamline.yml"
    assert "plan_dir" not in svc
    assert "excluded_plans" not in svc


def test_plan_dir_mount_and_env_round_trip_through_compose(tmp_path: Path) -> None:
    """A configured plan_dir renders both the read-only bind mount and the
    in-container BLUESKY_PLAN_DIRS env var the loader (plan_loader.py) reads
    — the host path must never leak into the container environment."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_bluesky(BlueskyConfig(plan_dir="/opt/facility/plans"), project_path)
    config = _read_config(project_path)
    rendered = _render_copied_compose(project_path, config)

    bridge = rendered["services"]["bluesky-bridge"]
    assert bridge["environment"]["BLUESKY_PLAN_DIRS"] == "/app/project/plans"
    assert "/opt/facility/plans:/app/project/plans:ro" in bridge["volumes"]
    # The host path never appears in the container's environment block.
    assert "/opt/facility/plans" not in bridge["environment"].values()


def test_plan_dir_absent_omits_mount_and_env(tmp_path: Path) -> None:
    """Regression: a bridge-only deploy (no plan_dir) renders no plan mount
    and no BLUESKY_PLAN_DIRS env var — unchanged from every prior build."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_bluesky(BlueskyConfig(), project_path)
    config = _read_config(project_path)
    rendered = _render_copied_compose(project_path, config)

    bridge = rendered["services"]["bluesky-bridge"]
    assert "BLUESKY_PLAN_DIRS" not in bridge["environment"]
    assert not any(str(v).endswith(":/app/project/plans:ro") for v in bridge["volumes"])


def test_excluded_plans_env_round_trips_through_compose(tmp_path: Path) -> None:
    """A configured excluded_plans renders the in-container BLUESKY_EXCLUDED_PLANS
    env var the plan loader reads to drop catalog entries — the os.pathsep-joined
    string flows straight through to the rendered compose environment (Task 3.3,
    success criterion 6, asserted through the rendered artifact)."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_bluesky(BlueskyConfig(excluded_plans=["orm"]), project_path)
    config = _read_config(project_path)
    rendered = _render_copied_compose(project_path, config)

    bridge = rendered["services"]["bluesky-bridge"]
    assert bridge["environment"]["BLUESKY_EXCLUDED_PLANS"] == "orm"


def test_excluded_plans_absent_omits_env(tmp_path: Path) -> None:
    """Regression: a deploy with no excluded_plans renders no BLUESKY_EXCLUDED_PLANS
    env var — an empty exclusion set contributes nothing to the container env."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_bluesky(BlueskyConfig(), project_path)
    config = _read_config(project_path)
    rendered = _render_copied_compose(project_path, config)

    bridge = rendered["services"]["bluesky-bridge"]
    assert "BLUESKY_EXCLUDED_PLANS" not in bridge["environment"]


def test_loopback_bind_and_failclosed_token_survive_plan_dir_wiring(tmp_path: Path) -> None:
    """Regression guard for the two invariants this task must not touch:
    the port bind stays loopback-only and BLUESKY_LAUNCH_TOKEN keeps no
    ``:-`` default (an unset token must fail closed, never boot with a
    guessable secret)."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_bluesky(BlueskyConfig(plan_dir="/opt/facility/plans"), project_path)

    # Asserted on the RENDER, not on the copied template's source: the template
    # renders one stack per Bluesky plan lane, so the port and the token are
    # per-lane expressions in the source and only become these literals once a
    # deployment's lanes are known. This is the single-lane render — every
    # project that has not opted into a second lane.
    rendered = _render_copied_compose(project_path, _read_config(project_path))
    bridge = rendered["services"]["bluesky-bridge"]

    assert bridge["ports"] == ["127.0.0.1:10080:10080"]
    assert bridge["environment"]["BLUESKY_LAUNCH_TOKEN"] == "${BLUESKY_LAUNCH_TOKEN}"
    assert ":-" not in bridge["environment"]["BLUESKY_LAUNCH_TOKEN"]


# ---------------------------------------------------------------------------
# Build-profile parsing: bluesky stays opt-in, not default-on.
# ---------------------------------------------------------------------------


def test_profile_without_bluesky_key_leaves_bluesky_none() -> None:
    profile = _parse_profile({"name": "no-plan-here"})
    assert profile.bluesky is None


def test_profile_bluesky_key_parses_overrides() -> None:
    profile = _parse_profile(
        {
            "name": "with-plan",
            "bluesky": {
                "port": 8123,
                "tiled_enabled": True,
                "tiled_port": 8124,
                "plan_dir": "/opt/facility/plans",
            },
        }
    )
    assert profile.bluesky is not None
    assert profile.bluesky.port == 8123
    assert profile.bluesky.tiled_enabled is True
    assert profile.bluesky.tiled_port == 8124
    assert profile.bluesky.plan_dir == "/opt/facility/plans"


def test_profile_bluesky_key_defaults_when_empty_mapping() -> None:
    profile = _parse_profile({"name": "with-plan-defaults", "bluesky": {}})
    assert profile.bluesky is not None
    assert profile.bluesky.port == 10080
    assert profile.bluesky.tiled_enabled is False
    assert profile.bluesky.tiled_port == 10070
    assert profile.bluesky.plan_dir is None
    assert profile.bluesky.excluded_plans == []


# ---------------------------------------------------------------------------
# Plan-catalog exclusions (Task 3.1): hide named plans from the agent while the
# bluesky server stays enabled (dev/local convenience).
# ---------------------------------------------------------------------------


def test_profile_bluesky_excluded_plans_parses() -> None:
    profile = _parse_profile({"name": "with-exclusions", "bluesky": {"excluded_plans": ["orm"]}})
    assert profile.bluesky is not None
    assert profile.bluesky.excluded_plans == ["orm"]


def test_profile_bluesky_excluded_plans_defaults_empty() -> None:
    profile = _parse_profile({"name": "no-exclusions", "bluesky": {}})
    assert profile.bluesky is not None
    assert profile.bluesky.excluded_plans == []


def test_profile_bluesky_excluded_plans_non_list_raises() -> None:
    with pytest.raises(BuildProfileError):
        _parse_profile({"name": "bad", "bluesky": {"excluded_plans": 5}})


def test_profile_bluesky_excluded_plans_non_str_element_raises() -> None:
    with pytest.raises(BuildProfileError):
        _parse_profile({"name": "bad", "bluesky": {"excluded_plans": ["orm", 7]}})


def test_profile_bluesky_devices_file_parses() -> None:
    profile = _parse_profile(
        {"name": "with-devices", "bluesky": {"devices_file": "/facility/devices.yml"}}
    )
    assert profile.bluesky is not None
    assert profile.bluesky.devices_file == "/facility/devices.yml"


def test_profile_bluesky_devices_file_defaults_to_the_project_relative_path() -> None:
    profile = _parse_profile({"name": "no-devices", "bluesky": {}})
    assert profile.bluesky is not None
    assert profile.bluesky.devices_file == "data/bluesky_devices.yml"


def test_profile_bluesky_devices_file_non_str_raises() -> None:
    with pytest.raises(BuildProfileError):
        _parse_profile({"name": "bad", "bluesky": {"devices_file": 5}})


def test_profile_bluesky_devices_file_empty_raises() -> None:
    """An empty string is a path to nothing, not an opt-out of the default."""
    with pytest.raises(BuildProfileError):
        _parse_profile({"name": "bad", "bluesky": {"devices_file": ""}})


# ---------------------------------------------------------------------------
# Device-listing page size through the rendered compose file.
#
# The other half of the omit-when-EQUALS-DEFAULT contract asserted above: what
# `_facility_plan_keys` writes into config.yml has to survive the template's
# `{% if svc.device_page_size %}` guard and arrive at the bridge under the name
# `device_page_size()` reads. Rendered end to end, because a key written to
# config.yml that no container ever sees is indistinguishable from a key that
# was never written.
# ---------------------------------------------------------------------------


def test_device_page_size_env_round_trips_through_compose(tmp_path: Path) -> None:
    """An authored page size reaches the bridge as BLUESKY_DEVICE_PAGE_SIZE.

    Quoted, because a compose environment carries strings: the bridge parses
    the number back out, and pinning the rendered spelling keeps the template
    from drifting into an unquoted int the YAML loader would type differently.
    """
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_bluesky(BlueskyConfig(device_page_size=200), project_path)
    config = _read_config(project_path)
    rendered = _render_copied_compose(project_path, config)

    bridge = rendered["services"]["bluesky-bridge"]
    assert bridge["environment"]["BLUESKY_DEVICE_PAGE_SIZE"] == "200"


def test_default_device_page_size_omits_the_env(tmp_path: Path) -> None:
    """Regression: an unauthored page size renders the key nowhere at all.

    This is what keeps every project built before the key existed rendering
    byte-for-byte what it rendered then — the guard has to emit zero bytes, not
    an empty assignment, so the assertion is against the whole rendered file
    rather than one container's environment.
    """
    import yaml as pyyaml

    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_bluesky(BlueskyConfig(), project_path)
    config = _read_config(project_path)
    rendered = _render_copied_compose(project_path, config)

    assert "BLUESKY_DEVICE_PAGE_SIZE" not in pyyaml.safe_dump(rendered)


def test_device_page_size_stops_at_the_bridge(tmp_path: Path) -> None:
    """The queueserver block carries no page size even when one is authored.

    The RE Manager's worker builds devices; it never serves a listing, so the
    bound on `GET /devices` and on the unknown-device refusal's inline
    threshold belongs to the HTTP facade alone.
    """
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_bluesky(BlueskyConfig(device_page_size=200), project_path)
    config = _read_config(project_path)
    rendered = _render_copied_compose(project_path, config)

    queueserver = rendered["services"]["queueserver"]
    assert "BLUESKY_DEVICE_PAGE_SIZE" not in (queueserver["environment"] or {})
