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
    assert svc["port"] == 8090
    assert svc["tiled_enabled"] is False
    assert svc["tiled_port"] == 8091
    # demo_runner defaults OFF — a facility profile must opt in explicitly;
    # it must never silently override real device/plan wiring.
    assert svc["demo_runner"] is False
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


def test_inject_bluesky_demo_runner_opt_in(tmp_path: Path) -> None:
    """demo_runner=True is written through to config.yml (2.14a's app.py hook
    reads BLUESKY_DEMO_RUNNER off the rendered compose env; this test only
    covers this task's half — the config.yml/compose passthrough)."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_bluesky(BlueskyConfig(demo_runner=True), project_path)

    config = _read_config(project_path)
    assert config["services"]["bluesky"]["demo_runner"] is True


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


def _render_copied_compose(project_path: Path, config: dict) -> dict:
    """Render the compose template `_inject_bluesky` copied, using the same
    context-key contract `compose_generator.render_template` uses, and parse
    the result. Closes the loop end-to-end: config.yml written by build_cmd.py
    -> env var read by docker-compose.yml.j2 -> app.py's guarded startup hook
    (2.14a) reads it back out of the container environment.
    """
    import yaml as pyyaml
    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader(str(project_path / "services" / "bluesky")))
    tmpl = env.get_template("docker-compose.yml.j2")
    ctx = {
        "osprey_labels": {
            "project_name": "p",
            "project_root": str(project_path),
            "deployed_at": "x",
        },
        "osprey_version": "",
        "system": {"timezone": "UTC"},
        "deployment": {},
        "services": config["services"],
    }
    return pyyaml.safe_load(tmpl.render(ctx))


def test_demo_runner_env_passthrough_round_trips_through_compose(tmp_path: Path) -> None:
    """The config.yml demo_runner flag `_inject_bluesky` writes must actually
    reach the container as BLUESKY_DEMO_RUNNER — the other half of 2.14a's
    contract (app.py reads the env var this template renders)."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_bluesky(BlueskyConfig(demo_runner=True), project_path)
    config = _read_config(project_path)
    rendered = _render_copied_compose(project_path, config)
    assert rendered["services"]["bluesky-bridge"]["environment"]["BLUESKY_DEMO_RUNNER"] == "true"


def test_demo_runner_off_by_default_omits_env_var(tmp_path: Path) -> None:
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_bluesky(BlueskyConfig(), project_path)
    config = _read_config(project_path)
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

    _inject_bluesky(BlueskyConfig(plan_dir="/opt/facility/scan_plans"), project_path)

    config = _read_config(project_path)
    assert config["services"]["bluesky"]["plan_dir"] == "/opt/facility/scan_plans"


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


def test_plan_dir_mount_and_env_round_trip_through_compose(tmp_path: Path) -> None:
    """A configured plan_dir renders both the read-only bind mount and the
    in-container BLUESKY_PLAN_DIRS env var the loader (plan_loader.py) reads
    — the host path must never leak into the container environment."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_bluesky(BlueskyConfig(plan_dir="/opt/facility/scan_plans"), project_path)
    config = _read_config(project_path)
    rendered = _render_copied_compose(project_path, config)

    bridge = rendered["services"]["bluesky-bridge"]
    assert bridge["environment"]["BLUESKY_PLAN_DIRS"] == "/app/project/plans"
    assert "/opt/facility/scan_plans:/app/project/plans:ro" in bridge["volumes"]
    # The host path never appears in the container's environment block.
    assert "/opt/facility/scan_plans" not in bridge["environment"].values()


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

    _inject_bluesky(BlueskyConfig(plan_dir="/opt/facility/scan_plans"), project_path)

    template_text = (project_path / "services" / "bluesky" / "docker-compose.yml.j2").read_text()
    assert (
        "\"{{ deployment.bind_address | default('127.0.0.1') }}:"
        "{{ services.bluesky.port | default(8090) }}:"
        '{{ services.bluesky.port | default(8090) }}"' in template_text
    )
    assert "BLUESKY_LAUNCH_TOKEN: ${BLUESKY_LAUNCH_TOKEN}" in template_text
    assert "BLUESKY_LAUNCH_TOKEN: ${BLUESKY_LAUNCH_TOKEN:-" not in template_text


# ---------------------------------------------------------------------------
# Build-profile parsing: bluesky stays opt-in, not default-on.
# ---------------------------------------------------------------------------


def test_profile_without_bluesky_key_leaves_bluesky_none() -> None:
    profile = _parse_profile({"name": "no-scan-here"})
    assert profile.bluesky is None


def test_profile_bluesky_key_parses_overrides() -> None:
    profile = _parse_profile(
        {
            "name": "with-scan",
            "bluesky": {
                "port": 8123,
                "tiled_enabled": True,
                "tiled_port": 8124,
                "demo_runner": True,
                "plan_dir": "/opt/facility/scan_plans",
            },
        }
    )
    assert profile.bluesky is not None
    assert profile.bluesky.port == 8123
    assert profile.bluesky.tiled_enabled is True
    assert profile.bluesky.tiled_port == 8124
    assert profile.bluesky.demo_runner is True
    assert profile.bluesky.plan_dir == "/opt/facility/scan_plans"


def test_profile_bluesky_key_defaults_when_empty_mapping() -> None:
    profile = _parse_profile({"name": "with-scan-defaults", "bluesky": {}})
    assert profile.bluesky is not None
    assert profile.bluesky.port == 8090
    assert profile.bluesky.tiled_enabled is False
    assert profile.bluesky.tiled_port == 8091
    assert profile.bluesky.demo_runner is False
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
