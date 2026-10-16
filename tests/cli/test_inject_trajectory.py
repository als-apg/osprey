"""Tests for the ``_inject_trajectory`` build step in ``osprey.cli.build_cmd``.

The step copies the bundled ``templates/services/trajectory/`` compose
template, writes ``services.trajectory {path, port, target, model?, network,
bind?}``, registers the service in ``deployed_services`` (additively), and
registers the ``web.panels.trajectory`` entry with a bare service-root ``url``
plus ``path``/``label``/``health_endpoint`` — explicit overrides winning.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ruamel.yaml import YAML

from osprey.cli.build_cmd import _inject_trajectory
from osprey.cli.build_profile_schema import TrajectoryConfig
from osprey.port_layout import default_port

#: The service's layout slot at the default base.
_PORT = default_port("trajectory")


def _write_config(project_path: Path, *, extra: dict | None = None) -> None:
    """Write a minimal config.yml with a pre-existing deployed service."""
    config: dict = {
        "control_system": {"type": "epics"},
        "deployed_services": ["postgresql"],
        "services": {"postgresql": {}},
    }
    if extra:
        config.update(extra)
    with open(project_path / "config.yml", "w") as fh:
        YAML().dump(config, fh)


def _read_config(project_path: Path) -> dict:
    with open(project_path / "config.yml") as fh:
        return YAML().load(fh)


def _project(tmp_path: Path, **kwargs) -> Path:
    project_path = tmp_path / "project"
    project_path.mkdir(parents=True)
    _write_config(project_path, **kwargs)
    return project_path


def test_copies_template_dir(tmp_path: Path) -> None:
    project_path = _project(tmp_path)

    _inject_trajectory(TrajectoryConfig(), project_path)

    assert (project_path / "services" / "trajectory" / "docker-compose.yml.j2").is_file()


def test_writes_service_block_with_defaults(tmp_path: Path) -> None:
    """Every default: the layout port, the baseline target, the default network,
    and no model or bind key at all."""
    project_path = _project(tmp_path)

    _inject_trajectory(TrajectoryConfig(), project_path)

    block = _read_config(project_path)["services"]["trajectory"]
    assert block["path"] == "./services/trajectory"
    assert block["port"] == _PORT
    # An epics control system is a real one: its baseline target is live.
    assert block["target"] == "live"
    assert block["network"] == "bridge"
    assert "model" not in block
    assert "bind" not in block


def test_baseline_target_follows_control_system(tmp_path: Path) -> None:
    """With no target in the profile the block names the deployment's baseline,
    the same target the service itself falls back to."""
    project_path = _project(tmp_path, extra={"control_system": {"type": "mock"}})

    _inject_trajectory(TrajectoryConfig(), project_path)

    from osprey_connectors.types import baseline_target

    block = _read_config(project_path)["services"]["trajectory"]
    assert block["target"] == baseline_target({"type": "mock"})


def test_profile_knobs_carried_through(tmp_path: Path) -> None:
    project_path = _project(tmp_path)

    _inject_trajectory(
        TrajectoryConfig(target="va", model="ring", network="host", bind="127.0.0.1"),
        project_path,
    )

    block = _read_config(project_path)["services"]["trajectory"]
    assert block["target"] == "va"
    assert block["model"] == "ring"
    assert block["network"] == "host"
    assert block["bind"] == "127.0.0.1"


def test_port_follows_resolved_base(tmp_path: Path) -> None:
    project_path = _project(tmp_path)

    _inject_trajectory(TrajectoryConfig(), project_path, base=20000)

    config = _read_config(project_path)
    assert config["services"]["trajectory"]["port"] == default_port("trajectory", base=20000)
    assert config["web"]["panels"]["trajectory"]["url"] == (
        f"${{TRAJECTORY_URL:-http://localhost:{default_port('trajectory', base=20000)}}}"
    )


def test_config_port_override_wins(tmp_path: Path) -> None:
    """A ``services.trajectory.port`` merged from the profile's ``config:``
    moves the service off its slot, and the panel url follows it."""
    project_path = _project(
        tmp_path,
        extra={
            "services": {"postgresql": {}, "trajectory": {"port": 31234, "env": ["FOO"]}},
        },
    )

    _inject_trajectory(TrajectoryConfig(), project_path)

    config = _read_config(project_path)
    block = config["services"]["trajectory"]
    assert block["port"] == 31234
    # An author-written axis survives the whole-block replacement.
    assert list(block["env"]) == ["FOO"]
    assert config["web"]["panels"]["trajectory"]["url"] == (
        "${TRAJECTORY_URL:-http://localhost:31234}"
    )


def test_deployed_services_additive_and_idempotent(tmp_path: Path) -> None:
    project_path = _project(tmp_path)

    _inject_trajectory(TrajectoryConfig(), project_path)
    _inject_trajectory(TrajectoryConfig(), project_path)

    deployed = [str(s) for s in _read_config(project_path)["deployed_services"]]
    assert deployed.count("trajectory") == 1
    assert "postgresql" in deployed


def test_registers_panel(tmp_path: Path) -> None:
    project_path = _project(tmp_path)

    _inject_trajectory(TrajectoryConfig(), project_path)

    panel = _read_config(project_path)["web"]["panels"]["trajectory"]
    assert panel["url"] == f"${{TRAJECTORY_URL:-http://localhost:{_PORT}}}"
    assert panel["path"] == "/panel/"
    assert panel["label"] == "TRAJECTORY"
    assert panel["health_endpoint"] == "/health"


def test_explicit_panel_override_wins(tmp_path: Path) -> None:
    """A partially authored panel keeps what it set; only the gaps are derived."""
    project_path = _project(
        tmp_path,
        extra={"web": {"panels": {"trajectory": {"url": "http://elsewhere:9", "label": "TBT"}}}},
    )

    _inject_trajectory(TrajectoryConfig(), project_path)

    panel = _read_config(project_path)["web"]["panels"]["trajectory"]
    assert panel["url"] == "http://elsewhere:9"
    assert panel["label"] == "TBT"
    assert panel["path"] == "/panel/"
    assert panel["health_endpoint"] == "/health"


def test_missing_config_is_noop(tmp_path: Path) -> None:
    project_path = tmp_path / "project"
    project_path.mkdir()

    _inject_trajectory(TrajectoryConfig(), project_path)

    assert not (project_path / "config.yml").exists()


def test_hint_names_token_and_shared_codes(tmp_path: Path, caplog) -> None:
    """The post-build hint names the service's token variable and the coded
    refusals every arming service shares."""
    from osprey.utils.launch_token import LAUNCH_TOKEN_INVALID, LAUNCH_TOKEN_REQUIRED

    project_path = _project(tmp_path)
    with caplog.at_level(logging.DEBUG):
        logging.getLogger().setLevel(logging.DEBUG)
        _inject_trajectory(TrajectoryConfig(), project_path)

    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "TRAJECTORY_LAUNCH_TOKEN" in text
    assert LAUNCH_TOKEN_REQUIRED in text
    assert LAUNCH_TOKEN_INVALID in text


def test_registered_in_inject_services(tmp_path: Path) -> None:
    """A profile with a ``trajectory:`` block deploys the service through
    ``_inject_services``; one without it deploys nothing of the kind."""
    from types import SimpleNamespace

    from osprey.cli import build_cmd

    def profile(trajectory: TrajectoryConfig | None) -> SimpleNamespace:
        return SimpleNamespace(
            deploy_services=True,
            services={},
            config={},
            dispatch=None,
            nextcloud_bridge=None,
            gchat_bridge=None,
            teams_bridge=None,
            bluesky=None,
            bluesky_web=None,
            virtual_accelerator=None,
            va_archiver=None,
            trajectory=trajectory,
        )

    with_block = _project(tmp_path / "a")
    injected = build_cmd._inject_services(profile(TrajectoryConfig()), tmp_path, with_block)
    assert "trajectory" in injected
    assert "trajectory" in [str(s) for s in _read_config(with_block)["deployed_services"]]

    without = _project(tmp_path / "b")
    assert "trajectory" not in build_cmd._inject_services(profile(None), tmp_path, without)
    assert "trajectory" not in _read_config(without)["services"]
    assert "_inject_trajectory" in build_cmd.__all__
