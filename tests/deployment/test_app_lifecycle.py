"""``osprey up`` over a deployment's projected application containers.

The deploy config is built from :func:`~osprey.apps.projection.project_deployment`
exactly as the build injects it: ``services.<key>`` is the projected ``config``
block under the service's ``path``, and ``<key>`` joins ``deployed_services``.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from osprey.apps.manifest import launch_token_env, parse_manifest
from osprey.apps.projection import project_deployment
from osprey.apps.rules import builtin_credential_names
from osprey.deployment import container_lifecycle
from osprey.deployment.compose_generator import AGENT_RECORD_TEMPLATE_DIR
from tests.deployment._dotenv import project_env

KNOB: dict[str, Any] = {
    "name": "knob-demo",
    "containers": {
        "knob-demo": {"runtime": "osprey", "command": ["python", "-m", "knob_demo.server"]},
        "knob-cache": {"image": "redis:7.2", "fragment": "compose/knob-cache/"},
    },
    "panel": {"label": "Knob", "container": "knob-demo", "launch_token": True},
}

VIEWER: dict[str, Any] = {
    "name": "viewer",
    "containers": {"viewer": {"image": "ghcr.io/example/viewer:1.0"}},
    "panel": {"label": "Viewer", "container": "viewer", "launch_token": True},
}


def _deploy_config(*apps: dict[str, Any]) -> dict[str, Any]:
    manifests = [parse_manifest(copy.deepcopy(data), app=data["name"]) for data in apps]
    projected = project_deployment(
        manifests, {"apps": {data["name"]: {} for data in apps}}, claimed_ports=frozenset()
    )
    config: dict[str, Any] = {"services": {}, "deployed_services": []}
    for key, entry in projected["services"].items():
        config["services"][key] = {"path": f"./services/{key}", **entry["config"]}
        config["deployed_services"].append(key)
    return config


# --- the project image --------------------------------------------------------------------


def test_a_runtime_container_runs_on_the_project_image() -> None:
    services = container_lifecycle._deployed_project_image_services(_deploy_config(KNOB))
    assert "knob-demo" in services
    assert "knob-cache" not in services


def test_an_image_container_never_joins_the_project_image() -> None:
    services = container_lifecycle._deployed_project_image_services(_deploy_config(VIEWER))
    assert "viewer" not in services


def test_an_application_alone_builds_the_project_image() -> None:
    """No dispatch worker, ARIEL sync, archive or record: the runtime container still
    needs the image it runs."""
    config = _deploy_config(KNOB)
    no_record = ["docker-compose.yml"]
    services = container_lifecycle._deployed_project_image_services(config, no_record)
    assert services == ["knob-demo"]
    assert AGENT_RECORD_TEMPLATE_DIR not in services
    assert container_lifecycle._project_image_build_target(config, {}, no_record) is not None


def test_an_undeployed_projected_row_builds_nothing() -> None:
    config = _deploy_config(KNOB)
    config["deployed_services"] = []
    services = container_lifecycle._deployed_project_image_services(config, ["docker-compose.yml"])
    assert services == []


# --- the launch token ---------------------------------------------------------------------


@pytest.fixture
def deploy(stub_deploy_up, monkeypatch, tmp_path):
    """Run ``deploy_up`` over a projected config, with no image build and no host."""
    monkeypatch.setattr(container_lifecycle, "_build_project_image", lambda *a, **k: None)
    for name in (launch_token_env("knob-demo"), launch_token_env("viewer")):
        monkeypatch.delenv(name, raising=False)

    def _run(config: dict[str, Any]) -> dict[str, str]:
        stub_deploy_up(config, seam="subprocess.run")
        container_lifecycle.deploy_up(str(tmp_path / "config.yml"), detached=True)
        return project_env(tmp_path)

    return _run


def test_a_runtime_panel_launch_token_is_minted(deploy) -> None:
    env = deploy(_deploy_config(KNOB))
    assert env.get(launch_token_env("knob-demo"))


def test_an_image_panel_launch_token_is_minted(deploy) -> None:
    env = deploy(_deploy_config(VIEWER))
    assert env.get(launch_token_env("viewer"))


def test_the_minted_token_is_the_recorded_name(deploy) -> None:
    """The row's ``launch_token_env`` is the name minted; it is never re-derived."""
    config = _deploy_config(KNOB)
    config["services"]["knob-demo"]["launch_token_env"] = "KNOB_PANEL_LAUNCH_TOKEN"
    env = deploy(config)
    assert env.get("KNOB_PANEL_LAUNCH_TOKEN")
    assert launch_token_env("knob-demo") not in env


def test_a_panel_without_a_launch_token_mints_none(deploy) -> None:
    data = copy.deepcopy(KNOB)
    data["panel"] = {"label": "Knob", "container": "knob-demo"}
    env = deploy(_deploy_config(data))
    assert launch_token_env("knob-demo") not in env


def test_the_launch_token_mint_is_idempotent(deploy, tmp_path) -> None:
    config = _deploy_config(KNOB)
    first = deploy(config)
    second = deploy(config)
    name = launch_token_env("knob-demo")
    assert first[name] == second[name]
    assert (tmp_path / ".env").read_text().count(f"{name}=") == 1


# --- OSPREY's own credentials stay the static set -----------------------------------------


def test_builtin_token_vars_are_the_static_table() -> None:
    expected = {
        name for names in container_lifecycle._SERVICE_TOKEN_VARS.values() for name in names
    }
    assert container_lifecycle.builtin_token_vars() == frozenset(expected)


def test_a_projected_app_own_token_is_not_a_builtin_credential(deploy) -> None:
    """Minting an application's token does not make it one of OSPREY's own names,
    so the collision rule never refuses the application it was minted for."""
    deploy(_deploy_config(KNOB))
    name = launch_token_env("knob-demo")
    assert name not in container_lifecycle.builtin_token_vars()
    assert name not in builtin_credential_names()
