"""``osprey up`` over a deployment's projected application containers.

The deploy config is built from :func:`~osprey.apps.projection.project_deployment`
exactly as the build injects it: ``services.<key>`` is the projected ``config``
block under the service's ``path``, and ``<key>`` joins ``deployed_services``.
"""

from __future__ import annotations

import copy
from typing import Any

from osprey.apps.manifest import parse_manifest
from osprey.apps.projection import project_deployment
from osprey.deployment import container_lifecycle
from osprey.deployment.compose_generator import AGENT_RECORD_TEMPLATE_DIR

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
