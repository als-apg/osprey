"""Projecting an application's containers into ``services.<key>`` entries."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from osprey.apps.errors import AppManifestError
from osprey.apps.manifest import AppManifest, launch_token_env, parse_manifest
from osprey.apps.projection import (
    APP_CONFIG_FILE,
    APP_RUNTIME_TEMPLATE,
    project_deployment,
)
from osprey.cli.build_profile_schema import ServiceDef
from osprey.port_layout import DEFAULT_PORT_BASE

FACILITY_BAND = DEFAULT_PORT_BASE + 900

KNOB: dict[str, Any] = {
    "name": "knob-demo",
    "containers": {
        "knob-demo": {
            "runtime": "osprey",
            "command": ["python", "-m", "knob_demo.server"],
            "health": "/health",
            "drives_target": True,
            "control_identity": True,
            "state": "var/knob-demo",
            "env": ["KNOB_DEMO_MODE"],
        },
        "knob-cache": {"image": "redis:7.2", "fragment": "compose/knob-cache/"},
    },
    "panel": {
        "label": "Knob",
        "container": "knob-demo",
        "launch_token": True,
        "roster_grant": {"single_user_alias": True},
    },
}

VIEWER: dict[str, Any] = {
    "name": "viewer",
    "containers": {"viewer": {"image": "ghcr.io/example/viewer:1.0", "port": 18123}},
    "audit_identity": "viewer-audit",
}


def _manifest(data: dict[str, Any]) -> AppManifest:
    return parse_manifest(copy.deepcopy(data), app=data["name"])


def _project(
    manifests: list[AppManifest],
    config: dict[str, Any],
    claimed: frozenset[int] = frozenset(),
) -> dict[str, Any]:
    return project_deployment(manifests, config, claimed_ports=claimed)


def _services(result: dict[str, Any]) -> dict[str, Any]:
    return result["services"]


def test_runtime_container_projects_the_bundled_template() -> None:
    result = _project([_manifest(KNOB)], {"apps": {"knob-demo": {}}})
    entry = _services(result)["knob-demo"]
    assert entry["template"] == APP_RUNTIME_TEMPLATE == "osprey.app_runtime"
    block = entry["config"]
    assert block["app"] == "knob-demo"
    assert block["command"] == ["python", "-m", "knob_demo.server"]
    assert block["port"] == FACILITY_BAND
    assert block["url"] == f"http://127.0.0.1:{FACILITY_BAND}"
    assert block["health"] == "/health"
    assert block["env"] == ["KNOB_DEMO_MODE"]
    assert block["state"] == "var/knob-demo"
    assert block["drives_target"] is True
    assert block["control_identity"] is True
    assert block["audit_identity"] == "knob-demo"
    assert block["config_file"] == f"./services/knob-demo/{APP_CONFIG_FILE}"


def test_panel_container_carries_the_roster_grant_and_launch_token_name() -> None:
    services = _services(_project([_manifest(KNOB)], {"apps": {"knob-demo": {}}}))
    panel = services["knob-demo"]["config"]
    assert panel["roster_grant"] == {"panel_id": "knob-demo", "single_user_alias": True}
    assert panel["launch_token_env"] == launch_token_env("knob-demo")
    other = services["knob-cache"]["config"]
    assert "roster_grant" not in other
    assert "launch_token_env" not in other


def test_panel_without_grant_or_token_projects_neither() -> None:
    data = copy.deepcopy(KNOB)
    data["panel"] = {"label": "Knob", "container": "knob-demo"}
    block = _services(_project([_manifest(data)], {"apps": {"knob-demo": {}}}))["knob-demo"]
    assert "roster_grant" not in block["config"]
    assert "launch_token_env" not in block["config"]


def test_image_container_projects_its_fragment_directory() -> None:
    services = _services(_project([_manifest(KNOB)], {"apps": {"knob-demo": {}}}))
    entry = services["knob-cache"]
    assert entry["template"] == "apps/knob-demo/compose/knob-cache"
    block = entry["config"]
    assert block["app"] == "knob-demo"
    assert block["image"] == "redis:7.2"
    assert block["port"] == FACILITY_BAND + 1
    assert block["url"] == f"http://127.0.0.1:{FACILITY_BAND + 1}"
    assert block["env"] == []
    assert block["config_file"] == f"./services/knob-cache/{APP_CONFIG_FILE}"
    assert block["drives_target"] is False
    assert block["control_identity"] is False
    assert block["audit_identity"] == "knob-demo"
    # The fragment owns its command, binds, volumes, env_file and depends_on.
    for key in ("command", "binds", "volumes", "env_file", "depends_on"):
        assert key not in block


def test_image_container_without_fragment_key_uses_its_compose_directory() -> None:
    services = _services(_project([_manifest(VIEWER)], {"apps": {"viewer": {}}}))
    assert services["viewer"]["template"] == "apps/viewer/compose/viewer"
    assert services["viewer"]["config"]["audit_identity"] == "viewer-audit"


def test_profile_axis_keys_are_carried_into_the_service() -> None:
    config = {
        "apps": {
            "knob-demo": {
                "services": {
                    "knob-demo": {
                        "network": "host",
                        "env": ["EPICS_CA_ADDR_LIST", "KNOB_DEMO_MODE"],
                    },
                    "knob-cache": {"env": ["REDIS_PASSWORD"]},
                }
            }
        }
    }
    services = _services(_project([_manifest(KNOB)], config))
    runtime = services["knob-demo"]["config"]
    assert runtime["network"] == "host"
    # The manifest's names first, then the profile's, each once.
    assert runtime["env"] == ["KNOB_DEMO_MODE", "EPICS_CA_ADDR_LIST"]
    cache = services["knob-cache"]["config"]
    assert "network" not in cache
    assert cache["env"] == ["REDIS_PASSWORD"]


def test_entries_are_profile_service_shaped() -> None:
    config = {"apps": {"knob-demo": {"services": {"knob-demo": {"network": "host"}}}}}
    services = _services(_project([_manifest(KNOB)], config))
    for key, entry in services.items():
        service = ServiceDef(**entry)
        assert service.config["app"] == "knob-demo", key
    runtime = ServiceDef(**services["knob-demo"])
    assert runtime.network_mode() == "host"
    assert runtime.env_names() == ["KNOB_DEMO_MODE"]


def test_only_applications_the_profile_names_are_projected() -> None:
    manifests = [_manifest(KNOB), _manifest(VIEWER)]
    assert set(_services(_project(manifests, {"apps": {"viewer": {}}}))) == {"viewer"}
    assert _project(manifests, {}) == {"services": {}}
    assert _project(manifests, {"apps": {"viewer": None}})["services"].keys() == {"viewer"}


def test_profile_naming_an_application_without_a_manifest_refuses() -> None:
    with pytest.raises(AppManifestError) as excinfo:
        _project([_manifest(KNOB)], {"apps": {"ghost": {}}})
    assert excinfo.value.app == "ghost"
    assert excinfo.value.rule == "missing_manifest"


def test_ports_follow_the_resolved_base_in_declaration_order() -> None:
    config = {"deployment": {"port_base": 20000}, "apps": {"viewer": {}, "knob-demo": {}}}
    services = _services(_project([_manifest(KNOB), _manifest(VIEWER)], config))
    assert list(services) == ["viewer", "knob-demo", "knob-cache"]
    assert services["viewer"]["config"]["port"] == 18123  # pinned
    assert services["knob-demo"]["config"]["port"] == 20900
    assert services["knob-cache"]["config"]["port"] == 20901


def test_derived_ports_skip_the_claimed_ledger_and_pinned_ports() -> None:
    viewer = copy.deepcopy(VIEWER)
    viewer["containers"]["viewer"]["port"] = FACILITY_BAND + 1
    claimed = frozenset({FACILITY_BAND, FACILITY_BAND + 2})
    config = {"apps": {"knob-demo": {}, "viewer": {}}}
    services = _services(_project([_manifest(KNOB), _manifest(viewer)], config, claimed))
    assert services["viewer"]["config"]["port"] == FACILITY_BAND + 1
    assert services["knob-demo"]["config"]["port"] == FACILITY_BAND + 3
    assert services["knob-cache"]["config"]["port"] == FACILITY_BAND + 4


def test_an_exhausted_band_refuses_naming_the_container() -> None:
    claimed = frozenset(range(FACILITY_BAND, FACILITY_BAND + 100))
    with pytest.raises(AppManifestError) as excinfo:
        _project([_manifest(KNOB)], {"apps": {"knob-demo": {}}}, claimed)
    error = excinfo.value
    assert error.app == "knob-demo"
    assert error.field == "containers.knob-demo.port"
    assert error.rule == "port_band_exhausted"
    assert f"{FACILITY_BAND}-{FACILITY_BAND + 99}" in str(error)


def test_projection_is_pure() -> None:
    manifests = [_manifest(KNOB)]
    config = {"apps": {"knob-demo": {"services": {"knob-demo": {"env": ["X"]}}}}}
    before = copy.deepcopy(config)
    first = _project(manifests, config)
    assert config == before
    assert _project(manifests, config) == first
    first["services"]["knob-demo"]["config"]["env"].append("Y")
    assert _project(manifests, config)["services"]["knob-demo"]["config"]["env"] == [
        "KNOB_DEMO_MODE",
        "X",
    ]
