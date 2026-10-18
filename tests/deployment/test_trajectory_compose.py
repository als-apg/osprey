"""The trajectory service's compose fragment, rendered through the deployment's context.

The fragment runs the project image beside the dispatch worker and takes every
path from the derived context keys, so each directory or module variable it
hands the service names exactly the container side of the bind it mounts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from jinja2 import Environment, FileSystemLoader

from osprey.agent_runner.build_artifacts.catalog import BuildArtifactCatalog
from osprey.deployment.compose_generator import (
    _CONTROL_IDENTITY_SERVICES,
    CONTROL_IDENTITY_FILENAME,
    _inject_project_metadata,
    _stage_control_identity_module,
)
from osprey.deployment.errors import DeploymentPreconditionError
from osprey.deployment.host_binding import BUNDLED_HOST_BINDINGS

_TEMPLATES_ROOT = Path(__file__).resolve().parents[2] / "src" / "osprey" / "templates"
_TEMPLATE = "services/trajectory/docker-compose.yml.j2"

#: A deployment on a real EPICS machine: its baseline target speaks Channel Access.
EPICS = {"type": "epics", "writes_enabled": True, "connector": {"epics": {"gateways": {}}}}

#: The simulator served in process: a target that dials nothing.
IN_PROCESS_VA = {
    "type": "virtual_accelerator",
    "connector": {"virtual_accelerator": {"serving": "in_process"}},
}

LIMITS_MOUNT = {"source": "./build/data/limits.json", "target": "/app/project/data/limits.json"}


def _config(block: dict[str, Any], control_system: Any = None, **extra: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "project_name": "proj",
        "project_root": "/deploy/proj",
        "deployment": {},
        "system": {"timezone": "UTC"},
        "services": {"trajectory": block},
        "deployed_services": ["trajectory"],
    }
    if control_system is not None:
        config["control_system"] = control_system
    config.update(extra)
    return config


def _render(config: dict[str, Any]) -> dict[str, Any]:
    context = _inject_project_metadata(config)
    environment = Environment(loader=FileSystemLoader(str(_TEMPLATES_ROOT)), autoescape=False)
    return yaml.safe_load(environment.get_template(_TEMPLATE).render(**context))


def _service(config: dict[str, Any]) -> dict[str, Any]:
    return _render(config)["services"]["trajectory"]


def _binds(service: dict[str, Any]) -> dict[str, str]:
    """Container target to the whole bind line, for every string volume."""
    binds = {}
    for volume in service["volumes"]:
        parts = volume.split(":")
        binds[parts[1]] = volume
    return binds


def test_host_network_publishes_no_ports_and_binds_loopback() -> None:
    document = _render(_config({"network": "host", "port": 9182}, EPICS))
    service = document["services"]["trajectory"]

    assert service["network_mode"] == "host"
    assert "ports" not in service
    assert "networks" not in service
    assert "networks" not in document
    assert service["environment"]["TRAJECTORY_BIND"] == "127.0.0.1"


def test_bridge_network_publishes_the_port_and_binds_every_interface() -> None:
    document = _render(_config({"network": "bridge", "port": 9182}, IN_PROCESS_VA))
    service = document["services"]["trajectory"]

    assert service["networks"] == ["osprey-network"]
    assert service["ports"] == ["127.0.0.1:9182:9182"]
    assert service["environment"]["TRAJECTORY_BIND"] == "0.0.0.0"
    assert "osprey-network" in document["networks"]


def test_a_declared_bind_wins() -> None:
    service = _service(_config({"network": "host", "bind": "10.0.0.5"}, EPICS))

    assert service["environment"]["TRAJECTORY_BIND"] == "10.0.0.5"


def test_the_bind_variable_is_the_host_binding_row() -> None:
    row = BUNDLED_HOST_BINDINGS["trajectory"]
    service = _service(_config({"network": "host"}, EPICS))

    assert row.binding.bind_env == "TRAJECTORY_BIND"
    assert row.binding.bind_env in service["environment"]


def test_a_bridge_render_refuses_a_channel_access_target() -> None:
    with pytest.raises(DeploymentPreconditionError, match="Channel Access") as refused:
        _render(_config({"network": "bridge"}, EPICS))

    assert "network: host" in refused.value.remedy


def test_an_unstated_network_is_the_bridge_and_refuses_too() -> None:
    with pytest.raises(DeploymentPreconditionError, match="Channel Access"):
        _render(_config({"target": "live"}, EPICS))


def test_the_bridge_admits_a_target_that_speaks_no_channel_access() -> None:
    service = _service(_config({"network": "bridge", "target": "va"}, IN_PROCESS_VA))

    assert service["networks"] == ["osprey-network"]


def test_the_command_starts_the_service_module() -> None:
    service = _service(_config({"network": "host"}, EPICS))

    assert service["command"] == ["python", "-m", "osprey.services.trajectory"]
    assert service["container_name"] == "proj-trajectory"
    assert "build" not in service


def test_the_image_is_the_project_image_unless_named() -> None:
    default = _service(_config({"network": "host"}, EPICS))
    named = _service(_config({"network": "host", "image": "reg/proj:9"}, EPICS))

    assert default["image"] == "${OSPREY_WORKER_IMAGE:-proj:local}"
    assert named["image"] == "${OSPREY_WORKER_IMAGE:-reg/proj:9}"


def test_no_terminal_user_is_rendered() -> None:
    service = _service(_config({"network": "host"}, EPICS))

    assert "OSPREY_TERMINAL_USER" not in service["environment"]


def test_identities_and_credentials() -> None:
    environment = _service(_config({"network": "host"}, EPICS))["environment"]

    assert environment["OSPREY_AUDIT_IDENTITY"] == "trajectory"
    assert environment["OSPREY_CONTROL_IDENTITY"] == "osprey-trajectory"
    assert environment["TRAJECTORY_LAUNCH_TOKEN"] == "${TRAJECTORY_LAUNCH_TOKEN}"
    assert environment["OSPREY_TERMINAL_SECRET"] == "${OSPREY_TERMINAL_SECRET}"
    assert environment["OSPREY_TERMINAL_BIND_HOST"] == "0.0.0.0"
    assert environment["CONFIG_FILE"] == "/app/proj/build/config.yml"


@pytest.mark.parametrize(
    ("variable", "mode"),
    [
        ("OSPREY_AUDIT_DIR", "rw"),
        ("OSPREY_CONTROL_CONTEXT_TREE", "ro"),
        ("OSPREY_GUARDED_RUN_DIR", "rw"),
        ("OSPREY_CONTROL_IDENTITY_MODULE", "ro"),
        ("OSPREY_TRAJECTORY_STATE_DIR", "rw"),
    ],
)
def test_each_directory_and_module_variable_is_its_bind_target(variable: str, mode: str) -> None:
    service = _service(_config({"network": "host"}, EPICS))
    target = service["environment"][variable]
    binds = _binds(service)

    assert target in binds, f"{variable}={target} names no bind target"
    assert binds[target].endswith(":ro") == (mode == "ro")


def test_the_binds_name_the_service_directories() -> None:
    binds = _binds(_service(_config({"network": "host"}, EPICS)))

    assert binds["/app/proj/var/audit/trajectory"] == (
        "./var/audit/trajectory:/app/proj/var/audit/trajectory"
    )
    assert binds["/app/proj/var/guarded_run"] == "./var/guarded_run:/app/proj/var/guarded_run"
    assert binds["/app/proj/var/trajectory"] == "./var/trajectory:/app/proj/var/trajectory"
    assert binds["/app/proj/build/config.yml"] == (
        "./build/services/trajectory/config.yml:/app/proj/build/config.yml:ro"
    )
    assert binds["/opt/osprey/control_identity.py"] == (
        "./build/services/trajectory/control_identity.py:/opt/osprey/control_identity.py:ro"
    )


def _limits_config(control_system: dict[str, Any], database_path: str) -> dict[str, Any]:
    control_system = {**control_system, "limits_checking": {"database_path": database_path}}
    return _config({"network": "host"}, control_system, limits_mount=LIMITS_MOUNT)


def test_an_armed_target_mounts_the_limits_database_beside_the_config() -> None:
    binds = _binds(_service(_limits_config(EPICS, "data/limits.json")))

    assert binds["/app/proj/build/data/limits.json"] == (
        "./build/data/limits.json:/app/proj/build/data/limits.json:ro"
    )


def test_an_absolute_limits_path_is_mounted_at_its_own_path() -> None:
    mount = {"source": "/srv/limits.json", "target": "/srv/limits.json"}
    control_system = {**EPICS, "limits_checking": {"database_path": "/srv/limits.json"}}
    binds = _binds(_service(_config({"network": "host"}, control_system, limits_mount=mount)))

    assert binds["/srv/limits.json"] == "/srv/limits.json:/srv/limits.json:ro"


def test_a_disarmed_target_mounts_no_limits_database() -> None:
    disarmed = {**EPICS, "writes_enabled": False}
    service = _service(_limits_config(disarmed, "data/limits.json"))

    assert not any("limits.json" in volume for volume in service["volumes"])


def test_no_limits_mount_renders_no_limits_bind() -> None:
    service = _service(_config({"network": "host"}, EPICS))

    assert not any("limits" in volume for volume in service["volumes"])


def test_the_render_does_not_mutate_the_input_config() -> None:
    config = _config({"network": "host"}, EPICS)

    _inject_project_metadata(config)

    assert config["services"]["trajectory"] == {"network": "host"}


def test_the_control_identity_module_is_staged_for_the_service(tmp_path: Path) -> None:
    out_dir = tmp_path / "build" / "services" / "trajectory"
    out_dir.mkdir(parents=True)

    assert "trajectory" in _CONTROL_IDENTITY_SERVICES
    assert _stage_control_identity_module({}, "services/trajectory", str(out_dir))
    assert (out_dir / CONTROL_IDENTITY_FILENAME).is_file()


def test_the_artifact_catalog_carries_the_template() -> None:
    artifact = BuildArtifactCatalog.default().get("services/trajectory")

    assert artifact is not None
    assert artifact.template_path == "trajectory"
    assert artifact.output_path == "services/trajectory"
    assert artifact.template_root == "services"
    assert artifact.is_directory


def test_the_stop_grace_period_covers_the_default_call_timeout() -> None:
    service = _service(_config({"network": "host"}, EPICS))

    assert service["stop_grace_period"] == "41s"


def test_the_stop_grace_period_grows_with_a_raised_connector_timeout() -> None:
    raised = {**EPICS, "connector": {"epics": {"gateways": {}, "timeout_s": 30}}}
    service = _service(_config({"network": "host"}, raised))

    assert service["stop_grace_period"] == "116s"


def test_the_stop_grace_period_reads_the_connector_its_target_selects() -> None:
    # The block's own target picks the connector whose timeout counts: the
    # simulator's here, though the deployment baseline is the EPICS machine.
    control_system = {
        **EPICS,
        "connector": {
            "epics": {"gateways": {}, "timeout_s": 30},
            "virtual_accelerator": {"serving": "in_process", "timeout_s": 5},
        },
    }
    service = _service(_config({"network": "host", "target": "va"}, control_system))

    assert service["stop_grace_period"] == "41s"
