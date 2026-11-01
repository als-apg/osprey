"""The render facts and compose section of a container that drives the target.

A container declaring ``drives_target`` judges owned writes in its own process.
What makes that judgeable is what this section binds: the staged config carrying
the login map, the read-only control tree, the guarded-run root and, on an armed
render, the limits database. The facts come from the deployment's own config, so
the role is the only thing a service has to declare.
"""

from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

import osprey.templates
from osprey.bluesky_bridge_connection import LANE_KEYS
from osprey.deployment.compose_generator import (
    _inject_project_metadata,
    drives_target_write_posture,
    setup_build_dir,
)
from osprey.deployment.errors import DeploymentPreconditionError

KEY = "knob_backend"
IDENTITY = "app-knob-example"
LOGIN_TIERS = {"alice": "allowed"}
LIMITS_MOUNT = {
    "source": "./build/data/channel_limits.json",
    "target": "/app/project/data/channel_limits.json",
}
_TEMPLATES = Path(next(iter(osprey.templates.__path__)))


def _section(writes: str = "allowed", **fields: Any) -> dict[str, Any]:
    return {"type": "virtual_accelerator", "writes": writes, **fields}


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": f"services/{KEY}",
        "template": "osprey.app_runtime",
        "app": "knob-example",
        "port": 10950,
        "drives_target": True,
        "control_identity": False,
        "audit_identity": IDENTITY,
    }
    row.update(overrides)
    return row


def _config(section: dict[str, Any], *, row: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "project_name": "fixture",
        "services": {
            KEY: row if row is not None else _row(),
            "virtual_accelerator": {"path": "services/virtual_accelerator", "port": 5071},
        },
        "deployed_services": [KEY, "virtual_accelerator"],
        "deployment": {},
        "system": {"timezone": "UTC"},
        "control_system": section,
    }


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A deployment repo holding the bundled template under the container's key."""
    repo = tmp_path / "repo"
    service = repo / "services" / KEY
    service.mkdir(parents=True)
    packaged = _TEMPLATES / "services"
    shutil.copy(packaged / "app_runtime" / "docker-compose.yml.j2", service)
    for macro in ("_network_axis.j2", "_env_axis.j2"):
        shutil.copy(packaged / macro, repo / "services")
    (repo / "config.yml").write_text("project_name: fixture\n")
    monkeypatch.chdir(repo)
    return repo


def _render(repo: Path, config: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    """Render through the build's own entry, as ``osprey build`` does."""
    config = {
        **copy.deepcopy(config),
        "project_root": str(repo),
        "config_dir": str(repo),
        "build_dir": "./build",
    }
    composed = setup_build_dir(
        f"services/{KEY}/docker-compose.yml.j2",
        config,
        config["services"][KEY],
        login_tiers=LOGIN_TIERS,
        service_key=KEY,
    )
    return yaml.safe_load(Path(composed).read_text())["services"][KEY], Path(composed).parent


def _tree_mount(metadata: dict[str, Any]) -> str:
    source = metadata["osprey_control_tree_mount_source"]
    return f"{source}:{metadata['osprey_container_control_tree_dir']}:ro"


@pytest.mark.parametrize(
    ("writes", "armed"), [("allowed", True), ("asked", True), ("never", False)]
)
def test_write_posture_is_the_login_tier_on_every_service_with_the_role(
    writes: str, armed: bool
) -> None:
    services = {
        **{lane: {} for lane in LANE_KEYS},
        "dispatch_worker": {},
        "projected": {"drives_target": True},
        "plain": {"drives_target": False},
    }

    posture = drives_target_write_posture(services, {"writes": writes})

    expected = dict.fromkeys((*LANE_KEYS, "dispatch_worker", "projected"), armed)
    assert posture == expected


def test_a_lane_key_with_no_block_is_no_lane() -> None:
    posture = drives_target_write_posture({"bluesky": None}, {"writes": "allowed"})

    assert posture == {}


@pytest.mark.parametrize("writes", ["allowed", "asked"])
def test_an_armed_render_binds_everything_an_owned_write_is_judged_by(
    repo: Path, writes: str
) -> None:
    config = {**_config(_section(writes)), "limits_mount": LIMITS_MOUNT}
    config["control_system"]["connector"] = {"virtual_accelerator": {"timeout_s": 7.5}}

    svc, out_dir = _render(repo, config)

    metadata = _inject_project_metadata(config)
    project = metadata["osprey_worker_project_dir"]
    guarded = metadata["osprey_guarded_run_mount"]
    assert svc["environment"]["CONFIG_FILE"] == f"{project}/build/config.yml"
    assert svc["environment"]["OSPREY_CONFIG"] == f"{project}/build/config.yml"
    tree = metadata["osprey_container_control_tree_dir"]
    assert svc["environment"]["OSPREY_CONTROL_CONTEXT_TREE"] == tree
    assert svc["environment"]["OSPREY_GUARDED_RUN_DIR"] == guarded["target"]
    assert svc["environment"]["OSPREY_AUDIT_IDENTITY"] == IDENTITY
    assert svc["environment"]["OSPREY_SPOOL_DIR"] == (
        f"{metadata['osprey_container_spool_dir']}/{IDENTITY}"
    )
    assert svc["stop_grace_period"] == "18s"
    assert set(svc["volumes"]) == {
        f"./build/services/{KEY}/config.yml:{project}/build/config.yml:ro",
        _tree_mount(metadata),
        f"{guarded['source']}:{guarded['target']}",
        f"{LIMITS_MOUNT['source']}:{LIMITS_MOUNT['target']}:ro",
        f"{metadata['osprey_spool_mount_source']}/{IDENTITY}:"
        f"{metadata['osprey_container_spool_dir']}/{IDENTITY}",
    }
    staged = yaml.safe_load((out_dir / "config.yml").read_text())
    assert staged["control_system"]["login_tiers"] == LOGIN_TIERS


def test_a_never_render_mounts_neither_the_tree_nor_the_limits(repo: Path) -> None:
    config = {**_config(_section("never")), "limits_mount": LIMITS_MOUNT}

    svc, _ = _render(repo, config)

    metadata = _inject_project_metadata(config)
    guarded = metadata["osprey_guarded_run_mount"]
    project = metadata["osprey_worker_project_dir"]
    assert "OSPREY_CONTROL_CONTEXT_TREE" not in svc["environment"]
    assert svc["stop_grace_period"] == "15s"
    assert set(svc["volumes"]) == {
        f"./build/services/{KEY}/config.yml:{project}/build/config.yml:ro",
        f"{guarded['source']}:{guarded['target']}",
        f"{metadata['osprey_spool_mount_source']}/{IDENTITY}:"
        f"{metadata['osprey_container_spool_dir']}/{IDENTITY}",
    }


def test_an_armed_render_with_no_limits_database_binds_no_limits(repo: Path) -> None:
    svc, _ = _render(repo, _config(_section()))

    assert not [volume for volume in svc["volumes"] if "channel_limits" in volume]


def test_the_shared_input_config_is_left_alone() -> None:
    config = _config(_section())
    before = copy.deepcopy(config)

    _inject_project_metadata(config)

    assert config == before


def test_channel_access_on_the_bridge_names_the_co_deployed_simulator(repo: Path) -> None:
    svc, _ = _render(repo, _config(_section()))

    assert svc["environment"]["EPICS_CA_NAME_SERVERS"] == "virtual-accelerator:5071"
    assert svc["environment"]["EPICS_CA_AUTO_ADDR_LIST"] == "NO"


def test_channel_access_on_the_bridge_names_the_co_deployed_stand_in(repo: Path) -> None:
    config = _config({"type": "live_standin", "writes": "allowed"})
    config["services"]["live_standin"] = {"path": "services/virtual_accelerator", "port": 5072}
    config["deployed_services"].append("live_standin")

    svc, _ = _render(repo, config)

    assert svc["environment"]["EPICS_CA_NAME_SERVERS"] == "live-standin:5072"


def test_channel_access_on_the_bridge_names_a_live_gateway(repo: Path) -> None:
    gateway = {"write_access": {"address": "gw.example.org", "port": 5084}}
    config = _config(
        {"type": "epics", "writes": "allowed", "connector": {"epics": {"gateways": gateway}}}
    )

    svc, _ = _render(repo, config)

    assert svc["environment"]["EPICS_CA_NAME_SERVERS"] == "gw.example.org:5084"


def test_a_live_gateway_with_no_address_refuses_naming_the_service() -> None:
    config = _config({"type": "epics", "writes": "allowed"})

    with pytest.raises(DeploymentPreconditionError, match=KEY) as refusal:
        _inject_project_metadata(config)

    assert "gateways" in refusal.value.remedy


def test_a_simulator_that_is_not_deployed_refuses_naming_the_service() -> None:
    config = _config(_section())
    config["deployed_services"].remove("virtual_accelerator")

    with pytest.raises(DeploymentPreconditionError, match=KEY):
        _inject_project_metadata(config)


def test_a_stand_in_with_no_port_refuses_naming_the_service() -> None:
    config = _config({"type": "live_standin", "writes": "allowed"})
    config["services"]["live_standin"] = {"path": "services/virtual_accelerator"}
    config["deployed_services"].append("live_standin")

    with pytest.raises(DeploymentPreconditionError, match=KEY):
        _inject_project_metadata(config)


def test_the_host_network_needs_no_addressing(repo: Path) -> None:
    config = _config({"type": "epics", "writes": "allowed"}, row=_row(network="host"))

    svc, _ = _render(repo, config)

    assert "EPICS_CA_NAME_SERVERS" not in svc["environment"]


def test_a_simulator_in_process_needs_no_addressing(repo: Path) -> None:
    serving = {"virtual_accelerator": {"serving": "in_process"}}
    svc, _ = _render(repo, _config(_section(connector=serving)))

    assert "EPICS_CA_NAME_SERVERS" not in svc["environment"]


def test_a_service_without_the_role_renders_none_of_the_section(repo: Path) -> None:
    config = {
        **_config(_section(), row=_row(drives_target=False)),
        "limits_mount": LIMITS_MOUNT,
    }

    svc, _ = _render(repo, config)

    assert "stop_grace_period" not in svc
    assert not {
        "CONFIG_FILE",
        "OSPREY_CONFIG",
        "OSPREY_CONTROL_CONTEXT_TREE",
        "OSPREY_GUARDED_RUN_DIR",
        "EPICS_CA_NAME_SERVERS",
    } & set(svc["environment"])
    assert not [volume for volume in svc["volumes"] if IDENTITY not in volume]


def test_a_bundled_service_with_the_role_gets_only_the_posture() -> None:
    config = _config({"type": "epics", "writes": "allowed"}, row=_row(drives_target=False))
    config["services"]["dispatch_worker"] = {"path": "services/dispatch_worker"}
    config["deployed_services"].append("dispatch_worker")

    worker = _inject_project_metadata(config)["services"]["dispatch_worker"]

    assert worker["writes_enabled"] is True
    assert "stop_grace_period" not in worker and "ca_env" not in worker
