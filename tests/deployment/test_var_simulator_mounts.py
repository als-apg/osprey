"""Every rendered container that runs the composite mounts ``var/simulator``.

The composite appends its model logs under ``var/simulator`` of the repo root
of the config the process loaded. The Virtual Accelerator instances run it in
their own container; the web terminals and the dispatch worker run it in
process whenever a session can be pointed at a simulated target. Each of them
binds the host's ``./var/simulator`` read-write, so every record lands in the
one directory ``osprey sim status`` names; a deployment with no simulated
target binds it nowhere.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml
from jinja2 import Environment, FileSystemLoader

from osprey.deployment.compose_generator import _inject_project_metadata
from osprey.deployment.web_terminals.render import render_web_terminals

from .web_terminals.test_golden_render import EXAMPLE_CONFIG

_TEMPLATES_ROOT = Path(__file__).resolve().parents[2] / "src" / "osprey" / "templates"

SOURCE = "./var/simulator"

#: A deployment whose only machine is a real control system.
LIVE_ONLY = {"type": "epics", "connector": {"epics": {"gateways": {}}}}

#: The same deployment with the Virtual Accelerator configured beside it.
LIVE_AND_VA = {
    "type": "epics",
    "connector": {"epics": {"gateways": {}}, "virtual_accelerator": {"port": 5064}},
}


def _render_service(rel_path: str, tmp_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    context = _inject_project_metadata(
        {
            "project_name": "proj",
            "project_root": str(tmp_path),
            "deployment": {},
            "system": {"timezone": "UTC"},
            **config,
        }
    )
    environment = Environment(loader=FileSystemLoader(str(_TEMPLATES_ROOT)), autoescape=False)
    return yaml.safe_load(environment.get_template(rel_path).render(**context))["services"]


def _simulator_mounts(service: dict[str, Any]) -> list[str]:
    return [
        volume
        for volume in service.get("volumes", [])
        if isinstance(volume, str) and volume.startswith(f"{SOURCE}")
    ]


def test_both_va_instances_mount_their_log_directory(tmp_path: Path) -> None:
    services = _render_service(
        "services/virtual_accelerator/docker-compose.yml.j2",
        tmp_path,
        {
            "control_system": LIVE_AND_VA,
            "deployed_services": ["virtual_accelerator", "live_standin"],
            "services": {
                "virtual_accelerator": {"port": 5064},
                "live_standin": {"port": 5074},
            },
        },
    )

    assert _simulator_mounts(services["virtual-accelerator"]) == ["./var/simulator:/var/simulator"]
    assert _simulator_mounts(services["live-standin"]) == ["./var/simulator/standin:/var/simulator"]


def _worker(tmp_path: Path, control_system: dict[str, Any] | None) -> dict[str, Any]:
    config: dict[str, Any] = {"services": {"dispatch_worker": {}}, "deployed_services": []}
    if control_system is not None:
        config["control_system"] = control_system
    services = _render_service("services/dispatch_worker/docker-compose.yml.j2", tmp_path, config)
    return services["dispatch-worker-1"]


@pytest.mark.parametrize(
    "control_system", [None, LIVE_AND_VA], ids=["in-process", "va-beside-live"]
)
def test_the_dispatch_worker_mounts_it_with_a_simulated_target(
    tmp_path: Path, control_system: dict[str, Any] | None
) -> None:
    worker = _worker(tmp_path, control_system)

    assert _simulator_mounts(worker) == ["./var/simulator:/app/proj/var/simulator"]
    assert worker["environment"]["OSPREY_SIMULATOR_LOG_DIR"] == "/app/proj/var/simulator"


def test_the_dispatch_worker_mounts_nothing_on_a_live_only_deployment(tmp_path: Path) -> None:
    worker = _worker(tmp_path, LIVE_ONLY)

    assert _simulator_mounts(worker) == []
    assert "OSPREY_SIMULATOR_LOG_DIR" not in worker["environment"]


def _terminals(control_system: dict[str, Any]) -> dict[str, dict[str, Any]]:
    config = copy.deepcopy(EXAMPLE_CONFIG)
    config["control_system"] = control_system
    compose = yaml.safe_load(render_web_terminals(config)["docker-compose.web.yml"])
    return {name: svc for name, svc in compose["services"].items() if name.startswith("web-")}


def _env(service: dict[str, Any]) -> dict[str, str]:
    return dict(entry.split("=", 1) for entry in service["environment"])


@pytest.mark.parametrize(
    "control_system",
    [
        {
            "type": "virtual_accelerator",
            "connector": {"virtual_accelerator": {"serving": "in_process"}},
        },
        LIVE_AND_VA,
    ],
    ids=["in-process", "va-beside-live"],
)
def test_every_web_terminal_mounts_it_with_a_simulated_target(
    control_system: dict[str, Any],
) -> None:
    terminals = _terminals(control_system)

    assert terminals
    for service in terminals.values():
        assert _simulator_mounts(service) == [
            "./var/simulator:/app/dls_controls-assistant/var/simulator"
        ]
        assert (
            _env(service)["OSPREY_SIMULATOR_LOG_DIR"] == "/app/dls_controls-assistant/var/simulator"
        )


def test_no_web_terminal_mounts_it_on_a_live_only_deployment() -> None:
    for service in _terminals(LIVE_ONLY).values():
        assert _simulator_mounts(service) == []
        assert "OSPREY_SIMULATOR_LOG_DIR" not in _env(service)
