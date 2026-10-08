"""Every container that runs the agent shares one host ``var/guarded_run``.

A guarded run's lock and journal have to be one per control target for the
whole deployment, so the deploy provisions ``var/guarded_run/<target>/`` on the
host before compose runs and binds ``var/guarded_run`` read-write under the
container repo root of every container that runs the agent.
"""

from __future__ import annotations

import copy
import stat
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader

from osprey.deployment.compose_generator import (
    _ensure_agent_data_structure,
    _inject_project_metadata,
    ensure_guarded_run_dirs,
    guarded_run_relpath,
)
from osprey.deployment.web_terminals.render import render_web_terminals
from osprey.utils.workspace import AUDIT_DIR_RELPATH
from tests.templates.test_render_defaults_golden import (
    TEMPLATES,
    _default_context,
    _probe_repo,
    _render_templates,
)

from .web_terminals.test_golden_render import EXAMPLE_CONFIG

_TEMPLATES_ROOT = Path(__file__).resolve().parents[2] / "src" / "osprey" / "templates"

SOURCE = "./var/guarded_run"

#: A deployment on a real control system with the Virtual Accelerator beside it.
LIVE_AND_VA = {
    "type": "epics",
    "connector": {"epics": {"gateways": {}}, "virtual_accelerator": {"port": 5064}},
}


def _assert_shared(path: Path) -> None:
    mode = path.stat().st_mode
    assert path.is_dir()
    assert mode & stat.S_ISGID
    assert mode & stat.S_IWGRP


def test_each_configured_target_is_provisioned(tmp_path: Path) -> None:
    gid = ensure_guarded_run_dirs(tmp_path, {"control_system": LIVE_AND_VA})

    root = tmp_path / "var" / "guarded_run"
    _assert_shared(root)
    assert sorted(path.name for path in root.iterdir()) == ["live", "va"]
    for target in root.iterdir():
        _assert_shared(target)
    assert gid == root.stat().st_gid


def test_a_deployment_without_a_control_system_gets_its_baseline(tmp_path: Path) -> None:
    ensure_guarded_run_dirs(tmp_path, {})

    assert [path.name for path in (tmp_path / "var" / "guarded_run").iterdir()] == ["live"]


def test_the_build_path_provisions_it(tmp_path: Path) -> None:
    config: dict[str, Any] = {"project_root": str(tmp_path), "control_system": {"type": "mock"}}

    _ensure_agent_data_structure(config)

    _assert_shared(tmp_path / "var" / "guarded_run" / "live")


def _guarded_run_mounts(service: dict[str, Any]) -> list[str]:
    return [
        volume
        for volume in service.get("volumes", [])
        if isinstance(volume, str) and volume.split(":", 1)[0] == SOURCE
    ]


def test_every_dispatch_worker_mounts_it(tmp_path: Path) -> None:
    context = _inject_project_metadata(
        {
            "project_name": "proj",
            "project_root": str(tmp_path),
            "deployment": {},
            "system": {"timezone": "UTC"},
            "services": {"dispatch_worker": {"worker_count": 2}},
            "deployed_services": [],
            "control_system": {"type": "epics", "connector": {"epics": {"gateways": {}}}},
        }
    )
    environment = Environment(loader=FileSystemLoader(str(_TEMPLATES_ROOT)), autoescape=False)
    rendered = environment.get_template("services/dispatch_worker/docker-compose.yml.j2")
    services = yaml.safe_load(rendered.render(**context))["services"]

    workers = [name for name in services if name.startswith("dispatch-worker-")]
    assert len(workers) == 2
    for name in workers:
        assert _guarded_run_mounts(services[name]) == [
            "./var/guarded_run:/app/proj/var/guarded_run"
        ]


def _environment(service: dict[str, Any]) -> dict[str, str]:
    environment = service.get("environment") or {}
    if isinstance(environment, dict):
        return {str(key): str(value) for key, value in environment.items()}
    return dict(str(entry).split("=", 1) for entry in environment)


def _rendered_services() -> dict[str, dict[str, Any]]:
    """Every service the bundled templates render, keyed ``<file>:<service>``.

    Each bundled service template at its defaults, plus the web-terminal
    overlay for a deployment with two users, so a template added later is
    enumerated here without anyone listing it.
    """
    with _probe_repo() as repo_root:
        renders = _render_templates(_default_context(repo_root), TEMPLATES)
    renders["docker-compose.web.yml"] = render_web_terminals(copy.deepcopy(EXAMPLE_CONFIG))[
        "docker-compose.web.yml"
    ]
    services: dict[str, dict[str, Any]] = {}
    for file_name, text in sorted(renders.items()):
        document = yaml.safe_load(text) or {}
        for name, service in sorted((document.get("services") or {}).items()):
            services[f"{file_name}:{name}"] = service
    return services


def _container_repo_root(service: dict[str, Any]) -> PurePosixPath | None:
    """The container repo root of a service that runs the python MCP server.

    The framework MCP servers record every tool call through the audit
    middleware, so a service that launches them is told where its records go
    (``OSPREY_AUDIT_DIR``, ``<container repo root>/var/audit/<identity>``). A
    service that names no such directory runs no framework MCP server.
    """
    audit_dir = _environment(service).get("OSPREY_AUDIT_DIR")
    if not audit_dir:
        return None
    path = PurePosixPath(audit_dir)
    depth = len(PurePosixPath(AUDIT_DIR_RELPATH).parts) + 1
    return PurePosixPath(*path.parts[:-depth])


def test_every_service_running_the_python_mcp_server_mounts_one_host_dir() -> None:
    services = _rendered_services()
    carriers = {
        key: root
        for key, service in services.items()
        if (root := _container_repo_root(service)) is not None
    }

    assert len({key.split(":", 1)[0] for key in carriers}) >= 2, carriers
    sources = set()
    for key, root in carriers.items():
        mounts = _guarded_run_mounts(services[key])
        assert mounts == [f"{SOURCE}:{root / guarded_run_relpath()}"], key
        assert _environment(services[key]).get("OSPREY_GUARDED_RUN_DIR") == str(
            root / guarded_run_relpath()
        ), key
        sources.add(mounts[0].split(":", 1)[0])
    assert sources == {SOURCE}
