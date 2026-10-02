"""The Bluesky containers load the config.yml they are handed.

Every container the ``bluesky`` and ``bluesky_web`` templates run from an OSPREY
image mounts the project's ``config.yml`` at one path and names it twice:
``CONFIG_FILE`` for the readers that take that name (the default config
singleton, the limits validator, the sidecar's ``/channels`` route) and
``OSPREY_CONFIG`` for the framework's own loader
(:func:`osprey_connectors.workspace.resolve_config_path`), which never reads
``CONFIG_FILE``. Those images run from ``WORKDIR /app`` and the config is mounted
under ``/app/project``, so a container handed only ``CONFIG_FILE`` loads nothing
through the framework loader: no ``bluesky.plan_module``, no preset plan
directories, no configured child-environment names, a one-lane roster on a
two-lane deployment, and an audit zone no bind covers.

These tests render both templates through the production injection, stand the
container's filesystem up under ``tmp_path`` — the image's ``WORKDIR`` as the
working directory, the config at its mount target, the two variables exactly as
rendered — and ask the real loaders what they see.
"""

from __future__ import annotations

import re
from importlib import resources
from pathlib import Path

import pytest
import yaml
from jinja2 import Environment, FileSystemLoader

import osprey
from osprey.bluesky_bridge_connection import resolve_lane_bridge_urls
from osprey.deployment.compose_generator import _inject_project_metadata
from osprey.interfaces.bluesky_web.app import _lane_roster
from osprey.mcp_server.sandbox_env import configured_child_env_passthrough
from osprey.services.bluesky_bridge import plan_loader
from osprey.utils.workspace import (
    AUDIT_DIR_RELPATH,
    load_osprey_config,
    reset_config_cache,
    resolve_project_root,
)

_TEMPLATES_ROOT = resources.files(osprey).joinpath("templates")

#: The image variable each OSPREY-built Bluesky image is spelled under. A
#: container whose ``image:`` names neither (Redis, Tiled) runs no OSPREY code.
_OSPREY_IMAGE_VARS = ("OSPREY_BLUESKY_BRIDGE_IMAGE", "OSPREY_BLUESKY_WEB_IMAGE")

#: The service template directory whose Dockerfile builds each image.
_IMAGE_TEMPLATE_DIR = {
    "OSPREY_BLUESKY_BRIDGE_IMAGE": "bluesky",
    "OSPREY_BLUESKY_WEB_IMAGE": "bluesky_web",
}


def _deployment() -> dict:
    """A two-lane deployment with the sidecar: ``bluesky`` and ``bluesky_va``."""
    services: dict = {
        Path(str(path)).parent.name: {}
        for path in _TEMPLATES_ROOT.glob("services/*/docker-compose*.yml.j2")
    }
    services["bluesky"] = {"port": 8095}
    services["bluesky_va"] = {"port": 8096, "target": "va"}
    return {
        "project_name": "demo",
        "project_root": "/r/demo",
        "services": services,
        "system": {"timezone": "UTC"},
        "deployment": {},
        "deployed_services": ["bluesky", "bluesky_va", "bluesky_web"],
    }


def _services(template: str) -> dict:
    """The parsed ``services:`` of one template, through the production injection."""
    env = Environment(loader=FileSystemLoader(str(_TEMPLATES_ROOT)), autoescape=False)
    rendered = env.get_template(f"services/{template}/docker-compose.yml.j2").render(
        _inject_project_metadata(_deployment())
    )
    return yaml.safe_load(rendered)["services"]


def _osprey_containers() -> dict[str, dict]:
    """Every rendered container that runs an OSPREY-built Bluesky image, by service key."""
    found: dict[str, dict] = {}
    for template in ("bluesky", "bluesky_web"):
        for key, service in _services(template).items():
            if any(var in str(service.get("image", "")) for var in _OSPREY_IMAGE_VARS):
                found[key] = service
    return found


def _workdir(service: dict) -> str:
    """The ``WORKDIR`` the service's image runs from, read from its Dockerfile."""
    var = next(var for var in _OSPREY_IMAGE_VARS if var in str(service["image"]))
    dockerfile = _TEMPLATES_ROOT.joinpath(
        "services", _IMAGE_TEMPLATE_DIR[var], "Dockerfile"
    ).read_text(encoding="utf-8")
    return re.findall(r"^WORKDIR\s+(\S+)\s*$", dockerfile, flags=re.MULTILINE)[-1]


def _config_mount_target(service: dict) -> str:
    """The container path the rendered ``config.yml`` bind lands on."""
    (target,) = [
        mount.split(":")[1]
        for mount in service["volumes"]
        if isinstance(mount, str) and mount.split(":")[0].endswith("/config.yml")
    ]
    return target


def _inside(root: Path, container_path: str) -> Path:
    """Where *container_path* lives in the container filesystem stood up at *root*."""
    return root / container_path.lstrip("/")


def _stand_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, service: dict, config: dict) -> Path:
    """Lay the container out under *tmp_path* and enter it, as the image runs.

    *config* is written at the bind's target, the working directory is the
    image's ``WORKDIR``, and ``OSPREY_CONFIG`` / ``CONFIG_FILE`` carry the
    rendered values re-rooted at *tmp_path*. Returns that root.
    """
    root = tmp_path / "container"
    config_path = _inside(root, _config_mount_target(service))
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    workdir = _inside(root, _workdir(service))
    workdir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(workdir)
    environment = service["environment"]
    for name in ("OSPREY_CONFIG", "CONFIG_FILE"):
        monkeypatch.delenv(name, raising=False)
        if name in environment:
            monkeypatch.setenv(name, str(_inside(root, environment[name])))
    reset_config_cache()
    return root


def test_every_osprey_container_names_its_mounted_config_by_both_variables() -> None:
    """``OSPREY_CONFIG`` and ``CONFIG_FILE`` name the one file the bind mounts."""
    containers = _osprey_containers()

    assert {"bluesky-bridge", "queueserver", "bluesky-web"} <= set(containers)
    for key, service in containers.items():
        environment = service["environment"]
        target = _config_mount_target(service)
        assert environment.get("OSPREY_CONFIG") == target, key
        assert environment.get("CONFIG_FILE") == target, key


_PLAN_CONFIG = {
    "bluesky": {
        "plan_module": "/app/project/facility_plans.py",
        "plan_dirs": ["/app/project/extra_plans"],
        "excluded_plans": ["retired_scan"],
    },
    "python_executor": {"child_env_passthrough": ["SITE_ARCHIVE_ROOT"]},
}


@pytest.fixture
def plan_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """No plan-layer variable leaks in, and the session directory is this test's."""
    for name in ("BLUESKY_PLAN_MODULE", "BLUESKY_PLAN_DIRS", "BLUESKY_EXCLUDED_PLANS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BLUESKY_SESSION_PLAN_DIR", str(tmp_path / "session_plans"))


@pytest.mark.parametrize("key", ["bluesky-bridge", "queueserver"])
@pytest.mark.usefixtures("plan_env")
def test_the_plan_loader_reads_the_mounted_config(
    key: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every config-side plan source arrives, in both containers that load plans."""
    _stand_up(tmp_path, monkeypatch, _osprey_containers()[key], _PLAN_CONFIG)

    assert plan_loader._resolve_plan_module_path() == "/app/project/facility_plans.py"
    assert (Path("/app/project/extra_plans"), "preset") in plan_loader._resolve_plan_dir_layers()
    assert "retired_scan" in plan_loader._resolve_excluded_plans()
    assert configured_child_env_passthrough(load_osprey_config()) == ("SITE_ARCHIVE_ROOT",)


@pytest.mark.usefixtures("plan_env")
def test_config_file_alone_loads_nothing_through_the_framework_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The loader never reads ``CONFIG_FILE``: without ``OSPREY_CONFIG`` the same
    container sees an empty config. This is why the templates set both."""
    _stand_up(tmp_path, monkeypatch, _osprey_containers()["queueserver"], _PLAN_CONFIG)
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)
    reset_config_cache()

    assert load_osprey_config() == {}
    assert plan_loader._resolve_plan_module_path() is None


@pytest.mark.parametrize("key", ["queueserver", "bluesky-va-queueserver", "bluesky-web"])
def test_the_audit_writer_resolves_into_the_containers_audit_bind(
    key: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Records land in the bind, not in the container's writable layer.

    ``writer.audit_dir``'s own derivation: the function itself is redirected
    suite-wide so no test writes into a real zone, so its body is what is asked.
    """
    service = _osprey_containers()[key]
    root = _stand_up(tmp_path, monkeypatch, service, {"project_name": "demo"})
    identity = service["environment"]["OSPREY_AUDIT_IDENTITY"]
    (audit_target,) = [
        mount.split(":")[1]
        for mount in service["volumes"]
        if isinstance(mount, str) and f"/{AUDIT_DIR_RELPATH}/" in mount.split(":")[1]
    ]

    resolved = resolve_project_root(load_osprey_config()) / AUDIT_DIR_RELPATH / identity

    assert resolved == _inside(root, audit_target)


def test_the_sidecar_reports_every_lane_the_deployment_renders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lane roster comes from the mounted config's ``services.<lane>`` blocks."""
    service = _osprey_containers()["bluesky-web"]
    deployment = _deployment()
    _stand_up(
        tmp_path,
        monkeypatch,
        service,
        {
            "services": {
                "bluesky": deployment["services"]["bluesky"],
                "bluesky_va": deployment["services"]["bluesky_va"],
            }
        },
    )
    bridge_urls = {
        name: value
        for name, value in service["environment"].items()
        if name.endswith("_BRIDGE_URL")
    }
    for name, value in bridge_urls.items():
        monkeypatch.setenv(name, value)

    urls = resolve_lane_bridge_urls()

    assert urls == {
        "bluesky": bridge_urls["BLUESKY_BRIDGE_URL"],
        "bluesky_va": bridge_urls["BLUESKY_VA_BRIDGE_URL"],
    }
    assert {"lane": "bluesky_va", "lane_target": "va"} in _lane_roster(urls)
