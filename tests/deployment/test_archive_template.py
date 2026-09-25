"""Render tests for the bundled ``archive`` compose template.

The archive copies the deployment's agent record into ``var/archive``. What is
load-bearing about its shape:

**It runs the watch loop as container root, with the image entrypoint
bypassed.** Container root is the invoking user under rootless podman and the
only identity that can read every source volume; the image entrypoint would drop
to the unprivileged user first.

**Every source is read-only; the archive root is the one writable mount.**

**It reads other services' volumes by their owners' own declarations.** A
worker or lane volume is declared by its owner in the same compose invocation
and never again here; a terminal volume, whose owner is rendered into the other
slice, is re-declared with exactly the web-terminal template's key and labels.
A volume no owner renders is never named.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.deployment.compose_generator import (
    _inject_project_metadata,
    _resolve_archive_render_context,
)
from osprey.deployment.web_terminals.render import render_web_terminals
from tests.deployment.test_compose_generator import (
    _packaged_compose_template,
    _render_service_template,
)
from tests.deployment.web_terminals.test_golden_render import EXAMPLE_CONFIG

TEMPLATE = "archive/docker-compose.yml.j2"
SERVICE_KEY = "archive"
REPO_ID = "0123456789ab"


def _config(
    *,
    users: tuple[str, ...] = (),
    deployed: tuple[str, ...] = ("archive",),
    services: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "services": {"archive": {}, **(services or {})},
        "deployed_services": list(deployed),
    }
    if users:
        config["modules"] = {"web_terminals": {"enabled": True, "users": list(users)}}
    return config


def _render(config: dict[str, Any] | None = None, *, project_name: str = "proj-a") -> str:
    config = config if config is not None else _config()
    return _render_service_template(
        TEMPLATE,
        project_name,
        services={**config["services"]},
        deployed_services=config["deployed_services"],
        osprey_labels={
            "project_name": project_name,
            "project_root": f"/r/{project_name}",
            "repo_id": REPO_ID,
        },
        osprey_archive=_resolve_archive_render_context(config),
    )


def _doc(rendered: str) -> dict[str, Any]:
    return yaml.safe_load(rendered)


def _service(rendered: str) -> dict[str, Any]:
    doc = _doc(rendered)
    assert list(doc["services"]) == [SERVICE_KEY], doc["services"]
    return doc["services"][SERVICE_KEY]


def _mounts(rendered: str) -> list[str]:
    return list(_service(rendered)["volumes"])


# ── The daemon shape ────────────────────────────────────────────────────────


def test_the_service_runs_the_watch_loop_as_root_with_the_entrypoint_bypassed() -> None:
    service = _service(_render())

    assert service["entrypoint"] == ["osprey", "archive", "--watch"]
    assert "command" not in service
    assert service["user"] == "0:0"
    assert service["restart"] == "unless-stopped"
    assert service["security_opt"] == ["no-new-privileges:true"]
    assert service["cap_drop"] == ["ALL"]
    assert service["cap_add"] == ["CHOWN", "DAC_OVERRIDE", "FOWNER"]


def test_the_container_name_is_per_project() -> None:
    assert _service(_render(project_name="proj-b"))["container_name"] == "proj-b-archive"


def test_no_ports_and_no_healthcheck() -> None:
    service = _service(_render())

    assert "ports" not in service
    assert "healthcheck" not in service


# ── Mounts ──────────────────────────────────────────────────────────────────


def test_dest_and_audit_are_mounted_and_every_source_is_read_only() -> None:
    config = _config(
        users=("alice",),
        deployed=("archive", "dispatch_worker", "bluesky"),
        services={"dispatch_worker": {"worker_count": 1}},
    )
    mounts = _mounts(_render(config))

    assert mounts[0] == "./var/archive:/archive/dest"
    assert mounts[1] == "./var/audit:/archive/sources/audit:ro"
    for mount in mounts[1:]:
        assert mount.endswith(":ro"), mount
        assert ":/archive/sources/" in mount, mount


def test_one_mount_per_roster_user_worker_and_lane() -> None:
    config = _config(
        users=("alice", "bob"),
        deployed=("archive", "dispatch_worker", "bluesky", "bluesky_va"),
        services={"dispatch_worker": {"worker_count": 2}},
    )

    assert _mounts(_render(config))[2:] == [
        "alice-claude-config:/archive/sources/terminals/alice:ro",
        "alice-agent-data:/archive/sources/terminal_agent_data/alice:ro",
        "bob-claude-config:/archive/sources/terminals/bob:ro",
        "bob-agent-data:/archive/sources/terminal_agent_data/bob:ro",
        "dispatch_workspace_1:/archive/sources/dispatch/dispatch-worker-1:ro",
        "dispatch_workspace_2:/archive/sources/dispatch/dispatch-worker-2:ro",
        "bluesky_queueserver_redis:/archive/sources/bluesky/bluesky:ro",
        "bluesky_va_queueserver_redis:/archive/sources/bluesky/bluesky_va:ro",
    ]


def test_shared_workspace_mode_mounts_one_worker_volume() -> None:
    config = _config(
        deployed=("archive", "dispatch_worker"),
        services={"dispatch_worker": {"worker_count": 3, "workspace_mode": "shared"}},
    )

    assert _mounts(_render(config))[2:] == [
        "dispatch_workspace:/archive/sources/dispatch/shared:ro"
    ]
    assert list(_service(_render(config))["depends_on"]) == ["dispatch-worker-1"]


def test_an_external_lane_is_not_mounted() -> None:
    config = _config(
        deployed=("archive", "bluesky", "bluesky_va"),
        services={"bluesky_va": {"external": True}},
    )

    assert _mounts(_render(config))[2:] == [
        "bluesky_queueserver_redis:/archive/sources/bluesky/bluesky:ro"
    ]


def test_depends_on_names_only_services_this_deployment_runs() -> None:
    config = _config(
        users=("alice",),
        deployed=("archive", "dispatch_worker", "bluesky"),
        services={"dispatch_worker": {"worker_count": 2}},
    )
    depends_on = _service(_render(config))["depends_on"]

    assert depends_on == {
        "dispatch-worker-1": {"condition": "service_started"},
        "dispatch-worker-2": {"condition": "service_started"},
        "bluesky-redis": {"condition": "service_started"},
    }
    assert "depends_on" not in _service(_render(_config(users=("alice",))))


def test_no_file_level_volumes_when_nothing_is_mounted() -> None:
    doc = _doc(_render())

    assert "volumes" not in doc


def test_a_worker_count_below_one_names_no_worker_volume() -> None:
    config = _config(
        deployed=("archive", "dispatch_worker"),
        services={"dispatch_worker": {"worker_count": 0}},
    )

    assert _mounts(_render(config))[2:] == []


def test_the_roster_is_mounted_only_when_the_archive_is_deployed() -> None:
    assert (
        _resolve_archive_render_context(_config(users=("alice",), deployed=()))["terminals"] == []
    )


# ── The telemetry store ─────────────────────────────────────────────────────


def test_openobserve_address_follows_the_network_axis() -> None:
    bridge = _config(deployed=("archive", "openobserve"))
    on_host = _config(
        deployed=("archive", "openobserve"),
        services={"archive": {"network": "host"}, "openobserve": {"port": 15080}},
    )

    bridge_env = _service(_render(bridge))["environment"]
    host_env = _service(_render(on_host))["environment"]

    assert bridge_env["OSPREY_ARCHIVE_OPENOBSERVE_URL"] == "http://openobserve:5080"
    assert host_env["OSPREY_ARCHIVE_OPENOBSERVE_URL"] == "http://localhost:15080"
    assert bridge_env["OSPREY_ARCHIVE_OPENOBSERVE_ORG"] == "default"
    assert bridge_env["OSPREY_ARCHIVE_OPENOBSERVE_BACKFILL_DAYS"] == "14"


def test_no_openobserve_variables_without_the_store() -> None:
    env = _service(_render())["environment"]

    assert not [key for key in env if key.startswith("OSPREY_ARCHIVE_OPENOBSERVE")]
    assert env["OSPREY_ARCHIVE_SOURCES"] == "/archive/sources"
    assert env["OSPREY_ARCHIVE_DEST"] == "/archive/dest"
    assert env["OSPREY_ARCHIVE_INTERVAL_SECONDS"] == "86400"


def test_the_interval_follows_its_key() -> None:
    config = _config(services={"archive": {"interval_seconds": 3600}})

    assert _service(_render(config))["environment"]["OSPREY_ARCHIVE_INTERVAL_SECONDS"] == "3600"


# ── Owner declarations ──────────────────────────────────────────────────────


def _render_with(context: dict[str, Any], rel_path: str) -> dict[str, Any]:
    return yaml.safe_load(_packaged_compose_template(f"services/{rel_path}").render(**context))


def _owner_context(tmp_path: Path, *, mode: str = "isolated") -> tuple[dict, dict]:
    config = copy.deepcopy(EXAMPLE_CONFIG)
    config.update(
        {
            "project_name": "dls",
            "project_root": str(tmp_path),
            "system": {"timezone": "UTC"},
            "deployment": {},
            "services": {
                "archive": {},
                "dispatch_worker": {"worker_count": 2, "workspace_mode": mode},
                "bluesky": {},
            },
            "deployed_services": ["archive", "dispatch_worker", "bluesky"],
            "control_system": {},
        }
    )
    return config, _inject_project_metadata(config)


def _source_volume(mount: str) -> str:
    return mount.split(":", 1)[0]


def test_declared_volumes_match_their_owner_templates_byte_for_byte(tmp_path: Path) -> None:
    """The terminals' volumes live in the other slice, so the archive declares them
    exactly as the web-terminal template does."""
    config, context = _owner_context(tmp_path)

    archive = _render_with(context, TEMPLATE)["volumes"]
    web = yaml.safe_load(render_web_terminals(config)["docker-compose.web.yml"])["volumes"]

    assert set(archive) == {
        "alice-claude-config",
        "alice-agent-data",
        "bob-claude-config",
        "bob-agent-data",
    }
    for key, declaration in archive.items():
        assert declaration == web[key], key


@pytest.mark.parametrize("mode", ["isolated", "shared"])
def test_worker_and_lane_volumes_are_their_owners_declarations_alone(
    tmp_path: Path, mode: str
) -> None:
    """A worker or lane volume the archive mounts is declared once, by its
    owner, in this slice; the archive only references it."""
    _config_unused, context = _owner_context(tmp_path, mode=mode)

    archive = _render_with(context, TEMPLATE)
    owners: set[str] = set()
    owners |= set(_render_with(context, "dispatch_worker/docker-compose.yml.j2")["volumes"])
    owners |= set(_render_with(context, "bluesky/docker-compose.yml.j2")["volumes"])
    mounted = {
        _source_volume(mount)
        for mount in archive["services"][SERVICE_KEY]["volumes"]
        if "/archive/sources/dispatch/" in mount or "/archive/sources/bluesky/" in mount
    }

    assert mounted
    assert mounted <= owners
    assert not set(archive["volumes"]) & owners
