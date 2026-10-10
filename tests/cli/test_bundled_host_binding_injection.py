"""The build step that writes OSPREY's own host-binding declarations.

OSPREY's host-capable templates each hold one fact about what they bind, kept in
one table keyed by bundled template. The build writes that fact into the
rendered ``services.<name>`` block, so the readers at ``osprey up`` see one
spelling whoever wrote it. It is written only into a host-mode block: readers
consult it only under ``network: host``, so a bridge-mode ``config.yml`` stays
byte-for-byte what it was.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml as pyyaml

from osprey.cli.build_cmd import _inject_services
from osprey.cli.build_injectors import _declare_bundled_host_bindings, _inject_dispatch
from osprey.cli.build_profile_schema import DispatchConfig, ServiceDef

_OUTBOUND_ONLY = ("nextcloud_bridge", "gchat_bridge", "teams_bridge", "ariel_sync", "archive")


def _project(tmp_path: Path, services: dict[str, Any], **extra: Any) -> Path:
    """A built project whose config.yml holds the given service blocks."""
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    document: dict[str, Any] = {"services": services, "deployed_services": list(services)}
    document.update(extra)
    (project / "config.yml").write_text(pyyaml.safe_dump(document), encoding="utf-8")
    return project


def _services(project: Path) -> dict[str, Any]:
    return pyyaml.safe_load((project / "config.yml").read_text(encoding="utf-8"))["services"]


@pytest.mark.parametrize("name", _OUTBOUND_ONLY)
def test_a_host_mode_outbound_service_is_declared_listening_on_nothing(
    tmp_path: Path, name: str
) -> None:
    project = _project(tmp_path, {name: {"path": f"./services/{name}", "network": "host"}})

    _declare_bundled_host_bindings(project, tmp_path / "profile", {})

    block = _services(project)[name]
    assert block["listens"] is False
    assert "bind_env" not in block


def test_a_host_dispatch_pair_declares_both_bind_variables(tmp_path: Path) -> None:
    project = _project(tmp_path, {})
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    dispatch = DispatchConfig(
        triggers="tutorial_triggers.yml",
        channel_strip_prefix="ERF:",
        network="host",
    )

    _inject_dispatch(dispatch, profile_dir=profile_dir, project_path=project)
    _declare_bundled_host_bindings(project, tmp_path / "profile", {})

    services = _services(project)
    assert services["event_dispatcher"]["bind_env"] == "FASTMCP_HOST"
    assert services["dispatch_worker"]["bind_env"] == "DISPATCH_WORKER_BIND"
    assert "listens" not in services["event_dispatcher"]
    assert "listens" not in services["dispatch_worker"]


def test_a_host_graphdb_declares_its_listen_variable(tmp_path: Path) -> None:
    project = _project(tmp_path, {"graphdb": {"path": "./services/graphdb", "network": "host"}})

    _declare_bundled_host_bindings(
        project, tmp_path / "profile", {"graphdb": ServiceDef(template="osprey.graphdb")}
    )

    assert _services(project)["graphdb"]["bind_env"] == "NEO4J_server_default__listen__address"


def test_qmd_is_left_undeclared(tmp_path: Path) -> None:
    """Its forwarder binds every interface and no variable narrows it."""
    project = _project(tmp_path, {"qmd": {"path": "./services/qmd", "network": "host"}})
    before = (project / "config.yml").read_bytes()

    _declare_bundled_host_bindings(
        project, tmp_path / "profile", {"qmd": ServiceDef(template="osprey.qmd")}
    )

    assert (project / "config.yml").read_bytes() == before


def test_a_bridge_mode_render_is_byte_for_byte_unchanged(tmp_path: Path) -> None:
    services = {
        name: {"path": f"./services/{name}"}
        for name in (*_OUTBOUND_ONLY, "event_dispatcher", "dispatch_worker", "graphdb")
    }
    services["archive"]["network"] = "bridge"
    project = _project(tmp_path, services)
    before = (project / "config.yml").read_bytes()

    _declare_bundled_host_bindings(
        project, tmp_path / "profile", {"archive": ServiceDef(template="osprey.archive")}
    )

    assert (project / "config.yml").read_bytes() == before


def test_a_claimed_service_is_left_to_its_author(tmp_path: Path) -> None:
    (tmp_path / "profile" / "services" / "archive").mkdir(parents=True)
    project = _project(tmp_path, {"archive": {"path": "./services/archive", "network": "host"}})

    _declare_bundled_host_bindings(
        project, tmp_path / "profile", {"archive": ServiceDef(template="osprey.archive")}
    )

    assert "listens" not in _services(project)["archive"]


def test_a_facility_template_of_a_bundled_name_is_left_to_its_author(tmp_path: Path) -> None:
    project = _project(tmp_path, {"archive": {"path": "./services/archive", "network": "host"}})

    _declare_bundled_host_bindings(
        project, tmp_path / "profile", {"archive": ServiceDef(template="services/archive")}
    )

    assert "listens" not in _services(project)["archive"]


def test_a_service_claimed_in_the_profile_keeps_only_its_authors_declaration(
    tmp_path: Path,
) -> None:
    """The claim is the profile's ``services/<name>`` directory, as validation reads it.

    The build injects services before it registers the profile's conventions
    as ``scaffold.user_owned``, so the render holds no ownership list yet.
    """
    profile_dir = tmp_path / "profile"
    claimed = profile_dir / "services" / "archive"
    claimed.mkdir(parents=True)
    (claimed / "docker-compose.yml.j2").write_text("services: {}\n", encoding="utf-8")
    project = _project(tmp_path, {})
    build_profile = SimpleNamespace(
        deploy_services=True,
        services={
            "archive": ServiceDef(
                template="osprey.archive",
                config={"network": "host", "bind_env": "ARCHIVE_BIND"},
            )
        },
        dispatch=None,
        nextcloud_bridge=None,
        gchat_bridge=None,
        teams_bridge=None,
        bluesky=None,
        bluesky_web=None,
        virtual_accelerator=None,
        va_archiver=None,
    )

    _inject_services(build_profile, profile_dir, project)

    block = _services(project)["archive"]
    assert block["bind_env"] == "ARCHIVE_BIND"
    assert "listens" not in block
