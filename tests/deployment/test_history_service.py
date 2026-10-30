"""The history service: rendered, provisioned and linked exactly when the page is deployed.

The page is derived, never a ``deployed_services`` entry: it exists when the
deployment's web terminals sign people in
(:func:`~osprey.agent_record.config.history_page_deployed`). Every
consumer keys on that one predicate — the audit directory, the project-image build, the compose render, and the dispatch
dashboard's link — so a password deployment gets all of them and an auth-none
one gets none.
"""

from __future__ import annotations

import copy
import stat
from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.agent_record.config import HISTORY_SERVICE
from osprey.deployment import container_lifecycle
from osprey.deployment.compose_generator import (
    HISTORY_CONTAINER_PORT,
    _ensure_agent_data_structure,
    _inject_project_metadata,
    service_audit_identities,
)
from osprey.port_layout import default_port, resolve_port_base
from osprey.utils.workspace import SPOOL_DIR_RELPATH, STATE_DIR_RELPATH
from tests._renders import tier_render

HISTORY_ORIGIN = "http://dls-deploy.dls.example.org:10002"


def _config(method: str = "password", **root: Any) -> dict[str, Any]:
    """A deployment with signed-in web terminals."""
    return {
        "project_name": "dls_controls",
        "project_root": "/r/dls_controls",
        "system": {"timezone": "UTC"},
        "deploy": {"host": "dls-deploy", "fqdn": "dls-deploy.dls.example.org"},
        "deployed_services": [],
        "modules": {
            "web_terminals": {
                "enabled": True,
                "auth": {"method": method, "allow_insecure_http": True},
                "users": [{"name": "alice", "index": 0}],
                "default_tier": "assistant",
                "renders": {"assistant": tier_render("assistant", "dls_controls-assistant")},
            }
        },
        **root,
    }


@pytest.mark.parametrize("method", ["password", "oidc"])
def test_a_signed_in_deployment_has_the_history_identity(method: str) -> None:
    assert HISTORY_SERVICE in service_audit_identities(_config(method))


@pytest.mark.parametrize(
    "config",
    [
        _config("none"),
        {**_config(), "modules": {"web_terminals": {"enabled": False}}},
    ],
    ids=["auth-none", "no-web-terminals"],
)
def test_no_history_identity_without_the_page(config: dict[str, Any]) -> None:
    assert HISTORY_SERVICE not in service_audit_identities(config)


@pytest.mark.parametrize("zone", [SPOOL_DIR_RELPATH, STATE_DIR_RELPATH])
def test_password_mode_provisions_the_history_zone_directories(tmp_path: Path, zone: str) -> None:
    config = {**_config(), "project_root": str(tmp_path)}
    _ensure_agent_data_structure(copy.deepcopy(config))

    directory = tmp_path / zone / HISTORY_SERVICE
    assert directory.is_dir()
    assert stat.S_IMODE(directory.stat().st_mode) & stat.S_ISGID


@pytest.mark.parametrize("zone", [SPOOL_DIR_RELPATH, STATE_DIR_RELPATH])
def test_auth_none_provisions_no_history_zone_directories(tmp_path: Path, zone: str) -> None:
    config = {**_config("none"), "project_root": str(tmp_path)}
    _ensure_agent_data_structure(copy.deepcopy(config))

    assert not (tmp_path / zone / HISTORY_SERVICE).exists()


# ---------------------------------------------------------------------------
# The project image is built for the page
# ---------------------------------------------------------------------------


def test_the_page_alone_builds_the_project_image() -> None:
    assert container_lifecycle._project_image_build_target(_config(), {}) == "dls_controls:local"


def test_auth_none_builds_no_image_for_the_page() -> None:
    """The project image is still built, for agent-record, which every
    deployment runs; the page is not among the services it is built for."""
    services = container_lifecycle._deployed_project_image_services(_config("none"))
    assert HISTORY_SERVICE not in services


def test_an_image_override_takes_the_page_build_away() -> None:
    assert (
        container_lifecycle._project_image_build_target(
            _config(), {"OSPREY_WORKER_IMAGE": "registry.example.org/osprey:1"}
        )
        is None
    )


# ---------------------------------------------------------------------------
# Render context, the service template and the dashboard link
# ---------------------------------------------------------------------------


def test_the_render_context_names_the_history_origin_and_the_page_url() -> None:
    context = _inject_project_metadata(copy.deepcopy(_config()))

    assert context["osprey_history"]["external_origin"] == HISTORY_ORIGIN
    assert context["osprey_history_url"] == f"{HISTORY_ORIGIN}/history/"
    assert context["osprey_history"]["port"] == default_port(
        "history", base=resolve_port_base(_config())
    )
    # The page reads the agent record alone.
    assert "store_url" not in context["osprey_history"]


def test_no_page_url_without_the_page() -> None:
    context = _inject_project_metadata(copy.deepcopy(_config("none")))

    assert context["osprey_history_url"] == ""
    assert context["osprey_history"]["external_origin"] == ""


def _render(rel_path: str, context: dict[str, Any]) -> dict[str, Any]:
    from tests.deployment.test_compose_generator import _packaged_compose_template

    return yaml.safe_load(_packaged_compose_template(rel_path).render(**context))


def _full_context(config: dict[str, Any]) -> dict[str, Any]:
    context = _inject_project_metadata(copy.deepcopy(config))
    context.setdefault("services", {})
    return context


def test_the_history_service_runs_the_project_image_on_loopback() -> None:
    (service,) = _render("services/history/docker-compose.yml.j2", _full_context(_config()))[
        "services"
    ].values()

    assert service["image"] == "${OSPREY_WORKER_IMAGE:-dls_controls:local}"
    assert service["container_name"] == "dls_controls-history"
    assert service["command"][:2] == ["osprey", "history"]
    port = default_port("history", base=resolve_port_base(_config()))
    assert service["ports"] == [f"127.0.0.1:{port}:{HISTORY_CONTAINER_PORT}"]
    assert service["env_file"] == [
        ".env.agent-record.login.history",
        ".env.agent-record.history-gate",
    ]
    assert "depends_on" not in service
    environment = service["environment"]
    assert environment["OSPREY_TERMINAL_ACCEPT_ROSTER_SECRETS"] == "1"
    assert environment["OSPREY_TERMINAL_ROSTER_OWNERS"] == "HISTORY=history"
    assert "OSPREY_TERMINAL_BIND_HOST" not in environment
    assert "OSPREY_TERMINAL_SECRET" not in environment
    assert environment["OSPREY_TERMINAL_EXTERNAL_ORIGIN"] == HISTORY_ORIGIN
    assert environment["OSPREY_AUDIT_IDENTITY"] == HISTORY_SERVICE
    assert environment["OSPREY_AGENT_RECORD_URL"] == "http://agent-record:8000"
    assert service["labels"] == {
        "osprey.agent-record.cred.digest": "${OSPREY_AGENT_RECORD_DIGEST_HISTORY:-}",
    }
    # Its spool and state directories, each addressed by its identity.
    sources = [volume.split(":")[0] for volume in service["volumes"]]
    assert [source.rsplit("/", 2)[-2:] for source in sources] == [
        ["spool", HISTORY_SERVICE],
        ["state", HISTORY_SERVICE],
    ]


def _dispatcher_env(config: dict[str, Any]) -> dict[str, Any]:
    from tests.deployment.test_compose_generator import _dispatcher_context

    context = _dispatcher_context()
    metadata = _inject_project_metadata(copy.deepcopy(config))
    context["osprey_history_url"] = metadata["osprey_history_url"]
    document = _render("services/event_dispatcher/docker-compose.yml.j2", context)
    return document["services"]["event-dispatcher"]["environment"]


def test_the_page_url_reaches_the_dispatcher_only_when_the_page_is_deployed() -> None:
    assert _dispatcher_env(_config())["OSPREY_HISTORY_URL"] == f"{HISTORY_ORIGIN}/history/"
    assert "OSPREY_HISTORY_URL" not in _dispatcher_env(_config("none"))


# ---------------------------------------------------------------------------
# A registry-published password deployment: the page reads the agent record alone
# ---------------------------------------------------------------------------


def test_a_password_deployment_renders_and_serves_history() -> None:
    from osprey.agent_record.config import history_page_deployed
    from osprey.deployment.compose_generator import derived_compose_dirs
    from osprey.deployment.web_terminals.render import render_web_terminals, serves_history_origin

    config = _config(
        registry={"url": "git.dls.example.org:5050/physics/production/dls-profiles"},
    )

    # The one predicate holds, and every consumer follows it.
    assert history_page_deployed(config)
    assert "history" in derived_compose_dirs(config)
    assert HISTORY_SERVICE in service_audit_identities(config)
    assert serves_history_origin(config)
    assert _inject_project_metadata(copy.deepcopy(config))["osprey_history_url"] == (
        f"{HISTORY_ORIGIN}/history/"
    )

    # The service reads the record with its login and waits on no other service.
    (service,) = _render("services/history/docker-compose.yml.j2", _full_context(config))[
        "services"
    ].values()
    assert ".env.agent-record.login.history" in service["env_file"]
    assert "depends_on" not in service
    assert not any("STORE" in name for name in service["environment"])

    # nginx serves /history/ on the history page's origin, gated by the sidecar.
    artifacts = render_web_terminals(config, facility_name="Demo Light Source")
    nginx = artifacts["nginx/nginx.conf"]
    assert "location /history/ {" in nginx
    assert "location = /verify-history {" in nginx
