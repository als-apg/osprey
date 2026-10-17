"""The history service: rendered, provisioned and linked exactly when the page is deployed.

The page is derived, never a ``deployed_services`` entry: it exists when the
deployment runs the telemetry store and its web terminals sign people in
(:func:`~osprey.deployment.telemetry_config.history_page_deployed`). Every
consumer keys on that one predicate — the audit directory and the gateway's
receiver for it, the project-image build, the compose render, and the dispatch
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

from osprey.deployment import container_lifecycle
from osprey.deployment import telemetry_config as tc
from osprey.deployment import telemetry_gateway as gw
from osprey.deployment.compose_generator import (
    HISTORY_CONTAINER_PORT,
    _ensure_agent_data_structure,
    _inject_project_metadata,
    service_audit_identities,
)
from osprey.port_layout import default_port
from osprey_connectors.workspace import AUDIT_DIR_RELPATH

READER_ORIGIN = "http://dls-deploy.dls.example.org:10002"


def _config(method: str = "password", **root: Any) -> dict[str, Any]:
    """A deployment with the store, the gateway and signed-in web terminals."""
    return {
        "project_name": "dls_controls",
        "project_root": "/r/dls_controls",
        "system": {"timezone": "UTC"},
        "deploy": {"host": "dls-deploy", "fqdn": "dls-deploy.dls.example.org"},
        "deployed_services": ["telemetry_store", "telemetry_gateway"],
        "modules": {
            "web_terminals": {
                "enabled": True,
                "auth": {"method": method, "allow_insecure_http": True},
                "users": [{"name": "alice", "index": 0}],
                "default_persona": "assistant",
                "personas": {"assistant": {"project": "dls_controls-assistant"}},
            }
        },
        **root,
    }


@pytest.mark.parametrize("method", ["password", "oidc"])
def test_a_signed_in_deployment_with_the_store_has_the_history_identity(method: str) -> None:
    assert tc.HISTORY_SERVICE in service_audit_identities(_config(method))


@pytest.mark.parametrize(
    "config",
    [
        _config("none"),
        {**_config(), "deployed_services": ["telemetry_gateway"]},
        {**_config(), "modules": {"web_terminals": {"enabled": False}}},
    ],
    ids=["auth-none", "no-store", "no-web-terminals"],
)
def test_no_history_identity_without_the_page(config: dict[str, Any]) -> None:
    assert tc.HISTORY_SERVICE not in service_audit_identities(config)


def test_password_mode_provisions_the_history_audit_directory_and_its_receiver(
    tmp_path: Path,
) -> None:
    config = {**_config(), "project_root": str(tmp_path)}
    _ensure_agent_data_structure(copy.deepcopy(config))

    directory = tmp_path / AUDIT_DIR_RELPATH / tc.HISTORY_SERVICE
    assert directory.is_dir()
    assert stat.S_IMODE(directory.stat().st_mode) & stat.S_ISGID

    receivers = gw.ledger_receivers(config, tmp_path / AUDIT_DIR_RELPATH)
    included = [
        include for receiver in receivers.values() for include in receiver.get("include", [])
    ]
    assert any(f"/{tc.HISTORY_SERVICE}/" in include for include in included), receivers


def test_auth_none_provisions_neither_the_directory_nor_a_receiver(tmp_path: Path) -> None:
    config = {**_config("none"), "project_root": str(tmp_path)}
    _ensure_agent_data_structure(copy.deepcopy(config))

    assert not (tmp_path / AUDIT_DIR_RELPATH / tc.HISTORY_SERVICE).exists()
    assert tc.HISTORY_SERVICE not in gw.configured_ledger_identities(config)
    receivers = gw.ledger_receivers(config, tmp_path / AUDIT_DIR_RELPATH)
    included = [
        include for receiver in receivers.values() for include in receiver.get("include", [])
    ]
    assert not any(f"/{tc.HISTORY_SERVICE}/" in include for include in included)


# ---------------------------------------------------------------------------
# The project image is built for the page
# ---------------------------------------------------------------------------


def test_the_page_alone_builds_the_project_image() -> None:
    assert container_lifecycle._project_image_build_target(_config(), {}) == "dls_controls:local"


def test_auth_none_with_the_store_alone_builds_nothing() -> None:
    assert container_lifecycle._project_image_build_target(_config("none"), {}) is None


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


def test_the_render_context_names_the_readers_origin_and_the_page_url() -> None:
    context = _inject_project_metadata(copy.deepcopy(_config()))

    assert context["osprey_history"]["external_origin"] == READER_ORIGIN
    assert context["osprey_history_url"] == f"{READER_ORIGIN}/history/"
    assert context["osprey_history"]["port"] == default_port(
        "history", base=tc.resolve_port_base(_config())
    )
    assert context["osprey_history"]["store_url"] == "http://telemetry_store:8123"


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
    port = default_port("history", base=tc.resolve_port_base(_config()))
    assert service["ports"] == [f"127.0.0.1:{port}:{HISTORY_CONTAINER_PORT}"]
    assert service["env_file"] == [".env.telemetry-history", ".env.telemetry-history-gate"]
    environment = service["environment"]
    assert environment["OSPREY_TERMINAL_ACCEPT_ROSTER_SECRETS"] == "1"
    assert environment["OSPREY_TERMINAL_ROSTER_OWNERS"] == "HISTORY=history"
    assert "OSPREY_TERMINAL_BIND_HOST" not in environment
    assert "OSPREY_TERMINAL_SECRET" not in environment
    assert environment["OSPREY_TERMINAL_EXTERNAL_ORIGIN"] == READER_ORIGIN
    assert environment["OSPREY_AUDIT_IDENTITY"] == tc.HISTORY_SERVICE
    assert service["labels"] == {
        "osprey.telemetry.cred.digest": "${OSPREY_TELEMETRY_DIGEST_HISTORY:-}"
    }
    (volume,) = service["volumes"]
    assert volume.split(":")[0].endswith(f"/{tc.HISTORY_SERVICE}")


def _dispatcher_env(config: dict[str, Any]) -> dict[str, Any]:
    from tests.deployment.test_compose_generator import _dispatcher_context

    context = _dispatcher_context()
    metadata = _inject_project_metadata(copy.deepcopy(config))
    context["osprey_history_url"] = metadata["osprey_history_url"]
    document = _render("services/event_dispatcher/docker-compose.yml.j2", context)
    return document["services"]["event-dispatcher"]["environment"]


def test_the_page_url_reaches_the_dispatcher_only_when_the_page_is_deployed() -> None:
    assert _dispatcher_env(_config())["OSPREY_HISTORY_URL"] == f"{READER_ORIGIN}/history/"
    assert "OSPREY_HISTORY_URL" not in _dispatcher_env(_config("none"))
