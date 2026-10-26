"""The history page's carriers into the auth sidecar's environment.

The render emits them, and the sidecar parses them back through the names in
:mod:`osprey.services.auth_sidecar.roster_env`; the round trip below holds the
two ends to one spelling.
"""

from __future__ import annotations

import copy
from typing import Any

import yaml

from osprey.agent_record.contract import KINDS
from osprey.deployment.web_terminals.render import history_carriers, render_web_terminals
from osprey.services.auth_sidecar.app import AuthSettings
from tests._renders import tier_render


def _config(**web: Any) -> dict[str, Any]:
    return {
        "project_name": "dls_controls",
        "facility": {"prefix": "dls"},
        "system": {"timezone": "UTC"},
        "registry": {"url": "git.dls.example.org:5050/physics/dls-profiles"},
        "deploy": {"host": "dls-deploy", "fqdn": "dls-deploy.dls.example.org"},
        "deployed_services": ["telemetry_store", "telemetry_gateway"],
        "services": {"telemetry_store": {"readers": [{"name": "store-only"}]}},
        "agent_record": {
            "readers": [
                {"name": "carol", "kinds": ["prompt", "answer", "prompt"]},
                {"name": "ext", "oidc_subject": " ext@partner.org "},
            ]
        },
        "modules": {
            "web_terminals": {
                "enabled": True,
                "auth": {"method": "password", "allow_insecure_http": True},
                "users": [
                    {"name": "alice", "index": 0, "history": "all"},
                    "bob",
                    {"name": "ops-desk", "index": 2, "role": "operator"},
                ],
                "authorization": {
                    "roles": {
                        "operator": {"history": "all", "tier": "assistant"},
                        "viewer": {"history": "own", "tier": "assistant"},
                    }
                },
                "default_tier": "assistant",
                "renders": {"assistant": tier_render("assistant", "dls_controls-assistant")},
                **web,
            }
        },
    }


def _sidecar_env(config: dict[str, Any]) -> dict[str, str]:
    compose = yaml.safe_load(render_web_terminals(config)["docker-compose.web.yml"])
    (sidecar,) = [
        service
        for service in compose["services"].values()
        if "auth_sidecar" in " ".join(service.get("command") or [])
    ]
    return dict(line.split("=", 1) for line in sidecar["environment"])


def test_the_carriers_are_rendered_into_the_sidecar() -> None:
    env = _sidecar_env(_config())
    assert env["OSPREY_AUTH_HISTORY_ORIGIN"] == "http://dls-deploy.dls.example.org:10002"
    assert env["OSPREY_AUTH_ROSTER_HISTORY_ALICE"] == "all"
    assert env["OSPREY_AUTH_ROSTER_HISTORY_BOB"] == "own"
    assert env["OSPREY_AUTH_ROSTER_HISTORY_OPS_DESK"] == "own"
    assert env["OSPREY_AUTH_ROLE_HISTORY_OPERATOR"] == "all"
    assert "OSPREY_AUTH_ROLE_HISTORY_VIEWER" not in env
    # The agent record's readers, never the old store's.
    assert env["OSPREY_AUTH_READERS"] == "carol,ext"
    assert env["OSPREY_AUTH_READER_TABLES_CAROL"] == "prompt,answer"
    # No kinds stated reads every kind.
    assert env["OSPREY_AUTH_READER_TABLES_EXT"] == ",".join(KINDS)
    assert env["OSPREY_AUTH_READER_SUBJECT_EXT"] == "ext@partner.org"
    assert "OSPREY_AUTH_READER_SUBJECT_CAROL" not in env
    assert not any("STORE_ONLY" in key for key in env)


def test_the_sidecar_reads_back_what_the_render_wrote() -> None:
    settings = AuthSettings.from_env(_sidecar_env(_config()))
    assert settings.history_origin == "http://dls-deploy.dls.example.org:10002"
    assert settings.user_history_all("alice")
    assert not settings.user_history_all("bob")
    assert settings.role_history_all("operator")
    assert not settings.role_history_all("viewer")
    assert settings.readers == ("carol", "ext")
    assert settings.tables_for_reader("carol") == ("prompt", "answer")
    assert settings.reader_subject("ext") == "ext@partner.org"


def test_a_stated_reader_origin_is_the_carried_one() -> None:
    config = _config(
        external_origin="https://terminals.dls.example.org",
        reader_origin="https://terminals.dls.example.org:8443",
    )
    assert history_carriers(config)[0] == (
        "OSPREY_AUTH_HISTORY_ORIGIN=https://terminals.dls.example.org:8443"
    )


def test_nothing_is_carried_without_the_history_page() -> None:
    no_sign_on = copy.deepcopy(_config())
    no_sign_on["modules"]["web_terminals"]["auth"] = {"method": "none"}
    assert history_carriers(no_sign_on) == []


def test_a_store_reader_subject_is_not_carried() -> None:
    """The sidecar's reader mapping comes from ``agent_record.readers`` alone."""
    config = copy.deepcopy(_config())
    config["services"]["telemetry_store"]["readers"] = [{"name": "old", "oidc_subject": "o@x"}]
    del config["agent_record"]
    carried = history_carriers(config)
    assert "OSPREY_AUTH_READERS=" in carried
    assert not any("_OLD=" in line or "o@x" in line for line in carried)


def test_the_page_is_carried_without_the_old_store() -> None:
    """The page reads the agent record: a signed-in deployment carries it with no store."""
    no_store = copy.deepcopy(_config())
    no_store["deployed_services"] = []
    carried = history_carriers(no_store)
    assert carried[0].startswith("OSPREY_AUTH_HISTORY_ORIGIN=")
    assert any(line.startswith("OSPREY_AUTH_ROSTER_HISTORY_") for line in carried)
