"""The panel-grant macro is the one renderer of a panel's accepted credentials.

``services/_panel_grant.j2`` turns ``panel_grant.credentials`` into the accept
flag, one variable per credential and the owners map the gate names operators
by. Every template serving a panel imports it, so the assertions here are about
the macro alone, about the gate reading back what it wrote, and about the two
bundled importers rendering one grant alike.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml
from jinja2 import Environment, FileSystemLoader

from osprey.deployment.compose_generator import _inject_project_metadata
from osprey.deployment.web_terminals.auth_credentials import (
    TERMINAL_SECRET_VAR_PREFIX,
    terminal_secret_var,
)
from osprey.interfaces import web_auth
from tests.deployment.test_compose_generator import _packaged_compose_template

SERVICES_DIR = Path(__file__).resolve().parents[2] / "src" / "osprey" / "templates" / "services"

_ALICE = terminal_secret_var("alice")
_OPS = terminal_secret_var("ops")

#: A roster user's credential and a roster-less alias, as roster.py writes them.
_ROSTER = {"var": _ALICE, "source": _ALICE, "owner": "alice", "required": False}
_ALIAS = {"var": _OPS, "source": "OSPREY_TERMINAL_SECRET", "owner": "ops", "required": True}


def _env(credentials: list[dict[str, Any]]) -> str:
    env = Environment(loader=FileSystemLoader(str(SERVICES_DIR)), autoescape=False)
    return env.get_template("_panel_grant.j2").module.env(credentials)


def _parsed(credentials: list[dict[str, Any]]) -> dict[str, str]:
    return yaml.safe_load("environment:" + _env(credentials))["environment"]


def test_no_credentials_render_nothing() -> None:
    assert _env([]) == ""


def test_required_decides_the_default() -> None:
    rendered = _env([_ROSTER, _ALIAS])

    assert rendered.startswith("\n") and not rendered.endswith("\n")
    env = _parsed([_ROSTER, _ALIAS])
    assert env["OSPREY_TERMINAL_ACCEPT_ROSTER_SECRETS"] == "1"
    assert env[_ALICE] == f"${{{_ALICE}:-}}"
    assert env[_OPS] == "${OSPREY_TERMINAL_SECRET}"


def test_the_gate_reads_the_owner_the_macro_wrote(monkeypatch: pytest.MonkeyPatch) -> None:
    credentials = [_ROSTER, _ALIAS]
    assert all(c["var"].startswith(TERMINAL_SECRET_VAR_PREFIX) for c in credentials)
    owners = _parsed(credentials)["OSPREY_TERMINAL_ROSTER_OWNERS"]

    monkeypatch.setenv(web_auth.ROSTER_OWNERS_ENV, owners)

    assert web_auth._roster_owners_from_env() == {
        c["var"].removeprefix(TERMINAL_SECRET_VAR_PREFIX): c["owner"] for c in credentials
    }


def _grant_env(rel_path: str, key: str, compose_key: str, grant: dict[str, Any]) -> dict:
    config = _inject_project_metadata(
        {
            "project_name": "fixture",
            "services": {key: {}, "bluesky": {}},
            "deployed_services": [key],
            "deployment": {},
            "system": {"timezone": "UTC"},
        }
    )
    context = {
        **config,
        "service_key": key,
        "service": config["services"][key],
        "panel_grant": copy.deepcopy(grant),
    }
    text = _packaged_compose_template(rel_path).render(**context)
    env = yaml.safe_load(text)["services"][compose_key]["environment"]
    return {
        name: value
        for name, value in env.items()
        if name.startswith(TERMINAL_SECRET_VAR_PREFIX)
        or name in ("OSPREY_TERMINAL_ACCEPT_ROSTER_SECRETS", "OSPREY_TERMINAL_ROSTER_OWNERS")
    }


@pytest.mark.parametrize("credential", [_ROSTER, _ALIAS], ids=["roster", "alias"])
def test_both_bundled_templates_render_one_grant_alike(credential: dict[str, Any]) -> None:
    grant = {"credentials": [credential]}
    app_runtime = _grant_env(
        "services/app_runtime/docker-compose.yml.j2", "knob_backend", "knob_backend", grant
    )
    bluesky_web = _grant_env(
        "services/bluesky_web/docker-compose.yml.j2", "bluesky_web", "bluesky-web", grant
    )

    assert app_runtime == bluesky_web
    assert app_runtime[credential["var"]]
    assert app_runtime["OSPREY_TERMINAL_ACCEPT_ROSTER_SECRETS"] == "1"
