"""The trajectory service's roster grant, rendered from the one panel→service map.

Each per-user web terminal proxies its trajectory tab into the trajectory
container with the operator secret ITS container holds, exactly as it proxies
the BLUESKY tab into the bluesky-web sidecar. The two grants are one mechanism:
a panel→service map in the services render, and one entitlement walk over the
personas' rendered ``config.yml`` files. In the single-user shape there is no
roster; the terminal proxies with the deployment-wide secret, and the trajectory
container lists that secret once more under the build account's name so its
gate names the one operator the deployment has.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from jinja2 import Environment, FileSystemLoader

from osprey.deployment import compose_generator
from osprey.deployment.compose_generator import (
    _inject_project_metadata,
    _roster_grant_context,
)
from osprey.deployment.web_terminals.auth_credentials import terminal_secret_var
from osprey.deployment.web_terminals.personas import (
    BLUESKY_PANEL_ID,
    TRAJECTORY_PANEL_ID,
    personas_declaring_trajectory_panel,
)
from osprey.interfaces import web_auth
from osprey.interfaces.web_auth import (
    OPERATOR_SECRET_ENV,
    ROSTER_ACCEPT_ENV,
    ROSTER_OWNERS_ENV,
)

_TEMPLATES_ROOT = Path(__file__).resolve().parents[3] / "src" / "osprey" / "templates"
_TEMPLATE = "services/trajectory/docker-compose.yml.j2"

#: A target the bridge network admits, so the render needs no host network.
_IN_PROCESS_VA = {
    "type": "virtual_accelerator",
    "connector": {"virtual_accelerator": {"serving": "in_process"}},
}

#: The build account the single-user tests run the build as.
_BUILD_ACCOUNT = "opsuser"


@pytest.fixture(autouse=True)
def _build_account(monkeypatch):
    """The build runs as :data:`_BUILD_ACCOUNT`, whatever account runs the test."""
    monkeypatch.delenv("OSPREY_TERMINAL_USER", raising=False)
    monkeypatch.setenv("OSPREY_AUDIT_IDENTITY", _BUILD_ACCOUNT)


def _write_persona(root: Path, name: str, panels: dict[str, Any]) -> str:
    """Render a persona's ``config.yml`` under ``root``; return its ``project_path``."""
    project_path = f"build/proj-{name}"
    directory = root / f"proj-{name}"
    directory.mkdir(parents=True)
    (directory / "config.yml").write_text(
        yaml.safe_dump({"web": {"panels": panels}}), encoding="utf-8"
    )
    return project_path


def _config(repo: Path, web_terminals: dict[str, Any] | None = None) -> dict[str, Any]:
    config: dict[str, Any] = {
        "project_name": "proj",
        "project_root": str(repo),
        "deployment": {},
        "system": {"timezone": "UTC"},
        "control_system": _IN_PROCESS_VA,
        "services": {"trajectory": {"network": "bridge", "target": "va"}},
        "deployed_services": ["trajectory"],
    }
    if web_terminals is not None:
        config["modules"] = {"web_terminals": web_terminals}
    return config


def _multi_user(repo: Path, persona_root: Path) -> dict[str, Any]:
    """alice runs a persona showing both write tabs, bob a read-only one, carol the default."""
    writer = _write_persona(
        persona_root,
        "writer",
        {"trajectory": {"enabled": True}, "bluesky": {"enabled": True}},
    )
    reader = _write_persona(persona_root, "reader", {"trajectory": {"enabled": False}})
    return _config(
        repo,
        {
            "enabled": True,
            "default_persona": "writer",
            "personas": {
                "writer": {"project_path": writer},
                "reader": {"project_path": reader},
            },
            "users": [
                {"name": "alice", "index": 0, "persona": "writer"},
                {"name": "bob", "index": 1, "persona": "reader"},
                {"name": "carol", "index": 2},
            ],
        },
    )


def _render(config: dict[str, Any], source_dir: str, persona_root: Path | None) -> dict:
    """The trajectory compose environment, rendered through the deployment's context."""
    context = _inject_project_metadata(
        {**config, **_roster_grant_context(config, source_dir, persona_root)}
    )
    environment = Environment(loader=FileSystemLoader(str(_TEMPLATES_ROOT)), autoescape=False)
    document = yaml.safe_load(environment.get_template(_TEMPLATE).render(**context))
    return document["services"]["trajectory"]["environment"]


def _roster_vars(environment: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value
        for name, value in environment.items()
        if name.startswith(f"{OPERATOR_SECRET_ENV}_")
    }


# ---------------------------------------------------------------------------
# The map
# ---------------------------------------------------------------------------


def test_one_map_names_the_service_of_each_panel() -> None:
    grants = compose_generator._ROSTER_GRANTS

    assert set(grants) == {BLUESKY_PANEL_ID, TRAJECTORY_PANEL_ID}
    assert grants[BLUESKY_PANEL_ID].service == "bluesky_web"
    assert grants[TRAJECTORY_PANEL_ID].service == "trajectory"


def test_a_grant_is_empty_off_its_own_service(tmp_path: Path) -> None:
    config = _multi_user(tmp_path, tmp_path / "staging")
    staging = tmp_path / "staging"

    on_trajectory = _roster_grant_context(config, "/x/services/trajectory", staging)
    on_bluesky = _roster_grant_context(config, "/x/services/bluesky_web", staging)
    elsewhere = _roster_grant_context(config, "/x/services/dispatch_worker", staging)

    assert on_trajectory["bluesky_panel_secret_env_vars"] == []
    assert on_trajectory["bluesky_panel_roster_owners"] == ""
    assert on_bluesky["trajectory_panel_secret_env_vars"] == []
    assert on_bluesky["trajectory_panel_roster_owners"] == ""
    assert all(not value for value in elsewhere.values()), elsewhere


# ---------------------------------------------------------------------------
# Personas
# ---------------------------------------------------------------------------


def test_the_predicate_names_the_personas_declaring_the_trajectory_tab(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    config = _multi_user(tmp_path, staging)

    assert personas_declaring_trajectory_panel(config, tmp_path, staging) == {"writer"}


def test_the_predicate_reads_the_published_zone_without_a_staging_tree(tmp_path: Path) -> None:
    config = _multi_user(tmp_path, tmp_path / "build")

    assert personas_declaring_trajectory_panel(config, tmp_path) == {"writer"}


def test_the_trajectory_roster_lists_each_entitled_user(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    config = _multi_user(tmp_path, staging)

    environment = _render(config, "/x/services/trajectory", staging)

    alice, carol = terminal_secret_var("alice"), terminal_secret_var("carol")
    assert environment[ROSTER_ACCEPT_ENV] == "1"
    assert _roster_vars(environment) == {alice: f"${{{alice}:-}}", carol: f"${{{carol}:-}}"}
    assert environment[ROSTER_OWNERS_ENV] == "ALICE=alice,CAROL=carol"
    assert environment[OPERATOR_SECRET_ENV] == f"${{{OPERATOR_SECRET_ENV}}}"


def test_a_roster_with_nobody_entitled_renders_no_grant(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    config = _multi_user(tmp_path, staging)
    config["modules"]["web_terminals"]["users"] = [{"name": "bob", "index": 0, "persona": "reader"}]

    environment = _render(config, "/x/services/trajectory", staging)

    assert ROSTER_ACCEPT_ENV not in environment
    assert ROSTER_OWNERS_ENV not in environment
    assert _roster_vars(environment) == {}


# ---------------------------------------------------------------------------
# The single-user alias
# ---------------------------------------------------------------------------


def test_single_user_aliases_the_deployment_secret_to_the_build_account(tmp_path: Path) -> None:
    environment = _render(_config(tmp_path), "/x/services/trajectory", None)

    variable = terminal_secret_var(_BUILD_ACCOUNT)
    assert environment[ROSTER_ACCEPT_ENV] == "1"
    assert _roster_vars(environment) == {variable: f"${{{OPERATOR_SECRET_ENV}}}"}
    assert environment[ROSTER_OWNERS_ENV] == f"OPSUSER={_BUILD_ACCOUNT}"


def test_a_disabled_web_terminal_module_is_the_single_user_shape(tmp_path: Path) -> None:
    config = _config(tmp_path, {"enabled": False, "users": [{"name": "alice", "index": 0}]})

    environment = _render(config, "/x/services/trajectory", None)

    assert environment[ROSTER_OWNERS_ENV] == f"OPSUSER={_BUILD_ACCOUNT}"


def test_an_unnameable_build_account_renders_no_alias(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OSPREY_AUDIT_IDENTITY", "Ops.User")

    environment = _render(_config(tmp_path), "/x/services/trajectory", None)

    assert ROSTER_ACCEPT_ENV not in environment
    assert _roster_vars(environment) == {}


def test_the_bluesky_sidecar_takes_no_single_user_alias(tmp_path: Path) -> None:
    context = _roster_grant_context(_config(tmp_path), "/x/services/bluesky_web", None)

    assert context["bluesky_panel_secret_env_vars"] == []
    assert context["bluesky_panel_roster_owners"] == ""
    assert context["bluesky_panel_secret_aliases"] == {}


def test_the_alias_makes_the_gate_name_the_build_account(tmp_path: Path, monkeypatch) -> None:
    """The rendered environment, as compose would interpolate it, at the web gate:
    the deployment-wide secret the single-user terminal proxies with names the
    build account rather than nobody."""
    environment = _render(_config(tmp_path), "/x/services/trajectory", None)
    secret = "deployment-wide-secret"
    for name, value in environment.items():
        if name == OPERATOR_SECRET_ENV or name.startswith(f"{OPERATOR_SECRET_ENV}_"):
            monkeypatch.setenv(name, secret)
        elif name in (ROSTER_ACCEPT_ENV, ROSTER_OWNERS_ENV, "OSPREY_TERMINAL_BIND_HOST"):
            monkeypatch.setenv(name, str(value))
    # The container's own audit identity is the SERVICE, never a person.
    monkeypatch.setenv("OSPREY_AUDIT_IDENTITY", "trajectory")

    web_auth.reset_web_credentials()
    try:
        credentials = web_auth.get_web_credentials()
        assert credentials.identify_operator(secret) == _BUILD_ACCOUNT
        assert credentials.identify_operator("not-the-secret") is None
    finally:
        web_auth.reset_web_credentials()
