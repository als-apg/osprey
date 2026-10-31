"""The panel roster grant: one table, one entitlement walk, every panel's service.

A panel whose service is reached by each per-user terminal with the secret ITS
container holds needs that service's compose to list the entitled users'
secrets and name whose each one is. The table says which service holds each
panel's grant: the bluesky row is built in, and an application's row is read off
the rendered ``services.<k>.roster_grant`` its projection writes — never off
``app.yml``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import yaml

from osprey.deployment.web_terminals import roster
from osprey.deployment.web_terminals.auth_credentials import terminal_secret_var
from osprey.deployment.web_terminals.roster import (
    BLUESKY_PANEL_ID,
    PanelRosterGrant,
    bluesky_panel_roster_owners,
    bluesky_panel_secret_env_vars,
    panel_roster_grant_table,
    service_panel_grant,
)
from tests._renders import tier_render


def _projected(panel_id: str = "knob-demo", single_user_alias: object = False) -> dict:
    return {"roster_grant": {"panel_id": panel_id, "single_user_alias": single_user_alias}}


# ---------------------------------------------------------------------------
# (1) the table
# ---------------------------------------------------------------------------


def test_the_bluesky_row_is_built_in() -> None:
    assert panel_roster_grant_table({}) == {
        BLUESKY_PANEL_ID: PanelRosterGrant(service="bluesky_web")
    }


def test_a_projected_row_is_read_off_the_rendered_service_block() -> None:
    config = {"services": {"knob-demo": _projected(single_user_alias=True), "plain": {}}}
    table = panel_roster_grant_table(config)
    assert table["knob-demo"] == PanelRosterGrant(service="knob-demo", single_user_alias=True)
    assert set(table) == {BLUESKY_PANEL_ID, "knob-demo"}


@pytest.mark.parametrize("value", ["true", 1, None])
def test_single_user_alias_holds_only_for_the_literal_true(value: object) -> None:
    config = {"services": {"knob": _projected(single_user_alias=value)}}
    assert panel_roster_grant_table(config)["knob-demo"].single_user_alias is False


@pytest.mark.parametrize("panel_id", [BLUESKY_PANEL_ID, "knob-demo"])
def test_two_rows_for_one_panel_are_refused(panel_id: str) -> None:
    config = {"services": {"a": _projected(panel_id), "b": _projected("knob-demo")}}
    with pytest.raises(ValueError, match=panel_id):
        panel_roster_grant_table(config)


@pytest.mark.parametrize("panel_id", ["", 7, None])
def test_a_row_without_a_panel_id_is_refused(panel_id: object) -> None:
    config = {"services": {"knob": {"roster_grant": {"panel_id": panel_id}}}}
    with pytest.raises(ValueError, match="knob"):
        panel_roster_grant_table(config)


def test_roster_does_not_import_the_application_loader() -> None:
    """Deployment code reads the rendered row, never ``app.yml``."""
    tree = ast.parse(Path(roster.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)
    assert not any(name.startswith("osprey.apps.loader") for name in imported), imported


# ---------------------------------------------------------------------------
# (2) the grant on a panel's service
# ---------------------------------------------------------------------------


def _deployment(tmp_path: Path, declared: dict[str, tuple[str, ...]]) -> dict:
    """A two-render roster: ``operator`` declares the panels named for it."""
    for tier in ("viewer", "operator"):
        project = tmp_path / "build" / f"demo-{tier}"
        project.mkdir(parents=True)
        panels = {pid: {"url": "http://localhost:1"} for pid in declared.get(tier, ())}
        (project / "config.yml").write_text(
            yaml.safe_dump({"web": {"panels": panels}}), encoding="utf-8"
        )
    return {
        "services": {"knob-demo": _projected()},
        "modules": {
            "web_terminals": {
                "default_tier": "viewer",
                "renders": {
                    tier: tier_render(tier, f"demo-{tier}", project_path=f"build/demo-{tier}")
                    for tier in ("viewer", "operator")
                },
                "users": [
                    {"name": "alice", "index": 0, "tier": "operator"},
                    {"name": "bob", "index": 1, "tier": "viewer"},
                    "carol",
                ],
            }
        },
    }


def test_a_projected_panels_grant_follows_each_users_own_render(tmp_path: Path) -> None:
    config = _deployment(tmp_path, {"operator": ("knob-demo",)})
    alice = terminal_secret_var("alice")
    assert service_panel_grant(config, tmp_path, "knob-demo") == {
        "secret_env_vars": [alice],
        "roster_owners": f"{alice.removeprefix('OSPREY_TERMINAL_SECRET_')}=alice",
    }


def test_every_other_service_gets_the_empty_grant(tmp_path: Path) -> None:
    config = _deployment(tmp_path, {"operator": ("knob-demo", BLUESKY_PANEL_ID)})
    empty = {"secret_env_vars": [], "roster_owners": ""}
    assert service_panel_grant(config, tmp_path, "plain") == empty
    assert service_panel_grant(config, tmp_path, "bluesky") == empty


def test_the_bluesky_grant_is_the_tables_row(tmp_path: Path) -> None:
    """The legacy names are views of the same grant the table gives bluesky_web."""
    config = _deployment(tmp_path, {"operator": (BLUESKY_PANEL_ID,)})
    config["modules"]["web_terminals"]["default_tier"] = "operator"
    grant = service_panel_grant(config, tmp_path, "bluesky_web")

    assert grant["secret_env_vars"] == [terminal_secret_var("alice"), terminal_secret_var("carol")]
    assert bluesky_panel_secret_env_vars(config, tmp_path) == grant["secret_env_vars"]
    assert bluesky_panel_roster_owners(config, tmp_path) == grant["roster_owners"]
    assert service_panel_grant(config, tmp_path, "knob-demo")["secret_env_vars"] == []


def test_a_deployment_with_no_roster_grants_nothing(tmp_path: Path) -> None:
    config = {"services": {"knob-demo": _projected(single_user_alias=False)}}
    assert service_panel_grant(config, tmp_path, "knob-demo") == {
        "secret_env_vars": [],
        "roster_owners": "",
    }
