"""A shared app's own ``access``: who may open it, and the ``tier:`` principal."""

from __future__ import annotations

import json

import pytest
import yaml

from osprey.deployment.web_terminals.lint import lint_web_terminals
from osprey.deployment.web_terminals.render import render_web_terminals
from osprey.deployment.web_terminals.roster import (
    expand_tier_principals,
    resolve_access_principals,
    resolve_cards,
    roster_entries,
)
from osprey.services.auth_sidecar.app import ENV_ROSTER_ACCESS_PREFIX, AuthSettings
from tests._renders import app_render, tier_render

EXPERT_ACCESS = ["tier:admin", "user:expert@example.org"]


def _web_terminals(*, method: str = "oidc", shared: list | None = None) -> dict:
    web_terminals = {
        "enabled": True,
        "default_tier": "viewer",
        "renders": {
            "viewer": tier_render("viewer", "demo-viewer"),
            "admin": tier_render("admin", "demo-admin"),
            "search": app_render("search", "demo-search"),
            "panel": app_render("panel", "demo-panel"),
        },
        "users": [
            {"name": "alice", "index": 0, "tier": "admin", "oidc_subject": "alice@example.org"},
            {"name": "bob", "index": 1, "tier": "viewer", "oidc_subject": "bob@example.org"},
        ],
        "shared": shared
        if shared is not None
        else ["search", {"name": "panel", "access": EXPERT_ACCESS}],
        "auth": {"method": method, "allow_insecure_http": True},
    }
    if method == "oidc":
        web_terminals["auth"]["oidc"] = {"issuer": "https://sso.example.org"}
    return web_terminals


def _config(**kwargs) -> dict:
    return {
        "facility": {"prefix": "demo"},
        "system": {"timezone": "UTC"},
        "registry": {"url": "registry.example.org/demo"},
        "deploy": {"host": "demo", "fqdn": "demo.example.org"},
        "modules": {"web_terminals": _web_terminals(**kwargs)},
    }


def _sidecar_settings(config: dict) -> AuthSettings:
    overlay = yaml.safe_load(render_web_terminals(config)["docker-compose.web.yml"])
    lines = overlay["services"]["auth"]["environment"]
    return AuthSettings.from_env(dict(line.split("=", 1) for line in lines))


def _codes(config: dict) -> list[str]:
    return [finding.code for finding in lint_web_terminals(config)]


def test_a_bare_app_name_still_opens_to_the_whole_roster():
    apps = {e["name"]: e for e in roster_entries(_web_terminals()) if e.get("kind") == "app"}
    assert apps["search"]["access"] == "any"


def test_a_mapping_item_carries_its_access_and_keeps_its_slot():
    apps = {e["name"]: e for e in roster_entries(_web_terminals()) if e.get("kind") == "app"}
    assert apps["panel"]["access"] == EXPERT_ACCESS
    assert apps["search"]["index"] - apps["panel"]["index"] == 1


def test_a_tier_principal_is_accepted_by_the_one_parser():
    principals = resolve_access_principals({"name": "panel", "access": EXPERT_ACCESS})
    assert principals == frozenset(EXPERT_ACCESS)


@pytest.mark.parametrize("member", ["tier:", "tier:a b", "tier:$X", "tier:a:b"])
def test_a_tier_principal_must_name_one_tier(member):
    with pytest.raises(ValueError):
        resolve_access_principals({"name": "panel", "access": [member]})


def test_a_tier_principal_expands_to_the_identities_of_that_tiers_people():
    cards = resolve_cards(_web_terminals(), {}, strict=False)
    expanded = expand_tier_principals(frozenset(EXPERT_ACCESS), cards)
    assert expanded == frozenset({"user:alice@example.org", "user:expert@example.org"})


def test_the_sidecar_admits_exactly_the_tier_and_the_named_expert():
    settings = _sidecar_settings(_config())
    assert settings.card_admits("panel", "alice@example.org")
    assert settings.card_admits("panel", "expert@example.org")
    assert not settings.card_admits("panel", "bob@example.org")
    assert settings.card_admits("search", "bob@example.org")


def test_no_tier_principal_reaches_the_sidecar():
    overlay = yaml.safe_load(render_web_terminals(_config())["docker-compose.web.yml"])
    lines = overlay["services"]["auth"]["environment"]
    (value,) = [
        line.split("=", 1)[1]
        for line in lines
        if line.startswith(f"{ENV_ROSTER_ACCESS_PREFIX}PANEL=")
    ]
    assert json.loads(value) == ["user:alice@example.org", "user:expert@example.org"]


def test_a_mapping_with_an_unknown_key_is_refused():
    config = _config(shared=["search", {"name": "panel", "acess": ["tier:admin"]}])
    assert "web_terminals.invalid_shared_entry" in _codes(config)


def test_a_malformed_app_access_is_refused():
    config = _config(shared=[{"name": "panel", "access": ["group:ops"]}])
    assert "web_terminals.invalid_shared_access" in _codes(config)


def test_a_tier_nobody_runs_is_refused():
    config = _config(shared=[{"name": "panel", "access": ["tier:readwrite"]}])
    assert "web_terminals.tier_principal_admits_nobody" in _codes(config)


def test_identity_principals_on_an_app_are_refused_under_password():
    config = _config(method="password")
    assert "web_terminals.access_principal_without_idp" in _codes(config)


def test_the_valid_shape_lints_clean_of_these_codes():
    codes = set(_codes(_config()))
    assert not codes & {
        "web_terminals.invalid_shared_entry",
        "web_terminals.invalid_shared_access",
        "web_terminals.tier_principal_admits_nobody",
        "web_terminals.access_principal_without_idp",
    }
