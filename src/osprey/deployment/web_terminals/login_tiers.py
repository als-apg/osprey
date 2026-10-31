"""The login→tier map shared containers read: ``control_system.login_tiers``.

A shared container (the Bluesky queue, a dispatch worker) runs one render for
every login whose job it carries, so the render's own tier cannot say whose
write a job is. The build therefore stages, beside that render, the tier each
login's OWN render sets: ``{login: "never" | "asked" | "allowed"}``, the word
verbatim. This module is the one producer of that map; nothing at runtime
writes it.

What is in it:

* each roster person whose tier no sign-in decides
  (:func:`~osprey.deployment.web_terminals.roster.sign_in_decides_tier`) and whose
  render (:func:`~osprey.deployment.web_terminals.roster.effective_tier`) has a
  readable staged ``config.yml``, at :func:`~osprey_connectors.types.tier_writes`
  of that render's ``control_system``;
* the account running the build (the single-user identity included), at the root
  render's tier — only when it names no roster card.

What is deliberately absent, so a shared container refuses it ``owner_unknown``:
a card whose tier is decided at sign-in (even with a ``default_tier`` or pin), a
card with no tier, a card whose binding does not resolve, a card whose render
cannot be read, and a build account that names any roster card the rule above
left out. Absent is never filled in with the root tier: that would be a guess.
"""

from __future__ import annotations

from typing import Any

from osprey.deployment.web_terminals.roster import (
    UnresolvedRoleError,
    _render_configs,
    as_dict,
    authored_people,
    effective_tier,
    resolve_authorization_roles,
    roster_entries,
    sign_in_decides_tier,
)
from osprey_connectors.types import TierWord

#: Where the map is rendered in a shared container's config.
LOGIN_TIERS_KEY = "login_tiers"

Tier = TierWord


def _entry_login(entry: Any) -> tuple[str | None, dict[str, Any]]:
    """A raw roster entry's login and its mapping form; ``None`` for no login."""
    if isinstance(entry, str):
        return (entry or None), {}
    if isinstance(entry, dict):
        name = entry.get("name")
        if isinstance(name, str) and name:
            return name, entry
    return None, {}


def _roster_card_names(web_terminals: dict[str, Any]) -> set[str]:
    """Every name a roster card goes by — people as authored, then app cards."""
    names = {login for login, _ in map(_entry_login, authored_people(web_terminals)) if login}
    for entry in roster_entries(web_terminals, strict=False):
        name = entry.get("name") if isinstance(entry, dict) else None
        if isinstance(name, str) and name:
            names.add(name)
    return names


def build_login_tiers(config: Any, repo_root: Any, *, layer_root: Any) -> dict[str, Tier]:
    """The ``control_system.login_tiers`` map for *config*'s shared containers.

    A pure read of *config* (the resolved root render) and the renders staged
    under *layer_root* (see
    :func:`~osprey.deployment.web_terminals.roster._render_configs`); it writes
    nothing.

    Args:
        config: The resolved root config.
        repo_root: The repo root the renders' ``project_path`` values are
            spelled against.
        layer_root: The staging tree a build in flight renders into, or
            ``None`` for the published build zone.

    Returns:
        ``{login: "never" | "asked" | "allowed"}``, each login's tier word
        verbatim, keys exactly as the roster names them.
    """
    from osprey.deployment.compose_generator import _control_tree_build_account
    from osprey_connectors.identity import UNKNOWN_IDENTITY
    from osprey_connectors.types import tier_writes

    root = as_dict(config)
    web_terminals = as_dict(as_dict(root.get("modules")).get("web_terminals"))
    has_roster = bool(web_terminals.get("enabled"))

    login_tiers: dict[str, Tier] = {}
    card_names: set[str] = set()

    if has_roster:
        card_names = _roster_card_names(web_terminals)
        try:
            roles = resolve_authorization_roles(web_terminals)
        except ValueError:
            roles = None
        renders = dict(_render_configs(root, repo_root, layer_root))
        default_tier = web_terminals.get("default_tier")
        seen: set[str] = set()
        for raw_entry in authored_people(web_terminals):
            login, entry = _entry_login(raw_entry)
            if login is None or login in seen:
                # A duplicate login is lint's finding; the first card decides.
                continue
            seen.add(login)
            if sign_in_decides_tier(entry, web_terminals):
                continue
            if roles is None and entry.get("role"):
                # An authorization stanza that does not parse binds no role.
                continue
            try:
                render = effective_tier(entry, roles or {}, default_tier, strict=True)
            except UnresolvedRoleError:
                continue
            render_config = renders.get(render) if render else None
            if not isinstance(render_config, dict):
                continue
            login_tiers[login] = tier_writes(render_config.get("control_system"))

    build_account = _control_tree_build_account(root, repo_root)
    if (
        isinstance(build_account, str)
        and build_account
        and build_account != UNKNOWN_IDENTITY
        and build_account not in card_names
    ):
        login_tiers[build_account] = tier_writes(root.get("control_system"))

    return login_tiers
