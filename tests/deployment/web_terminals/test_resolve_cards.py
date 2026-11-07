"""Resolving every card the roster runs: the people, then the shared apps.

``resolve_cards`` turns each roster card into the image and project it runs.
A person resolves through the tier they are bound to (``tier:``, a ``role:``,
or the deployment's ``default_tier``). A shared app resolves to its own render,
``build/<project>-<app>``, named by ``render_project``. A name is a tier or an
app, never both, and the naming refuses either direction. The derived app
entries live in memory only: a real build of a repo with ``shared:`` leaves
``build/config.yml`` and ``profile.yml`` listing people only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from osprey.deployment.web_terminals.render_naming import (
    RENDER_KIND_SHARED,
    RENDER_KIND_TIER,
    render_kind_conflict,
    render_project,
)
from osprey.deployment.web_terminals.roster import (
    RENDER_CATALOG_REQUIRED,
    UnresolvedRoleError,
    effective_tier,
    is_app_entry,
    resolve_cards,
)
from osprey.port_layout import INDEX_MAX
from tests._renders import tier_render

PROJECT = "facility"

TITLES = {"logbook": "Logbook Research", "knowledge": "Facility Knowledge"}

PEOPLE: list[Any] = [
    "alice",
    {"name": "bob", "index": 1, "tier": "readwrite", "display_name": "Bob's terminal"},
    {"name": "carol", "index": 2, "tier": "admin"},
]


def _catalog() -> dict[str, Any]:
    return {
        tier: tier_render(tier, f"{PROJECT}-{tier}", project_path=f"build/{PROJECT}-{tier}")
        for tier in ("readonly", "readwrite", "admin")
    }


def _subtree(**extra: Any) -> dict[str, Any]:
    """The tier catalog plus *extra*; extra ``renders`` entries join the tiers."""
    subtree: dict[str, Any] = {
        "enabled": True,
        "users": list(PEOPLE),
        "default_tier": "readonly",
        "image_source": "local",
    }
    renders = {**_catalog(), **extra.pop("renders", {})}
    subtree.update(extra)
    subtree["renders"] = renders
    return subtree


def _apps(resolved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [entry for entry in resolved if is_app_entry(entry)]


def _people(resolved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [entry for entry in resolved if not is_app_entry(entry)]


# ── The people resolve as they always did ────────────────────────────────────


def test_people_resolve_identically_with_and_without_shared() -> None:
    # Arrange
    without = _subtree()
    with_apps = _subtree(shared=["logbook", "knowledge"])

    # Act
    plain = resolve_cards(without, {}, project_name=PROJECT)
    mixed = resolve_cards(with_apps, {}, project_name=PROJECT, shared_titles=TITLES)

    # Assert
    assert _people(mixed) == plain
    assert [entry["name"] for entry in mixed] == ["alice", "bob", "carol", "logbook", "knowledge"]


def test_people_resolve_through_their_tier_and_the_default_tier() -> None:
    resolved = {entry["name"]: entry for entry in resolve_cards(_subtree(), {})}
    assert resolved["alice"]["render"] == "readonly"  # default_tier
    assert resolved["bob"]["render"] == "readwrite"
    assert resolved["carol"]["image"] == f"{PROJECT}-admin:local"
    assert resolved["bob"]["display_name"] == "Bob's terminal"


def test_default_tier_alone_names_the_default() -> None:
    resolved = resolve_cards(_subtree(default_tier="readwrite"), {})
    assert resolved[0]["render"] == "readwrite"


def test_no_default_tier_is_still_refused_under_strict() -> None:
    subtree = _subtree()
    del subtree["default_tier"]
    with pytest.raises(ValueError, match="default_tier"):
        resolve_cards(subtree, {})
    assert RENDER_CATALOG_REQUIRED  # the refusal is the shared constant


# ── effective_tier reads `tier` ───────────────────────────────────────────


def test_effective_tier_reads_the_tier_pin() -> None:
    assert effective_tier({"name": "a", "tier": "admin"}, {}, "readonly") == "admin"


def test_effective_tier_reads_no_other_pin_key() -> None:
    # Only `tier` pins: any other key on the entry leaves it on the default.
    assert effective_tier({"name": "a", "render": "readwrite"}, {}, "readonly") == "readonly"


def test_effective_tier_falls_back_to_the_default_tier() -> None:
    assert effective_tier({"name": "a"}, {}, "readonly") == "readonly"
    assert effective_tier({"name": "a", "tier": ""}, {}, "readonly") == "readonly"


def test_effective_tier_role_binds_over_no_pin() -> None:
    entry = {"name": "a", "role": "operator"}
    assert effective_tier(entry, {"operator": "readwrite"}, "readonly") == "readwrite"


def test_effective_tier_refuses_a_role_beside_a_tier_pin() -> None:
    entry = {"name": "a", "role": "operator", "tier": "admin"}
    with pytest.raises(UnresolvedRoleError, match="tier 'admin' and role 'operator'"):
        effective_tier(entry, {"operator": "readwrite"}, "readonly")
    # Lenient: the pre-roles answer, which lint pairs with its own error.
    assert effective_tier(entry, {"operator": "readwrite"}, "readonly", strict=False) == "admin"


# ── The app branch ───────────────────────────────────────────────────────────


def test_each_shared_app_resolves_to_its_own_render() -> None:
    # Arrange
    subtree = _subtree(shared=["logbook", "knowledge"])

    # Act
    apps = _apps(resolve_cards(subtree, {}, project_name=PROJECT, shared_titles=TITLES))

    # Assert
    project, project_path = render_project(PROJECT, "logbook")
    assert project_path == f"build/{PROJECT}-logbook"
    assert apps[0] == {
        "kind": "app",
        "app": "logbook",
        "name": "logbook",
        "index": INDEX_MAX,
        "render": "logbook",
        "image": f"{project}:local",
        "project": project,
        "container_project_dir": f"/app/{project}",
        "extra_mounts": [],
        "seed_base": True,
        "display_name": "Logbook Research",
        "access": "any",
    }
    assert apps[1]["index"] == INDEX_MAX - 1
    assert apps[1]["display_name"] == "Facility Knowledge"
    assert apps[1]["project"] == f"{PROJECT}-knowledge"


def test_an_untitled_app_is_shown_under_its_name() -> None:
    apps = _apps(resolve_cards(_subtree(shared=["logbook"]), {}, project_name=PROJECT))
    assert apps[0]["display_name"] == "logbook"


def test_app_image_in_registry_mode_is_named_by_the_app() -> None:
    subtree = _subtree(shared=["logbook"], image_source="registry", image_tag="1.2")
    apps = _apps(resolve_cards(subtree, {"url": "reg.example/osprey"}, project_name=PROJECT))
    assert apps[0]["image"] == "reg.example/osprey/web-terminal-logbook:1.2"
    assert apps[0]["container_project_dir"] == f"/app/{PROJECT}-logbook"


def test_app_carries_its_catalog_mounts_and_seed_base() -> None:
    catalog = {
        "logbook": tier_render(
            "logbook",
            f"{PROJECT}-logbook",
            extra_mounts=["/srv/logbook:/data:ro", 3],
            seed_base=False,
        )
    }
    subtree = _subtree(shared=["logbook"], renders=catalog)
    apps = _apps(resolve_cards(subtree, {}, project_name=PROJECT))
    assert apps[0]["extra_mounts"] == ["/srv/logbook:/data:ro"]
    assert apps[0]["seed_base"] is False


def test_without_project_name_the_render_is_read_off_the_catalog() -> None:
    renders = {"logbook": {"kind": "app", "project": f"{PROJECT}-logbook"}}
    subtree = _subtree(shared=["logbook"], renders=renders)
    apps = _apps(resolve_cards(subtree, {}))
    assert apps[0]["project"] == f"{PROJECT}-logbook"
    assert apps[0]["image"] == f"{PROJECT}-logbook:local"


def test_app_with_no_nameable_render_is_refused_under_strict() -> None:
    subtree = _subtree(shared=["logbook"])
    with pytest.raises(ValueError, match="shared app 'logbook' has no render"):
        resolve_cards(subtree, {})


def test_app_with_no_nameable_render_is_unresolved_when_lenient() -> None:
    apps = _apps(resolve_cards(_subtree(shared=["logbook"]), {}, strict=False))
    assert apps[0]["name"] == "logbook"
    assert apps[0]["image"] is None
    assert apps[0]["project"] is None
    assert apps[0]["access"] == "any"


def test_apps_resolve_through_a_lenient_caller_too() -> None:
    apps = _apps(
        resolve_cards(_subtree(shared=["logbook"]), {}, strict=False, project_name=PROJECT)
    )
    assert apps[0]["project"] == f"{PROJECT}-logbook"


# ── A name is a tier or an app, never both ───────────────────────────────────


def test_naming_refuses_a_tier_that_is_a_shared_login() -> None:
    problem = render_kind_conflict("logbook", RENDER_KIND_TIER, shared={"logbook"})
    assert problem is not None
    assert "modules.web_terminals.shared" in problem
    with pytest.raises(ValueError, match="never both"):
        render_project(PROJECT, "logbook", kind=RENDER_KIND_TIER, shared={"logbook"})


def test_naming_refuses_a_shared_login_that_is_a_tier() -> None:
    assert render_kind_conflict("admin", RENDER_KIND_SHARED, tiers={"admin"}) is not None
    with pytest.raises(ValueError, match="never both"):
        render_project(PROJECT, "admin", kind=RENDER_KIND_SHARED, tiers={"admin"})


def test_naming_accepts_distinct_names_and_unchecked_calls() -> None:
    assert render_kind_conflict("admin", RENDER_KIND_TIER, shared={"logbook"}) is None
    assert render_kind_conflict("logbook", RENDER_KIND_SHARED, tiers={"admin"}) is None
    checked = render_project(PROJECT, "logbook", kind=RENDER_KIND_SHARED, tiers={"admin"})
    assert checked == render_project(PROJECT, "logbook")


def test_a_person_whose_tier_is_an_app_is_refused() -> None:
    users = [*PEOPLE, {"name": "dana", "index": 3, "tier": "logbook"}]
    subtree = _subtree(users=users, shared=["logbook"])
    with pytest.raises(ValueError, match="'dana'.*'logbook' is used as a tier"):
        resolve_cards(subtree, {}, project_name=PROJECT)


def test_a_person_whose_tier_is_an_app_is_unresolved_when_lenient() -> None:
    users = [*PEOPLE, {"name": "dana", "index": 3, "tier": "logbook"}]
    subtree = _subtree(users=users, shared=["logbook"])
    resolved = {
        entry["name"]: entry
        for entry in _people(resolve_cards(subtree, {}, strict=False, project_name=PROJECT))
    }
    assert resolved["dana"]["image"] is None
    assert resolved["dana"]["render"] == "logbook"


def test_a_default_tier_that_is_an_app_is_refused() -> None:
    subtree = _subtree(default_tier="logbook", shared=["logbook"])
    with pytest.raises(ValueError, match="never both"):
        resolve_cards(subtree, {}, project_name=PROJECT)


def test_an_app_named_like_a_role_tier_is_refused() -> None:
    authorization = {"roles": {"operator": {"tier": "readwrite"}}}
    subtree = _subtree(shared=["readwrite"], authorization=authorization)
    with pytest.raises(ValueError, match="never both"):
        resolve_cards(subtree, {}, project_name=PROJECT)


def test_a_shared_login_rendered_from_tiers_resolves() -> None:
    """Every login file renders a ``kind: tier`` projection entry, the shared
    ones included: being projected is no use as a tier, so it is no collision."""
    renders = {"knowledge": {"kind": "tier", "project": f"{PROJECT}-knowledge"}}
    subtree = _subtree(shared=["knowledge"], renders=renders)

    apps = _apps(resolve_cards(subtree, {}, project_name=PROJECT))

    assert apps[0]["project"] == f"{PROJECT}-knowledge"


def test_a_shared_login_a_role_uses_as_a_tier_is_refused() -> None:
    subtree = _subtree(
        shared=["knowledge"],
        authorization={"roles": {"reader": {"tier": "knowledge"}}},
    )
    with pytest.raises(ValueError, match="shared login 'knowledge'.*also used as a tier"):
        resolve_cards(subtree, {}, project_name=PROJECT)
    # Lenient: the card stays on the roster, unresolved.
    apps = _apps(resolve_cards(subtree, {}, strict=False, project_name=PROJECT))
    assert apps[0]["image"] is None


# ── Derived entries are never written ────────────────────────────────────────


def test_a_built_repo_lists_people_only_in_config_and_profile(
    tmp_path: Path, built_control_assistant
) -> None:
    # Arrange: the run's shared build (tests/_builds.py), whose shipped roster
    # lists people under users: and the apps under shared:, and the profile a
    # fresh init of the same name writes, which that build started from.
    from osprey.cli.init_cmd import init

    target = built_control_assistant.repo
    fresh = tmp_path / target.name
    created = CliRunner().invoke(init, [str(fresh), "--preset", "control-assistant", "--no-git"])
    assert created.exit_code == 0, created.output
    profile_before = (fresh / "profile.yml").read_text(encoding="utf-8")

    # Assert: the build left the profile as init wrote it, and both files
    # list people only.
    assert (target / "profile.yml").read_text(encoding="utf-8") == profile_before
    profile_roster = yaml.safe_load(profile_before)["config"]["modules.web_terminals"]
    config = yaml.safe_load((target / "build" / "config.yml").read_text(encoding="utf-8"))
    web_terminals = config["modules"]["web_terminals"]
    people = ["alice", "bob", "carol"]
    for roster in (profile_roster, web_terminals):
        assert [entry["name"] for entry in roster["users"]] == people
        assert all("kind" not in entry for entry in roster["users"])
        assert roster["shared"] == ["logbook", "knowledge", "channel-finder"]

    # And resolving the built config derives the app cards in memory.
    titles = {**TITLES, "channel-finder": "Channel Finder"}
    resolved = resolve_cards(
        web_terminals, {}, strict=False, project_name=config["project_name"], shared_titles=titles
    )
    apps = _apps(resolved)
    assert [entry["name"] for entry in _people(resolved)] == people
    assert [(entry["name"], entry["project"], entry["display_name"]) for entry in apps] == [
        ("logbook", f"{config['project_name']}-logbook", "Logbook Research"),
        ("knowledge", f"{config['project_name']}-knowledge", "Facility Knowledge"),
        ("channel-finder", f"{config['project_name']}-channel-finder", "Channel Finder"),
    ]
    rewritten = yaml.safe_load((target / "build" / "config.yml").read_text(encoding="utf-8"))
    assert [entry["name"] for entry in rewritten["modules"]["web_terminals"]["users"]] == people
