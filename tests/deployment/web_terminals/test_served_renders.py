"""The served render set: one answer to "which renders does this deployment serve".

Every card's render, ``default_tier``, every role's tier and the dispatch tier
— built, verified, granted and linted from one list,
:func:`~osprey.deployment.web_terminals.roster.served_renders`.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from osprey.cli.build_profile_deploy import build_lint_view, deploy_aware_config_errors
from osprey.deployment.web_terminals.lint import lint_web_terminals
from osprey.deployment.web_terminals.roster import (
    ServedRender,
    dispatch_tier_name,
    served_render_names,
    served_renders,
)

_SRC = Path(__file__).resolve().parents[3] / "src"


def _five_sources() -> dict[str, Any]:
    """A roster serving one render through each of the five ways a render is served."""
    return {
        "enabled": True,
        "default_tier": "viewer",
        "users": [
            {"name": "alice", "index": 0, "tier": "operator"},
            {"name": "bob", "index": 1, "role": "op"},
        ],
        "shared": ["logbook"],
        "authorization": {
            "roles": {"op": {"tier": "operator"}, "expert": {"tier": "physicist"}},
        },
    }


def _projection(*tiers: str, apps: tuple[str, ...] = ()) -> dict[str, Any]:
    renders = {
        name: {"kind": "tier", "project": f"demo-{name}", "project_path": f"build/demo-{name}"}
        for name in tiers
    }
    renders.update(
        {
            name: {"kind": "tier", "project": f"demo-{name}", "project_path": f"build/demo-{name}"}
            for name in apps
        }
    )
    return renders


# ── the served set ──────────────────────────────────────────────────────────


def test_every_source_serves_its_render_with_its_kind() -> None:
    """A person on a pinned tier, a role-only tier nobody holds, an app, a
    default-only tier and a dispatch-only tier: all five, each with its kind."""
    # Arrange
    web_terminals = _five_sources()

    # Act
    served = served_render_names(web_terminals, "headless")

    # Assert
    assert served == {
        "operator": "tier",
        "viewer": "tier",
        "logbook": "shared",
        "physicist": "tier",
        "headless": "tier",
    }


def test_names_come_cards_then_default_then_roles_then_dispatch() -> None:
    """The order is part of the contract: a reader that wants it stable can
    sort, one that reports in roster order need not re-derive it."""
    # Arrange
    web_terminals = _five_sources()

    # Act
    names = list(served_render_names(web_terminals, "headless"))

    # Assert: alice/bob's operator, then the app card, then the default,
    # then the role nobody holds, then the dispatch tier.
    assert names == ["operator", "logbook", "viewer", "physicist", "headless"]


def test_a_person_with_no_tier_serves_the_default_once() -> None:
    # Arrange
    web_terminals = {"enabled": True, "default_tier": "viewer", "users": [{"name": "a"}]}

    # Act / Assert
    assert served_render_names(web_terminals, "viewer") == {"viewer": "tier"}


def test_a_name_both_a_tier_and_shared_is_a_tier() -> None:
    """The collision is reported by the refusal that a card renders from one
    file, never hidden by the served set picking the app."""
    # Arrange
    web_terminals = {
        "default_tier": "viewer",
        "users": [{"name": "a", "index": 0, "tier": "logbook"}],
        "shared": ["logbook"],
    }

    # Act / Assert
    assert served_render_names(web_terminals, None)["logbook"] == "tier"


def test_no_dispatch_tier_serves_nothing_extra() -> None:
    # Arrange
    web_terminals = {"default_tier": "viewer", "users": []}

    # Act / Assert
    assert served_render_names(web_terminals, None) == {"viewer": "tier"}
    assert served_render_names(web_terminals, "") == {"viewer": "tier"}


def test_strict_refuses_an_undeclared_role() -> None:
    # Arrange
    web_terminals = _five_sources()
    web_terminals["users"][1]["role"] = "admin"

    # Act / Assert
    with pytest.raises(ValueError, match="admin"):
        served_render_names(web_terminals, None, strict=True)


def test_lenient_degrades_an_undeclared_role_to_the_default() -> None:
    # Arrange
    web_terminals = _five_sources()
    web_terminals["users"][1]["role"] = "admin"

    # Act
    served = served_render_names(web_terminals, None)

    # Assert
    assert "viewer" in served and "physicist" in served


def test_lenient_reads_an_unparseable_authorization_as_no_roles() -> None:
    # Arrange
    web_terminals = _five_sources()
    web_terminals["authorization"] = {"roles": {"op": "not-a-mapping"}}
    web_terminals["users"] = [{"name": "alice", "index": 0, "tier": "operator"}]

    # Act
    served = served_render_names(web_terminals, None)

    # Assert
    assert served == {"operator": "tier", "logbook": "shared", "viewer": "tier"}
    with pytest.raises(ValueError):
        served_render_names(web_terminals, None, strict=True)


# ── joined to the projection ────────────────────────────────────────────────


def _rendered(web_terminals: dict[str, Any], *, dispatch: str | None = "headless") -> dict:
    config: dict[str, Any] = {"modules": {"web_terminals": web_terminals}}
    if dispatch is not None:
        config["services"] = {"dispatch_worker": {"tier": dispatch}}
    return config


def test_served_renders_joins_each_name_to_its_projection_entry() -> None:
    # Arrange
    web_terminals = _five_sources()
    web_terminals["renders"] = _projection(
        "operator", "viewer", "physicist", "headless", apps=("logbook",)
    )

    # Act
    served = served_renders(_rendered(web_terminals))

    # Assert
    assert set(served) == {"operator", "viewer", "physicist", "headless", "logbook"}
    logbook = served["logbook"]
    assert isinstance(logbook, ServedRender)
    assert logbook.kind == "shared"
    assert logbook.project == "demo-logbook"
    assert logbook.project_path == "build/demo-logbook"
    assert logbook.entry is web_terminals["renders"]["logbook"]
    assert served["headless"] is not None and served["headless"].kind == "tier"


def test_a_name_the_projection_lacks_maps_to_none() -> None:
    # Arrange
    web_terminals = _five_sources()
    web_terminals["renders"] = _projection("operator", "viewer", apps=("logbook",))

    # Act
    served = served_renders(_rendered(web_terminals))

    # Assert
    assert served["physicist"] is None
    assert served["headless"] is None
    assert served["operator"] is not None


def test_an_entry_without_a_project_path_is_served_with_none() -> None:
    # Arrange
    web_terminals = {"default_tier": "viewer", "renders": {"viewer": {"kind": "tier"}}}

    # Act
    (render,) = served_renders(_rendered(web_terminals, dispatch=None)).values()

    # Assert
    assert render is not None
    assert render.project_path is None and render.project is None


def test_dispatch_tier_name_reads_the_worker_block() -> None:
    assert dispatch_tier_name({"services": {"dispatch_worker": {"tier": "ro"}}}) == "ro"
    assert dispatch_tier_name({"services": {"dispatch_worker": {"tier": ""}}}) is None
    assert dispatch_tier_name({"services": {"dispatch_worker": None}}) is None
    assert dispatch_tier_name({}) is None


# ── lint reads the served set ───────────────────────────────────────────────


def _lint_config(**web_terminals: Any) -> dict[str, Any]:
    stanza: dict[str, Any] = {
        "enabled": True,
        "image_source": "local",
        "default_tier": "viewer",
        "users": [{"name": "alice", "index": 0}],
        "renders": _projection("viewer"),
    }
    stanza.update(web_terminals)
    return {"project_name": "demo", "modules": {"web_terminals": stanza}}


def _codes(findings) -> list[str]:
    return [f.code for f in findings if f.severity == "error"]


def test_lint_holds_a_shared_render_to_the_project_path_rule(tmp_path: Path) -> None:
    """A shared login's render is judged at rendered altitude like a tier: no
    ``project_path`` in local mode is the same refusal."""
    # Arrange
    renders = _projection("viewer")
    renders["logbook"] = {"kind": "tier", "project": "demo-logbook"}
    config = _lint_config(shared=["logbook"], renders=renders)

    # Act
    findings = lint_web_terminals(config, project_root=tmp_path)

    # Assert
    missing = [f for f in findings if f.code == "web_terminals.render_missing_project_path"]
    assert [m for m in missing if "'logbook'" in m.message]


def test_lint_refuses_an_app_name_as_default_tier() -> None:
    """An app carries no access level, so it never stands in for a tier."""
    # Arrange
    renders = _projection("viewer", apps=("logbook",))
    config = _lint_config(default_tier="logbook", shared=["logbook"], renders=renders)

    # Act
    codes = _codes(lint_web_terminals(config, rendered_project=False))

    # Assert
    assert "web_terminals.unknown_default_tier" in codes


def _unknown_messages(config: dict[str, Any]) -> list[str]:
    return [
        f.message
        for f in lint_web_terminals(config, rendered_project=False)
        if f.code == "web_terminals.unknown_tier_reference"
    ]


def test_unknown_tier_reported_against_the_user_who_pins_it() -> None:
    # Arrange
    config = _lint_config(users=[{"name": "alice", "index": 0, "tier": "ghost"}])

    # Act
    messages = _unknown_messages(config)

    # Assert
    assert len(messages) == 1
    assert "user 'alice'" in messages[0] and "tiers/ghost.yml" in messages[0]


def test_unknown_tier_reported_against_default_tier() -> None:
    # Arrange
    config = _lint_config(default_tier="ghost")

    # Act
    messages = _unknown_messages(config)

    # Assert
    assert len(messages) == 1
    assert "modules.web_terminals.default_tier" in messages[0]
    assert "'ghost'" in messages[0]


def test_unknown_tier_reported_against_the_role_that_binds_it() -> None:
    """A role nobody holds is served, so its tier is checked; a person carrying
    the role is not reported a second time."""
    # Arrange
    config = _lint_config(
        users=[{"name": "alice", "index": 0, "role": "op"}],
        authorization={"roles": {"op": {"tier": "ghost"}, "idle": {"tier": "phantom"}}},
    )

    # Act
    messages = _unknown_messages(config)

    # Assert
    assert len(messages) == 2
    assert any("authorization.roles.op.tier" in m and "'ghost'" in m for m in messages)
    assert any("authorization.roles.idle.tier" in m and "'phantom'" in m for m in messages)


def test_unknown_tier_reported_against_dispatch_tier() -> None:
    # Arrange
    config = _lint_config()
    config["services"] = {"dispatch_worker": {"tier": "ghost"}}

    # Act
    messages = _unknown_messages(config)

    # Assert
    assert len(messages) == 1
    assert messages[0].startswith("dispatch.tier references tier 'ghost'")


def test_known_role_and_dispatch_tiers_are_clean() -> None:
    # Arrange
    config = _lint_config(
        renders=_projection("viewer", "operator", "headless"),
        authorization={"roles": {"op": {"tier": "operator"}}},
    )
    config["services"] = {"dispatch_worker": {"tier": "headless"}}

    # Act / Assert
    assert _unknown_messages(config) == []


# ── profile altitude learns the dispatch tier ───────────────────────────────


def test_build_lint_view_writes_the_dispatch_tier_where_the_build_does() -> None:
    # Act
    view = build_lint_view(None, {"x": 1}, None, dispatch_tier="readonly")

    # Assert
    assert view["services.dispatch_worker.tier"] == "readonly"
    assert "services.dispatch_worker.tier" not in build_lint_view(None, {"x": 1}, None)


def test_profile_validation_names_a_dispatch_tier_with_no_tier_file(tmp_path: Path) -> None:
    """The proof the profile gate sees ``dispatch.tier``: a tier missing from
    ``tiers/`` is refused against that key."""
    # Arrange
    (tmp_path / "tiers").mkdir()
    (tmp_path / "tiers" / "viewer.yml").write_text("name: viewer\nconfig: {}\n")
    config = {
        "modules.web_terminals": {
            "enabled": True,
            "default_tier": "viewer",
            "users": [{"name": "alice", "index": 0}],
        }
    }

    # Act
    errors = deploy_aware_config_errors(
        None, config, profile_root=tmp_path, project_name="demo", dispatch_tier="ghost"
    )
    clean = deploy_aware_config_errors(
        None, config, profile_root=tmp_path, project_name="demo", dispatch_tier="viewer"
    )

    # Assert
    assert any(e.startswith("dispatch.tier references tier 'ghost'") for e in errors)
    assert not any("dispatch.tier" in e for e in clean)


# ── one definition ──────────────────────────────────────────────────────────

#: The one image-derivation projection over the served set, the start step
#: that wraps it, and the per-render directory walk over `served_renders`;
#: none decides membership.
_ALLOWED_REFERENCED_RENDER_DEFS = {
    ("osprey/deployment/web_terminals/render_images.py", "_referenced_renders"),
    ("osprey/deployment/container_lifecycle.py", "_build_referenced_renders"),
    ("osprey/deployment/web_terminals/roster.py", "referenced_render_project_dirs"),
}


def test_no_second_definition_of_the_served_set() -> None:
    """Every reader asks `served_renders`; no module grows its own walk."""
    # Arrange / Act
    found = set()
    for path in sorted((_SRC / "osprey").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                if "referenced_render" in node.name:
                    found.add((path.relative_to(_SRC).as_posix(), node.name))

    # Assert
    assert found == _ALLOWED_REFERENCED_RENDER_DEFS


# ── fail-closed readers of the served set ───────────────────────────────────


def test_a_served_tier_the_projection_lacks_counts_as_denying_nothing(tmp_path: Path) -> None:
    """A served name with no projection entry has no artifact to prove a deny,
    so the Bash guard counts it unsafe rather than skipping it."""
    # Arrange
    from osprey.deployment.web_terminals.roster import renders_with_open_shell

    web_terminals = {"default_tier": "ghost", "renders": _projection("viewer")}

    # Act
    unsafe = renders_with_open_shell(_rendered(web_terminals, dispatch=None), tmp_path)

    # Assert
    assert "ghost" in unsafe


def test_a_seal_refusal_travels_as_the_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A login file that reads cleanly but that the build refuses to seal is
    reported with the seal's reason, not as an unreadable file."""
    # Arrange
    import osprey.cli.build_profile_merge as merge
    from osprey.deployment.web_terminals import lint
    from osprey_connectors.errors import BuildProfileError

    (tmp_path / "tiers").mkdir()
    (tmp_path / "tiers" / "logbook.yml").write_text("lists: own\nconfig: {}\n", encoding="utf-8")

    def _refuse(_render: Any, rel: str) -> Any:
        raise BuildProfileError(f"{rel}: this app may not hold the door")

    monkeypatch.setattr(merge, "seal_shared_render", _refuse)

    # Act
    root = {"modules": {"web_terminals": {"shared": ["logbook"]}}}
    record = lint._profile_composed_layers(root, "tiers/logbook.yml", profile_root=tmp_path)

    # Assert
    assert isinstance(record, lint._UnreadableRender)
    clause = lint._unreadable_render_clause(record)
    assert "refused by the build" in clause
    assert "this app may not hold the door" in clause


def test_a_shared_login_on_a_host_that_opens_the_web_doors_holds_neither() -> None:
    """Lint judges shared renders as the build seals them, so the seal must
    close the Config panel and gallery writes whatever the host or login spells."""
    from osprey.cli.build_profile_merge import render_document
    from osprey.deployment.web_terminals.roster import PRIVILEGE_CONFIG_PANEL, tier_privileges

    host = {
        "config": {
            "web": {"config_panel": {"enabled": True}},
            "web.scaffold_gallery.write_enabled": True,
            "modules.web_terminals.shared": ["logbook"],
        }
    }
    app = {
        "name": "Logbook",
        "lists": "own",
        "config": {"web.scaffold_gallery": {"write_enabled": True}},
    }
    config = render_document(host, app, "tiers/logbook.yml")["config"]

    assert config["web.config_panel.enabled"] is False
    assert config["web.scaffold_gallery.write_enabled"] is False
    assert "web" not in config and "web.scaffold_gallery" not in config
    assert PRIVILEGE_CONFIG_PANEL not in tier_privileges(config)
