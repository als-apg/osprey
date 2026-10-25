"""An app render holds no deployment-editing privilege, whatever the host sets.

One door table (:data:`~osprey.deployment.web_terminals.roster.DEPLOYMENT_EDIT_DOORS`)
names every way a render can edit the deployment it runs in. The composer closes
every door, :func:`~osprey.deployment.web_terminals.lint.lint_app_render` checks
every door, and init, lint and the build read a render's privileges off one
per-kind document, :func:`~osprey.cli.build_profile_merge.render_document`.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from osprey.cli.build_profile_merge import compose_app_render, render_document
from osprey.cli.profile_root import ProfileKind
from osprey.deployment.web_terminals import lint
from osprey.deployment.web_terminals.lint import lint_app_render, lint_profile_config
from osprey.deployment.web_terminals.roster import (
    ALL_PRIVILEGES,
    DEPLOYMENT_EDIT_DOORS,
    PRIVILEGE_CONFIG_PANEL,
    PRIVILEGE_SCAFFOLD_WRITES,
    PRIVILEGE_SETUP_TOOL,
    tier_privileges,
)
from osprey_connectors.errors import BuildProfileError

_SETUP_PATCH = "mcp__osprey_workspace__setup_patch"
_CODE = "web_terminals.app_render_holds_privilege"
_PANEL = "web.config_panel.enabled"
_GALLERY = "web.scaffold_gallery.write_enabled"
_DENY = "claude_code.permissions.deny"
_SHIPPED_APPS = ("logbook", "knowledge", "channel-finder")


def _pinned_render(**config: Any) -> dict[str, Any]:
    """A composed-looking app render: every write switch and door closed."""
    base: dict[str, Any] = {
        "control_system.writes_enabled": False,
        _DENY: [_SETUP_PATCH],
        _PANEL: False,
        _GALLERY: False,
    }
    base.update(config)
    return {"hooks": ["writes-check"], "config": base}


def _codes(findings: list[Any]) -> list[str]:
    return [finding.code for finding in findings if finding.severity == "error"]


# ── the one door table ──────────────────────────────────────────────────────


def test_the_door_table_lists_the_three_deployment_editing_doors() -> None:
    rows = [(door.key, door.closed, door.privilege) for door in DEPLOYMENT_EDIT_DOORS]

    assert rows == [
        (_DENY, _SETUP_PATCH, PRIVILEGE_SETUP_TOOL),
        (_PANEL, False, PRIVILEGE_CONFIG_PANEL),
        (_GALLERY, False, PRIVILEGE_SCAFFOLD_WRITES),
    ]
    assert ALL_PRIVILEGES == (
        PRIVILEGE_SETUP_TOOL,
        PRIVILEGE_CONFIG_PANEL,
        PRIVILEGE_SCAFFOLD_WRITES,
    )


@pytest.mark.parametrize(
    "layer",
    [
        {_GALLERY: True},
        {"web": {"scaffold_gallery": {"write_enabled": True}}},
        {"web.scaffold_gallery": {"write_enabled": "yes"}},
        {},  # unmentioned: the server's default is ON
    ],
)
def test_a_tier_with_gallery_writes_on_holds_the_third_privilege(layer: dict) -> None:
    floored = {_DENY: [_SETUP_PATCH], _PANEL: False}

    assert tier_privileges(floored, layer) == (PRIVILEGE_SCAFFOLD_WRITES,)


def test_a_later_layer_closing_gallery_writes_wins() -> None:
    held = tier_privileges(
        {_GALLERY: True}, {"web": {"scaffold_gallery": {"write_enabled": False}}}
    )

    assert PRIVILEGE_SCAFFOLD_WRITES not in held


# ── the composer closes every door ──────────────────────────────────────────


def test_a_host_and_app_opening_every_door_compose_to_no_privilege() -> None:
    host = {
        "config": {
            _DENY: [],
            "web": {"config_panel": {"enabled": True}},
        }
    }
    app = {"name": "Logbook", "config": {_GALLERY: True}}

    render = compose_app_render(host, app, "apps/logbook.yml")

    assert tier_privileges(render["config"]) == ()
    assert render["config"][_PANEL] is False
    assert render["config"][_GALLERY] is False
    assert _SETUP_PATCH in render["config"][_DENY]


def test_the_host_floor_denies_survive_the_door_pin() -> None:
    host = {"config": {_DENY: ["mcp__other__tool"]}}

    render = compose_app_render(host, {"name": "Logbook"}, "apps/logbook.yml")

    assert render["config"][_DENY] == ["mcp__other__tool", _SETUP_PATCH]


def test_an_app_that_lifts_the_setup_deny_is_refused_with_the_new_code_message() -> None:
    app = {"name": "Logbook", "config": {"claude_code.permissions.remove_deny": [_SETUP_PATCH]}}

    with pytest.raises(BuildProfileError, match="must not be able to write or edit the deployment"):
        compose_app_render({"config": {}}, app, "apps/logbook.yml")


# ── lint_app_render checks every door ───────────────────────────────────────


def test_a_render_with_every_door_closed_passes() -> None:
    assert lint_app_render(_pinned_render()) == []


@pytest.mark.parametrize(
    ("render", "named"),
    [
        (_pinned_render(**{_DENY: []}), _SETUP_PATCH),
        (_pinned_render(**{_PANEL: True}), _PANEL),
        (_pinned_render(**{_GALLERY: True}), _GALLERY),
        (_pinned_render(**{_PANEL: "false"}), _PANEL),
    ],
    ids=["setup-tool", "config-panel", "gallery-writes", "panel-quoted-false"],
)
def test_one_open_door_is_refused(render: dict, named: str) -> None:
    findings = lint_app_render(render)

    assert _codes(findings) == [_CODE]
    assert named in findings[0].message


@pytest.mark.parametrize("door", [_PANEL, _GALLERY])
def test_a_door_no_spelling_pins_is_refused(door: str) -> None:
    render = _pinned_render()
    del render["config"][door]

    findings = lint_app_render(render)

    assert _codes(findings) == [_CODE]
    assert f"{door} is not pinned to false" in findings[0].message


def test_a_second_open_spelling_beside_the_pin_is_refused() -> None:
    render = _pinned_render(web={"config_panel": {"enabled": True}})

    findings = lint_app_render(render)

    assert _codes(findings) == [_CODE]
    assert PRIVILEGE_CONFIG_PANEL in findings[0].message


def test_a_scalar_at_the_door_section_is_refused() -> None:
    render = _pinned_render(**{"web.scaffold_gallery": True})

    findings = lint_app_render(render)

    assert _codes(findings) == [_CODE]
    assert "web.scaffold_gallery is True" in findings[0].message


def test_a_hand_edited_render_with_the_panel_on_is_refused() -> None:
    render = compose_app_render({"config": {}}, {"name": "Logbook"}, "apps/logbook.yml")
    render["config"][_PANEL] = True

    findings = lint_app_render(render)

    assert _codes(findings) == [_CODE]
    assert findings[0].message.startswith(
        "An app render must not be able to write or edit the deployment, and this one can: "
    )


# ── one per-kind render document ────────────────────────────────────────────


def test_render_document_composes_an_app_and_merges_a_tier() -> None:
    host = {"config": {_DENY: [], _PANEL: True}, "hooks": ["target-state"]}
    layer = {"name": "Layer", "config": {_GALLERY: True}}

    app = render_document(ProfileKind.APP, copy.deepcopy(host), copy.deepcopy(layer), "apps/x.yml")
    tier = render_document(
        ProfileKind.TIER, copy.deepcopy(host), copy.deepcopy(layer), "tiers/x.yml"
    )

    assert tier_privileges(app["config"]) == ()
    assert "writes-check" in app["hooks"]
    assert tier_privileges(tier["config"]) == ALL_PRIVILEGES


# ── lint at profile altitude reads apps as the composer builds them ─────────


def _shipped() -> tuple[dict[str, Any], Path]:
    from osprey.cli.build_profile import resolve_build_profile

    resolved, profile_dir = resolve_build_profile(None, "control-assistant")
    return copy.deepcopy(resolved.config), Path(profile_dir)


def _open_every_host_door(config: dict[str, Any]) -> dict[str, Any]:
    config[_DENY] = []
    config[_PANEL] = True
    config[_GALLERY] = True
    return config


def test_profile_lint_on_an_open_host_reports_no_app_but_still_a_shared_person(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, profile_root = _shipped()
    _open_every_host_door(config)
    for user in config["modules.web_terminals"]["users"]:
        if user["name"] == "carol":
            user["access"] = "any"
    read: dict[str, Any] = {}
    real = lint._render_documents

    def spy(*args: Any, **kwargs: Any) -> Any:
        documents, unreadable = real(*args, **kwargs)
        read.update(documents=documents, unreadable=unreadable)
        return documents, unreadable

    monkeypatch.setattr(lint, "_render_documents", spy)

    findings = lint_profile_config(config, profile_root=profile_root)

    shared = [f.message for f in findings if f.code == "web_terminals.shared_card_privileged"]
    for app in _SHIPPED_APPS:
        assert app in read["documents"], f"lint never read app {app!r}"
        assert app not in read["unreadable"]
        assert tier_privileges(*read["documents"][app]) == ()
        assert not any(f"{app!r}" in message for message in shared)
    assert len(shared) == 1
    assert "'carol'" in shared[0]


def test_the_shared_card_rule_names_no_app_entry_with_every_host_door_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, profile_root = _shipped()
    _open_every_host_door(config)
    seen: dict[str, Any] = {}
    real = lint.shared_card_privileged_problems

    def spy(resolved: Any, absolute: Any) -> list[str]:
        seen.update(resolved=list(resolved), absolute=dict(absolute))
        return real(seen["resolved"], absolute)

    monkeypatch.setattr(lint, "shared_card_privileged_problems", spy)
    lint_profile_config(config, profile_root=profile_root)

    app_entries = [e for e in seen["resolved"] if e.get("name") in _SHIPPED_APPS]
    assert {e["name"] for e in app_entries} == set(_SHIPPED_APPS)
    problems = real(seen["resolved"], seen["absolute"])
    for app in _SHIPPED_APPS:
        assert not any(f"user {app!r}" in problem for problem in problems)


# ── a --set edit cannot arm an app render ───────────────────────────────────


@pytest.mark.parametrize(
    "pair",
    [
        "config.control_system.writes_enabled=true",
        "config.control_system.connector.epics.writes_enabled=true",
        "config.control_system={writes_enabled: true}",
        f"config.{_PANEL}=true",
        f"config.{_GALLERY}=true",
    ],
)
def test_a_set_edit_on_an_app_build_stays_pinned(pair: str, tmp_path: Path) -> None:
    """The edit lands after the composer pinned the render, so the build seals it again."""
    import yaml

    from osprey.cli.build_profile_resolve import resolve_build_document

    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True)
    (repo / "profile.yml").write_text(
        yaml.safe_dump(
            {"name": "Facility", "data": "data", "config": {"control_system.writes_enabled": True}}
        ),
        encoding="utf-8",
    )
    app = repo / "apps" / "logbook.yml"
    app.parent.mkdir()
    app.write_text(
        yaml.safe_dump(
            {
                "name": "Logbook",
                "claude_md_template": "CLAUDE.logbook.md.j2",
                "hooks": [],
                "config": {"claude_code.servers.controls.enabled": False},
            }
        ),
        encoding="utf-8",
    )

    loaded = resolve_build_document(app, None, (), (pair,))

    render = {"hooks": loaded.profile.hooks, "config": loaded.profile.config}
    assert lint_app_render(render) == []
    assert loaded.profile.config["control_system.writes_enabled"] is False
