"""An application panel a roster card shows has a container holding its roster grant.

A card's terminal proxies each panel its render shows with the card's own
secret; an application panel whose container was handed none refuses every
such request. One predicate finds those panels, and the lint rule and the
compose render's refusal both speak its message.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from osprey.cli.build_cmd import _copy_service_templates
from osprey.deployment.compose_generator import prepare_compose_files
from osprey.deployment.web_terminals.lint import lint_web_terminals
from osprey.deployment.web_terminals.roster import (
    BLUESKY_PANEL_ID,
    app_panel_grant_missing,
    app_panels_without_roster_grant,
)
from tests._renders import tier_render

_RULE = "web_terminals.app_panel_without_roster_grant"

_KNOB = {"url": "http://localhost:1", "app": "knob"}


def _write_render(project: Path, name: str, panels: dict) -> dict:
    """Render one project showing *panels*; return its catalog entry."""
    render_dir = project / "build" / f"demo-{name}"
    render_dir.mkdir(parents=True, exist_ok=True)
    (render_dir / "config.yml").write_text(
        yaml.safe_dump({"web": {"panels": panels}}), encoding="utf-8"
    )
    return tier_render(name, f"demo-{name}", project_path=f"build/demo-{name}")


def _deployment(project: Path, operator_panels: dict, *, granted: bool = False) -> dict:
    """A roster of one person on ``operator``, one on ``viewer`` and one shared card.

    ``operator`` and the shared card ``radiation`` show *operator_panels*;
    ``viewer`` shows nothing.
    """
    services: dict = {"knob": {}}
    if granted:
        services["knob"]["roster_grant"] = {"panel_id": "knob"}
    return {
        "services": services,
        "modules": {
            "web_terminals": {
                "enabled": True,
                "default_tier": "viewer",
                "renders": {
                    "operator": _write_render(project, "operator", operator_panels),
                    "viewer": _write_render(project, "viewer", {}),
                    "radiation": _write_render(project, "radiation", operator_panels),
                },
                "users": [
                    {"name": "alice", "index": 0, "tier": "operator"},
                    {"name": "bob", "index": 1, "tier": "viewer"},
                ],
                "shared": ["radiation"],
            }
        },
    }


def test_a_card_showing_an_app_panel_with_no_grant_is_named(tmp_path: Path) -> None:
    config = _deployment(tmp_path, {"knob": _KNOB})

    assert app_panels_without_roster_grant(config, tmp_path) == [
        app_panel_grant_missing("alice", "knob", "knob"),
        app_panel_grant_missing("radiation", "knob", "knob"),
    ]


def test_the_message_names_the_card_the_app_the_panel_and_the_cure() -> None:
    message = app_panel_grant_missing("alice", "knob", "knob-panel")

    assert "'alice'" in message
    assert "'knob'" in message
    assert "'knob-panel'" in message
    assert "roster_grant: {}" in message


def test_a_granted_app_panel_is_not_named(tmp_path: Path) -> None:
    config = _deployment(tmp_path, {"knob": _KNOB}, granted=True)

    assert app_panels_without_roster_grant(config, tmp_path) == []


def test_a_builtin_panel_without_app_is_not_named(tmp_path: Path) -> None:
    config = _deployment(tmp_path, {"plain": {"url": "http://localhost:1"}})

    assert app_panels_without_roster_grant(config, tmp_path) == []


def test_a_panel_switched_off_is_not_named(tmp_path: Path) -> None:
    config = _deployment(tmp_path, {"knob": {**_KNOB, "enabled": False}})

    assert app_panels_without_roster_grant(config, tmp_path) == []


def test_a_card_whose_render_does_not_show_the_panel_is_not_named(tmp_path: Path) -> None:
    config = _deployment(tmp_path, {BLUESKY_PANEL_ID: {"url": "http://localhost:1"}})

    assert app_panels_without_roster_grant(config, tmp_path) == []


def test_a_deployment_with_no_roster_names_nothing(tmp_path: Path) -> None:
    config = _deployment(tmp_path, {"knob": _KNOB})
    config["modules"]["web_terminals"]["enabled"] = False

    assert app_panels_without_roster_grant(config, tmp_path) == []


def test_the_lint_reports_the_card_the_app_and_the_panel(tmp_path: Path) -> None:
    config = _deployment(tmp_path, {"knob": _KNOB})

    findings = [f for f in lint_web_terminals(config, project_root=tmp_path) if f.code == _RULE]

    assert [(f.severity, f.message) for f in findings] == [
        ("error", app_panel_grant_missing("alice", "knob", "knob")),
        ("error", app_panel_grant_missing("radiation", "knob", "knob")),
    ]


def test_the_lint_is_silent_on_a_granted_panel(tmp_path: Path) -> None:
    config = _deployment(tmp_path, {"knob": _KNOB}, granted=True)

    assert not [f for f in lint_web_terminals(config, project_root=tmp_path) if f.code == _RULE]


def test_the_render_refuses_with_the_lint_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _deployment(tmp_path, {"knob": _KNOB})
    config.update(
        {
            "project_name": "app-panel-grant-fixture",
            "project_root": str(tmp_path),
            "build_dir": str(tmp_path / "build"),
            "deployed_services": ["bluesky_web"],
            "system": {"timezone": "UTC"},
        }
    )
    config["services"].update(
        {"bluesky_web": {"path": "./services/bluesky_web", "port": 10071}, "bluesky": {"port": 1}}
    )
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    _copy_service_templates(tmp_path)
    monkeypatch.chdir(tmp_path)
    lint_messages = [
        f.message for f in lint_web_terminals(config, project_root=tmp_path) if f.code == _RULE
    ]

    with pytest.raises(ValueError) as refused:
        prepare_compose_files(str(config_path))

    assert str(refused.value).splitlines() == lint_messages
