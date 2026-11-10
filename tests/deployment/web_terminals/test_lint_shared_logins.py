"""Lint of the logins ``modules.web_terminals.shared`` names.

A shared login is a file in ``tiers/`` like any login. Lint reads it the way the
build renders it — sealed read-only — so a shared login whose own file arms
writes is the build's refusal, reported before any render, and one whose file
says nothing about writes holds no deployment-editing door whatever the host
opens. The landing section shared cards show in is ``type: shared``; the retired
``type: apps`` is read as it for one release, with one warning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from osprey.deployment.web_terminals.lint import lint_profile_config, lint_shared_render
from osprey_connectors.errors import BuildProfileError

_FLOOR = {
    "claude_code.permissions.deny": ["mcp__osprey_workspace__setup_patch"],
    "web.config_panel.enabled": False,
    "web.scaffold_gallery.write_enabled": False,
}


def _repo(tmp_path: Path, login: str, config: dict[str, Any] | None = None) -> Path:
    """A profile root: ``profile.yml`` carrying *config*, and two logins beside it."""
    import yaml

    (tmp_path / "tiers").mkdir()
    (tmp_path / "tiers" / "readonly.yml").write_text("name: ro\n", encoding="utf-8")
    (tmp_path / "tiers" / "logbook.yml").write_text(login, encoding="utf-8")
    profile = {"name": "x", "config": config if config is not None else _config()}
    (tmp_path / "profile.yml").write_text(yaml.safe_dump(profile), encoding="utf-8")
    return tmp_path


def _config(**web_terminals: Any) -> dict[str, Any]:
    return {
        "modules.web_terminals": {
            "enabled": True,
            "default_tier": "readonly",
            "users": [{"name": "alice", "index": 0}],
            "shared": ["logbook"],
            "auth": {"method": "password", "allow_insecure_http": True},
            **web_terminals,
        },
        **_FLOOR,
    }


def _errors_about(findings: list, name: str) -> list:
    return [f for f in findings if f.severity == "error" and f"'{name}'" in f.message]


def test_a_shared_login_arming_writes_is_refused_before_any_render(tmp_path: Path) -> None:
    repo = _repo(tmp_path, "name: Logbook\nlists: own\nconfig:\n  control_system.writes: allowed\n")

    findings = lint_profile_config(_config(), profile_root=repo)

    assert _errors_about(findings, "logbook"), findings


def test_a_shared_login_whose_retired_keys_armed_is_sealed_as_the_build_seals_it(
    tmp_path: Path,
) -> None:
    """The build warns once and pins ``never``; lint does not call it an authored ``allowed``."""
    repo = _repo(
        tmp_path, "name: Logbook\nlists: own\nconfig:\n  control_system.writes_enabled: true\n"
    )

    findings = lint_profile_config(_config(), profile_root=repo)

    assert _errors_about(findings, "logbook") == [], findings


def test_the_build_refuses_the_same_file_naming_shared() -> None:
    from osprey.cli.build_profile_merge import render_document

    host = {"config": {**_FLOOR, "modules.web_terminals.shared": ["logbook"]}}
    login = {"name": "Logbook", "lists": "own", "config": {"control_system.writes": "allowed"}}

    with pytest.raises(BuildProfileError, match="modules.web_terminals.shared"):
        render_document(host, login, "tiers/logbook.yml")


def test_a_shared_login_silent_on_writes_holds_no_door_on_an_open_host(tmp_path: Path) -> None:
    config = _config()
    config["web.config_panel.enabled"] = True
    config["control_system.writes"] = "allowed"
    repo = _repo(tmp_path, "name: Logbook\nlists: own\nhooks: []\n", config)

    findings = lint_profile_config(config, profile_root=repo)

    assert _errors_about(findings, "logbook") == [], findings


def test_a_hand_built_render_that_could_write_is_named_by_the_shared_check() -> None:
    findings = lint_shared_render({"hooks": ["writes-check"], "config": {**_FLOOR}})

    assert [f.code for f in findings] == ["web_terminals.shared_render_holds_privilege"]
    assert "control_system.writes is not pinned to never" in findings[0].message


def test_the_retired_apps_landing_section_warns_once_and_names_the_rewrite(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path, "name: Logbook\nlists: own\n")
    config = _config(landing={"groups": [{"type": "users"}, {"type": "apps"}]})

    findings = [
        f
        for f in lint_profile_config(config, profile_root=repo)
        if f.code == "web_terminals.retired_landing_group_type"
    ]

    assert len(findings) == 1
    assert findings[0].severity == "warn"
    assert "type: shared" in findings[0].message
    assert "osprey init --revalidate" in findings[0].message


def test_the_shared_landing_section_draws_no_warning(tmp_path: Path) -> None:
    repo = _repo(tmp_path, "name: Logbook\nlists: own\n")
    config = _config(landing={"groups": [{"type": "users"}, {"type": "shared"}]})

    codes = {f.code for f in lint_profile_config(config, profile_root=repo)}

    assert "web_terminals.retired_landing_group_type" not in codes


def test_a_person_named_like_a_shared_login_is_refused(tmp_path: Path) -> None:
    repo = _repo(tmp_path, "name: Logbook\nlists: own\n")
    config = _config(users=[{"name": "logbook", "index": 0}])

    codes = {f.code for f in lint_profile_config(config, profile_root=repo)}

    assert "web_terminals.user_named_like_app" in codes


def test_a_person_named_like_a_tier_is_not_a_shared_login_clash(tmp_path: Path) -> None:
    repo = _repo(tmp_path, "name: Logbook\nlists: own\n")
    config = _config(users=[{"name": "readonly", "index": 0}])

    codes = {f.code for f in lint_profile_config(config, profile_root=repo)}

    assert "web_terminals.user_named_like_app" not in codes
