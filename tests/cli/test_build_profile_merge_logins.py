"""One login kind: how a ``tiers/<name>.yml`` login renders over its deployment.

Every login is a file in ``tiers/``. One that says ``lists: own`` takes its
lists from its own file alone and is laid over the host's facts; the host's
PreToolUse guards reach it whatever its hooks list says. A login the host's
``modules.web_terminals.shared`` names is sealed read-only — writes ``never``,
every deployment-edit door closed, ``writes-check`` selected — and a ``--set``
edit cannot unseal it. A narrow login nobody shares is never pinned: it reads
its tier like any login and keeps the host's write and edit-door switches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from osprey.cli.build_profile_load import load_profile_document
from osprey.cli.build_profile_merge import render_document
from osprey.cli.build_profile_resolve import resolve_build_document
from osprey_connectors.errors import BuildProfileError

_WRITES = "control_system.writes"
_PANEL = "web.config_panel.enabled"
_DENY = "claude_code.permissions.deny"
_SETUP_PATCH = "mcp__osprey_workspace__setup_patch"

_HOST: dict[str, Any] = {
    "name": "Facility",
    "data": "data",
    "model": "opus",
    "hooks": ["approval", "limits", "turn-state"],
    "skills": ["diagnose"],
    "config": {
        _WRITES: "allowed",
        _PANEL: True,
        "claude_code.hooks.SessionStart": ["hooks/note.py"],
        "claude_code.hooks.PreToolUse": ["hooks/guard.py"],
    },
}

_NARROW: dict[str, Any] = {
    "name": "Narrow",
    "lists": "own",
    "hooks": ["turn-state"],
    "skills": [],
    "config": {"web.ui_mode": "expert"},
}


def _write_yaml(path: Path, body: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return path


def _repo(tmp_path: Path, login: dict[str, Any], *, shared: bool, host: dict | None = None):
    """A repo with one login ``tiers/narrow.yml``; ``shared`` names it in ``shared:``."""
    root = dict(host or _HOST)
    config = dict(root.get("config") or {})
    if shared:
        config["modules.web_terminals.shared"] = ["narrow"]
    root["config"] = config
    (tmp_path / "data").mkdir(exist_ok=True)
    _write_yaml(tmp_path / "profile.yml", root)
    return _write_yaml(tmp_path / "tiers" / "narrow.yml", login)


# ── a narrow login nobody shares ─────────────────────────────────────────────


def test_a_narrow_login_silent_on_its_tier_inherits_the_roots() -> None:
    rendered = render_document(dict(_HOST), dict(_NARROW), "tiers/narrow.yml")

    assert rendered["config"][_WRITES] == "allowed"


def test_a_narrow_login_keeps_the_hosts_write_and_edit_door_switches() -> None:
    rendered = render_document(dict(_HOST), dict(_NARROW), "tiers/narrow.yml")

    assert rendered["config"][_PANEL] is True
    assert _SETUP_PATCH not in (rendered["config"].get(_DENY) or [])
    assert "writes-check" not in rendered["hooks"]


def test_a_narrow_login_owns_its_lists_and_keeps_the_host_guard_floor() -> None:
    rendered = render_document(dict(_HOST), dict(_NARROW), "tiers/narrow.yml")

    # Its lists are its own: the host's skills never reach it.
    assert rendered["skills"] == []
    # Library guards reach it whatever its list says; a non-guard host hook does not.
    assert {"approval", "limits", "turn-state"} <= set(rendered["hooks"])
    assert rendered["config"]["claude_code.hooks.PreToolUse"] == ["hooks/guard.py"]
    assert "claude_code.hooks.SessionStart" not in rendered["config"]


def test_a_narrow_login_omitting_an_owned_key_is_absent_rather_than_inherited() -> None:
    rendered = render_document(dict(_HOST), dict(_NARROW), "tiers/narrow.yml")

    assert "agents" not in rendered
    assert rendered["name"] == "Narrow"


# ── a login the host's shared: names ─────────────────────────────────────────


@pytest.mark.parametrize("lists", ["own", "add"])
def test_a_shared_login_is_sealed_whatever_its_lists(lists: str) -> None:
    host = {**_HOST, "config": {**_HOST["config"], "modules.web_terminals.shared": ["narrow"]}}

    rendered = render_document(host, {**_NARROW, "lists": lists}, "tiers/narrow.yml")

    assert rendered["config"][_WRITES] == "never"
    assert rendered["config"][_PANEL] is False
    assert _SETUP_PATCH in rendered["config"][_DENY]
    assert "writes-check" in rendered["hooks"]


def test_a_shared_login_authoring_allowed_is_refused_naming_shared() -> None:
    host = {**_HOST, "config": {**_HOST["config"], "modules.web_terminals.shared": ["narrow"]}}
    login = {**_NARROW, "config": {_WRITES: "allowed"}}

    with pytest.raises(BuildProfileError) as excinfo:
        render_document(host, login, "tiers/narrow.yml")

    message = str(excinfo.value)
    assert "tiers/narrow.yml" in message
    assert "modules.web_terminals.shared" in message


@pytest.mark.usefixtures("facility_data_tree")
def test_a_set_edit_cannot_unseal_a_shared_login(tmp_path: Path) -> None:
    login = _repo(tmp_path, _NARROW, shared=True)

    loaded = resolve_build_document(login, None, (), (f"config.{_WRITES}=allowed",))

    assert loaded.profile.config[_WRITES] == "never"


@pytest.mark.usefixtures("facility_data_tree")
def test_a_set_edit_on_a_narrow_login_nobody_shares_is_its_own(tmp_path: Path) -> None:
    login = _repo(tmp_path, {**_NARROW, "config": {_WRITES: "never"}}, shared=False)

    loaded = resolve_build_document(login, None, (), (f"config.{_WRITES}=allowed",))

    assert loaded.profile.config[_WRITES] == "allowed"


# ── model ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("lists", ["own", "add"])
def test_a_login_naming_no_model_or_a_null_one_runs_the_hosts(lists: str) -> None:
    for login in ({"name": "x", "lists": lists}, {"name": "x", "lists": lists, "model": None}):
        rendered = render_document(dict(_HOST), dict(login), "tiers/x.yml")
        assert rendered["model"] == "opus", login


@pytest.mark.parametrize("lists", ["own", "add"])
def test_a_login_naming_a_model_runs_it(lists: str) -> None:
    rendered = render_document(
        dict(_HOST), {"name": "x", "lists": lists, "model": "haiku"}, "tiers/x.yml"
    )

    assert rendered["model"] == "haiku"


# ── lists is login-only, the root included ───────────────────────────────────


@pytest.mark.usefixtures("facility_data_tree")
def test_a_set_edit_spelling_lists_on_the_root_is_refused_like_the_file(tmp_path: Path) -> None:
    _repo(tmp_path, _NARROW, shared=False)

    with pytest.raises(BuildProfileError) as from_set:
        resolve_build_document(tmp_path / "profile.yml", None, (), ("lists=own",))

    root = yaml.safe_load((tmp_path / "profile.yml").read_text(encoding="utf-8"))
    _write_yaml(tmp_path / "profile.yml", {**root, "lists": "own"})
    with pytest.raises(BuildProfileError) as from_file:
        load_profile_document(tmp_path / "profile.yml")

    assert (
        str(from_set.value).split(" spells ", 1)[1] == str(from_file.value).split(" spells ", 1)[1]
    )
    assert "tiers/" in str(from_set.value)
