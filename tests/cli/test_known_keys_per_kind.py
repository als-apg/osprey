"""Known profile keys are a function of the file's kind.

A root ``profile.yml`` may spell every key the schema defines. A tier
(``tiers/<name>.yml``) is a small delta over the deployment's own agent: the
root set minus every deployment key, plus the per-render keys. An app
(``apps/<name>.yml``) is a whole agent definition offered beside the
deployment's own. ``extends`` is legal in a root profile only. The loader,
``osprey set`` and ``osprey validate <file>`` judge a file against its kind's
set, and their refusal names the keys legal in that file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.build_profile_load import (
    _KNOWN_PROFILE_KEYS,
    APP_KIND_KEYS,
    RENDER_KEYS,
    ROOT_KIND_KEYS,
    ROOT_ONLY_KEYS,
    TIER_KIND_KEYS,
    _parse_profile,
    known_profile_keys,
    load_profile_document,
    reject_unknown_closed_config_keys,
    reject_unknown_kind_keys,
)
from osprey.cli.profile_root import ProfileKind
from osprey.utils.workspace import LAYER_DIR_KINDS, RETIRED_LAYER_DIRS
from osprey_connectors.errors import BuildProfileError

_DEPLOYMENT_KEYS = frozenset(
    {
        "extends",
        "data",
        "project_name",
        "deploy",
        "deploy_services",
        "services",
        "lifecycle",
        "env",
        "environment",
        "dependencies",
        "exclude",
        "agent_record",
    }
)

_EXPECTED_APP_KEYS = frozenset(
    {
        "name",
        "model",
        "hooks",
        "rules",
        "skills",
        "agents",
        "output_styles",
        "web_panels",
        "default_panel",
        "claude_md_template",
        "channel_finder_mode",
        "config",
        "mcp_servers",
        "extra_mounts",
        "seed_base",
        "image",
        "provenance",
    }
)


def _write_yaml(path: Path, body: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return path


def _repo(tmp_path: Path, root: dict[str, Any] | None = None) -> Path:
    """A deployment repo with a root profile and its data tree."""
    (tmp_path / "data").mkdir(exist_ok=True)
    _write_yaml(tmp_path / "profile.yml", root or {"name": "site", "data": "data"})
    return tmp_path


# ---------------------------------------------------------------------------
# The three key sets
# ---------------------------------------------------------------------------


def test_root_set_is_todays_schema() -> None:
    assert ROOT_KIND_KEYS == _KNOWN_PROFILE_KEYS
    assert known_profile_keys(ProfileKind.ROOT) == _KNOWN_PROFILE_KEYS


def test_tier_set_is_root_minus_deployment_keys_plus_render_keys() -> None:
    assert ROOT_ONLY_KEYS == _DEPLOYMENT_KEYS
    assert RENDER_KEYS == {"extra_mounts", "seed_base", "image"}
    assert known_profile_keys(ProfileKind.TIER) == (
        (_KNOWN_PROFILE_KEYS - _DEPLOYMENT_KEYS) | {"extra_mounts", "seed_base", "image"}
    )
    assert TIER_KIND_KEYS.isdisjoint(_DEPLOYMENT_KEYS)


def test_app_set_is_the_agent_definition() -> None:
    assert APP_KIND_KEYS == _EXPECTED_APP_KEYS
    assert known_profile_keys(ProfileKind.APP) == _EXPECTED_APP_KEYS


@pytest.mark.parametrize("kind", [ProfileKind.TIER, ProfileKind.APP])
def test_extends_is_legal_in_a_root_profile_only(kind: ProfileKind) -> None:
    assert "extends" in known_profile_keys(ProfileKind.ROOT)
    assert "extends" not in known_profile_keys(kind)


@pytest.mark.parametrize("kind", list(ProfileKind))
@pytest.mark.parametrize("retired", ["app_template", "facility"])
def test_no_kind_accepts_a_retired_spelling(kind: ProfileKind, retired: str) -> None:
    """Retired spellings are rewritten before validation, never accepted here."""
    assert retired not in known_profile_keys(kind)


def test_render_keys_are_not_root_keys() -> None:
    """A root profile never spells what one card mounts, seeds from or runs as."""
    assert RENDER_KEYS.isdisjoint(ROOT_KIND_KEYS)


# ---------------------------------------------------------------------------
# The kind-aware refusal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", [ProfileKind.TIER, ProfileKind.APP])
def test_every_legal_key_passes(kind: ProfileKind) -> None:
    reject_unknown_kind_keys(dict.fromkeys(known_profile_keys(kind)), kind, "x/y.yml")


@pytest.mark.parametrize("key", sorted(_DEPLOYMENT_KEYS))
def test_a_deployment_key_in_a_tier_is_refused_naming_the_tier_keys(key: str) -> None:
    with pytest.raises(BuildProfileError) as excinfo:
        reject_unknown_kind_keys({"name": "readonly", key: {}}, ProfileKind.TIER, "tiers/ro.yml")

    message = str(excinfo.value)
    assert "tiers/ro.yml" in message
    assert f"{key!r} (legal only in the deployment's profile.yml)" in message
    assert "valid keys in a tier are:" in message
    listed = message.split("valid keys in a tier are:", 1)[1]
    for legal in ("extra_mounts", "seed_base", "image", "skills", "config"):
        assert legal in listed
    assert f" {key}," not in listed and not listed.rstrip(".").endswith(f" {key}")


@pytest.mark.parametrize("key", ["provider", "dispatch", "extends", "services"])
def test_a_host_key_in_an_app_is_refused_naming_the_app_keys(key: str) -> None:
    with pytest.raises(BuildProfileError) as excinfo:
        reject_unknown_kind_keys({"name": "logbook", key: "x"}, ProfileKind.APP, "apps/lb.yml")

    message = str(excinfo.value)
    assert "Unknown app key(s) in apps/lb.yml" in message
    assert f"{key!r} (legal only in the deployment's profile.yml)" in message
    listed = message.split("valid keys in an app are:", 1)[1]
    assert sorted(_EXPECTED_APP_KEYS) == [k.strip(" .") for k in listed.split(",")]


def test_a_typo_in_a_layer_gets_a_suggestion_from_its_own_set() -> None:
    with pytest.raises(BuildProfileError) as excinfo:
        reject_unknown_kind_keys({"seed_bas": "x"}, ProfileKind.APP, "apps/lb.yml")

    assert "'seed_bas' (did you mean 'seed_base'?)" in str(excinfo.value)


def test_every_offender_is_named_at_once() -> None:
    with pytest.raises(BuildProfileError) as excinfo:
        reject_unknown_kind_keys(
            {"services": {}, "exclude": {}, "zzz": 1}, ProfileKind.TIER, "tiers/ro.yml"
        )

    message = str(excinfo.value)
    for key in ("'services'", "'exclude'", "'zzz'"):
        assert key in message


def test_the_root_kind_uses_the_root_set() -> None:
    reject_unknown_kind_keys({"name": "p", "extends": "x", "services": {}}, ProfileKind.ROOT, "p")
    with pytest.raises(BuildProfileError, match="'image'"):
        reject_unknown_kind_keys({"name": "p", "image": "x"}, ProfileKind.ROOT, "p")


def test_a_retired_preset_key_keeps_its_own_message_in_a_layer() -> None:
    with pytest.raises(BuildProfileError, match="no longer a profile key"):
        reject_unknown_kind_keys({"app_template": "x"}, ProfileKind.TIER, "tiers/ro.yml")


# ---------------------------------------------------------------------------
# The loader judges a layer against its kind, before the merge
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dirname", "kind_word"),
    [
        ("tiers", "tier"),
        ("apps", "app"),
        *(
            (retired, LAYER_DIR_KINDS[successor])
            for retired, successor in RETIRED_LAYER_DIRS.items()
        ),
    ],
)
def test_the_loader_refuses_a_deployment_key_in_a_layer(
    tmp_path: Path, dirname: str, kind_word: str
) -> None:
    repo = _repo(tmp_path)
    layer = _write_yaml(repo / dirname / "reader.yml", {"name": "reader", "deploy_services": False})

    with pytest.raises(BuildProfileError) as excinfo:
        load_profile_document(layer)

    message = str(excinfo.value)
    assert f"Unknown {kind_word} key(s) in {dirname}/reader.yml" in message
    assert "'deploy_services' (legal only in the deployment's profile.yml)" in message


def test_the_loader_refuses_extends_in_a_tier_naming_the_tier_keys(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    layer = _write_yaml(repo / "tiers" / "ro.yml", {"name": "ro", "extends": "hello-world"})

    with pytest.raises(BuildProfileError, match="valid keys in a tier are:"):
        load_profile_document(layer)


def test_the_loader_refuses_a_host_key_in_an_app(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    layer = _write_yaml(repo / "apps" / "lb.yml", {"name": "lb", "provider": "anthropic"})

    with pytest.raises(BuildProfileError, match="valid keys in an app are:"):
        load_profile_document(layer)


def test_the_loader_accepts_a_tier_that_only_adds(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    layer = _write_yaml(
        repo / "tiers" / "rw.yml",
        {"name": "rw", "config": {"control_system.writes_enabled": True}},
    )

    loaded = load_profile_document(layer)

    assert loaded.profile_dir == repo.resolve()


def test_a_root_profile_still_spells_deployment_keys(tmp_path: Path) -> None:
    repo = _repo(tmp_path, {"name": "site", "data": "data", "deploy_services": False})

    load_profile_document(repo / "profile.yml")


# ---------------------------------------------------------------------------
# The roster keys of the new vocabulary are accepted by the schema
# ---------------------------------------------------------------------------


_ROSTER_CONFIG = {
    "modules.web_terminals.default_tier": "readonly",
    "modules.web_terminals.users": [
        {"name": "alice", "index": 0, "tier": "readwrite"},
        {"name": "bob", "index": 1, "tier": "readonly"},
    ],
    "modules.web_terminals.authorization.roles.operators.tier": "readwrite",
    "modules.web_terminals.shared": ["logbook", "knowledge", "channel-finder"],
    "modules.web_terminals.landing.groups": [{"type": "users"}, {"type": "apps"}],
}


def test_the_closed_config_check_accepts_the_tier_roster() -> None:
    reject_unknown_closed_config_keys(_ROSTER_CONFIG)


def test_the_parser_accepts_a_profile_spelling_the_tier_roster() -> None:
    profile = _parse_profile({"name": "site", "config": dict(_ROSTER_CONFIG)})

    assert profile.config["modules.web_terminals.default_tier"] == "readonly"


def test_a_tier_may_carry_roster_config() -> None:
    """``config:`` is legal in a tier; the dotted roster keys live inside it."""
    reject_unknown_kind_keys({"config": dict(_ROSTER_CONFIG)}, ProfileKind.TIER, "tiers/a.yml")


# ---------------------------------------------------------------------------
# `osprey validate <file>` and `osprey set`
# ---------------------------------------------------------------------------


def test_validate_refuses_a_deployment_key_in_a_tier_file(tmp_path: Path) -> None:
    from osprey.cli.validate_cmd import validate

    repo = _repo(tmp_path)
    layer = _write_yaml(repo / "tiers" / "ro.yml", {"name": "ro", "services": {}})

    result = CliRunner().invoke(validate, [str(layer)])

    assert result.exit_code == 2, result.output
    assert "Unknown tier key(s) in tiers/ro.yml" in result.output
    assert "valid keys in a tier are:" in result.output


def test_validate_refuses_a_host_key_in_an_app_file(tmp_path: Path) -> None:
    from osprey.cli.validate_cmd import validate

    repo = _repo(tmp_path)
    layer = _write_yaml(repo / "apps" / "lb.yml", {"name": "lb", "provider": "x"})

    result = CliRunner().invoke(validate, [str(layer)])

    assert result.exit_code == 2, result.output
    assert "Unknown app key(s) in apps/lb.yml" in result.output
    assert "valid keys in an app are:" in result.output


def test_set_names_the_keys_legal_in_the_profile(tmp_path: Path) -> None:
    from osprey.cli.set_cmd import set as set_command

    repo = _repo(tmp_path)

    result = CliRunner().invoke(
        set_command, ["--repo", str(repo), "seed_base=/x"], catch_exceptions=False
    )

    assert result.exit_code == 0, result.output
    assert "Not a profile key: seed_base" in result.stderr
    assert "Keys legal in profile.yml:" in result.stderr
    legal = result.stderr.split("Keys legal in profile.yml:", 1)[1]
    assert "deploy_services" in legal
    assert "extends" in legal


def test_set_judges_the_root_profile_by_the_root_set(tmp_path: Path) -> None:
    from osprey.cli.set_cmd import _unrecognized_top_level_keys

    repo = _repo(tmp_path)

    assert _unrecognized_top_level_keys(
        ("deploy_services=false", "image=x", "config.a=1"), repo / "profile.yml"
    ) == ["image"]
