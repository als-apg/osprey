"""A tier or app file resolves as a union over its deployment's root profile.

A tier adds by name: its lists union with the deployment's, its config keys win,
and it never subtracts. An app is attached to its deployment by what it is, not
by what it spells: the resolver sets the attachment facts (no services of its
own, no web-terminal tier of its own) and refuses a file that spells them. Root
profiles and host-variant overlays keep ``exclude:`` — it is the only way an
overlay subtracts — and a the retired layer directory file is read with its old meaning for
the one release it is still accepted.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from osprey.cli.build_profile_document import _read_profile_document
from osprey.cli.build_profile_merge import (
    _deep_merge,
    merge_layer_delta,
    resolve_profile_document,
)
from osprey.cli.build_profile_resolve import resolve_build_profile
from osprey.cli.profile_root import ProfileKind
from osprey_connectors.errors import BuildProfileError
from tests._retired_layout import RETIRED_LAYER_DIRNAME

_ROOT = {
    "name": "Facility",
    "data": "data",
    "deploy_services": True,
    "skills": ["diagnose", "session-report"],
    "web_panels": ["ariel"],
    "services": {"postgres": {"image": "postgres:16"}},
    "config": {
        "control_system.writes_enabled": False,
        "modules.web_terminals.enabled": True,
        "deployed_services": ["alpha", "beta"],
    },
}

_EXCLUDE_MESSAGE = "adds by name; edit the deployment's own lists in profile.yml instead"


def _write_yaml(path: Path, document: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _repo(tmp_path: Path, layer: str, document: dict, root: dict | None = None) -> Path:
    """A deployment repo with one layer file; returns the layer's path."""
    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True, exist_ok=True)
    _write_yaml(repo / "profile.yml", root if root is not None else _ROOT)
    return _write_yaml(repo / layer, document)


def _resolve(path: Path):
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return resolve_profile_document(raw, path, warn=False)


def _root_writing(word: str) -> dict:
    """The root document with the tier key in place of its retired leaf."""
    config = {k: v for k, v in _ROOT["config"].items() if k != "control_system.writes_enabled"}
    return {**_ROOT, "config": {**config, "control_system.writes": word}}


# ── the tier merge ───────────────────────────────────────────────────────────


def test_tier_is_a_deep_merge_over_the_root_with_lists_unioned(tmp_path: Path) -> None:
    tier = {
        "name": "Ops (Read-Write)",
        "skills": ["session-report", "write-check"],
        "web_panels": ["events", "bluesky"],
        "config": {"control_system.writes_enabled": True, "web.theme": "retro"},
    }
    path = _repo(tmp_path, "tiers/readwrite.yml", tier)
    document = _resolve(path)

    assert document.is_layer
    assert document.root_dir == (tmp_path / "repo").resolve()
    raw = document.raw
    assert raw["name"] == "Ops (Read-Write)"
    assert raw["skills"] == ["diagnose", "session-report", "write-check"]
    assert raw["web_panels"] == ["ariel", "events", "bluesky"]
    assert raw["config"]["control_system.writes_enabled"] is True
    assert raw["config"]["web.theme"] == "retro"
    assert raw["config"]["deployed_services"] == ["alpha", "beta"]


def test_tier_merge_matches_deep_merge_except_attachment(tmp_path: Path) -> None:
    """Outside the attachment facts, the tier result IS _deep_merge(root, tier)."""
    tier = {"skills": ["write-check"], "config": {"web.theme": "main"}}
    path = _repo(tmp_path, "tiers/readonly.yml", tier)
    raw = _resolve(path).raw

    # The root as the pipeline reads it: its retired posture leaf is rewritten
    # to ``control_system.writes`` at the parse point.
    root = _read_profile_document(tmp_path / "repo" / "profile.yml")
    expected = _deep_merge(root, tier)
    for key in ("deploy_services", "services"):
        raw.pop(key)
        expected.pop(key)
    raw["config"].pop("modules.web_terminals.enabled")
    expected["config"].pop("modules.web_terminals.enabled")
    assert raw == expected


def test_resolving_a_tier_does_not_mutate_the_root_or_layer(tmp_path: Path) -> None:
    tier = {"skills": ["write-check"]}
    path = _repo(tmp_path, "tiers/readonly.yml", tier)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    before = yaml.safe_dump(raw)
    resolve_profile_document(raw, path, warn=False)
    assert yaml.safe_dump(raw) == before


# ── attachment is a fact of the kind ─────────────────────────────────────────


@pytest.mark.parametrize("lists", ["add", "own"])
def test_attachment_facts_are_set_for_every_login(tmp_path: Path, lists: str) -> None:
    path = _repo(
        tmp_path, "tiers/readonly.yml", {"name": "Layer", "lists": lists, "skills": ["write-check"]}
    )
    raw = _resolve(path).raw

    assert raw["deploy_services"] is False
    assert raw["services"] == {}
    assert raw["config"]["modules.web_terminals.enabled"] is False


def test_attachment_overrides_a_nested_web_terminals_spelling_in_the_root(
    tmp_path: Path,
) -> None:
    """Whatever spelling the root uses, the render reads exactly one: off."""
    root = dict(_ROOT)
    root["config"] = {"modules.web_terminals": {"enabled": True, "port": 8080}}
    path = _repo(tmp_path, "tiers/readonly.yml", {"skills": ["x"]}, root=root)
    config = _resolve(path).raw["config"]

    assert config["modules.web_terminals.enabled"] is False
    assert config["modules.web_terminals"] == {"port": 8080}


def test_attachment_of_a_shared_login_over_a_bare_root(tmp_path: Path) -> None:
    root = {k: v for k, v in _ROOT.items() if k != "config"}
    root["config"] = {"modules.web_terminals.shared": ["logbook"]}
    path = _repo(tmp_path, "tiers/logbook.yml", {"name": "Logbook", "lists": "own"}, root=root)
    config = _resolve(path).raw["config"]
    assert config.pop("modules.web_terminals.enabled") is False
    assert config.pop("modules.web_terminals.shared") == ["logbook"]
    # What remains is only the shared login's seal: writes off, setup tool
    # denied, Config panel and gallery writes off.
    assert config == {
        "control_system.writes": "never",
        "claude_code.permissions.deny": ["mcp__osprey_workspace__setup_patch"],
        "web.config_panel.enabled": False,
        "web.scaffold_gallery.write_enabled": False,
    }


@pytest.mark.parametrize(("tier", "root"), [("asked", "allowed"), ("allowed", "asked")])
def test_a_tier_word_wins_over_the_root(tmp_path: Path, tier: str, root: str) -> None:
    document = {"name": "Ops", "config": {"control_system.writes": tier}}
    path = _repo(tmp_path, "tiers/ops.yml", document, root=_root_writing(root))
    assert _resolve(path).raw["config"]["control_system.writes"] == tier


def test_a_shared_login_over_an_asked_root_is_pinned_never(tmp_path: Path) -> None:
    root = _root_writing("asked")
    root["config"] = {**root["config"], "modules.web_terminals.shared": ["logbook"]}
    path = _repo(tmp_path, "tiers/logbook.yml", {"name": "Logbook", "lists": "own"}, root=root)
    assert _resolve(path).raw["config"]["control_system.writes"] == "never"


@pytest.mark.parametrize("layer", ["tiers/readonly.yml"])
@pytest.mark.parametrize(
    ("document", "offender"),
    [
        ({"deploy_services": False}, "deploy_services"),
        ({"deploy_services": True}, "deploy_services"),
        ({"services": {}}, "services"),
        ({"config": {"modules.web_terminals.enabled": False}}, "modules.web_terminals.enabled"),
        (
            {"config": {"modules.web_terminals": {"enabled": False}}},
            "modules.web_terminals.enabled",
        ),
        (
            {"config": {"modules": {"web_terminals": {"enabled": True}}}},
            "modules.web_terminals.enabled",
        ),
    ],
)
def test_attachment_keys_spelled_in_a_layer_are_refused(
    tmp_path: Path, layer: str, document: dict, offender: str
) -> None:
    path = _repo(tmp_path, layer, {"name": "Layer", **document})
    with pytest.raises(BuildProfileError) as excinfo:
        _resolve(path)
    message = str(excinfo.value)
    assert offender in message
    assert layer in message


def test_other_web_terminals_keys_in_a_tier_are_not_attachment(tmp_path: Path) -> None:
    path = _repo(
        tmp_path, "tiers/readonly.yml", {"config": {"modules.web_terminals.idle_timeout": 30}}
    )
    config = _resolve(path).raw["config"]
    assert config["modules.web_terminals.idle_timeout"] == 30
    assert config["modules.web_terminals.enabled"] is False


# ── exclude: refused in every login ──────────────────────────────────────────


@pytest.mark.parametrize("lists", ["add", "own"])
def test_exclude_in_a_login_is_refused(tmp_path: Path, lists: str) -> None:
    layer, article = "tiers/readonly.yml", "a tier"
    path = _repo(
        tmp_path, layer, {"name": "Layer", "lists": lists, "exclude": {"skills": ["diagnose"]}}
    )
    with pytest.raises(BuildProfileError) as excinfo:
        _resolve(path)
    message = str(excinfo.value)
    assert f"{article} {_EXCLUDE_MESSAGE}" in message
    assert layer in message


def test_tier_exclude_message_is_the_ruled_sentence(tmp_path: Path) -> None:
    path = _repo(tmp_path, "tiers/readonly.yml", {"exclude": {"skills": ["diagnose"]}})
    with pytest.raises(
        BuildProfileError,
        match=r"a tier adds by name; edit the deployment's own lists in profile\.yml instead",
    ):
        _resolve(path)


def test_empty_exclude_in_a_tier_is_still_refused(tmp_path: Path) -> None:
    path = _repo(tmp_path, "tiers/readonly.yml", {"exclude": {}})
    with pytest.raises(BuildProfileError, match="adds by name"):
        _resolve(path)


@pytest.mark.parametrize("layer", ["tiers/readonly.yml"])
def test_extends_in_a_layer_names_its_kind(tmp_path: Path, layer: str) -> None:
    path = _repo(tmp_path, layer, {"extends": "control-assistant"})
    with pytest.raises(BuildProfileError, match="'extends'") as excinfo:
        _resolve(path)
    assert layer.split("/")[0] + "/" in str(excinfo.value)


# ── root profiles and overlays keep exclude: (A2) ────────────────────────────


def test_root_profile_keeps_exclude(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    _write_yaml(tmp_path / "base.yml", {"name": "Base", "skills": ["diagnose", "session-report"]})
    path = _write_yaml(
        tmp_path / "profile.yml",
        {"extends": "./base.yml", "name": "Child", "exclude": {"skills": ["diagnose"]}},
    )
    document = _resolve(path)
    assert document.raw["skills"] == ["session-report"]
    assert "exclude" not in document.raw
    assert not document.is_layer


def test_golden_variant_overlay_subtracts_a_service(tmp_path: Path) -> None:
    """The one place a subtraction is still spelled: a host deploying fewer services.

    The deployment carries tiers, and they do not change what the root may do:
    an overlay on ``profile.yml`` still subtracts a service with ``exclude:``.
    """
    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True)
    _write_yaml(
        repo / "base.yml",
        {
            "name": "Base",
            "data": "data",
            "config": {"deployed_services": ["alpha", "beta", "gamma"]},
        },
    )
    profile = _write_yaml(repo / "profile.yml", {"extends": "./base.yml", "name": "Facility"})
    _write_yaml(repo / "tiers" / "readonly.yml", {"name": "Facility (Read-Only)"})
    overlay = _write_yaml(
        repo / "variants" / "small-host.yml",
        {"exclude": {"config": {"deployed_services": ["beta"]}}},
    )

    resolved, profile_dir = resolve_build_profile(
        profile.resolve(), preset=None, overlays=(overlay.resolve(),)
    )
    assert resolved.config["deployed_services"] == ["alpha", "gamma"]
    assert profile_dir == repo.resolve()


# ── the retired layer directory during the deprecation window ──────────────────────────────────


def test_layer_file_keeps_exclude_for_its_deprecation_window(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = _repo(
        tmp_path,
        f"{RETIRED_LAYER_DIRNAME}/reader.yml",
        {"name": "Reader", "deploy_services": False, "exclude": {"skills": ["diagnose"]}},
    )
    with caplog.at_level(logging.WARNING):
        raw = _resolve(path).raw
    assert raw["skills"] == ["session-report"]
    assert raw["deploy_services"] is False
    assert raw["services"] == {}
    assert raw["config"]["modules.web_terminals.enabled"] is False


def test_layer_file_contradicting_its_attachment_is_refused(tmp_path: Path) -> None:
    path = _repo(tmp_path, f"{RETIRED_LAYER_DIRNAME}/reader.yml", {"deploy_services": True})
    with pytest.raises(BuildProfileError, match="deploy_services"):
        _resolve(path)


# ── merge_layer_delta keeps serving the exclude path ───────────────────────


def test_merge_layer_delta_still_applies_exclude() -> None:
    merged = merge_layer_delta(_ROOT, {"exclude": {"skills": ["diagnose"]}})
    assert merged["skills"] == ["session-report"]
    assert "exclude" not in merged


def test_kind_is_location_not_content(tmp_path: Path) -> None:
    """The same document is refused under tiers/ and accepted as a root profile."""
    document = {"name": "X", "deploy_services": False, "exclude": {"skills": ["a"]}}
    standalone = _write_yaml(tmp_path / "solo" / "profile.yml", document)
    assert not _resolve(standalone).is_layer
    layer = _repo(tmp_path, "tiers/x.yml", document)
    with pytest.raises(BuildProfileError):
        _resolve(layer)
    assert ProfileKind.TIER.is_layer
