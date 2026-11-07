"""Preset listings by kind, and the one refusal a login name gets as a deployment.

``list_presets()`` lists deployments only; ``list_tiers()`` lists the bundled
``presets/tiers/`` logins beside it, the shared ones (``lists: own``) among
them. A login name handed to a verb that wants a deployment is refused with the
same sentence everywhere: it names the kind and says where the file renders.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from osprey.cli import build_profile_presets
from osprey.cli.build_profile_presets import (
    _load_preset_raw,
    _normalize_preset_name,
    _preset_exists,
    layer_preset_kind,
    layer_preset_refusal,
    list_presets,
    list_tiers,
)
from osprey.cli.profile_root import ProfileKind
from osprey_connectors.errors import BuildProfileError


def _expected_refusal(name: str, kind: str = "tier") -> str:
    folder = "tiers"
    return (
        f"`{name}` is a {kind}, not a deployment. It renders inside a repo: "
        f"`osprey init <dir> --preset control-assistant` writes {folder}/{name}.yml, "
        f"and `osprey build` there renders build/<project>-{name}"
    )


@pytest.fixture
def fake_presets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "presets"
    (root / "tiers").mkdir(parents=True)
    (root / "deploy-a.yml").write_text("name: a\n")
    (root / "_hidden.yml").write_text("name: hidden\n")
    (root / "tiers" / "viewer.yml").write_text("name: viewer\n")
    (root / "tiers" / "_draft.yml").write_text("name: draft\n")
    (root / "tiers" / "notes.txt").write_text("not a tier\n")
    monkeypatch.setattr(build_profile_presets, "_presets_dir", lambda: root)
    return root


# --- listings ---------------------------------------------------------------


def test_bundled_logins_listed() -> None:
    assert list_tiers() == [
        "admin",
        "channel-finder",
        "knowledge",
        "logbook",
        "readonly",
        "readwrite",
    ]


def test_list_presets_lists_deployments_only() -> None:
    presets = list_presets()
    assert {"control-assistant", "hello-world"} <= set(presets)
    assert not set(presets) & set(list_tiers())
    assert "tiers" not in presets and "apps" not in presets
    assert not list((build_profile_presets._presets_dir() / "apps").glob("*.yml"))


@pytest.mark.usefixtures("fake_presets")
def test_listings_skip_underscore_and_non_yaml() -> None:
    assert list_presets() == ["deploy-a"]
    assert list_tiers() == ["viewer"]


def test_listings_empty_when_kind_dir_missing(fake_presets: Path) -> None:
    for path in (fake_presets / "tiers").iterdir():
        path.unlink()
    (fake_presets / "tiers").rmdir()
    assert list_tiers() == []


# --- kind lookup ------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("readonly", ProfileKind.TIER),
        ("admin", ProfileKind.TIER),
        ("logbook", ProfileKind.TIER),
        ("channel-finder", ProfileKind.TIER),
        ("channel_finder", ProfileKind.TIER),
        ("control-assistant", None),
        ("no-such-thing", None),
    ],
)
def test_layer_preset_kind(name: str, kind: ProfileKind | None) -> None:
    assert layer_preset_kind(name) is kind


@pytest.mark.parametrize(("name", "kind"), [("readonly", "tier"), ("logbook", "tier")])
def test_refusal_wording(name: str, kind: str) -> None:
    assert layer_preset_refusal(name) == _expected_refusal(name, kind)


def test_refusal_none_for_deployment_and_unknown() -> None:
    assert layer_preset_refusal("control-assistant") is None
    assert layer_preset_refusal("no-such-thing") is None


def test_refusal_names_normalized_spelling() -> None:
    assert layer_preset_refusal("channel_finder") == _expected_refusal("channel-finder")


# --- deployment-only helpers stay deployment-only ---------------------------


def test_deployment_helpers_ignore_layers() -> None:
    assert _normalize_preset_name("channel_finder") == "channel-finder"
    assert _preset_exists("readonly") is None
    assert _preset_exists("logbook") is None
    assert _preset_exists("control-assistant") is not None


# --- the refusal at every verb ----------------------------------------------


@pytest.mark.parametrize(("name", "kind"), [("readwrite", "tier"), ("knowledge", "tier")])
def test_build_preset_mode_refuses_layer(name: str, kind: str) -> None:
    """`osprey build --preset` resolves through the preset reader."""
    from osprey.cli.build_profile import resolve_build_document

    with pytest.raises(BuildProfileError) as excinfo:
        _load_preset_raw(name)
    assert str(excinfo.value) == _expected_refusal(name, kind)
    with pytest.raises(BuildProfileError) as excinfo:
        resolve_build_document(None, name)
    assert str(excinfo.value) == _expected_refusal(name, kind)


def test_unknown_preset_message_unchanged() -> None:
    with pytest.raises(BuildProfileError, match="Unknown preset 'no-such-thing'"):
        _load_preset_raw("no-such-thing")


@pytest.mark.parametrize(("name", "kind"), [("readonly", "tier"), ("logbook", "tier")])
def test_init_refuses_layer_preset(tmp_path: Path, name: str, kind: str) -> None:
    from osprey.cli.init_cmd import init

    target = tmp_path / "repo"
    result = CliRunner().invoke(init, [str(target), "--preset", name, "--no-git"])
    assert result.exit_code == 2, result.output
    assert _expected_refusal(name, kind) in " ".join(result.output.split())
    assert not target.exists()


@pytest.mark.parametrize(
    ("folder", "name", "kind"), [("tiers", "admin", "tier"), ("tiers", "logbook", "tier")]
)
def test_validate_refuses_bundled_layer_file(folder: str, name: str, kind: str) -> None:
    from osprey.cli.validate_cmd import validate

    bundled = build_profile_presets._presets_dir() / folder / f"{name}.yml"
    result = CliRunner().invoke(validate, [str(bundled)])
    assert result.exit_code == 2, result.output
    assert _expected_refusal(name, kind) in " ".join(result.output.split())


@pytest.mark.parametrize(("folder", "kind"), [("tiers", "tier")])
def test_set_refuses_bundled_layer_dir(folder: str, kind: str) -> None:
    from osprey.cli.set_cmd import set as set_command

    bundled_dir = build_profile_presets._presets_dir() / folder
    result = CliRunner().invoke(set_command, ["--repo", str(bundled_dir), "model=x"])
    assert result.exit_code == 2, result.output
    flat = " ".join(result.output.split())
    assert f"is a {kind}, not a deployment" in flat
    assert f"writes {folder}/<name>.yml" in flat
