"""Tests for :func:`resolve_profile_root` — the profile-relative path anchor.

Location is the type. A profile file is a login (kind ``tier``) when its parent
directory is ``tiers`` and a ``profile.yml`` sits beside that directory, and a
root otherwise. The bundled ``presets/tiers/`` directory marks login presets in
the package. A file in the retired layer directory is read as a tier for the
deprecation window, with one warning naming the rewrite. The predicate looks
exactly one level and never walks upward. A layer file with no root beside it
cannot stand alone, so it raises rather than building something silently
incomplete, and a plain ``<name>.yml`` in ``apps/`` — which holds applications
only — is refused by name.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from osprey.cli import build_profile_presets
from osprey.cli.profile_root import ProfileKind, ProjectProfile, resolve_profile_root
from osprey.utils.workspace import APPS_DIRNAME, TIERS_DIRNAME
from osprey_connectors.errors import BuildProfileError
from tests._retired_layout import RETIRED_LAYER_DIRNAME


def _repo(tmp_path: Path, *layers: str) -> Path:
    """A tiny repo: ``profile.yml`` plus one ``<dir>/<stem>.yml`` per layer."""
    root = tmp_path / "my-profile"
    root.mkdir(parents=True)
    (root / "profile.yml").write_text("name: my-profile\n")
    for layer in layers:
        file = root / layer
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(f"name: {file.stem}\n")
    return root


# ── Kind by location ────────────────────────────────────────────────────────


def test_tier_file_anchors_at_the_profile_root(tmp_path: Path) -> None:
    """A tiers/ file beside a profile.yml is a tier anchored one level up."""
    root = _repo(tmp_path, "tiers/readonly.yml")

    root_dir, kind = resolve_profile_root(root / "tiers" / "readonly.yml")

    assert root_dir == root.resolve()
    assert kind is ProfileKind.TIER


def test_a_plain_file_in_apps_beside_a_profile_is_refused(tmp_path: Path) -> None:
    """apps/ holds applications only: a login file left there is named, never built."""
    root = _repo(tmp_path, "apps/logbook.yml")

    with pytest.raises(BuildProfileError) as excinfo:
        resolve_profile_root(root / "apps" / "logbook.yml")

    message = str(excinfo.value)
    assert "apps/logbook.yml is a login file" in message
    assert "osprey init --revalidate" in message
    assert "tiers/" in message


def test_root_profile_is_kind_root(tmp_path: Path) -> None:
    """The profile.yml itself is a root, anchored at its own directory."""
    root = _repo(tmp_path, "tiers/readonly.yml")

    root_dir, kind = resolve_profile_root(root / "profile.yml")

    assert root_dir == root.resolve()
    assert kind is ProfileKind.ROOT


def test_dirname_constants() -> None:
    """The layer directories are named by one definition each."""
    assert TIERS_DIRNAME == "tiers"
    assert APPS_DIRNAME == "apps"


def test_kind_values_and_truthiness() -> None:
    """Kinds carry their public spelling; only a layer kind is truthy."""
    assert [k.value for k in ProfileKind] == ["root", "tier"]
    assert not ProfileKind.ROOT
    assert ProfileKind.TIER
    assert not ProfileKind.ROOT.is_layer
    assert ProfileKind.TIER.is_layer


@pytest.mark.parametrize("dirname", ["tiers"])
def test_layer_file_without_a_root_raises(tmp_path: Path, dirname: str) -> None:
    """A tiers/ file with no profile.yml beside it is a hard error."""
    layer_dir = tmp_path / "stray" / dirname
    layer_dir.mkdir(parents=True)
    orphan = layer_dir / "reader.yml"
    orphan.write_text("name: reader\n")

    with pytest.raises(BuildProfileError, match="profile root is missing") as excinfo:
        resolve_profile_root(orphan)

    assert str((tmp_path / "stray" / "profile.yml").resolve()) in str(excinfo.value)
    assert f"'{dirname}/'" in str(excinfo.value)


@pytest.mark.parametrize("dirname", ["tiers", "apps"])
def test_layer_predicate_does_not_walk_upward(tmp_path: Path, dirname: str) -> None:
    """A file two levels below tiers/ or apps/ is a root of its own."""
    root = _repo(tmp_path)
    deep = root / dirname / "drafts"
    deep.mkdir(parents=True)
    draft = deep / "reader.yml"
    draft.write_text("name: reader\n")

    root_dir, kind = resolve_profile_root(draft)

    assert root_dir == deep.resolve()
    assert kind is ProfileKind.ROOT


# ── the retired layer directory during the deprecation window ─────────────────────────────────


def test_layer_file_resolves_as_a_tier_with_one_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A the retired layer directory file is a tier, with a single warning naming the rewrite."""
    root = _repo(tmp_path, f"{RETIRED_LAYER_DIRNAME}/reader.yml")
    render = root / RETIRED_LAYER_DIRNAME / "reader.yml"

    with caplog.at_level(logging.WARNING, logger="osprey.cli.profile_root"):
        first = resolve_profile_root(render)
        second = resolve_profile_root(render)

    assert first == second == (root.resolve(), ProfileKind.TIER)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1
    text = warnings[0].getMessage()
    assert "tiers/" in text
    assert text.rstrip(".").endswith("run `osprey init --revalidate`")


def test_retired_layer_file_without_a_root_raises(tmp_path: Path) -> None:
    """A the retired layer directory file with no profile.yml beside it is a hard error."""
    renders = tmp_path / "stray" / RETIRED_LAYER_DIRNAME
    renders.mkdir(parents=True)
    orphan = renders / "reader.yml"
    orphan.write_text("name: reader\n")

    with pytest.raises(BuildProfileError, match="profile root is missing") as excinfo:
        resolve_profile_root(orphan)

    assert str((tmp_path / "stray" / "profile.yml").resolve()) in str(excinfo.value)


def test_retired_layer_directory_beside_a_differently_named_profile_raises(tmp_path: Path) -> None:
    """Only a file named profile.yml makes a root."""
    renders = tmp_path / "loose" / RETIRED_LAYER_DIRNAME
    renders.mkdir(parents=True)
    (tmp_path / "loose" / "site.yml").write_text("name: loose\n")
    render = renders / "reader.yml"
    render.write_text("name: reader\n")

    with pytest.raises(BuildProfileError, match="profile root is missing"):
        resolve_profile_root(render)


# ── Bundled presets ─────────────────────────────────────────────────────────


def test_bundled_tier_preset_is_a_tier() -> None:
    """presets/tiers/<n>.yml in the package resolves as a tier at the presets dir."""
    presets = build_profile_presets._presets_dir()

    root_dir, kind = resolve_profile_root(presets / "tiers" / "readonly.yml")

    assert root_dir == presets.resolve()
    assert kind is ProfileKind.TIER


@pytest.mark.parametrize(("dirname", "expected"), [("tiers", "tier")])
def test_package_presets_subdirs_decide_the_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dirname: str, expected: str
) -> None:
    """The packaged presets dir needs no profile.yml for its tiers/."""
    presets = tmp_path / "presets"
    (presets / dirname).mkdir(parents=True)
    preset = presets / dirname / "sample.yml"
    preset.write_text("name: sample\n")
    monkeypatch.setattr(build_profile_presets, "_presets_dir", lambda: presets)

    root_dir, kind = resolve_profile_root(preset)

    assert root_dir == presets.resolve()
    assert kind is ProfileKind(expected)


def test_bundled_deployment_preset_is_a_root() -> None:
    """A top-level bundled preset stays a root anchored at the presets dir."""
    presets = build_profile_presets._presets_dir()

    root_dir, kind = resolve_profile_root(presets / "hello-world.yml")

    assert root_dir == presets.resolve()
    assert kind is ProfileKind.ROOT


# ── Standalone roots ────────────────────────────────────────────────────────


def test_bare_file_is_standalone(tmp_path: Path) -> None:
    """A profile written straight into a temp dir is standalone, root = its parent."""
    profile = tmp_path / "profile.yml"
    profile.write_text("name: bare\n")

    root_dir, kind = resolve_profile_root(profile)

    assert root_dir == tmp_path.resolve()
    assert kind is ProfileKind.ROOT


def test_standalone_file_needs_no_sibling_profile_yml(tmp_path: Path) -> None:
    """An arbitrarily named file with no profile.yml beside it still resolves."""
    profile = tmp_path / "scratch-profile.yaml"
    profile.write_text("name: scratch\n")

    root_dir, kind = resolve_profile_root(profile)

    assert root_dir == tmp_path.resolve()
    assert kind is ProfileKind.ROOT


def test_nested_directory_with_another_name_is_standalone(tmp_path: Path) -> None:
    """A nested dir with another name is standalone even beside a profile.yml."""
    root = _repo(tmp_path, "variants/reader.yml")

    root_dir, kind = resolve_profile_root(root / "variants" / "reader.yml")

    assert root_dir == (root / "variants").resolve()
    assert kind is ProfileKind.ROOT


def test_relative_path_resolves_to_an_absolute_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative profile path yields an absolute root, so callers never join onto cwd."""
    root = _repo(tmp_path, "tiers/reader.yml")
    monkeypatch.chdir(tmp_path)

    root_dir, kind = resolve_profile_root(Path("my-profile/tiers/reader.yml"))

    assert root_dir.is_absolute()
    assert root_dir == root.resolve()
    assert kind is ProfileKind.TIER


def test_missing_file_still_resolves_a_root(tmp_path: Path) -> None:
    """The helper anchors on path shape, not existence — callers report missing files."""
    root_dir, kind = resolve_profile_root(tmp_path / "never-written.yml")

    assert root_dir == tmp_path.resolve()
    assert kind is ProfileKind.ROOT


def test_missing_layer_file_still_resolves_its_kind(tmp_path: Path) -> None:
    """A not-yet-written tiers/ file beside a root resolves as a login."""
    root = _repo(tmp_path)

    root_dir, kind = resolve_profile_root(root / "tiers" / "never-written.yml")

    assert root_dir == root.resolve()
    assert kind is ProfileKind.TIER


# ── ProjectProfile ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("kind", list(ProfileKind))
def test_project_profile_env_path_is_the_root_env(tmp_path: Path, kind: ProfileKind) -> None:
    """Every kind shares the root's .env; a layer never keeps its own."""
    profile = ProjectProfile(profile_path=tmp_path / "x.yml", root=tmp_path, kind=kind)

    assert profile.env_path == tmp_path / ".env"
    assert profile.is_layer is kind.is_layer


def test_project_profile_defaults_to_root_kind() -> None:
    """An unresolved project carries no layer kind and no .env."""
    profile = ProjectProfile()

    assert profile.kind is ProfileKind.ROOT
    assert profile.env_path is None
