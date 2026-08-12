"""Tests for ``extends:`` accepting a bundled preset name in addition to a path.

The path-style ``extends: ./base.yml`` flow remains unchanged. The new flow
makes user-owned profiles able to inherit from upstream presets by name
(``extends: hello-world``) without depending on a sibling YAML file.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli import build_profile_presets
from osprey.cli.build_cmd import build
from osprey.cli.build_profile import (
    _load_preset_raw,
    _preset_exists,
    _presets_dir,
    list_presets,
    resolve_build_profile,
)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_preset_exists_returns_path_for_bundled_preset() -> None:
    """`_preset_exists` returns the file path for a known preset."""
    path = _preset_exists("hello-world")
    assert path is not None
    assert path.is_file()
    assert path.name == "hello-world.yml"


def test_preset_exists_normalizes_underscore_form() -> None:
    """`_preset_exists` honors the same _→- normalization as the loader."""
    assert _preset_exists("hello_world") is not None
    assert _preset_exists("control_assistant") is not None


def test_preset_exists_returns_none_for_unknown_name() -> None:
    """`_preset_exists` is non-raising — unknown names return None."""
    assert _preset_exists("definitely-not-a-real-preset") is None


def test_preset_exists_returns_none_for_path_shaped_value() -> None:
    """Backward-compat guard: ``extends: als-base.yml`` (path-shaped) must NOT
    accidentally match a preset, so it falls through to filesystem resolution.
    """
    # Normalization is `_→-`; the probe is for `als-base.yml.yml`, which doesn't exist.
    assert _preset_exists("als-base.yml") is None


def test_extends_by_preset_name_resolves(tmp_path: Path) -> None:
    """A profile with ``extends: hello-world`` (bare name) resolves to the bundled preset."""
    profile = tmp_path / "p.yml"
    profile.write_text("extends: hello-world\nname: ChildOfPreset\n")
    resolved, profile_dir = resolve_build_profile(profile.resolve(), preset=None)
    assert resolved.name == "ChildOfPreset"  # child wins
    # Preset's data_bundle/provider must flow through.
    assert resolved.data_bundle == "hello_world"
    assert resolved.provider == "anthropic"
    # Preset's hook list must be present (child added nothing → list is the preset's).
    assert "hook-log" in resolved.hooks


def test_extends_by_path_still_works(tmp_path: Path) -> None:
    """Path-style ``extends: ./base.yml`` continues to resolve via filesystem."""
    base = tmp_path / "base.yml"
    base.write_text(
        "name: BaseFromFile\n"
        "data_bundle: hello_world\n"
        "provider: anthropic\n"
        "model: claude-haiku-4-5\n"
        "hooks: [hook-log]\n"
    )
    child = tmp_path / "child.yml"
    child.write_text("extends: ./base.yml\nname: ChildOfFile\n")
    resolved, _ = resolve_build_profile(child.resolve(), preset=None)
    assert resolved.name == "ChildOfFile"
    assert resolved.data_bundle == "hello_world"


def test_extends_path_shaped_value_resolves_via_filesystem(tmp_path: Path) -> None:
    """Backward-compat: an extends value with a ``.yml`` suffix is treated as a path,
    not a preset. Mirrors ALS's ``extends: als-base.yml`` pattern.
    """
    base = tmp_path / "als-base.yml"
    base.write_text(
        "name: AlsBase\ndata_bundle: hello_world\nprovider: anthropic\nhooks: [hook-log]\n"
    )
    child = tmp_path / "client.yml"
    child.write_text("extends: als-base.yml\nname: AlsClient\n")
    resolved, _ = resolve_build_profile(child.resolve(), preset=None)
    assert resolved.name == "AlsClient"


def test_extends_nonexistent_name_has_combined_diagnostic(tmp_path: Path) -> None:
    """Unknown extends value: error message must mention BOTH the preset list
    and the resolved file path so the user knows where to look.
    """
    from osprey.errors import BuildProfileError

    profile = tmp_path / "p.yml"
    profile.write_text("extends: not-a-real-thing\nname: Orphan\n")
    with pytest.raises(BuildProfileError) as exc:
        resolve_build_profile(profile.resolve(), preset=None)
    msg = str(exc.value)
    assert "not-a-real-thing" in msg
    assert "no bundled preset" in msg.lower()
    assert "no file at" in msg.lower()
    # Available list must appear so the user can spot typos.
    for name in list_presets():
        assert name in msg


def test_extends_preset_chain_via_intermediate_file(tmp_path: Path) -> None:
    """Preset → file chain: intermediate file extends a preset, child extends the file.

    Exercises ``_resolve_extends``'s existing recursion + cycle-detection path
    when the chain mixes a bundled preset (looked up by name) and a sibling file.
    """
    mid = tmp_path / "mid.yml"
    mid.write_text("extends: hello-world\nname: MidLayer\nmodel: claude-opus-4-5\n")
    child = tmp_path / "child.yml"
    child.write_text("extends: ./mid.yml\nname: Leaf\n")
    resolved, _ = resolve_build_profile(child.resolve(), preset=None)
    # Leaf wins on name; mid wins on model; preset provides data_bundle.
    assert resolved.name == "Leaf"
    assert resolved.model == "claude-opus-4-5"
    assert resolved.data_bundle == "hello_world"


def test_build_from_a_repo_whose_profile_extends_a_preset_by_name(
    runner: CliRunner, tmp_path: Path
) -> None:
    """End-to-end: a deployment repo's ``profile.yml`` with ``extends: hello-world``
    (a bare preset name) builds through the real render pipeline, not just
    through ``resolve_build_profile`` in isolation — so a mistake in how the
    build wires preset resolution into the zero-argument render would show up
    here even if the resolver itself still behaved correctly on its own."""
    repo = tmp_path / "ext-repo"
    repo.mkdir()
    (repo / "profile.yml").write_text("extends: hello-world\nname: ExtTest\n", encoding="utf-8")

    result = runner.invoke(build, ["--repo", str(repo), "--skip-deps", "--skip-lifecycle"])

    assert result.exit_code == 0, result.output
    config = yaml.safe_load((repo / "build" / "config.yml").read_text(encoding="utf-8"))
    # The preset's own provider flowed through the extends chain into the render.
    assert config["claude_code"]["provider"] == "anthropic"


def test_presets_dir_returns_directory_with_known_presets() -> None:
    """`_presets_dir` should resolve to a real directory containing the bundled YAMLs."""
    presets = _presets_dir()
    assert presets.is_dir()
    files = {p.name for p in presets.glob("*.yml")}
    assert "hello-world.yml" in files


def test_load_preset_raw_invalid_yaml_raises_build_profile_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A preset file with invalid YAML raises `BuildProfileError` naming the preset."""
    from osprey.errors import BuildProfileError

    monkeypatch.setattr(build_profile_presets, "_presets_dir", lambda: tmp_path)
    (tmp_path / "broken.yml").write_text("name: [unterminated\n", encoding="utf-8")
    with pytest.raises(BuildProfileError) as exc:
        _load_preset_raw("broken")
    msg = str(exc.value)
    assert "Invalid YAML" in msg
    assert "broken" in msg


def test_load_preset_raw_non_mapping_yaml_raises_build_profile_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A preset file whose YAML parses to a list (not a mapping) is rejected."""
    from osprey.errors import BuildProfileError

    monkeypatch.setattr(build_profile_presets, "_presets_dir", lambda: tmp_path)
    (tmp_path / "listy.yml").write_text("- one\n- two\n", encoding="utf-8")
    with pytest.raises(BuildProfileError) as exc:
        _load_preset_raw("listy")
    msg = str(exc.value)
    assert "must be a YAML mapping" in msg
    assert "listy" in msg
