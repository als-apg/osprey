"""Tests for resolved-preset content hashing (deploy staleness provenance).

The hash must cover the preset as *resolved* (post-``extends`` merge), so a
change in a parent preset is visible through every child that inherits it.
"""

from __future__ import annotations

import json

import pytest

from osprey.cli import build_profile, build_profile_presets


@pytest.fixture
def presets_dir(tmp_path, monkeypatch):
    d = tmp_path / "presets"
    d.mkdir()
    monkeypatch.setattr(build_profile_presets, "_presets_dir", lambda: d)
    return d


def _write(d, name, text):
    (d / f"{name}.yml").write_text(text, encoding="utf-8")


def test_compute_preset_hash_is_stable(presets_dir):
    _write(presets_dir, "demo", "name: Demo\nservices: {}\n")
    h1 = build_profile.compute_preset_hash("demo")
    h2 = build_profile.compute_preset_hash("demo")
    assert h1 == h2
    assert h1.startswith("sha256:")


def test_compute_preset_hash_ignores_formatting_only_changes(presets_dir):
    """Reflowing YAML without changing content must not read as preset drift."""
    _write(presets_dir, "demo", "name: Demo\nservices: {}\n")
    before = build_profile.compute_preset_hash("demo")
    _write(presets_dir, "demo", "# a comment\nservices: {}\nname: Demo\n")
    assert build_profile.compute_preset_hash("demo") == before


def test_compute_preset_hash_tracks_content_change(presets_dir):
    _write(presets_dir, "demo", "name: Demo\n")
    before = build_profile.compute_preset_hash("demo")
    _write(presets_dir, "demo", "name: Demo\nmodules.web_terminals:\n  enabled: true\n")
    assert build_profile.compute_preset_hash("demo") != before


def test_compute_preset_hash_sees_extends_parent_change(presets_dir):
    _write(presets_dir, "base", "name: Base\n")
    _write(presets_dir, "child", "extends: base\nname: Child\n")
    before = build_profile.compute_preset_hash("child")
    _write(presets_dir, "base", "name: Base\nservices:\n  openobserve: {}\n")
    assert build_profile.compute_preset_hash("child") != before


def test_compute_preset_hash_unknown_preset_returns_none(presets_dir):
    assert build_profile.compute_preset_hash("no-such-preset") is None


def test_compute_profile_hash_is_stable(tmp_path):
    """Re-hashing unchanged content yields the same digest (the staleness no-drift case)."""
    profile = tmp_path / "my-profile.yml"
    profile.write_text("name: Custom\nskills: [a, b]\n", encoding="utf-8")
    assert build_profile.compute_profile_hash(profile) == build_profile.compute_profile_hash(
        profile
    )


def test_compute_profile_hash_tracks_content_change(tmp_path):
    profile = tmp_path / "my-profile.yml"
    profile.write_text("name: Custom\n", encoding="utf-8")
    before = build_profile.compute_profile_hash(profile)
    assert before.startswith("sha256:")
    profile.write_text("name: Custom\nservices:\n  postgresql: {}\n", encoding="utf-8")
    assert build_profile.compute_profile_hash(profile) != before


def test_compute_profile_hash_missing_file_returns_none(tmp_path):
    assert build_profile.compute_profile_hash(tmp_path / "gone.yml") is None


def test_compute_profile_hash_non_mapping_yaml_returns_none(tmp_path):
    """A profile file whose YAML parses to a list (not a mapping) is unhashable."""
    profile = tmp_path / "list-profile.yml"
    profile.write_text("- a\n- b\n", encoding="utf-8")
    assert build_profile.compute_profile_hash(profile) is None


def test_generate_manifest_stamps_preset_hash(presets_dir, tmp_path):
    """A --preset build records creation.preset_hash matching the resolved preset."""
    from osprey.cli.templates import manifest as manifest_mod

    _write(presets_dir, "demo", "name: Demo\nservices: {}\n")
    project_dir = tmp_path / "proj"
    project_dir.mkdir()

    data = manifest_mod.generate_manifest(
        template_root=tmp_path / "templates",
        jinja_env=None,
        project_dir=project_dir,
        project_name="proj",
        template_name="demo_bundle",
        context={},
        preset_name="demo",
    )

    assert data["creation"]["preset_hash"] == build_profile.compute_preset_hash("demo")
    on_disk = json.loads((project_dir / manifest_mod.MANIFEST_FILENAME).read_text())
    assert on_disk["creation"]["preset_hash"] == data["creation"]["preset_hash"]


def test_generate_manifest_records_both_profile_path_forms(tmp_path, monkeypatch):
    """The typed string is kept for display; the resolved path is kept for reuse.

    ``reproducible_command`` has to echo what the user typed, but every later
    reader — the deploy-side staleness advisory above all — needs a path that
    still resolves from somewhere other than the build's working directory.
    """
    from osprey.cli.templates import manifest as manifest_mod

    profile = tmp_path / "profiles" / "facility.yml"
    profile.parent.mkdir()
    profile.write_text("name: Facility\n", encoding="utf-8")
    project_dir = tmp_path / "proj"
    project_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    data = manifest_mod.generate_manifest(
        template_root=tmp_path / "templates",
        jinja_env=None,
        project_dir=project_dir,
        project_name="proj",
        template_name="demo_bundle",
        context={"profile_path_abs": str(profile)},
        profile_path="profiles/facility.yml",
    )

    assert data["build_args"]["profile_path"] == "profiles/facility.yml"
    assert data["build_args"]["profile_path_abs"] == str(profile)
    assert data["reproducible_command"] == "osprey build proj profiles/facility.yml"
    assert data["creation"]["preset_hash"] == build_profile.compute_profile_hash(profile)


def test_generate_manifest_omits_absolute_path_when_build_gave_none(tmp_path):
    """A --preset build has no profile path to resolve; the key stays absent."""
    from osprey.cli.templates import manifest as manifest_mod

    project_dir = tmp_path / "proj"
    project_dir.mkdir()

    data = manifest_mod.generate_manifest(
        template_root=tmp_path / "templates",
        jinja_env=None,
        project_dir=project_dir,
        project_name="proj",
        template_name="demo_bundle",
        context={},
        preset_name="demo",
    )

    assert "profile_path_abs" not in data["build_args"]


def test_generate_manifest_survives_unhashable_preset(tmp_path, monkeypatch):
    """Hash stamping is best-effort: a failing hash must not break the build."""
    from osprey.cli.templates import manifest as manifest_mod

    def _boom(name):
        raise RuntimeError("hash exploded")

    monkeypatch.setattr(build_profile, "compute_preset_hash", _boom)
    project_dir = tmp_path / "proj"
    project_dir.mkdir()

    data = manifest_mod.generate_manifest(
        template_root=tmp_path / "templates",
        jinja_env=None,
        project_dir=project_dir,
        project_name="proj",
        template_name="demo_bundle",
        context={},
        preset_name="demo",
    )

    assert "preset_hash" not in data["creation"]
