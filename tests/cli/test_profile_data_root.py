"""Tests for the profile ``data:`` tree replacing the bundle's data tree.

A profile that declares ``data:`` sources the project's ``data/`` directory
from its own tree instead of ``apps/<data_bundle>/data/``. That is a full
replacement, not a layer, and the tree is content rather than templates:

* a stray ``.j2`` file lands byte-identical, extension intact
* neither ``copy_template_data`` branch that reads the bundle contributes a
  file (the ``apps/<bundle>/data`` derivation nor the rglob fallback/merge)
* the build-time pipeline is otherwise untouched: tier materialization runs
  against the profile-sourced tree, and the project/ mirror applied later still
  wins
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml
from jinja2 import Environment, FileSystemLoader

from osprey.cli.templates.scaffolding import copy_template_data


def _bundle_data_dir() -> Path:
    """Path to the bundled control_assistant data tree."""
    import osprey

    return Path(osprey.__file__).parent / "templates" / "apps" / "control_assistant" / "data"


def _write_profile(profile_dir: Path, **extra) -> Path:
    """Materialize a profile directory carrying its own copy of the bundle data.

    The copy starts byte-identical to ``apps/control_assistant/data/`` so the
    tier-routed subtrees the build materializes from are present; individual
    tests then mutate it to make the replacement observable.
    """
    profile_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(_bundle_data_dir(), profile_dir / "data")

    profile: dict = {
        "name": "Data Root Test",
        "data_bundle": "control_assistant",
        "data": "data",
        "provider": "cborg",
        "model": "haiku",
        "channel_finder_mode": "in_context",
    }
    profile.update(extra)
    path = profile_dir / "profile.yml"
    path.write_text(yaml.dump(profile, default_flow_style=False))
    return path


def _build(profile_path: Path, output_dir: Path, project_name: str = "data-proj") -> Path:
    """Run `osprey build` against ``profile_path`` and return the project dir."""
    from click.testing import CliRunner

    from osprey.cli.main import cli

    output_dir.mkdir(parents=True, exist_ok=True)
    result = CliRunner().invoke(
        cli,
        [
            "build",
            project_name,
            str(profile_path),
            "--output-dir",
            str(output_dir),
            "--skip-deps",
            "--skip-lifecycle",
        ],
    )
    assert result.exit_code == 0, (
        f"build failed (exit={result.exit_code})\n"
        f"--- output ---\n{result.output}\n"
        f"--- exception ---\n{result.exception}"
    )
    return output_dir / project_name


class TestProfileDataIsContentNotTemplates:
    """The profile data tree is copied verbatim — no Jinja pass."""

    def test_stray_j2_survives_byte_identical(self, tmp_path: Path) -> None:
        profile_path = _write_profile(tmp_path / "profile")
        stray = tmp_path / "profile" / "data" / "facility_notes.md.j2"
        stray.write_bytes(
            b"{{ facility_name }} raw braces\n{% if channel_finder_mode %}kept{% endif %}\n"
        )

        project_dir = _build(profile_path, tmp_path / "out")

        landed = project_dir / "data" / "facility_notes.md.j2"
        assert landed.exists(), "stray .j2 was dropped from the profile data tree"
        assert landed.read_bytes() == stray.read_bytes(), (
            "stray .j2 was rendered instead of copied verbatim"
        )
        assert not (project_dir / "data" / "facility_notes.md").exists(), (
            ".j2 extension was stripped — the profile tree went through the render path"
        )


class TestFullReplacement:
    """No bundle file reaches a project built from a profile data tree."""

    def test_bundle_files_absent_from_build(self, tmp_path: Path) -> None:
        profile_dir = tmp_path / "profile"
        profile_path = _write_profile(profile_dir)

        # Drop two distinctive bundle artifacts from the profile's copy: if the
        # bundle tree were layered under (or merged into) the profile tree,
        # they would reappear in the rendered project.
        (profile_dir / "data" / "channel_limits.json").unlink()
        shutil.rmtree(profile_dir / "data" / "lattice")
        (profile_dir / "data" / "facility_marker.txt").write_text("profile tree\n")

        project_dir = _build(profile_path, tmp_path / "out")

        assert (project_dir / "data" / "facility_marker.txt").exists(), (
            "profile-only file did not land — the profile tree was not the source"
        )
        assert not (project_dir / "data" / "channel_limits.json").exists(), (
            "bundle channel_limits.json leaked into a full-replacement build"
        )
        assert not (project_dir / "data" / "lattice").exists(), (
            "bundle lattice/ leaked into a full-replacement build"
        )

    @pytest.mark.parametrize("bundle_data_at_top", [True, False])
    def test_both_bundle_derivation_branches_are_bypassed(
        self, tmp_path: Path, bundle_data_at_top: bool
    ) -> None:
        """``data_root`` short-circuits before either bundle-reading branch.

        ``copy_template_data`` finds bundle data two ways: the direct
        ``apps/<bundle>/data`` derivation, and — when that is absent — an rglob
        for a nested ``data/`` directory that merges into an existing
        destination. Parametrizing over the bundle's shape exercises both.
        """
        template_root = tmp_path / "templates"
        bundle_dir = template_root / "apps" / "demo"
        if bundle_data_at_top:
            bundle_data = bundle_dir / "data"
        else:
            bundle_data = bundle_dir / "pkg" / "data"
        bundle_data.mkdir(parents=True)
        (bundle_data / "from_bundle.txt").write_text("bundle\n")

        data_root = tmp_path / "facility_data"
        (data_root / "nested").mkdir(parents=True)
        (data_root / "from_profile.txt").write_text("profile\n")
        (data_root / "nested" / "seed.json.j2").write_text("{{ not_rendered }}\n")

        project_dir = tmp_path / "project"
        # Pre-populate data/ so the fallback's merge-into-existing path would
        # be taken if the bypass were missing.
        (project_dir / "data").mkdir(parents=True)

        copy_template_data(
            template_root,
            project_dir,
            "demo_pkg",
            "demo",
            {},
            jinja_env=Environment(loader=FileSystemLoader(str(template_root))),
            data_root=data_root,
        )

        assert (project_dir / "data" / "from_profile.txt").read_text() == "profile\n"
        assert (project_dir / "data" / "nested" / "seed.json.j2").read_text() == (
            "{{ not_rendered }}\n"
        )
        assert not (project_dir / "data" / "from_bundle.txt").exists(), (
            "bundle data leaked past the data_root bypass"
        )


class TestBuildPipelineOrderingHolds:
    """Materialization and the mirror behave as for a bundle-sourced tree."""

    def test_tier_materialization_runs_against_the_profile_tree(self, tmp_path: Path) -> None:
        profile_dir = tmp_path / "profile"
        profile_path = _write_profile(profile_dir)

        # A facility DB that is recognizably NOT the bundle's, staged where the
        # tier materializer reads from.
        tier_src = (
            profile_dir / "data" / "channel_databases" / "tiers" / "tier1" / "in_context.json"
        )
        tier_src.write_text('{"channels": {"FACILITY:TIER:SRC": {"description": "profile"}}}\n')

        project_dir = _build(profile_path, tmp_path / "out")

        flat = project_dir / "data" / "channel_databases" / "in_context.json"
        assert flat.read_text() == tier_src.read_text(), (
            "flat channel DB was not materialized from the profile's tier tree"
        )
        assert not (project_dir / "data" / "channel_databases" / "tiers").exists(), (
            "tiers/ subtree was not pruned from the profile-sourced tree"
        )

    def test_project_mirror_wins_over_profile_data(self, tmp_path: Path) -> None:
        """The project/ mirror applies after the data tree lands, so it wins."""
        profile_dir = tmp_path / "profile"
        mirror_src = profile_dir / "project" / "data" / "channel_databases" / "in_context.json"
        mirror_body = '{"channels": {"MIRROR:CH": {"description": "mirrored"}}}\n'

        profile_path = _write_profile(profile_dir)
        mirror_src.parent.mkdir(parents=True, exist_ok=True)
        mirror_src.write_text(mirror_body)

        # Both the profile's flat DB and the tier source it is materialized from
        # carry different content, so a passing assertion can only mean the
        # mirror landed last.
        profile_flat = profile_dir / "data" / "channel_databases" / "in_context.json"
        profile_flat.write_text('{"channels": {"PROFILE:FLAT": {}}}\n')
        tier_src = (
            profile_dir / "data" / "channel_databases" / "tiers" / "tier1" / "in_context.json"
        )
        tier_src.write_text('{"channels": {"PROFILE:TIER": {}}}\n')
        (profile_dir / "data" / "facility_marker.txt").write_text("profile tree\n")

        project_dir = _build(profile_path, tmp_path / "out")

        assert (project_dir / "data" / "facility_marker.txt").exists(), (
            "data/ was not sourced from the profile tree — the ordering claim is untested"
        )
        landed = project_dir / "data" / "channel_databases" / "in_context.json"
        assert landed.read_text() == mirror_body, (
            "project/ mirror did not win over the profile data tree / tier materialization"
        )
