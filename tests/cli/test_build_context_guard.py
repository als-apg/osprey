"""Profile context artifacts must not delete the web-terminal context baseline.

Every build installs the framework's fallback ``docker/web-terminal-context/
base.md``; a profile that ships ``web-terminal-context/base.md`` replaces it
through the convention's first-class slot. Per-user directories copy *below*
that path, so the baseline survives them by construction — these tests pin
that, pin the slot's override precedence (which otherwise rests on nothing but
build-step ordering), and pin the error a profile gets when it puts any other
loose file where a per-user directory belongs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.cli.build_persistence import _apply_conventions
from osprey.cli.templates.manager import TemplateManager
from osprey.errors import BuildProfileError


def _bundle_data_root(bundle: str = "control_assistant") -> Path:
    """The tree these fixtures hand the render as the profile's ``data:``.

    A build copies the tree its profile's ``data:`` key names, and that key is
    required — nothing falls back to a packaged tree any more. These fixtures
    render straight from a bundle rather than from a profile, so they name the
    tree that bundle packages, which is the content the render used to reach
    for on its own.
    """
    return Path(TemplateManager().template_root) / "apps" / bundle / "data"


def _create_project(manager: TemplateManager, **kwargs) -> Path:
    """``create_project`` plus the three steps a real build takes next.

    A build renders the framework template, overlays the resolved profile's
    ``config:`` block onto the result, stamps ``.osprey-manifest.json``, and
    regenerates ``.claude/`` from the finished config. The template carries
    only derived and profile-field-derived keys, so a fixture that stops after
    the render holds half a config — the declarative half is the preset's, and
    the artifacts rendered before it landed do not know about the deployment's
    control system, services or servers. These fixtures render from a bundle
    rather than from a profile, so they overlay the preset ``osprey init``
    pairs with that bundle.
    """
    from osprey.cli.build_profile import resolve_build_profile
    from osprey.utils.config_writer import config_update_fields

    bundle = kwargs.setdefault("data_bundle", "control_assistant")
    preset = bundle.replace("_", "-")
    kwargs.setdefault("data_root", _bundle_data_root(bundle))
    project = manager.create_project(**kwargs)
    profile, _preset_dir = resolve_build_profile(None, preset=preset)
    config_update_fields(project / "config.yml", profile.config)
    manager.generate_manifest(
        project, kwargs["project_name"], preset, {}, artifacts=kwargs.get("artifacts")
    )
    # The build's last render, and the one that ships: `create_project` wrote
    # `.claude/` from a config.yml that did not yet carry the preset's block.
    manager.regenerate_claude_code(project)
    return project


CONTEXT_DIR = "docker/web-terminal-context"


def _built_project(tmp_path: Path, name: str) -> Path:
    """Render a minimal project the way the build pipeline does."""
    return _create_project(
        TemplateManager(),
        project_name=name,
        output_dir=tmp_path,
        data_bundle="hello_world",
    )


def _profile_with_user_dir(tmp_path: Path, user: str) -> Path:
    """Profile directory holding a ``web-terminal-context/<user>`` seed."""
    profile_dir = tmp_path / "profile"
    user_dir = profile_dir / "web-terminal-context" / user
    user_dir.mkdir(parents=True)
    (user_dir / "extra.md").write_text(f"# {user}'s notes\n", encoding="utf-8")
    return profile_dir


def test_per_user_context_keeps_baseline_and_seeds_user(tmp_path: Path) -> None:
    """A roster user's context lands beside base.md, which stays in place."""
    project_path = _built_project(tmp_path, "context-per-user")
    profile_dir = _profile_with_user_dir(tmp_path, "alice")

    _apply_conventions(profile_dir, project_path, ["alice"])

    base_md = project_path / CONTEXT_DIR / "base.md"
    assert base_md.is_file()
    assert base_md.read_text(encoding="utf-8").strip() != ""
    assert (project_path / CONTEXT_DIR / "alice" / "extra.md").is_file()


def test_profile_base_md_overrides_the_framework_fallback(tmp_path: Path) -> None:
    """The slot's whole point: the profile's baseline text wins over the
    framework's, and the precedence is pinned here rather than left to the
    accident that the framework install runs before the convention copies."""
    project_path = _built_project(tmp_path, "context-base-override")
    profile_dir = _profile_with_user_dir(tmp_path, "alice")
    authored = "# Facility baseline\n\nOur own ground rules.\n"
    (profile_dir / "web-terminal-context" / "base.md").write_text(authored, encoding="utf-8")

    framework_text = (project_path / CONTEXT_DIR / "base.md").read_text(encoding="utf-8")
    assert framework_text != authored  # the override must be observable

    _apply_conventions(profile_dir, project_path, ["alice"])

    assert (project_path / CONTEXT_DIR / "base.md").read_text(encoding="utf-8") == authored
    assert (project_path / CONTEXT_DIR / "alice" / "extra.md").is_file()


def test_loose_file_in_context_dir_is_rejected(tmp_path: Path) -> None:
    """The context convention holds per-user directories, never loose files.

    This is a structural guard rather than a whole-directory one: a profile
    cannot address the context root at all, and the one shape that tries is
    rejected before anything is copied.
    """
    project_path = _built_project(tmp_path, "context-loose-file")
    profile_dir = _profile_with_user_dir(tmp_path, "alice")
    (profile_dir / "web-terminal-context" / "everyone.md").write_text("# all\n", encoding="utf-8")

    with pytest.raises(BuildProfileError) as excinfo:
        _apply_conventions(profile_dir, project_path, ["alice"])

    message = str(excinfo.value)
    assert "web-terminal-context/everyone.md" in message
    assert "one directory per web-terminal user" in message
    assert (project_path / CONTEXT_DIR / "base.md").is_file()


def test_departed_user_context_is_skipped(tmp_path: Path) -> None:
    """Context for someone off the roster is left alone, baseline untouched."""
    project_path = _built_project(tmp_path, "context-departed")
    profile_dir = _profile_with_user_dir(tmp_path, "alice")
    departed = profile_dir / "web-terminal-context" / "bob"
    departed.mkdir()
    (departed / "extra.md").write_text("# bob\n", encoding="utf-8")

    applied = _apply_conventions(profile_dir, project_path, ["alice"])

    assert applied.departed_users == ["bob"]
    assert not (project_path / CONTEXT_DIR / "bob").exists()
    assert (project_path / CONTEXT_DIR / "alice" / "extra.md").is_file()
    assert (project_path / CONTEXT_DIR / "base.md").is_file()
    # Never deleted from the profile — the operator decides that.
    assert (departed / "extra.md").is_file()
