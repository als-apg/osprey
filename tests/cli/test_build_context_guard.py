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

CONTEXT_DIR = "docker/web-terminal-context"


def _built_project(tmp_path: Path, name: str) -> Path:
    """Render a minimal project the way the build pipeline does."""
    return TemplateManager().create_project(
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
