"""Overlay mappings must not delete the web-terminal persona baseline.

Every build installs the framework's ``docker/web-terminal-context/base.md``;
a *directory* overlay replaces its destination wholesale, so a mapping aimed at
that directory would remove the baseline and leave the deployment unable to
seed a web-terminal user. Per-user subdirectories below it stay allowed — that
is how facilities restore user-authored persona content after ``--force``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.cli.build_persistence import _copy_overlay_files
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
    """Profile directory holding an ``overlays/web-terminal-context/<user>`` seed."""
    profile_dir = tmp_path / "profile"
    user_dir = profile_dir / "overlays" / "web-terminal-context" / user
    user_dir.mkdir(parents=True)
    (user_dir / "extra.md").write_text(f"# {user}'s notes\n", encoding="utf-8")
    return profile_dir


def test_per_user_overlay_keeps_baseline_and_seeds_user(tmp_path: Path) -> None:
    """The documented per-user mapping leaves base.md in place beside the user dir."""
    project_path = _built_project(tmp_path, "overlay-per-user")
    profile_dir = _profile_with_user_dir(tmp_path, "alice")

    _copy_overlay_files(
        profile_dir,
        project_path,
        {"overlays/web-terminal-context/alice": f"{CONTEXT_DIR}/alice"},
    )

    base_md = project_path / CONTEXT_DIR / "base.md"
    assert base_md.is_file()
    assert base_md.read_text(encoding="utf-8").strip() != ""
    assert (project_path / CONTEXT_DIR / "alice" / "extra.md").is_file()


@pytest.mark.parametrize(
    "destination",
    [CONTEXT_DIR, f"{CONTEXT_DIR}/", f"./{CONTEXT_DIR}"],
)
def test_whole_directory_overlay_is_rejected(tmp_path: Path, destination: str) -> None:
    """Mapping the context root itself errors, naming the per-user form, and
    leaves the installed baseline untouched."""
    project_path = _built_project(tmp_path, "overlay-whole-dir")
    profile_dir = _profile_with_user_dir(tmp_path, "alice")

    with pytest.raises(BuildProfileError) as excinfo:
        _copy_overlay_files(
            profile_dir,
            project_path,
            {"overlays/web-terminal-context": destination},
        )

    message = str(excinfo.value)
    assert "base.md" in message
    assert f"overlays/web-terminal-context/alice: {CONTEXT_DIR}/alice" in message
    assert (project_path / CONTEXT_DIR / "base.md").is_file()


def test_file_overlay_into_context_dir_still_allowed(tmp_path: Path) -> None:
    """The guard is about directory replacement — a single file landing inside
    the context tree is unaffected."""
    project_path = _built_project(tmp_path, "overlay-single-file")
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    (profile_dir / "extra.md").write_text("# shared\n", encoding="utf-8")

    _copy_overlay_files(profile_dir, project_path, {"extra.md": f"{CONTEXT_DIR}/bob/extra.md"})

    assert (project_path / CONTEXT_DIR / "base.md").is_file()
    assert (project_path / CONTEXT_DIR / "bob" / "extra.md").is_file()
