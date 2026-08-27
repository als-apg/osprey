"""Pins that the image context's ``.dockerignore`` excludes the deployment
repo's Claude Code WORKING state, not just its deployment source.

``_stage_source_zone`` (``osprey.cli.build_cmd``) copies the repo's source zone
into a container image by exclusion, not by naming what to take — so anything
under the repo's ``.claude/`` tree that isn't explicitly excluded travels into
every project and persona image. Four of those subtrees are host-side working
state, not deployment source an image's runtime ever reads: ``.claude/plans/``
(open planning artifacts), ``.claude/epics/`` (epic/phase state),
``.claude/worktrees/`` (nested git worktrees), and ``.claude/.logs/`` (agent
logs and receipts). This module pins that the shipped ``.dockerignore``
template names all four, and that the two functions that turn a
``.dockerignore`` into an actual pruned tree
(``_write_image_context_dockerignore`` and ``_prune_ignored_entries``) remove
them from an image context while leaving deployment source — including other
``.claude/`` subtrees a running agent does read, like ``.claude/skills/`` and
``.claude/settings.json`` — untouched.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

from osprey.cli.build_cmd import _prune_ignored_entries, _write_image_context_dockerignore
from osprey.utils.workspace import BUILD_DIR_NAME

#: The four working-state subtrees this module pins. Stated literally rather
#: than read off the template, so a future edit to the template's wording (or
#: an accidental deletion of an entry) is caught by comparing against this
#: independent list, not against itself.
_EXCLUDED_WORKING_STATE = (
    ".claude/plans/",
    ".claude/epics/",
    ".claude/worktrees/",
    ".claude/.logs/",
)


def _seed_image_context(image_root: Path) -> None:
    # Arrange: a minimal image context with the shipped .dockerignore staged
    # at build/.dockerignore, exactly where _write_image_context_dockerignore
    # reads it from.
    shipped = resources.files("osprey").joinpath("templates/project/dockerignore").read_text()
    (image_root / BUILD_DIR_NAME).mkdir(parents=True)
    (image_root / BUILD_DIR_NAME / ".dockerignore").write_text(shipped, encoding="utf-8")

    # Working state that must be pruned.
    (image_root / ".claude" / "plans").mkdir(parents=True)
    (image_root / ".claude" / "plans" / "x.md").write_text("plan\n", encoding="utf-8")
    (image_root / ".claude" / "epics" / "e").mkdir(parents=True)
    (image_root / ".claude" / "epics" / "e" / "STATE.json").write_text("{}\n", encoding="utf-8")
    (image_root / ".claude" / "worktrees" / "w").mkdir(parents=True)
    (image_root / ".claude" / "worktrees" / "w" / "profile.yml").write_text(
        "project_name: w\n", encoding="utf-8"
    )
    (image_root / ".claude" / ".logs" / "receipts").mkdir(parents=True)
    (image_root / ".claude" / ".logs" / "receipts" / "r.json").write_text("{}\n", encoding="utf-8")

    # Deployment source that must survive: an agent-readable .claude/ subtree,
    # the render's own settings, and the repo marker.
    (image_root / ".claude" / "skills" / "foo").mkdir(parents=True)
    (image_root / ".claude" / "skills" / "foo" / "SKILL.md").write_text("# foo\n", encoding="utf-8")
    (image_root / BUILD_DIR_NAME / ".claude").mkdir(parents=True)
    (image_root / BUILD_DIR_NAME / ".claude" / "settings.json").write_text("{}\n", encoding="utf-8")
    (image_root / "profile.yml").write_text("project_name: fixture\n", encoding="utf-8")


def test_working_state_pruned_deployment_source_kept(tmp_path: Path) -> None:
    image_root = tmp_path / "image"
    _seed_image_context(image_root)

    # Act
    patterns = _write_image_context_dockerignore(image_root)
    _prune_ignored_entries(image_root, patterns)

    # Assert: none of the four working-state subtrees survive.
    assert not (image_root / ".claude" / "plans").exists()
    assert not (image_root / ".claude" / "epics").exists()
    assert not (image_root / ".claude" / "worktrees").exists()
    assert not (image_root / ".claude" / ".logs").exists()

    # Assert: deployment source is untouched.
    assert (image_root / ".claude" / "skills" / "foo" / "SKILL.md").is_file()
    assert (image_root / BUILD_DIR_NAME / ".claude" / "settings.json").is_file()
    assert (image_root / "profile.yml").is_file()


def test_context_root_dockerignore_names_all_four_working_state_entries(
    tmp_path: Path,
) -> None:
    image_root = tmp_path / "image"
    _seed_image_context(image_root)

    # Act
    _write_image_context_dockerignore(image_root)

    # Assert: each pattern is present, re-spelled `**/`-anchored for the
    # context-root depth (see _write_image_context_dockerignore).
    emitted = (image_root / ".dockerignore").read_text(encoding="utf-8")
    for entry in _EXCLUDED_WORKING_STATE:
        assert f"**/{entry}" in emitted, f"{entry!r} missing from the context-root .dockerignore"
