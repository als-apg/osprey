"""Regen-safety test: osprey claude regen must never touch the OKF bundle.

The facility knowledge bundle lives at a path declared by
``facility_knowledge.bundle_path`` in config.yml (resolved relative to the
config file directory when not absolute).  It is explicitly outside the
template-rendered output set — the regen walk in ``claude_code.py`` only
visits files under ``.claude/`` that originate from the template's ``claude/``
directory, plus the root-level ``CLAUDE.md`` and ``.mcp.json``.

Bundle location: ``{project_dir}/data/facility_knowledge/`` — matching the
value shipped in the control-assistant ``config.yml.j2``
(``facility_knowledge.bundle_path: data/facility_knowledge``).  The bundle
is structurally outside the regen output set.

Test design — three interlocking assertions:

1. **Byte-identity**: every bundle file has the same SHA-256 before and after
   regen (the mechanics the previous version used, now pointed at the correct path).
2. **Positive control**: regen DID rewrite at least one ``.claude/`` file, proving
   the regen invocation actually ran (not a no-op).  Without this, the
   byte-identity check is vacuous.
3. **Emit-layer exclusion**: ``regenerate_claude_code`` returns the set of
   relative paths it wrote (``changed`` + ``unchanged``).  Assert that no path
   starting with ``data/`` appears in that set.  This guards at the output-set
   layer — if someone later changed the regen walk to include ``data/``, this
   assertion would catch it structurally rather than hoping byte hashes happened
   not to change.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from osprey.cli.templates.manager import TemplateManager

# ---------------------------------------------------------------------------
# Constants — canonical bundle path (matches control-assistant config.yml.j2)
# ---------------------------------------------------------------------------

#: Relative bundle path shipped in control-assistant config.yml.j2.
BUNDLE_REL_PATH = "data/facility_knowledge"

#: Top-level directory prefix regen is allowed to write inside.
REGEN_ALLOWED_PREFIXES = (".claude/", "CLAUDE.md", ".mcp.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of *path*'s contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checksum_tree(root: Path) -> dict[str, str]:
    """Return a ``{relative-path: sha256}`` mapping for every file under *root*."""
    return {str(p.relative_to(root)): _sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}


def _make_project_with_bundle(tmp_path: Path) -> tuple[Path, Path, TemplateManager]:
    """Create a minimal OSPREY project and populate a facility knowledge bundle.

    The bundle is placed at ``{project_dir}/data/facility_knowledge/`` to match
    the shipping control-assistant ``config.yml.j2`` default.  The config is
    updated to declare ``facility_knowledge.bundle_path: data/facility_knowledge``.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Tuple of (project_dir, bundle_dir, manager).
    """
    manager = TemplateManager()
    project_dir = manager.create_project(
        project_name="bundle-regen-test",
        output_dir=tmp_path,
        data_bundle="control_assistant",
        context={"channel_finder_mode": "hierarchical"},
    )

    # Populate the bundle at the canonical path.
    bundle_dir = project_dir / BUNDLE_REL_PATH
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "facility_overview.md").write_text(
        "---\ntype: facility_overview\ntitle: Test Facility\ndescription: A test.\n---\n\nBody.\n",
        encoding="utf-8",
    )
    (bundle_dir / "tables").mkdir()
    (bundle_dir / "tables" / "beam_params.md").write_text(
        "---\ntype: data_table\ntitle: Beam Params\ndescription: Key parameters.\n---\n\nData.\n",
        encoding="utf-8",
    )

    # Declare the bundle path in config.yml.
    config_path = project_dir / "config.yml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config.setdefault("facility_knowledge", {})["bundle_path"] = BUNDLE_REL_PATH
    config_path.write_text(yaml.dump(config, allow_unicode=True), encoding="utf-8")

    return project_dir, bundle_dir, manager


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBundlePathResolution:
    """Bundle path is resolved correctly from config."""

    def test_bundle_path_in_config(self, tmp_path):
        """facility_knowledge.bundle_path is present and resolves to the bundle dir."""
        project_dir, bundle_dir, _ = _make_project_with_bundle(tmp_path)
        config = yaml.safe_load((project_dir / "config.yml").read_text(encoding="utf-8"))
        raw = config["facility_knowledge"]["bundle_path"]

        assert raw == BUNDLE_REL_PATH
        resolved = (project_dir / raw).resolve()
        assert resolved == bundle_dir.resolve()
        assert resolved.is_dir()

    def test_bundle_path_resolver_returns_absolute(self, tmp_path):
        """_resolve_bundle_path returns an absolute path for a relative config value."""
        from osprey.mcp_server.facility_knowledge.server import _resolve_bundle_path

        project_dir, bundle_dir, _ = _make_project_with_bundle(tmp_path)
        config = yaml.safe_load((project_dir / "config.yml").read_text(encoding="utf-8"))

        resolved = _resolve_bundle_path(config, project_dir)

        assert resolved.is_absolute()
        assert resolved == bundle_dir.resolve()

    def test_bundle_path_resolver_passes_through_absolute(self, tmp_path):
        """_resolve_bundle_path leaves an already-absolute path unchanged."""
        from osprey.mcp_server.facility_knowledge.server import _resolve_bundle_path

        bundle_dir = tmp_path / "custom_bundle"
        bundle_dir.mkdir()
        config = {"facility_knowledge": {"bundle_path": str(bundle_dir)}}

        resolved = _resolve_bundle_path(config, tmp_path / "project")

        assert resolved == bundle_dir


class TestRegenDoesNotTouchBundle:
    """osprey claude regen must leave every bundle file byte-identical.

    Each test uses three interlocking assertions so the check is non-vacuous:
    byte-identity (hash check), positive control (regen DID write .claude/),
    and emit-layer exclusion (bundle paths absent from regen's output set).
    """

    def test_regen_leaves_bundle_byte_identical(self, tmp_path):
        """Regen must not modify any bundle file — SHA-256 identical before and after.

        Positive control: asserts that regen DID write at least one ``.claude/``
        file (proving the invocation was real, not a no-op).
        Emit-layer check: asserts no ``data/`` path appears in the regen output set.
        """
        project_dir, bundle_dir, manager = _make_project_with_bundle(tmp_path)

        before = _checksum_tree(bundle_dir)
        assert before, "Bundle must be non-empty for this test to be meaningful"

        result = manager.regenerate_claude_code(project_dir)

        # --- Positive control: regen must have written something in .claude/ ---
        all_written = result.get("changed", []) + result.get("unchanged", [])
        dot_claude_files = [p for p in all_written if p.startswith(".claude/")]
        assert dot_claude_files, (
            "Positive control failed: regen reported nothing written under .claude/. "
            "The regen invocation may have been a no-op or the test fixture is broken."
        )

        # --- Emit-layer exclusion: no bundle path in the regen output set ---
        bundle_prefix = BUNDLE_REL_PATH + "/"
        bundle_in_output = [p for p in all_written if p.startswith(bundle_prefix)]
        assert not bundle_in_output, (
            f"Regen's output set must not include any bundle path. Found: {bundle_in_output}"
        )

        # --- Byte-identity: every bundle file unchanged ---
        after = _checksum_tree(bundle_dir)
        assert before == after, (
            "Regen must not modify any bundle file.\n"
            f"Bundle dir: {bundle_dir}\n"
            f"Modified: {[k for k in before if before[k] != after.get(k, '')]}\n"
            f"Deleted: {[k for k in before if k not in after]}\n"
            f"Created: {[k for k in after if k not in before]}"
        )

    def test_regen_does_not_create_files_in_bundle(self, tmp_path):
        """Regen must not create new files inside the bundle directory."""
        project_dir, bundle_dir, manager = _make_project_with_bundle(tmp_path)

        before = set(_checksum_tree(bundle_dir))
        manager.regenerate_claude_code(project_dir)
        after = set(_checksum_tree(bundle_dir))

        new_files = after - before
        assert not new_files, f"Regen must not create files in the bundle. New files: {new_files}"

    def test_regen_twice_still_leaves_bundle_intact(self, tmp_path):
        """Two consecutive regen runs must not accumulate bundle mutations."""
        project_dir, bundle_dir, manager = _make_project_with_bundle(tmp_path)

        before = _checksum_tree(bundle_dir)
        manager.regenerate_claude_code(project_dir)
        manager.regenerate_claude_code(project_dir)
        after = _checksum_tree(bundle_dir)

        assert before == after

    def test_bundle_outside_dot_claude(self, tmp_path):
        """Bundle directory must not be nested inside ``.claude/``."""
        project_dir, bundle_dir, _ = _make_project_with_bundle(tmp_path)
        dot_claude = project_dir / ".claude"

        try:
            bundle_dir.relative_to(dot_claude)
            raise AssertionError(f"Bundle {bundle_dir} must not be nested inside {dot_claude}")
        except ValueError:
            pass  # Expected: bundle_dir is outside .claude/

    def test_regen_output_set_contains_no_data_prefix(self, tmp_path):
        """Regen's returned output set must never include paths starting with ``data/``.

        This is the emit-layer guard in isolation.  It would catch a future change
        that broadened the regen walk to include ``data/`` regardless of whether
        files happened to change on disk.
        """
        project_dir, _, manager = _make_project_with_bundle(tmp_path)

        result = manager.regenerate_claude_code(project_dir)

        all_written = result.get("changed", []) + result.get("unchanged", [])
        data_paths = [p for p in all_written if p.startswith("data/")]
        assert not data_paths, (
            f"Regen output set must not include any data/ paths. Found: {data_paths}"
        )
