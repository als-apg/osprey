"""Tests that PromptGalleryService reads files from disk, not re-rendered templates.

Verifies the fix for the bug where framework-managed artifacts showed unresolved
env vars (e.g. ``${TZ:-America/Los_Angeles}``) because get_content() re-rendered
the .j2 template instead of reading the already-correct file on disk.
"""

from __future__ import annotations

import pytest

from osprey.cli.build_cmd import build
from osprey.interfaces.web_terminal.prompt_gallery_service import PromptGalleryService

# Use an artifact known to exist after osprey build and backed by a .j2 template
SAFE_ARTIFACT = "rules/safety"


@pytest.fixture()
def project_dir(tmp_path):
    """Create a real OSPREY project via ``osprey build --preset hello-world``."""
    from click.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(
        build,
        [
            "env-test",
            "--preset", "hello-world",
            "--skip-deps", "--skip-lifecycle",
            "--output-dir", str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    return tmp_path / "env-test"


@pytest.fixture()
def service(project_dir):
    return PromptGalleryService(project_dir)


class TestGetContentReadsDisk:
    """get_content() should always return the file on disk, not a re-rendered template."""

    def test_get_content_reads_disk_for_framework_artifacts(self, service, project_dir):
        """Framework artifact: get_content returns what's on disk, not re-rendered.

        We modify the file on disk to have a known marker. If get_content
        re-renders the template, it won't have the marker. If it reads from
        disk, it will.
        """
        from osprey.services.prompts.catalog import PromptCatalog

        registry = PromptCatalog.default()
        art = registry.get(SAFE_ARTIFACT)
        assert art is not None

        # Modify the on-disk file to include a unique marker
        disk_path = project_dir / art.output_path
        assert disk_path.exists(), f"Expected file on disk: {disk_path}"
        marker = "# DISK-MARKER-12345\n"
        original = disk_path.read_text(encoding="utf-8")
        disk_path.write_text(marker + original, encoding="utf-8")

        # Re-create service to avoid any stale caching
        svc = PromptGalleryService(project_dir)
        result = svc.get_content(SAFE_ARTIFACT)

        # get_content should return the disk content (with marker)
        assert result["content"].startswith(marker)
        assert result["source"] == "framework"

    def test_get_content_falls_back_to_render_when_file_missing(self, service, project_dir):
        """When the on-disk file is missing, get_content falls back to _render_framework."""
        from osprey.services.prompts.catalog import PromptCatalog

        registry = PromptCatalog.default()
        art = registry.get(SAFE_ARTIFACT)
        assert art is not None

        # Delete the file from disk
        disk_path = project_dir / art.output_path
        disk_path.unlink()
        assert not disk_path.exists()

        # Re-create service after deleting file
        svc = PromptGalleryService(project_dir)
        result = svc.get_content(SAFE_ARTIFACT)

        # Should still return content (from template), not error
        assert result["source"] == "framework"
        assert isinstance(result["content"], str)
        assert len(result["content"]) > 0

    def test_get_content_source_label_reflects_ownership(self, service, project_dir):
        """source is 'framework' for unowned, 'user-owned' for claimed artifacts."""
        # Before claiming: framework
        result_before = service.get_content(SAFE_ARTIFACT)
        assert result_before["source"] == "framework"

        # Claim it
        service.scaffold_override(SAFE_ARTIFACT)
        svc = PromptGalleryService(project_dir)

        # After claiming: user-owned
        result_after = svc.get_content(SAFE_ARTIFACT)
        assert result_after["source"] == "user-owned"
