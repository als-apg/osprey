"""Tests for PromptGalleryService — bridges PromptRegistry + TemplateManager for web UI."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.init_cmd import init
from osprey.cli.prompt_registry import PromptRegistry
from osprey.interfaces.web_terminal.prompt_gallery_service import PromptGalleryService

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAFE_ARTIFACT = "rules/safety"  # Always available, real content


@pytest.fixture()
def project_dir(tmp_path):
    """Create a real OSPREY project via ``osprey init``."""
    runner = CliRunner()
    result = runner.invoke(init, ["gallery-test", "--output-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    return tmp_path / "gallery-test"


@pytest.fixture()
def service(project_dir):
    return PromptGalleryService(project_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_config(project_dir: Path) -> dict:
    with open(project_dir / "config.yml", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get_overrides(project_dir: Path) -> dict[str, str]:
    cfg = _read_config(project_dir)
    return cfg.get("prompts", {}).get("overrides", {})


# ===========================================================================
# List artifacts
# ===========================================================================


class TestListArtifacts:
    """Tests for PromptGalleryService.list_artifacts()."""

    def test_list_artifacts_count(self, service):
        """Total count matches PromptRegistry.default().all_artifacts()."""
        registry = PromptRegistry.default()
        expected = len(registry.all_artifacts())
        result = service.list_artifacts()
        assert len(result) == expected

    def test_list_artifacts_status_framework(self, service):
        """An artifact without an override has status 'framework'."""
        result = service.list_artifacts()
        by_name = {a["name"]: a for a in result}
        assert SAFE_ARTIFACT in by_name
        assert by_name[SAFE_ARTIFACT]["status"] == "framework"

    def test_list_artifacts_status_overridden(self, service, project_dir):
        """After scaffolding, the artifact shows status 'overridden'."""
        service.scaffold_override(SAFE_ARTIFACT)
        # Re-create service to pick up refreshed config
        svc = PromptGalleryService(project_dir)
        result = svc.list_artifacts()
        by_name = {a["name"]: a for a in result}
        assert by_name[SAFE_ARTIFACT]["status"] == "overridden"

    def test_list_artifacts_categories(self, service):
        """Category is derived from the canonical name prefix."""
        result = service.list_artifacts()
        by_name = {a["name"]: a for a in result}

        # "agents/channel-finder" -> category "agents"
        assert by_name["agents/channel-finder"]["category"] == "agents"
        # "rules/safety" -> category "rules"
        assert by_name["rules/safety"]["category"] == "rules"
        # "claude-md" (no slash) -> category "config"
        assert by_name["claude-md"]["category"] == "config"

    def test_list_artifacts_summary_counts(self, service):
        """Sum of framework + overridden equals total."""
        result = service.list_artifacts()
        framework = sum(1 for a in result if a["status"] == "framework")
        overridden = sum(1 for a in result if a["status"] == "overridden")
        assert framework + overridden == len(result)


# ===========================================================================
# Content retrieval
# ===========================================================================


class TestGetContent:
    """Tests for get_content, get_framework_content, get_override_content."""

    def test_get_content_framework(self, service):
        """Framework artifact returns non-empty content with source='framework'."""
        result = service.get_content(SAFE_ARTIFACT)
        assert result["source"] == "framework"
        assert isinstance(result["content"], str)
        assert len(result["content"]) > 0

    def test_get_content_overridden(self, service, project_dir):
        """After scaffold + modify, get_content returns the override."""
        service.scaffold_override(SAFE_ARTIFACT)
        custom = "# Custom safety rules\nDo not touch anything.\n"
        service.save_override(SAFE_ARTIFACT, custom)

        svc = PromptGalleryService(project_dir)
        result = svc.get_content(SAFE_ARTIFACT)
        assert result["source"] == "override"
        assert result["content"] == custom

    def test_get_framework_content_renders(self, service):
        """get_framework_content returns non-empty rendered content."""
        content = service.get_framework_content(SAFE_ARTIFACT)
        assert isinstance(content, str)
        assert len(content) > 0

    def test_get_framework_content_unknown(self, service):
        """Unknown artifact name raises KeyError."""
        with pytest.raises(KeyError, match="Unknown artifact"):
            service.get_framework_content("nonexistent/artifact")

    def test_get_override_content_not_overridden(self, service):
        """Non-overridden artifact returns None."""
        result = service.get_override_content(SAFE_ARTIFACT)
        assert result is None

    def test_get_override_content_exists(self, service, project_dir):
        """After scaffolding, get_override_content returns file content."""
        scaffold_result = service.scaffold_override(SAFE_ARTIFACT)
        expected_content = scaffold_result["content"]

        svc = PromptGalleryService(project_dir)
        content = svc.get_override_content(SAFE_ARTIFACT)
        assert content is not None
        assert content == expected_content


# ===========================================================================
# Diff
# ===========================================================================


class TestComputeDiff:
    """Tests for compute_diff."""

    def test_compute_diff_identical(self, service):
        """Scaffold without modification yields has_diff=False."""
        service.scaffold_override(SAFE_ARTIFACT)
        result = service.compute_diff(SAFE_ARTIFACT)
        assert result["has_diff"] is False
        assert result["additions"] == 0
        assert result["deletions"] == 0

    def test_compute_diff_with_changes(self, service, project_dir):
        """Scaffold, modify, diff shows additions and deletions."""
        service.scaffold_override(SAFE_ARTIFACT)
        service.save_override(SAFE_ARTIFACT, "# Completely replaced content\n")

        svc = PromptGalleryService(project_dir)
        result = svc.compute_diff(SAFE_ARTIFACT)
        assert result["has_diff"] is True
        assert result["additions"] > 0
        assert result["deletions"] > 0
        assert isinstance(result["unified_diff"], str)
        assert len(result["unified_diff"]) > 0

    def test_compute_diff_not_overridden(self, service):
        """Diff on a non-overridden artifact raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not overridden"):
            service.compute_diff(SAFE_ARTIFACT)


# ===========================================================================
# Scaffold
# ===========================================================================


class TestScaffold:
    """Tests for scaffold_override."""

    def test_scaffold_creates_file_and_config(self, service, project_dir):
        """Scaffolding creates the override file and updates config.yml."""
        result = service.scaffold_override(SAFE_ARTIFACT)

        assert result["status"] == "created"
        assert "override_path" in result
        assert len(result["content"]) > 0

        # Override file exists on disk
        override_file = project_dir / result["override_path"]
        assert override_file.exists()
        assert override_file.read_text(encoding="utf-8") == result["content"]

        # config.yml has the override entry
        overrides = _get_overrides(project_dir)
        assert SAFE_ARTIFACT in overrides
        assert overrides[SAFE_ARTIFACT] == result["override_path"]

    def test_scaffold_already_overridden(self, service):
        """Scaffolding the same artifact twice raises FileExistsError."""
        service.scaffold_override(SAFE_ARTIFACT)
        with pytest.raises(FileExistsError, match="already exists"):
            service.scaffold_override(SAFE_ARTIFACT)


# ===========================================================================
# Save
# ===========================================================================


class TestSaveOverride:
    """Tests for save_override."""

    def test_save_override_writes_file(self, service, project_dir):
        """Save writes new content to the override file."""
        service.scaffold_override(SAFE_ARTIFACT)
        new_content = "# Updated safety rules\nAll writes require approval.\n"
        result = service.save_override(SAFE_ARTIFACT, new_content)

        assert result["status"] == "saved"

        # Verify file content on disk
        override_file = project_dir / result["path"]
        assert override_file.read_text(encoding="utf-8") == new_content

    def test_save_override_not_overridden(self, service):
        """Saving to a non-overridden artifact raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not overridden"):
            service.save_override(SAFE_ARTIFACT, "some content")


# ===========================================================================
# Unoverride
# ===========================================================================


class TestUnoverride:
    """Tests for unoverride."""

    def test_unoverride_removes_config(self, service, project_dir):
        """Unoverride removes the config entry but keeps the file."""
        scaffold_result = service.scaffold_override(SAFE_ARTIFACT)
        override_file = project_dir / scaffold_result["override_path"]

        svc = PromptGalleryService(project_dir)
        result = svc.unoverride(SAFE_ARTIFACT)

        assert result["status"] == "removed"
        assert result["deleted_file"] is False

        # Config entry is gone
        overrides = _get_overrides(project_dir)
        assert SAFE_ARTIFACT not in overrides

        # File still exists on disk
        assert override_file.exists()

    def test_unoverride_deletes_file(self, service, project_dir):
        """Unoverride with delete_file=True removes config and file."""
        scaffold_result = service.scaffold_override(SAFE_ARTIFACT)
        override_file = project_dir / scaffold_result["override_path"]
        assert override_file.exists()

        svc = PromptGalleryService(project_dir)
        result = svc.unoverride(SAFE_ARTIFACT, delete_file=True)

        assert result["status"] == "removed"
        assert result["deleted_file"] is True

        # Config entry is gone
        overrides = _get_overrides(project_dir)
        assert SAFE_ARTIFACT not in overrides

        # File is deleted
        assert not override_file.exists()

    def test_unoverride_not_overridden(self, service):
        """Unoverriding a non-overridden artifact raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not overridden"):
            service.unoverride(SAFE_ARTIFACT)


# ===========================================================================
# Description extraction
# ===========================================================================


class TestDescriptionExtraction:
    """Tests for two-tier summary/description extraction from front matter."""

    def test_agent_has_summary_from_front_matter(self, service):
        """Agent artifact summary comes from template front matter, not registry."""
        result = service.list_artifacts()
        by_name = {a["name"]: a for a in result}
        art = by_name["agents/data-visualizer"]
        assert art["summary"] == (
            "Creates plots, charts, dashboards, and compiles LaTeX reports"
        )
        # Summary should differ from the full description
        assert art["summary"] != art["description"]

    def test_hook_has_summary_from_front_matter(self, service):
        """Hook artifact summary comes from docstring YAML front matter."""
        result = service.list_artifacts()
        by_name = {a["name"]: a for a in result}
        art = by_name["hooks/limits"]
        assert art["summary"] == (
            "Validates channel write values against the limits database"
        )

    def test_config_falls_back_to_registry(self, service):
        """Config artifacts (JSON templates) fall back to registry description."""
        registry = PromptRegistry.default()
        reg_art = registry.get("claude-md")
        result = service.list_artifacts()
        by_name = {a["name"]: a for a in result}
        art = by_name["claude-md"]
        # No front matter in JSON templates → falls back to registry
        assert art["summary"] == reg_art.description

    def test_description_is_full_from_front_matter(self, service):
        """Agent description field comes from full front matter description."""
        result = service.list_artifacts()
        by_name = {a["name"]: a for a in result}
        art = by_name["agents/data-visualizer"]
        assert "Creates data visualizations" in art["description"]

    def test_rule_has_summary_and_description(self, service):
        """Rule artifacts get both summary and description from front matter."""
        result = service.list_artifacts()
        by_name = {a["name"]: a for a in result}
        art = by_name["rules/safety"]
        assert art["summary"] == "Safety boundaries and tool confinement"
        assert "MCP tool interactions" in art["description"]

    def test_command_has_summary_from_front_matter(self, service):
        """Command artifacts get summary from added front matter."""
        result = service.list_artifacts()
        by_name = {a["name"]: a for a in result}
        art = by_name["commands/diagnose"]
        assert art["summary"] == "Investigate operational failures"
