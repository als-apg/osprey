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


def _get_user_owned(project_dir: Path) -> list[str]:
    cfg = _read_config(project_dir)
    return cfg.get("prompts", {}).get("user_owned", [])


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
        """An artifact without user-ownership has status 'framework'."""
        result = service.list_artifacts()
        by_name = {a["name"]: a for a in result}
        assert SAFE_ARTIFACT in by_name
        assert by_name[SAFE_ARTIFACT]["status"] == "framework"

    def test_list_artifacts_status_user_owned(self, service, project_dir):
        """After claiming, the artifact shows status 'user-owned'."""
        service.scaffold_override(SAFE_ARTIFACT)
        # Re-create service to pick up refreshed config
        svc = PromptGalleryService(project_dir)
        result = svc.list_artifacts()
        by_name = {a["name"]: a for a in result}
        assert by_name[SAFE_ARTIFACT]["status"] == "user-owned"

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
        owned = sum(1 for a in result if a["status"] == "user-owned")
        assert framework + owned == len(result)


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

    def test_get_content_user_owned(self, service, project_dir):
        """After claim + modify, get_content returns the user's version."""
        service.scaffold_override(SAFE_ARTIFACT)
        custom = "# Custom safety rules\nDo not touch anything.\n"
        service.save_override(SAFE_ARTIFACT, custom)

        svc = PromptGalleryService(project_dir)
        result = svc.get_content(SAFE_ARTIFACT)
        assert result["source"] == "user-owned"
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

    def test_get_override_content_not_owned(self, service):
        """Non-owned artifact returns None."""
        result = service.get_override_content(SAFE_ARTIFACT)
        assert result is None

    def test_get_override_content_exists(self, service, project_dir):
        """After claiming, get_override_content returns file content."""
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
        """Claim without modification yields has_diff=False."""
        service.scaffold_override(SAFE_ARTIFACT)
        result = service.compute_diff(SAFE_ARTIFACT)
        assert result["has_diff"] is False
        assert result["additions"] == 0
        assert result["deletions"] == 0

    def test_compute_diff_with_changes(self, service, project_dir):
        """Claim, modify, diff shows additions and deletions."""
        service.scaffold_override(SAFE_ARTIFACT)
        service.save_override(SAFE_ARTIFACT, "# Completely replaced content\n")

        svc = PromptGalleryService(project_dir)
        result = svc.compute_diff(SAFE_ARTIFACT)
        assert result["has_diff"] is True
        assert result["additions"] > 0
        assert result["deletions"] > 0
        assert isinstance(result["unified_diff"], str)
        assert len(result["unified_diff"]) > 0

    def test_compute_diff_not_owned(self, service):
        """Diff on a non-owned artifact raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not user-owned"):
            service.compute_diff(SAFE_ARTIFACT)


# ===========================================================================
# Scaffold (Claim)
# ===========================================================================


class TestScaffold:
    """Tests for scaffold_override (claim)."""

    def test_scaffold_claims_and_updates_config(self, service, project_dir):
        """Claiming creates/keeps the file and updates config.yml."""
        result = service.scaffold_override(SAFE_ARTIFACT)

        assert result["status"] == "claimed"
        assert "output_path" in result
        assert len(result["content"]) > 0

        # config.yml has the user_owned entry
        user_owned = _get_user_owned(project_dir)
        assert SAFE_ARTIFACT in user_owned

    def test_scaffold_already_owned(self, service):
        """Claiming the same artifact twice raises FileExistsError."""
        service.scaffold_override(SAFE_ARTIFACT)
        with pytest.raises(FileExistsError, match="already user-owned"):
            service.scaffold_override(SAFE_ARTIFACT)


# ===========================================================================
# Save
# ===========================================================================


class TestSaveOverride:
    """Tests for save_override."""

    def test_save_override_writes_file(self, service, project_dir):
        """Save writes new content to the user-owned file."""
        service.scaffold_override(SAFE_ARTIFACT)
        new_content = "# Updated safety rules\nAll writes require approval.\n"
        result = service.save_override(SAFE_ARTIFACT, new_content)

        assert result["status"] == "saved"

        # Verify file content on disk
        output_file = project_dir / result["path"]
        assert output_file.read_text(encoding="utf-8") == new_content

    def test_save_override_not_owned(self, service):
        """Saving to a non-owned artifact raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not user-owned"):
            service.save_override(SAFE_ARTIFACT, "some content")


# ===========================================================================
# Unclaim
# ===========================================================================


class TestUnoverride:
    """Tests for unoverride (unclaim)."""

    def test_unclaim_removes_config(self, service, project_dir):
        """Unclaim removes the config entry."""
        service.scaffold_override(SAFE_ARTIFACT)

        svc = PromptGalleryService(project_dir)
        result = svc.unoverride(SAFE_ARTIFACT)

        assert result["status"] == "removed"

        # Config entry is gone
        user_owned = _get_user_owned(project_dir)
        assert SAFE_ARTIFACT not in user_owned

    def test_unclaim_not_owned(self, service):
        """Unclaiming a non-owned artifact raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not user-owned"):
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
        # No front matter in JSON templates -> falls back to registry
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
        assert art["summary"] == "Safety boundaries, channel write safety, and data integrity"
        assert "tool confinement" in art["description"]

    def test_command_has_summary_from_front_matter(self, service):
        """Command artifacts get summary from added front matter."""
        result = service.list_artifacts()
        by_name = {a["name"]: a for a in result}
        art = by_name["commands/diagnose"]
        assert art["summary"] == "Investigate operational failures"


# ===========================================================================
# Untracked file detection
# ===========================================================================


class TestScanUntracked:
    """Tests for scan_untracked — detecting files active in Claude Code but not managed."""

    def test_scan_untracked_finds_orphaned_files(self, service, project_dir):
        """A .md file in .claude/rules/ not in the registry is reported as untracked."""
        orphan = project_dir / ".claude" / "rules" / "my-custom-rule.md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_text("# My Custom Rule\nDo something special.\n", encoding="utf-8")

        result = service.scan_untracked()
        names = [u["canonical_name"] for u in result]
        assert "rules/my-custom-rule" in names

    def test_scan_untracked_excludes_registered(self, service, project_dir):
        """Registry artifacts that exist on disk are NOT reported as untracked."""
        safety_file = project_dir / ".claude" / "rules" / "safety.md"
        safety_file.parent.mkdir(parents=True, exist_ok=True)
        safety_file.write_text("# Safety\nExisting framework content.\n", encoding="utf-8")

        result = service.scan_untracked()
        names = [u["canonical_name"] for u in result]
        assert "rules/safety" not in names

    def test_scan_untracked_excludes_user_owned(self, service, project_dir):
        """Custom files already in user_owned are NOT reported as untracked."""
        orphan = project_dir / ".claude" / "rules" / "already-claimed.md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_text("# Already Claimed\n", encoding="utf-8")

        service.register_untracked("rules/already-claimed")

        svc = PromptGalleryService(project_dir)
        result = svc.scan_untracked()
        names = [u["canonical_name"] for u in result]
        assert "rules/already-claimed" not in names

    def test_scan_untracked_returns_correct_category(self, service, project_dir):
        """Category is derived from the first path component."""
        orphan = project_dir / ".claude" / "agents" / "rogue-agent.md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_text("# Rogue Agent\n", encoding="utf-8")

        result = service.scan_untracked()
        by_name = {u["canonical_name"]: u for u in result}
        assert by_name["agents/rogue-agent"]["category"] == "agents"

    def test_scan_untracked_returns_preview(self, service, project_dir):
        """Untracked files include a text preview."""
        orphan = project_dir / ".claude" / "rules" / "preview-test.md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        content = "# Preview Test\nSome content here.\n"
        orphan.write_text(content, encoding="utf-8")

        result = service.scan_untracked()
        by_name = {u["canonical_name"]: u for u in result}
        assert by_name["rules/preview-test"]["preview"] == content

    def test_scan_untracked_empty_when_no_orphans(self, service):
        """Returns empty list when all files are tracked."""
        result = service.scan_untracked()
        assert result == []


class TestRegisterUntracked:
    """Tests for register_untracked — adding custom files to config."""

    def test_register_untracked_adds_to_config(self, service, project_dir):
        """Registering adds the canonical name to prompts.user_owned in config."""
        orphan = project_dir / ".claude" / "rules" / "new-rule.md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_text("# New Rule\n", encoding="utf-8")

        result = service.register_untracked("rules/new-rule")
        assert result["status"] == "registered"

        user_owned = _get_user_owned(project_dir)
        assert "rules/new-rule" in user_owned

    def test_register_untracked_file_must_exist(self, service):
        """Registering a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found on disk"):
            service.register_untracked("rules/nonexistent")

    def test_register_untracked_already_registered(self, service, project_dir):
        """Registering an already-registered file raises FileExistsError."""
        orphan = project_dir / ".claude" / "rules" / "dupe.md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_text("# Dupe\n", encoding="utf-8")

        service.register_untracked("rules/dupe")
        svc = PromptGalleryService(project_dir)
        with pytest.raises(FileExistsError, match="already registered"):
            svc.register_untracked("rules/dupe")


class TestDeleteUntracked:
    """Tests for delete_untracked — removing orphaned files from disk."""

    def test_delete_untracked_removes_file(self, service, project_dir):
        """Deleting removes the file from disk."""
        orphan = project_dir / ".claude" / "rules" / "to-delete.md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_text("# Delete Me\n", encoding="utf-8")

        result = service.delete_untracked("rules/to-delete")
        assert result["status"] == "deleted"
        assert not orphan.exists()

    def test_delete_untracked_file_must_exist(self, service):
        """Deleting a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found on disk"):
            service.delete_untracked("rules/ghost")

    def test_delete_untracked_rejects_framework_artifact(self, service, project_dir):
        """Cannot delete a framework-registered artifact via delete_untracked."""
        safety_file = project_dir / ".claude" / "rules" / "safety.md"
        safety_file.parent.mkdir(parents=True, exist_ok=True)
        safety_file.write_text("# Safety\n", encoding="utf-8")

        with pytest.raises(ValueError, match="framework artifact"):
            service.delete_untracked("rules/safety")


class TestCustomArtifacts:
    """Tests for custom user artifacts appearing in list_artifacts and get_content."""

    def test_list_artifacts_includes_custom(self, service, project_dir):
        """After registering a custom file, it appears in list_artifacts."""
        orphan = project_dir / ".claude" / "rules" / "custom.md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_text("---\nsummary: My custom rule\n---\n# Custom\n", encoding="utf-8")

        service.register_untracked("rules/custom")

        svc = PromptGalleryService(project_dir)
        result = svc.list_artifacts()
        by_name = {a["name"]: a for a in result}
        assert "rules/custom" in by_name
        art = by_name["rules/custom"]
        assert art["status"] == "user-owned"
        assert art["custom"] is True
        assert art["category"] == "rules"
        assert art["summary"] == "My custom rule"

    def test_get_content_custom_artifact(self, service, project_dir):
        """get_content works for registered custom files."""
        orphan = project_dir / ".claude" / "rules" / "readable.md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        content = "# Readable Custom Rule\nContent here.\n"
        orphan.write_text(content, encoding="utf-8")

        service.register_untracked("rules/readable")

        svc = PromptGalleryService(project_dir)
        result = svc.get_content("rules/readable")
        assert result["source"] == "user-owned"
        assert result["content"] == content

    def test_get_content_unknown_artifact_raises(self, service):
        """get_content for a completely unknown name raises KeyError."""
        with pytest.raises(KeyError, match="Unknown artifact"):
            service.get_content("rules/totally-unknown")

    def test_save_override_custom_artifact(self, service, project_dir):
        """save_override works for registered custom files."""
        orphan = project_dir / ".claude" / "rules" / "editable.md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_text("# Original\n", encoding="utf-8")

        service.register_untracked("rules/editable")

        svc = PromptGalleryService(project_dir)
        new_content = "# Edited Custom Rule\nUpdated content.\n"
        result = svc.save_override("rules/editable", new_content)
        assert result["status"] == "saved"
        assert orphan.read_text(encoding="utf-8") == new_content

    def test_unoverride_custom_with_delete(self, service, project_dir):
        """Unclaiming a custom artifact with delete_file=True removes the file."""
        orphan = project_dir / ".claude" / "rules" / "removable.md"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_text("# Removable\n", encoding="utf-8")

        service.register_untracked("rules/removable")

        svc = PromptGalleryService(project_dir)
        result = svc.unoverride("rules/removable", delete_file=True)
        assert result["status"] == "removed"
        assert result["deleted_file"] is True
        assert not orphan.exists()

        user_owned = _get_user_owned(project_dir)
        assert "rules/removable" not in user_owned
