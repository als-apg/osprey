"""Tests for project migration command and manifest generation.

Tests the manifest generation during project init and the
migration command for upgrading projects between versions.
"""

import json

import pytest
from click.testing import CliRunner

from osprey.cli.init_cmd import init
from osprey.cli.migrate_cmd import (
    FileCategory,
    _calculate_file_hash,
    _classify_file,
    _detect_project_settings,
    _load_manifest,
    migrate,
)
from osprey.cli.templates import (
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    TemplateManager,
)


class TestManifestGeneration:
    """Test manifest generation during project creation."""

    def test_manifest_created_on_init(self, tmp_path):
        """Test that manifest is created when project is initialized."""
        runner = CliRunner()

        result = runner.invoke(init, ["test-project", "--output-dir", str(tmp_path)])

        assert result.exit_code == 0

        manifest_path = tmp_path / "test-project" / MANIFEST_FILENAME
        assert manifest_path.exists(), "Manifest should be created during init"

    def test_manifest_schema_version(self, tmp_path):
        """Test that manifest has correct schema version."""
        manager = TemplateManager()
        project_dir = manager.create_project("test-app", tmp_path, "minimal")

        # Generate manifest
        manifest = manager.generate_manifest(
            project_dir=project_dir,
            project_name="test-app",
            template_name="minimal",
            registry_style="extend",
            context={"default_provider": "cborg"},
        )

        assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION

    def test_manifest_creation_fields(self, tmp_path):
        """Test that manifest has all required creation fields."""
        manager = TemplateManager()
        project_dir = manager.create_project("test-app", tmp_path, "minimal")

        manifest = manager.generate_manifest(
            project_dir=project_dir,
            project_name="test-app",
            template_name="minimal",
            registry_style="extend",
            context={},
        )

        # Check creation section
        assert "creation" in manifest
        creation = manifest["creation"]
        assert "osprey_version" in creation
        assert "timestamp" in creation
        assert "template" in creation
        assert "registry_style" in creation

        assert creation["template"] == "minimal"
        assert creation["registry_style"] == "extend"

    def test_manifest_init_args(self, tmp_path):
        """Test that manifest captures init arguments."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            "test-app",
            tmp_path,
            "control_assistant",
            registry_style="extend",
            context={
                "default_provider": "cborg",
                "default_model": "claude-haiku",
                "channel_finder_mode": "all",
            },
        )

        manifest = manager.generate_manifest(
            project_dir=project_dir,
            project_name="test-app",
            template_name="control_assistant",
            registry_style="extend",
            context={
                "default_provider": "cborg",
                "default_model": "claude-haiku",
                "channel_finder_mode": "all",
            },
        )

        assert "init_args" in manifest
        args = manifest["init_args"]
        assert args["project_name"] == "test-app"
        assert args["template"] == "control_assistant"
        assert args["registry_style"] == "extend"
        assert args["provider"] == "cborg"
        assert args["model"] == "claude-haiku"
        assert args["channel_finder_mode"] == "all"

    def test_manifest_reproducible_command(self, tmp_path):
        """Test that manifest includes reproducible command."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            "my-project",
            tmp_path,
            "control_assistant",
            context={"default_provider": "openai"},
        )

        manifest = manager.generate_manifest(
            project_dir=project_dir,
            project_name="my-project",
            template_name="control_assistant",
            registry_style="extend",
            context={"default_provider": "openai"},
        )

        assert "reproducible_command" in manifest
        cmd = manifest["reproducible_command"]
        assert "osprey init my-project" in cmd
        assert "--template control_assistant" in cmd
        assert "--provider openai" in cmd

    def test_manifest_file_checksums(self, tmp_path):
        """Test that manifest includes file checksums."""
        manager = TemplateManager()
        project_dir = manager.create_project("test-app", tmp_path, "minimal")

        manifest = manager.generate_manifest(
            project_dir=project_dir,
            project_name="test-app",
            template_name="minimal",
            registry_style="extend",
            context={},
        )

        assert "file_checksums" in manifest
        checksums = manifest["file_checksums"]

        # Should have checksums for key files
        assert "config.yml" in checksums
        assert "README.md" in checksums
        assert "pyproject.toml" in checksums

        # Checksums should be in expected format
        for path, checksum in checksums.items():
            assert checksum.startswith("sha256:"), f"Checksum for {path} should start with sha256:"

    def test_manifest_excludes_env_files(self, tmp_path):
        """Test that manifest excludes .env files from checksums."""
        manager = TemplateManager()
        project_dir = manager.create_project("test-app", tmp_path, "minimal")

        # Create a .env file
        (project_dir / ".env").write_text("SECRET=value")

        manifest = manager.generate_manifest(
            project_dir=project_dir,
            project_name="test-app",
            template_name="minimal",
            registry_style="extend",
            context={},
        )

        checksums = manifest["file_checksums"]
        assert ".env" not in checksums

    def test_manifest_written_to_file(self, tmp_path):
        """Test that manifest is written to JSON file."""
        manager = TemplateManager()
        project_dir = manager.create_project("test-app", tmp_path, "minimal")

        manager.generate_manifest(
            project_dir=project_dir,
            project_name="test-app",
            template_name="minimal",
            registry_style="extend",
            context={},
        )

        manifest_path = project_dir / MANIFEST_FILENAME
        assert manifest_path.exists()

        # Should be valid JSON
        with open(manifest_path) as f:
            loaded = json.load(f)

        assert loaded["schema_version"] == MANIFEST_SCHEMA_VERSION


class TestFileClassification:
    """Test file classification logic for migration."""

    def test_classify_data_directory(self):
        """Test that data directories are classified as DATA."""
        result = _classify_file("data/channels.json", "abc", "def", "ghi")
        assert result == FileCategory.DATA

        result = _classify_file("_agent_data/scripts/test.py", "abc", "def", "ghi")
        assert result == FileCategory.DATA

    def test_classify_new_file(self):
        """Test that new template files are classified as NEW."""
        result = _classify_file("new_feature.py", None, None, "abc123")
        assert result == FileCategory.NEW

    def test_classify_removed_file(self):
        """Test that removed template files are classified as REMOVED."""
        result = _classify_file("old_feature.py", None, "abc123", None)
        assert result == FileCategory.REMOVED

    def test_classify_auto_copy(self):
        """Test auto-copy classification (template changed, facility didn't)."""
        # facility == old_vanilla, but old_vanilla != new_vanilla
        result = _classify_file("config.py", "abc", "abc", "def")
        assert result == FileCategory.AUTO_COPY

    def test_classify_preserve(self):
        """Test preserve classification (facility changed, template didn't)."""
        # facility != old_vanilla, but old_vanilla == new_vanilla
        result = _classify_file("config.py", "custom", "original", "original")
        assert result == FileCategory.PRESERVE

    def test_classify_merge(self):
        """Test merge classification (both changed)."""
        # facility != old_vanilla AND old_vanilla != new_vanilla
        result = _classify_file("config.py", "custom", "original", "new")
        assert result == FileCategory.MERGE

    def test_classify_unchanged(self):
        """Test unchanged files are preserved."""
        # All hashes equal - preserve (no action needed)
        result = _classify_file("config.py", "same", "same", "same")
        assert result == FileCategory.PRESERVE


class TestProjectSettingsDetection:
    """Test detection of project settings for retroactive manifest creation."""

    def test_detect_from_config_yml(self, tmp_path):
        """Test detecting settings from config.yml."""
        # Create a project directory with config.yml
        project_dir = tmp_path / "test-project"
        project_dir.mkdir()

        config_content = """
llm:
  default_provider: openai
  default_model: gpt-4
channel_finder:
  default_pipeline: hierarchical
"""
        (project_dir / "config.yml").write_text(config_content)

        settings = _detect_project_settings(project_dir)

        assert settings["provider"] == "openai"
        assert settings["model"] == "gpt-4"
        assert settings["template"] == "control_assistant"  # Detected from channel_finder

    def test_detect_registry_style(self, tmp_path):
        """Test detecting registry style from registry.py."""
        project_dir = tmp_path / "test-project"
        src_dir = project_dir / "src" / "test_project"
        src_dir.mkdir(parents=True)

        registry_content = """
from osprey.registry import extend_framework_registry, OspreyFrameworkRegistry

def get_registry():
    return extend_framework_registry(...)
"""
        (src_dir / "registry.py").write_text(registry_content)

        settings = _detect_project_settings(project_dir)

        assert settings["registry_style"] == "extend"
        assert settings["package_name"] == "test_project"

    def test_detect_code_generator(self, tmp_path):
        """Test detecting code generator from config files."""
        project_dir = tmp_path / "test-project"
        project_dir.mkdir()

        (project_dir / "claude_generator_config.yml").write_text("enabled: true")

        settings = _detect_project_settings(project_dir)

        assert settings["code_generator"] == "claude_code"


class TestManifestLoading:
    """Test manifest loading functionality."""

    def test_load_valid_manifest(self, tmp_path):
        """Test loading a valid manifest file."""
        manifest_data = {
            "schema_version": "1.0.0",
            "creation": {"osprey_version": "0.10.0"},
        }

        manifest_path = tmp_path / MANIFEST_FILENAME
        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f)

        result = _load_manifest(tmp_path)

        assert result is not None
        assert result["schema_version"] == "1.0.0"

    def test_load_missing_manifest(self, tmp_path):
        """Test loading when manifest doesn't exist."""
        result = _load_manifest(tmp_path)
        assert result is None

    def test_load_invalid_json(self, tmp_path):
        """Test loading invalid JSON gracefully."""
        manifest_path = tmp_path / MANIFEST_FILENAME
        manifest_path.write_text("not valid json {{{")

        result = _load_manifest(tmp_path)
        assert result is None


class TestFileHashing:
    """Test file hashing utilities."""

    def test_hash_file(self, tmp_path):
        """Test hashing a file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        result = _calculate_file_hash(test_file)

        assert result is not None
        assert len(result) == 64  # SHA256 hex length

    def test_hash_same_content(self, tmp_path):
        """Test that same content produces same hash."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"

        file1.write_text("identical content")
        file2.write_text("identical content")

        assert _calculate_file_hash(file1) == _calculate_file_hash(file2)

    def test_hash_different_content(self, tmp_path):
        """Test that different content produces different hash."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"

        file1.write_text("content A")
        file2.write_text("content B")

        assert _calculate_file_hash(file1) != _calculate_file_hash(file2)

    def test_hash_missing_file(self, tmp_path):
        """Test hashing a missing file returns None."""
        result = _calculate_file_hash(tmp_path / "nonexistent.txt")
        assert result is None


class TestMigrateCLI:
    """Test migration CLI commands."""

    def test_migrate_check_no_manifest(self, tmp_path):
        """Test migrate check with no manifest."""
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(migrate, ["check"])

            assert "No manifest found" in result.output

    def test_migrate_check_with_manifest(self, tmp_path):
        """Test migrate check with existing manifest."""
        runner = CliRunner()

        # Create project with manifest
        result = runner.invoke(init, ["test-project", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0

        # Run check from project directory
        project_dir = tmp_path / "test-project"
        result = runner.invoke(migrate, ["check", "--project", str(project_dir)])

        assert "Project OSPREY version:" in result.output
        assert "Installed OSPREY version:" in result.output

    def test_migrate_init_creates_manifest(self, tmp_path):
        """Test that migrate init creates manifest for existing project."""
        runner = CliRunner()

        # Create project structure without manifest
        project_dir = tmp_path / "existing-project"
        src_dir = project_dir / "src" / "existing_project"
        src_dir.mkdir(parents=True)

        # Create minimal config
        config_content = """
llm:
  default_provider: cborg
  default_model: claude-haiku
"""
        (project_dir / "config.yml").write_text(config_content)

        # Create registry
        registry_content = """
from osprey.registry import extend_framework_registry

def get_registry():
    return extend_framework_registry(
        app_class_name="ExistingProject",
        app_display_name="existing-project"
    )
"""
        (src_dir / "registry.py").write_text(registry_content)

        # Run migrate init
        result = runner.invoke(
            migrate,
            ["init", "--project", str(project_dir), "--version", "0.10.0"],
            input="\n",  # Accept default version
        )

        assert result.exit_code == 0
        assert (project_dir / MANIFEST_FILENAME).exists()

    def test_migrate_init_force_overwrite(self, tmp_path):
        """Test that migrate init --force overwrites existing manifest."""
        runner = CliRunner()

        # Create project with manifest
        result = runner.invoke(init, ["test-project", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0

        project_dir = tmp_path / "test-project"

        # Try to init again without force
        result = runner.invoke(migrate, ["init", "--project", str(project_dir)])
        assert "already exists" in result.output

        # Try with force
        result = runner.invoke(
            migrate,
            ["init", "--project", str(project_dir), "--force", "--version", "0.9.0"],
        )
        assert result.exit_code == 0


class TestMigrateIntegration:
    """Integration tests for full migration workflow."""

    def test_full_workflow_dry_run(self, tmp_path):
        """Test full migration workflow in dry-run mode."""
        runner = CliRunner()

        # Create project
        result = runner.invoke(init, ["test-project", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0

        project_dir = tmp_path / "test-project"

        # Run migration dry-run
        # Note: This will likely fail to recreate exact old version, but should
        # handle gracefully with --use-current-version
        result = runner.invoke(
            migrate,
            ["run", "--project", str(project_dir), "--use-current-version", "--dry-run"],
        )

        # Should complete and create _migration directory
        assert "Migration" in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
