"""Tests for ClaudeCodeFileService."""

from __future__ import annotations

import pytest

from osprey.interfaces.web_terminal.claude_code_files import ClaudeCodeFileService


@pytest.fixture
def project_dir(tmp_path):
    """Create a minimal project directory with Claude Code files."""
    # Root files
    (tmp_path / "CLAUDE.md").write_text("# Test CLAUDE.md\n")
    (tmp_path / ".mcp.json").write_text('{"servers": {}}\n')

    # .claude subdirectories
    claude = tmp_path / ".claude"
    claude.mkdir()
    (claude / "settings.json").write_text('{"permissions": {}}\n')

    rules = claude / "rules"
    rules.mkdir()
    (rules / "safety.md").write_text("# Safety rules\n")

    agents = claude / "agents"
    agents.mkdir()
    (agents / "test-agent.md").write_text("# Test agent\n")

    hooks = claude / "hooks"
    hooks.mkdir()
    (hooks / "pre-check.sh").write_text("#!/bin/bash\necho ok\n")

    return tmp_path


@pytest.fixture
def service(project_dir):
    return ClaudeCodeFileService(project_dir)


class TestListFiles:
    def test_discovers_expected_files(self, service):
        files = service.list_files()
        paths = {f["path"] for f in files}

        assert "CLAUDE.md" in paths
        assert ".mcp.json" in paths
        assert ".claude/settings.json" in paths
        assert ".claude/rules/safety.md" in paths
        assert ".claude/agents/test-agent.md" in paths
        assert ".claude/hooks/pre-check.sh" in paths

    def test_categories_assigned(self, service):
        files = service.list_files()
        by_path = {f["path"]: f for f in files}

        assert by_path["CLAUDE.md"]["category"] == "System Prompt"
        assert by_path[".mcp.json"]["category"] == "MCP Servers"
        assert by_path[".claude/settings.json"]["category"] == "Permissions"
        assert by_path[".claude/rules/safety.md"]["category"] == "Safety"
        assert by_path[".claude/agents/test-agent.md"]["category"] == "Agents"
        assert by_path[".claude/hooks/pre-check.sh"]["category"] == "Hooks"

    def test_languages_detected(self, service):
        files = service.list_files()
        by_path = {f["path"]: f for f in files}

        assert by_path["CLAUDE.md"]["language"] == "markdown"
        assert by_path[".mcp.json"]["language"] == "json"
        assert by_path[".claude/hooks/pre-check.sh"]["language"] == "shell"

    def test_skips_hidden_files(self, project_dir, service):
        (project_dir / ".claude" / "rules" / ".hidden").write_text("secret\n")
        files = service.list_files()
        paths = {f["path"] for f in files}
        assert ".claude/rules/.hidden" not in paths

    def test_empty_project(self, tmp_path):
        svc = ClaudeCodeFileService(tmp_path)
        files = svc.list_files()
        assert files == []


class TestReadFile:
    def test_reads_existing_file(self, service):
        result = service.read_file("CLAUDE.md")
        assert result["content"] == "# Test CLAUDE.md\n"
        assert result["name"] == "CLAUDE.md"
        assert result["language"] == "markdown"

    def test_file_not_found(self, service):
        with pytest.raises(FileNotFoundError):
            service.read_file("nonexistent.md")


class TestWriteFile:
    def test_writes_existing_file(self, service, project_dir):
        result = service.write_file("CLAUDE.md", "# Updated\n")
        assert result["status"] == "saved"
        assert (project_dir / "CLAUDE.md").read_text() == "# Updated\n"

    def test_traversal_blocked(self, service):
        with pytest.raises(PermissionError, match="traversal"):
            service.write_file("../../etc/passwd", "hacked")

    def test_validates_json(self, service):
        with pytest.raises(ValueError, match="Invalid JSON"):
            service.write_file(".mcp.json", "not valid json {{{")

    def test_valid_json_accepted(self, service, project_dir):
        new_content = '{"servers": {"test": {}}}'
        result = service.write_file(".mcp.json", new_content)
        assert result["status"] == "saved"
        assert (project_dir / ".mcp.json").read_text() == new_content

    def test_file_not_found_returns_error(self, service):
        with pytest.raises(FileNotFoundError):
            service.write_file("does-not-exist.md", "content")

    def test_markdown_no_validation(self, service, project_dir):
        """Markdown files should not be syntax-checked."""
        weird_content = "{{{{not yaml not json\n"
        result = service.write_file("CLAUDE.md", weird_content)
        assert result["status"] == "saved"


class TestCreateFile:
    def test_create_in_allowed_dir(self, service, project_dir):
        result = service.create_file(".claude/rules/new-rule.md", "# New rule\n")
        assert result["status"] == "created"
        assert (project_dir / ".claude" / "rules" / "new-rule.md").exists()
        assert result["category"] == "Safety"

    def test_create_in_agents_dir(self, service, project_dir):
        result = service.create_file(".claude/agents/my-agent.md", "# Agent\n")
        assert result["status"] == "created"
        assert result["category"] == "Agents"

    def test_create_outside_allowed_dir(self, service):
        with pytest.raises(PermissionError, match="must be in .claude"):
            service.create_file("src/malicious.py", "import os\n")

    def test_create_in_root(self, service):
        with pytest.raises(PermissionError, match="must be in .claude"):
            service.create_file("evil.md", "# Evil\n")

    def test_create_file_already_exists(self, service):
        with pytest.raises(FileExistsError):
            service.create_file(".claude/rules/safety.md", "# Duplicate\n")

    def test_create_validates_json(self, service):
        with pytest.raises(ValueError, match="Invalid JSON"):
            service.create_file(".claude/commands/bad.json", "not json")

    def test_create_subdirectory(self, service, project_dir):
        """Should create parent directories if needed."""
        result = service.create_file(".claude/commands/sub/deep.md", "# Deep\n")
        assert result["status"] == "created"
        assert (project_dir / ".claude" / "commands" / "sub" / "deep.md").exists()

    def test_create_traversal_blocked(self, service):
        with pytest.raises(PermissionError, match="traversal"):
            service.create_file("../../etc/evil.md", "hacked")


class TestCategorize:
    def test_known_files(self):
        assert ClaudeCodeFileService.categorize("CLAUDE.md", "CLAUDE.md") == "System Prompt"
        assert ClaudeCodeFileService.categorize(".mcp.json", ".mcp.json") == "MCP Servers"

    def test_agent_file(self):
        assert (
            ClaudeCodeFileService.categorize(
                "resolver-agent.md", ".claude/agents/resolver-agent.md"
            )
            == "Agents"
        )

    def test_hooks_file(self):
        assert (
            ClaudeCodeFileService.categorize("pre-check.sh", ".claude/hooks/pre-check.sh")
            == "Hooks"
        )

    def test_commands_file(self):
        assert (
            ClaudeCodeFileService.categorize("deploy.md", ".claude/commands/deploy.md")
            == "Commands"
        )

    def test_unknown_file(self):
        assert ClaudeCodeFileService.categorize("random.txt", "random.txt") == "Other"


class TestDetectLanguage:
    def test_markdown(self):
        assert ClaudeCodeFileService.detect_language("file.md") == "markdown"

    def test_json(self):
        assert ClaudeCodeFileService.detect_language("config.json") == "json"

    def test_yaml_variants(self):
        assert ClaudeCodeFileService.detect_language("config.yml") == "yaml"
        assert ClaudeCodeFileService.detect_language("config.yaml") == "yaml"

    def test_shell(self):
        assert ClaudeCodeFileService.detect_language("script.sh") == "shell"

    def test_python(self):
        assert ClaudeCodeFileService.detect_language("script.py") == "python"

    def test_unknown(self):
        assert ClaudeCodeFileService.detect_language("file.txt") == "text"
