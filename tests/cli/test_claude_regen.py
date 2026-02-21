"""Tests for Claude Code artifact regeneration.

Tests that `osprey claude regen` correctly rebuilds Claude Code artifacts
from config.yml, preserves user files, creates backups, and maintains
safety hooks across regeneration cycles.
"""

import json
import os

import pytest
import yaml

from osprey.cli.templates import MANIFEST_FILENAME, TemplateManager


class TestBuildClaudeCodeContext:
    """Test _build_claude_code_context() reconstructs correct template vars."""

    def test_minimal_config(self, tmp_path):
        """Minimal config produces correct base context vars."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="ctx-minimal",
            output_dir=tmp_path,
            template_name="minimal",
        )

        config = yaml.safe_load((project_dir / "config.yml").read_text())
        ctx = manager._build_claude_code_context(project_dir, config)

        assert ctx["project_name"] == "ctx-minimal"
        assert ctx["package_name"] == "ctx_minimal"
        assert ctx["project_root"] == str(project_dir.absolute())
        assert "current_python_env" in ctx
        assert ctx["template_name"] == "minimal"

    def test_control_assistant_config(self, tmp_path):
        """Control assistant config produces channel_finder context vars."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="ctx-control",
            output_dir=tmp_path,
            template_name="control_assistant",
        )
        # Generate manifest so _build_claude_code_context can discover template_name
        manager.generate_manifest(
            project_dir, "ctx-control", "control_assistant", "extend", {}
        )

        config = yaml.safe_load((project_dir / "config.yml").read_text())
        ctx = manager._build_claude_code_context(project_dir, config)

        assert ctx["template_name"] == "control_assistant"
        assert "channel_finder_pipeline" in ctx
        assert "channel_finder_mode" in ctx

    def test_config_with_confluence(self, tmp_path):
        """Config with confluence section passes dict to context."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="ctx-confluence",
            output_dir=tmp_path,
            template_name="minimal",
        )

        # Add confluence to config
        config = yaml.safe_load((project_dir / "config.yml").read_text())
        config["confluence"] = {"url": "https://wiki.example.com"}
        (project_dir / "config.yml").write_text(yaml.dump(config))

        ctx = manager._build_claude_code_context(project_dir, config)
        assert "confluence" in ctx
        assert ctx["confluence"]["url"] == "https://wiki.example.com"

    def test_config_with_matlab(self, tmp_path):
        """Config with matlab section passes dict to context."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="ctx-matlab",
            output_dir=tmp_path,
            template_name="minimal",
        )

        config = yaml.safe_load((project_dir / "config.yml").read_text())
        config["matlab"] = {"db_path": "~/.matlab-mml/mml.db"}
        (project_dir / "config.yml").write_text(yaml.dump(config))

        ctx = manager._build_claude_code_context(project_dir, config)
        assert "matlab" in ctx
        assert ctx["matlab"]["db_path"] == "~/.matlab-mml/mml.db"

    def test_uses_manifest_template(self, tmp_path):
        """When manifest exists, template_name is read from it."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="ctx-manifest",
            output_dir=tmp_path,
            template_name="control_assistant",
        )
        # Generate manifest
        manager.generate_manifest(
            project_dir, "ctx-manifest", "control_assistant", "extend", {}
        )

        config = yaml.safe_load((project_dir / "config.yml").read_text())
        ctx = manager._build_claude_code_context(project_dir, config)

        assert ctx["template_name"] == "control_assistant"


class TestRegenerationCorrectness:
    """Test that regeneration produces correct output."""

    def test_regen_produces_same_output_as_init(self, tmp_path):
        """Init → regen with no config changes → identical artifacts."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="regen-idempotent",
            output_dir=tmp_path,
            template_name="minimal",
        )

        # Capture checksums
        files_to_check = [".mcp.json", "CLAUDE.md", ".claude/settings.json"]
        original_checksums = {}
        for f in files_to_check:
            fp = project_dir / f
            if fp.exists():
                original_checksums[f] = fp.read_text()

        # Regen
        result = manager.regenerate_claude_code(project_dir)

        # Compare
        for f, original_content in original_checksums.items():
            assert (project_dir / f).read_text() == original_content, f"{f} changed unexpectedly"
        assert f in result["unchanged"] or not result["changed"]

    def test_regen_updates_mcp_json_when_confluence_added(self, tmp_path):
        """Adding confluence to config.yml → .mcp.json gets confluence server."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="regen-confluence",
            output_dir=tmp_path,
            template_name="minimal",
        )

        # Verify confluence NOT in .mcp.json initially
        mcp_data = json.loads((project_dir / ".mcp.json").read_text())
        assert "confluence" not in mcp_data["mcpServers"]

        # Add confluence to config.yml
        config = yaml.safe_load((project_dir / "config.yml").read_text())
        config["confluence"] = {"url": "https://wiki.example.com"}
        (project_dir / "config.yml").write_text(yaml.dump(config))

        # Regen
        result = manager.regenerate_claude_code(project_dir)

        # Verify
        mcp_data = json.loads((project_dir / ".mcp.json").read_text())
        assert "confluence" in mcp_data["mcpServers"]
        assert ".mcp.json" in result["changed"]

    def test_regen_updates_settings_json_when_confluence_added(self, tmp_path):
        """Adding confluence → settings.json gets confluence permissions."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="regen-settings",
            output_dir=tmp_path,
            template_name="minimal",
        )

        # Add confluence
        config = yaml.safe_load((project_dir / "config.yml").read_text())
        config["confluence"] = {"url": "https://wiki.example.com"}
        (project_dir / "config.yml").write_text(yaml.dump(config))

        manager.regenerate_claude_code(project_dir)

        settings = json.loads((project_dir / ".claude" / "settings.json").read_text())
        assert "mcp__confluence" in settings["permissions"]["allow"]
        assert "Task(wiki-search)" in settings["permissions"]["allow"]

    def test_regen_updates_claude_md_when_confluence_added(self, tmp_path):
        """Adding confluence → CLAUDE.md mentions wiki-search delegation."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="regen-claude-md",
            output_dir=tmp_path,
            template_name="minimal",
        )

        # Initially no wiki section
        content = (project_dir / "CLAUDE.md").read_text()
        assert "wiki-search" not in content

        # Add confluence
        config = yaml.safe_load((project_dir / "config.yml").read_text())
        config["confluence"] = {"url": "https://wiki.example.com"}
        (project_dir / "config.yml").write_text(yaml.dump(config))

        manager.regenerate_claude_code(project_dir)

        content = (project_dir / "CLAUDE.md").read_text()
        assert "wiki-search" in content

    def test_regen_removes_features_when_config_section_removed(self, tmp_path):
        """Init with confluence → remove section → regen → confluence gone."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="regen-remove",
            output_dir=tmp_path,
            template_name="minimal",
            context={"confluence": {"url": "https://wiki.example.com"}},
        )

        # Confluence should be present
        mcp_data = json.loads((project_dir / ".mcp.json").read_text())
        assert "confluence" in mcp_data["mcpServers"]

        # Remove confluence from config.yml
        config = yaml.safe_load((project_dir / "config.yml").read_text())
        config.pop("confluence", None)
        (project_dir / "config.yml").write_text(yaml.dump(config))

        manager.regenerate_claude_code(project_dir)

        # Confluence should be gone
        mcp_data = json.loads((project_dir / ".mcp.json").read_text())
        assert "confluence" not in mcp_data["mcpServers"]


class TestSafetyPreservation:
    """Test that regeneration always preserves safety layers."""

    @pytest.fixture()
    def regen_project(self, tmp_path):
        """Create and regenerate a project."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="safety-test",
            output_dir=tmp_path,
            template_name="minimal",
        )
        manager.regenerate_claude_code(project_dir)
        return project_dir

    def test_always_includes_safety_hooks(self, regen_project):
        """After regen, settings.json still has PreToolUse/PostToolUse hook chains."""
        settings = json.loads(
            (regen_project / ".claude" / "settings.json").read_text()
        )
        assert "PreToolUse" in settings["hooks"]
        assert "PostToolUse" in settings["hooks"]

    def test_always_denies_dangerous_tools(self, regen_project):
        """After regen, settings.json still denies Bash, Edit, Write, WebFetch, WebSearch."""
        settings = json.loads(
            (regen_project / ".claude" / "settings.json").read_text()
        )
        deny = settings["permissions"]["deny"]
        for tool in ["Bash", "Edit", "Write", "WebFetch", "WebSearch"]:
            assert tool in deny, f"{tool} should be in deny list after regen"

    def test_preserves_writes_check_hook(self, regen_project):
        """osprey_writes_check.py is in PreToolUse for channel_write after regen."""
        settings = json.loads(
            (regen_project / ".claude" / "settings.json").read_text()
        )
        pre_tool_use = settings["hooks"]["PreToolUse"]
        channel_write_hooks = [
            h for h in pre_tool_use
            if h.get("matcher") == "mcp__controls__channel_write"
        ]
        assert len(channel_write_hooks) > 0
        hook_commands = [
            hook["command"]
            for entry in channel_write_hooks
            for hook in entry["hooks"]
        ]
        assert any("osprey_writes_check.py" in cmd for cmd in hook_commands)

    def test_hooks_remain_executable(self, regen_project):
        """After regen, all hook .py files retain executable permissions."""
        hooks_dir = regen_project / ".claude" / "hooks"
        for hook in hooks_dir.iterdir():
            if hook.is_file() and hook.suffix == ".py":
                mode = os.stat(hook).st_mode
                assert mode & 0o111, f"Hook {hook.name} should be executable after regen"


class TestUserFilePreservation:
    """Test that regeneration preserves user-maintained files."""

    def test_creates_backup(self, tmp_path):
        """Regen creates backup directory in osprey-workspace/backup/."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="backup-test",
            output_dir=tmp_path,
            template_name="minimal",
        )

        result = manager.regenerate_claude_code(project_dir)

        backup_dir = result["backup_dir"]
        assert backup_dir is not None
        assert os.path.exists(backup_dir)
        # Backup should contain the original files
        assert os.path.exists(os.path.join(backup_dir, ".mcp.json"))
        assert os.path.exists(os.path.join(backup_dir, "CLAUDE.md"))

    def test_claude_md_has_generated_header(self, tmp_path):
        """CLAUDE.md has the generated-file header comment."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="header-test",
            output_dir=tmp_path,
            template_name="minimal",
        )

        content = (project_dir / "CLAUDE.md").read_text()
        assert "GENERATED BY OSPREY" in content
        assert "osprey prompts scaffold" in content


class TestErrorHandling:
    """Test error handling in regeneration."""

    def test_no_config_error(self, tmp_path):
        """Run in directory without config.yml → clear error."""
        manager = TemplateManager()

        with pytest.raises(FileNotFoundError, match="No config.yml found"):
            manager.regenerate_claude_code(tmp_path)

    def test_dry_run_no_changes(self, tmp_path):
        """Dry run does not modify any files."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="dry-run-test",
            output_dir=tmp_path,
            template_name="minimal",
        )

        # Record file mtimes
        mcp_mtime = (project_dir / ".mcp.json").stat().st_mtime

        result = manager.regenerate_claude_code(project_dir, dry_run=True)

        assert result["backup_dir"] is None
        # File should not be modified
        assert (project_dir / ".mcp.json").stat().st_mtime == mcp_mtime

    def test_dry_run_detects_changes(self, tmp_path):
        """Dry run correctly detects what would change."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="dry-detect",
            output_dir=tmp_path,
            template_name="minimal",
        )

        # Add confluence to config (will cause .mcp.json to change)
        config = yaml.safe_load((project_dir / "config.yml").read_text())
        config["confluence"] = {"url": "https://wiki.example.com"}
        (project_dir / "config.yml").write_text(yaml.dump(config))

        result = manager.regenerate_claude_code(project_dir, dry_run=True)

        assert ".mcp.json" in result["changed"]


class TestGitignore:
    """Test that gitignore includes generated artifacts."""

    def test_gitignore_includes_generated_artifacts(self, tmp_path):
        """Project .gitignore includes CLAUDE.md, .mcp.json, .claude/."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="gitignore-gen-test",
            output_dir=tmp_path,
            template_name="minimal",
        )

        gitignore = (project_dir / ".gitignore").read_text()
        assert "CLAUDE.md" in gitignore
        assert ".mcp.json" in gitignore
        assert ".claude/" in gitignore


class TestDisableServers:
    """Test disable_servers functionality."""

    def _create_and_regen(self, tmp_path, disable_servers=None, disable_agents=None,
                          extra_servers=None, template="minimal"):
        """Helper: create project, set claude_code overrides, regen."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="disable-test",
            output_dir=tmp_path,
            template_name=template,
        )

        config = yaml.safe_load((project_dir / "config.yml").read_text())
        cc = {}
        if disable_servers:
            cc["disable_servers"] = disable_servers
        if disable_agents:
            cc["disable_agents"] = disable_agents
        if extra_servers:
            cc["extra_servers"] = extra_servers
        if cc:
            config["claude_code"] = cc
            (project_dir / "config.yml").write_text(yaml.dump(config))

        result = manager.regenerate_claude_code(project_dir)
        return project_dir, result

    def test_disable_server_removes_from_mcp_json(self, tmp_path):
        """Disabling accelpapers removes it from .mcp.json."""
        project_dir, _ = self._create_and_regen(
            tmp_path, disable_servers=["accelpapers"]
        )

        mcp_data = json.loads((project_dir / ".mcp.json").read_text())
        assert "accelpapers" not in mcp_data["mcpServers"]
        # Core servers should still be present
        assert "controls" in mcp_data["mcpServers"]

    def test_disable_server_removes_from_settings_allow(self, tmp_path):
        """Disabling accelpapers removes its tools from settings allow list."""
        project_dir, _ = self._create_and_regen(
            tmp_path, disable_servers=["accelpapers"]
        )

        settings = json.loads((project_dir / ".claude" / "settings.json").read_text())
        allow = settings["permissions"]["allow"]
        for entry in allow:
            assert "accelpapers" not in entry, f"accelpapers found in allow: {entry}"

    def test_disable_server_removes_from_claude_md(self, tmp_path):
        """Disabling accelpapers removes accelpapers row from CLAUDE.md tool table."""
        project_dir, _ = self._create_and_regen(
            tmp_path, disable_servers=["accelpapers"]
        )

        content = (project_dir / "CLAUDE.md").read_text()
        assert "accelpapers" not in content.lower() or "accelpapers" not in content

    def test_disable_agent_removes_agent_file(self, tmp_path):
        """Disabling literature-search produces an empty agent file."""
        project_dir, _ = self._create_and_regen(
            tmp_path, disable_agents=["literature-search"]
        )

        agent_file = project_dir / ".claude" / "agents" / "literature-search.md"
        if agent_file.exists():
            content = agent_file.read_text().strip()
            # Disabled agents render to empty (or whitespace-only) files
            assert content == "", f"Expected empty file, got: {content[:100]}"

    def test_disable_agent_removes_task_from_settings(self, tmp_path):
        """Disabling literature-search removes Task(literature-search) from allow."""
        project_dir, _ = self._create_and_regen(
            tmp_path, disable_agents=["literature-search"]
        )

        settings = json.loads((project_dir / ".claude" / "settings.json").read_text())
        allow = settings["permissions"]["allow"]
        assert "Task(literature-search)" not in allow

    def test_disable_agent_removes_from_claude_md(self, tmp_path):
        """Disabling literature-search removes its delegation section from CLAUDE.md."""
        project_dir, _ = self._create_and_regen(
            tmp_path, disable_agents=["literature-search"]
        )

        content = (project_dir / "CLAUDE.md").read_text()
        # The delegation section should be gone
        assert "literature-search` sub-agent" not in content
        # The usage pattern section should be gone
        assert "### Literature search" not in content

    def test_extra_server_added_to_mcp_json(self, tmp_path):
        """Extra server appears in .mcp.json."""
        project_dir, _ = self._create_and_regen(
            tmp_path,
            extra_servers={
                "my-server": {"command": "node", "args": ["server.js"]}
            },
        )

        mcp_data = json.loads((project_dir / ".mcp.json").read_text())
        assert "my-server" in mcp_data["mcpServers"]
        assert mcp_data["mcpServers"]["my-server"]["command"] == "node"

    def test_extra_server_ask_permission_in_settings(self, tmp_path):
        """Extra server gets ask permission in settings.json."""
        project_dir, _ = self._create_and_regen(
            tmp_path,
            extra_servers={
                "my-server": {"command": "node", "args": ["server.js"]}
            },
        )

        settings = json.loads((project_dir / ".claude" / "settings.json").read_text())
        ask = settings["permissions"]["ask"]
        assert any("mcp__my-server" in entry for entry in ask)

    def test_disable_does_not_remove_safety_hooks(self, tmp_path):
        """Disabling a server doesn't remove hook script files."""
        project_dir, _ = self._create_and_regen(
            tmp_path, disable_servers=["accelpapers"]
        )

        hooks_dir = project_dir / ".claude" / "hooks"
        assert hooks_dir.exists()
        hook_files = list(hooks_dir.glob("*.py"))
        assert len(hook_files) > 0
        # Safety hooks should still exist
        hook_names = [f.name for f in hook_files]
        assert "osprey_writes_check.py" in hook_names
        assert "osprey_error_guidance.py" in hook_names

    def test_regen_summary_includes_active_lists(self, tmp_path):
        """Result dict contains active/disabled lists."""
        _, result = self._create_and_regen(
            tmp_path, disable_servers=["accelpapers"]
        )

        assert "active_servers" in result
        assert "disabled_servers" in result
        assert "active_agents" in result
        assert "disabled_agents" in result
        assert "accelpapers" not in result["active_servers"]
        assert "accelpapers" in result["disabled_servers"]
        assert "controls" in result["active_servers"]

    def test_disable_core_server_allowed(self, tmp_path):
        """Users can disable any server including core ones."""
        project_dir, result = self._create_and_regen(
            tmp_path, disable_servers=["ariel"]
        )

        mcp_data = json.loads((project_dir / ".mcp.json").read_text())
        assert "ariel" not in mcp_data["mcpServers"]
        assert "ariel" in result["disabled_servers"]

    def test_context_includes_overrides(self, tmp_path):
        """_build_claude_code_context includes disable_servers, disable_agents, extra_servers."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="ctx-overrides",
            output_dir=tmp_path,
            template_name="minimal",
        )

        config = yaml.safe_load((project_dir / "config.yml").read_text())
        config["claude_code"] = {
            "disable_servers": ["accelpapers"],
            "disable_agents": ["literature-search"],
            "extra_servers": {"my-srv": {"command": "echo"}},
        }
        (project_dir / "config.yml").write_text(yaml.dump(config))

        ctx = manager._build_claude_code_context(project_dir, config)
        assert ctx["disable_servers"] == ["accelpapers"]
        assert ctx["disable_agents"] == ["literature-search"]
        assert "my-srv" in ctx["extra_servers"]

    def test_defaults_empty_when_no_claude_code_section(self, tmp_path):
        """Without claude_code section, overrides default to empty."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="ctx-defaults",
            output_dir=tmp_path,
            template_name="minimal",
        )

        config = yaml.safe_load((project_dir / "config.yml").read_text())
        ctx = manager._build_claude_code_context(project_dir, config)
        assert ctx["disable_servers"] == []
        assert ctx["disable_agents"] == []
        assert ctx["extra_servers"] == {}


class TestFacilityMd:
    """Test facility.md creation and preservation."""

    def test_facility_md_created_on_init(self, tmp_path):
        """osprey init creates .claude/rules/facility.md."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="facility-init",
            output_dir=tmp_path,
            template_name="minimal",
        )

        facility_file = project_dir / ".claude" / "rules" / "facility.md"
        assert facility_file.exists()
        content = facility_file.read_text()
        assert "Facility Identity" in content
        assert "Example Research Facility" in content

    def test_facility_md_override_registered_on_init(self, tmp_path):
        """Init auto-registers facility.md as a prompt override in config.yml."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="facility-override",
            output_dir=tmp_path,
            template_name="minimal",
        )

        # Verify override registered in config.yml
        config = yaml.safe_load((project_dir / "config.yml").read_text())
        override_rel = config["prompts"]["overrides"]["rules/facility"]
        assert override_rel == "overrides/.claude/rules/facility.md"

        # Verify override source file was created
        override_file = project_dir / override_rel
        assert override_file.exists()

    def test_facility_md_preserved_via_override(self, tmp_path):
        """Regen preserves customized facility.md via the override mechanism."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="facility-preserve",
            output_dir=tmp_path,
            template_name="minimal",
        )

        # Customize facility.md via the override source (the user-editable copy)
        facility_file = project_dir / ".claude" / "rules" / "facility.md"
        override_file = project_dir / "overrides" / ".claude" / "rules" / "facility.md"
        custom_content = "# My Custom Facility\n\nAdvanced Light Source at LBNL\n"
        override_file.write_text(custom_content)

        # Regen
        manager.regenerate_claude_code(project_dir)

        # Verify override content was applied to the output file
        assert facility_file.read_text() == custom_content

    def test_facility_md_created_on_regen_if_missing(self, tmp_path):
        """If facility.md doesn't exist, regen creates it."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="facility-regen-create",
            output_dir=tmp_path,
            template_name="minimal",
        )

        # Delete facility.md
        facility_file = project_dir / ".claude" / "rules" / "facility.md"
        facility_file.unlink()
        assert not facility_file.exists()

        # Regen should recreate it
        manager.regenerate_claude_code(project_dir)

        assert facility_file.exists()
        content = facility_file.read_text()
        assert "Facility Identity" in content


class TestManagedFiles:
    """Test managed_files configuration."""

    def test_managed_files_default_fallback(self, tmp_path):
        """Without managed_files in config, uses default list."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="managed-defaults",
            output_dir=tmp_path,
            template_name="minimal",
        )

        config = yaml.safe_load((project_dir / "config.yml").read_text())
        ctx = manager._build_claude_code_context(project_dir, config)

        # Should have the default managed_files list
        assert "managed_files" in ctx
        assert "CLAUDE.md" in ctx["managed_files"]
        assert ".mcp.json" in ctx["managed_files"]
        assert ".claude/settings.json" in ctx["managed_files"]

    def test_managed_files_from_config(self, tmp_path):
        """managed_files from config.yml is respected."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="managed-config",
            output_dir=tmp_path,
            template_name="minimal",
        )

        # Set custom managed_files in config
        config = yaml.safe_load((project_dir / "config.yml").read_text())
        config["claude_code"] = {
            "managed_files": ["CLAUDE.md", ".mcp.json"],
        }
        (project_dir / "config.yml").write_text(yaml.dump(config))

        ctx = manager._build_claude_code_context(project_dir, config)
        assert ctx["managed_files"] == ["CLAUDE.md", ".mcp.json"]

    def test_managed_files_controls_regen(self, tmp_path):
        """Removing a file from managed_files prevents overwrite on regen."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="managed-regen",
            output_dir=tmp_path,
            template_name="minimal",
        )

        # Customize safety.md
        safety_file = project_dir / ".claude" / "rules" / "safety.md"
        custom_safety = "# My Custom Safety Rules\n\nCustom content here.\n"
        safety_file.write_text(custom_safety)

        # Remove safety.md from managed_files
        config = yaml.safe_load((project_dir / "config.yml").read_text())
        default_files = manager._get_default_managed_files()
        custom_files = [f for f in default_files if f != ".claude/rules/safety.md"]
        config["claude_code"] = {"managed_files": custom_files}
        (project_dir / "config.yml").write_text(yaml.dump(config))

        # Regen
        manager.regenerate_claude_code(project_dir)

        # safety.md should be preserved
        assert safety_file.read_text() == custom_safety

    def test_agents_always_managed(self, tmp_path):
        """Agent files are always managed regardless of managed_files list."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="managed-agents",
            output_dir=tmp_path,
            template_name="minimal",
        )

        # Set minimal managed_files (no agent paths)
        config = yaml.safe_load((project_dir / "config.yml").read_text())
        config["claude_code"] = {
            "managed_files": ["CLAUDE.md", ".mcp.json", ".claude/settings.json"],
        }
        (project_dir / "config.yml").write_text(yaml.dump(config))

        # Regen
        manager.regenerate_claude_code(project_dir)

        # Agent files should still exist (they're auto-managed)
        agents_dir = project_dir / ".claude" / "agents"
        if agents_dir.exists():
            agent_files = list(agents_dir.glob("*.md"))
            assert len(agent_files) > 0


class TestSettingsJsonValidity:
    """Test that settings.json.j2 renders valid JSON across all configurations.

    Regression test for trailing-comma bug where conditional Jinja2 blocks
    in the hooks section could leave a trailing comma before a closing bracket,
    producing invalid JSON (e.g., `},]`).
    """

    @pytest.mark.parametrize(
        "template_name", ["minimal", "hello_world_weather", "control_assistant"]
    )
    def test_all_templates_produce_valid_settings_json(self, tmp_path, template_name):
        """Every built-in template produces valid settings.json."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name=f"json-valid-{template_name}",
            output_dir=tmp_path,
            template_name=template_name,
        )
        settings_path = project_dir / ".claude" / "settings.json"
        data = json.loads(settings_path.read_text())
        assert "permissions" in data
        assert "hooks" in data

    @pytest.mark.parametrize(
        "disable_servers,label",
        [
            (["controls"], "controls-disabled"),
            (["python"], "python-disabled"),
            (["controls", "python"], "controls-and-python-disabled"),
            (["ariel"], "ariel-disabled"),
            (["workspace"], "workspace-disabled"),
            (["accelpapers"], "accelpapers-disabled"),
            (
                ["controls", "python", "workspace", "ariel", "accelpapers"],
                "all-core-disabled",
            ),
        ],
        ids=lambda x: x if isinstance(x, str) else None,
    )
    def test_disable_servers_produces_valid_json(self, tmp_path, disable_servers, label):
        """Disabling various server combinations still produces valid JSON."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name=f"json-{label}",
            output_dir=tmp_path,
            template_name="minimal",
        )

        config = yaml.safe_load((project_dir / "config.yml").read_text())
        config["claude_code"] = {"disable_servers": disable_servers}
        (project_dir / "config.yml").write_text(yaml.dump(config))

        manager.regenerate_claude_code(project_dir)

        settings_path = project_dir / ".claude" / "settings.json"
        data = json.loads(settings_path.read_text())
        assert "permissions" in data
        assert "hooks" in data

    def test_all_optional_features_enabled(self, tmp_path):
        """Valid JSON when all optional features (confluence, matlab, deplot) are enabled."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="json-all-features",
            output_dir=tmp_path,
            template_name="control_assistant",
            context={
                "confluence": {"url": "https://wiki.example.com"},
                "matlab": {"db_path": "~/.matlab-mml/mml.db"},
                "deplot": {"host": "127.0.0.1", "port": 8095},
            },
        )
        settings_path = project_dir / ".claude" / "settings.json"
        data = json.loads(settings_path.read_text())
        assert "permissions" in data
        assert "hooks" in data

    def test_extra_servers_produce_valid_json(self, tmp_path):
        """Adding extra_servers still produces valid JSON."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="json-extra-servers",
            output_dir=tmp_path,
            template_name="minimal",
        )

        config = yaml.safe_load((project_dir / "config.yml").read_text())
        config["claude_code"] = {
            "extra_servers": {
                "my-server": {"command": "node", "args": ["server.js"]},
            },
        }
        (project_dir / "config.yml").write_text(yaml.dump(config))

        manager.regenerate_claude_code(project_dir)

        settings_path = project_dir / ".claude" / "settings.json"
        data = json.loads(settings_path.read_text())
        assert "permissions" in data
        assert any("mcp__my-server" in entry for entry in data["permissions"]["ask"])

    def test_all_mcp_json_files_are_valid(self, tmp_path):
        """Every template also produces valid .mcp.json."""
        for template_name in ["minimal", "hello_world_weather", "control_assistant"]:
            project_dir = TemplateManager().create_project(
                project_name=f"mcp-valid-{template_name}",
                output_dir=tmp_path,
                template_name=template_name,
            )
            mcp_path = project_dir / ".mcp.json"
            data = json.loads(mcp_path.read_text())
            assert "mcpServers" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
