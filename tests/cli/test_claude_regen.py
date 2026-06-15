"""Tests for Claude Code artifact regeneration.

Tests that `osprey claude regen` correctly rebuilds Claude Code artifacts
from config.yml, preserves user files, creates backups, and maintains
safety hooks across regeneration cycles.
"""

import json
import os

import pytest
import yaml

from osprey.cli.templates import claude_code
from osprey.cli.templates.manager import TemplateManager


class TestBuildClaudeCodeContext:
    """Test build_claude_code_context() reconstructs correct template vars."""

    def test_basic_config(self, tmp_path):
        """Basic config produces correct base context vars."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="ctx-basic",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )

        config = yaml.safe_load((project_dir / "config.yml").read_text())
        ctx = claude_code.build_claude_code_context(
            manager.template_root, manager.jinja_env, project_dir, config
        )

        assert ctx["project_name"] == "ctx-basic"
        assert ctx["package_name"] == "ctx_basic"
        assert ctx["project_root"] == str(project_dir.absolute())
        assert "current_python_env" in ctx
        assert ctx["template_name"] == "control_assistant"

    def test_project_root_override(self, tmp_path):
        """project_root_override replaces project_root in context but keeps file I/O path."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="ctx-override",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )

        config = yaml.safe_load((project_dir / "config.yml").read_text())
        ctx = claude_code.build_claude_code_context(
            manager.template_root,
            manager.jinja_env,
            project_dir,
            config,
            project_root_override="/app/als-assistant",
        )

        # project_root uses the override
        assert ctx["project_root"] == "/app/als-assistant"
        # project_name still derived from actual project (file I/O uses project_dir)
        assert ctx["project_name"] == "ctx-override"

    def test_control_assistant_config(self, tmp_path):
        """Control assistant config produces channel_finder context vars."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="ctx-control",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )
        # Generate manifest so build_claude_code_context can discover template_name
        manager.generate_manifest(project_dir, "ctx-control", "control_assistant", {})

        config = yaml.safe_load((project_dir / "config.yml").read_text())
        ctx = claude_code.build_claude_code_context(
            manager.template_root, manager.jinja_env, project_dir, config
        )

        assert ctx["template_name"] == "control_assistant"
        assert "channel_finder_pipeline" in ctx
        assert "channel_finder_mode" in ctx

    def test_uses_manifest_template(self, tmp_path):
        """When manifest exists, template_name is read from it."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="ctx-manifest",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )
        # Generate manifest
        manager.generate_manifest(project_dir, "ctx-manifest", "control_assistant", {})

        config = yaml.safe_load((project_dir / "config.yml").read_text())
        ctx = claude_code.build_claude_code_context(
            manager.template_root, manager.jinja_env, project_dir, config
        )

        assert ctx["template_name"] == "control_assistant"


class TestRegenerationCorrectness:
    """Test that regeneration produces correct output."""

    def test_regen_produces_same_output_as_init(self, tmp_path):
        """Init → regen with no config changes → identical artifacts."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="regen-idempotent",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
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

    def test_regen_resolves_env_var_in_timezone(self, tmp_path, monkeypatch):
        """${TZ:-UTC} in config.yml is resolved in timezone.md."""
        monkeypatch.delenv("TZ", raising=False)
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="regen-tz",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )

        # Set timezone to an env-var pattern in config.yml
        config = yaml.safe_load((project_dir / "config.yml").read_text())
        config.setdefault("system", {})["timezone"] = "${TZ:-UTC}"
        (project_dir / "config.yml").write_text(yaml.dump(config))

        manager.regenerate_claude_code(project_dir)

        timezone_file = project_dir / ".claude" / "rules" / "timezone.md"
        assert timezone_file.exists(), "timezone.md should be created when timezone is set"
        content = timezone_file.read_text()
        assert "UTC" in content
        assert "${TZ" not in content, "Env var pattern should be resolved, not literal"

    def test_regen_resolves_env_var_in_timezone_with_env_set(self, tmp_path, monkeypatch):
        """${TZ:-...} resolves to actual TZ env value when set."""
        monkeypatch.setenv("TZ", "Europe/Berlin")
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="regen-tz-env",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )

        config = yaml.safe_load((project_dir / "config.yml").read_text())
        config.setdefault("system", {})["timezone"] = "${TZ:-UTC}"
        (project_dir / "config.yml").write_text(yaml.dump(config))

        manager.regenerate_claude_code(project_dir)

        timezone_file = project_dir / ".claude" / "rules" / "timezone.md"
        assert timezone_file.exists()
        content = timezone_file.read_text()
        assert "Europe/Berlin" in content


class TestSafetyPreservation:
    """Test that regeneration always preserves safety layers."""

    @pytest.fixture()
    def regen_project(self, tmp_path):
        """Create and regenerate a project."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="safety-test",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )
        manager.regenerate_claude_code(project_dir)
        return project_dir

    def test_always_includes_safety_hooks(self, regen_project):
        """After regen, settings.json still has PreToolUse/PostToolUse hook chains."""
        settings = json.loads((regen_project / ".claude" / "settings.json").read_text())
        assert "PreToolUse" in settings["hooks"]
        assert "PostToolUse" in settings["hooks"]

    def test_always_denies_dangerous_tools(self, regen_project):
        """After regen, settings.json denies Bash, Edit, WebFetch, WebSearch.

        Write is intentionally NOT denied — Claude Code's native auto memory
        system requires Write access to persist memories across sessions.
        """
        settings = json.loads((regen_project / ".claude" / "settings.json").read_text())
        deny = settings["permissions"]["deny"]
        for tool in ["Bash", "Edit", "WebFetch", "WebSearch"]:
            assert tool in deny, f"{tool} should be in deny list after regen"
        assert "Write" not in deny, "Write must not be denied (auto memory needs it)"

    def test_preserves_writes_check_hook(self, regen_project):
        """osprey_writes_check.py is in PreToolUse for channel_write after regen."""
        settings = json.loads((regen_project / ".claude" / "settings.json").read_text())
        pre_tool_use = settings["hooks"]["PreToolUse"]
        channel_write_hooks = [
            h for h in pre_tool_use if h.get("matcher") == "mcp__controls__channel_write"
        ]
        assert len(channel_write_hooks) > 0
        hook_commands = [
            hook["command"] for entry in channel_write_hooks for hook in entry["hooks"]
        ]
        assert any("osprey_writes_check.py" in cmd for cmd in hook_commands)

    def test_has_memory_guard_hook(self, regen_project):
        """Write PreToolUse hook for memory guard is present."""
        settings_path = regen_project / ".claude" / "settings.json"
        data = json.loads(settings_path.read_text())
        pre_matchers = [r["matcher"] for r in data["hooks"]["PreToolUse"]]
        assert "Write" in pre_matchers

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
        """Regen creates backup directory in _agent_data/backup/."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="backup-test",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
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
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )

        content = (project_dir / "CLAUDE.md").read_text()
        assert "GENERATED BY OSPREY" in content
        assert "osprey scaffold claim" in content


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
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )

        # Record file mtimes
        mcp_mtime = (project_dir / ".mcp.json").stat().st_mtime

        result = manager.regenerate_claude_code(project_dir, dry_run=True)

        assert result["backup_dir"] is None
        # File should not be modified
        assert (project_dir / ".mcp.json").stat().st_mtime == mcp_mtime

    def test_dry_run_clean_after_regen_reports_no_changes(self, tmp_path):
        """Dry run on a freshly regenerated project reports nothing to change.

        facility.md is create-only (auto-registered user-owned) — a real regen
        never rewrites an existing copy, so the dry-run comparison must not
        report it as drift. A phantom diff here makes regen_if_drift perform a
        full regen (and churn a backup dir) on every web launch / config write,
        and makes `osprey claude status` report permanent false drift.
        """
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="dry-clean",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )
        manager.regenerate_claude_code(project_dir)

        result = manager.regenerate_claude_code(project_dir, dry_run=True)
        assert result["changed"] == []

    def test_dry_run_skips_user_owned_artifacts(self, tmp_path):
        """Dry run mirrors real-regen semantics for ejected (user-owned) files.

        A real regen never rewrites a user-owned artifact, so a customized copy
        must not show up as drift in the dry-run report.
        """
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="dry-owned",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )
        safety_file = project_dir / ".claude" / "rules" / "safety.md"
        safety_file.write_text("# My Custom Safety Rules\n")
        config = yaml.safe_load((project_dir / "config.yml").read_text())
        config.setdefault("scaffold", {}).setdefault("user_owned", []).append("rules/safety")
        (project_dir / "config.yml").write_text(yaml.dump(config))
        manager.regenerate_claude_code(project_dir)
        assert safety_file.read_text() == "# My Custom Safety Rules\n"

        result = manager.regenerate_claude_code(project_dir, dry_run=True)
        assert ".claude/rules/safety.md" not in result["changed"]
        assert result["changed"] == []

    def test_dry_run_detects_changes(self, tmp_path):
        """Dry run correctly detects what would change."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="dry-detect",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )

        # Disable a core server in config (will cause .mcp.json to change)
        config = yaml.safe_load((project_dir / "config.yml").read_text())
        config["claude_code"] = {"servers": {"ariel": {"enabled": False}}}
        (project_dir / "config.yml").write_text(yaml.dump(config))

        result = manager.regenerate_claude_code(project_dir, dry_run=True)

        assert ".mcp.json" in result["changed"]


class TestGitignore:
    """Test that gitignore includes local (personal) settings but NOT shared artifacts."""

    def test_gitignore_includes_local_settings(self, tmp_path):
        """Project .gitignore includes local settings but not shared generated files."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="gitignore-gen-test",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )

        gitignore = (project_dir / ".gitignore").read_text()
        # Local/personal settings ARE ignored
        assert "CLAUDE.local.md" in gitignore
        assert ".claude/settings.local.json" in gitignore
        # Shared generated artifacts are NOT ignored (they're committed)
        lines = [line.strip() for line in gitignore.splitlines()]
        assert "CLAUDE.md" not in lines
        assert ".mcp.json" not in lines


class TestDisableServers:
    """Test disable_servers functionality."""

    def _create_and_regen(
        self,
        tmp_path,
        disable_servers=None,
        disable_agents=None,
        custom_servers=None,
        template="control_assistant",
    ):
        """Helper: create project, set claude_code overrides, regen."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="disable-test",
            output_dir=tmp_path,
            data_bundle=template,
            context={"channel_finder_mode": "hierarchical"},
        )

        config = yaml.safe_load((project_dir / "config.yml").read_text())
        cc = {}
        servers = {}
        agents = {}
        if disable_servers:
            for name in disable_servers:
                servers[name] = {"enabled": False}
        if custom_servers:
            servers.update(custom_servers)
        if disable_agents:
            for name in disable_agents:
                agents[name] = {"enabled": False}
        if servers:
            cc["servers"] = servers
        if agents:
            cc["agents"] = agents
        if cc:
            config["claude_code"] = cc
            (project_dir / "config.yml").write_text(yaml.dump(config))

        result = manager.regenerate_claude_code(project_dir)
        return project_dir, result

    def test_disable_server_removes_from_mcp_json(self, tmp_path):
        """Disabling ariel removes it from .mcp.json."""
        project_dir, _ = self._create_and_regen(tmp_path, disable_servers=["ariel"])

        mcp_data = json.loads((project_dir / ".mcp.json").read_text())
        assert "ariel" not in mcp_data["mcpServers"]
        # Core servers should still be present
        assert "controls" in mcp_data["mcpServers"]

    def test_disable_server_removes_from_settings_allow(self, tmp_path):
        """Disabling ariel removes its tools from settings allow list."""
        project_dir, _ = self._create_and_regen(tmp_path, disable_servers=["ariel"])

        settings = json.loads((project_dir / ".claude" / "settings.json").read_text())
        allow = settings["permissions"]["allow"]
        for entry in allow:
            assert "ariel" not in entry, f"ariel found in allow: {entry}"

    def test_disable_server_removes_from_claude_md(self, tmp_path):
        """Disabling ariel removes ariel row from CLAUDE.md tool table."""
        project_dir, _ = self._create_and_regen(tmp_path, disable_servers=["ariel"])

        content = (project_dir / "CLAUDE.md").read_text()
        assert "ariel" not in content.lower() or "ariel" not in content

    def test_disable_agent_removes_agent_file(self, tmp_path):
        """Disabling logbook-search produces an empty agent file."""
        project_dir, _ = self._create_and_regen(tmp_path, disable_agents=["logbook-search"])

        agent_file = project_dir / ".claude" / "agents" / "logbook-search.md"
        if agent_file.exists():
            content = agent_file.read_text().strip()
            # Disabled agents render to empty (or whitespace-only) files
            assert content == "", f"Expected empty file, got: {content[:100]}"

    def test_disable_agent_removes_task_from_settings(self, tmp_path):
        """Disabling logbook-search removes Task(logbook-search) from allow."""
        project_dir, _ = self._create_and_regen(tmp_path, disable_agents=["logbook-search"])

        settings = json.loads((project_dir / ".claude" / "settings.json").read_text())
        allow = settings["permissions"]["allow"]
        assert "Task(logbook-search)" not in allow

    def test_disable_agent_removes_from_claude_md(self, tmp_path):
        """Disabling logbook-search removes its delegation section from CLAUDE.md."""
        project_dir, _ = self._create_and_regen(tmp_path, disable_agents=["logbook-search"])

        content = (project_dir / "CLAUDE.md").read_text()
        # The delegation section should be gone
        assert "logbook-search` sub-agent" not in content

    def test_extra_server_added_to_mcp_json(self, tmp_path):
        """Custom server appears in .mcp.json."""
        project_dir, _ = self._create_and_regen(
            tmp_path,
            custom_servers={"my-server": {"command": "node", "args": ["server.js"]}},
        )

        mcp_data = json.loads((project_dir / ".mcp.json").read_text())
        assert "my-server" in mcp_data["mcpServers"]
        assert mcp_data["mcpServers"]["my-server"]["command"] == "node"

    def test_custom_server_permission_in_settings(self, tmp_path):
        """Custom server with explicit permissions gets them in settings.json."""
        project_dir, _ = self._create_and_regen(
            tmp_path,
            custom_servers={
                "my-server": {
                    "command": "node",
                    "args": ["server.js"],
                    "permissions": {"ask": ["do_stuff"]},
                }
            },
        )

        settings = json.loads((project_dir / ".claude" / "settings.json").read_text())
        ask = settings["permissions"]["ask"]
        assert "mcp__my-server__do_stuff" in ask

    def test_regen_preserves_url_servers(self, tmp_path):
        """URL/SSE servers in config.yml survive regen and appear in .mcp.json."""
        project_dir, _ = self._create_and_regen(
            tmp_path,
            custom_servers={
                "remote-api": {
                    "url": "http://remote:8001/sse",
                    "permissions": {"allow": ["search"]},
                }
            },
        )

        mcp_data = json.loads((project_dir / ".mcp.json").read_text())
        assert "remote-api" in mcp_data["mcpServers"]
        entry = mcp_data["mcpServers"]["remote-api"]
        assert entry == {"type": "http", "url": "http://remote:8001/sse"}

        # Permissions should be in settings.json
        settings = json.loads((project_dir / ".claude" / "settings.json").read_text())
        assert "mcp__remote-api__search" in settings["permissions"]["allow"]

    def test_custom_server_env_preserves_var_placeholders(self, tmp_path, monkeypatch):
        """`${VAR}` placeholders in custom stdio server env survive into .mcp.json.

        Claude Code expands `${VAR}` / `${VAR:-default}` in `.mcp.json` env blocks
        at MCP server launch time. OSPREY must preserve the literal so the
        expansion happens at runtime — never bake an expanded (possibly empty)
        value into the generated artifact at build/regen time.
        """
        # Var is unset at build time — common in CI image builds where
        # secrets aren't passed to image builds for safety.
        monkeypatch.delenv("EXAMPLE_SECRET", raising=False)

        project_dir, _ = self._create_and_regen(
            tmp_path,
            custom_servers={
                "example-stdio": {
                    "command": "node",
                    "args": ["server.js"],
                    "env": {
                        "EXAMPLE_TOKEN": "${EXAMPLE_SECRET:-}",
                        "EXAMPLE_URL": "https://example.invalid",
                    },
                    "permissions": {"allow": ["do_stuff"]},
                }
            },
        )

        mcp_data = json.loads((project_dir / ".mcp.json").read_text())
        env = mcp_data["mcpServers"]["example-stdio"]["env"]
        assert env["EXAMPLE_TOKEN"] == "${EXAMPLE_SECRET:-}", (
            f"Expected ${{EXAMPLE_SECRET:-}} to survive into .mcp.json verbatim, "
            f"got {env['EXAMPLE_TOKEN']!r}. Build-time expansion bakes empty secrets "
            f"into the artifact, blanking the inherited env at MCP launch time."
        )
        # Non-var values should pass through unchanged
        assert env["EXAMPLE_URL"] == "https://example.invalid"

    def test_custom_server_env_preserves_placeholder_when_var_set(self, tmp_path, monkeypatch):
        """`${VAR}` placeholders survive even when VAR is set at build time.

        Build-time expansion is wrong even when the var IS set: secrets get
        baked into a build artifact that may be cached, distributed, or
        committed. Claude Code reads env at MCP launch — that's where
        expansion belongs.
        """
        monkeypatch.setenv("EXAMPLE_SECRET", "shhh-not-supposed-to-be-baked-in")

        project_dir, _ = self._create_and_regen(
            tmp_path,
            custom_servers={
                "example-stdio": {
                    "command": "node",
                    "args": ["server.js"],
                    "env": {"EXAMPLE_TOKEN": "${EXAMPLE_SECRET}"},
                    "permissions": {"allow": ["do_stuff"]},
                }
            },
        )

        mcp_data = json.loads((project_dir / ".mcp.json").read_text())
        env = mcp_data["mcpServers"]["example-stdio"]["env"]
        assert env["EXAMPLE_TOKEN"] == "${EXAMPLE_SECRET}"
        assert "shhh" not in json.dumps(mcp_data), (
            "Secret leaked into build artifact — env-var expansion happened at "
            "build time instead of being deferred to MCP server launch."
        )

    def test_regen_with_project_root_override(self, tmp_path):
        """project_root_override changes paths in rendered output."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="override-test",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )
        manager.generate_manifest(project_dir, "override-test", "control_assistant", {})

        result = manager.regenerate_claude_code(
            project_dir, project_root_override="/app/als-assistant"
        )

        # Verify the override path appears in rendered files
        mcp_data = json.loads((project_dir / ".mcp.json").read_text())
        # Framework servers use project_root for env vars
        controls = mcp_data["mcpServers"].get("controls", {})
        if "env" in controls:
            config_path = controls["env"].get("OSPREY_CONFIG", "")
            assert config_path.startswith("/app/als-assistant")

        assert "changed" in result or "unchanged" in result

    def test_disable_does_not_remove_safety_hooks(self, tmp_path):
        """Disabling a server doesn't remove hook script files."""
        project_dir, _ = self._create_and_regen(tmp_path, disable_servers=["ariel"])

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
        _, result = self._create_and_regen(tmp_path, disable_servers=["ariel"])

        assert "active_servers" in result
        assert "disabled_servers" in result
        assert "active_agents" in result
        assert "disabled_agents" in result
        assert "ariel" not in result["active_servers"]
        assert "ariel" in result["disabled_servers"]
        assert "controls" in result["active_servers"]

    def test_disable_core_server_allowed(self, tmp_path):
        """Users can disable any server including core ones."""
        project_dir, result = self._create_and_regen(tmp_path, disable_servers=["ariel"])

        mcp_data = json.loads((project_dir / ".mcp.json").read_text())
        assert "ariel" not in mcp_data["mcpServers"]
        assert "ariel" in result["disabled_servers"]

    def test_context_includes_overrides(self, tmp_path):
        """build_claude_code_context includes enabled_servers and enabled_agents."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="ctx-overrides",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )

        config = yaml.safe_load((project_dir / "config.yml").read_text())
        config["claude_code"] = {
            "servers": {"ariel": {"enabled": False}, "my-srv": {"command": "echo"}},
            "agents": {"logbook-search": {"enabled": False}},
        }
        (project_dir / "config.yml").write_text(yaml.dump(config))

        ctx = claude_code.build_claude_code_context(
            manager.template_root, manager.jinja_env, project_dir, config
        )
        assert "ariel" not in ctx["enabled_servers"]
        assert "logbook-search" not in ctx["enabled_agents"]
        # Data-driven lists
        assert any(s["name"] == "my-srv" and s["enabled"] for s in ctx["servers"])
        assert any(s["name"] == "ariel" and not s["enabled"] for s in ctx["servers"])

    def test_defaults_empty_when_no_claude_code_section(self, tmp_path):
        """Without claude_code section, core servers are enabled."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="ctx-defaults",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )

        config = yaml.safe_load((project_dir / "config.yml").read_text())
        ctx = claude_code.build_claude_code_context(
            manager.template_root, manager.jinja_env, project_dir, config
        )
        # Core servers should all be enabled
        assert {"controls", "osprey_workspace", "ariel"} <= ctx["enabled_servers"]


class TestRegenRuntimeRoot:
    """Test `osprey claude regen --runtime-root` path relocation.

    The flag rewrites recorded host paths in config.yml (comment-preserving)
    and re-renders artifacts against the new root — the supported way to fix
    up a project copied into a container image.
    """

    def _create(self, tmp_path, name):
        manager = TemplateManager()
        return manager.create_project(
            project_name=name,
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )

    def _invoke(self, args):
        from click.testing import CliRunner

        from osprey.cli.main import cli

        return CliRunner().invoke(cli, args)

    def test_project_root_rewritten_and_comments_preserved(self, tmp_path):
        """config.yml project_root is rewritten; a known comment survives."""
        project_dir = self._create(tmp_path, "rr-rewrite")
        config_file = project_dir / "config.yml"
        config_file.write_text("# KEEP-THIS-COMMENT\n" + config_file.read_text())

        result = self._invoke(
            ["claude", "regen", "--project", str(project_dir), "--runtime-root", "/app/rr-rewrite"]
        )
        assert result.exit_code == 0, result.output

        text = config_file.read_text()
        assert "# KEEP-THIS-COMMENT" in text, "ruamel rewrite must preserve comments"
        config = yaml.safe_load(text)
        assert config["project_root"] == "/app/rr-rewrite"

    def test_stale_python_env_path_replaced(self, tmp_path):
        """A recorded python_env_path that doesn't exist is healed to sys.executable."""
        import sys

        from osprey.utils.config_writer import config_update_fields

        project_dir = self._create(tmp_path, "rr-stale-env")
        config_file = project_dir / "config.yml"
        config_update_fields(
            config_file, {"execution.python_env_path": "/nonexistent/host/.venv/bin/python"}
        )

        result = self._invoke(
            ["claude", "regen", "--project", str(project_dir), "--runtime-root", str(project_dir)]
        )
        assert result.exit_code == 0, result.output

        config = yaml.safe_load(config_file.read_text())
        assert config["execution"]["python_env_path"] == sys.executable

    def test_valid_python_env_path_untouched(self, tmp_path):
        """An existing python_env_path is left alone."""
        from osprey.utils.config_writer import config_update_fields

        project_dir = self._create(tmp_path, "rr-valid-env")
        config_file = project_dir / "config.yml"
        fake_python = tmp_path / "some-other-venv-python"
        fake_python.touch()
        config_update_fields(config_file, {"execution.python_env_path": str(fake_python)})

        result = self._invoke(
            ["claude", "regen", "--project", str(project_dir), "--runtime-root", str(project_dir)]
        )
        assert result.exit_code == 0, result.output

        config = yaml.safe_load(config_file.read_text())
        assert config["execution"]["python_env_path"] == str(fake_python)

    def test_rendered_artifacts_reference_runtime_root(self, tmp_path):
        """.mcp.json paths point at the runtime root after relocation."""
        project_dir = self._create(tmp_path, "rr-artifacts")

        result = self._invoke(
            [
                "claude",
                "regen",
                "--project",
                str(project_dir),
                "--runtime-root",
                "/app/rr-artifacts",
            ]
        )
        assert result.exit_code == 0, result.output

        mcp_data = json.loads((project_dir / ".mcp.json").read_text())
        controls = mcp_data["mcpServers"].get("controls", {})
        if "env" in controls:
            config_path = controls["env"].get("OSPREY_CONFIG", "")
            assert config_path.startswith("/app/rr-artifacts")

    def test_dry_run_leaves_config_unchanged(self, tmp_path):
        """--dry-run with --runtime-root must not rewrite config.yml."""
        project_dir = self._create(tmp_path, "rr-dry-run")
        config_file = project_dir / "config.yml"
        original = config_file.read_text()

        result = self._invoke(
            [
                "claude",
                "regen",
                "--project",
                str(project_dir),
                "--runtime-root",
                "/app/rr-dry-run",
                "--dry-run",
            ]
        )
        assert result.exit_code == 0, result.output
        assert config_file.read_text() == original

    def test_regen_without_flag_is_unchanged_behavior(self, tmp_path):
        """Plain regen (no --runtime-root) never rewrites config.yml."""
        project_dir = self._create(tmp_path, "rr-no-flag")
        config_file = project_dir / "config.yml"
        original = config_file.read_text()

        result = self._invoke(["claude", "regen", "--project", str(project_dir)])
        assert result.exit_code == 0, result.output
        assert config_file.read_text() == original


class TestFacilityMd:
    """Test facility.md creation and preservation."""

    def test_facility_md_created_on_init(self, tmp_path):
        """create_project() (osprey build path) creates .claude/rules/facility.md."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="facility-init",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )

        facility_file = project_dir / ".claude" / "rules" / "facility.md"
        assert facility_file.exists()
        content = facility_file.read_text()
        assert "Facility Identity" in content
        assert "Example Research Facility" in content

    def test_facility_md_user_owned_on_init(self, tmp_path):
        """Init auto-registers facility.md as user-owned in config.yml."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="facility-owned",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )

        config = yaml.safe_load((project_dir / "config.yml").read_text())
        user_owned = config.get("scaffold", {}).get("user_owned", [])
        assert "rules/facility" in user_owned

        # No overrides/ directory should exist
        assert not (project_dir / "overrides").exists()

    def test_facility_md_preserved_via_user_owned(self, tmp_path):
        """Regen preserves customized facility.md via user_owned mechanism."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="facility-preserve",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )

        # Customize facility.md in-place
        facility_file = project_dir / ".claude" / "rules" / "facility.md"
        custom_content = "# My Custom Facility\n\nAdvanced Light Source at LBNL\n"
        facility_file.write_text(custom_content)

        # Regen
        manager.regenerate_claude_code(project_dir)

        # Verify custom content preserved
        assert facility_file.read_text() == custom_content

    def test_facility_md_created_on_regen_if_missing(self, tmp_path):
        """If facility.md doesn't exist and not user-owned, regen creates it."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="facility-regen-create",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )

        # Delete facility.md and remove from user_owned
        facility_file = project_dir / ".claude" / "rules" / "facility.md"
        facility_file.unlink()
        assert not facility_file.exists()

        # Remove from user_owned so regen will recreate it
        config = yaml.safe_load((project_dir / "config.yml").read_text())
        if "scaffold" in config and "user_owned" in config["scaffold"]:
            config["scaffold"]["user_owned"] = [
                n for n in config["scaffold"]["user_owned"] if n != "rules/facility"
            ]
        (project_dir / "config.yml").write_text(yaml.dump(config))

        # Regen should recreate it
        manager.regenerate_claude_code(project_dir)

        assert facility_file.exists()
        content = facility_file.read_text()
        assert "Facility Identity" in content


class TestUserOwned:
    """Test user_owned configuration for skipping files during regen."""

    def test_user_owned_default_empty(self, tmp_path):
        """Without user_owned in config, list is empty."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="owned-defaults",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )

        config = yaml.safe_load((project_dir / "config.yml").read_text())
        # Remove auto-registered facility
        if "scaffold" in config and "user_owned" in config["scaffold"]:
            config["scaffold"]["user_owned"] = []
        (project_dir / "config.yml").write_text(yaml.dump(config))

        ctx = claude_code.build_claude_code_context(
            manager.template_root, manager.jinja_env, project_dir, config
        )
        assert ctx["user_owned"] == []

    def test_user_owned_controls_regen(self, tmp_path):
        """Adding a file to user_owned prevents overwrite on regen."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="owned-regen",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )

        # Customize safety.md
        safety_file = project_dir / ".claude" / "rules" / "safety.md"
        custom_safety = "# My Custom Safety Rules\n\nCustom content here.\n"
        safety_file.write_text(custom_safety)

        # Add safety to user_owned
        config = yaml.safe_load((project_dir / "config.yml").read_text())
        if "scaffold" not in config:
            config["scaffold"] = {}
        user_owned = config["scaffold"].get("user_owned", [])
        user_owned.append("rules/safety")
        config["scaffold"]["user_owned"] = user_owned
        (project_dir / "config.yml").write_text(yaml.dump(config))

        # Regen
        manager.regenerate_claude_code(project_dir)

        # safety.md should be preserved
        assert safety_file.read_text() == custom_safety

    def test_agents_never_user_owned(self, tmp_path):
        """Agent files are always regenerated even if user_owned list is long."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="owned-agents",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )

        # Regen with some user_owned entries
        config = yaml.safe_load((project_dir / "config.yml").read_text())
        if "scaffold" not in config:
            config["scaffold"] = {}
        config["scaffold"]["user_owned"] = ["rules/safety", "rules/error-handling"]
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

    @pytest.mark.parametrize("data_bundle", ["control_assistant"])
    def test_all_templates_produce_valid_settings_json(self, tmp_path, data_bundle):
        """Every built-in template produces valid settings.json."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name=f"json-valid-{data_bundle}",
            output_dir=tmp_path,
            data_bundle=data_bundle,
            context={"channel_finder_mode": "hierarchical"},
        )
        settings_path = project_dir / ".claude" / "settings.json"
        data = json.loads(settings_path.read_text())
        assert "permissions" in data
        assert "hooks" in data

    @pytest.mark.parametrize(
        "disable_servers,label",
        [
            (["controls"], "controls-disabled"),
            (["ariel"], "ariel-disabled"),
            (["osprey_workspace"], "workspace-disabled"),
            (["controls", "osprey_workspace"], "controls-and-workspace-disabled"),
            (
                ["controls", "osprey_workspace", "ariel"],
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
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )

        config = yaml.safe_load((project_dir / "config.yml").read_text())
        servers_override = {name: {"enabled": False} for name in disable_servers}
        config["claude_code"] = {"servers": servers_override}
        (project_dir / "config.yml").write_text(yaml.dump(config))

        manager.regenerate_claude_code(project_dir)

        settings_path = project_dir / ".claude" / "settings.json"
        data = json.loads(settings_path.read_text())
        assert "permissions" in data
        assert "hooks" in data

    def test_all_optional_features_enabled(self, tmp_path):
        """Valid JSON when all optional features (channel-finder) are on."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="json-all-features",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={
                "channel_finder_pipeline": "hierarchical",
                "channel_finder_mode": "hierarchical",
            },
        )
        settings_path = project_dir / ".claude" / "settings.json"
        data = json.loads(settings_path.read_text())
        assert "permissions" in data
        assert "hooks" in data

    def test_custom_servers_produce_valid_json(self, tmp_path):
        """Adding custom servers still produces valid JSON."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="json-custom-servers",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )

        config = yaml.safe_load((project_dir / "config.yml").read_text())
        config["claude_code"] = {
            "servers": {
                "my-server": {"command": "node", "args": ["server.js"]},
            },
        }
        (project_dir / "config.yml").write_text(yaml.dump(config))

        manager.regenerate_claude_code(project_dir)

        settings_path = project_dir / ".claude" / "settings.json"
        data = json.loads(settings_path.read_text())
        assert "permissions" in data

    def test_all_mcp_json_files_are_valid(self, tmp_path):
        """Every template also produces valid .mcp.json."""
        for data_bundle in ["control_assistant"]:
            project_dir = TemplateManager().create_project(
                project_name=f"mcp-valid-{data_bundle}",
                output_dir=tmp_path,
                data_bundle=data_bundle,
                context={"channel_finder_mode": "hierarchical"},
            )
            mcp_path = project_dir / ".mcp.json"
            data = json.loads(mcp_path.read_text())
            assert "mcpServers" in data


class TestWritesToggleRegen:
    """Flipping control_system.writes_enabled re-renders the kill-switch deny entry.

    This is the GitHub #244 regression: editing config.yml alone leaves
    settings.json's permissions.deny stale, so a regen must add/remove
    mcp__controls__channel_write to match the config.
    """

    @staticmethod
    def _deny(project_dir):
        settings = json.loads((project_dir / ".claude" / "settings.json").read_text())
        return settings["permissions"]["deny"]

    def test_toggle_writes_updates_deny(self, tmp_path):
        from osprey.utils.config_writer import config_update_fields

        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="writes-toggle",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )
        config_path = project_dir / "config.yml"

        # Writes disabled → kill-switch baked into deny.
        config_update_fields(config_path, {"control_system.writes_enabled": False})
        manager.regenerate_claude_code(project_dir)
        assert "mcp__controls__channel_write" in self._deny(project_dir)

        # Enable writes → regen drops the deny entry and reports settings.json changed.
        config_update_fields(config_path, {"control_system.writes_enabled": True})
        result = manager.regenerate_claude_code(project_dir)
        assert "mcp__controls__channel_write" not in self._deny(project_dir)
        assert any("settings.json" in f for f in result["changed"]), (
            f"settings.json should be in changed list, got {result['changed']}"
        )

    def test_regen_if_drift_only_regens_when_changed(self, tmp_path):
        """regen_if_drift returns [] when in sync and the changed list when drifted."""
        from osprey.utils.config_writer import config_update_fields

        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="drift-gate",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )
        manager.regenerate_claude_code(project_dir)

        # Already in sync → no-op, and crucially NO real regen (no new backup dir).
        backup_dir = project_dir / "_agent_data" / "backup"

        def _backup_count():
            return len(list(backup_dir.glob("claude-code-*"))) if backup_dir.exists() else 0

        before = _backup_count()
        assert manager.regen_if_drift(project_dir) == []
        assert _backup_count() == before, "in-sync regen_if_drift must not create a backup"

        # Drift the config by flipping writes_enabled → regen_if_drift reports the change.
        config_path = project_dir / "config.yml"
        cfg = yaml.safe_load(config_path.read_text())
        current = bool(cfg.get("control_system", {}).get("writes_enabled", False))
        config_update_fields(config_path, {"control_system.writes_enabled": not current})
        changed = manager.regen_if_drift(project_dir)
        assert any("settings.json" in f for f in changed)

    def test_regen_if_drift_stamps_settings_mtime_on_cosmetic_edit(self, tmp_path):
        """An edit that changes no artifact must still clear the mtime drift signal.

        The SessionStart drift hook compares config.yml's mtime against
        settings.json's. A cosmetic edit (e.g. a YAML comment, or a runtime-read
        field) makes config.yml newer while regen_if_drift correctly finds nothing
        to regenerate — without an mtime stamp the hook would warn at every
        session start until a full `osprey claude regen`, training operators to
        ignore the warning.
        """
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="mtime-stamp",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )
        manager.regenerate_claude_code(project_dir)

        config_path = project_dir / "config.yml"
        settings_path = project_dir / ".claude" / "settings.json"
        # Cosmetic edit: a YAML comment changes no rendered artifact.
        config_path.write_text(config_path.read_text() + "\n# operator note\n")
        base = 1_700_000_000
        os.utime(settings_path, (base, base))
        os.utime(config_path, (base + 100, base + 100))

        assert manager.regen_if_drift(project_dir) == []
        assert settings_path.stat().st_mtime >= config_path.stat().st_mtime, (
            "in-sync regen_if_drift must stamp settings.json so the drift hook "
            "does not warn on an already-verified config"
        )

    def test_regen_if_drift_noops_without_rendered_claude(self, tmp_path):
        """regen_if_drift is re-sync only — it never bootstraps a .claude/ from scratch."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="no-claude",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )
        # A project with config.yml but no rendered settings.json (never built/regenerated).
        settings = project_dir / ".claude" / "settings.json"
        if settings.exists():
            settings.unlink()
        assert manager.regen_if_drift(project_dir) == []
        # The guard must not create the artifact it was guarding against.
        assert not settings.exists()

    def test_session_start_drift_hook_wired_in_settings(self, tmp_path):
        """The rendered settings.json wires the config-drift hook on SessionStart."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="drift-wiring",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )
        manager.regenerate_claude_code(project_dir)
        settings = json.loads((project_dir / ".claude" / "settings.json").read_text())
        session_start = settings["hooks"]["SessionStart"]
        cmds = [h["command"] for entry in session_start for h in entry["hooks"]]
        assert any("osprey_config_drift.py" in c for c in cmds), cmds
        # And the hook file is actually deployed alongside the wiring.
        assert (project_dir / ".claude" / "hooks" / "osprey_config_drift.py").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
