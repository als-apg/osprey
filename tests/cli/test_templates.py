"""Tests for template generation system.

Tests the TemplateManager class and template rendering,
including validation that generated projects use the new
registry helper pattern correctly.
"""

import pytest
from click.testing import CliRunner

from osprey.cli.build_cmd import build
from osprey.cli.templates import claude_code, manifest
from osprey.cli.templates.manager import TemplateManager

# All CLI tests below build the hello-world preset with deps + lifecycle skipped.
# Centralised so future flag changes are one-line edits.
_BUILD_FLAGS = ["--preset", "hello-world", "--skip-deps", "--skip-lifecycle"]


def _build_args(name: str, output_dir: str, *extra: str) -> list[str]:
    return [name, *_BUILD_FLAGS, "--output-dir", output_dir, *extra]


class TestTemplateManager:
    """Test TemplateManager class."""

    def test_template_manager_initialization(self):
        """Test that TemplateManager initializes correctly."""
        manager = TemplateManager()

        assert manager.template_root is not None
        assert manager.template_root.exists()
        assert manager.jinja_env is not None

    def test_list_app_templates(self):
        """Test listing available application templates."""
        manager = TemplateManager()
        templates = manager.list_app_templates()

        assert "control_assistant" in templates
        assert len(templates) >= 1

    def test_create_project_control_assistant(self, tmp_path):
        """Test creating project with control_assistant template."""
        manager = TemplateManager()

        project_dir = manager.create_project(
            project_name="test-project",
            output_dir=tmp_path,
            data_bundle="control_assistant",
        )

        # Verify structure (Claude Code mode — no src/ or pyproject.toml)
        assert project_dir.exists()
        assert (project_dir / "config.yml").exists()
        assert (project_dir / ".env.example").exists()
        assert (project_dir / "README.md").exists()
        assert (project_dir / "_agent_data").exists()

        # Claude Code integration
        assert (project_dir / "CLAUDE.md").exists()
        assert (project_dir / ".mcp.json").exists()

    def test_duplicate_project_raises_error(self, tmp_path):
        """Test that creating duplicate project raises error."""
        manager = TemplateManager()

        # Create first project
        manager.create_project("test-project", tmp_path, "control_assistant")

        # Try to create again
        with pytest.raises(ValueError, match="already exists"):
            manager.create_project("test-project", tmp_path, "control_assistant")

    def test_invalid_template_raises_error(self, tmp_path):
        """Test that invalid template name raises error."""
        manager = TemplateManager()

        with pytest.raises(ValueError, match="not found"):
            manager.create_project("test-project", tmp_path, "nonexistent_template")


class TestGitIsolation:
    """Test that osprey build creates a self-contained git repo."""

    def test_build_creates_git_repo(self, tmp_path):
        """osprey build creates a .git directory."""
        runner = CliRunner()
        result = runner.invoke(build, _build_args("git-test", str(tmp_path)))

        assert result.exit_code == 0, result.output
        project_dir = tmp_path / "git-test"
        assert (project_dir / ".git").exists(), ".git directory should be created"

    def test_build_initial_commit(self, tmp_path):
        """osprey build creates exactly one initial commit with expected files tracked."""
        import subprocess

        runner = CliRunner()
        result = runner.invoke(build, _build_args("commit-test", str(tmp_path)))

        assert result.exit_code == 0, result.output
        project_dir = tmp_path / "commit-test"

        log = subprocess.run(
            ["git", "log", "--oneline"],
            cwd=project_dir,
            capture_output=True,
            text=True,
        )
        assert log.returncode == 0
        commits = log.stdout.strip().splitlines()
        assert len(commits) == 1
        assert "Initial project from osprey build" in commits[0]

        tracked = subprocess.run(
            ["git", "ls-files"],
            cwd=project_dir,
            capture_output=True,
            text=True,
        )
        tracked_files = tracked.stdout.strip().splitlines()
        assert "config.yml" in tracked_files
        assert "CLAUDE.md" in tracked_files
        assert ".mcp.json" in tracked_files

    def test_gitignore_tracks_claude_dir(self, tmp_path):
        """.claude/ must NOT be gitignored (it's part of the project)."""
        runner = CliRunner()
        result = runner.invoke(build, _build_args("track-test", str(tmp_path)))

        assert result.exit_code == 0, result.output
        project_dir = tmp_path / "track-test"
        gitignore_content = (project_dir / ".gitignore").read_text()
        lines = [line.strip() for line in gitignore_content.splitlines()]
        assert ".claude/" not in lines, ".claude/ should not be ignored"
        assert "CLAUDE.md" not in lines, "CLAUDE.md should not be ignored"
        assert ".mcp.json" not in lines, ".mcp.json should not be ignored"

    def test_gitignore_ignores_local_settings(self, tmp_path):
        """Personal local settings (CLAUDE.local.md, .claude/settings.local.json) are ignored."""
        runner = CliRunner()
        result = runner.invoke(build, _build_args("local-test", str(tmp_path)))

        assert result.exit_code == 0, result.output
        project_dir = tmp_path / "local-test"
        gitignore_content = (project_dir / ".gitignore").read_text()
        assert ".claude/settings.local.json" in gitignore_content
        assert "CLAUDE.local.md" in gitignore_content

    def test_claude_dir_in_initial_commit(self, tmp_path):
        """.claude/ artifacts are included in the initial commit."""
        import subprocess

        runner = CliRunner()
        result = runner.invoke(build, _build_args("claude-track-test", str(tmp_path)))

        assert result.exit_code == 0, result.output
        project_dir = tmp_path / "claude-track-test"

        tracked = subprocess.run(
            ["git", "ls-files"],
            cwd=project_dir,
            capture_output=True,
            text=True,
        )
        tracked_files = tracked.stdout.strip().splitlines()
        # At least some .claude/ files should be tracked
        claude_files = [f for f in tracked_files if f.startswith(".claude/")]
        assert len(claude_files) > 0, ".claude/ files should be tracked in git"

    def test_build_inside_existing_repo_warns(self, tmp_path, caplog):
        """Build inside an existing git repo still creates an isolated repo + warns."""
        import logging
        import subprocess

        # Create a parent git repo
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "--allow-empty", "-m", "parent init"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            env={
                **__import__("os").environ,
                "GIT_AUTHOR_NAME": "test",
                "GIT_AUTHOR_EMAIL": "test@test",
                "GIT_COMMITTER_NAME": "test",
                "GIT_COMMITTER_EMAIL": "test@test",
            },
        )

        runner = CliRunner()
        with caplog.at_level(logging.WARNING, logger="build"):
            result = runner.invoke(build, _build_args("nested-test", str(tmp_path)))

        assert result.exit_code == 0, result.output
        # Should still create the git repo (for isolation)
        project_dir = tmp_path / "nested-test"
        assert (project_dir / ".git").exists()
        # The build logger emits a WARNING about the nested repo. Check via
        # caplog (not result.output) — RichHandler routing differs depending on
        # whether pytest's log-capture has hooked the root logger first.
        assert any("nested git repo" in rec.getMessage() for rec in caplog.records), (
            f"expected 'nested git repo' warning; got records: "
            f"{[r.getMessage()[:80] for r in caplog.records]}"
        )

    def test_build_inside_existing_repo_still_isolates(self, tmp_path):
        """Test that nested repo has its own independent git root."""
        import subprocess
        from pathlib import Path

        # Create a parent git repo
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "--allow-empty", "-m", "parent init"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            env={
                **__import__("os").environ,
                "GIT_AUTHOR_NAME": "test",
                "GIT_AUTHOR_EMAIL": "test@test",
                "GIT_COMMITTER_NAME": "test",
                "GIT_COMMITTER_EMAIL": "test@test",
            },
        )

        runner = CliRunner()
        result = runner.invoke(build, _build_args("isolated-test", str(tmp_path)))

        assert result.exit_code == 0
        project_dir = tmp_path / "isolated-test"

        # The nested project's git root should be itself, not the parent
        git_root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=project_dir,
            capture_output=True,
            text=True,
        )
        assert git_root.returncode == 0
        assert Path(git_root.stdout.strip()).resolve() == project_dir.resolve()

    def test_build_clears_stale_trust_state(self, tmp_path):
        """Test that build clears trust state even when directory was already deleted."""
        import json
        from pathlib import Path

        project_path = (tmp_path / "stale-test").resolve()

        # Simulate stale Claude Code trust state (project was deleted but
        # ~/.claude.json still has the entry)
        claude_json = Path.home() / ".claude.json"
        backup_data = None
        if claude_json.exists():
            backup_data = claude_json.read_text()

        try:
            data = json.loads(claude_json.read_text()) if claude_json.exists() else {}
            data.setdefault("projects", {})[str(project_path)] = {
                "hasTrustDialogAccepted": True,
            }
            claude_json.write_text(json.dumps(data, indent=2) + "\n")

            # Directory does NOT exist — simulates rm -rf before osprey build
            assert not project_path.exists()

            # Create project (no --force needed, directory is gone)
            runner = CliRunner()
            result = runner.invoke(build, _build_args("stale-test", str(tmp_path)))
            assert result.exit_code == 0, result.output

            # Trust entry should be gone
            after = json.loads(claude_json.read_text())
            assert str(project_path) not in after.get("projects", {}), (
                "Stale trust entry should be cleared on build"
            )
        finally:
            if backup_data is not None:
                claude_json.write_text(backup_data)

    def test_force_clears_claude_code_session_state(self, tmp_path):
        """Test that --force removes Claude Code's session directory."""
        from pathlib import Path

        from osprey.cli.project_utils import encode_claude_project_path

        runner = CliRunner()

        # Create project first time
        result = runner.invoke(build, _build_args("session-test", str(tmp_path)))
        assert result.exit_code == 0, result.output

        project_path = (tmp_path / "session-test").resolve()

        # Simulate Claude Code's session directory
        encoded_key = encode_claude_project_path(project_path)
        claude_project_dir = Path.home() / ".claude" / "projects" / encoded_key
        claude_project_dir.mkdir(parents=True, exist_ok=True)
        (claude_project_dir / "sessions-index.json").write_text("{}")

        assert claude_project_dir.exists()

        # Re-create with --force
        result = runner.invoke(build, _build_args("session-test", str(tmp_path), "--force"))
        assert result.exit_code == 0, result.output

        # Session directory should be gone
        assert not claude_project_dir.exists(), "Session directory should be removed on --force"


class TestBuildClaudeCodeContextHierarchy:
    """Tests for hierarchy embedding in build_claude_code_context."""

    def _make_manager_and_config(self, tmp_path, db_data):
        """Create a TemplateManager and config pointing at a hierarchy database."""
        import json as _json

        db_file = tmp_path / "channels.json"
        db_file.write_text(_json.dumps(db_data))

        # Manifest must declare control_assistant template for the
        # channel_finder block to activate.
        manifest = tmp_path / ".osprey-manifest.json"
        manifest.write_text(_json.dumps({"creation": {"template": "control_assistant"}}))

        config = {
            "facility_name": "TestFacility",
            "channel_finder": {
                "pipeline_mode": "hierarchical",
                "pipelines": {
                    "hierarchical": {
                        "database": {"path": "channels.json"},
                    },
                },
            },
        }
        return TemplateManager(), config

    @pytest.mark.unit
    def test_build_claude_code_context_embeds_hierarchy_info(self, tmp_path):
        """Hierarchy levels, config, and naming pattern are embedded in context."""
        manager, config = self._make_manager_and_config(
            tmp_path,
            {
                "hierarchy": {
                    "levels": [
                        {"name": "system", "type": "tree"},
                        {"name": "device", "type": "instances"},
                    ],
                    "naming_pattern": "{system}:{device}",
                },
                "tree": {
                    "SR": {
                        "DEVICE": {
                            "_expansion": {
                                "_type": "range",
                                "_pattern": "D{:02d}",
                                "_range": [1, 3],
                            }
                        }
                    }
                },
            },
        )
        ctx = claude_code.build_claude_code_context(
            manager.template_root, manager.jinja_env, tmp_path, config
        )
        hier = ctx["channel_finder_hierarchy"]
        assert hier is not None
        assert hier["hierarchy_levels"] == ["system", "device"]
        assert hier["naming_pattern"] == "{system}:{device}"
        assert "system" in hier["hierarchy_config"]["levels"]

    @pytest.mark.unit
    def test_build_claude_code_context_hierarchy_missing_path(self, tmp_path):
        """Graceful fallback to None when database path is missing."""
        config = {
            "facility_name": "TestFacility",
            "channel_finder": {
                "pipeline_mode": "hierarchical",
                "pipelines": {
                    "hierarchical": {
                        "database": {},
                    },
                },
            },
        }
        manager = TemplateManager()
        ctx = claude_code.build_claude_code_context(
            manager.template_root, manager.jinja_env, tmp_path, config
        )
        assert ctx["channel_finder_hierarchy"] is None

    @pytest.mark.unit
    def test_build_claude_code_context_hierarchy_non_hierarchical(self, tmp_path):
        """Non-hierarchical pipeline mode: channel_finder_hierarchy is None."""
        config = {
            "facility_name": "TestFacility",
            "channel_finder": {
                "pipeline_mode": "in_context",
            },
        }
        manager = TemplateManager()
        ctx = claude_code.build_claude_code_context(
            manager.template_root, manager.jinja_env, tmp_path, config
        )
        assert ctx["channel_finder_hierarchy"] is None

    @pytest.mark.unit
    def test_create_project_embeds_hierarchy_info(self, tmp_path, monkeypatch):
        """create_project renders hierarchy info into the agent prompt."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        manager = TemplateManager()

        project_dir = manager.create_project(
            project_name="test-hier-embed",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )

        agent_prompt = (project_dir / ".claude" / "agents" / "channel-finder.md").read_text()
        # Must contain embedded hierarchy info, NOT the fallback text
        assert "hierarchy_levels" in agent_prompt
        assert "Call `get_options()` at the first level to discover" not in agent_prompt


class TestTemplateManifest:
    """Test template manifest loading, resolution, and filtering."""

    def test_control_assistant_example_profile_has_expected_artifacts(self):
        """Control assistant example profile declares all expected artifact categories."""
        import importlib.resources

        import yaml

        profile_text = (
            importlib.resources.files("osprey.profiles.presets")
            .joinpath("control-assistant.yml")
            .read_text(encoding="utf-8")
        )
        profile = yaml.safe_load(profile_text)

        assert "hooks" in profile
        assert "rules" in profile
        assert "skills" in profile
        assert "agents" in profile
        assert "output_styles" in profile
        assert "approval" in profile["hooks"]
        assert "channel-finder" in profile["agents"]

    def test_load_manifest_nonexistent_template(self):
        """Returns None for unknown template."""
        manager = TemplateManager()
        mf = manifest.load_template_manifest(manager.template_root, "nonexistent_template")
        assert mf is None

    def test_load_manifest_preset_profile_fallback_includes_web_panels(self):
        """Preset-profile fallback must surface web_panels in the artifacts dict.

        Direct ``TemplateManager.create_project()`` callers rely on this fallback
        (control_assistant has no manifest.yml). If web_panels is dropped,
        config.yml renders ``panels: {}`` and the web terminal shows only the
        universal panels — the exact bug reported when ARIEL / channel-finder /
        tuning panels went missing.
        """
        manager = TemplateManager()
        mf = manifest.load_template_manifest(manager.template_root, "control_assistant")
        assert mf is not None
        artifacts = mf.get("artifacts", {})
        # The preset profile declares these panels; they must round-trip through
        # the fallback so create_project() → Jinja → config.yml wires them up.
        assert "web_panels" in artifacts, (
            "web_panels stripped by preset-profile fallback — TemplateManager will "
            "render `panels: {}` and no built-in panels will appear."
        )
        assert set(artifacts["web_panels"]) >= {"ariel", "channel-finder", "tuning"}

    def test_create_project_without_artifacts_enables_builtin_panels(self, tmp_path):
        """create_project() without explicit artifacts (legacy direct-call path)
        must render the builtin panels block from the preset profile."""
        import yaml as _yaml

        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="init-panels-test",
            output_dir=tmp_path,
            data_bundle="control_assistant",
            context={},
        )
        config = _yaml.safe_load((project_dir / "config.yml").read_text())
        panels = config["web"]["panels"]
        assert panels, "web.panels block is empty — no builtin panels were enabled"
        assert panels.get("ariel", {}).get("enabled") is True
        assert panels.get("channel-finder", {}).get("enabled") is True

    def test_resolve_manifest_outputs_includes_config_artifacts(self):
        """Resolved outputs always contain config artifacts."""
        mf = {"artifacts": {"hooks": ["approval"], "rules": ["safety"], "skills": [], "agents": []}}
        outputs = manifest.resolve_manifest_outputs(mf)

        assert "CLAUDE.md" in outputs
        assert ".mcp.json" in outputs
        assert ".claude/settings.json" in outputs

    def test_resolve_manifest_outputs_maps_hooks(self):
        """hooks: [approval] resolves to .claude/hooks/osprey_approval.py."""
        mf = {"artifacts": {"hooks": ["approval"]}}
        outputs = manifest.resolve_manifest_outputs(mf)

        assert ".claude/hooks/osprey_approval.py" in outputs

    def test_resolve_manifest_outputs_session_report_includes_reference(self):
        """skills: [session-report] resolves to both SKILL.md and reference.md."""
        mf = {"artifacts": {"skills": ["session-report"]}}
        outputs = manifest.resolve_manifest_outputs(mf)

        assert ".claude/skills/session-report/SKILL.md" in outputs
        assert ".claude/skills/session-report/reference.md" in outputs

    def test_control_assistant_has_all_hooks(self, tmp_path):
        """Control assistant project must have all 8 hook files."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="ctrl-hooks-test",
            output_dir=tmp_path,
            data_bundle="control_assistant",
        )

        expected_hooks = [
            "osprey_hook_log.py",
            "osprey_approval.py",
            "osprey_writes_check.py",
            "osprey_limits.py",
            "osprey_error_guidance.py",
            "osprey_memory_guard.py",
            "osprey_notebook_update.py",
            "osprey_cf_feedback_capture.py",
        ]
        hooks_dir = project_dir / ".claude" / "hooks"
        for hook_name in expected_hooks:
            assert (hooks_dir / hook_name).exists(), f"Missing hook: {hook_name}"

    def test_backward_compat_no_manifest(self, tmp_path):
        """If manifest doesn't exist, all files are generated (backward compat)."""
        manager = TemplateManager()

        # get_tracked_files falls back to REGEN_TRACKED_FILES when no manifest
        tracked = manifest.get_tracked_files(manager.template_root, "nonexistent_template")
        assert tracked == list(manifest.REGEN_TRACKED_FILES)

        # resolve_manifest_outputs with allowed_outputs=None means no filtering
        # Verify by checking that load_template_manifest returns None
        mf = manifest.load_template_manifest(manager.template_root, "nonexistent_template")
        assert mf is None


def test_get_framework_version_unknown_on_import_failure(monkeypatch):
    """C8: fallback returns 'unknown', not a stale hard-coded version."""
    import builtins

    import osprey.cli.templates.manifest as manifest_mod

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "osprey":
            raise ImportError("simulated failure")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert manifest_mod.get_framework_version() == "unknown"


def test_registry_style_parameter_is_gone():
    """C1 regression guard: removing the 'standalone' branch must remove the
    parameter from every entry point that no longer needs it. Catches anyone
    re-adding it without a real CLI flag."""
    import inspect

    from osprey.cli.templates.manager import TemplateManager

    sig = inspect.signature(TemplateManager.create_project)
    assert "registry_style" not in sig.parameters, (
        "registry_style should be removed from create_project; resurrect only "
        "with a real CLI flag, not as a dead constant."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
