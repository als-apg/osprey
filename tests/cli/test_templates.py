"""Tests for template generation system.

Tests the TemplateManager class and template rendering,
including validation that generated projects use the new
registry helper pattern correctly.
"""

import pytest
from click.testing import CliRunner

from osprey.cli.init_cmd import init
from osprey.cli.templates import claude_code, manifest
from osprey.cli.templates.manager import TemplateManager


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
            template_name="control_assistant",
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


class TestCLIIntegration:
    """Test CLI command integration with templates."""

    def test_init_command_basic(self, tmp_path):
        """Test basic init command."""
        runner = CliRunner()

        result = runner.invoke(init, ["test-project", "--output-dir", str(tmp_path)])

        assert result.exit_code == 0
        assert "Project created successfully" in result.output
        # New init shows Mode: Claude Code
        assert "Mode: Claude Code" in result.output

    def test_init_command_shows_next_steps(self, tmp_path):
        """Test that init command shows helpful next steps."""
        runner = CliRunner()

        result = runner.invoke(init, ["test-project", "--output-dir", str(tmp_path)])

        assert result.exit_code == 0
        # Should show next steps
        assert "Next steps:" in result.output
        assert "cd test-project" in result.output
        assert "claude" in result.output

    def test_init_command_with_channel_finder_mode(self, tmp_path):
        """Test init command with --channel-finder-mode option."""
        runner = CliRunner()

        result = runner.invoke(
            init,
            [
                "cf-app",
                "--template",
                "control_assistant",
                "--channel-finder-mode",
                "in_context",
                "--output-dir",
                str(tmp_path),
            ],
        )

        assert result.exit_code == 0

        # Check manifest has the option
        import json

        manifest_path = tmp_path / "cf-app" / ".osprey-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["init_args"]["channel_finder_mode"] == "in_context"
        assert "--channel-finder-mode in_context" in manifest["reproducible_command"]

    def test_init_command_with_code_generator(self, tmp_path):
        """Test init command with --code-generator option."""
        runner = CliRunner()

        result = runner.invoke(
            init,
            [
                "gen-app",
                "--template",
                "control_assistant",
                "--code-generator",
                "basic",
                "--output-dir",
                str(tmp_path),
            ],
        )

        assert result.exit_code == 0

        # Check manifest has the option
        import json

        manifest_path = tmp_path / "gen-app" / ".osprey-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["init_args"]["code_generator"] == "basic"
        assert "--code-generator basic" in manifest["reproducible_command"]

    def test_init_command_reproducible_command_complete(self, tmp_path):
        """Test that reproducible_command includes all options."""
        runner = CliRunner()

        result = runner.invoke(
            init,
            [
                "full-app",
                "--template",
                "control_assistant",
                "--provider",
                "cborg",
                "--model",
                "haiku",
                "--channel-finder-mode",
                "hierarchical",
                "--code-generator",
                "claude_code",
                "--output-dir",
                str(tmp_path),
            ],
        )

        assert result.exit_code == 0

        import json

        manifest_path = tmp_path / "full-app" / ".osprey-manifest.json"
        manifest = json.loads(manifest_path.read_text())

        cmd = manifest["reproducible_command"]
        # --template always included for reproducibility
        assert "--template control_assistant" in cmd
        assert "--provider cborg" in cmd
        assert "--model haiku" in cmd
        assert "--channel-finder-mode hierarchical" in cmd
        assert "--code-generator claude_code" in cmd


class TestGeneratorConfigRendering:
    """Test code generator config file rendering."""

    def test_claude_code_generator_config_rendered(self, tmp_path):
        """Test that claude_code generator creates config file."""
        manager = TemplateManager()

        project_dir = manager.create_project(
            project_name="claude-app",
            output_dir=tmp_path,
            template_name="control_assistant",
            context={"code_generator": "claude_code"},
        )

        config_file = project_dir / "claude_generator_config.yml"

        # Should exist
        assert config_file.exists(), "claude_generator_config.yml should be created"

        # Should have expected content
        content = config_file.read_text()
        assert "profile:" in content or "phases:" in content
        # Should have system prompt extensions section
        assert "system_prompt_extensions" in content

    def test_basic_generator_config_rendered(self, tmp_path):
        """Test that basic generator creates config file."""
        manager = TemplateManager()

        project_dir = manager.create_project(
            project_name="basic-app",
            output_dir=tmp_path,
            template_name="control_assistant",
            context={"code_generator": "basic"},
        )

        config_file = project_dir / "basic_generator_config.yml"

        # Should exist
        assert config_file.exists(), "basic_generator_config.yml should be created"

        # Should have expected content
        content = config_file.read_text()
        assert "system_role:" in content
        assert "core_requirements:" in content
        assert "system_prompt_extensions:" in content

    def test_no_generator_config_for_default(self, tmp_path):
        """Test that no generator config is created when generator not specified."""
        manager = TemplateManager()

        project_dir = manager.create_project(
            project_name="default-app",
            output_dir=tmp_path,
            template_name="control_assistant",
            # No code_generator in context
        )

        # Neither config should exist
        assert not (project_dir / "claude_generator_config.yml").exists()
        assert not (project_dir / "basic_generator_config.yml").exists()

    def test_generator_configs_mutually_exclusive(self, tmp_path):
        """Test that only one generator config is created at a time."""
        manager = TemplateManager()

        # Create with claude_code
        project_dir = manager.create_project(
            project_name="exclusive-app",
            output_dir=tmp_path,
            template_name="control_assistant",
            context={"code_generator": "claude_code"},
        )

        # Only claude config should exist
        assert (project_dir / "claude_generator_config.yml").exists()
        assert not (project_dir / "basic_generator_config.yml").exists()

    def test_generator_config_at_project_root(self, tmp_path):
        """Test that generator config is at project root."""
        manager = TemplateManager()

        project_dir = manager.create_project(
            project_name="placement-test",
            output_dir=tmp_path,
            template_name="control_assistant",
            context={"code_generator": "basic"},
        )

        # Should be at project root
        assert (project_dir / "basic_generator_config.yml").exists()


class TestGitIsolation:
    """Test that osprey init creates a self-contained git repo."""

    def test_init_creates_git_repo(self, tmp_path):
        """Test that osprey init creates a .git directory."""
        runner = CliRunner()

        result = runner.invoke(init, ["git-test", "--output-dir", str(tmp_path)])

        assert result.exit_code == 0
        project_dir = tmp_path / "git-test"
        assert (project_dir / ".git").exists(), ".git directory should be created"

    def test_init_initial_commit(self, tmp_path):
        """Test that osprey init creates an initial commit with expected files."""
        import subprocess

        runner = CliRunner()

        result = runner.invoke(init, ["commit-test", "--output-dir", str(tmp_path)])

        assert result.exit_code == 0
        project_dir = tmp_path / "commit-test"

        # Verify there is exactly one commit
        log = subprocess.run(
            ["git", "log", "--oneline"],
            cwd=project_dir,
            capture_output=True,
            text=True,
        )
        assert log.returncode == 0
        commits = log.stdout.strip().splitlines()
        assert len(commits) == 1
        assert "Initial project from osprey init" in commits[0]

        # Verify key files are tracked
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
        """Test that .claude/ is NOT in .gitignore (should be tracked)."""
        runner = CliRunner()

        result = runner.invoke(init, ["track-test", "--output-dir", str(tmp_path)])

        assert result.exit_code == 0
        project_dir = tmp_path / "track-test"

        gitignore_content = (project_dir / ".gitignore").read_text()
        # .claude/ should NOT appear as a top-level ignore pattern
        lines = [line.strip() for line in gitignore_content.splitlines()]
        assert ".claude/" not in lines, ".claude/ should not be ignored"
        assert "CLAUDE.md" not in lines, "CLAUDE.md should not be ignored"
        assert ".mcp.json" not in lines, ".mcp.json should not be ignored"

    def test_gitignore_ignores_local_settings(self, tmp_path):
        """Test that personal local settings are ignored."""
        runner = CliRunner()

        result = runner.invoke(init, ["local-test", "--output-dir", str(tmp_path)])

        assert result.exit_code == 0
        project_dir = tmp_path / "local-test"

        gitignore_content = (project_dir / ".gitignore").read_text()
        assert ".claude/settings.local.json" in gitignore_content
        assert "CLAUDE.local.md" in gitignore_content

    def test_init_output_mentions_git(self, tmp_path):
        """Test that init output confirms git initialization."""
        runner = CliRunner()

        result = runner.invoke(init, ["msg-test", "--output-dir", str(tmp_path)])

        assert result.exit_code == 0
        assert "Initialized git repository" in result.output
        assert "standalone git repo" in result.output

    def test_claude_dir_in_initial_commit(self, tmp_path):
        """Test that .claude/ files are included in the initial commit."""
        import subprocess

        runner = CliRunner()

        result = runner.invoke(init, ["claude-track-test", "--output-dir", str(tmp_path)])

        assert result.exit_code == 0
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

    def test_init_inside_existing_repo_warns(self, tmp_path):
        """Test that creating a project inside an existing git repo shows a warning."""
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
        result = runner.invoke(init, ["nested-test", "--output-dir", str(tmp_path)])

        assert result.exit_code == 0
        # Should still create the git repo (for isolation)
        project_dir = tmp_path / "nested-test"
        assert (project_dir / ".git").exists()
        # Should warn about nested repo
        assert "nested git repo" in result.output

    def test_init_inside_existing_repo_still_isolates(self, tmp_path):
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
        result = runner.invoke(init, ["isolated-test", "--output-dir", str(tmp_path)])

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

    def test_init_clears_stale_trust_state(self, tmp_path):
        """Test that init clears trust state even when directory was already deleted."""
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

            # Directory does NOT exist — simulates rm -rf before osprey init
            assert not project_path.exists()

            # Create project (no --force needed, directory is gone)
            runner = CliRunner()
            result = runner.invoke(init, ["stale-test", "--output-dir", str(tmp_path)])
            assert result.exit_code == 0

            # Trust entry should be gone
            after = json.loads(claude_json.read_text())
            assert str(project_path) not in after.get("projects", {}), (
                "Stale trust entry should be cleared on init"
            )
        finally:
            if backup_data is not None:
                claude_json.write_text(backup_data)

    def test_force_clears_claude_code_session_state(self, tmp_path):
        """Test that --force removes Claude Code's session directory."""
        from pathlib import Path

        runner = CliRunner()

        # Create project first time
        result = runner.invoke(init, ["session-test", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0

        project_path = (tmp_path / "session-test").resolve()

        # Simulate Claude Code's session directory
        encoded_key = str(project_path).replace("/", "-")
        claude_project_dir = Path.home() / ".claude" / "projects" / encoded_key
        claude_project_dir.mkdir(parents=True, exist_ok=True)
        (claude_project_dir / "sessions-index.json").write_text("{}")

        assert claude_project_dir.exists()

        # Re-create with --force
        result = runner.invoke(init, ["session-test", "--output-dir", str(tmp_path), "--force"])
        assert result.exit_code == 0

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
            template_name="control_assistant",
            context={"channel_finder_mode": "hierarchical"},
        )

        agent_prompt = (project_dir / ".claude" / "agents" / "channel-finder.md").read_text()
        # Must contain embedded hierarchy info, NOT the fallback text
        assert "hierarchy_levels" in agent_prompt
        assert "Call `get_options()` at the first level to discover" not in agent_prompt


class TestLatticeSkillIsolation:
    """Test that lattice-specific skills/rules only appear for lattice_design."""

    LATTICE_SKILLS = [
        # lattice-evaluation skill was replaced by the lattice dashboard panel
    ]

    def test_control_assistant_has_no_lattice_skills(self, tmp_path):
        """control_assistant projects must not contain lattice skill directories."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="ctrl-test",
            output_dir=tmp_path,
            template_name="control_assistant",
        )

        skills_dir = project_dir / ".claude" / "skills"
        for skill_name in self.LATTICE_SKILLS:
            skill_dir = skills_dir / skill_name
            assert not skill_dir.exists(), (
                f"Lattice skill '{skill_name}' should not exist in control_assistant project"
            )

    def test_control_assistant_has_no_lattice_rule(self, tmp_path):
        """control_assistant projects must not contain lattice-physics rule."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="ctrl-rule-test",
            output_dir=tmp_path,
            template_name="control_assistant",
        )

        rule_file = project_dir / ".claude" / "rules" / "lattice-physics.md"
        assert not rule_file.exists(), (
            "lattice-physics.md rule should not exist in control_assistant project"
        )

    def test_lattice_design_has_all_skills(self, tmp_path):
        """lattice_design projects must contain all lattice skills with content."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="lat-test",
            output_dir=tmp_path,
            template_name="lattice_design",
        )

        skills_dir = project_dir / ".claude" / "skills"
        for skill_name in self.LATTICE_SKILLS:
            skill_file = skills_dir / skill_name / "SKILL.md"
            assert skill_file.exists(), (
                f"Lattice skill '{skill_name}/SKILL.md' should exist in lattice_design project"
            )
            content = skill_file.read_text(encoding="utf-8").strip()
            assert len(content) > 0, f"Lattice skill '{skill_name}/SKILL.md' should not be empty"

    def test_lattice_design_has_lattice_rule(self, tmp_path):
        """lattice_design projects must contain lattice-physics rule with content."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="lat-rule-test",
            output_dir=tmp_path,
            template_name="lattice_design",
        )

        rule_file = project_dir / ".claude" / "rules" / "lattice-physics.md"
        assert rule_file.exists(), "lattice-physics.md rule should exist in lattice_design project"
        content = rule_file.read_text(encoding="utf-8").strip()
        assert len(content) > 0, "lattice-physics.md rule should not be empty"


class TestTemplateManifest:
    """Test template manifest loading, resolution, and filtering."""

    def test_load_manifest_control_assistant(self):
        """Control assistant manifest loads with all expected categories."""
        manager = TemplateManager()
        mf = manifest.load_template_manifest(manager.template_root, "control_assistant")

        assert mf is not None
        artifacts = mf["artifacts"]
        assert "hooks" in artifacts
        assert "rules" in artifacts
        assert "skills" in artifacts
        assert "agents" in artifacts
        assert "output_styles" in artifacts
        assert "approval" in artifacts["hooks"]
        assert "channel-finder" in artifacts["agents"]

    def test_load_manifest_lattice_design(self):
        """Lattice design manifest loads with lattice-specific entries."""
        manager = TemplateManager()
        mf = manifest.load_template_manifest(manager.template_root, "lattice_design")

        assert mf is not None
        artifacts = mf["artifacts"]
        assert "lattice-physics" in artifacts["rules"]
        # Should NOT have control-system-specific entries
        assert "limits" not in artifacts["hooks"]
        assert "cf-feedback-capture" not in artifacts["hooks"]
        assert "control-system-safety" not in artifacts.get("rules", [])

    def test_load_manifest_nonexistent_template(self):
        """Returns None for unknown template."""
        manager = TemplateManager()
        mf = manifest.load_template_manifest(manager.template_root, "nonexistent_template")
        assert mf is None

    def test_resolve_manifest_outputs_includes_config_artifacts(self):
        """Resolved outputs always contain config artifacts."""
        manager = TemplateManager()
        mf = manifest.load_template_manifest(manager.template_root, "control_assistant")
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

    def test_lattice_init_no_control_artifacts(self, tmp_path):
        """Lattice project must not contain control-system-specific artifacts."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="lat-manifest-test",
            output_dir=tmp_path,
            template_name="lattice_design",
        )

        # These should NOT exist
        assert not (project_dir / ".claude" / "hooks" / "osprey_limits.py").exists()
        assert not (project_dir / ".claude" / "hooks" / "osprey_cf_feedback_capture.py").exists()
        assert not (project_dir / ".claude" / "rules" / "control-system-safety.md").exists()

    def test_lattice_init_has_lattice_artifacts(self, tmp_path):
        """Lattice project must have lattice-physics rule and reference scripts."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="lat-artifacts-test",
            output_dir=tmp_path,
            template_name="lattice_design",
        )

        # Lattice-physics rule
        rule = project_dir / ".claude" / "rules" / "lattice-physics.md"
        assert rule.exists()
        assert len(rule.read_text(encoding="utf-8").strip()) > 0

        # lattice-evaluation SKILL.md was replaced by the dashboard panel
        # but the 4 reference scripts should still be deployed
        refs_dir = project_dir / ".claude" / "skills" / "lattice-evaluation" / "references"
        for ref_name in [
            "analyze_working_point.py",
            "resonance_diagram.py",
            "dynamic_aperture.py",
            "frequency_map.py",
        ]:
            ref_file = refs_dir / ref_name
            assert ref_file.exists(), f"Missing reference script: {ref_name}"
            assert len(ref_file.read_text(encoding="utf-8").strip()) > 0

    def test_lattice_init_agents(self, tmp_path):
        """Lattice project must have expected agents."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="lat-agents-test",
            output_dir=tmp_path,
            template_name="lattice_design",
        )

        agents_dir = project_dir / ".claude" / "agents"
        agent_files = sorted(f.name for f in agents_dir.iterdir() if f.suffix == ".md")
        assert agent_files == [
            "data-visualizer.md",
            "literature-search.md",
            "textbook-expert.md",
        ]

    def test_control_assistant_has_all_hooks(self, tmp_path):
        """Control assistant project must have all 8 hook files."""
        manager = TemplateManager()
        project_dir = manager.create_project(
            project_name="ctrl-hooks-test",
            output_dir=tmp_path,
            template_name="control_assistant",
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
