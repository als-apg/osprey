"""Tests for the osprey build command and build profile system."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from textwrap import dedent

import pytest
import yaml

from osprey.cli.build_profile import (
    BuildProfile,
    EnvConfig,
    LifecycleConfig,
    LifecycleStep,
    McpServerDef,
    load_profile,
)
from osprey.errors import BuildProfileError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def profile_dir(tmp_path: Path) -> Path:
    """Create a minimal profile directory with overlay sources."""
    # Create overlay source files
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "channels.json").write_text('{"pvs": ["SR:DCCT"]}')

    mcp_dir = tmp_path / "mcp_servers" / "test_server"
    mcp_dir.mkdir(parents=True)
    (mcp_dir / "__init__.py").write_text("")
    (mcp_dir / "server.py").write_text("# test server")

    return tmp_path


@pytest.fixture()
def minimal_profile_yaml(profile_dir: Path) -> Path:
    """Write a minimal valid profile YAML and return its path."""
    profile = {
        "name": "Test Profile",
        "data_bundle": "control_assistant",
        "provider": "cborg",
        "model": "haiku",
        "config": {
            "control_system.type": "mock",
        },
        "overlay": {
            "data/channels.json": "data/channel_databases/channels.json",
        },
        "mcp_servers": {
            "test_server": {
                "command": "python",
                "args": ["-m", "test_server"],
                "env": {
                    "CONFIG": "{project_root}/config.yml",
                },
                "permissions": {
                    "allow": ["test_tool"],
                    "ask": ["dangerous_tool"],
                },
            },
        },
    }
    path = profile_dir / "test-profile.yml"
    path.write_text(yaml.dump(profile, default_flow_style=False))
    return path


# ---------------------------------------------------------------------------
# Profile Loading
# ---------------------------------------------------------------------------


class TestProfileLoading:
    """Tests for load_profile() and YAML parsing."""

    def test_load_minimal_profile(self, minimal_profile_yaml: Path):
        profile = load_profile(minimal_profile_yaml)
        assert profile.name == "Test Profile"
        assert profile.data_bundle == "control_assistant"
        assert profile.provider == "cborg"
        assert profile.model == "haiku"

    def test_load_profile_not_found(self, tmp_path: Path):
        with pytest.raises(BuildProfileError, match="Profile not found"):
            load_profile(tmp_path / "nonexistent.yml")

    def test_load_profile_invalid_yaml(self, tmp_path: Path):
        bad_yaml = tmp_path / "bad.yml"
        bad_yaml.write_text("{{invalid yaml: [")
        with pytest.raises(BuildProfileError, match="Invalid YAML"):
            load_profile(bad_yaml)

    def test_load_profile_not_a_mapping(self, tmp_path: Path):
        list_yaml = tmp_path / "list.yml"
        list_yaml.write_text("- item1\n- item2\n")
        with pytest.raises(BuildProfileError, match="must be a YAML mapping"):
            load_profile(list_yaml)

    def test_load_profile_config_parsed(self, minimal_profile_yaml: Path):
        profile = load_profile(minimal_profile_yaml)
        assert profile.config["control_system.type"] == "mock"

    def test_load_profile_mcp_servers_parsed(self, minimal_profile_yaml: Path):
        profile = load_profile(minimal_profile_yaml)
        assert "test_server" in profile.mcp_servers
        server = profile.mcp_servers["test_server"]
        assert server.command == "python"
        assert server.args == ["-m", "test_server"]
        assert server.env == {"CONFIG": "{project_root}/config.yml"}
        assert server.permissions == {"allow": ["test_tool"], "ask": ["dangerous_tool"]}

    def test_load_profile_mcp_server_url(self, tmp_path: Path):
        """URL-only MCP server should parse correctly."""
        p = tmp_path / "profile.yml"
        p.write_text(
            yaml.dump(
                {
                    "name": "Test",
                    "mcp_servers": {
                        "remote": {
                            "url": "http://host:8001/sse",
                            "permissions": {"allow": ["search"]},
                        }
                    },
                }
            )
        )
        profile = load_profile(p)
        server = profile.mcp_servers["remote"]
        assert server.url == "http://host:8001/sse"
        assert server.command == ""
        assert server.permissions == {"allow": ["search"], "ask": []}

    def test_load_profile_mcp_server_both_command_and_url_rejected(self, tmp_path: Path):
        """Server with both command and url should be rejected at parse time."""
        p = tmp_path / "profile.yml"
        p.write_text(
            yaml.dump(
                {
                    "name": "Test",
                    "mcp_servers": {
                        "bad": {"command": "npx", "url": "http://host:8001/sse"}
                    },
                }
            )
        )
        with pytest.raises(BuildProfileError, match="both 'command' and 'url'"):
            load_profile(p)

    def test_load_profile_defaults(self, tmp_path: Path):
        """Profile with only name should use defaults."""
        simple = tmp_path / "simple.yml"
        simple.write_text("name: Simple\n")
        profile = load_profile(simple)
        assert profile.data_bundle == "control_assistant"
        assert profile.provider is None
        assert profile.config == {}
        assert profile.overlay == {}
        assert profile.mcp_servers == {}
        assert profile.lifecycle == LifecycleConfig()
        assert profile.env == EnvConfig()
        assert profile.dependencies == []
        assert profile.hooks == []
        assert profile.rules == []
        assert profile.skills == []
        assert profile.agents == []
        assert profile.output_styles == []
        assert profile.web_panels == []

    def test_load_profile_lifecycle_parsed(self, tmp_path: Path):
        profile_data = {
            "name": "Lifecycle Test",
            "lifecycle": {
                "pre_build": [{"name": "check deps", "run": "pip check"}],
                "post_build": [{"name": "build index", "run": "python index.py", "cwd": "data"}],
                "validate": [{"name": "smoke test", "run": "python -c 'print(1)'"}],
            },
        }
        path = tmp_path / "lc.yml"
        path.write_text(yaml.dump(profile_data, default_flow_style=False))
        profile = load_profile(path)
        assert len(profile.lifecycle.pre_build) == 1
        assert profile.lifecycle.pre_build[0].name == "check deps"
        assert profile.lifecycle.pre_build[0].run == "pip check"
        assert len(profile.lifecycle.post_build) == 1
        assert profile.lifecycle.post_build[0].cwd == "data"
        assert len(profile.lifecycle.validate) == 1

    def test_load_profile_env_parsed(self, tmp_path: Path):
        profile_data = {
            "name": "Env Test",
            "env": {
                "required": ["API_KEY", "DB_HOST"],
                "defaults": {"LOG_LEVEL": "info", "PORT": "8080"},
            },
        }
        path = tmp_path / "env.yml"
        path.write_text(yaml.dump(profile_data, default_flow_style=False))
        profile = load_profile(path)
        assert profile.env.required == ["API_KEY", "DB_HOST"]
        assert profile.env.defaults == {"LOG_LEVEL": "info", "PORT": "8080"}

    def test_load_profile_dependencies_parsed(self, tmp_path: Path):
        profile_data = {
            "name": "Deps Test",
            "dependencies": ["numpy>=1.24", "pandas", "scipy~=1.11"],
        }
        path = tmp_path / "deps.yml"
        path.write_text(yaml.dump(profile_data, default_flow_style=False))
        profile = load_profile(path)
        assert profile.dependencies == ["numpy>=1.24", "pandas", "scipy~=1.11"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    """Tests for BuildProfile.validate()."""

    def test_missing_overlay_source(self, tmp_path: Path):
        profile = BuildProfile(
            name="Test",
            overlay={"nonexistent/file.json": "data/file.json"},
        )
        with pytest.raises(BuildProfileError, match="Overlay source not found"):
            profile.validate(tmp_path)

    def test_path_traversal_blocked(self, tmp_path: Path):
        profile = BuildProfile(
            name="Test",
            overlay={},
        )
        # Manually add a traversal path
        profile.overlay["data/x.json"] = "../../../etc/passwd"
        # Create the source file so we don't hit that error
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "x.json").write_text("{}")
        with pytest.raises(BuildProfileError, match="must be relative without"):
            profile.validate(tmp_path)

    def test_absolute_overlay_destination_blocked(self, tmp_path: Path):
        profile = BuildProfile(
            name="Test",
            overlay={"x.json": "/tmp/evil.json"},
        )
        (tmp_path / "x.json").write_text("{}")
        with pytest.raises(BuildProfileError, match="must be relative without"):
            profile.validate(tmp_path)

    def test_missing_mcp_server_command_or_url(self, tmp_path: Path):
        profile = BuildProfile(
            name="Test",
            mcp_servers={"broken": McpServerDef()},
        )
        with pytest.raises(BuildProfileError, match="missing 'command' or 'url'"):
            profile.validate(tmp_path)

    def test_mcp_server_url_only_passes_validation(self, tmp_path: Path):
        profile = BuildProfile(
            name="Test",
            mcp_servers={"remote": McpServerDef(url="http://host:8001/sse")},
        )
        profile.validate(tmp_path)  # Should not raise

    def test_missing_name_reported(self, tmp_path: Path):
        profile = BuildProfile(name="")
        with pytest.raises(BuildProfileError, match="'name' is required"):
            profile.validate(tmp_path)

    def test_lifecycle_step_missing_name(self, tmp_path: Path):
        profile = BuildProfile(
            name="Test",
            lifecycle=LifecycleConfig(
                pre_build=[LifecycleStep(name="", run="echo hello")],
            ),
        )
        with pytest.raises(BuildProfileError, match="missing 'name'"):
            profile.validate(tmp_path)

    def test_lifecycle_step_missing_run(self, tmp_path: Path):
        profile = BuildProfile(
            name="Test",
            lifecycle=LifecycleConfig(
                post_build=[LifecycleStep(name="broken", run="")],
            ),
        )
        with pytest.raises(BuildProfileError, match="missing 'run'"):
            profile.validate(tmp_path)

    def test_lifecycle_step_absolute_cwd_blocked(self, tmp_path: Path):
        profile = BuildProfile(
            name="Test",
            lifecycle=LifecycleConfig(
                pre_build=[LifecycleStep(name="bad", run="echo", cwd="/tmp/evil")],
            ),
        )
        with pytest.raises(BuildProfileError, match="cwd must be relative without"):
            profile.validate(tmp_path)

    def test_lifecycle_step_traversal_cwd_blocked(self, tmp_path: Path):
        profile = BuildProfile(
            name="Test",
            lifecycle=LifecycleConfig(
                pre_build=[LifecycleStep(name="bad", run="echo", cwd="../escape")],
            ),
        )
        with pytest.raises(BuildProfileError, match="cwd must be relative without"):
            profile.validate(tmp_path)

    def test_invalid_env_var_name(self, tmp_path: Path):
        profile = BuildProfile(
            name="Test",
            env=EnvConfig(required=["lowercase_bad"]),
        )
        with pytest.raises(BuildProfileError, match="Invalid env var name"):
            profile.validate(tmp_path)

    def test_empty_dependency_rejected(self, tmp_path: Path):
        profile = BuildProfile(
            name="Test",
            dependencies=["numpy", ""],
        )
        with pytest.raises(BuildProfileError, match="non-empty string"):
            profile.validate(tmp_path)

    def test_valid_profile_passes(self, profile_dir: Path, minimal_profile_yaml: Path):
        profile = load_profile(minimal_profile_yaml)
        # Should not raise
        profile.validate(profile_dir)


# ---------------------------------------------------------------------------
# Build Command Helpers
# ---------------------------------------------------------------------------


class TestBuildHelpers:
    """Tests for build_cmd.py helper functions."""

    def test_persist_mcp_servers_url(self, tmp_path: Path):
        """_persist_mcp_servers writes URL server to config.yml claude_code.servers."""
        from osprey.cli.build_cmd import _persist_mcp_servers

        project_path = tmp_path / "project"
        project_path.mkdir()
        (project_path / "config.yml").write_text("facility_name: test\n")

        servers = {
            "matlab": McpServerDef(url="http://matlab:8001/sse"),
        }
        _persist_mcp_servers(project_path, servers)

        config = yaml.safe_load((project_path / "config.yml").read_text())
        assert "matlab" in config["claude_code"]["servers"]
        assert config["claude_code"]["servers"]["matlab"]["url"] == "http://matlab:8001/sse"

    def test_persist_mcp_servers_stdio(self, tmp_path: Path):
        """_persist_mcp_servers writes stdio server with command/args/env."""
        from osprey.cli.build_cmd import _persist_mcp_servers

        project_path = tmp_path / "project"
        project_path.mkdir()
        (project_path / "config.yml").write_text("facility_name: test\n")

        servers = {
            "confluence": McpServerDef(
                command="uvx",
                args=["--python=3.12", "mcp-atlassian"],
                env={"CONFLUENCE_URL": "https://wiki.example.com"},
            ),
        }
        _persist_mcp_servers(project_path, servers)

        config = yaml.safe_load((project_path / "config.yml").read_text())
        entry = config["claude_code"]["servers"]["confluence"]
        assert entry["command"] == "uvx"
        assert entry["args"] == ["--python=3.12", "mcp-atlassian"]
        assert entry["env"]["CONFLUENCE_URL"] == "https://wiki.example.com"

    def test_persist_mcp_servers_permissions(self, tmp_path: Path):
        """_persist_mcp_servers preserves permission structure in config.yml."""
        from osprey.cli.build_cmd import _persist_mcp_servers

        project_path = tmp_path / "project"
        project_path.mkdir()
        (project_path / "config.yml").write_text("facility_name: test\n")

        servers = {
            "phoebus": McpServerDef(
                url="http://phoebus:8003/sse",
                permissions={"allow": ["phoebus_launch"], "ask": ["dangerous_op"]},
            ),
        }
        _persist_mcp_servers(project_path, servers)

        config = yaml.safe_load((project_path / "config.yml").read_text())
        entry = config["claude_code"]["servers"]["phoebus"]
        assert entry["permissions"]["allow"] == ["phoebus_launch"]
        assert entry["permissions"]["ask"] == ["dangerous_op"]

    def test_copy_overlay_path_traversal_guard(self, tmp_path: Path):
        """_copy_overlay_files should reject destinations that escape project root."""
        from osprey.cli.build_cmd import _copy_overlay_files

        profile_dir = tmp_path / "profile"
        profile_dir.mkdir()
        (profile_dir / "evil.txt").write_text("evil")

        project_path = tmp_path / "project"
        project_path.mkdir()

        overlay = {"evil.txt": "../../../tmp/evil.txt"}
        with pytest.raises(ValueError, match="escapes project root"):
            _copy_overlay_files(profile_dir, project_path, overlay)

    def test_copy_overlay_files(self, tmp_path: Path):
        """_copy_overlay_files should copy files into the project."""
        from osprey.cli.build_cmd import _copy_overlay_files

        profile_dir = tmp_path / "profile"
        profile_dir.mkdir()
        (profile_dir / "data.json").write_text('{"key": "value"}')

        project_path = tmp_path / "project"
        project_path.mkdir()

        overlay = {"data.json": "config/data.json"}
        _copy_overlay_files(profile_dir, project_path, overlay)

        assert (project_path / "config" / "data.json").exists()
        assert json.loads((project_path / "config" / "data.json").read_text()) == {"key": "value"}

    def test_copy_overlay_directory(self, tmp_path: Path):
        """_copy_overlay_files should handle directory overlays."""
        from osprey.cli.build_cmd import _copy_overlay_files

        profile_dir = tmp_path / "profile"
        src_dir = profile_dir / "server_pkg"
        src_dir.mkdir(parents=True)
        (src_dir / "__init__.py").write_text("")
        (src_dir / "main.py").write_text("# main")

        project_path = tmp_path / "project"
        project_path.mkdir()

        overlay = {"server_pkg": "_mcp_servers/server_pkg"}
        _copy_overlay_files(profile_dir, project_path, overlay)

        assert (project_path / "_mcp_servers" / "server_pkg" / "__init__.py").exists()
        assert (project_path / "_mcp_servers" / "server_pkg" / "main.py").exists()

    def test_persist_mcp_servers_writes_to_config(self, tmp_path: Path):
        """_persist_mcp_servers adds servers to claude_code.servers in config.yml."""
        from osprey.cli.build_cmd import _persist_mcp_servers

        project_path = tmp_path / "project"
        project_path.mkdir()
        (project_path / "config.yml").write_text("facility_name: test\n")

        servers = {
            "remote_server": McpServerDef(
                url="http://remote-host:8001/sse",
                permissions={"allow": ["search", "get"], "ask": []},
            ),
            "local_tool": McpServerDef(
                command="npx",
                args=["-y", "some-mcp-server"],
                env={"API_KEY": "secret"},
                permissions={"allow": ["do_thing"], "ask": []},
            ),
        }
        _persist_mcp_servers(project_path, servers)

        config = yaml.safe_load((project_path / "config.yml").read_text())
        persisted = config["claude_code"]["servers"]

        # URL server
        assert persisted["remote_server"]["url"] == "http://remote-host:8001/sse"
        assert "command" not in persisted["remote_server"]
        assert persisted["remote_server"]["permissions"]["allow"] == ["search", "get"]

        # Stdio server
        assert persisted["local_tool"]["command"] == "npx"
        assert persisted["local_tool"]["args"] == ["-y", "some-mcp-server"]
        assert persisted["local_tool"]["env"]["API_KEY"] == "secret"
        assert "url" not in persisted["local_tool"]

    def test_persist_mcp_servers_preserves_existing_config(self, tmp_path: Path):
        """_persist_mcp_servers doesn't clobber existing config.yml entries."""
        from osprey.cli.build_cmd import _persist_mcp_servers

        project_path = tmp_path / "project"
        project_path.mkdir()
        (project_path / "config.yml").write_text(
            "facility_name: test\nclaude_code:\n  permissions:\n    allow: [existing]\n"
        )

        servers = {
            "my_server": McpServerDef(url="http://host:8001/sse"),
        }
        _persist_mcp_servers(project_path, servers)

        config = yaml.safe_load((project_path / "config.yml").read_text())
        # Original config preserved
        assert config["facility_name"] == "test"
        assert config["claude_code"]["permissions"]["allow"] == ["existing"]
        # Server added
        assert config["claude_code"]["servers"]["my_server"]["url"] == "http://host:8001/sse"

    def test_apply_config_overrides(self, tmp_path: Path):
        """_apply_config_overrides should update config.yml fields."""
        from osprey.cli.build_cmd import _apply_config_overrides

        project_path = tmp_path / "project"
        project_path.mkdir()
        config_path = project_path / "config.yml"
        config_path.write_text(
            dedent("""\
            control_system:
              type: mock
            archiver:
              type: mock_archiver
            """)
        )

        _apply_config_overrides(
            project_path,
            {
                "control_system.type": "epics",
                "system.timezone": "America/Los_Angeles",
            },
        )

        updated = yaml.safe_load(config_path.read_text())
        assert updated["control_system"]["type"] == "epics"
        assert updated["system"]["timezone"] == "America/Los_Angeles"
        # Verify untouched fields preserved
        assert updated["archiver"]["type"] == "mock_archiver"


# ---------------------------------------------------------------------------
# Environment Template
# ---------------------------------------------------------------------------


class TestEnvTemplate:
    """Tests for _generate_env_template()."""

    def test_generates_required_vars(self, tmp_path: Path):
        from osprey.cli.build_cmd import _generate_env_template

        project_path = tmp_path / "project"
        project_path.mkdir()
        env = EnvConfig(required=["API_KEY", "DB_HOST"])
        _generate_env_template(project_path, env)

        content = (project_path / ".env.template").read_text()
        assert "# Required" in content
        assert "API_KEY=" in content
        assert "DB_HOST=" in content

    def test_generates_defaults(self, tmp_path: Path):
        from osprey.cli.build_cmd import _generate_env_template

        project_path = tmp_path / "project"
        project_path.mkdir()
        env = EnvConfig(defaults={"LOG_LEVEL": "info", "PORT": "8080"})
        _generate_env_template(project_path, env)

        content = (project_path / ".env.template").read_text()
        assert "# Defaults" in content
        assert "LOG_LEVEL=info" in content
        assert "PORT=8080" in content

    def test_generates_both_sections(self, tmp_path: Path):
        from osprey.cli.build_cmd import _generate_env_template

        project_path = tmp_path / "project"
        project_path.mkdir()
        env = EnvConfig(required=["API_KEY"], defaults={"PORT": "8080"})
        _generate_env_template(project_path, env)

        content = (project_path / ".env.template").read_text()
        assert "# Required" in content
        assert "API_KEY=" in content
        assert "# Defaults" in content
        assert "PORT=8080" in content
        # Required section comes before defaults
        assert content.index("# Required") < content.index("# Defaults")


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


class TestRequirementsRecording:
    """Tests for the requirements.txt recording in _create_project_venv()."""

    def _make_profile(self, deps: list[str], osprey_install: str = "local") -> BuildProfile:
        return BuildProfile(
            name="test",
            dependencies=deps,
            osprey_install=osprey_install,
        )

    def test_records_deps_in_requirements_txt(self, monkeypatch, tmp_path: Path):
        from osprey.cli.build_cmd import _create_project_venv

        project_path = tmp_path / "project"
        project_path.mkdir()
        # Pre-seed requirements.txt (template may have created it)
        (project_path / "requirements.txt").write_text("# base\n")

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr("osprey.cli.build_cmd.subprocess.run", fake_run)
        monkeypatch.setenv("UV", "/usr/bin/uv")

        _create_project_venv(project_path, self._make_profile(["numpy>=1.24", "pandas"]))

        content = (project_path / "requirements.txt").read_text()
        assert "# base" in content  # original content preserved
        assert "# Profile dependencies" in content
        assert "numpy>=1.24" in content
        assert "pandas" in content

    def test_records_osprey_spec(self, monkeypatch, tmp_path: Path):
        from osprey.cli.build_cmd import _create_project_venv

        project_path = tmp_path / "project"
        project_path.mkdir()

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr("osprey.cli.build_cmd.subprocess.run", fake_run)
        monkeypatch.setenv("UV", "/usr/bin/uv")

        _create_project_venv(project_path, self._make_profile([], osprey_install="pip"))

        content = (project_path / "requirements.txt").read_text()
        assert "osprey-framework" in content


# ---------------------------------------------------------------------------
# Lifecycle Phase Runner
# ---------------------------------------------------------------------------


class TestLifecyclePhaseRunner:
    """Tests for _run_lifecycle_phase()."""

    def test_successful_step(self, tmp_path: Path):
        from osprey.cli.build_cmd import _run_lifecycle_phase

        steps = [LifecycleStep(name="echo test", run="echo hello")]
        # Should not raise
        _run_lifecycle_phase("post_build", steps, tmp_path, tmp_path)

    def test_failing_step_aborts(self, tmp_path: Path):
        from osprey.cli.build_cmd import _run_lifecycle_phase

        steps = [LifecycleStep(name="bad cmd", run="false")]
        with pytest.raises(BuildProfileError, match="'bad cmd' failed"):
            _run_lifecycle_phase("pre_build", steps, tmp_path, tmp_path)

    def test_failing_step_warns_when_no_abort(self, tmp_path: Path):
        from osprey.cli.build_cmd import _run_lifecycle_phase

        steps = [LifecycleStep(name="bad validate", run="false")]
        # Should not raise
        _run_lifecycle_phase("validate", steps, tmp_path, tmp_path, abort_on_failure=False)

    def test_step_with_cwd(self, tmp_path: Path):
        from osprey.cli.build_cmd import _run_lifecycle_phase

        subdir = tmp_path / "subdir"
        subdir.mkdir()
        steps = [LifecycleStep(name="check cwd", run="pwd", cwd="subdir")]
        # Should not raise — cwd relative to default_cwd
        _run_lifecycle_phase("post_build", steps, tmp_path, tmp_path)

    def test_project_root_placeholder(self, tmp_path: Path):
        from osprey.cli.build_cmd import _run_lifecycle_phase

        marker = tmp_path / "marker.txt"
        steps = [
            LifecycleStep(
                name="touch marker",
                run="touch {project_root}/marker.txt",
            )
        ]
        _run_lifecycle_phase("post_build", steps, tmp_path, tmp_path)
        assert marker.exists()

    def test_shell_metacharacters_handled(self, tmp_path: Path):
        from osprey.cli.build_cmd import _run_lifecycle_phase

        steps = [LifecycleStep(name="piped cmd", run="echo hello | cat")]
        # Should not raise — shell=True for pipe
        _run_lifecycle_phase("post_build", steps, tmp_path, tmp_path)


# ---------------------------------------------------------------------------
# Install Dependencies
# ---------------------------------------------------------------------------


class TestCreateProjectVenv:
    """Tests for _create_project_venv() — creates venv and installs deps."""

    def _make_profile(self, deps: list[str], osprey_install: str = "local") -> BuildProfile:
        return BuildProfile(
            name="test",
            dependencies=deps,
            osprey_install=osprey_install,
        )

    def test_creates_venv_and_installs_with_uv(self, monkeypatch, tmp_path):
        """Should create project venv then install deps with uv."""
        from osprey.cli.build_cmd import _create_project_venv

        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr("osprey.cli.build_cmd.subprocess.run", fake_run)
        monkeypatch.setenv("UV", "/home/user/.local/bin/uv")

        _create_project_venv(tmp_path, self._make_profile(["numpy>=1.24", "pandas"]))

        assert len(calls) == 2
        # First call: create venv
        venv_cmd = calls[0]
        assert venv_cmd[0] == "/home/user/.local/bin/uv"
        assert "venv" in venv_cmd
        assert str(tmp_path / ".venv") in venv_cmd
        # Second call: install deps
        install_cmd = calls[1]
        assert install_cmd[0] == "/home/user/.local/bin/uv"
        assert install_cmd[1:3] == ["pip", "install"]
        assert "numpy>=1.24" in install_cmd
        assert "pandas" in install_cmd

    def test_falls_back_to_stdlib_venv_and_pip(self, monkeypatch, tmp_path):
        """Should use python -m venv + pip when uv is not available."""
        import sys

        from osprey.cli.build_cmd import _create_project_venv

        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr("osprey.cli.build_cmd.subprocess.run", fake_run)
        monkeypatch.delenv("UV", raising=False)
        monkeypatch.setattr("shutil.which", lambda name: None)

        _create_project_venv(tmp_path, self._make_profile(["numpy>=1.24"]))

        assert len(calls) == 2
        # First call: python -m venv
        venv_cmd = calls[0]
        assert venv_cmd[0] == sys.executable
        assert "-m" in venv_cmd and "venv" in venv_cmd
        # Second call: pip install via venv python
        install_cmd = calls[1]
        assert str(tmp_path / ".venv" / "bin" / "python") in install_cmd
        assert "-m" in install_cmd and "pip" in install_cmd

    def test_raises_on_venv_failure(self, monkeypatch, tmp_path):
        """Should raise BuildProfileError if venv creation fails."""
        from osprey.cli.build_cmd import _create_project_venv

        def fake_run(cmd, **kwargs):
            return subprocess.CompletedProcess(
                args=cmd, returncode=1, stdout="", stderr="venv error"
            )

        monkeypatch.setattr("osprey.cli.build_cmd.subprocess.run", fake_run)
        monkeypatch.setenv("UV", "/usr/bin/uv")

        with pytest.raises(BuildProfileError, match="Failed to create project venv"):
            _create_project_venv(tmp_path, self._make_profile(["pkg"]))

    def test_raises_on_install_failure(self, monkeypatch, tmp_path):
        """Should raise BuildProfileError when pip install fails."""
        from osprey.cli.build_cmd import _create_project_venv

        call_count = 0

        def fake_run(cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Venv creation succeeds
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
            # Install fails
            return subprocess.CompletedProcess(
                args=cmd, returncode=1, stdout="", stderr="ERROR: No matching distribution"
            )

        monkeypatch.setattr("osprey.cli.build_cmd.subprocess.run", fake_run)
        monkeypatch.setenv("UV", "/usr/bin/uv")

        with pytest.raises(BuildProfileError, match="Failed to install project dependencies"):
            _create_project_venv(tmp_path, self._make_profile(["nonexistent-xyz"]))


# ---------------------------------------------------------------------------
# CLI Integration
# ---------------------------------------------------------------------------


class TestBuildCLI:
    """Tests for the Click command integration."""

    def test_build_command_exists(self):
        """Verify build command is registered in the CLI."""
        from click.testing import CliRunner

        from osprey.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["build", "--help"])
        assert result.exit_code == 0
        assert "Build a facility-specific assistant" in result.output

    def test_build_command_missing_profile(self):
        """Build should fail if profile file doesn't exist."""
        from click.testing import CliRunner

        from osprey.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["build", "test-proj", "/nonexistent/profile.yml"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Profile Inheritance (extends:)
# ---------------------------------------------------------------------------


def _write_yaml(path: Path, data: dict) -> Path:
    """Helper to write a YAML file and return its path."""
    path.write_text(yaml.dump(data, default_flow_style=False))
    return path


class TestProfileExtends:
    """Tests for profile inheritance via the ``extends:`` keyword."""

    def _make_base(self, tmp_path: Path) -> Path:
        """Create a base profile with overlay source files."""
        # Create overlay source so validation passes
        data_dir = tmp_path / "data"
        data_dir.mkdir(exist_ok=True)
        (data_dir / "channels.json").write_text("{}")

        return _write_yaml(
            tmp_path / "base.yml",
            {
                "name": "Base Profile",
                "data_bundle": "control_assistant",
                "provider": "cborg",
                "model": "opus",
                "hooks": ["hook-a", "hook-b"],
                "rules": ["rule-x"],
                "config": {
                    "control_system.type": "mock",
                    "archiver.type": "mock",
                },
                "overlay": {
                    "data/channels.json": "data/channels.json",
                },
                "mcp_servers": {
                    "server_one": {
                        "command": "python",
                        "args": ["-m", "server_one"],
                        "permissions": {"allow": ["tool_a", "tool_b"]},
                    },
                },
                "env": {
                    "required": ["API_KEY"],
                    "defaults": {"LOG_LEVEL": "info"},
                },
                "dependencies": ["fastmcp>=2.0", "pyepics>=3.5"],
            },
        )

    def test_basic_extends(self, tmp_path: Path):
        """Child inherits base fields and overrides name."""
        self._make_base(tmp_path)
        child_path = _write_yaml(
            tmp_path / "child.yml",
            {"extends": "base.yml", "name": "Child Profile"},
        )

        profile = load_profile(child_path)
        assert profile.name == "Child Profile"
        assert profile.hooks == ["hook-a", "hook-b"]
        assert profile.rules == ["rule-x"]
        assert profile.provider == "cborg"

    def test_scalar_override(self, tmp_path: Path):
        """Child overrides scalar fields; unmentioned scalars inherited."""
        self._make_base(tmp_path)
        child_path = _write_yaml(
            tmp_path / "child.yml",
            {"extends": "base.yml", "name": "Child", "model": "haiku"},
        )

        profile = load_profile(child_path)
        assert profile.model == "haiku"
        assert profile.provider == "cborg"  # inherited

    def test_dict_deep_merge(self, tmp_path: Path):
        """Config dicts are deep-merged; child keys override, base keys preserved."""
        self._make_base(tmp_path)
        child_path = _write_yaml(
            tmp_path / "child.yml",
            {
                "extends": "base.yml",
                "name": "Child",
                "config": {
                    "archiver.type": "epics_archiver",  # override
                    "system.timezone": "UTC",  # new key
                },
            },
        )

        profile = load_profile(child_path)
        assert profile.config["control_system.type"] == "mock"  # inherited
        assert profile.config["archiver.type"] == "epics_archiver"  # overridden
        assert profile.config["system.timezone"] == "UTC"  # new

    def test_list_union_dedup(self, tmp_path: Path):
        """String lists are unioned with dedup, base order preserved."""
        self._make_base(tmp_path)
        child_path = _write_yaml(
            tmp_path / "child.yml",
            {
                "extends": "base.yml",
                "name": "Child",
                "hooks": ["hook-b", "hook-c"],  # hook-b is a dup
            },
        )

        profile = load_profile(child_path)
        assert profile.hooks == ["hook-a", "hook-b", "hook-c"]

    def test_mcp_server_deep_merge(self, tmp_path: Path):
        """MCP servers are deep-merged: child adds url, inherits permissions."""
        # Base with permissions-only server (can't be built standalone)
        data_dir = tmp_path / "data"
        data_dir.mkdir(exist_ok=True)
        (data_dir / "channels.json").write_text("{}")

        _write_yaml(
            tmp_path / "base.yml",
            {
                "name": "Base",
                "data_bundle": "control_assistant",
                "config": {"control_system.type": "mock"},
                "mcp_servers": {
                    "matlab": {
                        "permissions": {"allow": ["mml_search", "mml_get"]},
                    },
                },
            },
        )
        child_path = _write_yaml(
            tmp_path / "child.yml",
            {
                "extends": "base.yml",
                "name": "Child",
                "mcp_servers": {
                    "matlab": {"url": "http://localhost:8001/sse"},
                },
            },
        )

        profile = load_profile(child_path)
        matlab = profile.mcp_servers["matlab"]
        assert matlab.url == "http://localhost:8001/sse"
        assert "mml_search" in matlab.permissions["allow"]
        assert "mml_get" in matlab.permissions["allow"]

    def test_overlay_merge(self, tmp_path: Path):
        """Child adds new overlay entries to base set."""
        self._make_base(tmp_path)
        # Create extra overlay source
        rules_dir = tmp_path / "overlays" / "rules"
        rules_dir.mkdir(parents=True)
        (rules_dir / "safety.md").write_text("# Safety")

        child_path = _write_yaml(
            tmp_path / "child.yml",
            {
                "extends": "base.yml",
                "name": "Child",
                "overlay": {
                    "overlays/rules/safety.md": ".claude/rules/safety.md",
                },
            },
        )

        profile = load_profile(child_path)
        assert "data/channels.json" in profile.overlay  # inherited
        assert "overlays/rules/safety.md" in profile.overlay  # added

    def test_env_merge(self, tmp_path: Path):
        """Env is deep-merged: file overridden, required unioned, defaults merged."""
        self._make_base(tmp_path)
        # Create the env file that the child references
        (tmp_path / ".env.local").write_text("API_KEY=test\n")

        child_path = _write_yaml(
            tmp_path / "child.yml",
            {
                "extends": "base.yml",
                "name": "Child",
                "env": {
                    "file": ".env.local",
                    "required": ["API_KEY", "EXTRA_VAR"],
                    "defaults": {"DEBUG": "true"},
                },
            },
        )

        profile = load_profile(child_path)
        assert profile.env.file == ".env.local"  # overridden
        assert "API_KEY" in profile.env.required  # from both (deduped)
        assert "EXTRA_VAR" in profile.env.required  # from child
        assert profile.env.defaults["LOG_LEVEL"] == "info"  # inherited
        assert profile.env.defaults["DEBUG"] == "true"  # from child

    def test_circular_extends(self, tmp_path: Path):
        """Circular extends chain is detected."""
        _write_yaml(tmp_path / "a.yml", {"extends": "b.yml", "name": "A"})
        _write_yaml(tmp_path / "b.yml", {"extends": "a.yml", "name": "B"})

        with pytest.raises(BuildProfileError, match="Circular extends"):
            load_profile(tmp_path / "a.yml")

    def test_missing_base_file(self, tmp_path: Path):
        """Extends referencing a nonexistent file raises a clear error."""
        child_path = _write_yaml(
            tmp_path / "child.yml",
            {"extends": "nonexistent.yml", "name": "Child"},
        )

        with pytest.raises(BuildProfileError, match="Extended profile not found"):
            load_profile(child_path)

    def test_multi_level_extends(self, tmp_path: Path):
        """A extends B extends C resolves correctly."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(exist_ok=True)
        (data_dir / "channels.json").write_text("{}")

        _write_yaml(
            tmp_path / "grandparent.yml",
            {
                "name": "Grandparent",
                "data_bundle": "control_assistant",
                "provider": "cborg",
                "model": "opus",
                "hooks": ["hook-a"],
                "config": {"control_system.type": "mock"},
                "mcp_servers": {
                    "srv": {"command": "python", "args": ["-m", "srv"]},
                },
            },
        )
        _write_yaml(
            tmp_path / "parent.yml",
            {
                "extends": "grandparent.yml",
                "name": "Parent",
                "hooks": ["hook-b"],
                "rules": ["rule-x"],
            },
        )
        child_path = _write_yaml(
            tmp_path / "child.yml",
            {
                "extends": "parent.yml",
                "name": "Child",
                "hooks": ["hook-c"],
            },
        )

        profile = load_profile(child_path)
        assert profile.name == "Child"
        assert profile.hooks == ["hook-a", "hook-b", "hook-c"]
        assert profile.rules == ["rule-x"]  # inherited from parent
        assert profile.provider == "cborg"  # inherited from grandparent

    def test_no_extends_unchanged(self, minimal_profile_yaml: Path):
        """Profiles without extends work exactly as before."""
        profile = load_profile(minimal_profile_yaml)
        assert profile.name == "Test Profile"
        assert profile.model == "haiku"

    def test_lifecycle_concatenation(self, tmp_path: Path):
        """Lifecycle step lists are concatenated (base first, child appended)."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(exist_ok=True)
        (data_dir / "channels.json").write_text("{}")

        _write_yaml(
            tmp_path / "base.yml",
            {
                "name": "Base",
                "data_bundle": "control_assistant",
                "config": {"control_system.type": "mock"},
                "mcp_servers": {
                    "srv": {"command": "python", "args": ["-m", "srv"]},
                },
                "lifecycle": {
                    "post_build": [{"name": "step-base", "run": "echo base"}],
                },
            },
        )
        child_path = _write_yaml(
            tmp_path / "child.yml",
            {
                "extends": "base.yml",
                "name": "Child",
                "lifecycle": {
                    "post_build": [{"name": "step-child", "run": "echo child"}],
                },
            },
        )

        profile = load_profile(child_path)
        names = [s.name for s in profile.lifecycle.post_build]
        assert names == ["step-base", "step-child"]

    def test_extends_key_stripped(self, tmp_path: Path):
        """The extends key does not leak into the parsed BuildProfile."""
        self._make_base(tmp_path)
        child_path = _write_yaml(
            tmp_path / "child.yml",
            {"extends": "base.yml", "name": "Child"},
        )

        profile = load_profile(child_path)
        assert not hasattr(profile, "extends")


# ---------------------------------------------------------------------------
# Web Panels Rendering (profile web_panels: drives config.yml)
# ---------------------------------------------------------------------------


def _build_for_web_panels(tmp_path: Path, web_panels: list[str] | None, overrides: dict | None = None) -> Path:
    """Build a minimal control_assistant project, optionally with web_panels
    and config overrides, and return the rendered config.yml path.

    Mirrors build_cmd.py steps 1b, 6, and 8 without running the full CLI.
    """
    from osprey.cli.build_cmd import _apply_config_overrides
    from osprey.cli.templates.artifact_library import validate_artifacts
    from osprey.cli.templates.manager import TemplateManager

    # Overlay source required by the control_assistant bundle
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "channels.json").write_text("{}")

    profile_data: dict = {
        "name": "Panels Test",
        "data_bundle": "control_assistant",
        "provider": "cborg",
        "model": "haiku",
        "overlay": {"data/channels.json": "data/channels.json"},
    }
    if web_panels is not None:
        profile_data["web_panels"] = web_panels
    if overrides:
        profile_data["config"] = overrides

    profile_path = tmp_path / "profile.yml"
    profile_path.write_text(yaml.dump(profile_data, default_flow_style=False))
    build_profile = load_profile(profile_path)

    artifacts: dict[str, list[str]] = {}
    for artifact_type in ("hooks", "rules", "skills", "agents", "output_styles"):
        names = getattr(build_profile, artifact_type, [])
        if names:
            artifacts[artifact_type] = list(names)
    if artifacts:
        validate_artifacts(artifacts)
    if build_profile.web_panels:
        artifacts["web_panels"] = list(build_profile.web_panels)

    manager = TemplateManager()
    project_dir = manager.create_project(
        project_name="panels-test",
        output_dir=tmp_path / "out",
        data_bundle=build_profile.data_bundle,
        registry_style="extend",
        context={},
        force=False,
        artifacts=artifacts or None,
    )
    if build_profile.config:
        _apply_config_overrides(project_dir, build_profile.config)

    return project_dir / "config.yml"


class TestWebPanelsRendering:
    """Profile web_panels: list drives which builtin panels are enabled in config.yml."""

    def test_builtin_panels_rendered_from_web_panels(self, tmp_path: Path):
        """Only builtin panels listed in web_panels appear with enabled: true."""
        config_path = _build_for_web_panels(tmp_path, web_panels=["ariel", "lattice"])
        config = yaml.safe_load(config_path.read_text())

        panels = config["web"]["panels"]
        assert panels.get("ariel", {}).get("enabled") is True
        assert panels.get("lattice", {}).get("enabled") is True
        assert "tuning" not in panels
        assert "channel-finder" not in panels

    def test_custom_panels_filtered_from_template_but_configurable(self, tmp_path: Path):
        """Non-builtin entries in web_panels are skipped by the template, but
        their dotted-override config still lands in config.yml."""
        config_path = _build_for_web_panels(
            tmp_path,
            web_panels=["ariel", "beam-viewer"],
            overrides={
                "web.panels.beam-viewer.label": "BEAM",
                "web.panels.beam-viewer.url": "http://localhost:8007",
            },
        )
        config = yaml.safe_load(config_path.read_text())

        panels = config["web"]["panels"]
        assert panels["ariel"]["enabled"] is True
        # beam-viewer is custom (not in _BUILTIN_PANELS) — no enabled: true from template,
        # but label/url from dotted overrides must land.
        assert panels["beam-viewer"]["label"] == "BEAM"
        assert panels["beam-viewer"]["url"] == "http://localhost:8007"
        assert "enabled" not in panels["beam-viewer"]

    def test_empty_web_panels_renders_empty_mapping(self, tmp_path: Path):
        """When web_panels is absent, template emits `panels: {}` and no builtins enable.
        Empty mapping (not None) is required so dotted-override merge stays safe."""
        config_path = _build_for_web_panels(tmp_path, web_panels=None)
        config = yaml.safe_load(config_path.read_text())

        panels = config["web"]["panels"]
        assert panels == {} or panels is None  # ruamel/pyyaml may parse either way
        # Critical: `web.panels` key exists and is not missing from the tree.
        assert "panels" in config["web"]

    def test_lattice_enabled_from_template_merges_with_dotted_label(self, tmp_path: Path):
        """lattice (builtin) gets enabled: true from the template, then the dotted
        override web.panels.lattice.label merges into the same dict."""
        config_path = _build_for_web_panels(
            tmp_path,
            web_panels=["lattice"],
            overrides={"web.panels.lattice.label": "LATTICE"},
        )
        config = yaml.safe_load(config_path.read_text())

        lattice = config["web"]["panels"]["lattice"]
        assert lattice["enabled"] is True
        assert lattice["label"] == "LATTICE"
