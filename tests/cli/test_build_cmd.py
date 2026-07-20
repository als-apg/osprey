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
                    "mcp_servers": {"bad": {"command": "npx", "url": "http://host:8001/sse"}},
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

    def test_default_panel_typo_rejected(self, tmp_path: Path):
        """A `default_panel` value that doesn't match any known panel is rejected.

        Without this check the frontend silently falls back to the framework
        default — a typo never surfaces. Validation catches it at build time.
        """
        profile = BuildProfile(name="Test", default_panel="areil")
        with pytest.raises(BuildProfileError, match="Unknown default_panel 'areil'"):
            profile.validate(tmp_path)

    def test_default_panel_builtin_accepted(self, tmp_path: Path):
        """A built-in panel id is accepted as default_panel without needing
        a matching web_panels entry."""
        profile = BuildProfile(name="Test", default_panel="ariel")
        profile.validate(tmp_path)  # must not raise

    def test_default_panel_custom_via_config_accepted(self, tmp_path: Path):
        """A custom panel backed by a `web.panels.<id>.url` config override
        is accepted as default_panel."""
        profile = BuildProfile(
            name="Test",
            default_panel="grafana",
            config={"web.panels.grafana.url": "http://localhost:3000"},
        )
        profile.validate(tmp_path)  # must not raise


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

    def test_load_profile_mcp_server_port_derives_url(self, tmp_path: Path):
        """A bare `port:` should yield url=http://localhost:<port>/mcp."""
        p = tmp_path / "profile.yml"
        p.write_text(
            yaml.dump(
                {
                    "name": "PortTest",
                    "mcp_servers": {
                        "matlab": {
                            "port": 8008,
                            "permissions": {"allow": ["mml_search"]},
                        }
                    },
                }
            )
        )
        profile = load_profile(p)
        server = profile.mcp_servers["matlab"]
        assert server.port == 8008
        assert server.url == "http://localhost:8008/mcp"
        assert server.command == ""

    def test_load_profile_mcp_server_port_with_explicit_url(self, tmp_path: Path):
        """An explicit `url:` plus `port:` should keep the explicit url verbatim."""
        p = tmp_path / "profile.yml"
        p.write_text(
            yaml.dump(
                {
                    "name": "ExternalClient",
                    "mcp_servers": {
                        "matlab": {
                            "port": 8008,
                            "url": "http://appsdev2:8008/mcp",
                        }
                    },
                }
            )
        )
        profile = load_profile(p)
        server = profile.mcp_servers["matlab"]
        assert server.port == 8008
        assert server.url == "http://appsdev2:8008/mcp"  # explicit wins, no derivation

    def test_load_profile_mcp_server_port_with_command_rejected(self, tmp_path: Path):
        """A stdio server (`command:`) cannot also declare a port."""
        p = tmp_path / "profile.yml"
        p.write_text(
            yaml.dump(
                {
                    "name": "Bad",
                    "mcp_servers": {
                        "confluence": {
                            "command": "uvx",
                            "port": 8001,
                        }
                    },
                }
            )
        )
        with pytest.raises(BuildProfileError, match="both 'command' and 'port'"):
            load_profile(p)

    @pytest.mark.parametrize(
        "bad_port",
        [0, -1, 65536, 80080, 100000],
        ids=["zero", "negative", "just-above-max", "typo-extra-digit", "way-too-big"],
    )
    def test_load_profile_mcp_server_port_out_of_range_rejected(
        self,
        tmp_path: Path,
        bad_port: int,
    ):
        """A `port:` outside 1..65535 must raise BuildProfileError at parse
        time, so a typo like `port: 80080` cannot silently flow into the
        derived url and the persisted network block."""
        p = tmp_path / "profile.yml"
        p.write_text(
            yaml.dump(
                {
                    "name": "OutOfRange",
                    "mcp_servers": {
                        "matlab": {
                            "port": bad_port,
                            "permissions": {"allow": ["mml_search"]},
                        }
                    },
                }
            )
        )
        with pytest.raises(BuildProfileError, match="port"):
            load_profile(p)

    def test_load_profile_mcp_server_port_non_integer_rejected(self, tmp_path: Path):
        """A string `port:` (YAML quoting accident) must be rejected at parse time."""
        p = tmp_path / "profile.yml"
        p.write_text(
            yaml.dump(
                {
                    "name": "Stringy",
                    "mcp_servers": {
                        "matlab": {
                            "port": "8008",
                            "permissions": {"allow": ["mml_search"]},
                        }
                    },
                }
            )
        )
        with pytest.raises(BuildProfileError, match="port"):
            load_profile(p)

    def test_persist_mcp_servers_port_emits_network_block(self, tmp_path: Path):
        """_persist_mcp_servers emits transport=http + network block when port is set."""
        from osprey.cli.build_cmd import _persist_mcp_servers

        project_path = tmp_path / "project"
        project_path.mkdir()
        (project_path / "config.yml").write_text("facility_name: test\n")

        servers = {
            "matlab": McpServerDef(
                url="http://localhost:8008/mcp",
                port=8008,
                permissions={"allow": ["mml_search"], "ask": []},
            ),
        }
        _persist_mcp_servers(project_path, servers)

        config = yaml.safe_load((project_path / "config.yml").read_text())
        entry = config["claude_code"]["servers"]["matlab"]
        assert entry["transport"] == "http"
        assert entry["url"] == "http://localhost:8008/mcp"
        assert entry["network"]["port"] == 8008
        assert entry["network"]["host_url"] == "http://localhost:8008/mcp"
        assert entry["network"]["docker_url"] == "http://matlab:8008/mcp"

    def test_persist_mcp_servers_stdio_emits_transport_stdio(self, tmp_path: Path):
        """Stdio servers get transport=stdio and no network block."""
        from osprey.cli.build_cmd import _persist_mcp_servers

        project_path = tmp_path / "project"
        project_path.mkdir()
        (project_path / "config.yml").write_text("facility_name: test\n")

        servers = {
            "confluence": McpServerDef(
                command="uvx",
                args=["--python=3.12", "mcp-atlassian"],
            ),
        }
        _persist_mcp_servers(project_path, servers)

        config = yaml.safe_load((project_path / "config.yml").read_text())
        entry = config["claude_code"]["servers"]["confluence"]
        assert entry["transport"] == "stdio"
        assert entry["command"] == "uvx"
        assert "network" not in entry

    def test_persist_mcp_servers_url_without_port_no_network_block(self, tmp_path: Path):
        """A url-only server (no port hint) gets transport=http but no network block."""
        from osprey.cli.build_cmd import _persist_mcp_servers

        project_path = tmp_path / "project"
        project_path.mkdir()
        (project_path / "config.yml").write_text("facility_name: test\n")

        servers = {
            "remote": McpServerDef(url="http://appsdev2:8008/mcp"),
        }
        _persist_mcp_servers(project_path, servers)

        config = yaml.safe_load((project_path / "config.yml").read_text())
        entry = config["claude_code"]["servers"]["remote"]
        assert entry["transport"] == "http"
        assert entry["url"] == "http://appsdev2:8008/mcp"
        assert "network" not in entry

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
# Osprey install spec resolution
# ---------------------------------------------------------------------------


class TestResolveOspreySpec:
    """Tests for _resolve_osprey_spec() — auto-detect install mode from metadata.

    Regression coverage for issue #216: building from a non-editable install
    (e.g., ``uv tool install osprey-framework``) used to error with
    "no pyproject.toml at <python lib>" because the resolver walked
    ``Path(__file__).parents[3]`` instead of consulting package metadata.
    """

    def _fake_dist(
        self,
        version: str = "2026.5.0",
        direct_url: dict | None = None,
    ):
        class _Dist:
            def __init__(self, ver, du):
                self.version = ver
                self._du = du

            def read_text(self, name):
                if name == "direct_url.json" and self._du is not None:
                    return json.dumps(self._du)
                return None

        return _Dist(version, direct_url)

    def test_editable_install_uses_source_path(self, monkeypatch):
        """Editable install (``pip install -e .``) → spec is the source dir."""
        from osprey.cli.build_cmd import _resolve_osprey_spec

        fake = self._fake_dist(
            direct_url={"url": "file:///abs/path/to/osprey", "dir_info": {"editable": True}},
        )
        monkeypatch.setattr("osprey.cli.build_cmd.distribution", lambda _name: fake)

        spec, label = _resolve_osprey_spec("local")
        assert spec == "/abs/path/to/osprey"
        assert "editable" in label

    def test_wheel_install_pins_to_version(self, monkeypatch):
        """uv tool / pip wheel install → pinned to ``osprey-framework==<version>``."""
        from osprey.cli.build_cmd import _resolve_osprey_spec

        fake = self._fake_dist(
            version="2026.5.0",
            direct_url={"url": "https://pypi/...", "archive_info": {"hash": "sha256=abc"}},
        )
        monkeypatch.setattr("osprey.cli.build_cmd.distribution", lambda _name: fake)

        spec, label = _resolve_osprey_spec("local")
        assert spec == "osprey-framework==2026.5.0"
        assert label == "osprey-framework==2026.5.0"

    def test_wheel_install_without_direct_url_pins_to_version(self, monkeypatch):
        """Older pip wheel install (no direct_url.json) → still pinned to version."""
        from osprey.cli.build_cmd import _resolve_osprey_spec

        fake = self._fake_dist(version="2026.5.0", direct_url=None)
        monkeypatch.setattr("osprey.cli.build_cmd.distribution", lambda _name: fake)

        spec, _label = _resolve_osprey_spec("local")
        assert spec == "osprey-framework==2026.5.0"

    def test_pip_keyword_uses_unpinned_pypi(self, monkeypatch):
        """Explicit ``osprey_install: pip`` → unpinned ``osprey-framework``."""
        from osprey.cli.build_cmd import _resolve_osprey_spec

        spec, _label = _resolve_osprey_spec("pip")
        assert spec == "osprey-framework"

    def test_pep508_spec_passthrough(self):
        """Explicit PEP 508 spec → passed through verbatim."""
        from osprey.cli.build_cmd import _resolve_osprey_spec

        spec, label = _resolve_osprey_spec("osprey-framework==2026.4.0")
        assert spec == "osprey-framework==2026.4.0"
        assert label == "osprey-framework==2026.4.0"

    def test_metadata_missing_falls_back_to_source_tree(self, monkeypatch, tmp_path):
        """If metadata is unavailable but a source tree exists at parents[3], use it."""
        from importlib.metadata import PackageNotFoundError

        from osprey.cli import build_cmd

        def _missing(_name):
            raise PackageNotFoundError

        monkeypatch.setattr(build_cmd, "distribution", _missing)
        # Real build_cmd.py is in a source checkout (pyproject.toml at parents[3]).
        spec, _label = build_cmd._resolve_osprey_spec("local")
        assert (Path(spec) / "pyproject.toml").exists()


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
        """Extends referencing a nonexistent file raises a clear, debuggable error.

        The diagnostic surfaces BOTH the preset list (so typos against bundled
        names are obvious) and the resolved filesystem path (so path-shaped
        values are debuggable).
        """
        child_path = _write_yaml(
            tmp_path / "child.yml",
            {"extends": "nonexistent.yml", "name": "Child"},
        )

        with pytest.raises(BuildProfileError) as exc:
            load_profile(child_path)
        msg = str(exc.value)
        assert "nonexistent.yml" in msg
        assert "no bundled preset" in msg.lower()
        assert "no file at" in msg.lower()

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


def _build_for_web_panels(
    tmp_path: Path,
    web_panels: list[str] | None,
    overrides: dict | None = None,
    default_panel: str | None = None,
) -> Path:
    """Build a minimal control_assistant project, optionally with web_panels,
    a default_panel pin, and config overrides, and return the rendered
    config.yml path.

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
    if default_panel is not None:
        profile_data["default_panel"] = default_panel
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

    template_context: dict = {}
    if build_profile.default_panel:
        template_context["default_panel"] = build_profile.default_panel

    manager = TemplateManager()
    project_dir = manager.create_project(
        project_name="panels-test",
        output_dir=tmp_path / "out",
        data_bundle=build_profile.data_bundle,
        context=template_context,
        force=False,
        artifacts=artifacts,
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

    def test_builtin_enabled_from_template_merges_with_dotted_label(self, tmp_path: Path):
        """A builtin panel gets enabled: true from the template, then the dotted
        override web.panels.<id>.label merges into the same dict."""
        config_path = _build_for_web_panels(
            tmp_path,
            web_panels=["lattice"],
            overrides={"web.panels.lattice.label": "LATTICE"},
        )
        config = yaml.safe_load(config_path.read_text())

        lattice = config["web"]["panels"]["lattice"]
        assert lattice["enabled"] is True
        assert lattice["label"] == "LATTICE"

    def test_default_panel_rendered_when_set(self, tmp_path: Path):
        """Profile default_panel: ariel ends up as web.default_panel in config.yml.

        The web terminal reads this key on startup; the frontend pins the
        cold-load tab to it.
        """
        config_path = _build_for_web_panels(tmp_path, web_panels=["ariel"], default_panel="ariel")
        config = yaml.safe_load(config_path.read_text())
        assert config["web"]["default_panel"] == "ariel"

    def test_default_panel_omitted_when_unset(self, tmp_path: Path):
        """Without a profile default_panel, the rendered config.yml omits
        the key entirely so the frontend falls back to DEFAULT_PANEL_FALLBACK."""
        config_path = _build_for_web_panels(tmp_path, web_panels=["ariel"])
        config = yaml.safe_load(config_path.read_text())
        assert "default_panel" not in config["web"]


# ---------------------------------------------------------------------------
# Tier Flattening (materialize_tier_artifacts)
# ---------------------------------------------------------------------------


def _preset_tier_source(tier: int, paradigm: str) -> Path:
    """Path to the bundled preset's tier-routed source DB."""
    import osprey

    osprey_root = Path(osprey.__file__).parent
    return (
        osprey_root
        / "templates"
        / "apps"
        / "control_assistant"
        / "data"
        / "channel_databases"
        / "tiers"
        / f"tier{tier}"
        / f"{paradigm}.json"
    )


def _write_tier_profile(profile_dir: Path, paradigm: str, tier: int | None = None) -> Path:
    """Write a minimal control_assistant profile pinned to a single paradigm
    and (optionally) a tier."""
    profile_data: dict = {
        "name": "Tier Test",
        "data_bundle": "control_assistant",
        "provider": "cborg",
        "model": "haiku",
        "channel_finder_mode": paradigm,
    }
    if tier is not None:
        profile_data["tier"] = tier
    path = profile_dir / "tier-profile.yml"
    path.write_text(yaml.dump(profile_data, default_flow_style=False))
    return path


_PARADIGMS_FOR_BUILD: tuple[str, ...] = ("in_context", "hierarchical", "middle_layer")


# Tier selection is restricted to {1, 3}, and tier 1 is in_context-only
# (it ships no hierarchical/middle_layer DB). Only these combos are buildable.
_VALID_TIER_PARADIGMS: tuple[tuple[int, str], ...] = (
    (1, "in_context"),
    (3, "in_context"),
    (3, "hierarchical"),
    (3, "middle_layer"),
)


@pytest.mark.parametrize("tier,paradigm", _VALID_TIER_PARADIGMS)
def test_build_tier_flatten(tmp_path: Path, tier: int, paradigm: str) -> None:
    """`osprey build --tier N` materializes the active paradigm's DB at the
    flat path and removes the ``tiers/`` subtree.

    - rendered config.yml emits ``data/channel_databases/<paradigm>.json``
      (no ``tiers/`` segment).
    - the file exists at that flat path and byte-equals the preset's
      ``tiers/tier{N}/<paradigm>.json`` source.
    - the other paradigms' flat files are NOT created.
    - the ``tiers/`` subdirectory has been removed.
    """
    from click.testing import CliRunner

    from osprey.cli.main import cli

    profile_path = _write_tier_profile(tmp_path, paradigm)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "build",
            "tier-proj",
            str(profile_path),
            "--output-dir",
            str(output_dir),
            "--tier",
            str(tier),
            "--skip-deps",
            "--skip-lifecycle",
        ],
    )
    assert result.exit_code == 0, (
        f"build failed (exit={result.exit_code})\n"
        f"--- output ---\n{result.output}\n"
        f"--- exception ---\n{result.exception}"
    )

    project_dir = output_dir / "tier-proj"
    config = yaml.safe_load((project_dir / "config.yml").read_text())
    pipelines = config["channel_finder"]["pipelines"]

    # (a) Rendered config points to the FLAT path — no tiers/ segment.
    assert pipelines[paradigm]["database"]["path"] == f"data/channel_databases/{paradigm}.json", (
        f"paradigm={paradigm} got {pipelines[paradigm]['database']['path']!r}"
    )

    # (b) The flat DB exists and byte-equals the preset tier source.
    flat_path = project_dir / "data" / "channel_databases" / f"{paradigm}.json"
    assert flat_path.exists(), f"flat DB missing: {flat_path}"

    src = _preset_tier_source(tier, paradigm)
    assert src.exists(), f"preset source missing: {src}"
    assert flat_path.read_bytes() == src.read_bytes(), (
        f"flat DB does not byte-equal preset tier{tier}/{paradigm}.json"
    )

    # (c) Other paradigms are NOT materialized to the flat root.
    for other in _PARADIGMS_FOR_BUILD:
        if other == paradigm:
            continue
        other_flat = project_dir / "data" / "channel_databases" / f"{other}.json"
        assert not other_flat.exists(), (
            f"{other}.json should not have been materialized for mode={paradigm!r}"
        )

    # (d) The tiers/ subtree has been pruned.
    assert not (project_dir / "data" / "channel_databases" / "tiers").exists(), (
        "tiers/ subtree was not pruned after materialization"
    )


# Re-tiering 1 → 3 only applies to in_context; tier 1 is in_context-only.
@pytest.mark.parametrize("paradigm", ["in_context"])
def test_build_force_retier(tmp_path: Path, paradigm: str) -> None:
    """Rebuilding with --force --tier 3 over a tier-1 project re-materializes
    the active paradigm's DB (byte-equals preset tier3 source)."""
    from click.testing import CliRunner

    from osprey.cli.main import cli

    profile_path = _write_tier_profile(tmp_path, paradigm)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    runner = CliRunner()

    # First build: tier 1
    result1 = runner.invoke(
        cli,
        [
            "build",
            "retier-proj",
            str(profile_path),
            "--output-dir",
            str(output_dir),
            "--tier",
            "1",
            "--skip-deps",
            "--skip-lifecycle",
        ],
    )
    assert result1.exit_code == 0, f"tier-1 build failed: {result1.output}\n{result1.exception}"

    # Second build: tier 3, --force overwrites the same path.
    result2 = runner.invoke(
        cli,
        [
            "build",
            "retier-proj",
            str(profile_path),
            "--output-dir",
            str(output_dir),
            "--tier",
            "3",
            "--force",
            "--skip-deps",
            "--skip-lifecycle",
        ],
    )
    assert result2.exit_code == 0, f"tier-3 rebuild failed: {result2.output}\n{result2.exception}"

    project_dir = output_dir / "retier-proj"
    flat_path = project_dir / "data" / "channel_databases" / f"{paradigm}.json"
    src = _preset_tier_source(3, paradigm)
    assert flat_path.read_bytes() == src.read_bytes(), (
        f"after --force --tier 3, {paradigm}.json does not byte-equal preset tier3 source"
    )


@pytest.mark.parametrize("paradigm", _PARADIGMS_FOR_BUILD)
def test_build_profile_only_tier(tmp_path: Path, paradigm: str) -> None:
    """When the profile sets ``tier: 3`` and no --tier is passed on the CLI,
    the profile value drives materialization."""
    from click.testing import CliRunner

    from osprey.cli.main import cli

    profile_path = _write_tier_profile(tmp_path, paradigm, tier=3)
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "build",
            "profile-tier-proj",
            str(profile_path),
            "--output-dir",
            str(output_dir),
            "--skip-deps",
            "--skip-lifecycle",
        ],
    )
    assert result.exit_code == 0, (
        f"profile-only-tier build failed: {result.output}\n{result.exception}"
    )

    project_dir = output_dir / "profile-tier-proj"
    flat_path = project_dir / "data" / "channel_databases" / f"{paradigm}.json"
    src = _preset_tier_source(3, paradigm)
    assert flat_path.read_bytes() == src.read_bytes(), (
        f"profile tier=3 not honored for {paradigm}.json"
    )


# ---------------------------------------------------------------------------
# Events-panel URL derivation (Task 1.4)
# ---------------------------------------------------------------------------


class TestEventsPanelUrlDerivation:
    """_inject_dispatch derives web.panels.events.url from dispatcher_port."""

    def _write_minimal_config(self, project_path: Path) -> None:
        """Write the smallest config.yml that _inject_dispatch needs."""
        from ruamel.yaml import YAML

        yaml = YAML()
        with open(project_path / "config.yml", "w") as fh:
            yaml.dump({"deployed_services": [], "services": {}}, fh)

    def _dispatch(self, dispatcher_port: int = 8020, **overrides: object):
        from osprey.cli.build_profile import DispatchConfig

        base: dict = {
            "triggers": "tutorial_triggers.yml",
            "worker_count": 1,
            "workspace_mode": "isolated",
            "max_concurrent_runs": 2,
            "max_queue_depth": 50,
            "dispatcher_port": dispatcher_port,
            "worker_port_base": 9190,
            "timeout_sec": 300,
            "facility_name": "generic-facility",
            "pv_strip_prefix": "",
        }
        base.update(overrides)
        return DispatchConfig(**base)  # type: ignore[arg-type]

    def _inject(self, dispatch, project_path: Path, profile_dir: Path) -> None:
        from osprey.cli.build_cmd import _inject_dispatch

        _inject_dispatch(dispatch, profile_dir=profile_dir, project_path=project_path)

    def _read_config(self, project_path: Path) -> dict:
        import yaml

        return yaml.safe_load((project_path / "config.yml").read_text()) or {}

    def test_events_panel_url_derived_from_dispatcher_port(self, tmp_path: Path) -> None:
        """events.url is a bare host and events.path carries the /dashboard route."""
        project_path = tmp_path / "project"
        project_path.mkdir()
        profile_dir = tmp_path / "profile"
        profile_dir.mkdir()
        self._write_minimal_config(project_path)

        self._inject(self._dispatch(dispatcher_port=8888), project_path, profile_dir)

        config = self._read_config(project_path)
        events = config["web"]["panels"]["events"]
        # Bare host in url + /dashboard in path matches the custom-panel proxy
        # convention (url.rstrip('/') + '/' + path); /dashboard must NOT be baked
        # into url or sub-routes would double-prefix.
        assert events["url"] == "http://localhost:8888"
        assert events["path"] == "/dashboard"

    def test_events_panel_url_reflects_custom_port(self, tmp_path: Path) -> None:
        """Different dispatcher_port values produce different derived bare-host URLs."""
        project_path = tmp_path / "project"
        project_path.mkdir()
        profile_dir = tmp_path / "profile"
        profile_dir.mkdir()
        self._write_minimal_config(project_path)

        self._inject(self._dispatch(dispatcher_port=9999), project_path, profile_dir)

        config = self._read_config(project_path)
        events = config["web"]["panels"]["events"]
        assert events["url"] == "http://localhost:9999"
        assert events["path"] == "/dashboard"

    def test_events_panel_path_pinned_by_profile_preserved(self, tmp_path: Path) -> None:
        """A facility-pinned web.panels.events.path is left untouched (setdefault)."""
        from ruamel.yaml import YAML

        project_path = tmp_path / "project"
        project_path.mkdir()
        profile_dir = tmp_path / "profile"
        profile_dir.mkdir()

        yaml = YAML()
        config_pre = {
            "deployed_services": [],
            "services": {},
            "web": {"panels": {"events": {"path": "/custom-route"}}},
        }
        with open(project_path / "config.yml", "w") as fh:
            yaml.dump(config_pre, fh)

        self._inject(self._dispatch(dispatcher_port=8020), project_path, profile_dir)

        config = self._read_config(project_path)
        events = config["web"]["panels"]["events"]
        # url is still derived (no explicit url was set), but the pinned path wins.
        assert events["url"] == "http://localhost:8020"
        assert events["path"] == "/custom-route"

    def test_explicit_events_url_not_overwritten(self, tmp_path: Path) -> None:
        """An explicit web.panels.events.url already in config.yml is not replaced,
        and the derivation does not force-inject a path alongside it."""
        from ruamel.yaml import YAML

        project_path = tmp_path / "project"
        project_path.mkdir()
        profile_dir = tmp_path / "profile"
        profile_dir.mkdir()

        # Pre-set an explicit URL via a profile config override (simulated here by
        # writing it directly into config.yml before _inject_dispatch runs).
        yaml = YAML()
        config_pre = {
            "deployed_services": [],
            "services": {},
            "web": {"panels": {"events": {"url": "http://custom-host:7777/dashboard"}}},
        }
        with open(project_path / "config.yml", "w") as fh:
            yaml.dump(config_pre, fh)

        self._inject(self._dispatch(dispatcher_port=8020), project_path, profile_dir)

        config = self._read_config(project_path)
        events = config["web"]["panels"]["events"]
        # Explicit URL must survive — derivation must not overwrite it.
        assert events["url"] == "http://custom-host:7777/dashboard"
        # Derivation is fully skipped when an explicit url exists; it must not
        # inject a path that would compose a double route against the explicit url.
        assert "path" not in events


def test_build_channel_finder_agent_requires_mode(tmp_path: Path) -> None:
    """A profile that selects the channel-finder agent but omits
    channel_finder_mode must fail with a clear BuildProfileError."""
    from click.testing import CliRunner

    from osprey.cli.main import cli

    profile_data = {
        "name": "no-mode",
        "data_bundle": "control_assistant",
        "provider": "cborg",
        "model": "haiku",
        "agents": ["channel-finder"],
        # NOTE: channel_finder_mode intentionally omitted.
    }
    profile_path = tmp_path / "no-mode.yml"
    profile_path.write_text(yaml.dump(profile_data, default_flow_style=False))
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "build",
            "no-mode-proj",
            str(profile_path),
            "--output-dir",
            str(output_dir),
            "--skip-deps",
            "--skip-lifecycle",
        ],
    )
    assert result.exit_code != 0, f"build should have failed; output:\n{result.output}"
    combined = result.output + (str(result.exception) if result.exception else "")
    assert "channel_finder_mode" in combined
    assert "required" in combined.lower()


# ---------------------------------------------------------------------------
# Approval-overlap guard (extends clones share bare-name approval policies)
# ---------------------------------------------------------------------------


def test_build_context_warns_when_remove_ask_overrides_gated_tool(tmp_path: Path, caplog) -> None:
    """A facility permissions.remove_ask (or allow) entry naming an enabled
    server's approval-gated tool — e.g. an extends clone's phoebus_drive —
    must emit a build-time warning: it auto-approves the tool at the
    permission layer, and approval policies apply to every instance of a
    template (no per-instance gating)."""
    import logging

    from osprey.cli.templates import claude_code
    from osprey.cli.templates.manager import TemplateManager

    manager = TemplateManager()
    project = manager.create_project(
        project_name="remove-ask-warn",
        output_dir=tmp_path,
        data_bundle="hello_world",
    )

    config = yaml.safe_load((project / "config.yml").read_text())
    cc = config.setdefault("claude_code", {})
    cc.setdefault("servers", {})["phoebus2"] = {"extends": "phoebus"}
    cc["permissions"] = {"remove_ask": ["mcp__phoebus2__phoebus_drive"]}

    with caplog.at_level(logging.WARNING):
        claude_code.build_claude_code_context(
            manager.template_root, manager.jinja_env, project, config
        )

    assert any(
        "mcp__phoebus2__phoebus_drive" in r.message and "approval-gated" in r.message
        for r in caplog.records
    ), f"expected approval-overlap warning; got: {[r.message for r in caplog.records]}"


def test_writes_disabled_hard_block_covers_extends_clones(tmp_path: Path) -> None:
    """With control_system.writes_enabled false, the kill-switch hard-block must
    cover extends clones of the controls/python templates, not just the literal
    template names: the clone's rewritten channel_write lands in the rendered
    settings.json permissions.deny, and the clone's execute is pulled out of
    permissions.ask exactly like mcp__python__execute."""
    from osprey.cli.templates.manager import TemplateManager
    from osprey.utils.config_writer import config_update_fields

    manager = TemplateManager()
    project = manager.create_project(
        project_name="writes-block-clone",
        output_dir=tmp_path,
        data_bundle="hello_world",
    )

    config_update_fields(
        project / "config.yml",
        {
            "control_system.writes_enabled": False,
            "claude_code.servers.controls2.extends": "controls",
            "claude_code.servers.python2.extends": "python",
        },
    )
    manager.regenerate_claude_code(project)

    settings = json.loads((project / ".claude" / "settings.json").read_text())
    deny = settings["permissions"]["deny"]
    ask = settings["permissions"]["ask"]
    # Template and clone both hard-blocked.
    assert "mcp__controls__channel_write" in deny
    assert "mcp__controls2__channel_write" in deny
    # Template and clone execute both pulled out of ask (remove_ask leg).
    assert "mcp__python__execute" not in ask
    assert "mcp__python2__execute" not in ask


# ---------------------------------------------------------------------------
# Service template bundling (_copy_service_templates)
# ---------------------------------------------------------------------------


class TestCopyServiceTemplates:
    """Tests for _copy_service_templates() — bundles package service templates.

    A service merely DECLARED under `services:` (an opt-in add-on left out of
    `deployed_services`) must still have its package template bundled, so it can
    be switched on later with a `deployed_services` edit + `osprey deploy up`
    without rebuilding.
    """

    def _write_config(self, project_path: Path, config: dict) -> None:
        from ruamel.yaml import YAML

        yaml_rt = YAML()
        with open(project_path / "config.yml", "w") as fh:
            yaml_rt.dump(config, fh)

    def test_declared_but_not_deployed_service_is_bundled(self, tmp_path: Path) -> None:
        """openobserve declared under `services:` but absent from
        deployed_services is still copied into the project's services/ tree,
        while a deployed service (postgresql) still bundles (no regression)."""
        from osprey.cli.build_cmd import _copy_service_templates

        project_path = tmp_path / "project"
        project_path.mkdir()
        self._write_config(
            project_path,
            {
                # postgresql is deployed; openobserve is only declared (opt-in).
                "deployed_services": ["postgresql"],
                "services": {
                    "postgresql": {"path": "./services/postgresql"},
                    "openobserve": {"path": "./services/openobserve"},
                },
            },
        )

        count = _copy_service_templates(project_path)

        # Deployed service still bundles (regression guard).
        assert (project_path / "services" / "postgresql" / "docker-compose.yml.j2").exists()
        # Declared-but-not-deployed add-on is bundled too (the new behavior).
        assert (project_path / "services" / "openobserve" / "docker-compose.yml.j2").exists()
        assert count == 2

    def test_declared_only_service_bundles_without_deployed_services(self, tmp_path: Path) -> None:
        """With an empty deployed_services, a declared service with a package
        template is still bundled — the copy path no longer early-returns when
        deployed_services is empty."""
        from osprey.cli.build_cmd import _copy_service_templates

        project_path = tmp_path / "project"
        project_path.mkdir()
        self._write_config(
            project_path,
            {
                "deployed_services": [],
                "services": {"openobserve": {"path": "./services/openobserve"}},
            },
        )

        count = _copy_service_templates(project_path)

        assert (project_path / "services" / "openobserve" / "docker-compose.yml.j2").exists()
        assert count == 1

    def test_declared_service_without_package_template_skipped_silently(
        self, tmp_path: Path, caplog
    ) -> None:
        """A declared-only service with no package template is skipped without
        a warning (it may be facility-injected elsewhere)."""
        import logging

        from osprey.cli.build_cmd import _copy_service_templates

        project_path = tmp_path / "project"
        project_path.mkdir()
        self._write_config(
            project_path,
            {
                "deployed_services": [],
                "services": {"typesense": {"path": "./services/typesense"}},
            },
        )

        with caplog.at_level(logging.WARNING, logger="osprey.cli.build_cmd"):
            count = _copy_service_templates(project_path)

        assert count == 0
        assert not (project_path / "services" / "typesense").exists()
        assert not any("typesense" in r.getMessage() for r in caplog.records), (
            "declared-only service without a template must not warn"
        )

    def test_deployed_service_without_package_template_warns(self, tmp_path: Path, caplog) -> None:
        """A *deployed* service missing its package template still warns —
        that would break `osprey deploy up`, so the operator must be told."""
        import logging

        from osprey.cli.build_cmd import _copy_service_templates

        project_path = tmp_path / "project"
        project_path.mkdir()
        self._write_config(
            project_path,
            {
                "deployed_services": ["typesense"],
                "services": {"typesense": {"path": "./services/typesense"}},
            },
        )

        with caplog.at_level(logging.WARNING, logger="osprey.cli.build_cmd"):
            count = _copy_service_templates(project_path)

        assert count == 0
        assert any("typesense" in r.getMessage() for r in caplog.records), (
            "a deployed service without a template must warn"
        )
