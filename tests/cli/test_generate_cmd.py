"""Tests for generate CLI command."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from osprey.cli.generate_cmd import (
    _determine_capabilities_path,
    generate,
    is_project_initialized,
)


@pytest.fixture
def cli_runner():
    """Provide a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_project(tmp_path, monkeypatch):
    """Set up a mock Osprey project directory."""
    # Create config.yml
    config_file = tmp_path / "config.yml"
    config_file.write_text("""
project_name: test_project
llm:
  default_provider: anthropic
""")

    # Create capabilities directory
    capabilities_dir = tmp_path / "capabilities"
    capabilities_dir.mkdir()

    # Create registry.py
    registry_file = tmp_path / "registry.py"
    registry_file.write_text("""
from osprey.registry import CapabilityRegistry

def get_registry():
    return CapabilityRegistry()
""")

    # Change to project directory
    monkeypatch.chdir(tmp_path)

    return tmp_path


# =============================================================================
# Test Project Detection
# =============================================================================


class TestProjectDetection:
    """Test project initialization detection."""

    def test_is_project_initialized_with_config(self, mock_project):
        """Test project detection when config.yml exists."""
        assert is_project_initialized() is True

    def test_is_project_initialized_without_config(self, tmp_path, monkeypatch):
        """Test project detection when config.yml does not exist."""
        monkeypatch.chdir(tmp_path)
        assert is_project_initialized() is False


class TestCapabilitiesPathDetermination:
    """Test capabilities path resolution."""

    def test_determine_path_with_registry(self, mock_project):
        """Test path determination when registry is found."""
        with patch("osprey.generators.registry_updater.find_registry_file") as mock_find:
            mock_find.return_value = mock_project / "src" / "myproject" / "registry.py"

            result = _determine_capabilities_path("test_capability")

            assert "capabilities" in str(result)
            assert "test_capability.py" in str(result)

    def test_determine_path_without_registry(self, mock_project):
        """Test path determination fallback when registry not found."""
        with patch("osprey.generators.registry_updater.find_registry_file") as mock_find:
            mock_find.return_value = None

            result = _determine_capabilities_path("test_capability")

            assert result == Path("capabilities/test_capability.py")

    def test_determine_path_with_exception(self, mock_project):
        """Test path determination when exception occurs."""
        with patch(
            "osprey.generators.registry_updater.find_registry_file", side_effect=Exception("Error")
        ):
            result = _determine_capabilities_path("test_capability")

            # Should fall back to simple path
            assert result == Path("capabilities/test_capability.py")


# =============================================================================
# Test Main Generate Command
# =============================================================================


class TestGenerateCommand:
    """Test main generate command group."""

    def test_generate_command_exists(self):
        """Verify generate command can be imported and is callable."""
        assert generate is not None
        assert callable(generate)

    def test_generate_help(self, cli_runner):
        """Test generate command help output."""
        result = cli_runner.invoke(generate, ["--help"])

        assert result.exit_code == 0
        assert "generate" in result.output.lower() or "Generate" in result.output
        assert "capability" in result.output
        assert "mcp-server" in result.output
        assert "claude-config" in result.output

    def test_generate_capability_subcommand_exists(self, cli_runner):
        """Test that capability subcommand exists."""
        result = cli_runner.invoke(generate, ["capability", "--help"])

        assert result.exit_code == 0
        assert "--from-mcp" in result.output
        assert "--from-prompt" in result.output

    def test_generate_mcp_server_subcommand_exists(self, cli_runner):
        """Test that mcp-server subcommand exists."""
        result = cli_runner.invoke(generate, ["mcp-server", "--help"])

        assert result.exit_code == 0
        assert "--name" in result.output
        assert "--output" in result.output

    def test_generate_claude_config_subcommand_exists(self, cli_runner):
        """Test that claude-config subcommand exists."""
        result = cli_runner.invoke(generate, ["claude-config", "--help"])

        assert result.exit_code == 0
        assert "--output" in result.output
        assert "--force" in result.output


# =============================================================================
# Test Capability Command - Validation
# =============================================================================


class TestCapabilityCommandValidation:
    """Test capability command input validation."""

    def test_capability_requires_source_option(self, cli_runner, mock_project):
        """Test that capability requires either --from-mcp or --from-prompt."""
        result = cli_runner.invoke(generate, ["capability"])

        assert result.exit_code != 0
        assert "Must specify exactly one of --from-mcp or --from-prompt" in result.output

    def test_capability_rejects_both_sources(self, cli_runner, mock_project):
        """Test that capability rejects both --from-mcp and --from-prompt."""
        result = cli_runner.invoke(
            generate, ["capability", "--from-mcp", "http://localhost:3001", "--from-prompt", "test"]
        )

        assert result.exit_code != 0
        assert "Must specify exactly one of --from-mcp or --from-prompt" in result.output

    def test_capability_from_mcp_requires_name(self, cli_runner, mock_project):
        """Test that --from-mcp requires --name option."""
        result = cli_runner.invoke(generate, ["capability", "--from-mcp", "http://localhost:3001"])

        assert result.exit_code != 0
        assert "--name is required when using --from-mcp" in result.output

    def test_capability_requires_project(self, cli_runner, tmp_path, monkeypatch):
        """Test that capability requires being in an Osprey project."""
        monkeypatch.chdir(tmp_path)  # No config.yml here

        result = cli_runner.invoke(
            generate, ["capability", "--from-mcp", "http://localhost:3001", "-n", "test"]
        )

        assert result.exit_code != 0
        assert "Not in an Osprey project directory" in result.output

    def test_capability_help_output(self, cli_runner):
        """Test capability command help text."""
        result = cli_runner.invoke(generate, ["capability", "--help"])

        assert result.exit_code == 0
        assert "--from-mcp" in result.output
        assert "--from-prompt" in result.output
        assert "--name" in result.output
        assert "--server-name" in result.output
        assert "--output" in result.output
        assert "--provider" in result.output
        assert "--model" in result.output
        assert "--quiet" in result.output


# =============================================================================
# Test Capability Command - MCP Mode
# =============================================================================


class TestCapabilityMCPMode:
    """Test capability generation from MCP server."""

    @patch("osprey.cli.generate_cmd.initialize_registry")
    @patch("osprey.cli.generate_cmd.get_mcp_generator")
    @patch("osprey.cli.generate_cmd.asyncio.run")
    def test_capability_from_mcp_basic(
        self, mock_asyncio, mock_get_gen, mock_init_reg, cli_runner, mock_project
    ):
        """Test basic MCP capability generation."""
        # Setup mocks
        mock_generator = Mock()
        mock_generator_class = Mock(return_value=mock_generator)
        mock_get_gen.return_value = mock_generator_class
        mock_init_reg.return_value = True

        cli_runner.invoke(
            generate,
            ["capability", "--from-mcp", "http://localhost:3001", "-n", "test_mcp", "--quiet"],
        )

        # Should initialize registry
        assert mock_init_reg.called

        # Should create generator
        assert mock_get_gen.called
        mock_generator_class.assert_called_once()

        # Should run async generation
        assert mock_asyncio.called

        # Verify call arguments
        call_kwargs = mock_generator_class.call_args.kwargs
        assert call_kwargs["capability_name"] == "test_mcp"
        assert call_kwargs["verbose"] is False  # --quiet flag

    @patch("osprey.cli.generate_cmd.initialize_registry")
    @patch("osprey.cli.generate_cmd.get_mcp_generator")
    @patch("osprey.cli.generate_cmd.asyncio.run")
    def test_capability_from_mcp_simulated(
        self, mock_asyncio, mock_get_gen, mock_init_reg, cli_runner, mock_project
    ):
        """Test MCP capability generation in simulated mode."""
        mock_generator = Mock()
        mock_generator_class = Mock(return_value=mock_generator)
        mock_get_gen.return_value = mock_generator_class
        mock_init_reg.return_value = True

        cli_runner.invoke(
            generate, ["capability", "--from-mcp", "simulated", "-n", "weather_mcp", "--quiet"]
        )

        # Should run successfully
        assert mock_asyncio.called

    @patch("osprey.cli.generate_cmd.initialize_registry")
    @patch("osprey.cli.generate_cmd.get_mcp_generator")
    @patch("osprey.cli.generate_cmd.asyncio.run")
    def test_capability_from_mcp_with_custom_output(
        self, mock_asyncio, mock_get_gen, mock_init_reg, cli_runner, mock_project
    ):
        """Test MCP capability generation with custom output path."""
        mock_generator = Mock()
        mock_generator_class = Mock(return_value=mock_generator)
        mock_get_gen.return_value = mock_generator_class
        mock_init_reg.return_value = True

        cli_runner.invoke(
            generate,
            [
                "capability",
                "--from-mcp",
                "http://localhost:3001",
                "-n",
                "test_mcp",
                "--output",
                "custom/path/capability.py",
                "--quiet",
            ],
        )

        # Should run successfully
        assert mock_asyncio.called

    @patch("osprey.cli.generate_cmd.initialize_registry")
    @patch("osprey.cli.generate_cmd.get_mcp_generator")
    @patch("osprey.cli.generate_cmd.asyncio.run")
    def test_capability_from_mcp_with_model_override(
        self, mock_asyncio, mock_get_gen, mock_init_reg, cli_runner, mock_project
    ):
        """Test MCP capability generation with model/provider override."""
        mock_generator = Mock()
        mock_generator_class = Mock(return_value=mock_generator)
        mock_get_gen.return_value = mock_generator_class
        mock_init_reg.return_value = True

        cli_runner.invoke(
            generate,
            [
                "capability",
                "--from-mcp",
                "http://localhost:3001",
                "-n",
                "test_mcp",
                "--provider",
                "anthropic",
                "--model",
                "claude-sonnet-4-20250514",
                "--quiet",
            ],
        )

        # Verify provider/model passed to generator
        call_kwargs = mock_generator_class.call_args.kwargs
        assert call_kwargs["provider"] == "anthropic"
        assert call_kwargs["model_id"] == "claude-sonnet-4-20250514"

    @patch("osprey.cli.generate_cmd.initialize_registry")
    @patch("osprey.cli.generate_cmd.get_mcp_generator")
    @patch("osprey.cli.generate_cmd.asyncio.run")
    def test_capability_from_mcp_keyboard_interrupt(
        self, mock_asyncio, mock_get_gen, mock_init_reg, cli_runner, mock_project
    ):
        """Test MCP capability generation handles KeyboardInterrupt."""
        mock_generator = Mock()
        mock_generator_class = Mock(return_value=mock_generator)
        mock_get_gen.return_value = mock_generator_class
        mock_init_reg.return_value = True
        mock_asyncio.side_effect = KeyboardInterrupt()

        result = cli_runner.invoke(
            generate,
            ["capability", "--from-mcp", "http://localhost:3001", "-n", "test_mcp", "--quiet"],
        )

        assert result.exit_code != 0
        assert "cancelled" in result.output.lower()

    @patch("osprey.cli.generate_cmd.initialize_registry")
    @patch("osprey.cli.generate_cmd.get_mcp_generator")
    @patch("osprey.cli.generate_cmd.asyncio.run")
    def test_capability_from_mcp_runtime_error(
        self, mock_asyncio, mock_get_gen, mock_init_reg, cli_runner, mock_project
    ):
        """Test MCP capability generation handles RuntimeError."""
        mock_generator = Mock()
        mock_generator_class = Mock(return_value=mock_generator)
        mock_get_gen.return_value = mock_generator_class
        mock_init_reg.return_value = True
        mock_asyncio.side_effect = RuntimeError("Test error")

        result = cli_runner.invoke(
            generate,
            ["capability", "--from-mcp", "http://localhost:3001", "-n", "test_mcp", "--quiet"],
        )

        assert result.exit_code != 0
        assert "Test error" in result.output


# =============================================================================
# Test Capability Command - Prompt Mode
# =============================================================================


class TestCapabilityPromptMode:
    """Test capability generation from natural language prompt."""

    @patch("osprey.cli.generate_cmd.initialize_registry")
    @patch("osprey.cli.generate_cmd.get_prompt_generator")
    @patch("osprey.cli.generate_cmd.asyncio.run")
    def test_capability_from_prompt_basic(
        self, mock_asyncio, mock_get_gen, mock_init_reg, cli_runner, mock_project
    ):
        """Test basic prompt-based capability generation."""
        mock_generator = Mock()
        mock_generator_class = Mock(return_value=mock_generator)
        mock_get_gen.return_value = mock_generator_class
        mock_init_reg.return_value = True

        cli_runner.invoke(
            generate, ["capability", "--from-prompt", "Fetch weather data", "--quiet"]
        )

        # Should initialize registry
        assert mock_init_reg.called

        # Should create generator
        assert mock_get_gen.called
        mock_generator_class.assert_called_once()

        # Verify call arguments
        call_kwargs = mock_generator_class.call_args.kwargs
        assert call_kwargs["prompt"] == "Fetch weather data"
        assert call_kwargs["capability_name"] is None  # Will be suggested by LLM
        assert call_kwargs["verbose"] is False

    @patch("osprey.cli.generate_cmd.initialize_registry")
    @patch("osprey.cli.generate_cmd.get_prompt_generator")
    @patch("osprey.cli.generate_cmd.asyncio.run")
    def test_capability_from_prompt_with_name(
        self, mock_asyncio, mock_get_gen, mock_init_reg, cli_runner, mock_project
    ):
        """Test prompt-based generation with explicit name."""
        mock_generator = Mock()
        mock_generator_class = Mock(return_value=mock_generator)
        mock_get_gen.return_value = mock_generator_class
        mock_init_reg.return_value = True

        cli_runner.invoke(
            generate,
            [
                "capability",
                "--from-prompt",
                "Fetch weather data",
                "-n",
                "weather_fetcher",
                "--quiet",
            ],
        )

        # Verify name passed to generator
        call_kwargs = mock_generator_class.call_args.kwargs
        assert call_kwargs["capability_name"] == "weather_fetcher"

    @patch("osprey.cli.generate_cmd.initialize_registry")
    @patch("osprey.cli.generate_cmd.get_prompt_generator")
    @patch("osprey.cli.generate_cmd.asyncio.run")
    def test_capability_from_prompt_keyboard_interrupt(
        self, mock_asyncio, mock_get_gen, mock_init_reg, cli_runner, mock_project
    ):
        """Test prompt-based generation handles KeyboardInterrupt."""
        mock_generator = Mock()
        mock_generator_class = Mock(return_value=mock_generator)
        mock_get_gen.return_value = mock_generator_class
        mock_init_reg.return_value = True
        mock_asyncio.side_effect = KeyboardInterrupt()

        result = cli_runner.invoke(
            generate, ["capability", "--from-prompt", "Fetch weather data", "--quiet"]
        )

        assert result.exit_code != 0
        assert "cancelled" in result.output.lower()


# =============================================================================
# Test MCP Server Command
# =============================================================================


class TestMCPServerCommand:
    """Test MCP server generation command."""

    def test_mcp_server_help(self, cli_runner):
        """Test mcp-server command help output."""
        result = cli_runner.invoke(generate, ["mcp-server", "--help"])

        assert result.exit_code == 0
        assert "mcp" in result.output.lower()
        assert "--name" in result.output
        assert "--output" in result.output
        assert "--port" in result.output

    @patch("osprey.cli.generate_cmd.get_server_template")
    def test_mcp_server_default_options(self, mock_get_template, cli_runner, tmp_path, monkeypatch):
        """Test MCP server generation with default options."""
        monkeypatch.chdir(tmp_path)

        mock_write_func = Mock(return_value=Path("demo_mcp_server.py"))
        mock_get_template.return_value = mock_write_func

        result = cli_runner.invoke(generate, ["mcp-server"])

        assert result.exit_code == 0
        assert mock_write_func.called

        # Verify default values
        call_kwargs = mock_write_func.call_args.kwargs
        assert call_kwargs["server_name"] == "demo_mcp"
        assert call_kwargs["port"] == 3001

    @patch("osprey.cli.generate_cmd.get_server_template")
    def test_mcp_server_custom_name(self, mock_get_template, cli_runner, tmp_path, monkeypatch):
        """Test MCP server generation with custom name."""
        monkeypatch.chdir(tmp_path)

        mock_write_func = Mock(return_value=Path("my_server_server.py"))
        mock_get_template.return_value = mock_write_func

        result = cli_runner.invoke(generate, ["mcp-server", "--name", "my_server"])

        assert result.exit_code == 0
        call_kwargs = mock_write_func.call_args.kwargs
        assert call_kwargs["server_name"] == "my_server"

    @patch("osprey.cli.generate_cmd.get_server_template")
    def test_mcp_server_custom_port(self, mock_get_template, cli_runner, tmp_path, monkeypatch):
        """Test MCP server generation with custom port."""
        monkeypatch.chdir(tmp_path)

        mock_write_func = Mock(return_value=Path("demo_mcp_server.py"))
        mock_get_template.return_value = mock_write_func

        result = cli_runner.invoke(generate, ["mcp-server", "--port", "3002"])

        assert result.exit_code == 0
        call_kwargs = mock_write_func.call_args.kwargs
        assert call_kwargs["port"] == 3002

    @patch("osprey.cli.generate_cmd.get_server_template")
    def test_mcp_server_custom_output(self, mock_get_template, cli_runner, tmp_path, monkeypatch):
        """Test MCP server generation with custom output path."""
        monkeypatch.chdir(tmp_path)

        custom_path = tmp_path / "servers" / "my_server.py"
        mock_write_func = Mock(return_value=custom_path)
        mock_get_template.return_value = mock_write_func

        result = cli_runner.invoke(generate, ["mcp-server", "--output", str(custom_path)])

        assert result.exit_code == 0
        call_args = mock_write_func.call_args.kwargs
        assert call_args["output_path"] == custom_path

    @patch("osprey.cli.generate_cmd.get_server_template")
    def test_mcp_server_generation_error(
        self, mock_get_template, cli_runner, tmp_path, monkeypatch
    ):
        """Test MCP server generation handles errors."""
        monkeypatch.chdir(tmp_path)

        mock_write_func = Mock(side_effect=Exception("Generation failed"))
        mock_get_template.return_value = mock_write_func

        result = cli_runner.invoke(generate, ["mcp-server"])

        assert result.exit_code != 0
        assert "Generation failed" in result.output


# =============================================================================
# Test Claude Config Command
# =============================================================================


class TestClaudeConfigCommand:
    """Test Claude Code configuration generation command."""

    def test_claude_config_help(self, cli_runner):
        """Test claude-config command help output."""
        result = cli_runner.invoke(generate, ["claude-config", "--help"])

        assert result.exit_code == 0
        assert "claude" in result.output.lower()
        assert "--output" in result.output
        assert "--force" in result.output

    @patch("osprey.cli.templates.TemplateManager")
    def test_claude_config_default_output(
        self, mock_template_mgr, cli_runner, tmp_path, monkeypatch
    ):
        """Test Claude config generation with default output path."""
        monkeypatch.chdir(tmp_path)

        mock_mgr_instance = Mock()
        mock_template_mgr.return_value = mock_mgr_instance

        result = cli_runner.invoke(generate, ["claude-config"])

        assert result.exit_code == 0
        assert mock_mgr_instance.render_template.called

        # Check default output filename
        call_args = mock_mgr_instance.render_template.call_args
        output_path = call_args[0][2]
        assert "claude_generator_config.yml" in str(output_path)

    @patch("osprey.cli.templates.TemplateManager")
    def test_claude_config_custom_output(
        self, mock_template_mgr, cli_runner, tmp_path, monkeypatch
    ):
        """Test Claude config generation with custom output path."""
        monkeypatch.chdir(tmp_path)

        mock_mgr_instance = Mock()
        mock_template_mgr.return_value = mock_mgr_instance

        result = cli_runner.invoke(generate, ["claude-config", "--output", "custom_claude.yml"])

        assert result.exit_code == 0
        call_args = mock_mgr_instance.render_template.call_args
        output_path = call_args[0][2]
        assert "custom_claude.yml" in str(output_path)

    def test_claude_config_file_exists_no_force(self, cli_runner, tmp_path, monkeypatch):
        """Test Claude config generation fails when file exists without --force."""
        monkeypatch.chdir(tmp_path)

        # Create existing file
        existing_file = tmp_path / "claude_generator_config.yml"
        existing_file.write_text("existing content")

        result = cli_runner.invoke(generate, ["claude-config"])

        assert result.exit_code != 0
        assert "already exists" in result.output
        assert "--force" in result.output

    @patch("osprey.cli.templates.TemplateManager")
    def test_claude_config_file_exists_with_force(
        self, mock_template_mgr, cli_runner, tmp_path, monkeypatch
    ):
        """Test Claude config generation overwrites with --force."""
        monkeypatch.chdir(tmp_path)

        # Create existing file
        existing_file = tmp_path / "claude_generator_config.yml"
        existing_file.write_text("existing content")

        mock_mgr_instance = Mock()
        mock_template_mgr.return_value = mock_mgr_instance

        result = cli_runner.invoke(generate, ["claude-config", "--force"])

        assert result.exit_code == 0
        assert mock_mgr_instance.render_template.called

    @patch("osprey.cli.templates.TemplateManager")
    def test_claude_config_detects_provider_from_config(
        self, mock_template_mgr, cli_runner, tmp_path, monkeypatch
    ):
        """Test Claude config detects provider from existing config.yml."""
        monkeypatch.chdir(tmp_path)

        # Create config.yml with cborg provider
        config_file = tmp_path / "config.yml"
        config_file.write_text("""
llm:
  default_provider: cborg
""")

        mock_mgr_instance = Mock()
        mock_template_mgr.return_value = mock_mgr_instance

        result = cli_runner.invoke(generate, ["claude-config"])

        assert result.exit_code == 0
        # Verify provider was detected
        call_args = mock_mgr_instance.render_template.call_args
        context = call_args[0][1]
        assert context["default_provider"] == "cborg"

    @patch("osprey.cli.templates.TemplateManager")
    def test_claude_config_generation_error(
        self, mock_template_mgr, cli_runner, tmp_path, monkeypatch
    ):
        """Test Claude config generation handles errors."""
        monkeypatch.chdir(tmp_path)

        mock_mgr_instance = Mock()
        mock_mgr_instance.render_template.side_effect = Exception("Template error")
        mock_template_mgr.return_value = mock_mgr_instance

        result = cli_runner.invoke(generate, ["claude-config"])

        assert result.exit_code != 0
        assert "Generation failed" in result.output
