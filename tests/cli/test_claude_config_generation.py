"""Tests for Claude Code generator configuration generation.

Tests the 'osprey generate claude-config' command and validates that
generated configurations are properly structured and ready for use.

These tests are fast and require no API calls.
"""

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from osprey.cli.generate_cmd import generate


class TestClaudeConfigGeneration:
    """Test Claude Code generator configuration file generation."""

    def test_generate_claude_config_basic(self, tmp_path):
        """Test basic claude-config generation with default settings."""
        runner = CliRunner()

        # Change to temp directory
        with runner.isolated_filesystem(temp_dir=tmp_path):
            output_file = "claude_generator_config.yml"

            result = runner.invoke(generate, ["claude-config", "--output", output_file])

            # Command should succeed
            assert result.exit_code == 0
            assert "Configuration generated" in result.output

            # File should be created
            assert Path(output_file).exists()

            # Load and validate YAML structure
            with open(output_file) as f:
                config = yaml.safe_load(f)

            # Should have required top-level keys
            assert "api_config" in config
            assert "phases" in config
            assert "profiles" in config
            assert "codebase_guidance" in config

            # API config should have provider settings
            assert "provider" in config["api_config"]

            # Phases should be a dict with phase definitions
            assert isinstance(config["phases"], dict)
            assert "scan" in config["phases"]
            assert "plan" in config["phases"]
            assert "implement" in config["phases"]
            assert "generate" in config["phases"]

            # Profiles should contain fast and robust
            assert isinstance(config["profiles"], dict)
            assert "fast" in config["profiles"]
            assert "robust" in config["profiles"]

            # Fast profile should have phases list
            assert "phases" in config["profiles"]["fast"]
            assert isinstance(config["profiles"]["fast"]["phases"], list)

            # Robust profile should have phases list
            assert "phases" in config["profiles"]["robust"]
            assert isinstance(config["profiles"]["robust"]["phases"], list)

    def test_generate_with_custom_output_path(self, tmp_path):
        """Test generating config with custom output path."""
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create subdirectory
            custom_dir = Path("configs")
            custom_dir.mkdir()

            output_file = custom_dir / "my_claude_config.yml"

            result = runner.invoke(generate, ["claude-config", "--output", str(output_file)])

            assert result.exit_code == 0
            assert output_file.exists()

    def test_generate_refuses_overwrite_without_force(self, tmp_path):
        """Test that command refuses to overwrite existing file without --force."""
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            output_file = "claude_generator_config.yml"

            # Create file first time
            result1 = runner.invoke(generate, ["claude-config", "--output", output_file])
            assert result1.exit_code == 0

            # Try to create again without --force
            result2 = runner.invoke(generate, ["claude-config", "--output", output_file])

            # Should fail
            assert result2.exit_code != 0
            assert "already exists" in result2.output
            assert "--force" in result2.output

    def test_generate_overwrites_with_force_flag(self, tmp_path):
        """Test that --force flag allows overwriting existing file."""
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            output_file = "claude_generator_config.yml"

            # Create file first time
            result1 = runner.invoke(generate, ["claude-config", "--output", output_file])
            assert result1.exit_code == 0

            # Modify the file
            with open(output_file, "w") as f:
                f.write("# Modified content\n")

            # Try to create again with --force
            result2 = runner.invoke(generate, ["claude-config", "--output", output_file, "--force"])

            # Should succeed
            assert result2.exit_code == 0

            # File should contain valid YAML (not "# Modified content")
            with open(output_file) as f:
                config = yaml.safe_load(f)
            assert "api_config" in config
            assert "phases" in config

    def test_generated_config_profile_phases_structure(self, tmp_path):
        """Test that generated profiles have correct phase lists."""
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            output_file = "claude_generator_config.yml"

            result = runner.invoke(generate, ["claude-config", "--output", output_file])
            assert result.exit_code == 0

            with open(output_file) as f:
                config = yaml.safe_load(f)

            # Fast profile should have single 'generate' phase
            fast_profile = config["profiles"]["fast"]
            assert "phases" in fast_profile
            assert fast_profile["phases"] == ["generate"]

            # Robust profile should have multi-phase workflow
            robust_profile = config["profiles"]["robust"]
            assert "phases" in robust_profile
            assert len(robust_profile["phases"]) > 1
            # Should include planning phases
            assert "scan" in robust_profile["phases"] or "plan" in robust_profile["phases"]

    def test_generated_config_phase_definitions(self, tmp_path):
        """Test that each phase has required definition fields."""
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            output_file = "claude_generator_config.yml"

            result = runner.invoke(generate, ["claude-config", "--output", output_file])
            assert result.exit_code == 0

            with open(output_file) as f:
                config = yaml.safe_load(f)

            phases = config["phases"]

            # Each phase should have a prompt
            for phase_name in ["scan", "plan", "implement", "generate"]:
                assert phase_name in phases
                phase_def = phases[phase_name]
                assert "prompt" in phase_def
                assert isinstance(phase_def["prompt"], str)
                assert len(phase_def["prompt"]) > 0

    def test_provider_detection_from_config(self, tmp_path):
        """Test that generator detects provider from existing config.yml."""
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Create a config.yml with cborg provider
            config_yml_content = """
llm:
  default_provider: cborg
  default_model: anthropic/claude-haiku
"""
            with open("config.yml", "w") as f:
                f.write(config_yml_content)

            output_file = "claude_generator_config.yml"

            result = runner.invoke(generate, ["claude-config", "--output", output_file])
            assert result.exit_code == 0

            # Should detect cborg provider
            assert "Detected provider:" in result.output
            assert "cborg" in result.output

            # Generated config should reflect detected provider
            with open(output_file) as f:
                config = yaml.safe_load(f)

            # API section should have cborg as provider
            assert config["api_config"]["provider"] == "cborg"

    def test_generated_config_has_codebase_guidance(self, tmp_path):
        """Test that generated config includes codebase guidance section."""
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            output_file = "claude_generator_config.yml"

            result = runner.invoke(generate, ["claude-config", "--output", output_file])
            assert result.exit_code == 0

            with open(output_file) as f:
                config = yaml.safe_load(f)

            # Should have codebase_guidance section
            assert "codebase_guidance" in config
            guidance = config["codebase_guidance"]

            # Should have at least one example library (like plotting)
            assert len(guidance) > 0

            # Each library should have directories and guidance
            for library_name, library_config in guidance.items():
                assert "directories" in library_config
                assert "guidance" in library_config
                assert isinstance(library_config["directories"], list)

    def test_next_steps_instructions_in_output(self, tmp_path):
        """Test that command output includes helpful next steps."""
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            output_file = "claude_generator_config.yml"

            result = runner.invoke(generate, ["claude-config", "--output", output_file])
            assert result.exit_code == 0

            # Should show what was created
            assert "What was created:" in result.output
            assert "API configuration" in result.output
            assert "Phase definitions" in result.output
            assert "Pre-configured profiles" in result.output

            # Should show next steps
            assert "Next Steps:" in result.output
            assert "config.yml" in result.output
            assert "code_generator:" in result.output
            assert "claude_code" in result.output


class TestClaudeConfigIntegration:
    """Test integration of generated config with ClaudeCodeGenerator."""

    def test_generated_config_loads_in_generator(self, tmp_path):
        """Test that generated config can be loaded by ClaudeCodeGenerator."""
        pytest.importorskip("anthropic_client")  # Skip if Claude SDK not available

        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            output_file = "claude_generator_config.yml"

            # Generate config
            result = runner.invoke(generate, ["claude-config", "--output", output_file])
            assert result.exit_code == 0

            # Try to load it with ClaudeCodeGenerator
            from osprey.services.python_executor.generation import ClaudeCodeGenerator

            # Load the generated config
            with open(output_file) as f:
                claude_config = yaml.safe_load(f)

            # Create generator with fast profile config
            generator_config = {
                "profile": "fast",
                "phases": claude_config["profiles"]["fast"]["phases"],
                "claude_config": claude_config,
            }

            # Should initialize without errors
            generator = ClaudeCodeGenerator(model_config=generator_config)

            # Should have the config loaded
            assert generator.config is not None
            assert generator.config.get("profile") == "fast"
            assert generator.config.get("profile_phases") == ["generate"]

    def test_fast_profile_workflow(self, tmp_path):
        """Test that fast profile configuration is valid for single-phase workflow."""
        pytest.importorskip("anthropic_client")  # Skip if Claude SDK not available

        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            output_file = "claude_generator_config.yml"

            # Generate config
            result = runner.invoke(generate, ["claude-config", "--output", output_file])
            assert result.exit_code == 0

            # Load config
            with open(output_file) as f:
                config = yaml.safe_load(f)

            fast_profile = config["profiles"]["fast"]

            # Fast profile should have single 'generate' phase
            assert fast_profile["phases"] == ["generate"]

            # Verify 'generate' phase is defined
            assert "generate" in config["phases"]
            assert "prompt" in config["phases"]["generate"]

    def test_robust_profile_workflow(self, tmp_path):
        """Test that robust profile configuration is valid for multi-phase workflow."""
        pytest.importorskip("anthropic_client")  # Skip if Claude SDK not available

        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            output_file = "claude_generator_config.yml"

            # Generate config
            result = runner.invoke(generate, ["claude-config", "--output", output_file])
            assert result.exit_code == 0

            # Load config
            with open(output_file) as f:
                config = yaml.safe_load(f)

            robust_profile = config["profiles"]["robust"]

            # Robust profile should have multiple phases
            assert len(robust_profile["phases"]) > 1

            # All phases in robust profile should be defined
            for phase in robust_profile["phases"]:
                assert phase in config["phases"]
                assert "prompt" in config["phases"][phase]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
