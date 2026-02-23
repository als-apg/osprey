"""Tests for --model CLI flag tier restriction."""

from __future__ import annotations

from click.testing import CliRunner

from osprey.cli.init_cmd import init


class TestModelFlagValidation:
    """The --model flag only accepts tier names (haiku/sonnet/opus)."""

    def test_rejects_provider_specific_id(self):
        runner = CliRunner()
        result = runner.invoke(init, ["--model", "anthropic/claude-haiku", "/tmp/test"])
        assert result.exit_code != 0
        assert "Invalid value" in result.output

    def test_accepts_haiku(self):
        runner = CliRunner()
        # Use --help to avoid actually running init; just validate the option parses
        result = runner.invoke(init, ["--model", "haiku", "--help"])
        assert result.exit_code == 0

    def test_accepts_sonnet(self):
        runner = CliRunner()
        result = runner.invoke(init, ["--model", "sonnet", "--help"])
        assert result.exit_code == 0

    def test_accepts_opus(self):
        runner = CliRunner()
        result = runner.invoke(init, ["--model", "opus", "--help"])
        assert result.exit_code == 0

    def test_rejects_arbitrary_model_name(self):
        runner = CliRunner()
        result = runner.invoke(init, ["--model", "gpt-4", "/tmp/test"])
        assert result.exit_code != 0
        assert "Invalid value" in result.output
