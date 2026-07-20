"""Tests for the channel-finder generate CLI subcommand."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from osprey.cli.channel_finder_cmd import channel_finder


@pytest.fixture
def runner():
    return CliRunner()


class TestGenerateSubcommand:
    """Tests for osprey channel-finder generate."""

    def test_help(self, runner):
        """--help shows flags and description."""
        result = runner.invoke(channel_finder, ["generate", "--help"])
        assert result.exit_code == 0
        assert "--output-dir" in result.output
        assert "--validate" in result.output
        assert "--source" in result.output
        assert "--format" in result.output
        assert "--tier" in result.output

    def test_generates_three_files(self, runner, tmp_path):
        """Running generate creates 3 database files."""
        result = runner.invoke(channel_finder, ["generate", "--output-dir", str(tmp_path)])
        assert result.exit_code == 0, f"Failed: {result.output}"

        assert (tmp_path / "in_context.json").exists()
        assert (tmp_path / "hierarchical.json").exists()
        assert (tmp_path / "middle_layer.json").exists()

    def test_in_context_channel_count_default(self, runner, tmp_path):
        """Default (no tier) in_context.json should have all channels from the template."""
        from osprey.services.channel_finder.benchmarks.generator import load_template

        _, all_channels = load_template()

        runner.invoke(channel_finder, ["generate", "--output-dir", str(tmp_path)])

        data = json.loads((tmp_path / "in_context.json").read_text())
        assert isinstance(data, dict)
        assert "_metadata" in data
        assert len(data["channels"]) == len(all_channels)

    def test_in_context_channel_count_tier1(self, runner, tmp_path):
        """--tier 1 in_context.json should hold exactly the tier-1 filtered channels."""
        from osprey.services.channel_finder.benchmarks.generator import (
            TIER_1,
            filter_channels,
            load_template,
        )

        _, all_channels = load_template()
        expected = len(filter_channels(all_channels, TIER_1))

        runner.invoke(
            channel_finder,
            ["generate", "--output-dir", str(tmp_path), "--tier", "1"],
        )

        data = json.loads((tmp_path / "in_context.json").read_text())
        assert len(data["channels"]) == expected

    def test_hierarchical_channel_count(self, runner, tmp_path):
        """hierarchical.json should have all template channels (default, no tier filter)."""
        from osprey.services.channel_finder.benchmarks.generator import (
            expand_hierarchy,
            load_template,
        )

        _, all_channels = load_template()

        runner.invoke(channel_finder, ["generate", "--output-dir", str(tmp_path)])

        data = json.loads((tmp_path / "hierarchical.json").read_text())
        channels = expand_hierarchy(data)
        assert len(channels) == len(all_channels)

    def test_middle_layer_channel_count(self, runner, tmp_path):
        """middle_layer.json should have all template channels (default, no tier filter)."""
        from osprey.services.channel_finder.benchmarks.generator import (
            collect_middle_layer_pvs,
            load_template,
        )

        _, all_channels = load_template()

        runner.invoke(channel_finder, ["generate", "--output-dir", str(tmp_path)])

        data = json.loads((tmp_path / "middle_layer.json").read_text())
        pvs = collect_middle_layer_pvs(data)
        assert len(pvs) == len(all_channels)

    def test_output_dir_flag(self, runner, tmp_path):
        """--output-dir creates files in specified directory."""
        custom = tmp_path / "custom_output"
        result = runner.invoke(channel_finder, ["generate", "--output-dir", str(custom)])
        assert result.exit_code == 0
        assert (custom / "in_context.json").exists()

    def test_validate_flag(self, runner, tmp_path):
        """--validate produces validation output."""
        result = runner.invoke(
            channel_finder, ["generate", "--output-dir", str(tmp_path), "--validate"]
        )
        assert result.exit_code == 0
        assert "validated" in result.output.lower() or "OK" in result.output

    def test_generate_single_format(self, runner, tmp_path):
        """--format in_context generates only one file."""
        result = runner.invoke(
            channel_finder,
            ["generate", "--output-dir", str(tmp_path), "--format", "in_context"],
        )
        assert result.exit_code == 0
        assert (tmp_path / "in_context.json").exists()
        assert not (tmp_path / "hierarchical.json").exists()
        assert not (tmp_path / "middle_layer.json").exists()

    def test_generate_tier1(self, runner, tmp_path):
        """--tier 1 generates only the in_context view at tier 1 scale."""
        from osprey.services.channel_finder.benchmarks.generator import (
            TIER_1,
            filter_channels,
            load_template,
        )

        _, all_channels = load_template()
        expected = len(filter_channels(all_channels, TIER_1))

        result = runner.invoke(
            channel_finder,
            ["generate", "--output-dir", str(tmp_path), "--tier", "1"],
        )
        assert result.exit_code == 0

        # Tier 1 is in_context-only: the flat view holds the filtered channels...
        ic_data = json.loads((tmp_path / "in_context.json").read_text())
        assert len(ic_data["channels"]) == expected

        # ...and the tree/middle-layer views are not emitted.
        assert not (tmp_path / "hierarchical.json").exists()
        assert not (tmp_path / "middle_layer.json").exists()

    def test_generate_tier1_rejects_non_in_context_format(self, runner, tmp_path):
        """--tier 1 with a non-in_context --format is refused."""
        result = runner.invoke(
            channel_finder,
            ["generate", "--output-dir", str(tmp_path), "--tier", "1", "--format", "hierarchical"],
        )
        assert result.exit_code != 0
        assert "in_context" in result.output

    def test_generate_custom_source(self, runner, tmp_path):
        """--source custom.json loads from a custom hierarchical template."""
        from osprey.services.channel_finder.benchmarks.generator import (
            TEMPLATE_DB_PATH,
            load_template,
        )

        _, all_channels = load_template()

        # Use the built-in template as a "custom" source to verify the flag works
        result = runner.invoke(
            channel_finder,
            [
                "generate",
                "--output-dir",
                str(tmp_path),
                "--source",
                str(TEMPLATE_DB_PATH),
                "--format",
                "in_context",
            ],
        )
        assert result.exit_code == 0
        data = json.loads((tmp_path / "in_context.json").read_text())
        assert len(data["channels"]) == len(all_channels)

    def test_validate_single_format(self, runner, tmp_path):
        """--validate works with a single format."""
        result = runner.invoke(
            channel_finder,
            [
                "generate",
                "--output-dir",
                str(tmp_path),
                "--format",
                "in_context",
                "--validate",
            ],
        )
        assert result.exit_code == 0
        assert "OK" in result.output
