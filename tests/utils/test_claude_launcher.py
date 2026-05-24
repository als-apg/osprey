"""Tests for the Claude Code launcher argv builder.

These tests verify that ``build_claude_launch_argv()`` produces the correct
argv prefix for both the unpinned and pinned cases, and that
``parse_claude_version()`` extracts semver from realistic CLI output.
"""

import pytest

from osprey.utils.claude_launcher import build_claude_launch_argv, parse_claude_version


class TestBuildClaudeLaunchArgv:
    """Test argv construction from a ``claude_code`` config dict."""

    def test_no_pin_returns_bare_claude(self):
        assert build_claude_launch_argv({}) == ["claude"]

    def test_unrelated_keys_do_not_trigger_pin(self):
        cc_config = {"provider": "anthropic", "default_model": "haiku"}
        assert build_claude_launch_argv(cc_config) == ["claude"]

    def test_pinned_returns_npx_invocation(self):
        assert build_claude_launch_argv({"cli_version": "2.1.146"}) == [
            "npx",
            "-y",
            "@anthropic-ai/claude-code@2.1.146",
        ]

    def test_pinned_strips_whitespace(self):
        assert build_claude_launch_argv({"cli_version": "  2.1.146  "}) == [
            "npx",
            "-y",
            "@anthropic-ai/claude-code@2.1.146",
        ]

    def test_empty_string_pin_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            build_claude_launch_argv({"cli_version": ""})

    def test_whitespace_only_pin_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            build_claude_launch_argv({"cli_version": "   "})

    def test_non_string_pin_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            build_claude_launch_argv({"cli_version": 2.1})


class TestParseClaudeVersion:
    """Test extraction of semver from ``claude --version`` output."""

    def test_extracts_simple_semver(self):
        assert parse_claude_version("2.1.146") == "2.1.146"

    def test_extracts_from_typical_cli_output(self):
        # Typical output looks like "2.1.146 (Claude Code)"
        assert parse_claude_version("2.1.146 (Claude Code)") == "2.1.146"

    def test_extracts_from_verbose_prefix(self):
        assert parse_claude_version("Claude Code version 2.1.146 on darwin") == "2.1.146"

    def test_returns_none_for_garbage(self):
        assert parse_claude_version("nope, no version here") is None

    def test_returns_none_for_empty(self):
        assert parse_claude_version("") is None
