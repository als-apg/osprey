"""Tests for the Claude Code launcher argv builder.

These tests verify that ``build_claude_launch_argv()`` produces the correct
argv prefix for both the unpinned and pinned cases, and that
``parse_claude_version()`` extracts semver from realistic CLI output.
"""

from unittest.mock import patch

import pytest

from osprey.agent_runner.launcher import (
    CLI_NAME,
    build_claude_launch_argv,
    parse_claude_version,
    resolve_cli_name,
)

_RESOLVER = "osprey.utils.shell_resolver.resolve_shell_command"


class TestBuildClaudeLaunchArgv:
    """Test argv construction from a ``claude_code`` config dict.

    Every launch path emits ``--setting-sources project`` so a user's global
    ``~/.claude/settings.json`` (or a gitignored ``.claude/settings.local.json``)
    cannot override the project's provider ``env`` via the settings-file scope
    that outranks the process environment. This mirrors the SDK
    launch paths (``agent_runner.primitives``, ``dispatch_worker.sdk_runner``),
    which set ``setting_sources=["project"]``.
    """

    def test_no_pin_returns_bare_claude(self):
        assert build_claude_launch_argv({}) == ["claude", "--setting-sources", "project"]

    def test_unrelated_keys_do_not_trigger_pin(self):
        cc_config = {"provider": "anthropic", "default_model": "claude-haiku-4-5"}
        assert build_claude_launch_argv(cc_config) == ["claude", "--setting-sources", "project"]

    def test_pinned_returns_npx_invocation(self):
        assert build_claude_launch_argv({"cli_version": "2.1.146"}) == [
            "npx",
            "-y",
            "@anthropic-ai/claude-code@2.1.146",
            "--setting-sources",
            "project",
        ]

    def test_pinned_strips_whitespace(self):
        assert build_claude_launch_argv({"cli_version": "  2.1.146  "}) == [
            "npx",
            "-y",
            "@anthropic-ai/claude-code@2.1.146",
            "--setting-sources",
            "project",
        ]

    def test_setting_sources_always_emitted(self):
        """Both the pinned and unpinned prefixes end with the isolation flag."""
        for cc_config in ({}, {"cli_version": "2.1.146"}):
            argv = build_claude_launch_argv(cc_config)
            assert argv[-2:] == ["--setting-sources", "project"]

    def test_no_pin_flag_forces_bare_claude_but_keeps_isolation(self):
        """``no_pin=True`` ignores a configured pin (matching ``osprey claude
        chat --no-pin``) yet still restricts setting sources to project scope."""
        argv = build_claude_launch_argv({"cli_version": "2.1.146"}, no_pin=True)
        assert argv == ["claude", "--setting-sources", "project"]

    def test_empty_string_pin_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            build_claude_launch_argv({"cli_version": ""})

    def test_whitespace_only_pin_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            build_claude_launch_argv({"cli_version": "   "})

    def test_non_string_pin_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            build_claude_launch_argv({"cli_version": 2.1})

    def test_no_pin_flag_skips_pin_validation(self):
        """With ``no_pin=True`` an invalid pin is irrelevant — it is never read."""
        assert build_claude_launch_argv({"cli_version": ""}, no_pin=True) == [
            "claude",
            "--setting-sources",
            "project",
        ]


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


class TestResolveCliName:
    """The bare program name is resolved to an absolute path; nothing else is."""

    def test_bare_name_is_resolved_and_flags_kept(self):
        with patch(_RESOLVER, return_value="/abs/claude") as resolver:
            argv = resolve_cli_name(build_claude_launch_argv({}))
        assert argv == ["/abs/claude", "--setting-sources", "project"]
        resolver.assert_called_once_with("claude")

    def test_pinned_prefix_is_left_to_path_lookup(self):
        pinned = build_claude_launch_argv({"cli_version": "2.1.146"})
        with patch(_RESOLVER) as resolver:
            argv = resolve_cli_name(pinned)
        assert argv == pinned
        assert argv is not pinned
        resolver.assert_not_called()

    def test_foreign_argv_is_untouched(self):
        with patch(_RESOLVER) as resolver:
            argv = resolve_cli_name(["/bin/bash", "-l"])
        assert argv == ["/bin/bash", "-l"]
        resolver.assert_not_called()

    def test_unresolvable_name_raises_file_not_found(self):
        with patch(_RESOLVER, side_effect=FileNotFoundError("x")):
            with pytest.raises(FileNotFoundError):
                resolve_cli_name(build_claude_launch_argv({}))

    def test_empty_argv_is_refused(self):
        with pytest.raises(ValueError, match="resolve_cli_name"):
            resolve_cli_name([])

    def test_cli_name_is_what_an_unpinned_launch_runs(self):
        assert build_claude_launch_argv({})[0] == CLI_NAME == "claude"
