"""Build argv prefixes for invoking the Claude Code CLI.

OSPREY projects can pin a specific Claude Code CLI version via the
``claude_code.cli_version`` config field. When set, OSPREY launches Claude
through ``npx`` rather than the user's global install, insulating projects
from upstream CC releases that break compatibility (see issue #218).
"""

from __future__ import annotations

import re

_VERSION_RE = re.compile(r"\b(\d+\.\d+\.\d+(?:[-+.][0-9A-Za-z.-]+)?)\b")


# Restrict Claude Code to project-scope settings files. A user's global
# ~/.claude/settings.json and a gitignored .claude/settings.local.json both
# outrank the inherited process environment, so an `env` block there would
# silently override the provider variables OSPREY injects at launch — including
# ANTHROPIC_BASE_URL, bypassing the translation proxy (issue #355). Loading only
# the project scope makes the process environment authoritative again. The SDK
# launch paths (agent_runner.primitives, dispatch_worker.sdk_runner) already
# pass setting_sources=["project"]; this keeps the subprocess paths consistent.
_SETTING_SOURCES_ARGS = ["--setting-sources", "project"]


def build_claude_launch_argv(cc_config: dict, *, no_pin: bool = False) -> list[str]:
    """Return the argv prefix used to launch Claude Code.

    Args:
        cc_config: The ``claude_code`` block from ``config.yml`` (may be empty).
        no_pin: When ``True``, ignore any ``cli_version`` pin and launch the
            globally installed ``claude`` (mirrors ``osprey claude chat
            --no-pin``). The ``--setting-sources`` restriction is applied
            regardless, so provider isolation cannot be opted out of.

    Returns:
        ``["claude", "--setting-sources", "project"]`` when no version is pinned,
        otherwise the ``npx -y @anthropic-ai/claude-code@<version>`` prefix with
        the same ``--setting-sources`` suffix.

    Raises:
        ValueError: If ``cli_version`` is present but empty/whitespace (only
            checked when ``no_pin`` is ``False``).
    """
    cli_version = None if no_pin else cc_config.get("cli_version")
    if cli_version is None:
        base = ["claude"]
    elif not isinstance(cli_version, str) or not cli_version.strip():
        raise ValueError(
            "claude_code.cli_version must be a non-empty string "
            '(e.g. "2.1.146"); got an empty value.'
        )
    else:
        base = ["npx", "-y", f"@anthropic-ai/claude-code@{cli_version.strip()}"]
    return base + _SETTING_SOURCES_ARGS


def parse_claude_version(version_output: str) -> str | None:
    """Extract a semver string from ``claude --version`` output.

    Returns ``None`` if no recognisable version token is present.
    """
    if not version_output:
        return None
    match = _VERSION_RE.search(version_output)
    return match.group(1) if match else None
