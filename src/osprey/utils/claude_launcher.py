"""Build argv prefixes for invoking the Claude Code CLI.

OSPREY projects can pin a specific Claude Code CLI version via the
``claude_code.cli_version`` config field. When set, OSPREY launches Claude
through ``npx`` rather than the user's global install, insulating projects
from upstream CC releases that break compatibility (see issue #218).
"""

from __future__ import annotations

import re

_VERSION_RE = re.compile(r"\b(\d+\.\d+\.\d+(?:[-+.][0-9A-Za-z.-]+)?)\b")


def build_claude_launch_argv(cc_config: dict) -> list[str]:
    """Return the argv prefix used to launch Claude Code.

    Args:
        cc_config: The ``claude_code`` block from ``config.yml`` (may be empty).

    Returns:
        ``["claude"]`` when no version is pinned, otherwise
        ``["npx", "-y", "@anthropic-ai/claude-code@<version>"]``.

    Raises:
        ValueError: If ``cli_version`` is present but empty/whitespace.
    """
    cli_version = cc_config.get("cli_version")
    if cli_version is None:
        return ["claude"]
    if not isinstance(cli_version, str) or not cli_version.strip():
        raise ValueError(
            "claude_code.cli_version must be a non-empty string "
            '(e.g. "2.1.146"); got an empty value.'
        )
    return ["npx", "-y", f"@anthropic-ai/claude-code@{cli_version.strip()}"]


def parse_claude_version(version_output: str) -> str | None:
    """Extract a semver string from ``claude --version`` output.

    Returns ``None`` if no recognisable version token is present.
    """
    if not version_output:
        return None
    match = _VERSION_RE.search(version_output)
    return match.group(1) if match else None
