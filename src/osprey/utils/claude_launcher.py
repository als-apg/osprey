"""Build argv prefixes for invoking the Claude Code CLI.

OSPREY projects can pin a specific Claude Code CLI version via the
``claude_code.cli_version`` config field. When set, OSPREY launches Claude
through ``npx`` rather than the user's global install, insulating projects
from upstream CC releases that break compatibility (see issue #218).

That pin governs the CLI OSPREY *spawns*. It does not govern the CLI the
Agent SDK runs, which prefers a binary bundled inside its own package, so a
pinned project runs two different Claude Code builds side by side. The
:func:`argv_cli_version` / :func:`bundled_cli_version` pair reads both, which
is what lets a caller say so out loud instead of leaving the difference to be
discovered as a behaviour one surface has and the other does not.
"""

from __future__ import annotations

import logging
import platform
import re
import subprocess
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_VERSION_RE = re.compile(r"\b(\d+\.\d+\.\d+(?:[-+.][0-9A-Za-z.-]+)?)\b")

#: The npm package a pinned launch runs through ``npx``. Spelled once so the
#: argv builder and the argv reader below cannot drift apart.
_CLI_PACKAGE = "@anthropic-ai/claude-code"

#: Directory inside the Agent SDK package that holds its own CLI binary.
_BUNDLED_DIRNAME = "_bundled"

#: How long the bundled binary gets to answer ``--version``. It is a large
#: single-file executable and this runs at server startup, so the probe is
#: bounded rather than allowed to hold a boot open; a timeout reads the same
#: as a missing bundle.
_VERSION_PROBE_TIMEOUT_S = 5.0


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
            globally installed ``claude`` (mirrors ``osprey chat --no-pin``).
            The ``--setting-sources`` restriction is applied
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
        base = ["npx", "-y", f"{_CLI_PACKAGE}@{cli_version.strip()}"]
    return base + _SETTING_SOURCES_ARGS


def parse_claude_version(version_output: str) -> str | None:
    """Extract a semver string from ``claude --version`` output.

    Returns ``None`` if no recognisable version token is present.
    """
    if not version_output:
        return None
    match = _VERSION_RE.search(version_output)
    return match.group(1) if match else None


def argv_cli_version(argv: Sequence[str]) -> str | None:
    """Extract the pinned CLI version from a launch argv.

    Args:
        argv: An argv prefix, normally one :func:`build_claude_launch_argv`
            returned.

    Returns:
        The pinned version, or ``None`` when the argv names no pin — an
        unpinned launch runs whatever ``claude`` PATH resolves to, whose
        version is not knowable from the argv alone.
    """
    prefix = f"{_CLI_PACKAGE}@"
    for arg in argv:
        if arg.startswith(prefix):
            return parse_claude_version(arg[len(prefix) :])
    return None


def bundled_cli_path() -> Path | None:
    """Return the Claude Code binary bundled inside the Agent SDK, if any.

    The SDK prefers this binary over anything on PATH, so it — not ``claude``
    — is what the chat surface actually runs. Resolved the same way the SDK
    resolves it, from the installed package's own location.

    Returns:
        The binary's path, or ``None`` when the SDK is not installed or ships
        without a bundle.
    """
    try:
        import claude_agent_sdk
    except ImportError:
        return None
    package_file = getattr(claude_agent_sdk, "__file__", None)
    if not package_file:
        return None
    name = "claude.exe" if platform.system() == "Windows" else "claude"
    candidate = Path(package_file).resolve().parent / _BUNDLED_DIRNAME / name
    return candidate if candidate.is_file() else None


@lru_cache(maxsize=1)
def bundled_cli_version() -> str | None:
    """Return the version of the SDK's bundled CLI, or ``None``.

    Cached for the life of the process: the answer cannot change under a
    running server, and the probe spawns a large executable. Every failure —
    no bundle, a binary that will not run, a non-zero exit, a timeout, output
    with no recognisable version — answers ``None``, because this is
    diagnostic information and no caller should be denied a start over it.
    """
    path = bundled_cli_path()
    if path is None:
        return None
    try:
        completed = subprocess.run(
            [str(path), "--version"],
            capture_output=True,
            text=True,
            timeout=_VERSION_PROBE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("Could not read the bundled Claude Code version: %s", exc)
        return None
    if completed.returncode != 0:
        return None
    return parse_claude_version(completed.stdout)
