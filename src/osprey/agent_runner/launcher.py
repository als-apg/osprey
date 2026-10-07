"""Build argv prefixes for invoking the Claude Code CLI.

OSPREY projects can pin a specific Claude Code CLI version via the
``claude_code.cli_version`` config field. When set, OSPREY launches Claude
through ``npx`` rather than the user's global install, insulating projects
from upstream CC releases that break compatibility.

This module is the harness adapter's single home for the agent CLI's command
line: the program, the pin, the isolation flag and the conversation flags.

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

#: The program an unpinned launch runs, found on PATH.
CLI_NAME = "claude"

_VERSION_RE = re.compile(r"\b(\d+\.\d+\.\d+(?:[-+.][0-9A-Za-z.-]+)?)\b")

#: The npm package a pinned launch runs through ``npx``. Spelled once so the
#: argv builder and the argv reader below cannot drift apart.
_CLI_PACKAGE = "@anthropic-ai/claude-code"

#: The MCP server config the build renders at the project root, and the only
#: one an agent launch loads.
RENDERED_MCP_CONFIG = ".mcp.json"

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
# ANTHROPIC_BASE_URL, bypassing the translation proxy. Loading only
# the project scope makes the process environment authoritative again. The SDK
# launch paths (agent_runner.primitives, dispatch_worker.sdk_runner) already
# pass setting_sources=["project"]; this keeps the subprocess paths consistent.
_SETTING_SOURCES_ARGS = ["--setting-sources", "project"]

# Load only the MCP servers the build rendered. ``--strict-mcp-config`` makes the
# named config the only MCP source, so plugin servers (``--plugin-dir``, settings,
# managed), claude.ai connectors and user- or local-scope servers never load. The
# path is relative because every launch runs with the project root as its working
# directory, and it is the file the CLI reads from there anyway. The ``=`` form is
# required: ``--mcp-config`` takes several values and would swallow a trailing
# opening message as a second config file.
_MCP_ISOLATION_ARGS = ["--strict-mcp-config", f"--mcp-config={RENDERED_MCP_CONFIG}"]

# The conversation flags :func:`build_session_argv` appends to a launch prefix.
_RESUME_FLAG = "--resume"
_SESSION_ID_FLAG = "--session-id"
_PRINT_FLAG = "--print"
_EFFORT_FLAG = "--effort"

#: What ``--resume <id>`` prints before exiting 1 when no transcript for the id
#: exists. A caller that resumes a conversation watches the child's early output
#: for it (:class:`NoConversationWatch`) to tell a missing transcript from any
#: other exit.
NO_CONVERSATION_MARKER = b"No conversation found with session ID"

#: How much of a ``--resume`` child's output is searched for the marker. The
#: verdict is the first thing the CLI prints, so anything beyond the first
#: few kilobytes is a session that resumed and is now doing real work.
NO_CONVERSATION_SCAN_LIMIT = 16 * 1024


def build_claude_launch_argv(cc_config: dict, *, no_pin: bool = False) -> list[str]:
    """Return the argv prefix used to launch Claude Code.

    Args:
        cc_config: The ``claude_code`` block from ``config.yml`` (may be empty).
        no_pin: When ``True``, ignore any ``cli_version`` pin and launch the
            globally installed ``claude`` (mirrors ``osprey chat --no-pin``).
            The ``--setting-sources`` restriction and the MCP restriction are
            applied regardless, so neither provider isolation nor MCP isolation
            can be opted out of.

    Returns:
        ``["claude", "--setting-sources", "project", "--strict-mcp-config",
        "--mcp-config=.mcp.json"]`` when no version is pinned, otherwise the
        ``npx -y @anthropic-ai/claude-code@<version>`` prefix with the same
        suffix.

    Raises:
        ValueError: If ``cli_version`` is present but empty/whitespace (only
            checked when ``no_pin`` is ``False``).
    """
    cli_version = None if no_pin else cc_config.get("cli_version")
    if cli_version is None:
        base = [CLI_NAME]
    elif not isinstance(cli_version, str) or not cli_version.strip():
        raise ValueError(
            "claude_code.cli_version must be a non-empty string "
            '(e.g. "2.1.146"); got an empty value.'
        )
    else:
        base = ["npx", "-y", f"{_CLI_PACKAGE}@{cli_version.strip()}"]
    return base + _SETTING_SOURCES_ARGS + _MCP_ISOLATION_ARGS


def resolve_cli_name(argv: Sequence[str]) -> list[str]:
    """Resolve the bare CLI name at the head of a launch argv to an absolute path.

    A stripped PATH (a systemd unit, a container entrypoint) must still find the
    CLI, so an argv that starts with :data:`CLI_NAME` gets its program resolved
    while every flag the launcher appended — notably ``--setting-sources
    project`` — is preserved. Any other argv comes back as a new, equal list: a
    pinned ``npx …`` prefix is left to the PATH lookup, and an argv that names
    some other program is not this function's to touch.

    Args:
        argv: A launch argv, normally one :func:`build_claude_launch_argv`
            returned.

    Returns:
        A new argv list.

    Raises:
        ValueError: If ``argv`` is empty.
        FileNotFoundError: From
            :func:`osprey.utils.shell_resolver.resolve_shell_command` when the
            CLI cannot be found.
    """
    from osprey.utils.shell_resolver import resolve_shell_command

    if not argv:
        raise ValueError("resolve_cli_name() needs a non-empty argv")
    if argv[0] == CLI_NAME:
        return [resolve_shell_command(argv[0]), *argv[1:]]
    return list(argv)


def build_session_argv(
    base: Sequence[str],
    *,
    resume_id: str | None = None,
    session_id: str | None = None,
    print_mode: bool = False,
    effort: str | None = None,
    prompt: str | None = None,
) -> list[str]:
    """Extend a launch prefix with the conversation flags, as a new list.

    The flags follow ``base`` in a fixed order: ``--resume <resume_id>`` or
    ``--session-id <session_id>``, then ``--print``, then ``--effort <effort>``,
    then ``prompt``. The prompt is last because the CLI reads one trailing
    positional as the opening message. An empty ``resume_id``, ``session_id``
    or ``effort`` adds no flag; an empty ``prompt`` is still passed.

    Args:
        base: The launch prefix, normally one :func:`build_claude_launch_argv`
            returned (resolved or not). It is never mutated.
        resume_id: The conversation to resume.
        session_id: The id a new conversation is forced onto.
        print_mode: Run non-interactively and print the answer.
        effort: The reasoning effort level.
        prompt: The opening message.

    Returns:
        A new argv list.

    Raises:
        ValueError: If both ``resume_id`` and ``session_id`` are given; a child
            either resumes a transcript or starts one under a forced id.
    """
    if resume_id and session_id:
        raise ValueError("build_session_argv() takes resume_id or session_id, not both")
    argv = list(base)
    if resume_id:
        argv.extend([_RESUME_FLAG, resume_id])
    elif session_id:
        argv.extend([_SESSION_ID_FLAG, session_id])
    if print_mode:
        argv.append(_PRINT_FLAG)
    if effort:
        argv.extend([_EFFORT_FLAG, effort])
    if prompt is not None:
        argv.append(prompt)
    return argv


class NoConversationWatch:
    """Watch a ``--resume`` child's early output for :data:`NO_CONVERSATION_MARKER`.

    Only the first :data:`NO_CONVERSATION_SCAN_LIMIT` bytes are searched; the
    chunk that crosses the limit is still scanned whole, and a marker split
    across two chunks is still found.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()
        #: Whether the marker has been seen. Once true it stays true.
        self.found = False

    def feed(self, data: bytes) -> None:
        """Add one chunk of the child's output to the scan."""
        if self.found or len(self._buffer) >= NO_CONVERSATION_SCAN_LIMIT:
            return
        self._buffer.extend(data)
        self.found = NO_CONVERSATION_MARKER in self._buffer


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
