"""Build argv prefixes for invoking the Claude Code CLI.

OSPREY projects can pin a specific Claude Code CLI version via the
``claude_code.cli_version`` config field. When set, OSPREY launches Claude
through ``npx`` rather than the user's global install, insulating projects
from upstream CC releases that break compatibility.

This module is the harness adapter's single home for the agent CLI's command
line: the program, the pin, the isolation flag and the conversation flags.

The launch argv (:func:`build_claude_launch_argv`) and the executable an
Agent SDK run is pointed at (:func:`agent_cli_path`) name the same build: the
version ``claude_code.cli_version`` pins, else the installed ``claude``. The
SDK is never left to pick a binary of its own.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

from osprey.agent_runner.errors import AgentCliUnavailableError

logger = logging.getLogger(__name__)

#: The program an unpinned launch runs, found on PATH.
CLI_NAME = "claude"

_VERSION_RE = re.compile(r"\b(\d+\.\d+\.\d+(?:[-+.][0-9A-Za-z.-]+)?)\b")

#: The npm package a pinned launch runs through ``npx``, and the one an
#: operator is told to install.
_CLI_PACKAGE = "@anthropic-ai/claude-code"

#: What a ``claude_code.cli_version`` pin may be: one exact version,
#: ``MAJOR.MINOR.PATCH`` with an optional ``-prerelease``, matched in full.
#: The pin is written after ``<package>@`` in an ``npx`` or ``npm`` command
#: line, where npm also accepts an alias, a URL, a git address, a folder, a
#: file in the working directory, a range and a dist-tag. None of those names
#: one build, and a tag is nothing an installed build reports as its version,
#: so the pattern admits a version and nothing else.
_CLI_VERSION_RE = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?")

#: The MCP server config the build renders at the project root, and the only
#: one an agent launch loads.
RENDERED_MCP_CONFIG = ".mcp.json"

#: How long a CLI binary gets to answer ``--version``. It is a large
#: single-file executable and the probe runs at server startup and ahead of a
#: run, so it is bounded rather than allowed to hold either open.
_VERSION_PROBE_TIMEOUT_S = 15.0


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
        ValueError: From :func:`validate_cli_version`, for a ``cli_version``
            that is present but not one exact version (only checked when
            ``no_pin`` is ``False``).
    """
    cli_version = None if no_pin else pinned_cli_version(cc_config)
    if cli_version is None:
        base = [CLI_NAME]
    else:
        base = ["npx", "-y", f"{_CLI_PACKAGE}@{cli_version}"]
    return base + _SETTING_SOURCES_ARGS + _MCP_ISOLATION_ARGS


def validate_cli_version(value: object) -> str:
    """Return a ``claude_code.cli_version`` value as the pin it states.

    This is the one check of the pin. Every reader of the config value goes
    through it (directly, or through :func:`pinned_cli_version`) before the
    value reaches a command line, an image build argument or a file.

    Args:
        value: The configured value.

    Returns:
        The value itself, unchanged: nothing is stripped or normalised.

    Raises:
        ValueError: If the value is anything other than one exact version.
    """
    if not isinstance(value, str) or not _CLI_VERSION_RE.fullmatch(value):
        raise ValueError(
            f"claude_code.cli_version must be one exact version of {_CLI_PACKAGE}, "
            f'such as "2.1.146" — not a range, a tag, an alias, a URL, a git address, '
            f"a file or a path; got {value!r}."
        )
    return value


def pinned_cli_version(cc_config: Mapping[str, Any]) -> str | None:
    """Return the validated ``cli_version`` pin of a ``claude_code`` block.

    Args:
        cc_config: The ``claude_code`` block from ``config.yml`` (may be empty).

    Returns:
        The pin, or ``None`` when the block states none.

    Raises:
        ValueError: From :func:`validate_cli_version`, for a pin that is
            present but not one exact version.
    """
    value = cc_config.get("cli_version")
    return None if value is None else validate_cli_version(value)


def project_claude_code_config(project_dir: Path) -> dict[str, Any]:
    """Return the ``claude_code`` block of a project's ``config.yml``.

    Args:
        project_dir: The directory holding ``config.yml`` — a rendered project.

    Returns:
        The block, or ``{}`` when the file is absent or states no such mapping.
    """
    config_path = Path(project_dir) / "config.yml"
    if not config_path.is_file():
        return {}
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    block = config.get("claude_code") if isinstance(config, dict) else None
    return block if isinstance(block, dict) else {}


def _probe_cli_version(executable: str) -> str | None:
    """Run ``executable --version`` and return the version it reports.

    Returns:
        The version, or ``None`` when the executable cannot be run, exits
        non-zero or prints no recognisable version.

    Raises:
        subprocess.TimeoutExpired: When it does not answer within
            :data:`_VERSION_PROBE_TIMEOUT_S`. Kept apart from ``None``: a slow
            host says nothing about which version is installed.
    """
    try:
        completed = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=_VERSION_PROBE_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("Could not read the Claude Code version of %s: %s", executable, exc)
        return None
    if completed.returncode != 0:
        return None
    return parse_claude_version(completed.stdout)


def cli_version_of(executable: str) -> str | None:
    """Return the version *executable* reports, or ``None`` for any failure.

    For a line in a log. A caller that decides anything by the version uses
    :func:`agent_cli_path`, which does not read a slow answer as a missing one.
    """
    try:
        return _probe_cli_version(executable)
    except subprocess.TimeoutExpired:
        return None


def _locate_claude() -> str | None:
    """Return the absolute path of the installed ``claude``, or ``None``.

    The lookup the interactive launch uses: ``PATH`` as it stands at the call,
    then the user-local bin directories a non-login process leaves off it.
    """
    from osprey.utils.shell_resolver import resolve_shell_command

    try:
        return os.path.abspath(resolve_shell_command(CLI_NAME))
    except FileNotFoundError:
        return None


def agent_cli_path(cc_config: Mapping[str, Any]) -> str:
    """Return the absolute path of the Claude Code a headless run executes.

    The Agent SDK is always handed this path. Left to itself it runs a binary
    bundled inside its own package, which no configuration names. A headless
    run executes an installed ``claude`` and nothing else: nothing is
    downloaded, and no other executable is substituted for it.

    The binary is whichever ``claude`` the process's ``PATH`` yields when it
    is resolved. Nothing here vouches for that file, so the trust boundary for
    it is the container image or the operating-system user that owns that
    ``PATH`` and the project directory: whoever can change the ``PATH`` the
    process sees, write a directory on it, or replace the file chooses what
    runs.

    Args:
        cc_config: The ``claude_code`` block from ``config.yml`` (may be empty).

    Returns:
        With a ``cli_version`` pin: the ``claude`` on ``PATH``, once it has
        reported exactly that version. Without a pin: the installed
        ``claude``, whatever its version.

    Raises:
        ValueError: From :func:`validate_cli_version`, for a pin that is not
            one exact version.
        AgentCliUnavailableError: When no installed ``claude`` reports the
            pinned version, when its version cannot be read in time, or when
            an unpinned project has no ``claude`` at all. The message is
            written for the operator and names the ways out.
    """
    pin = pinned_cli_version(cc_config)
    installed = _locate_claude()
    if pin is None:
        if installed is None:
            raise AgentCliUnavailableError(
                f"No `{CLI_NAME}` was found on PATH, so this run has no Claude Code to execute. "
                f"Either install it:\n  npm install -g {_CLI_PACKAGE}\n"
                f"or pin a version by setting claude_code.cli_version under `config:` in "
                f"profile.yml, running `osprey build`, and installing that version."
            )
        return installed

    if installed is None:
        found = f"no `{CLI_NAME}` was found on PATH"
    else:
        try:
            reported = _probe_cli_version(installed)
        except subprocess.TimeoutExpired:
            raise AgentCliUnavailableError(
                f"`{installed} --version` did not answer within "
                f"{_VERSION_PROBE_TIMEOUT_S:.0f} s, so it could not be checked against the "
                f"Claude Code version this project pins ({pin}, claude_code.cli_version). "
                f"Nothing was started. Run that command by hand to see why it is slow, "
                f"then try again."
            ) from None
        if reported == pin:
            return installed
        found = (
            f"the installed `{CLI_NAME}` ({installed}) reports {reported}"
            if reported
            else f"the installed `{CLI_NAME}` ({installed}) did not report a version"
        )
    raise AgentCliUnavailableError(
        f"This project pins Claude Code {pin} (claude_code.cli_version), but {found}. "
        f"A headless run executes only an installed `{CLI_NAME}` that reports the pinned "
        f"version, so nothing was started. Either install that version:\n"
        f"  npm install -g {_CLI_PACKAGE}@{pin}\n"
        f"or change claude_code.cli_version under `config:` in profile.yml and run "
        f"`osprey build`."
    )


def project_agent_cli_path(project_dir: Path) -> str:
    """Return the Claude Code every headless run of *project_dir* executes.

    Resolved once per process and reused: the project's pin is read, the
    executable located and its version checked the first time a run needs it,
    and a later edit of ``config.yml`` or of the installed binary's location
    changes nothing until the process restarts. A refusal is not remembered, so a run started after the
    pinned version is installed goes ahead.

    Raises:
        ValueError: As :func:`agent_cli_path`.
        AgentCliUnavailableError: As :func:`agent_cli_path`.
    """
    return _project_agent_cli_path(os.path.abspath(project_dir))


@cache
def _project_agent_cli_path(project_dir: str) -> str:
    return agent_cli_path(project_claude_code_config(Path(project_dir)))


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
