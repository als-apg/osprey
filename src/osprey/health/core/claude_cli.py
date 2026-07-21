"""Core Claude Code CLI health categories.

Two categories share this module, split by cost class:

* ``claude_cli`` (poll): the cheap ``claude --version`` availability check
  (5s). Row ``claude_cli_version``; warns on a missing, timed-out or
  unparseable CLI, informational ``ok`` otherwise.
* ``claude_cli_pinned`` (on_demand, always registered): verifies a
  ``claude_code.cli_version`` pin by running
  ``npx -y @anthropic-ai/claude-code@<version> --version`` (60s first-run
  download budget). Row ``claude_cli_pinned``; emits a single ``skip`` row
  when no pin is configured, ``error`` on npx failure and ``warning`` on a
  version mismatch.

Both categories run their subprocesses via ``asyncio.create_subprocess_exec``
so the runner bounds them without blocking its event loop.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from osprey.health.models import CheckResult, Status

_CATEGORY_CLI = "claude_cli"
_CATEGORY_PINNED = "claude_cli_pinned"

_POLL_TIMEOUT_S = 5.0
_PINNED_TIMEOUT_S = 60.0


async def _run_version_command(argv: list[str], timeout_s: float) -> tuple[int | None, str, str]:
    """Run ``argv`` and return ``(returncode, stdout, stderr)``.

    Args:
        argv: Command and arguments to execute.
        timeout_s: Wall-clock budget; on expiry the child is killed and reaped.

    Returns:
        The exit code and decoded stdout/stderr.

    Raises:
        FileNotFoundError: If the executable is not on ``PATH``.
        TimeoutError: If the command exceeds ``timeout_s``.
    """
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except TimeoutError:
        proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()
        raise
    return (
        proc.returncode,
        stdout.decode(errors="replace"),
        stderr.decode(errors="replace"),
    )


def claude_cli(config: dict[str, Any] | None, context: Any = None) -> Any:
    """Build the poll-class ``claude_cli`` category callable.

    Args:
        config: Parsed config mapping (unused; the poll check needs no config).
        context: Unused; the check needs no connector.

    Returns:
        An async no-argument callable producing the category's rows.
    """

    async def _run() -> list[CheckResult]:
        return await _check_claude_cli()

    return _run


async def _check_claude_cli() -> list[CheckResult]:
    """Report the globally-installed ``claude`` CLI version (informational)."""
    from osprey.utils.claude_launcher import parse_claude_version

    try:
        returncode, stdout, stderr = await _run_version_command(
            ["claude", "--version"], _POLL_TIMEOUT_S
        )
    except FileNotFoundError:
        return [
            CheckResult(
                name="claude_cli_version",
                category=_CATEGORY_CLI,
                status=Status.WARNING,
                message="`claude` not found in PATH",
                details=(
                    "Install with `npm install -g @anthropic-ai/claude-code` or "
                    "pin via claude_code.cli_version in config.yml."
                ),
            )
        ]
    except TimeoutError:
        return [
            CheckResult(
                name="claude_cli_version",
                category=_CATEGORY_CLI,
                status=Status.WARNING,
                message="`claude --version` timed out (5s)",
            )
        ]

    detected = parse_claude_version(stdout) if returncode == 0 else None
    if detected:
        return [
            CheckResult(
                name="claude_cli_version",
                category=_CATEGORY_CLI,
                status=Status.OK,
                message=f"Claude Code CLI {detected}",
            )
        ]
    return [
        CheckResult(
            name="claude_cli_version",
            category=_CATEGORY_CLI,
            status=Status.WARNING,
            message="Could not parse `claude --version` output",
            details=f"stdout={stdout.strip()!r} stderr={stderr.strip()!r}",
        )
    ]


def claude_cli_pinned(config: dict[str, Any] | None, context: Any = None) -> Any:
    """Build the on_demand ``claude_cli_pinned`` category callable.

    Args:
        config: Parsed config mapping; ``claude_code.cli_version`` selects the pin.
        context: Unused; the check needs no connector.

    Returns:
        An async no-argument callable producing the category's rows.
    """
    cc_config = (config or {}).get("claude_code", {}) or {}
    pinned = cc_config.get("cli_version")

    async def _run() -> list[CheckResult]:
        return await _check_claude_cli_pinned(pinned)

    return _run


async def _check_claude_cli_pinned(pinned: str | None) -> list[CheckResult]:
    """Verify a pinned Claude Code CLI version via ``npx``."""
    from osprey.utils.claude_launcher import parse_claude_version

    if not pinned:
        return [
            CheckResult(
                name="claude_cli_pinned",
                category=_CATEGORY_PINNED,
                status=Status.SKIP,
                message="no cli_version pin configured",
            )
        ]

    argv = ["npx", "-y", f"@anthropic-ai/claude-code@{pinned}", "--version"]
    try:
        returncode, stdout, stderr = await _run_version_command(argv, _PINNED_TIMEOUT_S)
    except FileNotFoundError:
        return [
            CheckResult(
                name="claude_cli_pinned",
                category=_CATEGORY_PINNED,
                status=Status.ERROR,
                message="npx not found in PATH",
                details="Install Node.js (which ships npx) to use claude_code.cli_version pinning.",
            )
        ]
    except TimeoutError:
        return [
            CheckResult(
                name="claude_cli_pinned",
                category=_CATEGORY_PINNED,
                status=Status.ERROR,
                message=f"npx timed out fetching @anthropic-ai/claude-code@{pinned} (60s)",
                details="Check network connectivity to the npm registry.",
            )
        ]

    if returncode != 0:
        return [
            CheckResult(
                name="claude_cli_pinned",
                category=_CATEGORY_PINNED,
                status=Status.ERROR,
                message=f"npx failed for @anthropic-ai/claude-code@{pinned}",
                details=stderr.strip() or "no stderr output",
            )
        ]

    detected = parse_claude_version(stdout)
    if detected == pinned:
        return [
            CheckResult(
                name="claude_cli_pinned",
                category=_CATEGORY_PINNED,
                status=Status.OK,
                message=f"Pinned Claude Code CLI {pinned}",
            )
        ]
    shown = detected or "unknown"
    return [
        CheckResult(
            name="claude_cli_pinned",
            category=_CATEGORY_PINNED,
            status=Status.WARNING,
            message=f"Pinned {pinned} but npx reported {shown}",
            details=f"raw --version output: {stdout.strip()!r}",
        )
    ]


__all__ = ["claude_cli", "claude_cli_pinned"]
