"""Core Claude Code CLI health categories.

Two categories share this module, split by cost class:

* ``claude_cli`` (poll): the cheap ``claude --version`` availability check
  (5s). Row ``claude_cli_version``; warns on a missing, timed-out or
  unparseable CLI, informational ``ok`` otherwise.
* ``claude_cli_pinned`` (on_demand, always registered): verifies a
  ``claude_code.cli_version`` pin by asking the launcher for the executable a
  headless run would be pointed at, so the row proves the path runs take and
  downloads nothing. Row ``claude_cli_pinned``; emits a single ``skip`` row
  when no pin is configured, ``ok`` naming the executable when an installed
  ``claude`` reports the pin, and ``error`` carrying the launcher's refusal
  otherwise — a pin that is not an exact version included.

Neither category blocks the runner's event loop: the poll check runs its
subprocess via ``asyncio.create_subprocess_exec``, the pinned check hands the
launcher's blocking probe to a worker thread.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from osprey.health.models import CheckResult, Status

_CATEGORY_CLI = "claude_cli"
_CATEGORY_PINNED = "claude_cli_pinned"

_POLL_TIMEOUT_S = 5.0


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


def claude_cli(
    config: dict[str, Any] | None,  # noqa: ARG001 - health category factory signature; most categories read the config state
    context: Any = None,  # noqa: ARG001 - health category factory signature; categories that probe a runtime read the context
) -> Any:
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
    from osprey.agent_runner.launcher import parse_claude_version

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
                    "pin by setting claude_code.cli_version under `config:` in "
                    "profile.yml and running `osprey build`."
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


def claude_cli_pinned(
    config: dict[str, Any] | None,
    context: Any = None,  # noqa: ARG001 - health category factory signature; categories that probe a runtime read the context
) -> Any:
    """Build the on_demand ``claude_cli_pinned`` category callable.

    Args:
        config: Parsed config mapping; ``claude_code.cli_version`` selects the pin.
        context: Unused; the check needs no connector.

    Returns:
        An async no-argument callable producing the category's rows.
    """
    cc_config = (config or {}).get("claude_code", {}) or {}

    async def _run() -> list[CheckResult]:
        return await _check_claude_cli_pinned(cc_config)

    return _run


async def _check_claude_cli_pinned(cc_config: dict[str, Any]) -> list[CheckResult]:
    """Verify that the pinned Claude Code is the one a headless run would execute."""
    from osprey.agent_runner import launcher
    from osprey.agent_runner.errors import AgentCliUnavailableError

    try:
        pinned = launcher.pinned_cli_version(cc_config)
    except ValueError as exc:
        return [_pinned_row(Status.ERROR, str(exc))]
    if pinned is None:
        return [_pinned_row(Status.SKIP, "no cli_version pin configured")]

    try:
        # A fresh answer on every run of the check, not the one a serving
        # process settled at its start. The probe blocks, so off the loop.
        path = await asyncio.to_thread(launcher.agent_cli_path, cc_config)
    except AgentCliUnavailableError as exc:
        return [
            _pinned_row(
                Status.ERROR,
                f"Pinned Claude Code CLI {pinned} is not available to headless runs",
                details=str(exc),
            )
        ]
    return [_pinned_row(Status.OK, f"Pinned Claude Code CLI {pinned} ({path})")]


def _pinned_row(status: Status, message: str, *, details: str = "") -> CheckResult:
    return CheckResult(
        name="claude_cli_pinned",
        category=_CATEGORY_PINNED,
        status=status,
        message=message,
        details=details,
    )


__all__ = ["claude_cli", "claude_cli_pinned"]
