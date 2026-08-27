"""Core ``systemd_unit`` health category.

Reports whether the ``systemd --user`` manager can actually *see* the boot unit
``osprey scaffold systemd`` wrote, which is a different question from whether
the unit file exists.

The failure this row exists for: when ``$HOME`` lives on NFS/autofs, a lingering
``systemd --user`` manager is started at boot — before the home is mounted. That
manager resolves its unit search path exactly once, finds an empty (or absent)
``~/.config/systemd/user``, and never looks again. The scaffolded unit is then
installed on disk and perfectly well-formed, yet ``systemctl --user`` answers
``not-found`` for it, and the deployment silently fails to come back after every
reboot. Read from the outside it looks like a broken unit; it is really a
mount-ordering problem, and the fix is a ``systemctl --user daemon-reload``
(plus, durably, making the manager start after the home is mounted). Nothing
else in the health suite would notice: the file is on disk, the build is clean,
and the only witness is the manager's own answer.

The category emits exactly one row, ``systemd_unit``, from a deliberate
trigger/report split:

* the **repo** copy (``<project>/osprey.service``) decides whether this
  deployment uses the systemd boot path at all. No repo unit means the operator
  never asked for one, so the row is ``skip`` — not a warning about a missing
  file nobody wanted;
* the **installed** copy plus the manager's ``LoadState`` decide the verdict:
  ``warning`` when the unit was scaffolded but never installed, ``error`` when
  it is installed and the manager still reports ``not-found``, ``ok`` for any
  other ``LoadState`` (``loaded``, ``masked``, ``error``, ``bad-setting`` — all
  states in which the manager can see the unit, which is what this row asks).

Keying off only one of the two produces a false alarm in one direction or the
other: repo-only would warn about every host that deliberately runs OSPREY some
other way, and install-only would stay silent on exactly the host where the
manager has gone blind.

Everything that is not a verdict degrades to ``skip``: no ``systemctl`` on
``PATH`` (macOS, containers), a ``systemctl --user`` that cannot reach a bus, a
timed-out query, or a home directory that cannot be resolved at all. A host with
no user manager is not an unhealthy host.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from osprey.health.models import CheckResult, Status

if TYPE_CHECKING:
    from collections.abc import Mapping

    from osprey.health.core import CategoryCallable
    from osprey.health.runtime import HealthRuntime

CATEGORY = "systemd_unit"

#: Wall-clock budget for the ``systemctl --user show`` query. Its own constant
#: rather than a share of the poll budget: the query is a single D-Bus
#: round-trip against a local manager, so anything slower than this is a wedged
#: or unreachable bus, and waiting longer only delays the ``skip``.
_SYSTEMCTL_TIMEOUT_S = 5.0

#: The boot unit ``osprey scaffold systemd`` writes, spelled as a literal rather
#: than imported from :mod:`~osprey.cli.deploy_scaffold_templates` (which owns
#: the name as ``SYSTEMD_UNIT_NAME``). Importing it would pull the CLI package
#: into the web and MCP health surfaces, which import this module. This is the
#: same choice :mod:`~osprey.cli.profile_conventions` made for its own copy of
#: the name, and the same answer: the two spellings are pinned to each other by
#: a test rather than by an import.
_UNIT_NAME = "osprey.service"

_LOAD_STATE_NOT_FOUND = "not-found"


def systemd_unit(
    config: Mapping[str, Any] | None = None,
    context: HealthRuntime | None = None,
    *,
    cwd: Path | None = None,
) -> CategoryCallable:
    """Build the ``systemd_unit`` category callable.

    Args:
        config: Loaded config mapping. Unused — the check reads the deployment
            repo and the user manager, not the rendered config.
        context: Health runtime. Unused — no control-system connector is needed.
        cwd: Deployment repo root, where the scaffolded ``osprey.service`` lives.
            Defaults to :func:`Path.cwd`, resolved when the callable runs so the
            CLI can thread a ``--project-path`` override; ``build_records``
            passes the project path here just as it does for ``file_system``.

    Returns:
        A no-argument async callable returning the category's single row.
    """
    base_dir = cwd

    async def _run() -> list[CheckResult]:
        return await _check_systemd_unit(base_dir or Path.cwd())

    return _run


async def _run_systemctl(argv: list[str], timeout_s: float) -> tuple[int | None, str, str]:
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


def _user_unit_dir() -> Path:
    """Resolve the directory the user manager loads units from.

    ``$XDG_CONFIG_HOME/systemd/user`` when that variable is set and non-empty,
    otherwise ``~/.config/systemd/user``. Honouring ``XDG_CONFIG_HOME`` matters:
    the user manager reads it when set, so hardcoding ``~/.config`` would report
    a properly installed unit as missing on every host that sets it.

    Raises:
        RuntimeError: If ``Path.home()`` cannot resolve a home directory.
        OSError: If the environment or path expansion fails.
    """
    xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "systemd" / "user"


def _row(status: Status, message: str, details: str = "", value: str = "") -> CheckResult:
    """Build this category's single row."""
    return CheckResult(
        name=CATEGORY,
        category=CATEGORY,
        status=status,
        message=message,
        value=value,
        details=details,
    )


async def _check_systemd_unit(repo_root: Path) -> list[CheckResult]:
    """Produce the one ``systemd_unit`` row for the deployment at ``repo_root``."""
    repo_unit = repo_root / _UNIT_NAME
    if not repo_unit.exists():
        return [
            _row(
                Status.SKIP,
                f"no scaffolded {_UNIT_NAME} in this deployment",
                details=(
                    f"Run `osprey scaffold systemd` if this deployment should "
                    f"start at boot; expected {repo_unit}."
                ),
            )
        ]

    if shutil.which("systemctl") is None:
        return [_row(Status.SKIP, "systemctl not found in PATH")]

    try:
        installed_unit = _user_unit_dir() / _UNIT_NAME
        installed = installed_unit.exists()
    except (RuntimeError, OSError) as exc:
        # Path.home() raises RuntimeError when neither $HOME nor a passwd entry
        # resolves (containers, some CI). Nothing to report against.
        return [
            _row(
                Status.SKIP,
                "cannot resolve the user unit directory",
                details=f"{type(exc).__name__}: {exc}",
            )
        ]

    if not installed:
        return [
            _row(
                Status.WARNING,
                f"{_UNIT_NAME} is scaffolded but not installed",
                details=(
                    f"Copy or symlink {repo_unit} to {installed_unit}, then run "
                    f"`systemctl --user daemon-reload` and "
                    f"`systemctl --user enable --now {_UNIT_NAME}`."
                ),
            )
        ]

    argv = ["systemctl", "--user", "show", "-p", "LoadState", "--value", _UNIT_NAME]
    try:
        returncode, stdout, stderr = await _run_systemctl(argv, _SYSTEMCTL_TIMEOUT_S)
    except FileNotFoundError:
        return [_row(Status.SKIP, "systemctl not found in PATH")]
    except TimeoutError:
        return [
            _row(
                Status.SKIP,
                f"`systemctl --user show` timed out ({_SYSTEMCTL_TIMEOUT_S:.0f}s)",
                details="The user manager did not answer; not treated as a fault.",
            )
        ]

    if returncode != 0:
        return [
            _row(
                Status.SKIP,
                "no reachable systemd --user manager",
                details=stderr.strip() or "systemctl --user exited non-zero with no stderr.",
            )
        ]

    load_state = stdout.strip() or "unknown"
    if load_state == _LOAD_STATE_NOT_FOUND:
        return [
            _row(
                Status.ERROR,
                f"{_UNIT_NAME} is installed but the user manager reports not-found",
                details=(
                    f"{installed_unit} exists, yet `systemctl --user` cannot see it. "
                    "This is the network-home failure: when $HOME is on NFS/autofs, "
                    "a systemd --user manager started before the home was mounted "
                    "resolved its unit search path once and never looked again, so "
                    "the deployment does not come back after a reboot. Run "
                    "`systemctl --user daemon-reload` to recover now, and order the "
                    "user manager after the home mount so it does not recur."
                ),
            )
        ]

    return [
        _row(
            Status.OK,
            f"{_UNIT_NAME} is installed and visible to the user manager",
            value=load_state,
        )
    ]


__all__ = ["CATEGORY", "systemd_unit"]
