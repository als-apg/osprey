"""Capture a child process's output to a spool file instead of the terminal.

The lifecycle verbs report progress as a handful of phase lines (see
:mod:`osprey.cli.phase_reporter`). That reporting is only readable if the
subprocesses underneath it — image builds, compose invocations — stop writing
their own thousands of lines to the same terminal. :func:`run_captured` is the
one seam that makes that true: the child's stdout and stderr go to
``<repo>/var/logs/<spool_name>-<timestamp>.log`` as a single merged stream, and
the file is named on the failure path so nothing is lost.

Under the global ``--verbose`` flag the helper is a pass-through: the child
inherits this process's stdio and streams straight to the terminal, with no
redirection kwargs at all. Verbosity is read from the reporter's module-level
singleton rather than threaded through every caller, so the two surfaces cannot
disagree about which mode a run is in.

The default reporter is a quiet ``NullReporter(verbose=False)``, so calls made
before a verb installs a reporter still capture. That is deliberate: capturing
is the safe default, and the spool path is reported on failure either way.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from osprey.cli.phase_reporter import current_reporter, is_verbose
from osprey.deployment.errors import CapturedProcessError
from osprey.utils.logger import get_logger

logger = get_logger("deployment.capture")

#: Spool directory, relative to the deployment repo root.
SPOOL_DIR = Path("var") / "logs"

#: How many spool files survive a prune. Older ones are removed at entry, so a
#: long-lived deployment repo keeps the recent runs without growing unbounded.
SPOOL_RETENTION = 20


class CapturedProcess(subprocess.CompletedProcess[bytes]):
    """A finished captured run, naming the file its output went to.

    A plain :class:`subprocess.CompletedProcess` plus ``spool_path``: the spool
    this run's output was written to, or ``None`` in verbose mode where it went
    straight to the terminal. Declaring it as a type rather than setting the
    attribute dynamically is what lets a caller read ``result.spool_path``
    without a type checker rejecting it.

    ``stdout``/``stderr`` stay ``None`` for a captured run — that output is in
    the spool file, not in memory — but the constructor accepts them so this
    remains substitutable for its base class.
    """

    def __init__(
        self,
        args: Sequence[str] | str,
        returncode: int,
        stdout: bytes | None = None,
        stderr: bytes | None = None,
        *,
        spool_path: Path | None = None,
    ) -> None:
        super().__init__(args, returncode, stdout, stderr)
        self.spool_path = spool_path


def _prune_spools(spool_dir: Path, keep: int = SPOOL_RETENTION) -> None:
    """Delete all but the newest ``keep`` spool files, best-effort.

    Retention is a housekeeping nicety; a directory that cannot be read or an
    entry that cannot be removed must never take down the deploy that was about
    to run, so every failure here is a debug line and nothing more.
    """
    try:
        existing = sorted(spool_dir.glob("*.log"), key=lambda p: p.stat().st_mtime)
    except OSError as exc:  # pragma: no cover - depends on filesystem state
        logger.debug("Could not list spool directory %s: %s", spool_dir, exc)
        return
    for stale in existing[: max(0, len(existing) - keep)]:
        try:
            stale.unlink()
        except OSError as exc:  # pragma: no cover - depends on filesystem state
            logger.debug("Could not prune spool %s: %s", stale, exc)


def _spool_path(spool_dir: Path, spool_name: str) -> Path:
    """Return an unused spool path for ``spool_name`` in ``spool_dir``.

    The timestamp sorts lexicographically, which is what makes the directory
    listing readable. Two runs can still land in the same second — a compose
    ``rm`` followed immediately by a build — so the name takes a numeric suffix
    until it is free rather than overwriting the earlier run's output.
    """
    stamp = time.strftime("%Y%m%d-%H%M%S")
    candidate = spool_dir / f"{spool_name}-{stamp}.log"
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = spool_dir / f"{spool_name}-{stamp}-{suffix}.log"
    return candidate


def run_captured(
    cmd: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | str | None = None,
    spool_name: str,
    repo_root: Path | str | None = None,
    check: bool = True,
    on_line: Callable[[str], None] | None = None,
) -> CapturedProcess:
    """Run ``cmd``, spooling its merged output to ``<repo>/var/logs/``.

    :param cmd: The argv to run. Passed through unchanged in both modes.
    :param env: Environment for the child, as for :func:`subprocess.run`.
    :param cwd: Working directory for the child, as for :func:`subprocess.run`.
        Unrelated to ``repo_root``: a scaffolded script is run from the project
        root so its own relative paths resolve, while its output still spools
        under the deployment repo. Forwarded in both modes, so a run behaves
        the same way under ``--verbose``.
    :param spool_name: Descriptive stem for the spool file (``"build-worker"``).
    :param repo_root: Deployment repo the ``var/logs/`` directory hangs off.
        Callers already hold this (``resolve_repo_root(config)``); ``None``
        means the current directory, the same default the compose ``--env-file``
        helpers use. No repo discovery happens here.
    :param check: Raise :class:`CapturedProcessError` on a non-zero exit.
    :param on_line: Watch the child's merged stream, line by line, as decoded
        text without its newline. The spool stays byte-for-byte complete either
        way — the callback is a tee, not a diversion — and it is cosmetic by
        contract: an exception it raises is logged at debug and changes nothing
        about the run's outcome. Ignored under ``--verbose``, where the child's
        own lines are already on the terminal and derived progress would only
        duplicate them.
    :returns: A :class:`CapturedProcess`, whose ``spool_path`` is the file this
        run's output went to, or ``None`` in verbose mode.
        ``stdout``/``stderr`` are ``None`` — the output is in that file (or on
        the terminal). A ``check=False`` caller that wants to point at the
        output reads it from there; the reporter's copy only covers runs made
        while a phase is open.
    :raises CapturedProcessError: ``check`` is set and the child exited
        non-zero. The exception carries the spool path.

    Under ``--verbose`` the child inherits this process's stdio: same argv, no
    redirection at all, so long-running builds stream live.
    """
    if is_verbose():
        # Spool-less in both senses: no file was written, and the result says so
        # explicitly, so a caller can read ``spool_path`` unconditionally.
        completed = subprocess.run(cmd, env=env, cwd=cwd)
        if check and completed.returncode != 0:
            raise CapturedProcessError(cmd, completed.returncode)
        return CapturedProcess(cmd, completed.returncode)

    spool_dir = Path(repo_root or Path.cwd()) / SPOOL_DIR
    spool_dir.mkdir(parents=True, exist_ok=True)
    _prune_spools(spool_dir)
    spool_path = _spool_path(spool_dir, spool_name)

    logger.debug("Capturing %s output to %s", spool_name, spool_path)
    with open(spool_path, "wb") as spool:
        # The reporter keeps the path for the whole phase, not just this call:
        # an interrupt during the child needs to name the partial output.
        current_reporter().note_spool(spool_path)
        if on_line is None:
            # stderr into the same handle, not a second file — a build's errors
            # are only legible interleaved with the step they interrupted.
            completed = subprocess.run(
                cmd, env=env, cwd=cwd, stdout=spool, stderr=subprocess.STDOUT
            )
        else:
            completed = _run_teeing(cmd, env=env, cwd=cwd, spool=spool, on_line=on_line)

    if check and completed.returncode != 0:
        raise CapturedProcessError(cmd, completed.returncode, spool_path)

    # The spool goes on the result, not only on the reporter: an advisory caller
    # that does not raise still has to be able to name the file its output went
    # to, and it may be running with no phase open at all.
    return CapturedProcess(cmd, completed.returncode, spool_path=spool_path)


def _run_teeing(
    cmd: Sequence[str],
    *,
    env: Mapping[str, str] | None,
    cwd: Path | str | None,
    spool,
    on_line: Callable[[str], None],
) -> subprocess.CompletedProcess:
    """The watched variant of the capture: every line to the spool AND the callback.

    The direct-to-file ``subprocess.run`` cannot offer a per-line hook, so this
    path reads the child's merged stream through a pipe and writes it on. The
    bytes reaching the spool are the child's own; only the callback gets a
    decoded copy, with U+FFFD standing in for anything that is not UTF-8 —
    a build log's occasional garbage must not take down the build.

    The callback is fenced per line rather than per run: one line the parser
    chokes on costs that line's step at debug level, not every step after it.
    """
    with subprocess.Popen(
        cmd, env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    ) as process:
        assert process.stdout is not None  # PIPE above guarantees it
        for raw in process.stdout:
            spool.write(raw)
            try:
                on_line(raw.decode("utf-8", errors="replace").rstrip("\r\n"))
            except Exception as exc:
                logger.debug("on_line callback failed on %r: %s", raw[:200], exc)
        returncode = process.wait()
    return subprocess.CompletedProcess(list(cmd), returncode)


def diagnose_captured_failure(exc: BaseException) -> str | None:
    """Translate a captured child's failure into a remedy, or ``None``.

    A :class:`~osprey.deployment.errors.CapturedProcessError` names the file its
    child's output went to, and some of those failures have a cause OSPREY can
    recognize and answer in words the raw text never uses. This is the one seam
    that reads that spool and asks the diagnosers, so every verb that runs a
    build gets the same translation rather than each growing its own.

    Total by construction. No spool (a ``--verbose`` run, where the output went
    to the terminal instead, or an exception of some other type), a spool that
    cannot be read, or output that nothing recognizes all yield ``None`` — and
    the caller then renders the failure exactly as it did before.

    :param exc: The exception a verb is about to fail on.
    :returns: Operator-facing remedy text, or ``None`` when there is nothing
        honest to add.
    """
    from osprey.deployment.runtime_helper import diagnose_build_failure

    spool_path = getattr(exc, "spool_path", None)
    if spool_path is None:
        return None
    try:
        output = Path(spool_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return diagnose_build_failure(output)


__all__ = [
    "SPOOL_DIR",
    "SPOOL_RETENTION",
    "CapturedProcess",
    "diagnose_captured_failure",
    "run_captured",
]
