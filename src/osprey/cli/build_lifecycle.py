"""Build lifecycle-phase execution helpers.

Runs the ``pre_build`` / ``post_build`` / ``validate`` command phases declared
in a build profile, with the shared subprocess environment (the project's
merged env chain, project venv on ``PATH``, ``_mcp_servers`` on ``PYTHONPATH``), streaming
and quiet output modes, timeout handling, and the JUnit test-results summary.

Output here splits two ways. What a step's run looks like from outside -- the
phase header, the line naming a streamed step as it starts -- is phase record,
so it goes through the installed reporter and is silenced under ``--verbose``
with the rest of the record. What a step actually reported -- its child's
lines, its pass or failure, the tests it ran -- is the operator's answer, so it
goes through :mod:`osprey.cli.output` and prints whatever reporter is
installed. Everything else the module knows is a diagnostic and stays on the
logger.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import IO, Any

from rich.text import Text

from osprey.errors import BuildProfileError
from osprey.utils.dotenv import chain_files, merge_chain
from osprey.utils.logger import get_logger

from .output import fail, report, warn
from .phase_reporter import current_reporter
from .styles import Styles, data_table

logger = get_logger("build")


_SHELL_METACHARACTERS = ("|", "&&", "||", "$(", "`")

_EXIT_GRACE_SECONDS = 5.0
"""Seconds a streamed step's child gets to exit once its stdout has drained.

Some test frameworks (a pyepics CA context, for one) keep background threads
alive that stop the process exiting promptly after its last line.
"""

_DRAIN_SHUTDOWN_SECONDS = 5.0
"""Seconds the drain thread gets to notice its step is over and end."""

_DRAIN_THREAD_NAME = "osprey-lifecycle-drain"


def _format_junit_summary(xml_path: Path) -> None:
    """Parse JUnit XML and print a summary table of the test results.

    Report content, not phase record: the step ran these tests because the
    operator asked it to, so the answer prints whichever reporter is installed.
    It goes through the reporter's own console rather than a private one, which
    is what puts it ABOVE a mounted live region instead of inside it, and takes
    the CLI's one data-table look from :func:`~osprey.cli.styles.data_table`.

    A missing or unparseable file is not an error here: most steps produce no
    test results at all, and a step that failed on its way to writing them has
    already reported why.

    Args:
        xml_path: The JUnit XML a step may have left behind.
    """
    import xml.etree.ElementTree as ET

    if not xml_path.exists():
        return

    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return

    root = tree.getroot()

    table = data_table(title="Integration Test Results")
    table.add_column("Test", style=Styles.BOLD, no_wrap=True)
    table.add_column("Status", justify="center", width=6)
    table.add_column("Time", justify="right", width=8)

    for testsuite in root.iter("testsuite"):
        for testcase in testsuite.iter("testcase"):
            name = testcase.get("name", "unknown")
            time_s = testcase.get("time", "0")

            failure = testcase.find("failure")
            error = testcase.find("error")
            skipped = testcase.find("skipped")

            # Pre-styled cells rather than markup tags: the echo path prints
            # with markup off, so a `[red]` tag would reach the operator as
            # four literal characters.
            if failure is not None or error is not None:
                status = Text("✗", style=Styles.ERROR)
            elif skipped is not None:
                status = Text("skip", style=Styles.DIM)
            else:
                status = Text("✓", style=Styles.SUCCESS)

            table.add_row(name, status, f"{float(time_s):.2f}s")

    if table.row_count > 0:
        current_reporter().out().print(table)


def _drain_stdout(stdout: IO[str], stop: threading.Event) -> None:
    """Emit a streaming step's stdout, line by line, until ``stop`` is set.

    Reads with ``readline`` rather than iterating the file so the loop reaches
    the stop check once per line instead of once per read-ahead buffer, and
    checks the flag after the read as well as before it: the read is where this
    thread spends its time, so a line that only arrives once the step is over
    must be dropped rather than emitted over whatever the CLI draws next.

    Lines go through the installed reporter rather than ``print`` so they take
    the reporter's ordering like every other CLI line. A raw write from this
    thread would land inside a mounted live region instead of above it.

    They go through :meth:`~osprey.cli.phase_reporter.PhaseReporter.echo` and
    not ``emit``: this is the step's own output, which the operator asked for
    with ``--stream``, and the emit path is silenced under global ``--verbose``.
    ``osprey --verbose build --stream`` would then be the one invocation asking
    hardest for the child's lines and the one showing none of them. ``echo``
    gives the same console ordering without the silencing.

    Args:
        stdout: The child's piped stdout, in text mode.
        stop: Set by :func:`_stop_drain` once the step's result is decided.
    """
    try:
        while not stop.is_set():
            line = stdout.readline()
            if not line:
                break
            if stop.is_set():
                break
            current_reporter().echo(f"    {line.rstrip()}")
    except (ValueError, OSError):
        # The pipe was closed under the read. The step is long over by then and
        # there is nobody left to tell.
        return


def _stop_drain(reader: threading.Thread, stop: threading.Event, stdout: IO[str]) -> None:
    """Retire a finished step's drain thread before its result is reported.

    The ordering here is load-bearing, and not the obvious one. Closing
    ``stdout`` to unblock the reader deadlocks the caller: a thread blocked in
    ``readline`` holds the buffered-IO lock, so ``close()`` from another thread
    waits for exactly the read it is trying to interrupt -- for as long as the
    child holds the pipe open. The child is therefore already dead by the time
    this is called (the caller's exit grace kills it), which drops the write end
    and lets the blocked ``readline`` return EOF; only then is the close free.

    The flag alone would not do either, since a reader blocked in a read that
    never returns never sees it. Together they bound the thread's life at
    ``_DRAIN_SHUTDOWN_SECONDS`` and, whether or not it ever ends, guarantee it
    emits nothing once its step has been reported.

    Args:
        reader: The drain thread for the step that just finished.
        stop: The stop flag that thread is watching.
        stdout: The child's piped stdout.
    """
    stop.set()
    reader.join(timeout=_DRAIN_SHUTDOWN_SECONDS)
    if reader.is_alive():
        # Still blocked on a pipe something outlived the child holding open --
        # a grandchild that inherited the write end. Leaving the descriptor to
        # the daemon thread is the honest trade: closing it here would hang the
        # CLI on the lock that thread holds, and with the flag set the thread is
        # already silent.
        logger.debug("Lifecycle drain thread outlived its step; left silenced")
        return
    try:
        stdout.close()
    except OSError:  # pragma: no cover - the pipe is at EOF by this point
        pass


def _run_lifecycle_phase(
    phase_name: str,
    steps: list[Any],
    default_cwd: Path,
    project_path: Path,
    *,
    abort_on_failure: bool = True,
    stream: bool = False,
) -> None:
    """Run lifecycle commands for a build phase.

    Args:
        phase_name: Phase name for display (pre_build, post_build, validate).
        steps: List of LifecycleStep objects.
        default_cwd: Default working directory for steps without explicit cwd.
        project_path: Project root path for {project_root} substitution.
        abort_on_failure: If True, raise BuildProfileError on failure.
            If False, warn and continue (used for validate phase).
        stream: If True, stream stdout/stderr in real-time instead of capturing.
    """
    # Auto-inject the project's env chain into the subprocess environment.
    # `merge_chain` resolves `.env.shared` under `.env` first, so a key both
    # files set arrives with the host-local value — the same local-wins
    # precedence the launch paths apply.
    env_files = chain_files(project_path)
    dotenv_vars = merge_chain(project_path)
    sub_env = {**os.environ, **dotenv_vars}
    if env_files:
        logger.info(
            "Loaded %d vars from %s into lifecycle environment",
            len(dotenv_vars),
            ", ".join(str(path) for path in env_files),
        )

    # Prepend project venv to PATH so `python` resolves to the project's
    # Python (with profile deps) rather than OSPREY's Python.
    venv_bin = project_path / ".venv" / "bin"
    if venv_bin.is_dir():
        sub_env["PATH"] = f"{venv_bin}{os.pathsep}{sub_env.get('PATH', '')}"
        logger.info("Prepended project venv to lifecycle PATH: %s", venv_bin)

    # Prepend _mcp_servers to PYTHONPATH so lifecycle commands can
    # ``import integration_tests`` (and other MCP server packages)
    # without manual PYTHONPATH wrappers in profile YAML.
    mcp_servers_dir = project_path / "_mcp_servers"
    if mcp_servers_dir.is_dir():
        existing = sub_env.get("PYTHONPATH", "")
        sub_env["PYTHONPATH"] = (
            f"{mcp_servers_dir}{os.pathsep}{existing}" if existing else str(mcp_servers_dir)
        )
        logger.info("Prepended _mcp_servers to lifecycle PYTHONPATH: %s", mcp_servers_dir)

    # Phase record: what the build is doing right now, in the same voice as the
    # phases around it, and silenced with the rest of the record under
    # `--verbose` (where the logger below is carrying the detail instead).
    current_reporter().emit(f"  Running {phase_name} commands...")
    for step in steps:
        cmd_str = step.run.replace("{project_root}", str(project_path))

        # Resolve cwd
        if step.cwd:
            cwd_str = step.cwd.replace("{project_root}", str(project_path))
            cwd = (default_cwd / cwd_str).resolve()
        else:
            cwd = default_cwd

        # Detect shell metacharacters
        use_shell = any(meta in cmd_str for meta in _SHELL_METACHARACTERS)

        t0 = time.monotonic()
        try:
            cmd = cmd_str if use_shell else shlex.split(cmd_str)

            if stream or step.stream:
                # Stream mode: show output in real-time, prefix with step name.
                # Uses a threaded reader so proc.wait(timeout=...) can enforce
                # the timeout even when the subprocess stalls mid-output.
                # Also phase record: it names the step whose lines are about to
                # arrive, and says nothing the lines themselves will not.
                current_reporter().emit(f"  > {step.name}")
                proc = subprocess.Popen(
                    cmd,
                    shell=use_shell,
                    cwd=cwd,
                    env=sub_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                assert proc.stdout is not None  # noqa: S101

                stop_drain = threading.Event()
                reader = threading.Thread(
                    target=_drain_stdout,
                    args=(proc.stdout, stop_drain),
                    name=_DRAIN_THREAD_NAME,
                    daemon=True,
                )
                reader.start()
                # Wait for stdout to drain (tests finished) with the full
                # timeout, then give the process a short grace period to exit.
                reader.join(timeout=step.timeout)
                try:
                    proc.wait(timeout=_EXIT_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                # The step is decided: retire its reader before the result line
                # is printed, so nothing of this step can print after it.
                _stop_drain(reader, stop_drain, proc.stdout)
                elapsed = time.monotonic() - t0
                if proc.returncode != 0:
                    msg = f"Lifecycle {phase_name} step '{step.name}' failed (exit {proc.returncode}, {elapsed:.1f}s)"
                    if abort_on_failure:
                        # No cause under the summary: the child's own output is
                        # already on the screen above it, streamed line by line.
                        fail(msg)
                        _format_junit_summary(project_path / "check_results.xml")
                        raise BuildProfileError(msg)
                    else:
                        warn(msg)
                else:
                    report(f"  ✓ {step.name} ({elapsed:.1f}s)", style=Styles.SUCCESS)
                # Show JUnit summary if test results were produced
                _format_junit_summary(project_path / "check_results.xml")
            else:
                # Quiet mode: capture output, show one-line summary
                result = subprocess.run(
                    cmd,
                    shell=use_shell,
                    cwd=cwd,
                    env=sub_env,
                    capture_output=True,
                    text=True,
                    timeout=step.timeout,
                )
                elapsed = time.monotonic() - t0

                if result.returncode != 0:
                    output = (result.stdout + result.stderr).strip()
                    headline = f"Lifecycle {phase_name} step '{step.name}' failed (exit {result.returncode}, {elapsed:.1f}s)"
                    msg = f"{headline}:\n{output}" if output else headline
                    if abort_on_failure:
                        # The captured output is the cause, printed under the
                        # summary rather than run into it: nobody streamed it,
                        # so this is the operator's only sight of it.
                        fail(headline, output)
                        _format_junit_summary(project_path / "check_results.xml")
                        raise BuildProfileError(msg)
                    else:
                        warn(headline, output)
                else:
                    success_msg = f"  ✓ {step.name} ({elapsed:.1f}s)"
                    output = (result.stdout + result.stderr).strip()
                    if output:
                        summary = output.rstrip().rsplit("\n", 1)[-1].strip()
                        if summary:
                            success_msg += f": {summary}"
                    report(success_msg, style=Styles.SUCCESS)
                # Show JUnit summary if test results were produced
                _format_junit_summary(project_path / "check_results.xml")

        except subprocess.TimeoutExpired as e:
            elapsed = time.monotonic() - t0
            headline = f"Lifecycle {phase_name} step '{step.name}' timed out ({elapsed:.0f}s)"
            msg = headline
            # Show partial output captured before timeout (quiet mode only;
            # stream mode already printed output in real-time).
            _out = (
                e.stdout.decode(errors="replace")
                if isinstance(e.stdout, bytes)
                else (e.stdout or "")
            )
            _err = (
                e.stderr.decode(errors="replace")
                if isinstance(e.stderr, bytes)
                else (e.stderr or "")
            )
            partial = _out + _err
            cause = ""
            if partial.strip():
                tail = "\n".join(partial.strip().splitlines()[-20:])
                cause = f"Last output:\n{tail}"
                msg += f"\n  {cause}"
            if abort_on_failure:
                fail(headline, cause)
                _format_junit_summary(project_path / "check_results.xml")
                raise BuildProfileError(msg) from None
            else:
                warn(headline, cause)
            _format_junit_summary(project_path / "check_results.xml")
        except OSError as exc:
            msg = f"Lifecycle {phase_name} step '{step.name}' failed to start"
            if abort_on_failure:
                fail(msg, str(exc))
                raise BuildProfileError(f"{msg}: {exc}") from exc
            else:
                warn(msg, str(exc))
