"""Build lifecycle-phase execution helpers.

Runs the ``pre_build`` / ``post_build`` / ``validate`` command phases declared
in a build profile, with the shared subprocess environment (project ``.env``
vars, project venv on ``PATH``, ``_mcp_servers`` on ``PYTHONPATH``), streaming
and quiet output modes, timeout handling, and the JUnit test-results summary.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from osprey.errors import BuildProfileError
from osprey.utils.dotenv import parse_dotenv_file as _load_dotenv
from osprey.utils.logger import get_logger

logger = get_logger("build")


_SHELL_METACHARACTERS = ("|", "&&", "||", "$(", "`")


def _format_junit_summary(xml_path: Path) -> None:
    """Parse JUnit XML and print a Rich summary table of test results."""
    import xml.etree.ElementTree as ET

    from rich.console import Console
    from rich.table import Table

    if not xml_path.exists():
        return

    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return

    root = tree.getroot()

    table = Table(title="Integration Test Results", show_header=True, padding=(0, 1))
    table.add_column("Test", style="bold", no_wrap=True)
    table.add_column("Status", justify="center", width=6)
    table.add_column("Time", justify="right", width=8)

    for testsuite in root.iter("testsuite"):
        for testcase in testsuite.iter("testcase"):
            name = testcase.get("name", "unknown")
            time_s = testcase.get("time", "0")

            failure = testcase.find("failure")
            error = testcase.find("error")
            skipped = testcase.find("skipped")

            if failure is not None or error is not None:
                status = "[red]✗[/red]"
            elif skipped is not None:
                status = "[dim]skip[/dim]"
            else:
                status = "[green]✓[/green]"

            table.add_row(name, status, f"{float(time_s):.2f}s")

    if table.row_count > 0:
        render_console = Console(force_terminal=True, width=120)
        render_console.print(table)


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
    # Auto-inject .env vars into subprocess environment
    env_file = project_path / ".env"
    if env_file.is_file():
        dotenv_vars = _load_dotenv(env_file)
        sub_env = {**os.environ, **dotenv_vars}
        logger.info("Loaded %d vars from %s into lifecycle environment", len(dotenv_vars), env_file)
    else:
        sub_env = os.environ.copy()

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

    logger.info("  Running %s commands...", phase_name)
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
                logger.info("  > %s", step.name)
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

                def _drain_stdout(stdout=proc.stdout) -> None:
                    for line in stdout:
                        print(f"    {line}", end="", flush=True)

                reader = threading.Thread(target=_drain_stdout, daemon=True)
                reader.start()
                # Wait for stdout to drain (tests finished) with the full
                # timeout, then give the process a short grace period to
                # exit.  Some test frameworks (pyepics CA context) keep
                # background threads alive that prevent clean exit.
                reader.join(timeout=step.timeout)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                elapsed = time.monotonic() - t0
                if proc.returncode != 0:
                    msg = f"Lifecycle {phase_name} step '{step.name}' failed (exit {proc.returncode}, {elapsed:.1f}s)"
                    if abort_on_failure:
                        logger.error("  ✗ %s", msg)
                        _format_junit_summary(project_path / "check_results.xml")
                        raise BuildProfileError(msg)
                    else:
                        logger.warning("  ! %s", msg)
                else:
                    logger.info("  ✓ %s (%.1fs)", step.name, elapsed)
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
                    msg = f"Lifecycle {phase_name} step '{step.name}' failed (exit {result.returncode}, {elapsed:.1f}s)"
                    if output:
                        msg += f":\n{output}"
                    if abort_on_failure:
                        logger.error("  ✗ %s", msg)
                        _format_junit_summary(project_path / "check_results.xml")
                        raise BuildProfileError(msg)
                    else:
                        logger.warning("  ! %s", msg)
                else:
                    success_msg = f"{step.name} ({elapsed:.1f}s)"
                    output = (result.stdout + result.stderr).strip()
                    if output:
                        summary = output.rstrip().rsplit("\n", 1)[-1].strip()
                        if summary:
                            success_msg += f" — {summary}"
                    logger.info("  ✓ %s", success_msg)
                # Show JUnit summary if test results were produced
                _format_junit_summary(project_path / "check_results.xml")

        except subprocess.TimeoutExpired as e:
            elapsed = time.monotonic() - t0
            msg = f"Lifecycle {phase_name} step '{step.name}' timed out ({elapsed:.0f}s)"
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
            if partial.strip():
                tail = "\n".join(partial.strip().splitlines()[-20:])
                msg += f"\n  Last output:\n{tail}"
            if abort_on_failure:
                logger.error("  ✗ %s", msg)
                _format_junit_summary(project_path / "check_results.xml")
                raise BuildProfileError(msg) from None
            else:
                logger.warning("  ! %s", msg)
            _format_junit_summary(project_path / "check_results.xml")
        except OSError as exc:
            msg = f"Lifecycle {phase_name} step '{step.name}' failed to start: {exc}"
            if abort_on_failure:
                logger.error("  ✗ %s", msg)
                raise BuildProfileError(msg) from exc
            else:
                logger.warning("  ! %s", msg)
