"""A real dispatch worker + dispatcher as subprocesses, for the e2e lanes that need one.

The dispatch tutorial sweep, the allowlist-parity lane and the three chat-bridge
lanes (Google Chat, Microsoft Teams, Nextcloud Talk) all run the same pair: a
control-assistant repo built once with ``osprey init`` + ``osprey build``, then
``python -m osprey.mcp_server.dispatch_worker`` and ``python -m osprey.dispatch``
started on free loopback ports with this interpreter, wired exactly as the
compose templates wire the deployed services.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests._live import free_port
from tests.e2e._deploy_support import find_osprey_console_script, format_failure


def drain_output(proc: subprocess.Popen) -> str:
    """Best-effort grab of a subprocess's combined output for failure messages."""
    if proc.stdout is None:
        return "(no captured output)"
    try:
        data = proc.stdout.read1(16384) if hasattr(proc.stdout, "read1") else b""
    except Exception:
        data = b""
    text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else str(data)
    return f"--- subprocess output (partial) ---\n{text}" if text else "(no captured output)"


def wait_for_health(url: str, timeout: float, proc: subprocess.Popen) -> None:
    """Poll ``url`` until it returns HTTP 200, or fail with the captured output."""
    deadline = time.monotonic() + timeout
    last_err = "(no response yet)"
    with httpx.Client(timeout=3.0, trust_env=False) as client:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise AssertionError(
                    f"subprocess for {url} exited early (rc={proc.returncode}).\n"
                    f"{drain_output(proc)}"
                )
            try:
                status = client.get(url).status_code
                if status == 200:
                    return
                last_err = f"HTTP {status}"
            except httpx.HTTPError as exc:
                last_err = str(exc)
            time.sleep(0.5)
    raise AssertionError(
        f"timed out after {timeout:.0f}s waiting for {url} (last error: {last_err}).\n"
        f"{drain_output(proc)}"
    )


def terminate(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        # Ignored SIGKILL within the grace window; teardown is best-effort, so leave
        # it for the OS to reap rather than hang the test.
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=5)


def build_repo(
    base: Path,
    *,
    provider: str,
    model: str,
    timeout: int = 300,
    env: Mapping[str, str] | None = None,
) -> Path:
    """Init + build a real control-assistant deployment repo at ``base / "proj"``.

    Two steps because the surface has two: ``init`` writes the repo's source zone
    from the preset, ``build`` renders ``build/`` from it. ``--skip-deps`` keeps it
    fast (no project venv); the worker and dispatcher run with this repo's
    interpreter, so the project venv is not needed. ``env`` adds to the CLI's
    environment.
    """
    repo = base / "proj"
    osprey_bin = find_osprey_console_script()

    def _osprey(argv: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(  # fixed argv, no shell
            [str(osprey_bin), *argv],
            cwd=str(base),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env={**os.environ, "CLAUDECODE": "", **(env or {})},
        )

    init = _osprey(
        [
            "init",
            str(repo),
            "--preset",
            "control-assistant",
            "--no-git",
            "--set",
            f"provider={provider}",
            "--set",
            f"model={model}",
        ]
    )
    if init.returncode != 0:
        pytest.fail(format_failure(f"osprey init ({provider})", init))

    build = _osprey(["build", "--repo", str(repo), "--skip-deps", "--skip-lifecycle"])
    if build.returncode != 0:
        pytest.fail(format_failure("osprey build", build))
    if not (repo / "build" / "config.yml").is_file():
        pytest.fail(f"build succeeded but build/config.yml missing under {repo}")
    return repo


def start_worker(
    repo: Path,
    port: int,
    *,
    token: str,
    health_timeout: float,
    run_cap_sec: int | None = None,
) -> subprocess.Popen:
    """Start the dispatch worker on ``port`` and wait for its ``/health``.

    The repo root and the render's config one level down are passed exactly as the
    dispatch_worker compose template wires the deployed worker. ``run_cap_sec`` is
    the worker's own per-run wall-clock cap (``DISPATCH_TIMEOUT_SEC``); a bridge
    lane sets it from the same constant it gives the bridge, so the bridge's
    poll-budget floor is validated against the right number.
    """
    env = {
        **os.environ,
        "DISPATCH_WORKER_PORT": str(port),
        "DISPATCH_WORKER_TOKEN": token,
        "OSPREY_PROJECT_DIR": str(repo),
        "CONFIG_FILE": str(repo / "build" / "config.yml"),
        "CLAUDECODE": "",
    }
    if run_cap_sec is not None:
        env["DISPATCH_TIMEOUT_SEC"] = str(run_cap_sec)
    proc = subprocess.Popen(  # fixed argv, no shell
        [sys.executable, "-m", "osprey.mcp_server.dispatch_worker"],
        cwd=str(repo),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        wait_for_health(f"http://127.0.0.1:{port}/health", health_timeout, proc)
    except BaseException:
        terminate(proc)
        raise
    return proc


@contextlib.contextmanager
def running_dispatch_stack(
    repo: Path,
    triggers_path: Path,
    write_triggers: Callable[[Path, int], None],
    *,
    token: str,
    health_timeout: float,
    run_cap_sec: int | None = None,
) -> Iterator[dict]:
    """A real worker + dispatcher pair on free ports, torn down on the way out.

    ``write_triggers(triggers_path, worker_port)`` writes the triggers document the
    dispatcher runs, once the worker's port is known. Yields the two base URLs, the
    repo and the worker process (whose output a failure message may want).
    """
    worker_port = free_port()
    dispatcher_port = free_port()
    write_triggers(triggers_path, worker_port)

    worker_proc: subprocess.Popen | None = None
    dispatcher_proc: subprocess.Popen | None = None
    try:
        worker_proc = start_worker(
            repo,
            worker_port,
            token=token,
            health_timeout=health_timeout,
            run_cap_sec=run_cap_sec,
        )
        dispatcher_env = {
            **os.environ,
            "TRIGGERS_YML": str(triggers_path),
            "EVENT_DISPATCHER_TOKEN": token,
            "DISPATCH_WORKER_TOKEN": token,
            "FASTMCP_TRANSPORT": "http",
            "FASTMCP_PORT": str(dispatcher_port),
            "FASTMCP_HOST": "127.0.0.1",
            "MCP_TRANSPORT": "http",
            "MCP_PORT": str(dispatcher_port),
            "CLAUDECODE": "",
        }
        dispatcher_proc = subprocess.Popen(  # fixed argv, no shell
            [sys.executable, "-m", "osprey.dispatch"],
            cwd=str(repo),
            env=dispatcher_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        wait_for_health(
            f"http://127.0.0.1:{dispatcher_port}/health", health_timeout, dispatcher_proc
        )

        yield {
            "dispatcher_url": f"http://127.0.0.1:{dispatcher_port}",
            "worker_url": f"http://127.0.0.1:{worker_port}",
            "repo": repo,
            "worker_proc": worker_proc,
        }
    finally:
        terminate(dispatcher_proc)
        terminate(worker_proc)


def await_bridge(bridge: Any, what: str, probe: Callable[[], Any], timeout: float) -> Any:
    """Poll ``probe`` until it returns non-``None``, or fail informatively.

    For a chat bridge running in a daemon thread: ``bridge`` carries the ``thread``,
    the ``failure`` list it appends to, and its persisted ``dedup`` store. ``None`` and
    only ``None`` means "not yet", so a probe may legitimately answer a falsy value.
    Checks the bridge thread on every pass: a crashed or exited bridge is reported as
    such immediately instead of surfacing as an opaque timeout minutes later.
    """
    deadline = time.monotonic() + timeout
    while True:
        if bridge.failure:
            raise AssertionError(f"bridge thread died before {what}: {bridge.failure[0]!r}")
        if not bridge.thread.is_alive():
            raise AssertionError(f"bridge thread exited before {what}")
        value = probe()
        if value is not None:
            return value
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"timed out after {timeout:.0f}s waiting for {what}\n"
                f"  dedup: {json.dumps(bridge.dedup, indent=2, default=str)[:2000]}"
            )
        time.sleep(0.5)


def drain_thread_alive() -> bool:
    """Whether a bridge engine's single drain thread is still running.

    It is a daemon named ``bridge-drain``, so it never shows up as a stuck process — it
    shows up as a store being written by a bridge the test thinks has stopped.
    """
    return any(t.name == "bridge-drain" and t.is_alive() for t in threading.enumerate())
