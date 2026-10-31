"""A real dispatch worker + dispatcher as subprocesses, for the e2e lanes that need one.

The dispatch tutorial sweep, the allowlist-parity lane and the three chat-bridge
lanes (Google Chat, Microsoft Teams, Nextcloud Talk) all run the same pair: a
control-assistant repo built once with ``osprey init`` + ``osprey build``, then
``python -m osprey.mcp_server.dispatch_worker`` and ``python -m osprey.dispatch``
started with this interpreter, wired exactly as the compose templates wire the
deployed services. Each binds port 0 and names the port it got in uvicorn's
startup line, which :func:`tests.e2e._server_port.bound_port` reads back from the
log file the process writes to.
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

import pytest

from tests.e2e._deploy_support import (
    find_osprey_console_script,
    format_failure,
    wait_for_health,
)
from tests.e2e._server_port import UVICORN_STARTUP, bound_port

#: Lines of a child's log a failure message carries.
_LOG_TAIL_LINES = 40


def log_tail(log: Path) -> str:
    """The tail of a child's log, for failure messages."""
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "(no captured output)"
    tail = "\n".join(lines[-_LOG_TAIL_LINES:])
    return f"--- subprocess log (tail) ---\n{tail}" if tail else "(no captured output)"


def _spawn_server(
    argv: list[str], *, cwd: Path, env: Mapping[str, str], log: Path, health_timeout: float
) -> tuple[subprocess.Popen, int]:
    """Start a server child logging to *log*; return it and the port it bound, once healthy.

    The child binds port 0 itself, so the port is read from its startup line and is
    the one it holds: no other process can have taken it in between.
    """
    with log.open("w") as log_fh:
        proc = subprocess.Popen(  # fixed argv, no shell
            argv, cwd=str(cwd), env=dict(env), stdout=log_fh, stderr=subprocess.STDOUT
        )
    try:
        port = bound_port(log, UVICORN_STARTUP, child=proc, timeout=health_timeout)
        url = f"http://127.0.0.1:{port}/health"
        try:
            wait_for_health(url, health_timeout, proc=proc)
        except AssertionError as exc:
            raise AssertionError(f"{exc}\n{log_tail(log)}") from exc
    except BaseException:
        terminate(proc)
        raise
    return proc, port


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
    log: Path,
    *,
    token: str,
    health_timeout: float,
    run_cap_sec: int | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[subprocess.Popen, int]:
    """Start the dispatch worker, logging to *log*; return it and its port once healthy.

    The repo root and the render's config one level down are passed exactly as the
    dispatch_worker compose template wires the deployed worker. ``run_cap_sec`` is
    the worker's own per-run wall-clock cap (``DISPATCH_TIMEOUT_SEC``); a bridge
    lane sets it from the same constant it gives the bridge, so the bridge's
    poll-budget floor is validated against the right number. ``env`` adds to the
    worker's environment: the agent record it records its runs in
    (:meth:`tests.e2e._worker_record.WorkerRecord.worker_env`).
    """
    worker_env = {
        **os.environ,
        "DISPATCH_WORKER_PORT": "0",
        "DISPATCH_WORKER_TOKEN": token,
        "OSPREY_PROJECT_DIR": str(repo),
        "CONFIG_FILE": str(repo / "build" / "config.yml"),
        "CLAUDECODE": "",
    }
    if run_cap_sec is not None:
        worker_env["DISPATCH_TIMEOUT_SEC"] = str(run_cap_sec)
    worker_env.update(env or {})
    return _spawn_server(
        [sys.executable, "-m", "osprey.mcp_server.dispatch_worker"],
        cwd=repo,
        env=worker_env,
        log=log,
        health_timeout=health_timeout,
    )


@contextlib.contextmanager
def running_dispatch_stack(
    repo: Path,
    triggers_path: Path,
    write_triggers: Callable[[Path, int], None],
    *,
    token: str,
    health_timeout: float,
    run_cap_sec: int | None = None,
    worker_env: Mapping[str, str] | None = None,
) -> Iterator[dict]:
    """A real worker + dispatcher pair, each on the port it bound, torn down on the way out.

    ``write_triggers(triggers_path, worker_port)`` writes the triggers document the
    dispatcher runs, once the worker's port is known. ``worker_env`` adds to the
    worker's environment, as :func:`start_worker`'s ``env``. The two logs sit beside
    ``triggers_path``. Yields the two base URLs, the repo, the worker process and
    its log (whose tail a failure message may want).
    """
    worker_log = triggers_path.parent / "worker.log"
    dispatcher_log = triggers_path.parent / "dispatcher.log"

    worker_proc: subprocess.Popen | None = None
    dispatcher_proc: subprocess.Popen | None = None
    try:
        worker_proc, worker_port = start_worker(
            repo,
            worker_log,
            token=token,
            health_timeout=health_timeout,
            run_cap_sec=run_cap_sec,
            env=worker_env,
        )
        write_triggers(triggers_path, worker_port)
        dispatcher_env = {
            **os.environ,
            "TRIGGERS_YML": str(triggers_path),
            "EVENT_DISPATCHER_TOKEN": token,
            "DISPATCH_WORKER_TOKEN": token,
            "FASTMCP_TRANSPORT": "http",
            "FASTMCP_PORT": "0",
            "FASTMCP_HOST": "127.0.0.1",
            "MCP_TRANSPORT": "http",
            "MCP_PORT": "0",
            "CLAUDECODE": "",
        }
        dispatcher_proc, dispatcher_port = _spawn_server(
            [sys.executable, "-m", "osprey.dispatch"],
            cwd=repo,
            env=dispatcher_env,
            log=dispatcher_log,
            health_timeout=health_timeout,
        )

        yield {
            "dispatcher_url": f"http://127.0.0.1:{dispatcher_port}",
            "worker_url": f"http://127.0.0.1:{worker_port}",
            "repo": repo,
            "worker_proc": worker_proc,
            "worker_log": worker_log,
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
