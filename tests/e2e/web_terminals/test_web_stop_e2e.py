"""End-to-end proof that a detached ``osprey web`` server ends on SIGTERM, and that
``osprey web stop`` says so only once it has.

Two halves, one launch shape each (``test_session_restart_e2e.py``'s, whose
helpers this module imports): a minimal deployment repo with a host record, a
detached server on a reserved port, ``--shell true`` so the PTY never spawns
an agent.

1. **The server alone.** A browser's terminal socket is open and has received
   its first frame. The server's process tree is read, the server is sent
   SIGTERM directly — not through ``osprey web stop`` — and within the bound
   the server and every process group it started are gone, apart from the
   host record's group, which is shared and ends through its lease.
2. **The command.** ``osprey web stop`` exits 0 and prints ``Stopped`` only
   after the server has exited: at the moment it returns, the pid is gone and
   the PID file with it.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
import yaml
from websockets.sync.client import connect as ws_connect

from osprey.agent_record import host as host_record
from osprey.cli.web_cmd import _LEASE_RELEASE_WAIT_S, _STOP_WAIT_S, PID_FILE
from osprey.deployment import process_table
from osprey.interfaces.web_terminal import process_tree
from tests.e2e.web_terminals._minimal_repo import stop_host_record, write_minimal_repo
from tests.e2e.web_terminals.test_session_restart_e2e import (
    _launch_detached,
    _launcher_env,
    _read_pid,
    _wait_for_pid_exit,
)

pytestmark = [pytest.mark.e2e_smoke, pytest.mark.model_free]

_HOST = "127.0.0.1"
#: The ceiling on a server ending after SIGTERM: the bound ``osprey web stop`` gives it.
_STOP_TIMEOUT = _STOP_WAIT_S
#: A hang ceiling on ``osprey web stop``, which waits for the exit and then the lease.
_STOP_COMMAND_TIMEOUT = 2 * (_STOP_WAIT_S + _LEASE_RELEASE_WAIT_S)
_HTTP_TIMEOUT = 15.0
_WS_TIMEOUT = 15.0
_TEARDOWN_TIMEOUT = 5.0
_POLL_INTERVAL = 0.2


def _repo(tmp_path: Path, record_port: int) -> Path:
    return write_minimal_repo(
        tmp_path / "web-stop-e2e",
        {
            "system": {"name": "web-stop-e2e"},
            "control_system": {
                "type": "virtual_accelerator",
                "connector": {"virtual_accelerator": {"serving": "in_process"}},
                "writes": "never",
            },
        },
        record_port,
    )


def _pid_gone(pid: int) -> bool:
    """Whether *pid* no longer exists, asked once."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _record_pgid(repo: Path) -> int | None:
    """The process group of *repo*'s host record, which outlives one terminal by design."""
    config = yaml.safe_load((repo / "build" / "config.yml").read_text(encoding="utf-8"))
    pid = host_record.running_record_pid(host_record.host_state_dir(repo, config))
    if pid is None:
        return None
    with contextlib.suppress(OSError):
        return os.getpgid(pid)
    return None


@contextlib.contextmanager
def _detached_server(tmp_path: Path, record_port: int, port: int):
    """A detached server on *port*; yields the repo, its login URL and its pid."""
    repo = _repo(tmp_path, record_port)
    pids: list[int] = []
    try:
        login_url, bound = _launch_detached(repo, port, tmp_path / "launch.log")
        assert bound == port, f"the server bound {bound}, not {port}"
        pid = _read_pid(repo)
        assert pid is not None, "the detached launcher wrote no readable PID file"
        pids.append(pid)
        yield repo, login_url, pid
    finally:
        for pid in pids:
            if _wait_for_pid_exit(pid, _TEARDOWN_TIMEOUT):
                continue
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)
        stop_host_record(repo)


def test_sigterm_ends_a_detached_server_with_an_open_terminal_socket(
    tmp_path: Path, record_port: int, reserve_port
) -> None:
    port = reserve_port()
    with _detached_server(tmp_path, record_port, port) as (repo, login_url, pid):
        base_url = f"http://{_HOST}:{port}"
        with httpx.Client(follow_redirects=False, timeout=_HTTP_TIMEOUT) as client:
            exchange = client.get(login_url)
        assert exchange.status_code == 303, f"token exchange did not 303: {exchange.status_code}"
        cookies = "; ".join(f"{name}={value}" for name, value in exchange.cookies.items())
        assert cookies, "the token exchange set no cookie"

        with ws_connect(
            f"ws://{_HOST}:{port}/ws/terminal?mode=new",
            additional_headers={"Cookie": cookies, "Origin": base_url},
            open_timeout=_WS_TIMEOUT,
        ) as socket:
            first = socket.recv(timeout=_WS_TIMEOUT)
            assert first, "the terminal socket sent an empty first frame"

            groups = process_tree.tree_groups(pid, process_table.snapshot())
            record = _record_pgid(repo)
            os.kill(pid, signal.SIGTERM)

            deadline = time.monotonic() + _STOP_TIMEOUT
            survivors = groups
            while time.monotonic() < deadline:
                survivors = [
                    g
                    for g in process_tree.still_running(groups, process_table.snapshot())
                    if g.pgid != record
                ]
                if not survivors and _pid_gone(pid):
                    break
                time.sleep(_POLL_INTERVAL)
            assert _pid_gone(pid), (
                f"the server (PID {pid}) was still running {_STOP_TIMEOUT}s after SIGTERM "
                "with a terminal socket open"
            )
            assert not survivors, (
                f"processes the server started outlived it: {process_tree.describe(survivors)}"
            )


def test_web_stop_says_stopped_after_the_server_and_its_children_are_gone(
    tmp_path: Path, record_port: int, reserve_port
) -> None:
    port = reserve_port()
    with _detached_server(tmp_path, record_port, port) as (repo, _login_url, pid):
        stopped = subprocess.run(
            [sys.executable, "-m", "osprey.cli.main", "web", "stop"],
            cwd=str(repo),
            env=_launcher_env(),
            capture_output=True,
            text=True,
            timeout=_STOP_COMMAND_TIMEOUT,
        )
        pid_gone = _pid_gone(pid)
        pid_file_gone = not (repo / PID_FILE).exists()

        assert stopped.returncode == 0, f"`osprey web stop` exited {stopped.returncode}: {stopped}"
        assert "Stopped" in stopped.stdout, stopped
        assert pid_gone, f"`osprey web stop` returned while the server (PID {pid}) still ran"
        assert pid_file_gone, "`osprey web stop` returned 0 and left the PID file"
