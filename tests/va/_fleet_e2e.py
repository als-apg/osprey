"""The processes of the server fleet's end-to-end run, and what watches them.

``tests/va/test_server_fleet_live.py`` runs the fleet the way a deployment runs
it: the real fleet service process (which spawns its refresher), the real
panel backend serving the Channels panel's IOCs view and IOC page, and the
real ``server_fleet`` tool behind the channel_directory MCP server, all reading one
rendered config of the demo. This module holds the parts that need no
Channel Access: preparing that config, starting and stopping the processes,
and reading what the refresher left behind (its snapshots, its memory and
its per-cycle audit records). It imports nothing that loads libca, so a
process can import it without a serving stack.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

from osprey.interfaces.common_middleware import OPERATOR_SECRET_HEADER
from osprey.interfaces.web_auth import OPERATOR_SECRET_ENV

#: The target the demo's fleet refreshes: the virtual accelerator.
TARGET = "va"

#: How long a started process may take to answer its health route.
PROCESS_START_S = 120.0

#: How long one HTTP read of a started process may take.
HTTP_TIMEOUT_S = 10.0

#: How long one ``server_fleet`` call through the MCP server may take.
TOOL_CALL_S = 60.0

#: How long a stopped process tree gets to exit on SIGTERM before it is killed.
STOP_GRACE_S = 15.0

#: The refresher's one audit record per cycle (``refresher._audit_cycle``).
_CYCLE_RECORD = re.compile(rf"target={TARGET} channel_count=(\d+) cycle=(\d+)")

#: The tool reader: the channel_directory MCP server built from the rendered config,
#: with ``server_fleet`` called through an in-memory MCP client. It answers one
#: command per stdin line -- ``call`` (one call, its answer) or ``loop SECONDS``
#: (call until the time is up, the number of calls) -- each with one line
#: starting ``OSPREY_TOOL ``.
_TOOL_READER = r"""
import asyncio, json, sys, time

async def main():
    from fastmcp import Client
    from osprey.mcp_server.channel_directory.server import create_server

    server = create_server()

    async def call(client):
        result = await client.call_tool("server_fleet", {"limit": 1000})
        return json.loads(result.content[0].text)

    async with Client(server) as client:
        sys.stdout.write("OSPREY_TOOL ready\n")
        sys.stdout.flush()
        loop = asyncio.get_running_loop()
        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                return
            verb, *args = line.split()
            if verb == "call":
                reply = await call(client)
            elif verb == "loop":
                deadline = time.monotonic() + float(args[0])
                calls = 0
                while time.monotonic() < deadline:
                    answer = await call(client)
                    if answer.get("outcome") != "ok":
                        raise RuntimeError(f"server_fleet answered {answer.get('outcome')}")
                    calls += 1
                reply = calls
            else:
                raise RuntimeError(f"unknown command {verb!r}")
            sys.stdout.write("OSPREY_TOOL " + json.dumps(reply) + "\n")
            sys.stdout.flush()

asyncio.run(main())
"""


# =============================================================================
# The rendered config
# =============================================================================


@dataclass(frozen=True)
class FleetProject:
    """A writable copy of the built demo, its config pointed at this run's servers."""

    repo: Path
    config_path: Path
    view_dir: Path
    fleet_port: int

    @property
    def build_dir(self) -> Path:
        return self.config_path.parent

    @property
    def fleet_url(self) -> str:
        return f"http://127.0.0.1:{self.fleet_port}"

    def config(self) -> dict[str, Any]:
        loaded: dict[str, Any] = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        return loaded


def point_config(config_path: Path, *, ca_port: int, fleet_port: int) -> None:
    """Point the rendered config at this run's served accelerator and fleet service.

    The virtual accelerator's two gateways become the served view's held CA
    port on loopback, searched directly (no name server), and
    ``services.server_fleet.port`` the port the fleet service binds -- the port
    every reader's fleet connection resolves. Nothing else changes: the
    period, the probe channel and the directory stay the demo's.
    """
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    gateway = {"address": "127.0.0.1", "port": ca_port, "use_name_server": False}
    gateways = config["control_system"]["connector"]["virtual_accelerator"]["gateways"]
    gateways["read_only"] = dict(gateway)
    gateways["write_access"] = dict(gateway)
    config.setdefault("services", {}).setdefault("server_fleet", {})["port"] = fleet_port
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


# =============================================================================
# Processes
# =============================================================================


def _descendants(pid: int) -> list[int]:
    """Every live descendant of ``pid`` (Linux ``/proc``; none elsewhere)."""
    found: list[int] = []
    pending = [pid]
    while pending:
        parent = pending.pop()
        for task in Path(f"/proc/{parent}/task").glob("*/children"):
            try:
                children = [int(word) for word in task.read_text().split()]
            except OSError:
                continue
            found.extend(children)
            pending.extend(children)
    return found


def stop_tree(proc: subprocess.Popen[Any]) -> None:
    """Stop ``proc`` and everything it started: SIGTERM, a grace period, then SIGKILL."""
    children = _descendants(proc.pid)
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=STOP_GRACE_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=STOP_GRACE_S)
    for child in children:
        with contextlib.suppress(ProcessLookupError):
            os.kill(child, signal.SIGKILL)


def environment(
    project: FleetProject, spool_dir: Path, extra: Mapping[str, str] | None = None
) -> dict[str, str]:
    """A started process's environment: this config, its own spool, no URL override."""
    env = dict(os.environ)
    env.pop("SERVER_FLEET_URL", None)
    env.update(
        {
            "OSPREY_CONFIG": str(project.config_path),
            "OSPREY_SPOOL_DIR": str(spool_dir),
            "PYTHONUNBUFFERED": "1",
            **dict(extra or {}),
        }
    )
    return env


def _await_health(
    proc: subprocess.Popen[Any],
    url: str,
    log: Path,
    headers: Mapping[str, str] | None = None,
) -> None:
    deadline = time.monotonic() + PROCESS_START_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise AssertionError(
                f"{url} exited ({proc.returncode}) before it served: "
                f"{log.read_text(errors='replace')[-2000:]}"
            )
        try:
            if httpx.get(url, timeout=2.0, headers=dict(headers or {})).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.2)
    raise AssertionError(f"{url} never answered: {log.read_text(errors='replace')[-2000:]}")


def start_fleet_service(
    project: FleetProject,
    state_dir: Path,
    spool_dir: Path,
    log: Path,
    ca_env: Mapping[str, str],
) -> subprocess.Popen[Any]:
    """``python -m osprey.services.channel_directory.fleet`` on the project's config; once it serves."""
    spool_dir.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as out:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "osprey.services.channel_directory.fleet",
                "--config",
                str(project.config_path),
                "--state-dir",
                str(state_dir),
                "--host",
                "127.0.0.1",
                "--port",
                str(project.fleet_port),
            ],
            cwd=project.build_dir,
            env=environment(project, spool_dir, ca_env),
            stdout=out,
            stderr=subprocess.STDOUT,
        )
    try:
        _await_health(proc, f"{project.fleet_url}/health", log)
    except BaseException:
        stop_tree(proc)
        raise
    return proc


@dataclass
class Panel:
    """One panel backend process: the Channels panel's app, as the web launcher serves it."""

    proc: subprocess.Popen[Any]
    url: str
    secret: str
    client: httpx.Client = field(init=False)

    def __post_init__(self) -> None:
        # The operator's secret, as the web terminal's proxy presents it: the
        # panel admits no request without a credential.
        self.client = httpx.Client(
            base_url=self.url,
            timeout=HTTP_TIMEOUT_S,
            headers={OPERATOR_SECRET_HEADER: self.secret},
        )

    def view(self) -> dict[str, Any]:
        """The IOCs view's answer: ``GET /api/directory/fleet``."""
        response = self.client.get("/api/directory/fleet")
        response.raise_for_status()
        answer: dict[str, Any] = response.json()
        return answer

    def server_page(self, name: str) -> dict[str, Any]:
        """An IOC page's facts: ``GET /api/directory/server/<name>``."""
        response = self.client.get(f"/api/directory/server/{name}")
        response.raise_for_status()
        facts: dict[str, Any] = response.json()
        return facts

    def stop(self) -> None:
        self.client.close()
        stop_tree(self.proc)


def start_panel(
    project: FleetProject, port: int, spool_dir: Path, log: Path, ca_env: Mapping[str, str]
) -> Panel:
    """The Channels panel's backend (``channel_finder.app:create_app``) under uvicorn; once it serves."""
    spool_dir.mkdir(parents=True, exist_ok=True)
    secret = secrets.token_hex(16)
    with log.open("wb") as out:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "osprey.interfaces.channel_finder.app:create_app",
                "--factory",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--log-level",
                "warning",
            ],
            cwd=project.build_dir,
            env=environment(project, spool_dir, {**ca_env, OPERATOR_SECRET_ENV: secret}),
            stdout=out,
            stderr=subprocess.STDOUT,
        )
    url = f"http://127.0.0.1:{port}"
    try:
        _await_health(proc, f"{url}/health", log, {OPERATOR_SECRET_HEADER: secret})
    except BaseException:
        stop_tree(proc)
        raise
    return Panel(proc, url, secret)


class ToolReader:
    """The channel_directory MCP server in a process of its own, calling ``server_fleet``."""

    def __init__(
        self, project: FleetProject, spool_dir: Path, log: Path, ca_env: Mapping[str, str]
    ) -> None:
        spool_dir.mkdir(parents=True, exist_ok=True)
        self._log = log
        self._err = log.open("wb")
        self.proc = subprocess.Popen(
            [sys.executable, "-P", "-c", _TOOL_READER],
            cwd=project.build_dir,
            env=environment(project, spool_dir, ca_env),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._err,
            text=True,
        )
        try:
            ready = self._reply(PROCESS_START_S)
        except BaseException:
            self.stop()
            raise
        assert ready == "ready", ready

    def _reply(self, timeout: float) -> Any:
        assert self.proc.stdout is not None
        result: list[str] = []

        def read() -> None:
            assert self.proc.stdout is not None
            for line in self.proc.stdout:
                if line.startswith("OSPREY_TOOL "):
                    result.append(line.removeprefix("OSPREY_TOOL ").strip())
                    return

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        reader.join(timeout)
        if not result:
            raise AssertionError(
                f"the tool reader gave no answer: {self._log.read_text(errors='replace')[-2000:]}"
            )
        text = result[0]
        return text if text == "ready" else json.loads(text)

    def _send(self, line: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def call(self) -> dict[str, Any]:
        """One ``server_fleet`` call with no filter; its JSON answer."""
        self._send("call")
        answer: dict[str, Any] = self._reply(TOOL_CALL_S)
        return answer

    def start_loop(self, seconds: float) -> None:
        """Call ``server_fleet`` over and over for ``seconds``; :meth:`loop_count` collects the count."""
        self._send(f"loop {seconds}")

    def loop_count(self, seconds: float) -> int:
        count: int = self._reply(seconds + TOOL_CALL_S)
        return count

    def stop(self) -> None:
        if self.proc.stdin is not None:
            with contextlib.suppress(OSError):
                self.proc.stdin.close()
        stop_tree(self.proc)
        self._err.close()


# =============================================================================
# What the refresher left behind
# =============================================================================


@dataclass(frozen=True)
class Cycle:
    """One published cycle: its snapshot, and whether it listed the directory."""

    number: int
    generated_at: float
    document: dict[str, Any]
    listed: bool
    seen_at: float


def cycle_reads(spool_dir: Path) -> dict[int, int]:
    """Each cycle's channel reads, from the refresher's one audit record per cycle."""
    reads: dict[int, int] = {}
    for path in spool_dir.rglob("*"):
        if not path.is_file():
            continue
        for match in _CYCLE_RECORD.finditer(path.read_text(encoding="utf-8", errors="replace")):
            reads[int(match.group(2))] = int(match.group(1))
    return reads


class SnapshotWatcher:
    """Records every cycle the refresher publishes, from its files alone.

    The snapshot file and the memory file are what the refresher writes; reading
    them asks nothing of the fleet service, the panel or the control system, so
    watching a quiet window costs it no reader. A cycle listed the directory
    when the memory's last good listing was taken at that cycle's start (the
    refresher stamps both with one clock reading).
    """

    def __init__(self, state_dir: Path, poll_s: float = 0.1) -> None:
        self._state_dir = state_dir
        self._poll_s = poll_s
        self._lock = threading.Lock()
        self._cycles: dict[int, Cycle] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        from osprey.services.channel_directory.fleet.memory import FleetMemory
        from osprey.services.channel_directory.fleet.snapshot import snapshot_path

        path = snapshot_path(self._state_dir, TARGET)
        last: int | None = None
        while not self._stop.wait(self._poll_s):
            try:
                document = json.loads(path.read_bytes())
            except (OSError, ValueError):
                continue
            number = document.get("cycle")
            if document.get("outcome") != "ok" or number is None or number == last:
                continue
            memory = FleetMemory.load(self._state_dir, retention=float("inf"))
            listing = memory.last_good_listing(TARGET, "virtual_accelerator")
            generated_at = float(document["generated_at"])
            cycle = Cycle(
                number=int(number),
                generated_at=generated_at,
                document=document,
                listed=listing is not None and listing.at == generated_at,
                seen_at=time.time(),
            )
            with self._lock:
                self._cycles[cycle.number] = cycle
            last = number

    def cycles(self) -> list[Cycle]:
        with self._lock:
            return [self._cycles[n] for n in sorted(self._cycles)]

    def latest(self) -> Cycle | None:
        cycles = self.cycles()
        return cycles[-1] if cycles else None

    def wait_for(self, predicate: Callable[[Cycle], bool], timeout: float) -> Cycle | None:
        """The first recorded cycle ``predicate`` holds for, waiting up to ``timeout``."""
        deadline = time.monotonic() + timeout
        while True:
            for cycle in self.cycles():
                if predicate(cycle):
                    return cycle
            if time.monotonic() > deadline:
                return None
            time.sleep(0.1)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


# =============================================================================
# Comparing answers
# =============================================================================


def states(document: Mapping[str, Any]) -> dict[str, str]:
    """Each row's state, by IOC name, of a C5 snapshot document."""
    return {row["name"]: row["state"] for row in document["servers"]}


def without_links(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The tool's rows as the snapshot's: the tool adds only each row's ``links``."""
    return [{key: value for key, value in row.items() if key != "links"} for row in rows]
