"""The review consumers' read of code review's runtime switch: off only when the terminal says so.

A real UNIX socket answers. In a deployment the terminal spawns the session, so the hook or the
executor asking is the terminal's descendant; here the asker is a child process of this test
process, which serves. An answer from anywhere else, this process asking itself included, is the
agent standing in for the terminal and must leave the review running.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterator

import pytest

from osprey.mcp_server import http as http_module
from osprey.services.code_review.contract import ReviewSwitchedOff
from osprey.services.code_review.switch import (
    SWITCH_ANSWER_LIMIT,
    SWITCH_SOCKET_ENV,
    switched_off,
)

ON = {"on": True, "off_by": None, "off_at": None}
OFF = {"on": False, "off_by": "alice", "off_at": 1700000000.5}

#: Run in a child: ask the switch, with the timeout in ``argv[1]`` when given, and print what
#: came back and how long it took.
_ASK = """
import json
import logging
import sys
import time
logging.basicConfig(level=logging.WARNING, stream=sys.stderr, force=True)
from osprey.mcp_server import http
from osprey.services.code_review.switch import switched_off
if len(sys.argv) > 1:
    real = http.fetch_code_review_switch
    http.fetch_code_review_switch = lambda: real(timeout=float(sys.argv[1]))
started = time.monotonic()
off = switched_off()
took = time.monotonic() - started
print(json.dumps([None if off is None else [off.off_by, off.off_at], took]))
"""

#: Run in a child: serve ``argv[2]`` on the socket ``argv[1]`` until killed.
_FORGE = """
import os, socket, sys
listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
listener.bind(sys.argv[1])
listener.listen()
print(os.getpid(), flush=True)
while True:
    conn, _ = listener.accept()
    with conn:
        conn.sendall(sys.argv[2].encode())
"""


class _Terminal:
    """A stand-in for the web terminal's switch socket, served from this process."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.body = json.dumps(ON).encode()
        self.delay = 0.0
        self.drip_from: int | None = None
        self.served = 0
        self.stop = threading.Event()
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(path)
        self.listener.listen()
        self.listener.settimeout(0.05)
        threading.Thread(target=self._serve, daemon=True).start()

    def answer(self, body: object) -> None:
        self.body = body if isinstance(body, bytes) else json.dumps(body).encode()

    def close(self) -> None:
        self.stop.set()
        self.listener.close()
        with contextlib.suppress(FileNotFoundError):
            os.unlink(self.path)

    def _serve(self) -> None:
        while not self.stop.is_set():
            try:
                conn, _ = self.listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            threading.Thread(target=self._answer, args=(conn,), daemon=True).start()

    def _answer(self, conn: socket.socket) -> None:
        conn.settimeout(None)
        with conn, contextlib.suppress(OSError):
            self.served += 1
            if self.stop.wait(self.delay):
                return
            cut = len(self.body) if self.drip_from is None else self.drip_from
            conn.sendall(self.body[:cut])
            for byte in self.body[cut:]:
                if self.stop.wait(0.05):
                    return
                conn.sendall(bytes([byte]))


@pytest.fixture
def socket_dir() -> Iterator[str]:
    """A short private directory: a socket path must fit in ``sun_path``."""
    directory = tempfile.mkdtemp(prefix="osprey-cr-")
    yield directory
    shutil.rmtree(directory, ignore_errors=True)


@pytest.fixture
def terminal(monkeypatch, socket_dir) -> Iterator[_Terminal]:
    served = _Terminal(os.path.join(socket_dir, "s"))
    monkeypatch.setenv(SWITCH_SOCKET_ENV, served.path)
    yield served
    served.close()


def _ask(timeout: float | None = None) -> tuple[ReviewSwitchedOff | None, float, str]:
    """Ask the switch from a child of this process, as a session's hook asks its terminal."""
    done = subprocess.run(
        [sys.executable, "-c", _ASK, *([] if timeout is None else [str(timeout)])],
        capture_output=True,
        text=True,
        timeout=60,
        env=dict(os.environ),
        check=True,
    )
    answer, took = json.loads(done.stdout.strip().splitlines()[-1])
    off = None if answer is None else ReviewSwitchedOff(off_by=answer[0], off_at=answer[1])
    return off, took, done.stderr


def _ask_as_a_descendant(*, expect_off: bool = False) -> ReviewSwitchedOff | None:
    """The child's answer; with *expect_off*, a ``None`` fails with the child's own log."""
    off, _took, log = _ask()
    if off is None and expect_off:
        pytest.fail(f"the terminal said off but the child read no answer:\n{log}")
    return off


@contextlib.contextmanager
def _forger(path: str, body: object) -> Iterator[int]:
    """A process this test spawns serving *body* on *path*: the agent's own server."""
    forger = subprocess.Popen(
        [sys.executable, "-c", _FORGE, path, json.dumps(body)], stdout=subprocess.PIPE, text=True
    )
    try:
        yield int(forger.stdout.readline())
    finally:
        forger.kill()
        forger.wait(timeout=10)


# ---- The terminal that spawned the session ----


def test_an_off_answer_from_the_terminal_names_who_and_when(terminal):
    terminal.answer(OFF)

    assert _ask_as_a_descendant(expect_off=True) == ReviewSwitchedOff(
        off_by="alice", off_at=1700000000.5
    )
    assert terminal.served == 1


def test_an_on_answer_reviews(terminal):
    assert _ask_as_a_descendant() is None
    assert terminal.served == 1


def test_an_off_answer_without_names_still_skips(terminal):
    terminal.answer({"on": False, "off_by": 7, "off_at": "noon"})

    assert _ask_as_a_descendant(expect_off=True) == ReviewSwitchedOff(off_by=None, off_at=None)


@pytest.mark.parametrize(
    "body",
    [{"on": "false"}, {"on": 0}, {"off_by": "alice"}, [{"on": False}], b"not json"],
    ids=["string", "zero", "no-on", "list", "not-json"],
)
def test_anything_but_a_literal_off_reviews(terminal, body):
    terminal.answer(body)

    assert _ask_as_a_descendant() is None


@pytest.mark.parametrize(
    "body",
    [
        b"",
        json.dumps(OFF).encode()[:-1],
        json.dumps(OFF).encode().ljust(SWITCH_ANSWER_LIMIT + 1),
    ],
    ids=["nothing", "cut-short", "too-long"],
)
def test_an_answer_that_is_not_whole_reviews_even_when_it_says_off(terminal, body):
    terminal.answer(body)

    assert _ask_as_a_descendant() is None
    assert terminal.served == 1


@pytest.mark.skipif(sys.platform != "linux", reason="a non-dumpable process is a Linux property")
def test_an_off_answer_from_a_terminal_that_hides_its_memory_is_read(terminal):
    """The web terminal marks itself non-dumpable; its descendants cannot read its ``/proc`` fds."""
    import ctypes

    pr_get_dumpable, pr_set_dumpable = 3, 4
    libc = ctypes.CDLL(None, use_errno=True)
    was = libc.prctl(pr_get_dumpable, 0, 0, 0, 0)
    terminal.answer(OFF)
    assert libc.prctl(pr_set_dumpable, 0, 0, 0, 0) == 0
    try:
        if os.geteuid() != 0:
            peek = f"import os; os.listdir('/proc/{os.getpid()}/fd')"
            hidden = subprocess.run([sys.executable, "-c", peek], capture_output=True, timeout=60)
            assert hidden.returncode != 0, "the terminal's fds are readable: nothing is hidden"

        assert _ask_as_a_descendant(expect_off=True) == ReviewSwitchedOff(
            off_by="alice", off_at=1700000000.5
        )
    finally:
        libc.prctl(pr_set_dumpable, was, 0, 0, 0)


# ---- Anyone answering in the terminal's place ----


def test_an_off_answer_from_a_process_that_did_not_spawn_the_session_reviews(terminal, caplog):
    """This process serves and asks itself: it is no ancestor of itself, as the agent is not."""
    terminal.answer(OFF)

    with caplog.at_level("WARNING", logger="osprey.mcp_server.http"):
        assert switched_off() is None
    assert any("outside this session" in r.getMessage() for r in caplog.records)


def test_an_off_answer_from_a_sibling_where_the_terminal_was_reviews(terminal):
    """The agent's own server bound at the path the terminal left: a sibling, not an ancestor."""
    terminal.close()
    with _forger(terminal.path, OFF):
        off, _took, log = _ask()

    assert off is None
    assert "outside this session" in log


def test_the_kernel_names_the_process_that_listens(terminal, socket_dir):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as ours:
        ours.connect(terminal.path)
        assert http_module._peer(ours) == (os.getpid(), os.getuid())

    path = os.path.join(socket_dir, "f")
    with _forger(path, OFF) as pid, socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as theirs:
        theirs.connect(path)
        assert http_module._peer(theirs) == (pid, os.getuid())


def _as_if_served_by(monkeypatch, pid: int, uid: int) -> None:
    monkeypatch.setattr(http_module, "_peer", lambda _sock: (pid, uid))


def test_an_ancestor_of_this_account_is_believed(terminal, monkeypatch):
    """The control for the two below: the same answer, believed when the peer qualifies."""
    terminal.answer(OFF)
    _as_if_served_by(monkeypatch, os.getppid(), os.getuid())

    assert switched_off() == ReviewSwitchedOff(off_by="alice", off_at=1700000000.5)


def test_an_ancestor_running_as_another_account_is_no_answer(terminal, monkeypatch):
    terminal.answer(OFF)
    _as_if_served_by(monkeypatch, os.getppid(), os.getuid() + 1)

    assert switched_off() is None


def test_a_process_table_that_cannot_be_read_is_no_proof(terminal, monkeypatch):
    def unreadable(*_a, **_k):
        raise PermissionError("no")

    terminal.answer(OFF)
    _as_if_served_by(monkeypatch, os.getppid(), os.getuid())
    monkeypatch.setattr("psutil.Process.parents", unreadable)

    assert switched_off() is None


def test_a_platform_that_names_no_peer_is_no_answer(terminal, monkeypatch):
    monkeypatch.setattr(http_module.sys, "platform", "win32")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.connect(terminal.path)

        assert http_module._peer(sock) is None
        assert http_module._served_by_an_ancestor(sock) is False


# ---- No answer ----


def test_a_terminal_too_slow_to_answer_reviews_in_time(terminal):
    terminal.answer(OFF)
    terminal.delay = 1.0

    off, took, _log = _ask(timeout=0.2)
    assert off is None
    assert took < 0.9

    assert _ask(timeout=5)[0] == ReviewSwitchedOff(off_by="alice", off_at=1700000000.5)


@pytest.mark.parametrize("drip_from", [0, 20], ids=["from-the-start", "after-a-prefix"])
def test_an_answer_dripped_slower_than_the_deadline_reviews_in_time(terminal, drip_from):
    """Every byte arrives well inside the timeout, yet the whole read gives up at it."""
    terminal.answer(json.dumps(OFF).encode().ljust(SWITCH_ANSWER_LIMIT))
    terminal.drip_from = drip_from

    off, took, _log = _ask(timeout=0.5)
    assert off is None
    assert took < 1.5


@pytest.mark.parametrize("left", ["no-file", "stale-file"])
def test_an_unreachable_terminal_reviews(terminal, left):
    terminal.close()
    if left == "stale-file":
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as dead:
            dead.bind(terminal.path)

    assert _ask_as_a_descendant() is None


@pytest.mark.parametrize("path", [None, "", "  ", "relative/s"], ids=str)
def test_no_terminal_named_reviews_and_dials_nothing(monkeypatch, path):
    if path is None:
        monkeypatch.delenv(SWITCH_SOCKET_ENV, raising=False)
    else:
        monkeypatch.setenv(SWITCH_SOCKET_ENV, path)
    dialed: list = []
    monkeypatch.setattr("socket.socket.connect", lambda *a: dialed.append(a))

    assert switched_off() is None
    assert dialed == []


def test_a_broken_reader_reviews(monkeypatch):
    def broken(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(http_module, "fetch_code_review_switch", broken)

    assert switched_off() is None
