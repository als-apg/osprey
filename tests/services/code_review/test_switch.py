"""The review consumers' read of code review's runtime switch: off only when the terminal says so.

A real loopback server answers. In a deployment the terminal spawns the session, so the hook or
the executor asking is the terminal's descendant; here the asker is a child process of this test
process, which serves. An answer from anywhere else, this process asking itself included, is the
agent standing in for the terminal and must leave the review running.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from collections import namedtuple
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from osprey.mcp_server import http as http_module
from osprey.services.code_review.contract import ReviewSwitchedOff
from osprey.services.code_review.switch import switched_off

TOKEN = "panel-token-for-the-switch"
OFF = {"on": False, "off_by": "alice", "off_at": 1700000000.5}

#: Run in a child: ask the switch and print what came back.
_ASK = """
import contextlib
import json
from osprey.services.code_review.switch import switched_off
off = switched_off()
print(json.dumps(None if off is None else [off.off_by, off.off_at]))
"""


class _Terminal:
    """A loopback stand-in for the web terminal's switch route."""

    def __init__(self) -> None:
        self.status = 200
        self.body: bytes = json.dumps({"on": True, "off_by": None, "off_at": None}).encode()
        self.delay = 0.0
        self.requests: list[tuple[str, str | None]] = []
        terminal = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:
                terminal.requests.append((self.path, self.headers.get("Authorization")))
                if terminal.delay:
                    threading.Event().wait(terminal.delay)
                self.send_response(terminal.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(terminal.body)))
                self.end_headers()
                self.wfile.write(terminal.body)

            def log_message(self, *_args) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]

    def answer(self, body: object, status: int = 200) -> None:
        self.body = json.dumps(body).encode() if not isinstance(body, bytes) else body
        self.status = status


@pytest.fixture
def terminal(monkeypatch) -> Iterator[_Terminal]:
    served = _Terminal()
    thread = threading.Thread(target=served.server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("OSPREY_WEB_PORT", str(served.port))
    monkeypatch.setenv("OSPREY_PANEL_TOKEN", TOKEN)
    # A proxy that would swallow the read: loopback must never go through it.
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    http_module.reset_panel_token_latch()
    yield served
    served.server.shutdown()
    served.server.server_close()
    http_module.reset_panel_token_latch()


def _ask_as_a_descendant() -> ReviewSwitchedOff | None:
    """Ask the switch from a child of this process, as a session's hook asks its terminal."""
    done = subprocess.run(
        [sys.executable, "-c", _ASK],
        capture_output=True,
        text=True,
        timeout=60,
        env=dict(os.environ),
        check=True,
    )
    answer = json.loads(done.stdout.strip().splitlines()[-1])
    return None if answer is None else ReviewSwitchedOff(off_by=answer[0], off_at=answer[1])


# ---- The terminal that spawned the session ----


def test_an_off_answer_from_the_terminal_names_who_and_when(terminal):
    terminal.answer(OFF)

    assert _ask_as_a_descendant() == ReviewSwitchedOff(off_by="alice", off_at=1700000000.5)
    [(path, auth)] = terminal.requests
    assert path == "/api/code-review/switch"
    assert auth == f"Bearer {TOKEN}"


def test_an_on_answer_reviews(terminal):
    assert _ask_as_a_descendant() is None
    assert terminal.requests


def test_an_off_answer_without_names_still_skips(terminal):
    terminal.answer({"on": False, "off_by": 7, "off_at": "noon"})

    assert _ask_as_a_descendant() == ReviewSwitchedOff(off_by=None, off_at=None)


@pytest.mark.parametrize(
    "body",
    [{"on": "false"}, {"on": 0}, {"off_by": "alice"}, [{"on": False}], b"not json"],
    ids=["string", "zero", "no-on", "list", "not-json"],
)
def test_anything_but_a_literal_off_reviews(terminal, body):
    terminal.answer(body)

    assert _ask_as_a_descendant() is None


@pytest.mark.parametrize("status", [401, 500])
def test_a_refusal_reviews_even_when_its_body_says_off(terminal, status):
    terminal.answer(OFF, status=status)

    assert _ask_as_a_descendant() is None


# ---- Anyone answering in the terminal's place ----


def test_an_off_answer_from_a_process_that_did_not_spawn_the_session_reviews(terminal):
    """This process serves and asks itself: it is no ancestor of itself, as the agent is not."""
    terminal.answer(OFF)

    assert switched_off() is None
    assert terminal.requests, "the forged answer was read, and refused for who served it"


def test_an_off_answer_from_a_sibling_on_the_port_reviews(monkeypatch):
    """The agent's own server on a port the terminal left free: a sibling, not an ancestor."""
    script = (
        "import http.server, json, sys\n"
        "class H(http.server.BaseHTTPRequestHandler):\n"
        "    protocol_version = 'HTTP/1.1'\n"
        "    def do_GET(self):\n"
        f"        body = json.dumps({OFF!r}).encode()\n"
        "        self.send_response(200); self.send_header('Content-Length', str(len(body)))\n"
        "        self.end_headers(); self.wfile.write(body)\n"
        "    def log_message(self, *a): pass\n"
        "s = http.server.HTTPServer(('127.0.0.1', 0), H)\n"
        "print(s.server_address[1], flush=True)\n"
        "s.serve_forever()\n"
    )
    forger = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    try:
        port = forger.stdout.readline().strip()
        monkeypatch.setenv("OSPREY_WEB_PORT", port)

        assert switched_off() is None
    finally:
        forger.kill()
        forger.wait(timeout=10)


# ---- No answer ----


def test_a_terminal_too_slow_to_answer_reviews(terminal, monkeypatch):
    terminal.answer(OFF)
    terminal.delay = 0.5
    real = http_module.fetch_code_review_switch
    monkeypatch.setattr(http_module, "fetch_code_review_switch", lambda: real(timeout=0.1))

    assert switched_off() is None


@pytest.mark.parametrize("stage", ["status", "headers", "body"])
def test_an_answer_dripped_slower_than_the_deadline_reviews_in_time(monkeypatch, stage):
    """Every byte arrives well inside the timeout, yet the whole read gives up at it."""
    answer = json.dumps(OFF).encode().ljust(4096)
    head = f"HTTP/1.1 200 OK\r\nContent-Length: {len(answer)}\r\n\r\n".encode()
    fast, drip = {
        "status": (b"", head + answer),
        "headers": (head[:20], head[20:] + answer),
        "body": (head, answer),
    }[stage]
    listener = socket.create_server(("127.0.0.1", 0))
    stop = threading.Event()

    def serve() -> None:
        conn, _ = listener.accept()
        with conn:
            conn.recv(4096)
            conn.sendall(fast)
            for byte in drip:
                if stop.wait(0.05):
                    return
                with contextlib.suppress(OSError):
                    conn.sendall(bytes([byte]))

    server = threading.Thread(target=serve, daemon=True)
    server.start()
    monkeypatch.setenv("OSPREY_WEB_PORT", str(listener.getsockname()[1]))
    try:
        started = time.monotonic()
        assert http_module.fetch_code_review_switch(timeout=0.5) is None
        assert time.monotonic() - started < 1.5
    finally:
        stop.set()
        server.join(timeout=5)
        listener.close()


def test_an_unreachable_terminal_reviews(terminal, monkeypatch):
    port = terminal.port
    terminal.server.shutdown()
    terminal.server.server_close()
    monkeypatch.setenv("OSPREY_WEB_PORT", str(port))

    assert switched_off() is None


@pytest.mark.parametrize("port", [None, "", "  ", "abc", "80; rm"], ids=str)
def test_no_terminal_named_reviews_and_dials_nothing(monkeypatch, port):
    if port is None:
        monkeypatch.delenv("OSPREY_WEB_PORT", raising=False)
    else:
        monkeypatch.setenv("OSPREY_WEB_PORT", port)
    dialed: list = []
    monkeypatch.setattr("http.client.HTTPConnection.connect", lambda *a: dialed.append(a))

    assert switched_off() is None
    assert dialed == []


def test_a_broken_reader_reviews(monkeypatch):
    def broken(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(http_module, "fetch_code_review_switch", broken)

    assert switched_off() is None


def test_a_process_table_that_cannot_be_read_is_no_proof(monkeypatch):
    def unreadable(*_a, **_k):
        raise PermissionError("no")

    monkeypatch.setattr("psutil.Process.parents", unreadable)

    assert http_module._served_by_an_ancestor(("127.0.0.1", 1), ("127.0.0.1", 2)) is False


def _ancestor_holding(*conns):
    class _Ancestor:
        def net_connections(self, **_kind):
            return list(conns)

    return [_Ancestor()]


_Addr = namedtuple("_Addr", "ip port")


def _conn(laddr, raddr, status):
    return SimpleNamespace(laddr=_Addr(*laddr), raddr=_Addr(*raddr) if raddr else (), status=status)


@pytest.mark.parametrize(
    ("listen_ip", "peer", "expected"),
    [
        ("127.0.0.1", ("127.0.0.1", 50000), True),
        ("0.0.0.0", ("::ffff:127.0.0.1", 50000), True),
        # Same ports, another client address: not this connection.
        ("127.0.0.1", ("10.0.0.5", 50000), False),
        # Listening on another address than the one we reached.
        ("127.0.0.2", ("127.0.0.1", 50000), False),
    ],
)
def test_the_ancestor_must_hold_this_very_connection(monkeypatch, listen_ip, peer, expected):
    import psutil

    held = _ancestor_holding(
        _conn((listen_ip, 8080), None, psutil.CONN_LISTEN),
        _conn(("127.0.0.1", 8080), peer, psutil.CONN_ESTABLISHED),
    )
    monkeypatch.setattr("psutil.Process.parents", lambda self: held)

    ours, theirs = ("127.0.0.1", 50000), ("127.0.0.1", 8080)
    assert http_module._served_by_an_ancestor(ours, theirs) is expected
