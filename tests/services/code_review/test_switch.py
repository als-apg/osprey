"""The review consumers' read of code review's runtime switch: off only when the terminal says so.

A real loopback server answers, so the read is exercised end to end: the port it dials, the
bearer it carries, the proxy it ignores, and every answer that must leave the review running.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from osprey.mcp_server import http as http_module
from osprey.services.code_review.contract import ReviewSwitchedOff
from osprey.services.code_review.switch import switched_off

TOKEN = "panel-token-for-the-switch"


class _Terminal:
    """A loopback stand-in for the web terminal's switch route."""

    def __init__(self) -> None:
        self.status = 200
        self.body: bytes = json.dumps({"on": True, "off_by": None, "off_at": None}).encode()
        self.delay = 0.0
        self.requests: list[tuple[str, str | None]] = []
        terminal = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                terminal.requests.append((self.path, self.headers.get("Authorization")))
                if terminal.delay:
                    threading.Event().wait(terminal.delay)
                self.send_response(terminal.status)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(terminal.body)

            def log_message(self, *_args) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
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


def test_an_off_answer_names_who_and_when(terminal):
    terminal.answer({"on": False, "off_by": "alice", "off_at": 1700000000.5})

    assert switched_off() == ReviewSwitchedOff(off_by="alice", off_at=1700000000.5)
    [(path, auth)] = terminal.requests
    assert path == "/api/code-review/switch"
    assert auth == f"Bearer {TOKEN}"


def test_an_on_answer_reviews(terminal):
    assert switched_off() is None
    assert terminal.requests


def test_an_off_answer_without_names_still_skips(terminal):
    terminal.answer({"on": False, "off_by": 7, "off_at": "noon"})

    assert switched_off() == ReviewSwitchedOff(off_by=None, off_at=None)


@pytest.mark.parametrize(
    "body",
    [
        {"on": "false"},
        {"on": 0},
        {"on": None},
        {"off_by": "alice"},
        [{"on": False}],
        "off",
        b"not json",
        b"",
    ],
    ids=["string", "zero", "null", "no-on", "list", "bare-string", "not-json", "empty"],
)
def test_anything_but_a_literal_off_reviews(terminal, body):
    terminal.answer(body)

    assert switched_off() is None


@pytest.mark.parametrize("status", [401, 404, 500])
def test_a_refusal_reviews_even_when_its_body_says_off(terminal, status):
    terminal.answer({"on": False, "off_by": "mallory"}, status=status)

    assert switched_off() is None


def test_a_terminal_too_slow_to_answer_reviews(terminal, monkeypatch):
    terminal.answer({"on": False, "off_by": "alice"})
    terminal.delay = 0.5
    real = http_module.fetch_code_review_switch
    monkeypatch.setattr(http_module, "fetch_code_review_switch", lambda: real(timeout=0.1))

    assert switched_off() is None


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
    monkeypatch.setattr("urllib.request.OpenerDirector.open", lambda *a, **k: dialed.append(a))

    assert switched_off() is None
    assert dialed == []


def test_a_broken_reader_reviews(monkeypatch):
    def broken(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(http_module, "fetch_code_review_switch", broken)

    assert switched_off() is None
