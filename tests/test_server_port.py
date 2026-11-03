"""Unit tests for :mod:`tests.e2e._server_port`, the reader of a server's startup line."""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.cli.web_cmd import _STARTUP_LINE
from tests._container_support import ContainerExitedError
from tests.e2e._server_port import UVICORN_STARTUP, bound_port, port_in


class _Child:
    """A child process stand-in: ``poll()`` runs *on_poll* each round and returns *code*."""

    def __init__(self, code: int | None = None, on_poll=None) -> None:
        self.pid = 4242
        self.code = code
        self.polls = 0
        self.on_poll = on_poll

    def poll(self) -> int | None:
        self.polls += 1
        if self.on_poll is not None:
            self.on_poll(self.polls)
        return self.code


@pytest.mark.parametrize(
    ("line", "prefix", "port"),
    [
        (
            "INFO:     Uvicorn running on http://127.0.0.1:43123 (Press CTRL+C to quit)",
            UVICORN_STARTUP,
            43123,
        ),
        (
            "INFO:     Uvicorn running on http://0.0.0.0:43124 (Press CTRL+C)",
            UVICORN_STARTUP,
            43124,
        ),
        ("INFO:     Uvicorn running on http://[::1]:43125 (Press CTRL+C)", UVICORN_STARTUP, 43125),
        (f"{_STARTUP_LINE} http://127.0.0.1:43126", _STARTUP_LINE, 43126),
    ],
)
def test_the_port_of_the_url_after_the_prefix(line, prefix, port):
    assert port_in(f"booting\n{line}\nINFO: GET /health 200\n", prefix) == port


def test_a_line_still_being_written_names_no_port():
    text = f"INFO:     {UVICORN_STARTUP} http://127.0.0.1:4312"
    assert port_in(text, UVICORN_STARTUP) is None
    assert port_in(text + "7 (Press CTRL+C)\n", UVICORN_STARTUP) == 43127


def test_a_line_without_the_prefix_names_no_port():
    text = "Starting MCP server 'dispatcher' on http://127.0.0.1:0/mcp\n"
    assert port_in(text, UVICORN_STARTUP) is None


def test_reads_the_startup_line_from_the_log(tmp_path: Path):
    log = tmp_path / "server.log"
    log.write_text(f"INFO:     {UVICORN_STARTUP} http://127.0.0.1:43127 (Press CTRL+C)\n")

    assert bound_port(log, UVICORN_STARTUP, child=_Child(), timeout=5) == 43127


def test_waits_for_a_line_that_arrives_late(tmp_path: Path):
    log = tmp_path / "server.log"
    log.write_text("INFO:     Started server process\n")

    def _late(polls: int) -> None:
        if polls == 3:
            with log.open("a") as fh:
                fh.write(f"{_STARTUP_LINE} http://127.0.0.1:43128\n")

    child = _Child(on_poll=_late)

    assert bound_port(log, _STARTUP_LINE, child=child, timeout=30) == 43128
    assert child.polls >= 3


def test_a_child_that_dies_first_is_reported_with_its_log(tmp_path: Path):
    log = tmp_path / "server.log"
    log.write_text("Traceback (most recent call last):\nOSError: address in use\n")

    with pytest.raises(ContainerExitedError) as excinfo:
        bound_port(log, UVICORN_STARTUP, child=_Child(code=3), timeout=30)

    message = str(excinfo.value)
    assert "exit code: 3" in message
    assert "OSError: address in use" in message
