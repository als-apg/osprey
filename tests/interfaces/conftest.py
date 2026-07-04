"""Shared fixtures for real-browser Playwright suites under ``tests/interfaces/``.

Extracted from the byte-identical boilerplate duplicated across
``design_system/test_behavioral.py``, ``design_system/test_visual.py``, and
``web_terminal/test_panels_browser.py``: the free-port/wait-for-port helpers,
the generic FastAPI-on-a-background-thread launcher, and the function-scoped
Playwright browser fixture. Suite-specific live-server wrappers (hub launchers
with their own patch sets) stay local to each file.
"""

from __future__ import annotations

import socket
import threading
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytest
import uvicorn
from fastapi import FastAPI

if TYPE_CHECKING:
    from collections.abc import Iterator

    from playwright.sync_api import Browser

# ---------------------------------------------------------------------------
# Playwright availability guard
# ---------------------------------------------------------------------------

try:
    from playwright.sync_api import sync_playwright

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers: ports, uvicorn lifecycle
# ---------------------------------------------------------------------------


def _free_port() -> int:
    """Return an unused TCP port on 127.0.0.1."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_for_port(port: int, timeout: float = 10.0) -> None:
    """Block until the server accepts TCP connections, or raise RuntimeError."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"Server did not become ready on port {port} within {timeout}s")


@contextmanager
def _run_app_server(app: FastAPI) -> Iterator[str]:
    """Run any FastAPI app on a free port in a background thread.

    Yields:
        The server's base URL, e.g. ``"http://127.0.0.1:54321"``.
    """
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    _wait_for_port(port)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        t.join(timeout=5)


@contextmanager
def _apply_all(patches: list) -> Iterator[None]:
    """Enter a variable-length list of ``unittest.mock`` patch objects together.

    Starts each patch in order and stops them in reverse on exit, so a suite can
    aggregate a patch set (some entries conditional) and apply it as a single
    context around ``create_app()`` + the server lifespan.
    """
    for p in patches:
        p.start()
    try:
        yield
    finally:
        for p in reversed(patches):
            p.stop()


# ---------------------------------------------------------------------------
# Function-scoped chromium fixture
# ---------------------------------------------------------------------------
#
# Intentionally function-scoped (not session-scoped): sync_playwright() runs
# an asyncio event loop on the main thread while alive, which makes
# asyncio.Runner.run() raise "cannot be called from a running event loop" in
# any pytest-asyncio async tests that share the session.  Closing and
# restarting playwright per test (~0.5s overhead) is cheaper than the
# ordering-dependent failures that a session-scoped fixture would cause.


@pytest.fixture
def chromium_browser() -> Iterator[Browser]:
    """Function-scoped Playwright browser. Skips if chromium binary is absent."""
    if not _PLAYWRIGHT_AVAILABLE:
        pytest.skip("playwright package not installed")

    # sync_playwright().start() spins up an asyncio loop on the main thread.  It
    # MUST be stopped on every exit path — including the skip taken when the
    # chromium binary is absent (the usual CI condition) and a failing test body.
    # Leaking it makes every later asyncio.run()/pytest-asyncio test in the
    # session raise "Runner.run() cannot be called from a running event loop".
    pw = sync_playwright().start()
    try:
        browser = pw.chromium.launch(headless=True)
    except Exception as exc:  # pragma: no cover
        pw.stop()
        pytest.skip(f"Chromium binary not available: {exc}")
        return  # unreachable — present only to satisfy type checkers

    try:
        yield browser
    finally:
        browser.close()
        pw.stop()
