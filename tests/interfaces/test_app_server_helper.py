"""Tests for ``run_app_server`` — the shared helper behind the browser suites.

The helper runs an interface's ``create_app()`` on a throwaway port in a
background thread. Its failure reporting matters as much as its happy path: it
is the entry point for every real-browser test, so when a server does not come
up, the message it raises is the only evidence anyone gets.
"""

from __future__ import annotations

import contextlib
import time

import httpx
import pytest
from fastapi import FastAPI

from osprey.interfaces._serving import run_app_server

# Bound for reporting a startup *failure*. The helper's readiness budget for a
# slow-but-alive startup is far larger (30 s), but a server whose thread died
# must be reported well inside even this small bound, never after burning the
# budget.
READINESS_BUDGET = 10.0


def _app_with_failing_lifespan() -> FastAPI:
    """An app that cannot start: its lifespan raises before yielding."""

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        raise RuntimeError("lifespan exploded")
        yield  # pragma: no cover - unreachable, documents the shape

    return FastAPI(lifespan=lifespan)


class TestStartupFailureReporting:
    """A server that dies must be reported as dead, not as slow."""

    def test_startup_failure_is_reported_without_burning_the_whole_budget(self):
        """A dead server is detected promptly instead of after the full timeout.

        Measured readiness for real interface apps is ~0.1 s (p90 0.11 s), so a
        run that consumes the entire 10 s budget always means the server never
        listened at all. Polling the port blindly cannot tell those apart and
        pays the full timeout for a failure that was knowable immediately.
        """
        # Arrange
        app = _app_with_failing_lifespan()

        # Act
        started = time.monotonic()
        with pytest.raises(RuntimeError) as excinfo:
            with run_app_server(app):
                pass  # pragma: no cover - the context must not be entered
        elapsed = time.monotonic() - started

        # Assert
        assert elapsed < READINESS_BUDGET / 2, (
            f"took {elapsed:.1f}s to report a server that never started; "
            "the helper polled the port blindly instead of noticing the thread exited"
        )
        assert "exited during startup" in str(excinfo.value), (
            f"error must say the server died, not that it was slow; got: {excinfo.value}"
        )


class TestHappyPath:
    def test_yields_a_url_served_by_the_supplied_app(self):
        """The yielded URL reaches the app that was passed in."""
        # Arrange
        app = FastAPI()

        @app.get("/ping")
        def ping() -> dict[str, str]:
            return {"pong": "from-this-app"}

        # Act
        with run_app_server(app) as base_url:
            resp = httpx.get(f"{base_url}/ping", timeout=5.0)

        # Assert — proves the port belongs to *our* server, not a coincidental
        # listener that a bare TCP connect check would have accepted.
        assert resp.status_code == 200
        assert resp.json() == {"pong": "from-this-app"}
