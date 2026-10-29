"""A per-user web terminal behind an nginx stand-in, and the landing page it returns to.

:func:`launch_terminal` serves the hub with ``OSPREY_TERMINAL_{USER,LANDING_URL}``
set, reached under ``/u/<user>/`` the way nginx would forward it, and
:func:`launch_landing` is the second origin that stands in for the operator
landing page. The logout and session-persistence browser suites share them.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

from osprey.interfaces._serving import run_app_server
from tests.interfaces._hub import LONG_LIVED_SHELL, hub_app


def nginx_stand_in(app, prefix: str):
    """ASGI wrapper reproducing nginx's prefix contract for a per-user app.

    In a real multi-user deployment the browser talks to nginx at
    ``/u/<user>/…`` and nginx strips that prefix before proxying, so the app
    only ever sees bare paths (see the ``root_path`` note in ``create_app``).
    This wrapper is that stripping proxy: http/websocket scopes whose path
    starts with the prefix are forwarded with the prefix removed; the
    ``lifespan`` scope passes through untouched so the inner app's startup
    still runs (a ``starlette`` ``Mount`` would swallow it).
    """

    async def asgi(scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            if scope["path"] == prefix:  # bare /u/<user>, as nginx's trailing-slash redirect
                scope = dict(scope, path="/")
            elif scope["path"].startswith(f"{prefix}/"):
                scope = dict(scope, path=scope["path"][len(prefix) :])
        await app(scope, receive, send)

    return asgi


@contextmanager
def launch_terminal(
    tmp_path, monkeypatch, *, terminal_user: str = "", landing_url: str = ""
) -> Iterator[str]:
    """Launch the web terminal with ``OSPREY_TERMINAL_{USER,LANDING_URL}`` set.

    The env vars are read by ``create_app`` (app.py:443-444) into ``app.state``,
    so they MUST be in ``os.environ`` before the factory runs. Companion backends
    (artifact server, panels) are bypassed via the same patch set the other
    web-terminal browser suites use.

    With a ``terminal_user`` the SPA self-addresses under ``/u/<user>/`` and the
    app serves bare paths, so the pair only functions behind a prefix-stripping
    proxy; the server is wrapped in ``nginx_stand_in`` and the yielded base URL
    carries the prefix, exactly as a user would reach it through nginx.
    """
    monkeypatch.chdir(tmp_path)
    env = {
        "OSPREY_TERMINAL_USER": terminal_user,
        "OSPREY_TERMINAL_LANDING_URL": landing_url,
    }
    with hub_app(tmp_path, patch.dict(os.environ, env), shell_command=LONG_LIVED_SHELL) as app:
        if terminal_user:
            prefix = f"/u/{terminal_user}"
            with run_app_server(nginx_stand_in(app, prefix)) as server_url:
                yield f"{server_url}{prefix}"
        else:
            with run_app_server(app) as base_url:
                yield base_url


@contextmanager
def launch_landing() -> Iterator[str]:
    """A trivial second origin standing in for the operator landing page."""
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse

    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def _root() -> str:
        return (
            "<!doctype html><html><body><h1 id='landing-marker'>OSPREY LANDING</h1></body></html>"
        )

    with run_app_server(app) as base_url:
        yield base_url
