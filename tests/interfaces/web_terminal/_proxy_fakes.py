"""Shared fakes for the panel reverse-proxy suites.

The proxy route suites each carried their own copy of the same upstream
doubles and the same app boot. This module centralizes them: the websocket
upstream (:class:`_FakeConnect` standing in for either connect type the proxy
picks from, :func:`_patch_connect` installing it for both, and the
:class:`_FakeUpstreamSocket` it opens), the streamed HTTP upstream
(:class:`_FakeStreamResponse`), the header-case helper :func:`_lower`, and
:func:`panel_app`, the whole app booted over a watched directory with the
universal panels plus the custom panels a test declares. The page-stamping
suites relay :data:`PANEL_DOCUMENT` through the real app with
:func:`relay_panel_response` and read the stamps off :func:`root_tag`.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.app import UNIVERSAL_PANELS, create_app
from tests.interfaces.web_terminal._started_app import terminal_client


def _lower(headers):
    return {k.lower(): v for k, v in headers.items()}


class _FakeUpstreamSocket:
    """A websocket upstream that stays open until the relay task is cancelled."""

    def __init__(self):
        self.sent: list[object] = []

    async def send(self, data):
        self.sent.append(data)

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.Event().wait()  # pragma: no cover - cancelled at teardown
        raise AssertionError("unreachable")


class _FakeConnect:
    """Stands in for a websocket connect type, recording the handshake arguments."""

    def __init__(self):
        self.target = None
        self.kwargs = None
        self.refuses_redirects = False

    def __call__(self, target, **kwargs):
        self.target = target
        self.kwargs = kwargs
        return self

    async def __aenter__(self):
        return _FakeUpstreamSocket()

    async def __aexit__(self, *exc_info):
        return False


@contextlib.contextmanager
def _patch_connect(fake):
    """Replace both connect types the WS proxy picks from with one *fake*."""
    with (
        patch("websockets.connect", fake),
        patch("osprey.interfaces.web_terminal.routes.proxy._RedirectRefusingConnect", fake),
    ):
        yield fake


class _FakeStreamResponse:
    """A streamed upstream response, as ``client.send(stream=True)`` returns one."""

    def __init__(self, *, status_code=200, headers=None, chunks=(b"data: hello\n\n",)):
        self.status_code = status_code
        default = {"content-type": "text/event-stream"}
        self.headers = httpx.Headers(default if headers is None else headers)
        self._chunks = chunks
        self.closed = False

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


def panel_app(
    workspace_dir: Path, custom_panels: list[dict]
) -> Iterator[tuple[FastAPI, TestClient]]:
    """Yield ``(app, client)`` for the whole app, lifespan run, with *custom_panels* declared.

    The built-in roster is :data:`UNIVERSAL_PANELS`; *workspace_dir* is the
    watched directory. A generator, so a fixture hands it on with
    ``yield from panel_app(...)``.
    """
    enabled = set(UNIVERSAL_PANELS)
    with (
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ),
        patch(
            "osprey.interfaces.web_terminal.app._load_panel_config",
            return_value=(enabled, custom_panels, None),
        ),
    ):
        app = create_app(shell_command="echo")
        with TestClient(app) as client:
            yield app, client


#: A panel document of the shape the companion servers serve.
PANEL_DOCUMENT = (
    '<!DOCTYPE html>\n<html lang="en" data-browse-orient="row">'
    "<head><title>panel</title></head><body></body></html>"
)


def root_tag(body: str) -> str:
    """The ``<html>`` start tag of a relayed document."""
    match = re.search(r"<html\b[^>]*>", body)
    assert match, "relayed document has no <html> element"
    return match.group(0)


def relay_panel_response(tmp_path, terminal_user, path, body, content_type):
    """Relay one panel response through the real app as *terminal_user*'s mount.

    ``None`` removes ``OSPREY_TERMINAL_USER`` entirely. The env patch wraps
    ``create_app`` and the lifespan both: the URL prefix is computed at
    construction and the terminal user is captured during the lifespan.

    Returns:
        ``(relayed_text, headers_the_upstream_received)``.
    """
    workspace = tmp_path / "_agent_data"
    workspace.mkdir()
    custom = [{"id": "my-dash", "label": "DASH", "url": "http://localhost:9000"}]
    env = {} if terminal_user is None else {"OSPREY_TERMINAL_USER": terminal_user}
    captured: dict[str, str] = {}

    # ``httpx.AsyncClient.request``'s signature: the proxy names every field it sends.
    async def fake_request(*, method, url, headers, content, follow_redirects=True):  # noqa: ARG001
        captured.update(headers)
        return httpx.Response(status_code=200, text=body, headers={"content-type": content_type})

    with patch.dict("os.environ", env):
        if terminal_user is None:
            os.environ.pop("OSPREY_TERMINAL_USER", None)
        with terminal_client(
            workspace,
            patch(
                "osprey.interfaces.web_terminal.app._load_panel_config",
                return_value=(set(UNIVERSAL_PANELS), custom, None),
            ),
        ) as client:
            client.app.state.proxy_client.request = AsyncMock(side_effect=fake_request)
            response = client.get(path)
            assert response.status_code == 200
            return response.text, captured
