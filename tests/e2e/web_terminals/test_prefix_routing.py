"""Master end-to-end proof of THE prefix contract (Task 2.1) across the whole
Web Terminal SPA, exercised through the REAL FastAPI app.

Multi-user deployments run one Web Terminal container per user behind a shared
nginx front door (``osprey.deployment.web_terminals``), each mounted at a
per-user path, ``/u/<user>/``. Task 2.1's ``compute_url_prefix()``
(``osprey.interfaces.web_terminal.app``) computes that prefix once from
``OSPREY_TERMINAL_USER``, and every surface downstream of it — the served HTML
documents, the panel/API JSON payloads, the panel reverse proxy, the websocket
routes, and nginx's own routing — must line up so the SPA behaves identically
to the Phase-1 direct-port root, just relocated under ``/u/<user>/``.

That contract is already covered piecemeal, unit-by-unit:

- ``tests/interfaces/web_terminal/test_prefix_injection.py`` — the HTML/import-map
  injection contract, plus the bare-path StaticFiles regression (see below).
- ``tests/interfaces/web_terminal/test_panels_prefix.py`` — panel URL producers
  (the five ``*_server_config`` endpoints, ``GET /api/panels``, panel
  register/focus broadcasts).
- ``tests/interfaces/web_terminal/test_proxy_rewrite.py`` — the panel reverse
  proxy's outer-prefix content rewriting and ``x-forwarded-prefix`` header.
- ``tests/deployment/web_terminals/test_render.py`` / ``test_nginx_validate.py``
  — the generator's nginx routing/redirect output, including a real ``nginx -t``
  run.

THIS file is the integration proof that all of those surfaces line up together
on the real app, in one place, end to end — not a replacement for any of them.

THE KEY ARCHITECTURAL FACT this file's static-asset coverage exists to pin:
nginx.conf.j2 strips the ``/u/<user>`` prefix before proxying to the container
(``location /u/<user>/ { proxy_pass http://<bind_host>:<web>/; }`` — trailing
slashes on both sides strip it), so the app itself **always** receives BARE
paths (``/static/...``, ``/design-system/...``, ``/api/...``, ``/ws/...``,
``/panel/<id>/...``) regardless of whether a prefix is configured. The prefix
only ever shows up in the app's own *generated content* — the injected
``window.__OSPREY_PREFIX__``/import map, and the URLs embedded in JSON/SSE
payloads by ``routes/panels.py`` and ``routes/proxy.py`` (which read
``compute_url_prefix()`` directly, never from the request path). Accordingly,
``create_app()`` deliberately never sets ``FastAPI(root_path=url_prefix)`` —
doing so once forced every real (bare, nginx-forwarded) StaticFiles request to
404 (Starlette's ``Mount``/``StaticFiles`` re-derive their route path from
``scope["root_path"]``, and a non-empty forced root_path makes that
re-derivation expect the prefix to already be in the path). That regression is
pinned at the unit level in ``test_prefix_injection.py``; this file's
``TestStaticAssetUnderPrefix`` guards the same fact end to end, through the
real asset mounts, alongside every other surface.

Every check below runs fully in-process via FastAPI's ``TestClient`` — no
docker, no provider credentials, no tokens — so the whole file is a fast,
deterministic default-gate test. The one surface that is genuinely nginx-only
(item 7, the trailing-slash bookmark redirect) is verified against the
rendered ``nginx.conf`` text rather than a live nginx process; a real
``nginx -t`` validation of that exact directive already exists in
``tests/deployment/web_terminals/test_nginx_validate.py``, so it is not
duplicated here.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from osprey.deployment.web_terminals.render import render_web_terminals
from osprey.interfaces.web_terminal.app import UNIVERSAL_PANELS, create_app

pytestmark = [pytest.mark.e2e, pytest.mark.e2e_smoke]

_ALICE = "alice"
_PREFIX = f"/u/{_ALICE}"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def workspace_dir(tmp_path):
    """A temp workspace dir for the file watcher -- content is irrelevant here."""
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    return ws


def _panel_config():
    """One url-backed custom panel ('my-dash') -- gives the /api/panels (item 4)
    and panel-proxy (item 6) checks something concrete to prefix, mirroring the
    fixture shape ``test_panels_prefix.py``/``test_proxy_rewrite.py`` use."""
    enabled = set(UNIVERSAL_PANELS)
    custom = [{"id": "my-dash", "label": "DASH", "url": "http://localhost:9000"}]
    return enabled, custom, None


def _make_client(workspace_dir):
    """Build the real app with the fixture panel config. Caller controls
    ``OSPREY_TERMINAL_USER`` (via monkeypatch) before entering the context."""
    enabled, custom, default = _panel_config()
    with (
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(workspace_dir)},
        ),
        patch(
            "osprey.interfaces.web_terminal.app._load_panel_config",
            return_value=(enabled, custom, default),
        ),
    ):
        app = create_app(shell_command="echo")
        with TestClient(app) as client:
            # Real FileEventBroadcaster replaced with a spy so panel-focus/
            # register broadcasts (item 6) are directly observable.
            app.state.broadcaster = MagicMock()
            yield app, client


@pytest.fixture
def alice_client(workspace_dir, monkeypatch):
    """The multi-user shape: OSPREY_TERMINAL_USER=alice -> /u/alice prefix."""
    monkeypatch.setenv("OSPREY_TERMINAL_USER", "alice")
    yield from _make_client(workspace_dir)


@pytest.fixture
def plain_client(workspace_dir, monkeypatch):
    """The Phase-1 shape: no OSPREY_TERMINAL_USER -> empty prefix, unchanged
    behavior. Kept alongside every prefixed check below (cheap) so this file
    also proves the single-user topology never regressed."""
    monkeypatch.delenv("OSPREY_TERMINAL_USER", raising=False)
    yield from _make_client(workspace_dir)


# ---------------------------------------------------------------------------
# 1 & 2: HTML shell -- window.__OSPREY_PREFIX__/import map + prefixed entrypoint
# ---------------------------------------------------------------------------


class TestHtmlShellUnderPrefix:
    """The served index page carries the full injection contract (Task 2.1),
    exercised through the real app rather than a bare template render. The
    exhaustive per-page/per-asset matrix (session.html, safety.html, every
    head asset) is ``test_prefix_injection.py``'s job; this only needs to
    prove the SAME mechanism holds on the real, fully-wired app.
    """

    def test_prefix_global_and_importmap_baked_into_index(self, alice_client):
        _app, client = alice_client
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.text

        prefix_idx = body.index(f'window.__OSPREY_PREFIX__ = "{_PREFIX}";')
        importmap_idx = body.index('type="importmap"')
        assert f'"/design-system/": "{_PREFIX}/design-system/"' in body
        assert f'"/static/": "{_PREFIX}/static/"' in body

        # The prefix global and import map must land before any module script
        # runs -- module specifiers resolve using whatever's in place already.
        first_module_idx = body.find('type="module"')
        assert first_module_idx != -1
        assert prefix_idx < first_module_idx
        assert importmap_idx < first_module_idx

    def test_module_entrypoint_src_is_prefixed(self, alice_client):
        """The import map alone cannot retarget the entrypoint's own ``src``
        (see ``_prefixed()``'s docstring) -- it must be explicitly rewritten."""
        _app, client = alice_client
        body = client.get("/").text
        assert f'src="{_PREFIX}/static/js/app.js"' in body
        assert 'src="/static/js/app.js"' not in body

    def test_empty_prefix_parity(self, plain_client):
        """Same page, no OSPREY_TERMINAL_USER -> byte-identical Phase-1 shape."""
        _app, client = plain_client
        body = client.get("/").text
        assert 'window.__OSPREY_PREFIX__ = "";' in body
        assert '"/static/": "/static/"' in body
        assert 'src="/static/js/app.js"' in body


# ---------------------------------------------------------------------------
# 3: /static asset -- reachable on the BARE path nginx actually forwards
# ---------------------------------------------------------------------------


class TestStaticAssetUnderPrefix:
    """StaticFiles-mounted assets (``/static``, ``/design-system``) must serve
    on their bare path even when a prefix is configured -- that bare path is
    the ONLY form the app ever actually receives in the real deployment (nginx
    strips the prefix before proxying; see this module's docstring). This is
    the exact regression a forced ``FastAPI(root_path=...)`` previously
    introduced (fixed; pinned at the unit level in
    ``test_prefix_injection.py::test_static_mount_served_on_bare_path_when_prefix_set``).
    """

    def test_bare_static_asset_reachable_with_prefix_configured(self, alice_client):
        _app, client = alice_client
        resp = client.get("/static/js/app.js")
        assert resp.status_code == 200
        assert resp.text  # a real, non-empty file was served

    def test_bare_design_system_asset_reachable_with_prefix_configured(self, alice_client):
        _app, client = alice_client
        resp = client.get("/design-system/js/theme-boot.js")
        assert resp.status_code == 200
        assert resp.text

    def test_literal_prefixed_static_path_is_not_a_route(self, alice_client):
        """Regression guard on the fix itself: the app must NOT tolerate the
        literal ``/u/alice/static/...`` form either -- nginx never sends that
        shape to the container, and re-introducing any accidental root_path-style
        tolerance here would silently mask the bare-path regression above the
        next time something reintroduces it."""
        _app, client = alice_client
        assert client.get(f"{_PREFIX}/static/js/app.js").status_code == 404

    def test_bare_static_asset_reachable_empty_prefix(self, plain_client):
        _app, client = plain_client
        assert client.get("/static/js/app.js").status_code == 200


# ---------------------------------------------------------------------------
# 4: /api -- a real route, reached on its bare path, embeds the prefix in JSON
# ---------------------------------------------------------------------------


class TestApiUnderPrefix:
    """``GET /api/panels`` needs no external services (health-checked servers
    are all optional/absent here) and its response embeds panel URLs via the
    exact same ``compute_url_prefix()``-reading path production traffic uses
    (``_browser_panel_url`` in ``routes/panels.py``)."""

    def test_bare_api_route_prefixes_custom_panel_url(self, alice_client):
        _app, client = alice_client
        resp = client.get("/api/panels")
        assert resp.status_code == 200
        payload = resp.json()
        by_id = {cp["id"]: cp for cp in payload["custom"]}
        assert by_id["my-dash"]["url"] == f"{_PREFIX}/panel/my-dash"

    def test_bare_api_route_empty_prefix_unchanged(self, plain_client):
        _app, client = plain_client
        resp = client.get("/api/panels")
        assert resp.status_code == 200
        by_id = {cp["id"]: cp for cp in resp.json()["custom"]}
        assert by_id["my-dash"]["url"] == "/panel/my-dash"


# ---------------------------------------------------------------------------
# 5: /ws handshake -- routed on its bare path; the literal prefixed form isn't
# ---------------------------------------------------------------------------


class _FakePtySession:
    """Minimal PtySession substitute -- stays alive, produces no output.
    Mirrors ``test_ws_resume_confirm.py``'s fixture of the same name; only
    what ``terminal_ws`` touches is implemented."""

    def __init__(self):
        self._alive = True

    @property
    def is_alive(self):
        return self._alive

    @property
    def exit_code(self):
        return None if self._alive else 0

    def start(self, initial_rows=24, initial_cols=80, extra_env=None, cwd=None):
        pass

    def resize(self, rows, cols):
        pass

    def write_input(self, data):
        pass

    def terminate(self):
        self._alive = False

    async def read_output(self):
        try:
            while self._alive:
                await asyncio.sleep(0.05)
        except (asyncio.CancelledError, GeneratorExit):
            return
        if False:  # pragma: no cover -- makes this an async generator
            yield b""


def _patch_spawn(app):
    """Replace ``_spawn_session`` so no real PTY/subprocess is created."""
    app.state.pty_registry._spawn_session = lambda *_a, **_kw: _FakePtySession()


def _recv_json(ws, msg_type: str, max_frames: int = 30) -> dict:
    """Receive frames until a JSON message of ``msg_type`` arrives (skipping
    binary frames), mirroring ``test_ws_resume_confirm.py``'s helper."""
    for _ in range(max_frames):
        raw = ws.receive()
        if "text" in raw:
            data = json.loads(raw["text"])
            if data.get("type") == msg_type:
                return data
    raise AssertionError(f"'{msg_type}' not received within {max_frames} frames")


@pytest.fixture
def ws_app(tmp_path, monkeypatch):
    """An isolated app for driving a real websocket handshake. Uses the
    ``mode=resume`` path (not the bare ``mode=new`` default): the ``new``-session
    path only sends ``session_info`` when discovery finds a NEWLY-created
    session file, so with ``discover_new_session`` patched to return ``None``
    (no real filesystem poll) it would never send anything and this test would
    hang forever waiting on a frame that never arrives. The resume path, by
    contrast, always confirms synchronously (real or discovered id) -- the
    exact, proven shape ``test_ws_resume_confirm.py`` already relies on.
    """
    monkeypatch.setenv("OSPREY_TERMINAL_USER", "alice")
    ws_dir = tmp_path / "_agent_data"
    ws_dir.mkdir()
    with (
        patch(
            "osprey.interfaces.web_terminal.app._load_web_config",
            return_value={"watch_dir": str(ws_dir)},
        ),
        patch(
            "osprey.interfaces.web_terminal.routes.websocket.SessionDiscovery.discover_new_session",
            return_value=None,
        ),
    ):
        app = create_app(shell_command="echo", project_dir=str(tmp_path))
        with TestClient(app) as client:
            _patch_spawn(app)
            yield app, client


class TestWebSocketHandshakeUnderPrefix:
    """The websocket route is a plain ``@router.websocket(...)`` route (not a
    Mount), so -- unlike the StaticFiles mounts above -- it is unaffected by
    the root_path/Mount interaction; it simply never sees the prefix in its
    path (nginx strips it) and is registered at its bare path only."""

    def test_bare_ws_path_completes_handshake_under_prefix(self, ws_app):
        _app, client = ws_app
        session_id = str(uuid.uuid4())
        url = f"/ws/terminal?session_id={session_id}&mode=resume"
        with client.websocket_connect(url) as ws:
            ws.send_json({"type": "resize", "cols": 80, "rows": 24})
            msg = _recv_json(ws, "session_info")
            assert msg["session_id"]

    def test_literal_prefixed_ws_path_is_not_routed(self, ws_app):
        """The app registers ``/ws/terminal`` only -- a literal ``/u/alice/ws/...``
        request (which nginx never actually sends; it strips the prefix first)
        must not resolve to a route at all."""
        _app, client = ws_app
        session_id = str(uuid.uuid4())
        url = f"{_PREFIX}/ws/terminal?session_id={session_id}&mode=resume"
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(url):
                pass


# ---------------------------------------------------------------------------
# 6: Panel iframe + internal asset -- proxy rewrite carries the outer prefix
# ---------------------------------------------------------------------------


class TestPanelProxyAndFocusUnderPrefix:
    """Exercises ``routes/proxy.py`` (a nested panel asset gets the outer
    prefix) and ``routes/panels.py``'s ``set_panel_focus`` broadcast (a
    relative panel URL gets prefixed; an absolute one passes through
    unchanged) -- both through the real, fully-wired app rather than a
    router-only harness."""

    def test_internal_panel_asset_gets_outer_prefix(self, alice_client):
        app, client = alice_client
        js_body = 'var x = "/static/js/foo.js";'

        captured_headers: dict = {}

        async def capturing_request(*, method, url, headers, content):
            captured_headers.update(headers)
            return httpx.Response(
                status_code=200,
                text=js_body,
                headers={"content-type": "application/javascript"},
            )

        app.state.proxy_client.request = AsyncMock(side_effect=capturing_request)

        resp = client.get("/panel/my-dash/static/js/gallery.js")

        assert resp.status_code == 200
        assert f'"{_PREFIX}/panel/my-dash/static/js/foo.js"' in resp.text
        assert captured_headers.get("x-forwarded-prefix") == f"{_PREFIX}/panel/my-dash"

    def test_panel_focus_relative_url_gets_prefixed(self, alice_client):
        app, client = alice_client
        resp = client.post("/api/panel-focus", json={"panel": "my-dash", "url": "/panel/my-dash"})
        assert resp.status_code == 200
        event = app.state.broadcaster.broadcast.call_args[0][0]
        assert event["url"] == f"{_PREFIX}/panel/my-dash"

    def test_panel_focus_absolute_url_passes_through_unchanged(self, alice_client):
        """An already-absolute (external) panel URL must never be corrupted
        by prefixing -- it isn't served through this container at all."""
        app, client = alice_client
        resp = client.post(
            "/api/panel-focus",
            json={"panel": "my-dash", "url": "https://grafana.lan:3000/d/abc"},
        )
        assert resp.status_code == 200
        event = app.state.broadcaster.broadcast.call_args[0][0]
        assert event["url"] == "https://grafana.lan:3000/d/abc"


# ---------------------------------------------------------------------------
# 7: Trailing-slash bookmark redirect -- nginx-only, verified via rendered conf
# ---------------------------------------------------------------------------


def _single_user_facility_config() -> dict:
    """A minimal-but-complete facility config for one user ("alice"), matching
    the shape ``tests/deployment/web_terminals/test_render.py``'s ``_config()``
    builds -- every field ``render_web_terminals()`` reads, nothing more."""
    return {
        "facility": {"name": "Demo Light Source", "prefix": "dls", "timezone": "UTC"},
        "registry": {"url": "git.dls.example.org:5050/physics/production/dls-profiles"},
        "deploy": {"host": "dls-deploy", "fqdn": "dls-deploy.dls.example.org"},
        "modules": {
            "web_terminals": {
                "enabled": True,
                "nginx_port": 9080,
                "web_base_port": 9091,
                "artifact_base_port": 9291,
                "ariel_base_port": 9391,
                "lattice_base_port": 9491,
                "users": [_ALICE],
            }
        },
    }


class TestTrailingSlashRedirectRenderedByNginx:
    """The no-trailing-slash bookmark (``/u/alice``) is redirected to
    ``/u/alice/`` by NGINX, not the app -- there is no such route in
    ``app.py``. That directive is proven syntactically valid nginx (a real
    ``nginx -t`` run) by ``tests/deployment/web_terminals/test_nginx_validate.py``;
    this only needs to prove the generator actually emits it for this user,
    so the in-process default gate stays docker-free.
    """

    def test_nginx_conf_redirects_bare_user_path_to_trailing_slash(self):
        artifacts = render_web_terminals(_single_user_facility_config())
        nginx_conf = artifacts["nginx/nginx.conf"]

        assert f"location = {_PREFIX} {{" in nginx_conf
        assert f"return 301 {_PREFIX}/;" in nginx_conf
        # And the reverse-proxy route itself strips the prefix before it
        # reaches the upstream (trailing slash on both sides of proxy_pass).
        assert f"location {_PREFIX}/ {{" in nginx_conf
        assert "proxy_pass http://127.0.0.1:9091/;" in nginx_conf
