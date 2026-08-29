"""Shared fixtures for the bluesky-web sidecar suite: a live uvicorn server.

The per-module suites in this directory exercise the composed app in-process
through ``TestClient``. The ``bluesky_live_server`` fixture below instead runs
the very same app object on a real localhost port, for tests that need actual
HTTP: the fixture's own smoke test (``test_browser_fixture_smoke.py``) and the
real-browser Playwright suites (``-m browser``) that navigate to
``{base_url}/bluesky/`` and type into the plan form's channel fields.

Two things make this fixture more than ``run_app_server(app)``:

- **A stand-in project directory.** The ``/channels`` route resolves the
  channel catalog as a sibling of the config named by ``CONFIG_FILE`` at
  request time, so the fixture writes a temp ``config.yml`` + ``channels.json``
  and points ``CONFIG_FILE`` at it — the deployed container layout, exactly as
  ``test_channels_route.py``'s ``project_dir`` reproduces it. Launching with
  ``catalog=None`` writes no catalog, which is the "this deployment has no
  catalog" 404 path.
- **A stubbed bridge.** The panel boots by fetching ``/plans``,
  ``/bridge/health``, ``/draft``, ``/queue``, ``/runs`` and the two SSE relays
  through the sidecar's proxy routers, all of which read
  ``request.app.state.client``/``bridge_url`` at request time. After the
  server has started (the real ``_lifespan`` has run), the fixture swaps in an
  ``httpx.MockTransport`` client — the ``test_app_integration.py`` seam — whose
  ``/plans`` answer carries one plan with role-tagged channel fields
  (``x-channel-role``, plus ``x-widget: channel-list`` on the list field), so
  the rendered form exposes both the single-channel and channel-list controls
  the autocomplete tests exercise. The lifespan closes its own local client
  reference on shutdown, never ``app.state.client``, so the swap leaks no
  double-close.

The app is a module-level singleton (no ``create_app()``), so the swapped
state persists on the object between tests in a worker. The fixture is
therefore function-scoped and re-sets both attributes on every launch rather
than trying to restore them; env vars go through ``monkeypatch``, and the root
conftest's ``restore_environ`` clears ``CONFIG_FILE`` around every test, so
nothing leaks sideways either.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import TYPE_CHECKING

import httpx
import pytest

from osprey.interfaces._serving import run_app_server as _run_app_server
from osprey.interfaces.bluesky_web.app import app
from tests.interfaces.conftest import use_process_web_credentials

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from contextlib import AbstractContextManager
    from pathlib import Path

# ---------------------------------------------------------------------------
# Stub data: the channel catalog and the bridge's plan catalog
# ---------------------------------------------------------------------------

# Default build-time catalog served at /channels — sorted, like `osprey build`
# writes it, with enough prefix variety for autocomplete tests to get
# distinguishable matches out of a fuzzy filter.
DEFAULT_CHANNEL_CATALOG: list[str] = [
    "BR:DIAG:BPM:01:POSITION:X",
    "BR:DIAG:BPM:01:POSITION:Y",
    "BR:DIAG:BPM:02:POSITION:X",
    "BR:MAG:HCM:01:CURRENT:SP",
    "BR:MAG:VCM:01:CURRENT:SP",
    "SR:DIAG:BPM:07:POSITION:Y",
    "SR:MAG:QF:03:CURRENT:SP",
    "SR:VAC:ION:12:PRESSURE",
]

STUB_BRIDGE_URL = "http://bridge.test"

STUB_PLAN_NAME = "channel_probe"

# One plan schema shaped like the bridge's own `PARAMS.model_json_schema()`
# output for a role-typed model (see `bluesky_bridge.plan_fields`): the role
# annotation sits on the FIELD, beside `type`/`items`, and the presentation
# hint rides alongside it. `target` is the plain-string channel field
# (schema-form's `buildString`); `monitors` is the channel-list field
# (`x-widget: channel-list` -> `buildChannelList`); `setpoint` pins that
# non-channel fields render untouched next to them.
STUB_PLAN_SCHEMA: dict = {
    "title": "PARAMS",
    "type": "object",
    "properties": {
        "target": {
            "title": "Target",
            "type": "string",
            "x-channel-role": "movable",
        },
        "monitors": {
            "title": "Monitors",
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "x-channel-role": "readable",
            "x-widget": "channel-list",
        },
        "setpoint": {"title": "Setpoint", "type": "number", "default": 0.0},
    },
    "required": ["target", "monitors"],
}

# The bridge's `GET /plans` catalog-entry shape, as pinned by
# `tests/integration/test_bluesky_queue_contract.py`: exactly
# {name, description, schema, metadata, provenance}, with metadata carrying
# the three authoring-declared fields.
STUB_PLANS: list[dict] = [
    {
        "name": STUB_PLAN_NAME,
        "description": "Stub plan exercising role-tagged channel fields.",
        "schema": STUB_PLAN_SCHEMA,
        "metadata": {
            "name": STUB_PLAN_NAME,
            "description": "Stub plan exercising role-tagged channel fields.",
            "writes": True,
        },
        "provenance": "shipped",
    }
]


def _stub_bridge(request: httpx.Request) -> httpx.Response:
    """Answer the read endpoints the panel touches on boot, harmlessly.

    Only ``/plans`` carries content the tests assert on; everything else
    returns the smallest body of the real endpoint's shape so the panel's boot
    fetches succeed without a bridge. The SSE relays get an immediately-ended
    empty stream — the panel's reconnect logic tolerates that, and nothing in
    these tests consumes events.
    """
    path = request.url.path
    if path == "/plans":
        return httpx.Response(200, json=STUB_PLANS)
    if path == f"/plans/{STUB_PLAN_NAME}/source":
        return httpx.Response(
            200, json={"name": STUB_PLAN_NAME, "source": "def channel_probe(): ..."}
        )
    if path == "/health":
        # The sidecar's /bridge/health relays the bridge's /health, capability
        # record and all. can_execute=True keeps the panel free of the
        # browse-only banner — one less overlay for browser tests.
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "capability": {"can_execute": True, "reason": None, "detail": None},
            },
        )
    if path == "/devices":
        return httpx.Response(200, json={"devices": [], "total": 0, "offset": 0, "limit": 500})
    if path == "/runs":
        return httpx.Response(200, json=[])
    if path == "/draft":
        return httpx.Response(200, json={"draft": None, "revision": 0})
    if path == "/queue":
        return httpx.Response(
            200,
            json={"status": {"available": True}, "items": [], "running_item": None},
        )
    if path in ("/draft/events", "/queue/events"):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=b"")
    return httpx.Response(404, json={"detail": f"stub bridge has no {path}"})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_ambient_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Mirrors the per-module isolation fixtures in this directory: point config
    # resolution at a config.yml that does not exist, so ambient repo/user
    # config can never leak a real launch token (or bridge URL) into the
    # lifespan's resolve_bridge_url() during server startup. Harmlessly
    # duplicates the module-level fixtures where both apply.
    monkeypatch.setenv("OSPREY_CONFIG", str(tmp_path / "does-not-exist.yml"))
    monkeypatch.delenv("BLUESKY_LAUNCH_TOKEN", raising=False)


@pytest.fixture
def bluesky_live_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Callable[..., AbstractContextManager[str]]:
    """Factory: launch the composed sidecar app on a free port, bridge stubbed.

    Returns a context manager factory::

        def test_something(bluesky_live_server):
            with bluesky_live_server() as base_url:
                ...  # httpx / page.goto against base_url

    Args (of the returned callable):
        catalog: Channel addresses to serve at ``/channels``
            (``DEFAULT_CHANNEL_CATALOG`` by default), or ``None`` to serve no
            ``channels.json`` at all — the 404 "deployment has no catalog"
            variant.

    Yields (inside the ``with``):
        The server's base URL, e.g. ``"http://127.0.0.1:54321"``.
    """

    @contextmanager
    def _launch(catalog: list[str] | None = DEFAULT_CHANNEL_CATALOG) -> Iterator[str]:
        project = tmp_path / "project"
        project.mkdir(exist_ok=True)
        (project / "config.yml").write_text("connector:\n  type: mock\n")
        catalog_file = project / "channels.json"
        if catalog is None:
            # A relaunch within one test may leave a previous catalog behind;
            # absent must mean absent.
            catalog_file.unlink(missing_ok=True)
        else:
            catalog_file.write_text(json.dumps(catalog))
        # /channels resolves the catalog beside CONFIG_FILE at request time,
        # so this only has to be set before the first request — but setting it
        # before startup matches the deployed container's environment.
        monkeypatch.setenv("CONFIG_FILE", str(project / "config.yml"))

        # `app` here is a module-level singleton, so its cached credential
        # holder goes stale the moment the root conftest resets the process
        # one; without this the gate refuses every seam-authenticated call.
        use_process_web_credentials(app)

        with _run_app_server(app) as base_url:
            # The lifespan has run: swap the bridge client out from under the
            # served app (routers read app.state per request). The lifespan
            # still closes the ORIGINAL client it created — it holds a local
            # reference — so nothing double-closes.
            app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(_stub_bridge))
            app.state.bridge_url = STUB_BRIDGE_URL
            yield base_url

    return _launch
