"""Shared fixtures for the web-terminal app tests.

Every test in this directory that drives ``create_app`` through the
``TestClient`` lifespan gets a real :class:`WorkspaceWatcher` — an OS-level
filesystem observer on ``watch_dir``, running on its own thread. That observer
calls ``app.state.broadcaster.broadcast`` for any event it sees, on the very
object tests replace with a mock to capture route broadcasts. A background
thread and the assertion therefore share one counter.

On macOS this is not merely theoretical. watchdog's FSEvents backend defaults
to ``suppress_history=False``, and its own documentation notes the API "may
emit historic events up to 30 sec before the watch was started" — so the
observer replays the ``tmp_path`` creation the fixture itself performed moments
earlier. Delivery is asynchronous, so whether that replay lands inside a test's
assertion window is decided by FSEvents daemon coalescing, not by the test.
That is what made ``test_valid_panel_broadcasts_panel_visibility_event``
intermittently report "Called 2 times" against ``assert_called_once``, and CI
covers ``macos-latest``.

No app test here exercises file watching — the watcher's own tests build it
directly from ``file_watcher``, and the SSE route tests build their own app —
so the observer is stubbed out by default. It is incidental machinery for these
tests, and starting it buys nothing but a 30-second window of foreign events.

Opt back in with ``@pytest.mark.real_workspace_watcher`` when a test genuinely
needs live file watching through the app factory.

The **audit zone** every test here records into is redirected by
``tests/conftest.py::_isolate_audit_zone``, suite-wide rather than per
directory. Any test that drives ``create_app`` files real records —
``HttpAuditMiddleware`` records one ``http_mutation`` line per state-changing
request — and those would otherwise land in the developer's or the runner's own
``var/audit/<identity>/`` ledger, indistinguishable from records of things that
really happened.

The **feedback and bar-items stores** are sited under
``resolve_shared_data_root()``, which reads ``agent_data.base_dir`` anchored on
the project root and never consults ``OSPREY_AGENT_DATA_ROOT``.
``tests/interfaces/conftest.py::_agent_data_root_in_tmp`` stamps that variable
for every test in this tree, and it moves the control-context record only. A
test that needs its own stores patches
``osprey.utils.workspace.resolve_shared_data_root``, the name the lifespan
imports at call time, as ``test_bar_items_routes.py``'s ``client`` fixture
does. A lifespan left unpatched resolves the stores to the checkout's
``var/agent_data``; ``tests/conftest.py::agent_data_never_the_checkout``
diverts that to one throwaway root per worker session, so the stores are then
isolated from the checkout but shared with every other test in the worker.

The second autouse fixture here keeps the panel-register route's deploy-host
check off the real machine, and resets that check's TTL cache between tests
through ``routes.panels.reset_host_addrs_cache()``. It is the same class of
leak-guard — machine state reaching into a test that never asked for it. Its
target constant is exported as ``HOST_ADDRS_TARGET`` for the tests that patch
the helper again from the inside to exercise the check itself.

Two more process-wide memos are reset around every test through their named
seams — the parsed-render memo in ``routes.websocket`` and the once-per-process
no-durable-store notice in ``ownership`` — and the deployment-identity
variables a developer shell may export are unset, so an "unset" row means
unset.

Two non-autouse builders serve the tests that ask for them:
:func:`bar_items_app` boots the whole app with both agent-data stores on
``tmp_path``, and :func:`bare_route_app` mounts routers on a bare ``FastAPI``
carrying the ``app.state`` the lifespan would have seeded.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path, PurePath
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from osprey.interfaces.web_terminal.file_watcher import FileEventBroadcaster


class StubWorkspaceWatcher:
    """Drop-in for :class:`WorkspaceWatcher` that starts no observer thread.

    Accepts exactly the call the lifespan makes (``app.py``'s
    ``WorkspaceWatcher(workspace_dir, broadcaster, concealed=…)``), not the
    whole constructor: ``observer_factory`` is a seam for the watcher's own
    tests, and the lifespan never passes it. This class is the second
    construction site for :class:`WorkspaceWatcher`, so an argument the
    lifespan starts passing has to be accepted here too, or every app test in
    this directory fails at startup with a ``TypeError``.

    It records what the lifespan wired up, so tests can still assert the
    watcher was pointed at the right directory without an OS-level observer
    being involved.
    """

    def __init__(
        self,
        workspace_dir: Path,
        broadcaster: FileEventBroadcaster,
        *,
        concealed: Sequence[PurePath] = (),
    ) -> None:
        self.workspace_dir = workspace_dir
        self.broadcaster = broadcaster
        self.concealed = tuple(concealed)
        self.started = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False


@pytest.fixture(autouse=True)
def stub_workspace_watcher(request):
    """Stop ``create_app`` starting a real filesystem observer.

    Patches the name bound in ``web_terminal.app`` rather than the class in
    ``file_watcher``, so tests importing ``WorkspaceWatcher`` from its own
    module keep the real implementation.
    """
    if request.node.get_closest_marker("real_workspace_watcher"):
        yield None
        return

    with patch(
        "osprey.interfaces.web_terminal.app.WorkspaceWatcher",
        StubWorkspaceWatcher,
    ) as stub:
        yield stub


#: Patch target for the panel-register route's own-address probe. Exported so a
#: test that exercises the deploy-host check can re-patch the same attribute
#: from inside the autouse stub below — the inner patch wins.
HOST_ADDRS_TARGET = "osprey.interfaces.web_terminal.routes.panels._host_interface_addresses"


@pytest.fixture(autouse=True)
def _stub_host_interface_addresses():
    """Keep the register route's deploy-host check off the real machine.

    ``_host_interface_addresses`` calls ``socket.getaddrinfo`` itself, so the
    canned ``getaddrinfo`` patches in the panel modules would otherwise feed it
    the same ``10.0.0.5`` the test URLs resolve to and every registration would
    be refused as a proxy back into the deployment. Tests that exercise the
    check patch the helper themselves — the inner patch wins.

    The module-level TTL cache inside the real helper is cleared as well. That
    cache is keyed on nothing but a timestamp, so one test that reaches the
    real probe (or stubs ``socket.getaddrinfo`` under it) would otherwise leave
    a result standing for the next minute of the run, and the test that
    inherits it would be reading a neighbour's machine picture rather than its
    own. Autouse and unconditional, for the same reason as the fixtures above:
    the tests that need it are the ones whose authors would not think to ask.
    """
    from osprey.interfaces.web_terminal.routes import panels

    panels.reset_host_addrs_cache()
    with patch(HOST_ADDRS_TARGET, return_value=frozenset()):
        yield
    panels.reset_host_addrs_cache()


@pytest.fixture(autouse=True)
def reset_rendered_config_memo():
    """Isolate every test from the process-wide parsed-render memo.

    ``routes.websocket`` parses the rendered ``config.yml`` once and memoizes
    it on ``(path, stat signature)``. Two tests that write a render at the same
    path within one mtime tick share a signature, so the second would be
    answered the first one's parse — its posture, its targets, its labels.
    """
    from osprey.interfaces.web_terminal.routes import websocket

    websocket.reset_rendered_config_memo()
    yield
    websocket.reset_rendered_config_memo()


@pytest.fixture(autouse=True)
def reset_store_notice():
    """Re-arm the once-per-process no-durable-store notice around every test.

    ``ownership`` warns once per process that a container render records no
    claim that outlives the container. Whichever container-mode
    ``resolve_ownership`` runs first on an xdist worker spends that warning,
    and a later test asserting the notice is said out loud then sees silence.
    """
    from osprey.interfaces.web_terminal import ownership

    ownership.reset_store_notice()
    yield
    ownership.reset_store_notice()


#: Deployment-identity variables ``create_app`` and its lifespan read.
_WEB_IDENTITY_ENV = (
    "OSPREY_WEB_THEME",
    "OSPREY_WEB_TOUR",
    "OSPREY_WEB_APP_NAME",
    "OSPREY_TERMINAL_USER",
    "OSPREY_TERMINAL_LANDING_URL",
)


@pytest.fixture(autouse=True)
def web_identity_env_unset(monkeypatch):
    """Unset the deployment-identity variables for every test.

    ``tests/conftest.py::restore_environ`` restores the environment after a
    test but does not clear these on the way in, so a developer shell that
    exports one — a theme, a terminal user, a landing URL — changes what the
    app renders, and every row asserting the unset case goes red on that
    machine only. A test that needs a value sets it itself; it runs after this.
    """
    for name in _WEB_IDENTITY_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def bar_items_app(tmp_path) -> Callable[..., Any]:
    """A factory booting the whole app with both agent-data stores on ``tmp_path``.

    The feedback and bar-items stores sit under ``resolve_shared_data_root()``,
    which an unpatched lifespan resolves to the per-worker shared root — so a
    bar-items test booted without this shares its document with every other
    test on the worker, and passes or fails by run order. Here the resolver
    always answers ``tmp_path / "agent_data"``, the watched tree is
    ``tmp_path / "_watch"``, the panel roster is fixed and no panel server is
    launched.

    Call it as a context manager; it yields the started ``TestClient``
    (``client.app`` is the app)::

        with bar_items_app(stored=b"{...}") as client:
            ...

    Keyword Args:
        enabled_panels: Enabled built-in panel ids; ``None`` means ``{"artifacts"}``.
        custom_panels: Config-declared panel dicts; ``None`` means none.
        env: Environment overrides applied across ``create_app`` and the lifespan.
        stored: Document bytes written to the bar-items store before boot.
        web: The ``web:`` section ``_load_web_ui_config`` answers; ``None``
            leaves the real reader in place.
        config_path: Set on ``app.state.config_path`` once the app has started.
        project_dir: Passed to ``create_app``.
    """
    from osprey.interfaces.web_terminal.app import create_app
    from osprey.interfaces.web_terminal.bar_items_store import LAYOUT_FILENAME

    @contextmanager
    def _boot(
        *,
        enabled_panels: set[str] | None = None,
        custom_panels: list[dict] | None = None,
        env: dict[str, str] | None = None,
        stored: bytes | None = None,
        web: dict | None = None,
        config_path: Path | None = None,
        project_dir: Path | None = None,
    ) -> Iterator[TestClient]:
        agent_data_root = tmp_path / "agent_data"
        agent_data_root.mkdir(exist_ok=True)
        watch_dir = tmp_path / "_watch"
        watch_dir.mkdir(exist_ok=True)
        if stored is not None:
            store = agent_data_root / "bar_items"
            store.mkdir(parents=True, exist_ok=True)
            (store / LAYOUT_FILENAME).write_bytes(stored)
        panels = {"artifacts"} if enabled_panels is None else set(enabled_panels)
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "osprey.interfaces.web_terminal.app._load_web_config",
                    return_value={"watch_dir": str(watch_dir)},
                )
            )
            stack.enter_context(
                patch(
                    "osprey.utils.workspace.resolve_shared_data_root",
                    return_value=agent_data_root,
                )
            )
            stack.enter_context(
                patch(
                    "osprey.interfaces.web_terminal.app._load_panel_config",
                    return_value=(panels, list(custom_panels or []), None),
                )
            )
            stack.enter_context(patch("osprey.interfaces.web_terminal.app._launch_panel_server"))
            if web is not None:
                stack.enter_context(
                    patch(
                        "osprey.interfaces.web_terminal.app._load_web_ui_config",
                        return_value=web,
                    )
                )
            stack.enter_context(patch.dict("os.environ", env or {}, clear=False))
            app = create_app(shell_command="echo", project_dir=project_dir)
            with TestClient(app) as client:
                if config_path is not None:
                    app.state.config_path = config_path
                yield client

    return _boot


def bare_route_app(*routers: APIRouter, **state: Any) -> FastAPI:
    """A bare ``FastAPI`` mounting *routers*, with the lifespan's shared state seeded.

    Route suites that skip ``create_app`` still reach code that reads
    ``app.state.broadcaster`` and ``app.state.agent_activity_ring`` — every
    agent-origin panel command records into the ring — so a bare mount without
    them tests a state the shipped app never has. Both are seeded here, a
    ``MagicMock`` broadcaster and an empty bounded ring; *state* sets further
    ``app.state`` attributes and overrides either default.
    """
    from osprey.interfaces.web_terminal.routes.agent_activity import ACTIVITY_RING_MAX

    app = FastAPI()
    for router in routers:
        app.include_router(router)
    app.state.broadcaster = MagicMock()
    app.state.agent_activity_ring = deque(maxlen=ACTIVITY_RING_MAX)
    for name, value in state.items():
        setattr(app.state, name, value)
    return app
