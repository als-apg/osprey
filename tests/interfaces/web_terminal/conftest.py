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

The **record spool** every test here writes into is redirected by
``tests/conftest.py::_isolate_spool_zone``, suite-wide rather than per
directory. Any test that drives ``create_app`` records real audit entries —
``HttpAuditMiddleware`` records one ``http_mutation`` entry per state-changing
request — and those would otherwise land in the developer's or the runner's own
``var/spool/<identity>/``, indistinguishable from records of things that really
happened. A test reads what it recorded through ``recorded_entries("audit")``.

The **feedback store** is sited under ``resolve_shared_data_root()``, which
reads ``agent_data.base_dir`` anchored on the project root and never consults
``OSPREY_AGENT_DATA_ROOT``. ``tests/interfaces/conftest.py::_agent_data_root_in_tmp``
stamps that variable for every test in this tree, and it moves the
control-context record only. A test that needs its own store patches
``osprey_connectors.workspace.resolve_shared_data_root``, the name the lifespan
imports at call time, as :func:`isolated_app` does. A lifespan left unpatched
resolves the store to the checkout's ``var/agent_data``;
``tests/conftest.py::agent_data_never_the_checkout`` diverts that to one
throwaway root per worker session, so the store is then isolated from the
checkout but shared with every other test in the worker.

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

``workspace_dir`` is the empty watched directory most app tests boot over;
``_started_app.terminal_client`` is the boot itself. ``ws_app`` and ``sessions_dir``
are the unstarted app and the transcript directory the ``/ws/terminal`` suites
drive through the harness in ``_ws``. ``agent_data_root`` and ``ledger`` are
the throwaway record root and the captured audit ledger of the posture and
target route suites (``_control_context`` holds their helpers).

Two non-autouse builders serve the tests that ask for them:
:func:`isolated_app` boots the whole app with its agent-data stores on
``tmp_path``, and :func:`bare_route_app` mounts routers on a bare ``FastAPI``
carrying the ``app.state`` the lifespan would have seeded.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager
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
def workspace_dir(tmp_path) -> Path:
    """An empty watched directory, ``tmp_path / "_agent_data"``."""
    ws = tmp_path / "_agent_data"
    ws.mkdir()
    return ws


@pytest.fixture
def ws_app(tmp_path) -> Iterator[FastAPI]:
    """The app the ``/ws/terminal`` suites drive, built over ``tmp_path`` and not started.

    Each test enters its own ``TestClient``; the spawn is swapped out through
    ``_ws.patch_spawn``, so the shell command is never run.
    """
    from osprey.interfaces.web_terminal.app import create_app

    with patch(
        "osprey.interfaces.web_terminal.app._load_web_config",
        return_value={"watch_dir": str(tmp_path / "ws")},
    ):
        yield create_app(shell_command="fake-not-used", project_dir=str(tmp_path))


@pytest.fixture
def sessions_dir(tmp_path, monkeypatch) -> Path:
    """Point ``SessionDiscovery`` at an empty transcript directory and return it."""
    from osprey.interfaces.web_terminal.session_discovery import SessionDiscovery

    sessions = tmp_path / "claude_sessions"
    sessions.mkdir()
    monkeypatch.setattr(SessionDiscovery, "_resolve_sessions_dir", lambda self: sessions)
    return sessions


@pytest.fixture
def agent_data_root(tmp_path, monkeypatch) -> Iterator[Path]:
    """Point every resolver at one throwaway agent-data root, verdict env clean.

    ``OSPREY_AGENT_DATA_ROOT`` is the single stamp the record reader and
    ``posture_store`` both prefer — the stamp this feature puts in every
    session child's environment. Patching ``resolve_shared_data_root`` instead
    would redirect only one of them: ``posture_store`` binds the resolver at
    import, so the other half would write into the repository's own
    ``var/agent_data``. Every variable that changes a write verdict behind a
    test's back — a stamped owner, a bound tree, a launch pin, a read-only run
    — is cleared, so each case states the one it means. Both caches are
    dropped on the way in and out.
    """
    from osprey_connectors import control_context, posture_store
    from osprey_connectors_tests._harness import VERDICT_ENV_VARS, state_dir_under

    root = tmp_path / "agent_data"
    state_dir_under(root).mkdir(parents=True)
    for name in VERDICT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OSPREY_AGENT_DATA_ROOT", str(root))
    posture_store.invalidate_cache()
    control_context.invalidate_cache()
    yield root
    posture_store.invalidate_cache()
    control_context.invalidate_cache()


@pytest.fixture
def ledger() -> Iterator[list[dict]]:
    """Capture every audit record a request would have written.

    Patches ``osprey.audit.writer.record``, which BOTH recorders resolve at
    call time — a route's ``record_and_mark`` and
    ``HttpAuditMiddleware._emit_audit_record`` — so the count is the true
    number of ledger lines one request produces, not just the route's own.
    """
    from osprey.audit import writer as audit_writer

    records: list[dict] = []

    def _record(**fields):
        records.append(fields)
        return Path("/dev/null/ledger.jsonl")

    with patch.object(audit_writer, "record", side_effect=_record):
        yield records


@pytest.fixture
def readable_process_table(monkeypatch):
    """Read the real process table through a wait for a readable one.

    A real table that is momentarily unreadable, or does not yet hold a
    just-spawned pid, makes the test wait, never decide: every
    ``process_table.snapshot`` call goes through
    :func:`tests.interfaces.web_terminal._pty_child.readable_rows`. A test that
    injects an unknown table patches ``snapshot`` after it.
    """
    from osprey.deployment import process_table
    from tests.interfaces.web_terminal._pty_child import readable_rows

    monkeypatch.setattr(process_table, "snapshot", readable_rows)


@pytest.fixture
def isolated_app(tmp_path) -> Callable[[], AbstractContextManager[TestClient]]:
    """A factory booting the whole app with its agent-data stores on ``tmp_path``.

    The agent-data stores sit under ``resolve_shared_data_root()``, which an
    unpatched lifespan resolves to the per-worker shared root, so an app booted
    without this shares its stores with every other test on the worker. Here
    the resolver always answers ``tmp_path / "agent_data"``, the watched tree is
    ``tmp_path / "_watch"``, the panel roster is ``{"artifacts"}`` and no panel
    server is launched.

    Call it as a context manager; it yields the started ``TestClient``
    (``client.app`` is the app)::

        with isolated_app() as client:
            ...
    """
    from osprey.interfaces.web_terminal.app import create_app

    @contextmanager
    def _boot() -> Iterator[TestClient]:
        agent_data_root = tmp_path / "agent_data"
        agent_data_root.mkdir(exist_ok=True)
        watch_dir = tmp_path / "_watch"
        watch_dir.mkdir(exist_ok=True)
        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "osprey.interfaces.web_terminal.app._load_web_config",
                    return_value={"watch_dir": str(watch_dir)},
                )
            )
            stack.enter_context(
                patch(
                    "osprey_connectors.workspace.resolve_shared_data_root",
                    return_value=agent_data_root,
                )
            )
            stack.enter_context(
                patch(
                    "osprey.interfaces.web_terminal.app._load_panel_config",
                    return_value=({"artifacts"}, [], None),
                )
            )
            stack.enter_context(patch("osprey.interfaces.web_terminal.app._launch_panel_server"))
            app = create_app(shell_command="echo")
            with TestClient(app) as client:
                yield client

    return _boot


def bare_route_app(*routers: APIRouter, **state: Any) -> FastAPI:
    """A bare ``FastAPI`` mounting *routers*, with the lifespan's shared state seeded.

    Route suites that skip ``create_app`` still reach code that reads
    ``app.state.broadcaster`` and ``app.state.agent_activity_buffer`` — every
    agent-origin panel command records into the buffer — so a bare mount without
    them tests a state the shipped app never has. Both are seeded here, a
    ``MagicMock`` broadcaster and an empty bounded buffer; *state* sets further
    ``app.state`` attributes and overrides either default.
    """
    from osprey.interfaces.web_terminal.routes.agent_activity import ACTIVITY_BUFFER_MAX

    app = FastAPI()
    for router in routers:
        app.include_router(router)
    app.state.broadcaster = MagicMock()
    app.state.agent_activity_buffer = deque(maxlen=ACTIVITY_BUFFER_MAX)
    for name, value in state.items():
        setattr(app.state, name, value)
    return app
