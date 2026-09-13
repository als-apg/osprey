"""
Pytest configuration and shared test utilities.

This module provides shared fixtures and utilities for all Osprey tests.
"""

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from rich.logging import RichHandler

from osprey.utils.logger import QUIET_THIRD_PARTY_LOGGERS
from tests import _env_scope_guard, ci_diagnostics
from tests._env_scope_guard import restore_module_environment

#: Repo root — the fallback when a test leaves the process in a deleted cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent

#: Root-logging state as it was before collection — see ``restore_root_logging``.
#: Captured at conftest import, which is the only moment guaranteed to precede
#: every test *and* every higher-scoped fixture.
_PRISTINE_LOGGING = {
    "root_level": logging.getLogger().level,
    "root_handlers": list(logging.getLogger().handlers),
    "third_party": {name: logging.getLogger(name).level for name in QUIET_THIRD_PARTY_LOGGERS},
}

# ===================================================================
# Collection-time environment pollution guard
# ===================================================================
#
# Importing a test module can import third-party code that loads a .env at
# import time — litellm calls dotenv.load_dotenv() on import, and python-dotenv
# walks UP from its package directory, so on a developer machine it can find an
# ancestor checkout's .env and inject real credentials, PROJECT_ROOT, and TZ
# into os.environ before the first test runs. CI has no ancestor .env, so
# whatever those variables change locally is invisible there.
#
# Only that injection is undone. A blanket restore would also strip the
# coordination variables some test modules legitimately export at import time
# (tests/va/* set EPICS_CA_SERVER_PORT et al. so the soft-IOC subprocesses they
# spawn share their ports), so a key is removed only when it BOTH appeared
# during collection AND carries exactly the value an ancestor .env defines.

# import-time required because `osprey.cli.styles` builds its module-level
# Rich Console at import time and Rich reads FORCE_COLOR in `Console.__init__`,
# so a fixture runs too late to un-force a console that already exists: the
# color-forcing variables must be gone BEFORE collection imports any osprey
# module. A developer terminal that
# exports FORCE_COLOR otherwise turns CliRunner captures into ANSI-laced text
# and fails seven CLI tests on pristine main — CI is green only because its
# runners never export it. `NO_COLOR` is left alone: unset is the same default
# CI runs under.
os.environ.pop("FORCE_COLOR", None)
os.environ.pop("CLICOLOR_FORCE", None)

# Rich deliberately clamps ``TERM=dumb`` consoles to 80 columns and suppresses
# cursor control even when a test explicitly requests ``force_terminal=True``.
# CI presents a capable TERM, so normalize only the dumb/unknown host values;
# tests for a restricted terminal can still set one explicitly after collection.
if os.environ.get("TERM", "").lower() in {"dumb", "unknown"}:
    os.environ["TERM"] = "xterm-256color"

_PRE_COLLECTION_ENV = dict(os.environ)


def _ancestor_dotenv_values() -> dict[str, str]:
    """Every assignment an ancestor .env could have injected, keyed by var."""
    from dotenv import dotenv_values

    injected: dict[str, str] = {}
    for directory in (_REPO_ROOT, *_REPO_ROOT.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            injected.update({k: v for k, v in dotenv_values(candidate).items() if v is not None})
    return injected


def pytest_collection_finish(session):
    """Undo ancestor-.env injection performed by imports during collection."""
    import time

    added = set(os.environ) - set(_PRE_COLLECTION_ENV)
    if added:
        injected = _ancestor_dotenv_values()
        for key in added:
            if key in injected and os.environ[key] == injected[key]:
                del os.environ[key]
    # Re-sync the C library's cached zone with the (possibly restored) TZ.
    # Collection can leave the two disagreeing (TZ injected, cache not), and
    # then the first test to call tzset() flips tzname mid-suite and reads as
    # a leaker to _no_host_timezone_leak below.
    time.tzset()


# ===================================================================
# Provider gateway stand-in
# ===================================================================

#: The endpoint every test uses when a gateway-backed provider needs a URL.
#:
#: Providers that front a site's own gateway (``als-apg``) ship no default
#: endpoint, so a test that exercises one has to supply the URL the same way a
#: deployment does. One shared value means no test states a real deployment's
#: host, and a reader can tell "this is the fixture endpoint" at a glance.
GATEWAY_BASE_URL = "https://gateway.example.org/v1"

#: :data:`GATEWAY_BASE_URL` as Claude Code receives it — no trailing ``/v1``,
#: which the resolver strips because Claude Code appends ``/v1/messages``.
GATEWAY_ORIGIN = "https://gateway.example.org"


# ===================================================================
# Environment guard
# ===================================================================
#
# Ordering note: autouse fixtures of equal scope are set up in *alphabetical*
# order, not declaration order — `pytest --setup-plan <test>` prints the real
# sequence. So `_no_host_timezone_leak` is the outermost function-scoped fixture
# here (leading underscore), not this one, and anything that must sit outside
# that guard's snapshot window has to be session-scoped rather than merely
# declared earlier.


@pytest.fixture(autouse=True, scope="function")
def restore_environ():
    """Snapshot and restore the whole of ``os.environ`` around every test.

    Leak guarded: plenty of code writes ``os.environ`` directly rather than
    through ``monkeypatch`` — ``.env`` loading in the health CLI and the config
    loader uses override semantics, and the web terminal lifespan assigns
    ``OSPREY_CONFIG`` by hand. Without a full snapshot those writes survive into
    sibling tests, which under xdist makes results depend on how the files
    happened to be distributed.

    ``OSPREY_CONFIG`` and ``CONFIG_FILE`` are additionally cleared on the way in,
    so a developer who exports either in their shell still gets a pristine run.
    ``TZ`` is handled once per session instead -- see ``_clear_dotenv_timezone``.

    This snapshot cannot see writes made by a *module*-scoped fixture, which is
    set up before it — ``restore_environ_per_module`` below covers those.
    """
    saved = dict(os.environ)

    os.environ.pop("OSPREY_CONFIG", None)
    os.environ.pop("CONFIG_FILE", None)

    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


@pytest.fixture(autouse=True, scope="module")
def restore_environ_per_module():
    """Snapshot and restore ``os.environ`` around every test *module*.

    Module-scoped fixtures — a real ``osprey build`` over a seeded exemplar repo,
    which loads that repo's ``.env`` with override semantics — are set up before
    the function-scoped snapshot above is taken, so their writes land inside that
    snapshot's ``saved`` copy and are restored, not removed, by every later test.
    This restores the environment at module teardown so they cannot outlive the
    module that created them.

    A fixture that outlives the module keeps its writes: the plugin registered
    in :func:`pytest_configure` records them, and :func:`~tests._env_scope_guard.restore_module_environment` replays
    them on top of the snapshot. Without that, a session-scoped fixture first
    requested by this module's second test would be left alive with its
    environment rolled back under it.
    """
    with restore_module_environment():
        yield


@pytest.fixture(autouse=True, scope="function")
def session_posture_leak_guard(monkeypatch):
    """Run every test outside a web-terminal session's write posture.

        Three variables carry that posture: ``OSPREY_EXECUTION_MODE`` (a read-only
        run), and the two anchors ``OSPREY_POSTURE_SESSION`` and
        ``OSPREY_AGENT_DATA_ROOT``, which the web server always stamps as a pair.
        A developer running the suite from inside a narrowed session would
        otherwise hand every record-touching test a session id and an
        agent-data root no test asked for — and, worse, a test that
        redirects the record by patching ``resolve_shared_data_root`` would be
        silently inert, because ``posture_store.agent_data_root()`` reads the
        variable FIRST and only falls back to the resolver.

    All three are CLEARED, not pointed elsewhere. Stamping a throwaway root here
        would look tidier — clearing the variable does not stop a write, it aims it
        at ``<repo>/var/agent_data`` via ``resolve_shared_data_root()`` — but the
        stamp is preferred over the resolver by both readers, so a suite-wide stamp
        silently overrides every test that redirects the root by patching
        ``resolve_shared_data_root``, which is most of them: measured, it turns 90
        tests across ``tests/mcp_server`` red and the whole hooks tree with it.
        The leak is therefore closed where it is caused — a test whose code writes
        under that root stamps its own — and kept closed by
        :func:`no_agent_data_in_the_repo` below, which fails the session if the run
        created the directory.

        Named without a leading underscore on purpose. Autouse fixtures of equal
        scope are set up in alphabetical order, so this sorts AFTER
        ``restore_environ`` and is therefore torn down before it — a name like
        ``_no_session_posture`` would delete the variables before that snapshot was
        taken, and the restore would then drop them for the rest of the session.
    """
    for anchor in (
        "OSPREY_EXECUTION_MODE",
        "OSPREY_POSTURE_SESSION",
        "OSPREY_AGENT_DATA_ROOT",
    ):
        monkeypatch.delenv(anchor, raising=False)
    yield


_REAL_DEPLOYMENT_LANES = ("tests/e2e/", "tests/va/e2e/")

#: ``<repo>/var/agent_data`` — the directory :func:`no_agent_data_in_the_repo`
#: watches. Spelled once, at import, because the creator hook below needs it
#: before any fixture has run.
_AGENT_DATA_MARKER = _REPO_ROOT / "var" / "agent_data"

#: Whether that directory was already there when this conftest was imported.
#: A pre-existing one belongs to a real local deployment, so nothing about it
#: is the suite's business and the creator hook stays quiet for the session.
_AGENT_DATA_PRE_EXISTED = _AGENT_DATA_MARKER.exists()

#: The first test in THIS worker whose teardown found the directory there,
#: ``None`` while none has. Recorded by :func:`pytest_runtest_teardown` and
#: read by the guard.
_AGENT_DATA_FIRST_SEEN: str | None = None


def pytest_runtest_teardown(item):
    """Timestamp the appearance of ``<repo>/var/agent_data`` against a test.

    The guard below runs once per worker at session teardown and can only
    report *that* the directory appeared, which is the least useful half of the
    finding: the run is over, and the reader is left grepping a whole tree of
    app fixtures for whichever one resolved the agent-data root to the
    checkout. This narrows it to a window instead — the first test that ran
    with the directory already present.

    Deliberately reported as an upper bound rather than as blame. The writer is
    typically not the test named: the last one measured was a uvicorn daemon
    thread that ``ServerLauncher`` auto-launched out of an interface app's
    lifespan and that reached its ``mkdir`` some tests later, in a worker that
    had moved on. Under ``-n`` the naming worker may not even be the one that
    created it. What the name is good for is bounding the search — nothing in
    this worker before it can be the cause.

    One ``stat`` per test, and only until the first hit — cheap enough to leave
    armed for every lane rather than gated behind a flag nobody would set
    before the leak had already cost them an afternoon.
    """
    global _AGENT_DATA_FIRST_SEEN
    if _AGENT_DATA_PRE_EXISTED or _AGENT_DATA_FIRST_SEEN is not None:
        return
    if _AGENT_DATA_MARKER.exists():
        worker = os.environ.get("PYTEST_XDIST_WORKER", "master")
        _AGENT_DATA_FIRST_SEEN = f"{item.nodeid} (worker {worker})"


@pytest.fixture(autouse=True, scope="session")
def agent_data_never_the_checkout(request, tmp_path_factory):
    """Divert a resolution that lands on THIS checkout to a throwaway root.

    Every path to the agent-data root funnels through
    ``osprey_connectors.workspace.resolve_project_root`` — ``resolve_shared_data_root``
    and ``resolve_agent_data_root`` both anchor ``agent_data.base_dir`` on
    whatever it answers, and ``writer.audit_dir`` anchors the ledger on it too.
    It is also the only seam in that chain that is looked up by module global at
    call time, so binding it here reaches every caller regardless of how it
    imported its own resolver: the modules that did ``from ... import
    resolve_shared_data_root`` at import time still run the original function
    body, and that body reads this name.

    Session-scoped because the leak it closes is won by a RACE, not by a test.
    The directory is created inside a uvicorn daemon thread — the artifacts
    gallery that ``ServerLauncher`` auto-launches from an interface app's
    lifespan, whose ``store_watcher.start()`` mkdirs
    ``<root>/artifacts`` — and that thread outlives the test that started it by
    an unbounded margin. A function-scoped redirection is therefore not merely
    untidy but unsound: ``monkeypatch`` undoes it at teardown and the thread
    then resolves against the real checkout, which is exactly the intermittency
    that made this leak look like a different module every run. A module-scoped
    one loses the same race, one module later.

    A WRAPPER, not a constant. A blanket redirect would answer the same tmp
    directory for a test that configured its own ``project_root`` and asserts
    what a tool derived from it — so only the one case that constitutes the
    leak is diverted: resolution landing on the checkout the suite is running
    in. Everything else, including every test that patches a resolver of its
    own further up the stack, is untouched — this sits BELOW those patches
    rather than overriding them, which is what a suite-wide
    ``OSPREY_AGENT_DATA_ROOT`` stamp cannot say for itself (see
    :func:`session_posture_leak_guard`).

    Disarmed for the real-deployment lanes, on the same predicate as
    :func:`no_agent_data_in_the_repo`: they run agents and servers with this
    checkout as the project root on purpose, and diverting it would be
    rewriting the thing under test.
    """
    from osprey_connectors import workspace as connector_workspace

    if any(item.nodeid.startswith(_REAL_DEPLOYMENT_LANES) for item in request.session.items):
        yield
        return

    real_resolve_project_root = connector_workspace.resolve_project_root
    diverted = tmp_path_factory.mktemp("project-root-not-the-checkout")

    def _never_the_checkout(config=None):
        root = real_resolve_project_root(config)
        try:
            landed_here = Path(root).resolve() == _REPO_ROOT
        except OSError:  # pragma: no cover - a root that vanished mid-run
            return root
        return diverted if landed_here else root

    # Installed for the lifetime of the process, with no undo. The thread this
    # exists to catch is a uvicorn daemon that outlives the fixture that started
    # it, so a redirect lifted at teardown leaves it resolving against the real
    # checkout for as long as the interpreter is still up -- the same unsoundness
    # a function-scoped redirect has, one scope later. Nothing runs after a
    # session fixture's teardown that needs the original back.
    connector_workspace.resolve_project_root = _never_the_checkout
    yield


@pytest.fixture(autouse=True, scope="session")
def no_agent_data_in_the_repo(request):
    """Fail the session if the suite created ``<repo>/var/agent_data``.

    The regression this exists for is silent by construction: ``var/`` is
    gitignored, the marker files are unlinked on the way out, and what is left
    behind is an empty directory that ``git status`` never mentions. It was
    found twice by hand; this is what finds it the third time.

    This is the guard that actually holds the line, rather than a suite-wide
    stamp of ``OSPREY_AGENT_DATA_ROOT`` — see
    :func:`session_posture_leak_guard`. A stamp would silence the symptom for
    every test at once, at the cost of overriding the resolver patch most
    store-touching tests use to redirect that root.

    Only a directory the RUN created is a failure. One that was already there
    belongs to a real local deployment and is none of the suite's business —
    checking for creation rather than existence is what keeps this from firing
    on a developer who has actually run OSPREY in this checkout.

    The real-deployment lanes (``tests/e2e/``, ``tests/va/e2e/``) are exempt:
    they run agents and servers with this checkout as the project root, so the
    executor's run folders land under ``var/agent_data`` by design, not by a
    fixture's mistake. The guard is armed only in a session that collects none
    of them — the unit lane it was written for.
    """
    marker = _AGENT_DATA_MARKER
    real_deployment_lane = any(
        item.nodeid.startswith(_REAL_DEPLOYMENT_LANES) for item in request.session.items
    )
    existed = marker.exists()
    yield
    if real_deployment_lane:
        return
    if not existed and marker.exists():
        culprit = (
            "\nalready present by the end of "
            f"{_AGENT_DATA_FIRST_SEEN} — the writer is at or before that point, "
            "and may be a background thread an earlier test started"
            if _AGENT_DATA_FIRST_SEEN
            else ""
        )
        raise AssertionError(
            f"the test run created {marker} — something resolved the agent-data root "
            "to the repository. A test that writes the posture store or a control-target "
            "state file must stamp OSPREY_AGENT_DATA_ROOT at a tmp path (see "
            "session_posture_leak_guard) rather than leave it to resolve_shared_data_root()."
            f"{culprit}"
            "\nThe bound above names only what this xdist worker ran: a writer in "
            "another worker, in a subprocess, or in a module that bound its resolver "
            "at import time is not narrowed by it."
        )


# ===================================================================
# Auto-reset Registry and Config Between Tests
# ===================================================================


@pytest.fixture(autouse=True, scope="function")
def reset_state_between_tests():
    """Auto-reset registry and config before each test to ensure isolation.

    This prevents state leakage between tests by:
    - Resetting the registry
    - Clearing config caches (utils.config and utils.workspace)

    Config-related environment variables are handled by ``restore_environ``.
    """
    # Reset before test
    from osprey.registry import reset_registry
    from osprey.utils.workspace import reset_config_cache

    reset_registry()
    reset_config_cache()

    yield

    # Reset after test
    reset_registry()
    reset_config_cache()


# ===================================================================
# Web-auth test seam
# ===================================================================
#
# ``WebAuthMiddleware`` is installed outermost on every interface app by
# ``configure_interface_app``, so every request a ``starlette.testclient``
# (a.k.a. ``fastapi.testclient``) ``TestClient`` — or an ``httpx.AsyncClient``
# wrapping an ``ASGITransport`` — makes is 401'd unless it carries a live
# operator credential. Hundreds of existing tests drive interface routes through
# these clients and predate the gate; they assert the route's own behaviour, not
# the absence of authentication. Rather than touch each of them, this seam makes
# every such client present the operator credential by default.
#
# Mechanism — request-time operator-secret injection, *not* a construction-time
# cookie. Two facts about the existing suite force this:
#
#   1. Some interface tests pin their own ``app.state.web_credentials`` (a fresh
#      :class:`WebCredentials` with a known operator secret) *after* the client
#      is constructed — ``tests/interfaces/web_terminal/test_proxy.py`` does. A
#      credential minted at client-construction time lands in the holder that
#      swap discards, so it never authenticates. Reading the holder on each
#      request, instead, always sees the live one.
#   2. The proxy-boundary tests attach *decoy* inbound headers (a wrong
#      ``X-Osprey-Terminal-Secret``, a stale cookie, a bearer) to prove the
#      proxy strips them. The gate checks the operator-secret header *first* and
#      refuses a wrong one outright, so a session cookie added alongside a decoy
#      secret header would never be reached. Overriding that header with the
#      process's real secret is the only thing that authenticates such a request.
#
# So the seam force-sets ``X-Osprey-Terminal-Secret`` (``OPERATOR_SECRET_HEADER``)
# to the operator secret held by the client's target app, per request, reading
# ``app.state.web_credentials`` at send time. This is the header path the task
# explicitly allows as the alternative to the cookie: the cookie path is the one
# real browsers use, but it cannot serve these two shapes, and the Origin/CSRF
# logic it would exercise is already covered directly by the raw-ASGI tests in
# ``tests/interfaces/test_auth_middleware.py``. The operator-secret header is not
# subject to the Origin check, so mutating and websocket requests pass too.
#
# The seam acts **only** on a client aimed at a gated interface app — one whose
# ``app.state.web_credentials`` is a real :class:`WebCredentials` (the object the
# gate reads, and the tell that ``configure_interface_app`` ran). A client aimed
# at any other ASGI app, or at a real network host, is left untouched. Both
# client shapes are covered: ``TestClient`` over HTTP *and* its
# ``websocket_connect`` handshake, and ``httpx.AsyncClient`` over ``ASGITransport``.
#
# Install is session-scoped so it wraps client fixtures of every scope — a
# module-scoped client is built before any function-scoped fixture would run.
# Opt-out is a per-test flag (``@pytest.mark.no_auth_seam``) the injection hooks
# read at send time, for the dedicated tests that assert an *unauthenticated*
# refusal through such a client. The raw-ASGI auth tests never build one, so they
# need no marker.

#: Flipped off by :func:`_auth_seam_optout` for a ``no_auth_seam`` test. Read by
#: the injection hooks at send time, so it governs even a client built by a
#: higher-scoped fixture and shared across tests.
_AUTH_SEAM_ON = True


@pytest.fixture(autouse=True, scope="function")
def reset_web_credentials_between_tests(monkeypatch: pytest.MonkeyPatch):
    """Forget the process web credentials before and after every test.

    The env-driven population path (``web_auth._populate``) is decided once per
    process and is idempotent, so without this the first test in a worker would
    pin the operator secret for every test after it, and a test that mints a
    secret would leak it into its neighbours. Resetting both sides keeps the
    population testable in isolation. It does not clear ``app.state`` on an app
    built before the reset — such an app keeps the holder it cached. The root
    TestClient seam reads its secret from ``app.state``, so that app stays
    reachable through it; the browser and live-server seams in
    ``tests/interfaces/conftest.py`` read the PROCESS holder instead, so a
    module-level app must be re-pointed at it with
    ``use_process_web_credentials(app)`` before those seams can authenticate.

    Also clears ``SESSION_LIFETIME_ENV`` and ``SESSION_STORE_DIR_ENV`` before
    every test, for the same leak-prevention reason: ``tests/cli/test_discovery_
    rewire.py`` runs the real ``osprey web`` command in-process, and the
    launcher publishes both ``OSPREY_TERMINAL_SESSION_LIFETIME`` and
    ``OSPREY_TERMINAL_SESSION_STORE_DIR`` into ``os.environ`` for the child
    process to read. Left in place, every later test sharing that worker would
    populate a holder pointed at a real, on-disk store directory instead of the
    unconfigured one most tests assume. ``monkeypatch`` restores whichever
    value (if any) was present before the test once it tears down, rather than
    each of the two sides here reset unconditionally the way the credentials
    calls above do.
    """
    from osprey.interfaces.web_auth import (
        SESSION_LIFETIME_ENV,
        SESSION_STORE_DIR_ENV,
        reset_web_credentials,
    )
    from osprey.mcp_server.http import reset_panel_token_latch

    monkeypatch.delenv(SESSION_LIFETIME_ENV, raising=False)
    monkeypatch.delenv(SESSION_STORE_DIR_ENV, raising=False)

    # The MCP client latches the last panel token it saw (so an in-process
    # companion launch scrubbing the carrier does not strip its bearer); that
    # latch is process state for the same reason and is reset on both sides too.
    reset_web_credentials()
    reset_panel_token_latch()
    yield
    reset_web_credentials()
    reset_panel_token_latch()


def _gated_interface_app(client):
    """Return the client's target app if it is a gated interface app, else None.

    A ``TestClient`` records the app on ``self.app``; an httpx client wrapping an
    ``ASGITransport`` carries it on the transport. The tell that the app installed
    :class:`WebAuthMiddleware` is a real :class:`WebCredentials` on
    ``app.state`` — seeded by ``configure_interface_app`` and read by the gate.
    """
    from osprey.interfaces.web_auth import WebCredentials

    app = getattr(client, "app", None)
    if app is None:
        app = getattr(getattr(client, "_transport", None), "app", None)
    credentials = getattr(getattr(app, "state", None), "web_credentials", None)
    return app if isinstance(credentials, WebCredentials) else None


def _current_operator_secret(app):
    """The operator secret ``app``'s gate will accept right now, or None."""
    from osprey.interfaces.web_auth import WebCredentials

    credentials = getattr(getattr(app, "state", None), "web_credentials", None)
    if isinstance(credentials, WebCredentials):
        return credentials.operator_secret
    return None


def _install_client_auth(client) -> None:
    """Arm one httpx client to present the operator secret to its gated app.

    A no-op for a client not aimed at a gated interface app. Otherwise it appends
    a request event hook (async for an ``AsyncClient``, sync otherwise) that
    force-sets the operator-secret header per request, and — for a client that
    speaks websockets — wraps ``websocket_connect`` to carry the same header on
    the handshake, which the request hooks never see.
    """
    import httpx

    app = _gated_interface_app(client)
    if app is None:
        return

    from osprey.interfaces.common_middleware import OPERATOR_SECRET_HEADER

    def _apply(request) -> None:
        if not _AUTH_SEAM_ON:
            return
        secret = _current_operator_secret(app)
        if secret:
            request.headers[OPERATOR_SECRET_HEADER] = secret

    async def _apply_async(request) -> None:
        _apply(request)

    hook = _apply_async if isinstance(client, httpx.AsyncClient) else _apply
    client.event_hooks.setdefault("request", []).append(hook)

    ws_connect = getattr(client, "websocket_connect", None)
    if callable(ws_connect):

        def _seamed_ws(url, *args, **kwargs):
            if _AUTH_SEAM_ON:
                secret = _current_operator_secret(app)
                if secret:
                    merged = dict(kwargs.get("headers") or {})
                    merged.setdefault(OPERATOR_SECRET_HEADER, secret)
                    kwargs["headers"] = merged
            return ws_connect(url, *args, **kwargs)

        client.websocket_connect = _seamed_ws


@pytest.fixture(autouse=True, scope="session")
def _install_auth_seam():
    """Patch the client constructors once so every client fixture is covered.

    Session-scoped on purpose: a module- or class-scoped client fixture is built
    before any function-scoped fixture runs, so a function-scoped patch would
    miss it. The per-test opt-out lives in :data:`_AUTH_SEAM_ON`, which the
    injection hooks read at send time.
    """
    import httpx
    from starlette.testclient import TestClient

    test_client_init = TestClient.__init__
    async_client_init = httpx.AsyncClient.__init__

    def _seamed_test_client_init(self, *args, **kwargs):
        test_client_init(self, *args, **kwargs)
        _install_client_auth(self)

    def _seamed_async_client_init(self, *args, **kwargs):
        async_client_init(self, *args, **kwargs)
        _install_client_auth(self)

    TestClient.__init__ = _seamed_test_client_init
    httpx.AsyncClient.__init__ = _seamed_async_client_init
    try:
        yield
    finally:
        TestClient.__init__ = test_client_init
        httpx.AsyncClient.__init__ = async_client_init


@pytest.fixture(autouse=True, scope="function")
def _auth_seam_optout(request):
    """Disable the auth seam for a ``@pytest.mark.no_auth_seam`` test."""
    global _AUTH_SEAM_ON
    previous = _AUTH_SEAM_ON
    _AUTH_SEAM_ON = request.node.get_closest_marker("no_auth_seam") is None
    try:
        yield
    finally:
        _AUTH_SEAM_ON = previous


# ===================================================================
# Host-timezone leak guard
# ===================================================================


@pytest.fixture(autouse=True, scope="session")
def _clear_dotenv_timezone():
    """Drop a ``.env``-injected ``TZ`` once, before any test runs.

    litellm calls a bare ``load_dotenv()`` at import, and the bare form searches
    *parent* directories -- so a worktree with no ``.env`` of its own still
    inherits the checkout's. That publishes ``TZ`` into ``os.environ`` during
    collection without a matching ``time.tzset()``, leaving the variable and the
    C library's cached zone disagreeing. The first test to call ``tzset()`` then
    reconciles them mid-test, and ``_no_host_timezone_leak`` reports the change
    against whichever test happened to make the call. It only bites where the
    host zone differs from the ``.env`` one, which is why CI -- which has no
    ``.env`` at all -- never sees it.

    Session scope is load-bearing: the clear has to land outside every
    function-scoped fixture's window, including the guard's own. Autouse
    fixtures of equal scope are ordered alphabetically, so ``_no_host_timezone``
    sets up first and tears down last; doing this per test would fall inside
    that window and read as a leak.
    """
    if os.environ.pop("TZ", None) is not None:
        time.tzset()
    yield


@pytest.fixture(autouse=True, scope="function")
def _no_host_timezone_leak():
    """Fail the test that leaves the process on a different host timezone.

    Finalizers run LIFO: a bare ``request.addfinalizer(time.tzset)`` runs
    *before* monkeypatch restores ``TZ``, leaving every later test on the
    wrong host zone. This fixture's teardown runs after monkeypatch's own,
    which is exactly the ordering the bug turns on.
    """
    import os as _os
    import time as _time

    before = (_os.environ.get("TZ"), _time.tzname)

    yield

    after = (_os.environ.get("TZ"), _time.tzname)
    assert after == before, (
        f"test leaked the host timezone: TZ/tzname was {before} before the test "
        f"and {after} after. Undo the monkeypatch *before* calling time.tzset() "
        f"so the C library re-reads the restored TZ, not the patched one."
    )


# ===================================================================
# Working-directory guard
# ===================================================================


@pytest.fixture(autouse=True, scope="function")
def restore_cwd():
    """Restore the process working directory after every test.

    Leak guarded: raw ``os.chdir`` calls scattered through the suite, plus any
    exception path that skips their ``finally``, leave the process parked in a
    foreign (often deleted ``tmp_path``) directory. The compose generator
    resolves ``SERVICES_DIR`` relative to the cwd, so a leaked cwd silently
    changes what later tests render.

    This is a backstop, not a licence: individual ``os.chdir`` sites still own
    their own restore. Deliberately cheap — it snapshots the path string and
    nothing else.
    """
    try:
        original = os.getcwd()
    except OSError:
        # Already parked in a deleted directory when this test started.
        original = str(_REPO_ROOT)

    yield

    try:
        os.chdir(original)
    except OSError:
        # The test deleted the directory it started in.
        os.chdir(_REPO_ROOT)


# ===================================================================
# Root-logging guard
# ===================================================================


@pytest.fixture(autouse=True, scope="function")
def restore_root_logging():
    """Return root logging to its pre-collection state before and after every test.

    Leak guarded: ``configure_logging()`` (``src/osprey/utils/logger.py``) makes
    three process-global changes — it sets the root logger level, installs an
    Osprey ``RichHandler``, and raises six third-party loggers to WARNING. Any
    test that reaches an entry point triggers all three, and they would then
    decide what ``caplog`` sees in every test that ran after it on the same
    worker, including whether httpx/urllib3 records are captured at all.

    The baseline is ``_PRISTINE_LOGGING``, captured at conftest import, rather
    than a per-test snapshot. A per-test snapshot cannot see far enough back:
    pytest sets higher-scoped fixtures up *before* this function-scoped one, so
    a module-scoped fixture that configures logging is already reflected in the
    snapshot and would be preserved rather than undone, for every later test on
    the worker. ``tests/cli/test_dockerfile_template.py`` runs the CLI from a
    module-scoped fixture and does exactly that.

    Only ``RichHandler`` instances are ever removed. ``caplog`` installs its
    ``LogCaptureHandler`` around each test, and anything a test or fixture owns
    itself, are left alone.

    The CLI adds a fourth process-global change: it installs its altitude gate
    (``osprey.cli.altitude``) as a filter on that ``RichHandler``, which decides
    what a run renders. A handler a test survives with — one that was already
    pristine, or one a fixture owns — would carry that gate into every later
    test on the worker, so any gate is stripped as well.
    """
    root = logging.getLogger()
    # `in` on handlers is an identity check — they define no __eq__.
    pristine_handlers = _PRISTINE_LOGGING["root_handlers"]

    def _strip_altitude_gate(handler: logging.Handler) -> None:
        # A RichHandler carries no filters until the CLI gates it, so the
        # import — which pulls in osprey.cli — stays out of runs that never
        # touch the CLI.
        if not handler.filters:
            return
        from osprey.cli.altitude import lift_gate

        lift_gate(handler)

    def _reset() -> None:
        root.setLevel(_PRISTINE_LOGGING["root_level"])
        for handler in list(root.handlers):
            if isinstance(handler, RichHandler):
                _strip_altitude_gate(handler)
                if handler not in pristine_handlers:
                    root.removeHandler(handler)
        for name, level in _PRISTINE_LOGGING["third_party"].items():
            logging.getLogger(name).setLevel(level)

    _reset()

    yield

    _reset()


# ===================================================================
# Health offload accounting guard
# ===================================================================


@pytest.fixture(autouse=True, scope="function")
def reset_health_offload_state():
    """Zero the health runner's abandoned-thread accounting around every test.

    Leak guarded: ``osprey.health.offload`` counts abandoned worker threads
    cumulatively for the life of the process — correct for a real run, shared
    state for a test suite. ``tests/health/test_offload.py`` really abandons
    threads, and afterwards every in-process ``osprey health`` invocation in
    ``tests/cli`` takes the ``os._exit`` branch
    (``src/osprey/cli/health_cmd.py:323-326``) instead of ``sys.exit``. Under
    xdist that kills the worker and everything still queued on it.

    Reset happens only between tests, never inside one:
    ``tests/health/test_offload.py`` asserts the real counter.
    """
    from osprey.health.offload import reset_abandoned_state

    reset_abandoned_state()

    yield

    reset_abandoned_state()


# ===================================================================
# One-real-machine warning guard
# ===================================================================


@pytest.fixture(autouse=True, scope="function")
def reset_second_real_block_warning():
    """Forget which config shapes have already drawn the one-real-machine warning.

    Leak guarded: ``osprey_connectors.types`` says once per process that a
    config carries a second real connector block no control target reaches —
    once, because the resolver behind it answers every roster render and a line
    repeated at that rate is a line nobody reads. Held for the life of the
    process, that set makes any test asserting the warning depend on whether an
    earlier test on the same worker happened to resolve the same shape first,
    which under xdist depends on how the files were distributed.
    """
    from osprey_connectors import types as connector_types

    connector_types._SECOND_REAL_BLOCK_WARNED.clear()

    yield

    connector_types._SECOND_REAL_BLOCK_WARNED.clear()


# ===================================================================
# Companion server launch guard
# ===================================================================


@pytest.fixture(autouse=True, scope="function")
def _no_companion_server_launches(request, monkeypatch):
    """Keep ``ServerLauncher`` from starting a real uvicorn server in a test.

    Leak guarded: ``ServerLauncher._launch_in_thread`` runs uvicorn in a daemon
    thread nothing ever stops, and two runtime paths reach it unasked — the
    web-terminal lifespan launches every companion panel whose ``auto_launch``
    is on (the default), and ``ArtifactStore`` launches the gallery on every
    save. A test that boots either leaves a real, gated server listening on the
    configured port for the rest of the xdist worker's life. A later launcher
    that finds that port held probes it with the unauthenticated-then-
    credentialed ``GET /`` pair of ``_adopt_or_refuse``, and the leaked server
    files both refusals through ``osprey.audit.writer.record`` — resolved at
    call time, so they land in whichever test currently holds that attribute
    patched to its own ledger. That test then fails on a record it never made.

    Only the thread start is replaced. Everything ``ensure_running`` decides
    before it — the auto-launch gate, the bind check, the held-port grace
    window and its verdict — still runs, so the launcher tests that patch
    those per instance are unaffected. A test of the launch itself opts out
    with ``@pytest.mark.real_server_launch``.
    """
    if request.node.get_closest_marker("real_server_launch"):
        return

    from osprey.infrastructure.server_launcher import ServerLauncher

    def _no_launch(self, host: str, port: int) -> None:
        return None

    monkeypatch.setattr(ServerLauncher, "_launch_in_thread", _no_launch)


# ===================================================================
# Marker-driven resource skip-gating
# ===================================================================
#
# Tests use @pytest.mark.requires_<resource> to declare what they need.
# This hook turns those markers into real skips when the resource is
# unavailable. Markers are registered in pyproject.toml under
# [tool.pytest.ini_options].markers; predicates live here.
#
# To add a new resource: append to _RESOURCE_CHECKS. Predicate must be
# zero-arg, side-effect-free, and fast (gets called once per item per
# resource at collection time).


def _has_als_apg_api_key() -> bool:
    import os as _os

    return bool(_os.environ.get("ALS_APG_API_KEY"))


def _has_anthropic_api_key() -> bool:
    import os as _os

    return bool(_os.environ.get("ANTHROPIC_API_KEY"))


def _has_any_provider_api_key() -> bool:
    """True if any supported LLM provider key is set.

    Mirrors the inline detection in tests/e2e/test_llm_channel_namer.py:
    als-apg, cborg, amsc-i2, anthropic.
    """
    import os as _os

    return any(
        _os.environ.get(k)
        for k in ("ALS_APG_API_KEY", "CBORG_API_KEY", "AMSC_I2_API_KEY", "ANTHROPIC_API_KEY")
    )


def _has_openai_api_key() -> bool:
    import os as _os

    return bool(_os.environ.get("OPENAI_API_KEY"))


def _has_google_api_key() -> bool:
    import os as _os

    return bool(_os.environ.get("GOOGLE_API_KEY"))


def _has_cborg_api_key() -> bool:
    import os as _os

    return bool(_os.environ.get("CBORG_API_KEY"))


def _e2e_provider_availability() -> tuple[bool, str]:
    """Whether the credential of the provider this run builds with is present.

    Two facts meet here and neither is authored: which provider the run was
    told to build with (``tests.e2e.provider``) and which variable holds that
    provider's key (``PROVIDER_API_KEYS``, whose own comment calls it the single
    source of truth). Gating on one gateway's key instead is what let a run
    naming another provider, and holding its credential, skip anyway.

    A run that named no provider reads as unavailable carrying the refusal's own
    message rather than raising: the e2e collection hook owns ending such a
    session, and keeping this predicate out of that business leaves the two free
    of any hook ordering.
    """
    import os as _os

    from osprey.models.provider_registry import PROVIDER_API_KEYS
    from tests.e2e.provider import E2E_PROVIDER_ENV, e2e_provider

    try:
        provider = e2e_provider()
    except RuntimeError as exc:
        return False, str(exc)
    if provider not in PROVIDER_API_KEYS:
        known = ", ".join(sorted(PROVIDER_API_KEYS))
        return False, (
            f"{E2E_PROVIDER_ENV} names {provider!r}, which is not a provider OSPREY "
            f"registers (known: {known})"
        )
    key_var = PROVIDER_API_KEYS[provider]
    if key_var is None or _os.environ.get(key_var):
        return True, ""
    return False, f"{key_var} not set — the provider this run builds with is {provider!r}"


def _has_e2e_provider_key() -> bool:
    return _e2e_provider_availability()[0]


def _e2e_provider_reason() -> str:
    return _e2e_provider_availability()[1]


def _is_ollama_available() -> bool:
    """True if a local Ollama server responds at localhost:11434."""
    try:
        import requests

        return requests.get("http://localhost:11434/api/tags", timeout=2).status_code == 200
    except Exception:
        return False


# The second element is the skip reason: a fixed string, or a zero-arg callable
# for a resource whose absence has more than one explanation to report.
_RESOURCE_CHECKS: dict[str, tuple[Callable[[], bool], str | Callable[[], str]]] = {
    "requires_e2e_provider": (_has_e2e_provider_key, _e2e_provider_reason),
    "requires_als_apg": (_has_als_apg_api_key, "ALS_APG_API_KEY not set"),
    "requires_anthropic": (_has_anthropic_api_key, "ANTHROPIC_API_KEY not set"),
    "requires_api": (
        _has_any_provider_api_key,
        "No LLM provider API key set (ALS_APG_API_KEY / CBORG_API_KEY / "
        "AMSC_I2_API_KEY / ANTHROPIC_API_KEY)",
    ),
    "requires_ollama": (_is_ollama_available, "Ollama not reachable at localhost:11434"),
    "requires_openai": (_has_openai_api_key, "OPENAI_API_KEY not set"),
    "requires_google": (_has_google_api_key, "GOOGLE_API_KEY not set"),
    "requires_cborg": (_has_cborg_api_key, "CBORG_API_KEY not set"),
}


def pytest_collection_modifyitems(config, items):
    """Auto-skip items whose `requires_<resource>` marker's resource is missing.

    Each predicate is evaluated at most once per pytest run via cache. A
    missing resource adds a real `pytest.mark.skip(reason=...)`; satisfied
    markers are no-ops.
    """
    cache: dict[str, tuple[bool, str]] = {}
    for item in items:
        for marker_name, (predicate, reason) in _RESOURCE_CHECKS.items():
            if marker_name not in item.keywords:
                continue
            resolved = cache.get(marker_name)
            if resolved is None:
                available = predicate()
                text = "" if available else (reason() if callable(reason) else reason)
                resolved = (available, text)
                cache[marker_name] = resolved
            if not resolved[0]:
                item.add_marker(pytest.mark.skip(reason=resolved[1]))


# ===================================================================
# CI diagnostics: what survives a kill
# ===================================================================
#
# A test that FAILS reports itself. A lane that is KILLED — job timeout, runner
# OOM, a wedged worker held until the step cap — reports nothing at all: the
# signal lands while pytest's output is still in a buffer. These hooks keep a
# continuously flushed record on disk instead, so the post-mortem has something
# to read. See tests/ci_diagnostics.py for the file formats and for why
# faulthandler must target a file rather than a worker's stderr.
#
# Entirely gated on OSPREY_CI_DIAG_DIR, which only the CI lanes set: with the
# variable unset this installs nothing and costs nothing.

_CI_DIAGNOSTICS: ci_diagnostics.DiagnosticsRecorder | None = None


def pytest_configure(config):
    """Open the per-worker records and arm the stack dumper."""
    # Registered as a plugin rather than left as a hook in this file: a
    # session-scoped fixture's ``pytest_fixture_setup`` is dispatched on the
    # Session node, whose hook proxy carries only the conftests above it — this
    # one is not among them. See tests/_env_scope_guard.py.
    if not config.pluginmanager.is_registered(_env_scope_guard):
        config.pluginmanager.register(_env_scope_guard, "osprey-env-scope-guard")

    # Registered here rather than in pyproject.toml so the seam and its opt-out
    # marker live in one file; `--strict-markers` would otherwise reject it.
    config.addinivalue_line(
        "markers",
        "no_auth_seam: opt this test out of the TestClient auth seam — for a "
        "test that asserts an unauthenticated 401/403 through a TestClient.",
    )

    global _CI_DIAGNOSTICS
    _CI_DIAGNOSTICS = ci_diagnostics.recorder_from_env()
    if _CI_DIAGNOSTICS is not None:
        _CI_DIAGNOSTICS.start()


def pytest_runtest_logstart(nodeid, location):
    """Fires before setup — so a test that never returns is still on record."""
    if _CI_DIAGNOSTICS is not None:
        _CI_DIAGNOSTICS.record("start", nodeid=nodeid)


def pytest_runtest_logfinish(nodeid, location):
    """Fires after teardown. A `start` left unmatched is the wedged test."""
    if _CI_DIAGNOSTICS is not None:
        _CI_DIAGNOSTICS.record("finish", nodeid=nodeid)


def pytest_unconfigure(config):
    if _CI_DIAGNOSTICS is not None:
        _CI_DIAGNOSTICS.stop()


# ===================================================================
# xdist distribution: file-or-group scheduling
# ===================================================================
#
# The unit lane runs `-n 4 --dist loadgroup`. Stock LoadGroupScheduling gives
# every nodeid without an `@group` suffix its own scope, so unmarked tests get
# no file-level pinning at all and module-level state can straddle workers.
# FileOrGroupScheduling keeps upstream's behaviour for `xdist_group`-marked
# nodeids and gives everything else exact `--dist loadfile` semantics.
#
# `_split_scope` is an underscore method of xdist's documented scheduler
# extension point; tests/infrastructure/test_xdist_scheduler.py pins both the
# override's behaviour and the fact that upstream still calls it.

try:
    from xdist.scheduler import LoadGroupScheduling as _LoadGroupScheduling
except ImportError:  # pragma: no cover - xdist is a dev/CI-only dependency
    FileOrGroupScheduling = None
else:

    class FileOrGroupScheduling(_LoadGroupScheduling):
        """LoadGroupScheduling that falls back to file scope, not per-test scope."""

        def _split_scope(self, nodeid: str) -> str:
            """Group by `@group` suffix when present, else by test file path.

            The `@`-after-`]` check mirrors upstream's, so a parametrize value
            containing `@` is not mistaken for a group suffix.
            """
            if nodeid.rfind("@") > nodeid.rfind("]"):
                return super()._split_scope(nodeid)
            return nodeid.split("::", 1)[0]


def pytest_xdist_make_scheduler(config, log):
    """Use FileOrGroupScheduling for `--dist loadgroup`, stock xdist otherwise.

    Returning None hands the choice back to xdist, leaving the e2e lane's
    `--dist loadfile` and ad-hoc `--dist load`/`worksteal` runs untouched.
    """
    if FileOrGroupScheduling is None:
        return None
    if config.getoption("dist", None) != "loadgroup":
        return None
    return FileOrGroupScheduling(config, log)


# ===================================================================
# Prompt Testing Helpers
# ===================================================================


class PromptTestHelpers:
    """Helper methods for testing prompt structure and content."""

    @staticmethod
    def extract_section(prompt: str, section_header: str) -> str:
        """Extract a specific section from the prompt by its header.

        Args:
            prompt: The full prompt text
            section_header: The header marking the start of the section

        Returns:
            The extracted section content (without the header)
        """
        lines = prompt.split("\n")
        section_lines = []
        in_section = False

        for line in lines:
            if section_header in line:
                in_section = True
                continue
            if in_section:
                # Stop at next all-caps header with colon
                if (
                    line.strip()
                    and line.strip().replace(" ", "").replace("'", "").isupper()
                    and ":" in line
                ):
                    break
                section_lines.append(line)

        return "\n".join(section_lines).strip()

    @staticmethod
    def get_section_positions(prompt: str, *section_headers: str) -> dict[str, int]:
        """Get the positions of multiple section headers in the prompt.

        Args:
            prompt: The full prompt text
            *section_headers: Variable number of section headers to find

        Returns:
            Dictionary mapping section headers to their positions (-1 if not found)
        """
        positions = {}
        for header in section_headers:
            try:
                positions[header] = prompt.index(header)
            except ValueError:
                positions[header] = -1
        return positions


# ===================================================================
# Pytest Fixtures
# ===================================================================


@pytest.fixture
def prompt_helpers():
    """Fixture providing prompt testing helper methods."""
    return PromptTestHelpers


# ===================================================================
# Test Configuration Helpers
# ===================================================================


@pytest.fixture
def test_config(tmp_path):
    """Fixture providing a minimal test configuration file.

    Creates a temporary config.yml with sensible defaults for testing.
    Tests can override or extend this as needed.

    Returns:
        Path to the created config file

    Examples:
        Basic usage::

            def test_something(test_config):
                # Config file exists at test_config
                assert test_config.exists()
                # Set environment to use it
                os.environ['CONFIG_FILE'] = str(test_config)

        With custom modifications::

            def test_custom(test_config):
                # Read existing config
                import yaml
                with open(test_config) as f:
                    config = yaml.safe_load(f)
                # Modify
                config['execution']['execution_method'] = 'subprocess'
                # Write back
                with open(test_config, 'w') as f:
                    yaml.dump(config, f)
    """
    import yaml

    config_file = tmp_path / "config.yml"

    # Create a test registry using extend_framework_registry helper
    registry_file = tmp_path / "registry.py"
    registry_file.write_text(
        """
# Test registry for integration tests - extends framework with empty additions
from osprey.registry import RegistryConfigProvider, extend_framework_registry

class TestRegistryProvider(RegistryConfigProvider):
    '''Test registry provider that extends framework defaults.'''

    def get_registry_config(self):
        # Use extend_framework_registry helper - the recommended way
        # This extends the framework registry with no additions
        return extend_framework_registry()
"""
    )

    # Minimal working configuration for tests
    config = {
        "project_root": str(tmp_path),
        "registry_path": str(registry_file),
        "approval": {
            "enabled": True,
            "default_policy": "selective",
        },
        "control_system": {
            "type": "mock",  # Use mock for tests - default patterns include write_channel/read_channel
        },
        "execution": {
            "execution_method": "subprocess",
            "limits": {"max_retries": 3, "max_execution_time_seconds": 30},
        },
        "models": {
            "orchestrator": {"provider": "openai", "model_id": "gpt-4"},
            "python_code_generator": {"provider": "openai", "model_id": "gpt-4"},
        },
    }

    with open(config_file, "w") as f:
        yaml.dump(config, f)

    # Do NOT initialize registry here - let each test handle registry initialization
    # to avoid state pollution between tests
    return config_file


@pytest.fixture
def test_config_with_approval(tmp_path):
    """Fixture providing a test configuration WITH approval enabled.

    This is specifically for testing approval workflows.
    """
    import yaml

    config_file = tmp_path / "config.yml"

    # Create a test registry using extend_framework_registry helper
    registry_file = tmp_path / "registry.py"
    registry_file.write_text(
        """
# Test registry for integration tests - extends framework with empty additions
from osprey.registry import RegistryConfigProvider, extend_framework_registry

class TestRegistryProvider(RegistryConfigProvider):
    '''Test registry provider that extends framework defaults.'''

    def get_registry_config(self):
        # Use extend_framework_registry helper - the recommended way
        # This extends the framework registry with no additions
        return extend_framework_registry()
"""
    )

    # Configuration with approval ENABLED
    config = {
        "project_root": str(tmp_path),
        "registry_path": str(registry_file),
        "approval": {
            "enabled": True,
            "default_policy": "selective",
        },
        "control_system": {
            "type": "mock",  # Use mock for tests - default patterns include write_channel/read_channel
        },
        "execution": {
            "execution_method": "subprocess",
            "limits": {"max_retries": 3, "max_execution_time_seconds": 30},
        },
        "models": {
            "orchestrator": {"provider": "openai", "model_id": "gpt-4"},
            "python_code_generator": {"provider": "openai", "model_id": "gpt-4"},
        },
    }

    with open(config_file, "w") as f:
        yaml.dump(config, f)

    # Do NOT initialize registry here - let the test do it
    # to avoid state pollution between tests

    return config_file


@pytest.fixture(scope="session")
def graphdb_plugin_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A directory holding neosemantics + APOC, resolved once per session.

    Every lane that starts a real graph store mounts the same two jars, and the
    expensive half of resolving them is a release download — so this is shared
    across lanes even though the *stores* are not (two of them wipe the graph
    between corpora and therefore run their own module-scoped container).

    Skips, with the reason, on a host that could not start the container:
    see :func:`tests._graphdb_container.resolve_plugin_dir`.
    """
    from tests._graphdb_container import resolve_plugin_dir

    return resolve_plugin_dir(tmp_path_factory)


# ===================================================================
# Control context — the record and the per-server reports
# ===================================================================
#
# Re-exported from ``tests._control_context_fixtures`` so that every suite
# downstream of the control target writes the two files the same way, and a
# schema change fails all of them at once instead of one at a time. The
# writers are plain functions there for suites that would rather import them
# at module level; these are the factory fixtures that hand the same functions
# to a test that asks for them by name.


@pytest.fixture
def write_control_context():
    """Write a control-context record under an explicit agent-data root.

    ``write_control_context(root, target="va", generation=3,
    posture={"va": "sandbox"})`` — see
    :func:`tests._control_context_fixtures.write_control_context` for the
    owner and ``last_switch`` arguments. Defaults to a record this process
    owns, so a test gets a live owner without asking for one.
    """
    from tests._control_context_fixtures import write_control_context as _write

    return _write


@pytest.fixture
def write_server_report():
    """Write one controls server's report under an explicit agent-data root.

    ``write_server_report(root, 4321, session="s", applied_target="va",
    applied_generation=3)`` — see
    :func:`tests._control_context_fixtures.write_server_report`. Every field
    but the PID defaults to "this server has not answered yet".
    """
    from tests._control_context_fixtures import write_server_report as _write

    return _write


@pytest.fixture
def control_context_root(tmp_path, monkeypatch):
    """A scratch agent-data root, stamped, with the reader cache dropped.

    The three lines every control-context suite opens with: a root that exists,
    ``OSPREY_AGENT_DATA_ROOT`` pointing at it, and no record parsed from a
    previous test still held in the module cache. The ``control_target/``
    directory is created, so a test can write a raw file into it without
    reaching for ``mkdir`` first.
    """
    from osprey_connectors import control_context, posture_store

    root = tmp_path / "agent_data"
    (root / posture_store.STATE_DIR_NAME).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(posture_store.AGENT_DATA_ROOT_ENV_VAR, str(root))
    control_context.invalidate_cache()
    yield root
    control_context.invalidate_cache()


# ===================================================================
# Graph search index
# ===================================================================


@dataclass(slots=True)
class GraphIndexCache:
    """Every graph search index this session has built, keyed by corpus digest.

    Held for the whole session because the thing being cached is not a
    render's file but the *parse* behind it: two builds of the same corpus,
    in two tests, in two directories, derive byte-identical rows from
    byte-identical text.
    """

    directory: Path
    """Where the cached indexes live — one ``<digest>.duckdb`` per corpus."""

    entries: dict[str, Path] = field(default_factory=dict)
    """Digest of the corpus TEXT -> the index built from it, inside :attr:`directory`."""


@pytest.fixture(scope="session")
def graph_index_cache(tmp_path_factory: pytest.TempPathFactory) -> GraphIndexCache:
    """The session's built indexes, shared by every render the suite performs.

    Requested by the autouse stub below, and by the tests that assert on the
    caching itself: a test that needs a corpus to be built for real can drop
    its entry here first, and get a deterministic builder count whatever ran
    before it.
    """
    return GraphIndexCache(directory=tmp_path_factory.mktemp("graph-index-cache"))


@pytest.fixture(autouse=True, scope="function")
def _stub_graph_index_builds(request, monkeypatch, graph_index_cache: GraphIndexCache):
    """Build each corpus the suite renders once per session, not once per render.

    ``osprey build`` derives a graph-mode project's channel search index from
    the Turtle corpus the render staged, which is an rdflib parse of the whole
    file — about a second and a half for the corpus the control-assistant
    preset ships. The suite renders that preset over and over, in the build
    tests, the deployment tests and every exemplar render, and each of those
    renders parses the identical bytes again.

    So the parse is memoized across the whole session on the digest of the
    corpus TEXT, the same key the build's own per-build memo uses
    (:attr:`osprey.cli.build_cmd._SharedRenderInputs.graph_indexes`), and for
    the same reason: every render re-copies the profile's ``data/`` tree, so
    the staged corpus is a different file with a fresh mtime every time and a
    path or stat key would miss on every pass.

    What is stubbed is only :func:`osprey.cli.build_cmd._build_graph_index` —
    the step that turns a corpus into a file. ``_graph_index_target`` still
    resolves the corpus and the index path out of the rendered config, and the
    index that lands in the render is the real builder's output for that exact
    corpus, so a test can still read it, query it, or hand it to the roster.
    A corpus that cannot be read or parsed is handed straight to the real hook,
    which owns the warning and the decision that a failure costs the index
    rather than the build — one copy of that fact, not two.

    Opt out with ``@pytest.mark.real_graph_index`` when the derivation itself
    is what a test is about.
    """
    if request.node.get_closest_marker("real_graph_index"):
        return

    import shutil

    from osprey.cli import build_cmd

    real_hook = build_cmd._build_graph_index

    def _cached_build_graph_index(shared, target, progress):
        # Imported here, at call time, for two reasons: the fixture stays
        # import-light for the tests that never render anything, and a test
        # that counts builder calls by patching the package attribute still
        # sees this path go through it.
        from osprey.services.channel_finder.graph_index import build_graph_index
        from osprey.services.facility_knowledge.seeder.graph_seeder import ttl_sha256

        try:
            digest = ttl_sha256(target.corpus_path.read_text(encoding="utf-8"))
            cached = graph_index_cache.entries.get(digest)
            if cached is None:
                cached = graph_index_cache.directory / f"{digest}.duckdb"
                build_graph_index(target.corpus_path, cached)
                if not cached.is_file():
                    raise FileNotFoundError(cached)
                # Recorded only once the file exists, so a build that wrote
                # nothing cannot poison every later render of this corpus for
                # the rest of the session.
                graph_index_cache.entries[digest] = cached
            target.index_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached, target.index_path)
        except Exception:  # noqa: BLE001 - the real hook decides what a failure costs
            return real_hook(shared, target, progress)

        # The build's own memo, kept consistent: a later render pass of the
        # same build reads it before asking for an index at all.
        shared.graph_indexes.setdefault(digest, target.index_path)
        progress(
            "  ✓ Reused the session-cached channel search index built from %s",
            target.corpus_path.name,
        )
        return target.index_path

    #: Marks this as the stub, so a test can assert which hook it is running
    #: against without inspecting where the function was defined.
    _cached_build_graph_index.stubbed_by_conftest = True

    monkeypatch.setattr(build_cmd, "_build_graph_index", _cached_build_graph_index)
