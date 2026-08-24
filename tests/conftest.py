"""
Pytest configuration and shared test utilities.

This module provides shared fixtures and utilities for all Osprey tests.
"""

import logging
import os
import time
from pathlib import Path

import pytest
from rich.logging import RichHandler

from osprey.utils.logger import QUIET_THIRD_PARTY_LOGGERS
from tests import ci_diagnostics

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
    """
    saved = dict(os.environ)

    os.environ.pop("OSPREY_CONFIG", None)
    os.environ.pop("CONFIG_FILE", None)

    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


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
def reset_web_credentials_between_tests():
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
    """
    from osprey.interfaces.web_auth import reset_web_credentials
    from osprey.mcp_server.http import reset_panel_token_latch

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


def _is_ollama_available() -> bool:
    """True if a local Ollama server responds at localhost:11434."""
    try:
        import requests

        return requests.get("http://localhost:11434/api/tags", timeout=2).status_code == 200
    except Exception:
        return False


_RESOURCE_CHECKS: dict[str, tuple[callable, str]] = {
    "requires_als_apg": (_has_als_apg_api_key, "ALS_APG_API_KEY not set"),
    "requires_anthropic": (_has_anthropic_api_key, "ANTHROPIC_API_KEY not set"),
    "requires_api": (
        _has_any_provider_api_key,
        "No LLM provider API key set (ALS_APG_API_KEY / CBORG_API_KEY / "
        "AMSC_I2_API_KEY / ANTHROPIC_API_KEY)",
    ),
    "requires_ollama": (_is_ollama_available, "Ollama not reachable at localhost:11434"),
}


def pytest_collection_modifyitems(config, items):
    """Auto-skip items whose `requires_<resource>` marker's resource is missing.

    Each predicate is evaluated at most once per pytest run via cache. A
    missing resource adds a real `pytest.mark.skip(reason=...)`; satisfied
    markers are no-ops.
    """
    cache: dict[str, bool] = {}
    for item in items:
        for marker_name, (predicate, reason) in _RESOURCE_CHECKS.items():
            if marker_name not in item.keywords:
                continue
            available = cache.get(marker_name)
            if available is None:
                available = predicate()
                cache[marker_name] = available
            if not available:
                item.add_marker(pytest.mark.skip(reason=reason))


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
