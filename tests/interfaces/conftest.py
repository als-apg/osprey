"""Shared fixtures for real-browser Playwright suites under ``tests/interfaces/``.

Extracted from the byte-identical boilerplate duplicated across
``design_system/test_behavioral.py``, ``design_system/test_visual.py``, and
``web_terminal/test_panels_browser.py``: the free-port/wait-for-port helpers,
the generic FastAPI-on-a-background-thread launcher, and the function-scoped
Playwright browser fixture. Suite-specific live-server wrappers (hub launchers
with their own patch sets) stay local to each file.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import pytest

# The port/uvicorn helpers now live in a supported module shared with the docs
# screenshot runner; re-export them under their historical underscore names so
# every ``tests/interfaces`` importer stays byte-for-byte unchanged.
from osprey.interfaces._serving import authorize_browser_context as _authorize_browser_context
from osprey.interfaces._serving import free_port as _free_port
from osprey.interfaces._serving import run_app_server as _run_app_server
from osprey.interfaces._serving import wait_for_port as _wait_for_port

if TYPE_CHECKING:
    from collections.abc import Iterator

    from playwright.sync_api import Browser

__all__ = [
    "_apply_all",
    "_authorize_browser_context",
    "_free_port",
    "_run_app_server",
    "_wait_for_port",
    "chromium_browser",
    "use_process_web_credentials",
]

# ---------------------------------------------------------------------------
# Playwright availability guard
# ---------------------------------------------------------------------------

try:
    from playwright.sync_api import sync_playwright

    _PLAYWRIGHT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLAYWRIGHT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Server-side ``requests`` auth seam
# ---------------------------------------------------------------------------
#
# The browser suites in this tree drive server-side events out of band from the
# browser — an agent MCP call, an SSE trigger — with the ``requests`` library,
# hitting the same gated interface app the page is looking at (POST
# /api/panel-visibility, /api/agent-activity, /__test__/… helpers). Those calls
# carry no browser cookie, so under WebAuthMiddleware they are refused with 401.
#
# The root-conftest auth seam (task 9.1) force-sets the operator-secret header on
# TestClient and httpx clients, but not on ``requests`` — which only these
# real-browser suites use. This mirrors that mechanism for ``requests``: it adds
# ``X-Osprey-Terminal-Secret`` (read from this process's credential holder, the
# same object the in-process server's gate verifies against) to every loopback
# request. Mutating POSTs and PATCHes pass too: the gate checks the ``Origin``
# header on every credential path, operator header included, but only when one is
# PRESENT — and neither ``requests`` nor ``TestClient`` sends an ``Origin`` on a
# programmatic call, so the check never applies to these.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _auth_seam_on() -> bool:
    """Whether the root conftest's auth seam is currently enabled.

    Read defensively so a ``@pytest.mark.no_auth_seam`` test — which flips the
    root ``_AUTH_SEAM_ON`` global to assert an *unauthenticated* refusal — is
    honored here too, and so this file does not hard-depend on that internal.
    """
    try:
        import tests.conftest as root_conftest

        return bool(getattr(root_conftest, "_AUTH_SEAM_ON", True))
    except Exception:  # pragma: no cover - the root conftest is always importable
        return True


def _process_operator_secret() -> str | None:
    """This process's operator secret, or ``None`` when it cannot be resolved.

    The in-process interface servers share this holder, so the secret it returns
    is exactly the one their gate accepts. Populating the holder here is harmless
    — the server reuses the same object — but a container-shape misconfiguration
    (a declared bind host with no secret) raises, which is mapped to ``None`` so
    the seam simply injects nothing rather than breaking request dispatch.
    """
    from osprey.interfaces.web_auth import get_web_credentials

    try:
        return get_web_credentials().operator_secret
    except RuntimeError:
        return None


def use_process_web_credentials(app: Any) -> None:
    """Re-point a **module-level singleton** app's gate at this test's credentials.

    ``configure_interface_app`` caches the process credential holder on
    ``app.state.web_credentials`` when the app is built, and
    :class:`WebAuthMiddleware` verifies against whatever is cached there. The
    root conftest's ``reset_web_credentials_between_tests`` then replaces the
    *process* holder before every test — deliberately, so population stays
    testable in isolation — but it cannot reach an app that was built before it
    ran.

    For an app created inside the test body (which is every other browser suite
    here) the cached holder and the process holder are the same object, so
    nothing has to be done. For an app imported at module scope
    (``bluesky_web.app.app``) they diverge after the first reset, and the gate
    ends up verifying against a holder nothing else in the process can see: the
    operator header the two seams above inject, and the session
    :func:`_authorize_browser_context` mints, both come from the *current*
    holder and are refused as "invalid credential".

    Call this immediately before serving such an app. It assigns rather than
    calling ``get_web_credentials(app)``, which would find the stale cached
    object and return it untouched.
    """
    from osprey.interfaces.web_auth import get_web_credentials

    app.state.web_credentials = get_web_credentials()


# Both seams below restore by hand in a ``finally`` rather than taking the
# ``monkeypatch`` fixture, and that is load-bearing rather than stylistic.
#
# An autouse fixture that requests ``monkeypatch`` pulls it into EVERY test's
# fixture closure, and autouse fixtures are set up first — so ``monkeypatch``
# ends up created before the test's own fixtures and therefore torn down AFTER
# them. That inversion breaks any fixture that wraps a ``mock.patch`` around a
# target the test body then re-points with ``monkeypatch.setattr``: the patch's
# ``__exit__`` puts the real object back first, and ``monkeypatch.undo()`` then
# restores the value it captured — the mock — leaving it installed for the rest
# of the worker. ``tests/interfaces/channel_finder/test_pending_review_api.py``
# is exactly that shape (its ``pending_review_client`` fixture patches
# ``osprey.utils.workspace.load_osprey_config``, and ``_artifact_store_dir``
# monkeypatches the same name), and the leaked ``MagicMock`` then answered every
# later config read in the worker — which is how an unrelated store suite ended
# up resolving its repo root one directory short.


@pytest.fixture(autouse=True)
def _requests_auth_seam() -> Iterator[None]:
    """Authenticate every loopback ``requests`` call the browser suites make.

    Wraps ``requests.sessions.Session.request`` (through which the module-level
    ``requests.post``/``get``/``patch`` helpers all funnel) to add the operator
    secret header for requests to a loopback host. Restored at the end of each
    test, so it never touches non-interface tests that run later in the session.
    """
    import requests

    from osprey.interfaces.common_middleware import OPERATOR_SECRET_HEADER

    original_request = requests.sessions.Session.request

    def _seamed_request(self, method, url, **kwargs):
        if _auth_seam_on() and urlsplit(str(url)).hostname in _LOOPBACK_HOSTS:
            secret = _process_operator_secret()
            if secret:
                headers = dict(kwargs.get("headers") or {})
                headers.setdefault(OPERATOR_SECRET_HEADER, secret)
                kwargs["headers"] = headers
        return original_request(self, method, url, **kwargs)

    requests.sessions.Session.request = _seamed_request
    try:
        yield
    finally:
        requests.sessions.Session.request = original_request


@pytest.fixture(autouse=True)
def _httpx_auth_seam() -> Iterator[None]:
    """Authenticate every loopback **sync** ``httpx`` call these suites make.

    Sibling of :func:`_requests_auth_seam`, for the other HTTP client this tree
    reaches a live server with: ``tests/interfaces/bluesky_web`` drives its
    :func:`bluesky_live_server` fixture with the module-level ``httpx.get``
    helpers, which under the hood build a throwaway :class:`httpx.Client`.

    Neither existing seam covers that client. The root-conftest seam
    (``tests/conftest.py``) patches ``httpx.AsyncClient.__init__`` and derives the
    secret from the ASGI app attached to the client's transport — a live server
    reached over a real socket has no such app, so that mechanism cannot apply
    here even in principle. The ``requests`` seam above covers a different
    library. Hence this one: same rule as ``requests`` (loopback host only, read
    the secret from *this process's* credential holder, which is the very holder
    the in-process server's gate verifies against), applied at
    ``httpx.Client.send`` — the single funnel every sync path reaches, including
    the module-level ``httpx.get``/``post`` helpers.

    ``httpx.AsyncClient`` does **not** inherit from ``httpx.Client``, so this
    never double-injects on top of the root seam. Starlette's ``TestClient``
    *does* subclass it, but its ``base_url`` host is ``testserver`` rather than a
    loopback address, so it is filtered out here and keeps using the root seam's
    app-derived secret.
    """
    import httpx

    from osprey.interfaces.common_middleware import OPERATOR_SECRET_HEADER

    original_send = httpx.Client.send

    def _seamed_send(self, request, **kwargs):
        if _auth_seam_on() and request.url.host in _LOOPBACK_HOSTS:
            secret = _process_operator_secret()
            if secret and OPERATOR_SECRET_HEADER not in request.headers:
                request.headers[OPERATOR_SECRET_HEADER] = secret
        return original_send(self, request, **kwargs)

    httpx.Client.send = _seamed_send
    try:
        yield
    finally:
        httpx.Client.send = original_send


# ---------------------------------------------------------------------------
# Helpers: mock-patch aggregation
# ---------------------------------------------------------------------------


@contextmanager
def _apply_all(patches: list) -> Iterator[None]:
    """Enter a variable-length list of ``unittest.mock`` patch objects together.

    Starts each patch in order and stops them in reverse on exit, so a suite can
    aggregate a patch set (some entries conditional) and apply it as a single
    context around ``create_app()`` + the server lifespan.

    Each patch's ``stop`` is registered before the next one starts, so a raising
    ``start()`` mid-list unwinds the ones already applied. Left unwound, those
    patches would stay live process-globally for the rest of the worker.
    """
    with ExitStack() as stack:
        for p in patches:
            p.start()
            stack.callback(p.stop)
        yield


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


def _install_auth_seam(browser: Browser) -> None:
    """Wrap ``browser.new_context``/``new_page`` so each is authorized on creation.

    Every interface app these suites launch (through :func:`_run_app_server`)
    installs :class:`~osprey.interfaces.common_middleware.WebAuthMiddleware`,
    which refuses an unauthenticated navigation with ``401`` — so a page opened
    from a bare context never renders and every DOM assertion times out. The apps
    run in *this* process, so :func:`_authorize_browser_context` can mint a
    session directly in the shared credential holder and install it as the
    session cookie; wrapping the two context/page factories here means individual
    tests keep calling ``browser.new_page()`` / ``browser.new_context()``
    unchanged and get an authorized context transparently.

    Both factories are wrapped because ``new_page`` builds its own throwaway
    context internally rather than routing through the public ``new_context``, so
    wrapping only one would leave the other path unauthenticated. The session is
    minted per call, at page/context creation time — which is inside the test
    body, after the live server (and thus this process's populated credential
    holder) exists. The Playwright sync objects carry no ``__slots__``, so the
    instance-level reassignment below is well defined.
    """
    original_new_context = browser.new_context
    original_new_page = browser.new_page

    def new_context(*args, **kwargs):
        context = original_new_context(*args, **kwargs)
        _authorize_browser_context(context)
        return context

    def new_page(*args, **kwargs):
        page = original_new_page(*args, **kwargs)
        _authorize_browser_context(page.context)
        return page

    browser.new_context = new_context
    browser.new_page = new_page


@pytest.fixture
def chromium_browser() -> Iterator[Browser]:
    """Function-scoped Playwright browser, pre-authorized against the auth gate.

    Skips if the chromium binary is absent. The yielded browser's
    ``new_context``/``new_page`` are wrapped by :func:`_install_auth_seam` so
    every page opened from it carries a valid operator session cookie for the
    in-process interface servers these suites drive.
    """
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

    _install_auth_seam(browser)
    try:
        yield browser
    finally:
        browser.close()
        pw.stop()
