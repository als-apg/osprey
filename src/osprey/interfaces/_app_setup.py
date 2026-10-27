"""Shared FastAPI app-setup helper for OSPREY interfaces.

Centralizes the middleware + static-mount block that every interface factory
repeats verbatim.  Each ``create_app()`` calls :func:`configure_interface_app`
last (after ``include_router``) so the mounts and middleware wrap the
fully-assembled app.

The block's job is now security-critical: it installs
:class:`~osprey.interfaces.common_middleware.WebAuthMiddleware` as the
outermost layer, so every interface's HTTP and websocket surface is gated
before any route or static asset is reached.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from osprey.interfaces._media_types import PinnedTypeStaticFiles
from osprey.interfaces.common_middleware import (
    ASSET_CORS_HEADERS,
    ExceptionLoggingMiddleware,
    HttpAuditMiddleware,
    NoCacheStaticMiddleware,
    WebAuthMiddleware,
)
from osprey.interfaces.web_auth import close_env_carriers, get_web_credentials


class AssetStaticFiles(PinnedTypeStaticFiles):
    """A static mount whose answers are C-ASSETS: readable from any origin.

    Built on :class:`~osprey.interfaces._media_types.PinnedTypeStaticFiles`,
    so every suffix that class maps is served with its pinned content type
    whatever the host's ``mimetypes`` table says.

    Adds :data:`~osprey.interfaces.common_middleware.ASSET_CORS_HEADERS`
    (``Access-Control-Allow-Origin: *``, never credentials) to every response
    the mount produces, so an agent-authored document running at an opaque
    origin can still import first-party modules, stylesheets and fonts. Keyed
    on the mount rather than on a global CORS layer: only what these package
    directories serve becomes cross-origin readable, and nothing agent-writable
    is ever served from one.
    """

    async def get_response(self, path: str, scope: Scope) -> Response:
        """Serve *path* with its pinned content type, plus the C-ASSETS header."""
        response = await super().get_response(path, scope)
        response.headers.update(ASSET_CORS_HEADERS)
        return response


#: Q1-C6's framing rule for console-origin HTML: only the console itself may
#: frame it. The hub frames its panels at the same origin, so they keep working.
FRAME_ANCESTORS_CSP = "frame-ancestors 'self'"

#: The legacy spelling of the same rule, for browsers that ignore CSP framing.
FRAME_OPTIONS = "SAMEORIGIN"


class FrameAncestorsMiddleware:
    """Pure-ASGI layer giving console-origin HTML Q1-C6's framing headers.

    Every ``text/html`` response that carries no ``Content-Security-Policy``
    of its own leaves with ``Content-Security-Policy: frame-ancestors 'self'``
    and, unless it already set one, ``X-Frame-Options: SAMEORIGIN``. That
    covers the console page, its static HTML and every first-party panel page
    the proxy relays onto the console origin, without a line per route.

    A response that already brings a CSP keeps exactly that one and gets no
    ``X-Frame-Options`` either: opaque agent content, ARIEL attachments and
    the Lab page each chose their own policy, and appending a second header
    would intersect it with this one.

    Pure ASGI rather than ``BaseHTTPMiddleware`` so streamed and SSE responses
    pass through unbuffered; websocket and lifespan scopes are not touched.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Wrap ``app``."""
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Add the framing headers to an HTML ``http.response.start`` lacking a CSP."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_frame_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                names = {name.lower() for name, _ in headers}
                content_type = next(
                    (value for name, value in headers if name.lower() == b"content-type"), b""
                )
                is_html = content_type.split(b";", 1)[0].strip().lower() == b"text/html"
                if is_html and b"content-security-policy" not in names:
                    headers.append((b"content-security-policy", FRAME_ANCESTORS_CSP.encode()))
                    if b"x-frame-options" not in names:
                        headers.append((b"x-frame-options", FRAME_OPTIONS.encode()))
                    message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_frame_headers)


def configure_interface_app(
    app: FastAPI, *, static_dir: Path | str, frame_ancestors: bool = False
) -> None:
    """Apply the shared middleware and static mounts to an interface app.

    Call this **last** in each ``create_app()`` factory (after
    ``app.include_router(...)``).

    It first seeds ``app.state.web_credentials`` by calling
    :func:`~osprey.interfaces.web_auth.get_web_credentials`, so this process's
    credentials are resolved at app construction.  This is deliberate: in the
    container shape a declared bind host with no supplied operator secret makes
    the holder raise, and seeding here surfaces that ``RuntimeError`` at startup
    rather than on the first request that reaches the gate.

    It then calls :func:`~osprey.interfaces.web_auth.close_env_carriers`, and
    that call is security-critical rather than tidy-up.  A launcher publishes
    the operator secret into ``os.environ`` on purpose, so that a ``--reload``
    worker or a ``--detach`` child inherits it; on the **direct-serve** path —
    the default ``osprey web``, and the per-user container's ``CMD`` — the
    launcher becomes the server in that same process, which is already
    populated, so no second ``_populate`` and therefore no second ``pop`` ever
    runs.  Without this call the re-published secret would stay in
    ``os.environ`` for the server's whole life, and claude-agent-sdk builds a
    spawned agent's environment as ``{**os.environ, **options.env}`` — meaning
    the agent would inherit the operator credential no matter what
    :mod:`osprey.agent_runner.clean_env` declined to copy.  App construction is
    the right seam because every launch shape passes through it exactly once,
    after the credentials are settled and before the process serves anything.

    It then registers the middleware.  Starlette wraps middleware
    inner-to-outer, so the layer added **last** is the **outermost** one and
    runs **first** on each request.  In add-order:

    1. :class:`~osprey.interfaces.common_middleware.HttpAuditMiddleware` — added
       first so it is **innermost**, sitting directly against the router.  That
       position is its contract: it records only what the gate admitted (so a
       refusal is never recorded twice), and the response status it files is
       the one the route actually produced.  This is what makes every
       state-changing request to *any* interface app show up in the audit
       ledger, including routes written long after this helper.
    2. :class:`~osprey.interfaces.common_middleware.NoCacheStaticMiddleware`.
    3. :class:`~osprey.interfaces.common_middleware.ExceptionLoggingMiddleware`.
    4. :class:`FrameAncestorsMiddleware` — only when ``frame_ancestors`` is
       true, which the web terminal (the console) passes and no other app
       does. Directly inside the gate, so every HTML answer to an admitted
       request, proxied panel pages included, passes through it.
    5. :class:`~osprey.interfaces.common_middleware.WebAuthMiddleware` — added
       last so it is outermost and authenticates every connection before
       anything downstream sees it.  It takes no constructor arguments: it
       resolves the session cookie name and external origin per request.

    There is no ``CORSMiddleware``. The one cross-origin read this system
    allows is C-ASSETS — first-party CSS/JS/fonts an opaque (``Origin: null``)
    agent document imports — and it is granted per mount by
    :class:`AssetStaticFiles` (``Access-Control-Allow-Origin: *``, never
    credentials), not by an app-wide layer that would reach ``/api``.

    It then mounts three static directories as :class:`AssetStaticFiles`, each
    guarded by ``.exists()`` and
    registered in declaration order (Starlette matches routes in order, so the
    ``/static`` catch-all is registered last):

    * ``/static/fonts`` → ``<interfaces>/shared_fonts``
    * ``/design-system`` → ``<interfaces>/design_system/static``
    * ``/static`` → ``static_dir``

    The shared-fonts and design-system directories are derived from this
    helper's own location, so every interface serves the same shared assets;
    only the interface-specific ``static_dir`` is supplied by the caller.  The
    three mount paths must stay in step with
    :data:`~osprey.interfaces.common_middleware.STATIC_MOUNT_PREFIXES`, which
    :class:`WebAuthMiddleware` treats as unauthenticated so a browser can fetch
    CSS/JS/fonts before it has a session.

    Args:
        app: The FastAPI application to configure.
        static_dir: The interface's own static directory, mounted at
            ``/static``.  Accepts a ``str`` or :class:`~pathlib.Path`.
        frame_ancestors: Install :class:`FrameAncestorsMiddleware`. The
            console opts in; the other interface apps are framed by it
            through the panel proxy and get the header there.

    Returns:
        None. The app is mutated in place.

    Raises:
        RuntimeError: from :func:`~osprey.interfaces.web_auth.get_web_credentials`
            in the container shape, when a bind host is declared but no operator
            secret was supplied.
    """
    static_dir = Path(static_dir)

    # Resolve this process's web credentials at construction, caching them on
    # app.state where WebAuthMiddleware reads them. Doing it here rather than
    # lazily per request surfaces the container-shape RuntimeError at startup.
    get_web_credentials(app)

    # ... then close the environment carriers again. The credentials are now in
    # this process's memory; a copy left in os.environ would be inherited by
    # every agent the SDK spawns, whose env is an OVERLAY on os.environ. Must
    # follow the resolve above: popping first would discard a
    # deployment-supplied secret.
    close_env_carriers()

    # Added first => innermost => runs last, closest to the router: it sees a
    # request only if the gate admitted it, and it sees the status the route
    # itself produced.
    app.add_middleware(HttpAuditMiddleware)
    app.add_middleware(NoCacheStaticMiddleware)
    app.add_middleware(ExceptionLoggingMiddleware)
    # Added last => outermost => runs first: authenticate before anything else.
    if frame_ancestors:
        # Directly inside the gate: it sees every HTML answer an admitted
        # request produces, while the gate keeps its outermost position.
        app.add_middleware(FrameAncestorsMiddleware)
    app.add_middleware(WebAuthMiddleware)

    base = Path(__file__).parent
    fonts_dir = base / "shared_fonts"
    design_system_dir = base / "design_system" / "static"

    # Mount shared fonts before /static (Starlette matches in declaration order).
    if fonts_dir.exists():
        app.mount("/static/fonts", AssetStaticFiles(directory=fonts_dir), name="shared-fonts")
    if design_system_dir.exists():
        app.mount(
            "/design-system", AssetStaticFiles(directory=design_system_dir), name="design-system"
        )
    if static_dir.exists():
        app.mount("/static", AssetStaticFiles(directory=static_dir), name="static")
