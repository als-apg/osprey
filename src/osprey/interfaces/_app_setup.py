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
from starlette.staticfiles import StaticFiles

from osprey.interfaces.common_middleware import (
    ExceptionLoggingMiddleware,
    NoCacheStaticMiddleware,
    WebAuthMiddleware,
)
from osprey.interfaces.web_auth import close_env_carriers, get_web_credentials


def configure_interface_app(app: FastAPI, *, static_dir: Path | str) -> None:
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

    1. :class:`~osprey.interfaces.common_middleware.NoCacheStaticMiddleware`.
    2. :class:`~osprey.interfaces.common_middleware.ExceptionLoggingMiddleware`.
    3. :class:`~osprey.interfaces.common_middleware.WebAuthMiddleware` — added
       last so it is outermost and authenticates every connection before
       anything downstream sees it.  It takes no constructor arguments: it
       resolves the session cookie name and external origin per request.

    There is no ``CORSMiddleware``: nothing in this system is cross-origin, so
    the previously-permissive CORS layer served no purpose and has been removed.

    It then mounts three static directories, each guarded by ``.exists()`` and
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

    app.add_middleware(NoCacheStaticMiddleware)
    app.add_middleware(ExceptionLoggingMiddleware)
    # Added last => outermost => runs first: authenticate before anything else.
    app.add_middleware(WebAuthMiddleware)

    base = Path(__file__).parent
    fonts_dir = base / "shared_fonts"
    design_system_dir = base / "design_system" / "static"

    # Mount shared fonts before /static (Starlette matches in declaration order).
    if fonts_dir.exists():
        app.mount("/static/fonts", StaticFiles(directory=fonts_dir), name="shared-fonts")
    if design_system_dir.exists():
        app.mount("/design-system", StaticFiles(directory=design_system_dir), name="design-system")
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
