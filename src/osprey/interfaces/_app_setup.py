"""Shared FastAPI app-setup helper for OSPREY interfaces.

Centralizes the CORS + middleware + static-mount block that every interface
factory repeats verbatim.  Each ``create_app()`` calls
:func:`configure_interface_app` last (after ``include_router``) so the mounts
and middleware wrap the fully-assembled app.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

from osprey.interfaces.common_middleware import ExceptionLoggingMiddleware, NoCacheStaticMiddleware


def configure_interface_app(app: FastAPI, *, static_dir: Path | str) -> None:
    """Apply the shared CORS, middleware, and static mounts to an interface app.

    Call this **last** in each ``create_app()`` factory (after
    ``app.include_router(...)``).  It registers, in order:

    1. :class:`~fastapi.middleware.cors.CORSMiddleware` with fully permissive
       origins/methods/headers and **no** ``allow_credentials`` (credentials are
       intentionally left at Starlette's default of ``False``).
    2. :class:`~osprey.interfaces.common_middleware.NoCacheStaticMiddleware`.
    3. :class:`~osprey.interfaces.common_middleware.ExceptionLoggingMiddleware`.

    It then mounts three static directories, each guarded by ``.exists()`` and
    registered in declaration order (Starlette matches routes in order, so the
    ``/static`` catch-all is registered last):

    * ``/static/fonts`` → ``<interfaces>/shared_fonts``
    * ``/design-system`` → ``<interfaces>/design_system/static``
    * ``/static`` → ``static_dir``

    The shared-fonts and design-system directories are derived from this
    helper's own location, so every interface serves the same shared assets;
    only the interface-specific ``static_dir`` is supplied by the caller.

    Args:
        app: The FastAPI application to configure.
        static_dir: The interface's own static directory, mounted at
            ``/static``.  Accepts a ``str`` or :class:`~pathlib.Path`.

    Returns:
        None. The app is mutated in place.
    """
    static_dir = Path(static_dir)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(NoCacheStaticMiddleware)
    app.add_middleware(ExceptionLoggingMiddleware)

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
