"""Parametrized cross-interface assertions for the shared app-setup block.

Every one of the six OSPREY interface ``create_app()`` factories now delegates
its CORS + middleware + static-mount wiring to
:func:`osprey.interfaces._app_setup.configure_interface_app`.  This module builds
each real app and asserts the shared invariants hold uniformly:

1. The three static mounts (``/static/fonts``, ``/design-system``, ``/static``)
   are all present.
2. Both :class:`NoCacheStaticMiddleware` and :class:`ExceptionLoggingMiddleware`
   are registered.
3. The CORS middleware does **not** enable ``allow_credentials``.

Each factory has a slightly different signature, so a small per-interface builder
mirrors exactly how that interface's own tests construct the app (see e.g.
``tests/interfaces/artifacts/*``, ``tests/interfaces/web_terminal/*``,
``tests/interfaces/channel_finder/conftest.py``, ``tests/interfaces/ariel/test_app.py``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.routing import Mount

from osprey.interfaces.common_middleware import (
    ExceptionLoggingMiddleware,
    NoCacheStaticMiddleware,
)

# ── Per-interface builders ────────────────────────────────────────────────
# Each mirrors how that interface's existing tests instantiate create_app().
# None of them enter the app lifespan (no TestClient), so no external service
# is contacted — we only inspect the assembled routes/middleware.


def _build_web_terminal(tmp_path: Path) -> FastAPI:
    from osprey.interfaces.web_terminal.app import create_app

    # Mirrors tests/interfaces/web_terminal/* (create_app(shell_command="echo")).
    return create_app(shell_command="echo", project_dir=str(tmp_path))


def _build_artifacts(tmp_path: Path) -> FastAPI:
    from osprey.interfaces.artifacts.app import create_app

    # Mirrors tests/interfaces/artifacts/* (create_app(workspace_root=tmp_path)).
    return create_app(workspace_root=tmp_path)


def _build_ariel(tmp_path: Path) -> FastAPI:
    from osprey.interfaces.ariel.app import create_app

    # Mirrors tests/interfaces/ariel/test_app.py::test_create_app_basic. The
    # lifespan is never entered here, so config loading / service creation do
    # not run; the factory returns the assembled app directly.
    return create_app()


def _build_channel_finder(tmp_path: Path) -> FastAPI:
    from osprey.interfaces.channel_finder.app import create_app

    # Mirrors tests/interfaces/channel_finder/conftest.py.
    return create_app(project_cwd=str(tmp_path))


def _build_lattice_dashboard(tmp_path: Path) -> FastAPI:
    from osprey.interfaces.lattice_dashboard.app import create_app

    # workspace_root is optional; supply tmp_path so the lattice state dir is
    # created under an isolated, writable location.
    return create_app(workspace_root=tmp_path)


INTERFACE_BUILDERS = [
    ("web_terminal", _build_web_terminal),
    ("artifacts", _build_artifacts),
    ("ariel", _build_ariel),
    ("channel_finder", _build_channel_finder),
    ("lattice_dashboard", _build_lattice_dashboard),
]


@pytest.fixture(params=INTERFACE_BUILDERS, ids=[name for name, _ in INTERFACE_BUILDERS])
def interface_app(request, tmp_path) -> FastAPI:
    """Build one interface app per parametrization case."""
    _name, builder = request.param
    return builder(tmp_path)


# ── Shared invariants ─────────────────────────────────────────────────────


def _mount_paths(app: FastAPI) -> set[str]:
    return {route.path for route in app.routes if isinstance(route, Mount)}


def test_all_three_static_mounts_present(interface_app: FastAPI) -> None:
    paths = _mount_paths(interface_app)
    for expected in ("/static/fonts", "/design-system", "/static"):
        assert expected in paths, f"missing mount {expected}; got {sorted(paths)}"


def test_both_middlewares_present(interface_app: FastAPI) -> None:
    classes = {mw.cls for mw in interface_app.user_middleware}
    assert NoCacheStaticMiddleware in classes
    assert ExceptionLoggingMiddleware in classes


def test_cors_allow_credentials_not_enabled(interface_app: FastAPI) -> None:
    cors = next(
        (mw for mw in interface_app.user_middleware if mw.cls is CORSMiddleware),
        None,
    )
    assert cors is not None, "CORSMiddleware not registered"
    # CORSMiddleware stores its constructor kwargs on the Middleware entry.
    # allow_credentials must be left at Starlette's default (False): either
    # absent, or present-but-falsy.
    assert not cors.kwargs.get("allow_credentials", False)
