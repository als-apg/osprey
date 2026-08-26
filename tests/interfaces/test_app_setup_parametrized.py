"""Parametrized cross-interface assertions for the shared app-setup block.

Every OSPREY app that delegates its middleware + static-mount wiring to
:func:`osprey.interfaces._app_setup.configure_interface_app` is built here —
all nine: ``web_terminal``, ``artifacts``, ``ariel``, ``channel_finder``,
``lattice_dashboard``, ``health``, ``okf_panel``, ``bluesky_web`` (which
assembles its app at module scope rather than via a ``create_app()`` factory)
and ``theme_lab`` (whose factory lives in ``osprey.cli.theme_lab_cmd``, not
under ``osprey/interfaces/``). This module builds each real app and asserts the
shared invariants hold uniformly:

1. The three static mounts (``/static/fonts``, ``/design-system``, ``/static``)
   are all present.
2. :class:`NoCacheStaticMiddleware`, :class:`ExceptionLoggingMiddleware` and
   :class:`HttpAuditMiddleware` (the audit layer for admitted mutations) are
   all registered.
3. :class:`WebAuthMiddleware` is registered (the auth gate).
4. The two safety layers sit at the two ends of the shared block:
   :class:`WebAuthMiddleware` outermost of everything,
   :class:`HttpAuditMiddleware` inside every other layer
   ``configure_interface_app`` installs. Pinned here, on the nine real apps,
   and not only on a synthetic one — "this lands in every interface app" is
   the audit layer's own spec, and an interface that grew its own middleware
   around the shared block would break it silently.

   ``bluesky_web`` is why the audit assertion is "inside the shared block"
   rather than ``user_middleware[-1]``: it registers its own ``@app.middleware
   ("http")`` no-cache wrapper at module scope, *before* calling
   ``configure_interface_app``, so that wrapper ends up innermost of all. It
   only fills in response headers — it decides nothing and rewrites no status
   — so the status the audit layer records is still the one the route
   produced.
5. ``CORSMiddleware`` is **absent** — nothing here is cross-origin.

Each factory has a slightly different signature, so a small per-interface builder
mirrors exactly how that interface's own tests construct the app (see e.g.
``tests/interfaces/artifacts/*``, ``tests/interfaces/web_terminal/*``,
``tests/interfaces/channel_finder/conftest.py``, ``tests/interfaces/ariel/test_app.py``).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.routing import Mount

from osprey.interfaces.common_middleware import (
    ExceptionLoggingMiddleware,
    HttpAuditMiddleware,
    NoCacheStaticMiddleware,
    WebAuthMiddleware,
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


def _build_health(tmp_path: Path) -> FastAPI:
    from osprey.interfaces.health.app import create_app

    # config_path=None is guaranteed-constructible (guarded fallback).
    return create_app()


def _build_okf_panel(tmp_path: Path) -> FastAPI:
    from osprey.interfaces.okf_panel.app import create_app

    # bundle_path=None → guarded app; the app is still fully assembled.
    return create_app()


def _build_theme_lab(tmp_path: Path) -> FastAPI:
    # The Theme Lab's factory lives in the CLI module, not under
    # osprey/interfaces/ — it serves the packaged design system and nothing
    # project-specific, so it needs no arguments.
    from osprey.cli.theme_lab_cmd import create_app

    return create_app()


def _build_bluesky_web(tmp_path: Path) -> FastAPI:
    # bluesky_web assembles its app at MODULE scope, not via a create_app()
    # factory, so importing the module runs configure_interface_app — and thus
    # get_web_credentials — at import time. Seed an operator secret first so
    # that import cannot hit the container-shape RuntimeError (a declared bind
    # host with no supplied secret); populated at most once per process.
    os.environ.setdefault("OSPREY_TERMINAL_SECRET", "test-bluesky-web-secret")
    from osprey.interfaces.bluesky_web.app import app

    return app


INTERFACE_BUILDERS = [
    ("web_terminal", _build_web_terminal),
    ("artifacts", _build_artifacts),
    ("ariel", _build_ariel),
    ("channel_finder", _build_channel_finder),
    ("lattice_dashboard", _build_lattice_dashboard),
    ("health", _build_health),
    ("okf_panel", _build_okf_panel),
    ("bluesky_web", _build_bluesky_web),
    ("theme_lab", _build_theme_lab),
]


#: Interfaces that ship no own ``static`` directory, so configure_interface_app's
#: ``.exists()``-guarded ``/static`` catch-all is (correctly) not mounted. The
#: two shared mounts (``/static/fonts``, ``/design-system``) are still present.
#: bluesky_web serves its panel bundle at ``/bluesky`` instead. (theme_lab is
#: not here: it passes the design-system directory as its own ``static_dir``,
#: so it does mount ``/static``.)
_INTERFACES_WITHOUT_OWN_STATIC = {"bluesky_web"}


@pytest.fixture(params=INTERFACE_BUILDERS, ids=[name for name, _ in INTERFACE_BUILDERS])
def interface_app(request, tmp_path) -> FastAPI:
    """Build one interface app per parametrization case."""
    name, builder = request.param
    app = builder(tmp_path)
    app.state.interface_name = name
    return app


# ── Shared invariants ─────────────────────────────────────────────────────


def _mount_paths(app: FastAPI) -> set[str]:
    return {route.path for route in app.routes if isinstance(route, Mount)}


def test_static_mounts_present(interface_app: FastAPI) -> None:
    paths = _mount_paths(interface_app)
    # The two shared-asset mounts are installed on every interface by
    # configure_interface_app, from the helper's own package location.
    for expected in ("/static/fonts", "/design-system"):
        assert expected in paths, f"missing mount {expected}; got {sorted(paths)}"
    # The /static catch-all is .exists()-guarded on the interface's own static
    # dir, so it is present for every interface that ships one and absent for
    # those that do not (see _INTERFACES_WITHOUT_OWN_STATIC).
    if interface_app.state.interface_name in _INTERFACES_WITHOUT_OWN_STATIC:
        assert "/static" not in paths, f"unexpected /static mount; got {sorted(paths)}"
    else:
        assert "/static" in paths, f"missing mount /static; got {sorted(paths)}"


def test_both_middlewares_present(interface_app: FastAPI) -> None:
    classes = {mw.cls for mw in interface_app.user_middleware}
    assert NoCacheStaticMiddleware in classes
    assert ExceptionLoggingMiddleware in classes
    assert HttpAuditMiddleware in classes, "HttpAuditMiddleware (the audit layer) not registered"


def test_web_auth_middleware_present(interface_app: FastAPI) -> None:
    classes = {mw.cls for mw in interface_app.user_middleware}
    assert WebAuthMiddleware in classes, "WebAuthMiddleware (the auth gate) not registered"


def test_web_auth_middleware_is_outermost(interface_app: FastAPI) -> None:
    # Added last => outermost => runs first: the gate authenticates before any
    # other middleware or route. Starlette lists user_middleware outermost-first.
    assert interface_app.user_middleware[0].cls is WebAuthMiddleware


def test_http_audit_middleware_is_innermost(interface_app: FastAPI) -> None:
    # Added first => innermost of the shared block => runs last: it never sees
    # a request the gate refused (so a refusal is recorded once), and the
    # status it records is the one the route produced rather than one an outer
    # layer rewrote. Asserted against the shared block rather than against
    # user_middleware[-1] because bluesky_web registers a header-only no-cache
    # wrapper of its own before the block (see this module's docstring).
    classes = [mw.cls for mw in interface_app.user_middleware]
    audit_at = classes.index(HttpAuditMiddleware)
    outside = [WebAuthMiddleware, ExceptionLoggingMiddleware, NoCacheStaticMiddleware]
    assert all(classes.index(cls) < audit_at for cls in outside), classes


def test_cors_middleware_absent(interface_app: FastAPI) -> None:
    classes = {mw.cls for mw in interface_app.user_middleware}
    assert CORSMiddleware not in classes, "CORSMiddleware should have been removed"


#: Files that mention ``configure_interface_app(`` without being a caller: the
#: helper's own definition site.
_NOT_A_CALLER = {"_app_setup.py"}


def _caller_name(path: Path) -> str:
    """The parametrization id for one file that calls ``configure_interface_app``.

    ``src/osprey/interfaces/<name>/app.py`` is named for its interface package;
    anything else — the Theme Lab's factory in ``osprey/cli/theme_lab_cmd.py`` —
    is named for its module, with the CLI module suffix dropped, so the id reads
    the same as an interface's.
    """
    if path.name == "app.py" and path.parent.parent.name == "interfaces":
        return path.parent.name
    return path.stem.removesuffix("_cmd")


def _configure_interface_app_callers() -> set[str]:
    """Every app in ``src/osprey`` that calls ``configure_interface_app``.

    Discovered by walking the **whole** source tree rather than
    ``interfaces/*/app.py``: the Theme Lab builds a gated app from
    ``osprey/cli/theme_lab_cmd.py``, and a glob scoped to the interfaces package
    could not see it — which is exactly the kind of blind spot this guard exists
    to prevent, since an app the guard cannot see is an app whose auth gate goes
    unasserted.
    """
    source_root = Path(__file__).resolve().parents[2] / "src" / "osprey"
    callers: set[str] = set()
    for module in source_root.rglob("*.py"):
        if module.name in _NOT_A_CALLER:
            continue
        text = module.read_text(encoding="utf-8")
        # Skip a mention that is only a comment or a docstring reference.
        if any(
            "configure_interface_app(" in line and not line.lstrip().startswith("#")
            for line in text.splitlines()
        ):
            callers.add(_caller_name(module))
    return callers


def test_builders_cover_every_configure_interface_app_caller() -> None:
    # The WebAuthMiddleware-present/outermost/CORS-absent guards above are only
    # as complete as INTERFACE_BUILDERS. If a new interface starts calling
    # configure_interface_app without being added here, its auth gate would go
    # unguarded — so assert the two sets agree exactly.
    built = {name for name, _ in INTERFACE_BUILDERS}
    callers = _configure_interface_app_callers()
    assert "theme_lab" in callers, (
        "discovery no longer sees the Theme Lab's create_app in "
        "src/osprey/cli/theme_lab_cmd.py — the guard has narrowed back to "
        "interfaces/*/app.py and would miss any gated app built elsewhere"
    )
    assert built == callers, (
        f"INTERFACE_BUILDERS out of sync with configure_interface_app callers.\n"
        f"  missing builders for: {sorted(callers - built)}\n"
        f"  builders with no caller: {sorted(built - callers)}"
    )
