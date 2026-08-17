"""OSPREY OKF Knowledge Panel — FastAPI application factory.

A read-only browser over a facility-knowledge bundle: concept tree, markdown
reader, search, and a bundle-health summary. Search is ranked through the qmd
sidecar where the deployment has one and falls back to a substring scan where
it does not — ``score`` on each hit says which answered. Backed directly by core
:class:`osprey.services.facility_knowledge.okf.bundle.OKFBundle` (no vendored
copy). Launched in-process by ``ServerLauncher`` and reverse-proxied at
``/panel/okf/`` — the ``channel_finder`` builtin pattern.

The bundle is parsed once at ``create_app`` time and cached on
``app.state.bundle`` so every request reuses the single parse.

Guarded construction (DA CC-1): a falsy ``bundle_path`` — or a path that fails
to open — never raises out of the factory (that exception would be swallowed by
the daemon-thread launcher, leaving a silent dead tab). Instead the app is built
with ``app.state.bundle = None``: ``/health`` still returns 200 and every data
endpoint returns a clear JSON "not configured" error.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

from osprey.interfaces._app_setup import configure_interface_app
from osprey.interfaces.okf_panel.helpers import (
    build_structure_markdown,
    format_qmd_snippet,
    group_concepts,
    make_snippet,
)
from osprey.interfaces.okf_panel.validation import (
    bundle_health,
    log_validation_summary,
    validate_bundle,
)

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

# Returned by every data endpoint when the panel has no bundle (unconfigured or
# unopenable path). HTTP 503 = the service is up but cannot serve data yet.
_NOT_CONFIGURED = {"error": "facility_knowledge.bundle_path not configured"}
_NOT_CONFIGURED_STATUS = 503


def _ranked_backend_kwargs():
    """Resolve the ranked-search backend for :class:`OKFBundle`, or nothing.

    The panel's factory is handed only ``bundle_path``, so it reads the project
    config itself for the two inputs ranked search needs: the sidecar endpoint
    (``services.qmd``) and the query knobs (``facility_knowledge.search``).
    Both are optional — a deployment with no sidecar is a supported
    configuration, and ``QMDClient(None)`` never opens a socket — so the
    unconfigured case costs nothing and says nothing.

    Returns:
        Keyword arguments for the :class:`OKFBundle` constructor. Empty when
        the config cannot be read or names a malformed value: search then
        degrades to the substring backend rather than taking the reading pane
        down with it, which a hard failure here would do.
    """
    from osprey.deployment.qmd_service import resolve_qmd_service_config
    from osprey.services.facility_knowledge.okf.bundle import OKFSearchSettings
    from osprey.services.qmd import QMDClient
    from osprey.utils.workspace import load_osprey_config

    # Everything the ranked backend needs sits inside the guard, config load
    # included: this helper's caller wraps it in the factory's own catch-all, so
    # anything escaping here does not cost the operator ranked search — it costs
    # them the whole panel. `Exception` is the right width for the same reason
    # `_load_bundle` uses it: nothing this helper can fail at is worth more than
    # substring search.
    try:
        config = load_osprey_config()
        return {
            "qmd_client": QMDClient(resolve_qmd_service_config(config)),
            "search_settings": OKFSearchSettings.from_config(config),
        }
    except Exception:  # noqa: BLE001 — see above; degrade, never kill the panel.
        logger.warning(
            "okf panel: qmd search is misconfigured; serving substring search only.",
            exc_info=True,
        )
        return {}


def _load_bundle(bundle_path):
    """Open the OKF bundle, or return ``None`` (never raise) if it cannot be opened.

    Args:
        bundle_path: Resolved ``facility_knowledge.bundle_path`` (or a falsy
            value when the section is absent/unconfigured).

    Returns:
        A validated :class:`OKFBundle`, or ``None`` when *bundle_path* is falsy
        or the bundle directory cannot be opened. Either ``None`` case logs one
        WARN and leaves the panel in guarded mode.
    """
    if not bundle_path:
        logger.warning(
            "okf panel: facility_knowledge.bundle_path not configured; "
            "serving guarded app (data endpoints return 'not configured')."
        )
        return None

    # Imported lazily so the guarded path never touches the OKF engine.
    from osprey.services.facility_knowledge.bundle_path import resolve_bundle_path
    from osprey.services.facility_knowledge.okf.bundle import OKFBundle

    try:
        bundle = OKFBundle(resolve_bundle_path(bundle_path), **_ranked_backend_kwargs())
    except Exception:  # noqa: BLE001 — a bad path must degrade, not kill the thread.
        logger.warning(
            "okf panel: could not open bundle at %s; serving guarded app.",
            bundle_path,
            exc_info=True,
        )
        return None

    log_validation_summary(validate_bundle(bundle), logger)
    return bundle


def create_app(bundle_path=None) -> FastAPI:
    """Create the OKF Knowledge Panel FastAPI application.

    Args:
        bundle_path: Filesystem path to the OKF bundle directory, resolved from
            ``facility_knowledge.bundle_path`` by the registry's
            ``factory_config_kwargs``. Falsy or unopenable → guarded app.

    Returns:
        Configured FastAPI application with the bundle cached on
        ``app.state.bundle`` (or ``None`` in guarded mode).
    """
    app = FastAPI(
        title="OSPREY OKF Knowledge Panel",
        description="Read-only browser over a facility-knowledge (OKF) bundle",
        version="1.0.0",
    )

    app.state.bundle = _load_bundle(bundle_path)

    def _bundle_or_error():
        """Return ``(bundle, None)`` when configured, else ``(None, JSONResponse)``."""
        bundle = app.state.bundle
        if bundle is None:
            return None, JSONResponse(_NOT_CONFIGURED, status_code=_NOT_CONFIGURED_STATUS)
        return bundle, None

    @app.get("/health")
    async def health():
        """Liveness probe — 200 even in guarded mode (panel process is up)."""
        return {
            "status": "ok",
            "service": "okf-panel",
            "configured": app.state.bundle is not None,
        }

    @app.get("/")
    async def root():
        """Serve the SPA shell, or a placeholder until the SPA task lands."""
        index_html = STATIC_DIR / "index.html"
        if index_html.exists():
            return FileResponse(index_html)
        return JSONResponse(
            {
                "service": "okf-panel",
                "detail": "SPA not built yet; JSON API at /api/concepts, "
                "/api/concept?id=..., /api/search?q=..., /api/bundle_health",
            }
        )

    @app.get("/api/concepts")
    async def api_concepts():
        """Grouped listing of every concept in the bundle ({"groups": [...]})."""
        bundle, err = _bundle_or_error()
        if err:
            return err
        return group_concepts(bundle.list_concepts())

    @app.get("/api/structure")
    async def api_structure():
        """Markdown overview of the whole knowledge base ({"markdown": ...})."""
        bundle, err = _bundle_or_error()
        if err:
            return err
        markdown = build_structure_markdown(group_concepts(bundle.list_concepts()))
        return {"markdown": markdown}

    @app.get("/api/concept")
    async def api_concept(id: str = ""):
        """Return one concept's frontmatter + body, looked up by ``?id=``.

        ``id`` is a query param (not a path segment) because concept IDs contain
        slashes (e.g. ``control-system/channel-finding``). Missing files,
        malformed frontmatter, and path-traversal escapes all surface as a 404.
        """
        bundle, err = _bundle_or_error()
        if err:
            return err
        from osprey.services.facility_knowledge.okf.bundle import OKFBundleError
        from osprey.services.facility_knowledge.okf.document import OKFDocumentError

        try:
            doc = bundle.read_concept(id)
        except (OKFBundleError, OKFDocumentError):
            return JSONResponse({"error": "not found", "id": id}, status_code=404)
        return {"id": id, "frontmatter": doc.frontmatter, "body": doc.body}

    # Deliberately `def`, not `async def` — the only handler here that is.
    # ``bundle.search`` is synchronous, and on the ranked path it makes blocking
    # HTTP round-trips to the sidecar (30 s timeout, and seconds at a time with
    # ``rerank: true`` even when the sidecar is healthy). Awaited on the event
    # loop that work would stall every other request in the process, including
    # the reading pane. A sync route runs in Starlette's threadpool instead, so
    # a slow or hung sidecar costs the search and nothing else. The other four
    # handlers are in-memory and stay `async`.
    @app.get("/api/search")
    def api_search(q: str = ""):
        """Search the bundle; returns snippet-only hits (never full bodies).

        Every hit carries a ``score``: a number when the qmd sidecar ranked it
        — the results are then in descending relevance order — and ``null``
        when the substring fallback produced it, which does not rank. The
        number is an **ordering signal, not a calibrated relevance
        probability** (see :class:`OKFSearchResult`), so the front end shows
        the rank it implies and never renders it as a percentage.

        Snippets come from qmd's match-centered excerpt when there is one, and
        otherwise from the local substring snippet, falling back last to the
        concept's own description. A ranked hit can have no local snippet at
        all — a semantic match need not contain the query text.
        """
        bundle, err = _bundle_or_error()
        if err:
            return err
        if not q.strip():
            return {"query": q, "results": []}

        results = []
        for hit in bundle.search(q):
            doc = hit.document
            snippet = format_qmd_snippet(hit.snippet) or make_snippet(doc.body, q)
            if not snippet:
                snippet = str(doc.frontmatter.get("description", ""))
            title = str(doc.frontmatter.get("title", hit.concept_id))
            results.append(
                {
                    "id": hit.concept_id,
                    "title": title,
                    "snippet": snippet,
                    "score": hit.score,
                }
            )

        return {"query": q, "results": results}

    @app.get("/api/bundle_health")
    async def api_bundle_health():
        """Report the panel's own validation summary for the served bundle."""
        bundle, err = _bundle_or_error()
        if err:
            return err
        return bundle_health(validate_bundle(bundle))

    # Shared CORS + middleware + static mounts (/design-system, /static/fonts,
    # /static) applied last so they wrap the fully-assembled app and never
    # shadow the API routes above. Carries no allow_credentials=True (the shared
    # helper deliberately omits it) and adds the design-system mounts the theme
    # trio in index.html needs.
    configure_interface_app(app, static_dir=STATIC_DIR)

    return app
