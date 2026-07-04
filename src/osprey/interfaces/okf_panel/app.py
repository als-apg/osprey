"""OSPREY OKF Knowledge Panel — FastAPI application factory.

A read-only browser over a facility-knowledge bundle: concept tree, markdown
reader, substring search, and a bundle-health summary. Backed directly by core
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


def _normalize_bundle_path(raw) -> Path:
    """Normalise a configured ``bundle_path`` so the panel opens the same bundle
    the other consumers of this SHARED config key do.

    ``facility_knowledge.bundle_path`` is read by three places, and the two
    existing ones do NOT normalise identically: the MCP server
    (``mcp_server/facility_knowledge/server.py::_resolve_bundle_path``) resolves
    a relative value against the ``config.yml`` directory but does not expand
    ``~``; the CLI (``cli/knowledge_cmd.py``) expands ``~`` and resolves relative
    to the CWD. This panel does both (expand ``~``, then resolve a relative value
    against the ``config.yml`` directory), which matches the MCP server EXACTLY
    for the documented/shipped relative form (e.g. ``data/facility_knowledge``).
    The point is to not regress that case: without config-dir resolution a valid
    relative ``bundle_path`` would silently drop the panel into guarded (empty-
    tab) mode whenever the web terminal isn't launched from the config directory.

    Absolute paths pass through unchanged. This never raises (the caller runs in
    a swallowed daemon-thread launcher).
    """
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p
    try:
        from osprey.utils.workspace import resolve_config_path

        return (resolve_config_path().parent / p).resolve()
    except Exception:  # noqa: BLE001 — fall back to CWD-relative; never raise.
        return p.resolve()


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
    from osprey.services.facility_knowledge.okf.bundle import OKFBundle

    try:
        bundle = OKFBundle(_normalize_bundle_path(bundle_path))
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

    @app.get("/api/search")
    async def api_search(q: str = ""):
        """Substring search; returns snippet-only hits (never full bodies)."""
        bundle, err = _bundle_or_error()
        if err:
            return err
        if not q.strip():
            return {"query": q, "results": []}

        results = []
        for concept_id, doc in bundle.search(q):
            snippet = make_snippet(doc.body, q)
            if not snippet:
                snippet = str(doc.frontmatter.get("description", ""))
            title = str(doc.frontmatter.get("title", concept_id))
            results.append({"id": concept_id, "title": title, "snippet": snippet})

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
    # shadow the API routes above. Drops the old allow_credentials=True (the
    # shared helper deliberately omits it) and adds the design-system mounts the
    # theme trio in index.html needs.
    configure_interface_app(app, static_dir=STATIC_DIR)

    return app
