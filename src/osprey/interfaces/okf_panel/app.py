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

The Graph view is a second, optional surface over the same panel: when the
deployment configures a facility graph (``services.graphdb.ttl_path``) the
Turtle file ``osprey build`` wrote is read once at ``create_app`` time and
served flattened at ``/api/graph``. Without one the page never shows the view
switch, and a graph that cannot be read degrades the same way.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.templating import Jinja2Templates

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
from osprey.interfaces.vendor import vendor_url

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

# ``index.html`` is rendered rather than served verbatim so it can name
# DOMPurify through ``vendor_url()``: the CDN by default, the locally fetched
# ``static/vendor/`` copy under ``OSPREY_OFFLINE``. Only a template can make
# that choice at request time.
templates = Jinja2Templates(directory=str(STATIC_DIR))
templates.env.globals["vendor_url"] = vendor_url

# Returned by every data endpoint when the panel has no bundle (unconfigured or
# unopenable path). HTTP 503 = the service is up but cannot serve data yet.
_NOT_CONFIGURED = {"error": "facility_knowledge.bundle_path not configured"}
_NOT_CONFIGURED_STATUS = 503

# Returned by ``/api/graph`` when the deployment has no facility graph.
_NO_GRAPH = {"error": "services.graphdb.ttl_path not configured"}


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
    from osprey.deployment.qmd_service import OKF_CORPUS, resolve_qmd_corpus_config
    from osprey.services.facility_knowledge.okf.bundle import OKFSearchSettings
    from osprey.services.qmd import QMDClient
    from osprey_connectors.workspace import load_osprey_config

    # Everything the ranked backend needs sits inside the guard, config load
    # included: this helper's caller wraps it in the factory's own catch-all, so
    # anything escaping here does not cost the operator ranked search — it costs
    # them the whole panel. `Exception` is the right width for the same reason
    # `_load_bundle` uses it: nothing this helper can fail at is worth more than
    # substring search.
    try:
        config = load_osprey_config()
        return {
            "qmd_client": QMDClient(resolve_qmd_corpus_config(config, OKF_CORPUS)),
            "search_settings": OKFSearchSettings.from_config(config),
        }
    except Exception:  # see above; degrade, never kill the panel.
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
    except Exception:  # a bad path must degrade, not kill the thread.
        logger.warning(
            "okf panel: could not open bundle at %s; serving guarded app.",
            bundle_path,
            exc_info=True,
        )
        return None

    log_validation_summary(validate_bundle(bundle), logger)
    return bundle


def _load_facility_records():
    """Read the render's facility file for resolving facility links, or nothing.

    The panel's factory is handed only ``bundle_path``, so it reads the project
    config itself for where the facility file sits.

    Returns:
        A :class:`~osprey.facility.records.FacilityRecords`, or ``None`` when
        no facility file is there or it cannot be read. The reading pane then
        marks every facility link unresolved rather than failing the page.
    """
    from osprey.channel_roster.sources import facility_file_path
    from osprey.facility.records import FacilityRecords
    from osprey_connectors.workspace import load_osprey_config

    # Config load included: nothing here is worth more than the reading pane.
    try:
        return FacilityRecords.from_file(facility_file_path(load_osprey_config()))
    except Exception:  # degrade to unresolved links, never kill the panel.
        logger.warning(
            "okf panel: the facility file cannot be read; facility links show unresolved.",
            exc_info=True,
        )
        return None


def _configured_graph_path():
    """Where the deployment's facility graph sits, or ``None`` when it has none.

    Returns:
        The absolute path ``services.graphdb.ttl_path`` names, resolved against
        the render as the graph store's seeder resolves it. ``None`` when the
        config names no graph or cannot be read.
    """
    from osprey.deployment.graphdb_service import (
        graph_corpus_configured,
        resolve_graphdb_service_config,
    )
    from osprey.utils.config_paths import resolve_render_relative_path
    from osprey_connectors.workspace import load_osprey_config

    try:
        config = load_osprey_config()
        if not graph_corpus_configured(config):
            return None
        resolved = resolve_graphdb_service_config(config)
        if resolved is None or not resolved.ttl_path:
            return None
        return resolve_render_relative_path(resolved.ttl_path)
    except Exception:  # degrade to no Graph view, never kill the panel.
        logger.warning(
            "okf panel: services.graphdb is misconfigured; serving no graph view.",
            exc_info=True,
        )
        return None


def _load_graph(graph_path, bundle, records):
    """Read the facility graph as the Graph view's response body, or ``None``.

    Args:
        graph_path: A Turtle file to read, or a falsy value to take the one the
            project config names.
        bundle: The served bundle, or ``None``; with ``records`` it tells which
            knowledge pages name which node.
        records: The facility records, or ``None``.

    Returns:
        The ``/api/graph`` body as JSON bytes, serialised once since the graph
        does not change while the panel runs. ``None`` when the deployment has
        no graph, or its file is absent, unreadable or empty: the page then
        shows no Graph view.
    """
    path = Path(graph_path) if graph_path else _configured_graph_path()
    if path is None:
        return None

    from osprey.interfaces.okf_panel.graph import attach_pages, page_references, read_graph

    try:
        payload = read_graph(path)
        if bundle is not None and records is not None:
            attach_pages(payload, page_references(bundle, records))
    except Exception:  # degrade to no Graph view, never kill the panel.
        logger.warning(
            "okf panel: the facility graph at %s cannot be read; serving no graph view.",
            path,
            exc_info=True,
        )
        return None
    if not payload["nodes"]:
        logger.warning("okf panel: the facility graph at %s is empty.", path)
        return None
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _facility_references(body, records):
    """The ``facility_references`` and ``unresolved`` fields of a concept response.

    Args:
        body: The concept's markdown body.
        records: The facility records, or ``None`` when none loaded.

    Returns:
        ``{"facility_references": [span, ...], "unresolved": bool}``: one
        ``{start, end, target, kinds, label, records}`` per facility link of
        the body, in body order. ``records`` lists the records the target
        resolves to (empty for a dangling link), or is ``None`` when no
        facility file loaded, which ``unresolved`` then states.
    """
    from osprey.facility.references import find_facility_links, resolved_references

    found = find_facility_links(body or "", records)
    return {
        "facility_references": resolved_references(found, records),
        "unresolved": records is None,
    }


def create_app(bundle_path=None, graph_path=None) -> FastAPI:
    """Create the OKF Knowledge Panel FastAPI application.

    Args:
        bundle_path: Filesystem path to the OKF bundle directory, resolved from
            ``facility_knowledge.bundle_path`` by the registry's
            ``factory_config_kwargs``. Falsy or unopenable → guarded app.
        graph_path: A facility-graph Turtle file to serve in the Graph view.
            Omitted, the one ``services.graphdb.ttl_path`` names is served, and
            a deployment that names none has no Graph view.

    Returns:
        Configured FastAPI application with the bundle cached on
        ``app.state.bundle`` (or ``None`` in guarded mode) and the Graph view's
        response body on ``app.state.graph`` (or ``None`` without a graph).
    """
    app = FastAPI(
        title="OSPREY OKF Knowledge Panel",
        description="Read-only browser over a facility-knowledge (OKF) bundle",
        version="1.0.0",
    )

    app.state.bundle = _load_bundle(bundle_path)
    app.state.facility_records = _load_facility_records()
    app.state.graph = _load_graph(graph_path, app.state.bundle, app.state.facility_records)

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
    async def root(request: Request) -> Response:
        """Serve the SPA shell, or a placeholder when it is not installed."""
        index_html = STATIC_DIR / "index.html"
        if index_html.exists():
            return templates.TemplateResponse(
                request, "index.html", {"graph": app.state.graph is not None}
            )
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
        ``facility_references`` lists each facility link of the body with the
        records it resolves to; ``unresolved`` says no facility file was read.
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
        return {
            "id": id,
            "frontmatter": doc.frontmatter,
            "body": doc.body,
            **_facility_references(doc.body, app.state.facility_records),
        }

    # Deliberately `def`, not `async def` — the only handler here that is.
    # ``bundle.search`` is synchronous, and on the ranked path it makes blocking
    # HTTP round-trips to the sidecar (30 s timeout, and seconds at a time with
    # ``rerank: true`` even when the sidecar is healthy). Awaited on the event
    # loop that work would stall every other request in the process, including
    # the reading pane. A sync route runs in Starlette's threadpool instead, so
    # a slow or hung sidecar costs the search and nothing else. The other
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

    @app.get("/api/graph")
    async def api_graph() -> Response:
        """The facility graph as ``{"nodes", "edges"}``.

        The shape is :func:`osprey.interfaces.okf_panel.graph.graph_payload`'s.
        404 when the deployment has no facility graph: unlike the bundle, a
        graph is optional, so its absence is not a service fault.
        """
        if app.state.graph is None:
            return JSONResponse(_NO_GRAPH, status_code=404)
        return Response(app.state.graph, media_type="application/json")

    # Shared CORS + middleware + static mounts (/design-system, /static/fonts,
    # /static) applied last so they wrap the fully-assembled app and never
    # shadow the API routes above. Carries no allow_credentials=True (the shared
    # helper deliberately omits it) and adds the design-system mounts the theme
    # trio in index.html needs.
    configure_interface_app(app, static_dir=STATIC_DIR)

    return app
