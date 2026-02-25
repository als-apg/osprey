"""Reverse proxy for CUI that constrains sessions to the OSPREY project.

Mounted at ``/cui/`` in the web terminal. All requests are forwarded to
the upstream CUI server with three interceptions:

1. ``GET /api/conversations`` — injects ``projectPath=<project_cwd>``
2. ``GET /api/working-directories`` — filters response to this project only
3. ``POST /api/conversations/start`` — forces ``workingDirectory`` to project_cwd

HTML responses are rewritten so that the CUI single-page application works
correctly under the ``/cui/`` sub-path.  Vite builds CUI with absolute asset
paths (``/assets/main-xxx.js``).  Without rewriting, the browser resolves
these against the Osprey origin root and gets 404s.  The rewriter:

* Prefixes ``src="/…"`` and ``href="/…"`` attributes with ``/cui``
* Injects a small ``<script>`` that patches ``fetch`` and ``EventSource``
  so runtime API calls also route through the proxy.

Everything else (streaming, static assets, config) passes through unchanged.
This also makes CUI same-origin, eliminating cross-origin auth/cookie issues.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urlencode

import httpx
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Mount, Route

logger = logging.getLogger("osprey.interfaces.cui.proxy")

_CUI_MOUNT = "/cui"

# ---- HTML rewriting for sub-path proxy ----------------------------------- #

# Script injected into CUI's index.html so the SPA works under ``/cui/``.
#
# The patch does three things:
#
# 1. **URL rewrite** — Uses ``history.replaceState`` to change the browser URL
#    from ``/cui/`` to ``/`` *before* React loads.  This makes ``BrowserRouter``
#    (which reads ``window.location.pathname``) see ``/`` and match its routes.
#    This runs before any modules execute because it's an inline script placed
#    before the ``<script type="module">`` tag.
#
# 2. **History API** — Wraps ``pushState`` / ``replaceState`` so that React
#    Router navigations like ``navigate("/c/123")`` become ``"/cui/c/123"`` in
#    the real browser URL, keeping all requests routed through the proxy.
#    Incoming reads strip the prefix so React Router always sees clean paths.
#
# 3. **fetch / EventSource** — Prefixes absolute API paths (``/api/…``) with
#    ``/cui`` so requests route through the proxy instead of hitting the Osprey
#    root.
_RUNTIME_PATCH = f"""<script data-osprey-proxy>
(function(){{
  var P='{_CUI_MOUNT}';

  /* ---- 1. URL rewrite (before React loads) ---- */
  /* Change /cui/… to /… so BrowserRouter sees clean paths.
     Uses the real (un-patched) replaceState so it actually changes the URL. */
  var _rs0=history.replaceState.bind(history);
  (function(){{
    var p=location.pathname;
    if(p.startsWith(P)){{
      _rs0(history.state,'',(p.slice(P.length)||'/')+location.search+location.hash);
    }}
  }})();

  /* ---- 2. History API wrappers ---- */
  var _ps=history.pushState.bind(history);
  /* pushState: add /cui prefix so browser URL keeps going through proxy */
  history.pushState=function(s,t,u){{
    if(typeof u==='string'&&u.startsWith('/')&&!u.startsWith(P+'/'))u=P+u;
    return _ps(s,t,u);
  }};
  /* replaceState: same prefix logic */
  history.replaceState=function(s,t,u){{
    if(typeof u==='string'&&u.startsWith('/')&&!u.startsWith(P+'/'))u=P+u;
    return _rs0(s,t,u);
  }};
  /* popstate: when browser navigates back/forward the URL has /cui prefix —
     strip it so React Router sees clean paths. */
  window.addEventListener('popstate',function(){{
    var p=location.pathname;
    if(p.startsWith(P)){{
      _rs0(history.state,'',p.slice(P.length)||'/');
    }}
  }});

  /* ---- 3. fetch / EventSource ---- */
  var _f=window.fetch;
  window.fetch=function(u,o){{
    if(typeof u==='string'&&u.startsWith('/')&&!u.startsWith(P+'/'))u=P+u;
    return _f.call(this,u,o);
  }};
  var _E=window.EventSource;
  window.EventSource=function(u,c){{
    if(typeof u==='string'&&u.startsWith('/')&&!u.startsWith(P+'/'))u=P+u;
    return new _E(u,c);
  }};
  window.EventSource.prototype=_E.prototype;
  Object.keys(_E).forEach(function(k){{window.EventSource[k]=_E[k];}});
}})();
</script>"""

# Matches src="/" or href="/" but NOT src="//" (protocol-relative) and
# NOT paths already prefixed with the proxy mount.
_ABS_PATH_RE = re.compile(
    r"""((?:src|href)=["'])(/(?!/|""" + re.escape(_CUI_MOUNT.lstrip("/")) + r"""/))""",
)


def _rewrite_html(html_bytes: bytes) -> bytes:
    """Rewrite CUI HTML so assets and API calls route through /cui/ proxy."""
    html = html_bytes.decode("utf-8", errors="replace")

    # 1. Rewrite absolute src/href paths → /cui/…
    html = _ABS_PATH_RE.sub(rf"\g<1>{_CUI_MOUNT}\2", html)

    # 2. Inject runtime fetch/EventSource patch right after <head>
    html = html.replace("<head>", "<head>" + _RUNTIME_PATCH, 1)

    return html.encode("utf-8")


# Timeout for upstream requests (generous for npx cold-start)
_TIMEOUT = httpx.Timeout(30.0, connect=10.0)


async def _proxy(request: Request) -> Response:
    """Forward a request to the upstream CUI server, with interceptions."""
    cui_url: str | None = getattr(request.app.state, "cui_server_url", None)
    project_cwd: str = getattr(request.app.state, "project_cwd", "")

    if not cui_url:
        return Response("CUI server not available", status_code=502)

    # Strip the /cui prefix to get the original CUI path
    raw_path = request.scope.get("path", "/")
    path = raw_path.removeprefix("/cui") or "/"

    # --- Interception 1: filter conversation list by project ---
    if request.method == "GET" and path == "/api/conversations":
        params = dict(request.query_params)
        if project_cwd and "projectPath" not in params:
            params["projectPath"] = project_cwd
        upstream_url = f"{cui_url}{path}?{urlencode(params)}"
        return await _forward(request, upstream_url)

    # --- Interception 2: force workingDirectory on new conversations ---
    if request.method == "POST" and path == "/api/conversations/start":
        body = await request.json()
        if project_cwd:
            body["workingDirectory"] = project_cwd
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.post(
                f"{cui_url}{path}",
                json=body,
                headers=_forward_headers(request),
            )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=dict(resp.headers),
        )

    # --- Interception 3: filter working directories response ---
    if request.method == "GET" and path == "/api/working-directories":
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                f"{cui_url}{path}",
                headers=_forward_headers(request),
            )
        if resp.status_code == 200 and project_cwd:
            data = resp.json()
            dirs = data.get("directories", [])
            filtered = [d for d in dirs if d.get("path") == project_cwd]
            # If project not in list (no sessions yet), add it
            if not filtered:
                import os

                filtered = [
                    {
                        "path": project_cwd,
                        "shortname": os.path.basename(project_cwd),
                        "lastDate": "",
                        "conversationCount": 0,
                    }
                ]
            data["directories"] = filtered
            data["totalCount"] = len(filtered)
            import json

            return Response(
                content=json.dumps(data),
                status_code=200,
                media_type="application/json",
            )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=dict(resp.headers),
        )

    # --- Interception 4: SSE streaming (special handling) ---
    if request.method == "GET" and path.startswith("/api/stream/"):
        return await _forward_sse(request, f"{cui_url}{path}")

    # --- Default: pass through ---
    qs = str(request.query_params)
    upstream_url = f"{cui_url}{path}" + (f"?{qs}" if qs else "")
    return await _forward(request, upstream_url)


async def _forward(request: Request, upstream_url: str) -> Response:
    """Forward a standard HTTP request to the upstream URL."""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        body = await request.body()
        resp = await client.request(
            method=request.method,
            url=upstream_url,
            content=body if body else None,
            headers=_forward_headers(request),
        )
    content = resp.content
    headers = dict(resp.headers)

    # Rewrite HTML responses so CUI's absolute asset paths and API calls
    # route through the /cui/ proxy mount instead of the Osprey root.
    ct = headers.get("content-type", "")
    if "text/html" in ct:
        content = _rewrite_html(content)
        # Content length changed; update or remove to avoid truncation.
        headers.pop("content-length", None)

    return Response(
        content=content,
        status_code=resp.status_code,
        headers=headers,
    )


async def _forward_sse(request: Request, upstream_url: str) -> StreamingResponse:
    """Forward an SSE stream from the upstream CUI server."""
    client = httpx.AsyncClient(timeout=httpx.Timeout(None))

    async def stream():
        try:
            async with client.stream(
                "GET", upstream_url, headers=_forward_headers(request)
            ) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk
        except httpx.ReadError:
            pass
        finally:
            await client.aclose()

    return StreamingResponse(stream(), media_type="text/event-stream")


def _forward_headers(request: Request) -> dict[str, str]:
    """Extract headers to forward, dropping hop-by-hop and cache headers.

    Cache-validation headers (``If-None-Match``, ``If-Modified-Since``) are
    stripped because the proxy rewrites HTML responses.  Allowing conditional
    requests would let the upstream return 304 (no body), preventing the
    rewriter from running — the browser would then use stale cached HTML
    with un-rewritten absolute asset paths.
    """
    skip = {
        "host",
        "connection",
        "transfer-encoding",
        "content-length",
        "if-none-match",
        "if-modified-since",
    }
    return {k: v for k, v in request.headers.items() if k.lower() not in skip}


def create_cui_proxy_mount() -> Mount:
    """Create a Starlette Mount that proxies everything under /cui/ to CUI."""
    return Mount(
        "/cui",
        routes=[
            Route("/{path:path}", _proxy, methods=["GET", "POST", "PUT", "DELETE", "PATCH"]),
            Route("/", _proxy, methods=["GET"]),
        ],
    )
