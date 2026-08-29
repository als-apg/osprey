"""Reverse proxy for companion panel servers running inside the container.

Companion servers (artifact gallery, ARIEL, channel-finder, lattice, OKF)
bind to 127.0.0.1 inside the Docker container.  The browser cannot reach them
directly, so this proxy forwards ``/panel/{panel_id}/{path}`` to the internal
server and rewrites root-absolute paths in HTML/JS/CSS responses.

The proxy is also a **credential boundary**. Every request arriving here was
authenticated as the operator at the terminal's own origin, and the browser
attaches that proof — the session cookie, and behind the multi-user reverse
proxy an ``X-Osprey-Terminal-Secret`` header — to panel requests too, because
they share that origin. A panel backend is a different trust domain: a
facility Grafana, a sidecar built from some other release, or, where runtime
registration is enabled, a URL an agent picked. So the inbound credentials are
stripped here (:data:`_STRIPPED_REQUEST_HEADERS`) and the operator secret is
re-injected only toward a backend that is both declared by configuration and
addressed at loopback (:func:`_terminal_secret_header`). Every case that is not
plainly both is treated as neither.

The boundary runs in **both directions**. A response coming back is emitted at
the terminal's own origin, which hands a backend a whole class of headers that
act on the *console* rather than on the panel's own document — evicting the
operator's session, relaxing the origin for an attacker's, navigating the
operator off-site. :func:`_filter_response_headers` drops that whole class
(:data:`_ORIGIN_ACTING_RESPONSE_HEADERS`) at every emit site. No relayed
response follows a redirect (:func:`_request_no_redirect`), so the proxy's
network reach cannot be steered by a backend either, and a redirect naming a
third origin is refused outright rather than relayed
(:func:`_rewrite_redirect_location`) so a console URL is never an open redirect.

What the injection still buys a backend, stated plainly: the header carries the
**full** operator secret, so a backend that earns it can turn around and act as
the operator against the terminal's own API. That is accepted here — the
backends that earn it are config-declared loopback sidecars of the same
deployment — and the follow-up is a scoped per-backend credential rather than
the operator's own.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import mimetypes
import os
import re
from pathlib import Path
from urllib.parse import ParseResult, urljoin, urlparse, urlunsplit

import httpx
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse

from osprey.interfaces.common_middleware import compute_url_prefix
from osprey.interfaces.web_auth import get_web_credentials
from osprey.registry.web import FRAMEWORK_WEB_SERVERS, panel_url_state_attr
from osprey.utils.http_proxy import HOP_BY_HOP

logger = logging.getLogger(__name__)

router = APIRouter()

# Paths that companion servers commonly use as root-absolute references.
# Only these prefixes are rewritten inside string delimiters to avoid
# false positives on arbitrary ``/`` characters.
_REWRITE_PREFIXES = (
    "/static/",
    "/design-system/",
    "/api/",
    "/files/",
    "/health",
    "/checks",
    "/ws/",
    "/assets/",
    "/dashboard/",
    # Tiled's built-in web UI (served under /ui/) references root-absolute
    # /ui/assets/... bundle paths.
    "/ui/",
    # Event dispatcher endpoints. Without these, the dashboard served through
    # /panel/events/* would call `/webhook/...` at the browser origin (the web
    # terminal) instead of going through this proxy — the dispatcher never
    # sees the request and the user gets a 404.
    "/webhook/",
    "/retry/",
    "/trigger/",
)

# Content types eligible for path rewriting.
_REWRITABLE_TYPES = {
    "text/html",
    "text/javascript",
    "application/javascript",
    "text/css",
}

# The proxy-wide caching default, applied to any proxied response whose
# upstream set no Cache-Control of its own. Panels ship unversioned asset
# filenames (panel.js, not panel-<hash>.js), and a header-less response lets
# the browser cache heuristically — so a panel redeploy silently doesn't
# reach an operator's already-open browser. Filled in with setdefault, never
# overridden: an upstream that made a deliberate caching decision (e.g. the
# artifact gallery's immutable versioned vendor bundles) keeps it.
_DEFAULT_NO_CACHE = "no-cache, no-store, must-revalidate"

# Panel ID → app.state attribute name, derived from the web-server registry
# rather than hand-listed. Registry keys and panel ids are two namespaces
# (``artifact``/``artifacts``, ``channel_finder``/``channel-finder``), and each
# consumer that kept its own translation table drifted from the others; the
# relation lives once, on ``WebServerDefinition.panel_id``.
#
# The attribute name comes from ``registry.web.panel_url_state_attr`` — the same
# function ``web_terminal/app.py`` publishes under — rather than being spelled
# out a second time here. Both ends import it from the registry because they
# cannot import each other, and a convention agreed on by two independent
# f-strings is a convention only until one of them is edited.
_PANEL_STATE_MAP = {
    definition.panel_id: panel_url_state_attr(key)
    for key, definition in FRAMEWORK_WEB_SERVERS.items()
}


def _resolve_panel_url(request: Request, panel_id: str) -> str | None:
    """Map a panel ID to its internal server URL, or ``None`` if unavailable."""
    attr = _PANEL_STATE_MAP.get(panel_id)
    if attr:
        return getattr(request.app.state, attr, None)

    # Custom panels: look up by ID in the custom panels list.
    for cp in getattr(request.app.state, "custom_panels", []):
        if cp.get("id") == panel_id:
            url = cp.get("url", "")
            return url if url else None

    return None


def _panel_is_config_defined(scope: Request | WebSocket, panel_id: str) -> bool:
    """True only if ``panel_id`` resolves to a config-declared custom panel.

    Runtime registrations (POST /api/panels/register) never carry the
    ``configDefined`` marker, so a registration that squats a config panel's id
    cannot inherit that panel's server-side credential injection below.

    Args:
        scope: The request or websocket being proxied; only ``app.state`` is
            read, so both connection kinds answer the same way.
        panel_id: The panel id from the proxied URL.
    """
    for cp in getattr(scope.app.state, "custom_panels", []):
        if cp.get("id") == panel_id:
            return bool(cp.get("configDefined"))
    return False


#: Request headers that are never forwarded to a panel backend.
#:
#: All three carry the operator's identity at the terminal: the browser session
#: cookie, an ``Authorization`` header, and the ``X-Osprey-Terminal-Secret``
#: the multi-user reverse proxy stamps on. The browser attaches them to panel
#: requests because a panel is served from the terminal's own origin, not
#: because the panel is entitled to them — and a backend handed any of the
#: three could replay it against the terminal as the operator.
#:
#: Stripped for every panel without exception, framework sidecars included. A
#: backend that genuinely needs to know its caller is the operator gets the
#: secret from :func:`_terminal_secret_header` instead: a decision this process
#: makes per backend, rather than one the browser's header list makes for it.
#:
#: The trade this buys, stated plainly: a panel backend cannot run a cookie
#: session of its own through this proxy, because its cookies ride the same
#: stripped header as the terminal's.
_STRIPPED_REQUEST_HEADERS = frozenset(
    {
        "cookie",
        "authorization",
        "x-osprey-terminal-secret",
    }
)


def _is_stripped_header(name: str) -> bool:
    """Whether an inbound header name must not be forwarded to a panel backend.

    Matches the names in :data:`_STRIPPED_REQUEST_HEADERS` case-insensitively,
    and also their underscore spelling. A backend behind a WSGI/CGI layer reads
    request headers as ``HTTP_`` environment variables with every ``-`` folded
    to ``_``, so ``X-Osprey-Terminal-Secret`` and ``X_Osprey_Terminal_Secret``
    arrive at the application as the very same ``HTTP_X_OSPREY_TERMINAL_SECRET``
    key. Stripping only the dash spelling would let the underscore spelling ride
    through and reconstitute the credential on the far side; folding ``_`` to
    ``-`` before the check closes that.
    """
    return name.lower().replace("_", "-") in _STRIPPED_REQUEST_HEADERS


#: Response headers that act on the **embedding origin** rather than on the
#: panel's own document — the class this proxy must refuse to relay.
#:
#: A proxied response is emitted at the terminal's own origin, so any header
#: whose effect is scoped to "the origin of this response" is really a header
#: the backend gets to aim at the operator's console. That is the shape of the
#: rule, and membership follows from it rather than from a list of known
#: attacks; a header a panel sends about *itself* (``Content-Security-Policy``,
#: ``Link``) is not in the class and is relayed.
#:
#: * ``set-cookie`` — a backend's cookie is stored *for the terminal*. The
#:   terminal's session cookie is ``osprey_terminal_session_<port>`` on
#:   ``Path=/``, a name a hostile panel can derive, so relaying the header would
#:   let a panel overwrite the operator's session and log them out — and behind
#:   the multi-user proxy, do it to every port it can enumerate.
#: * ``set-cookie2`` — the obsolete RFC 2965 spelling, so a backend cannot
#:   reach a lenient browser by the older name.
#: * ``clear-site-data`` — reaches the same eviction without naming a cookie:
#:   the directive clears cookies and storage for the *response's* origin, which
#:   here is the console's, and browsers scope the clear to the registrable
#:   domain.
#: * ``refresh`` — the header form of ``<meta http-equiv=refresh>``, honoured by
#:   Chrome and Safari. It navigates the browser off-site from a URL that reads
#:   as the console, on **any** status: unlike ``Location`` it needs no ``30x``,
#:   so a plain 200 panel page can carry it.
#: * ``www-authenticate`` — raises a native credential dialog whose origin line
#:   reads as the console and whose realm text the backend writes. Stripping
#:   costs nothing: what the operator types comes back as ``Authorization``,
#:   which :data:`_STRIPPED_REQUEST_HEADERS` drops before the backend sees it,
#:   so a backend running its own HTTP auth cannot be driven through this proxy
#:   in any case.
#:
#: Nothing legitimate is lost by dropping the cookie headers either: the request
#: direction already strips ``Cookie``, so a backend cannot run a cookie session
#: through this proxy at all.
_ORIGIN_ACTING_RESPONSE_HEADERS = frozenset(
    {
        "set-cookie",
        "set-cookie2",
        "clear-site-data",
        "refresh",
        "www-authenticate",
    }
)

#: Prefix of the CORS response headers, dropped for the same reason.
#:
#: A panel proxied here is *same-origin* with the terminal, so it needs no CORS
#: headers at all. Relaying them lets a backend publish
#: ``Access-Control-Allow-Origin: https://attacker.test`` with
#: ``Allow-Credentials: true`` **on the terminal's origin**, which is an
#: attacker-chosen site being told it may read credentialed responses from the
#: operator's console.
_CORS_RESPONSE_PREFIX = "access-control-"


def _filter_response_headers(headers: httpx.Headers) -> dict[str, str]:
    """The upstream headers that may be re-emitted to the browser.

    One helper for every emit site — the standard response, the SSE stream and
    the relayed redirect — because three copies of a filter list is three
    chances for one of them to be the stale one, and the header that goes
    missing from a copy is a credential boundary, not a formatting detail.

    Args:
        headers: The upstream response's headers.

    Returns:
        A plain dict carrying neither hop-by-hop headers, nor any header in
        :data:`_ORIGIN_ACTING_RESPONSE_HEADERS`, nor any CORS header.
    """
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in HOP_BY_HOP
        and key.lower() not in _ORIGIN_ACTING_RESPONSE_HEADERS
        and not key.lower().startswith(_CORS_RESPONSE_PREFIX)
    }


#: Status codes on which an HTTP client follows the response's ``Location``.
#: ``304`` is deliberately absent: it shares the ``3xx`` range but is a cache
#: validation answer, and relaying it as a redirect would strip the body
#: semantics the conditional-request cycle depends on.
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


def _is_redirect(resp: httpx.Response) -> bool:
    """Whether this upstream response is a redirect to relay rather than emit."""
    return resp.status_code in _REDIRECT_STATUSES


#: The header carrying the operator secret to a trusted backend.
#:
#: Spelled exactly as ``deployment.web_terminals.render.TERMINAL_SECRET_HEADER``
#: writes it into the generated nginx ``proxy_set_header`` line, so a panel
#: backend reads one header name whether the request came through nginx or
#: through this proxy. A test pins the two spellings together rather than this
#: request-path module importing the deployment renderer.
TERMINAL_SECRET_HEADER = "X-Osprey-Terminal-Secret"


def _backend_is_loopback(backend_url: str) -> bool:
    """Whether ``backend_url``'s host is unambiguously this machine's loopback.

    Literal addresses are classified with :mod:`ipaddress`, unwrapping
    IPv4-mapped IPv6 first so ``::ffff:127.0.0.1`` cannot present itself as
    "some other address". Only the exact name ``localhost`` counts among names,
    not the ``*.localhost`` subtree. Everything else, including an unparseable
    URL or a missing host, is False.

    **No DNS lookup happens here, and that is a security choice before it is a
    performance one.** Resolving would put a round trip on every proxied
    request, but worse, a name that resolves to 127.0.0.1 at lookup time is
    still a name somebody else controls: the zone can be re-pointed between the
    lookup and the connection, and the secret would follow it off-box. Only a
    host that is loopback by its own spelling earns injection.

    ``*.localhost`` is *not* trusted here even though RFC 6761 reserves it for
    loopback, because the promise is only advisory: the actual ``connect()``
    still goes through the system resolver, and glibc without
    ``systemd-resolved``, musl, and Docker's embedded DNS can all forward a
    name like ``panel.localhost`` to an upstream server that answers with any
    address it likes. Trusting the spelling while the socket obeys DNS is
    exactly the split that would carry the secret off-box. No shipped config
    uses such a name; a backend that wants the secret spells its host
    ``127.0.0.1`` or ``localhost``.

    Args:
        backend_url: The panel's raw backend URL, as configured or registered.

    Returns:
        True when the URL's host is a loopback literal or the exact name
        ``localhost``; False in every other case, including every error.
    """
    try:
        host = (urlparse(backend_url).hostname or "").strip().lower()
    except ValueError:
        return False

    if not host:
        return False
    if host == "localhost":
        return True

    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.is_loopback


def _panel_origin_is_declared(scope: Request | WebSocket, panel_id: str) -> bool:
    """Whether this panel's backend URL was declared by OSPREY, not registered at runtime.

    Two kinds of declaration count. A framework panel's URL comes from
    ``app.state``, written by the launcher that started the sidecar, and
    :func:`_resolve_panel_url` consults that first — so a runtime registration
    squatting a framework panel id never becomes the forward target and never
    inherits its trust. A custom panel counts only when the config loader
    stamped ``configDefined`` on it, which ``POST /api/panels/register`` cannot
    do.

    Args:
        scope: The request or websocket being proxied.
        panel_id: The panel id from the proxied URL.
    """
    attr = _PANEL_STATE_MAP.get(panel_id)
    if attr and getattr(scope.app.state, attr, None):
        return True
    return _panel_is_config_defined(scope, panel_id)


def _terminal_secret_header(
    scope: Request | WebSocket, panel_id: str, backend_url: str
) -> dict[str, str]:
    """The terminal-secret header to send upstream, or ``{}`` for no injection.

    Both halves are required. A backend addressed at loopback but registered at
    runtime is not enough: runtime registration is a path the agent's own
    sandbox can reach, and a loopback listener it registered would be handed
    the operator secret out of this process. A config-declared backend pointing
    off-box is not enough either: the secret would leave the machine to a host
    the operator's browser never authenticated to, over a hop that may not even
    be TLS.

    When either half is missing — or the process cannot produce credentials at
    all — nothing is injected and the panel simply loads unauthenticated. That
    is the failure an operator can see and act on; the alternative failure,
    a leaked secret, is silent.

    Args:
        scope: The request or websocket being proxied.
        panel_id: The panel id from the proxied URL.
        backend_url: The resolved backend URL the request is being forwarded to.

    Returns:
        A single-entry mapping ready to merge into the outbound headers, or an
        empty mapping when this backend must not receive the secret.
    """
    if not _panel_origin_is_declared(scope, panel_id):
        return {}
    if not _backend_is_loopback(backend_url):
        return {}

    try:
        credentials = get_web_credentials(getattr(scope, "app", None))
    except Exception:  # pragma: no cover - see web_auth._populate's fatal path
        logger.warning(
            "No web credentials available; proxying panel %s without the operator secret",
            panel_id,
        )
        return {}

    secret = credentials.operator_secret
    return {TERMINAL_SECRET_HEADER: secret} if secret else {}


async def _request_no_redirect(
    client: httpx.AsyncClient,
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    content: bytes | None,
) -> httpx.Response:
    """Send one panel request without letting httpx chase redirects.

    The shared proxy client is built with ``follow_redirects=True`` (see
    ``web_terminal.app``'s lifespan). Applied to **every** proxied request, not
    only the credential-bearing ones, for two separate reasons:

    * **The injected secret.** On a cross-origin redirect httpx re-sends every
      header except ``Authorization`` and ``Cookie``, so a config-declared
      loopback backend answering ``30x`` with an off-box ``Location`` would hand
      the operator secret straight to that host.
    * **The proxy's network reach.** A request carrying no credential still has
      something to leak: this process's position. It runs inside the container,
      on the compose network, next to loopback services the browser cannot
      address — so an off-box panel answering ``302 Location:
      http://127.0.0.1:<admin-port>/`` would have the proxy fetch that for it.
      "Nothing on it to leak" was true of the headers and false of the socket.

    The redirect is relayed to the browser instead (:func:`_relay_redirect`),
    whose re-request is re-gated by the terminal's own auth and, if it comes
    back through a panel path, earns a freshly evaluated injection.

    The ``TypeError`` fallback covers a client double that does not accept the
    per-request ``follow_redirects`` keyword. Such a stub does no redirect
    following of its own, so one un-parameterised call is already safe; only the
    keyword-specific ``TypeError`` is caught, so an unrelated one still surfaces.
    """
    try:
        return await client.request(
            method=method,
            url=url,
            headers=headers,
            content=content,
            follow_redirects=False,
        )
    except TypeError as exc:
        if "follow_redirects" not in str(exc):
            raise
        return await client.request(method=method, url=url, headers=headers, content=content)


def _same_origin(parsed: ParseResult, other_url: str) -> bool:
    """Whether an absolute (or protocol-relative) ``Location`` names ``other_url``.

    A protocol-relative ``//host/path`` carries no scheme of its own — it
    inherits the document's — so only the authority is compared for that form.
    The authority is compared whole, userinfo included, so a ``Location`` that
    decorates a matching host with credentials counts as a different origin
    rather than as the backend's own.
    """
    other = urlparse(other_url)
    if parsed.netloc.lower() != other.netloc.lower():
        return False
    if parsed.scheme and other.scheme and parsed.scheme.lower() != other.scheme.lower():
        return False
    return True


def _rewrite_redirect_location(
    location: str,
    backend_url: str,
    panel_prefix: str,
    *,
    base_path: str,
    terminal_origin: str,
) -> str | None:
    """Point a relayed redirect's ``Location`` back through the panel proxy.

    A backend's own ``Location`` is written for the backend's origin, so a
    redirect meant to stay inside the panel — the common ``/`` → ``/dashboard/``
    shape — must be re-based onto ``<prefix>/panel/<id>`` or the browser would
    resolve it against the terminal's origin and escape the frame. Three forms
    are re-based: an absolute URL to the backend's own origin, a root-absolute
    path, and a relative reference.

    The relative case is resolved here rather than left to the browser because
    the two do not always agree. The browser resolves against the *request* URL,
    and :func:`proxy_panel_root` serves ``/panel/<id>`` with no trailing slash —
    so a backend answering ``Location: dashboard/`` (meaning ``/dashboard/`` on
    its own root) would send the operator to ``/panel/dashboard/``, a different
    panel id. Resolving against ``base_path``, which spells the panel prefix as
    the directory it is, keeps the id.

    An absolute ``Location`` naming a **third** origin — neither the backend's
    nor the terminal's — is refused rather than relayed, and this returns
    ``None`` to say so. Relaying it would make every ``/panel/<id>/...`` URL an
    open redirect *on the console's own origin*: a link that reads as the
    operator's trusted console and lands on a page the panel backend chose.
    Nothing shipped needs the pass-through — every preset panel URL is loopback,
    and a backend that bounces its callers to an identity provider already
    cannot be driven as a custom panel, because the proxy strips the credentials
    such a flow depends on. A ``Location`` naming the terminal itself is not a
    third origin and is relayed unchanged; it names the same destination a
    root-absolute path would, and the browser's re-request is re-gated either
    way.

    Args:
        location: The backend's raw ``Location`` header value.
        backend_url: The panel's backend URL, whose origin is the panel's own.
        panel_prefix: ``<outer prefix>/panel/<id>``, the namespace to re-base into.
        base_path: The proxied request's own path with the panel prefix treated
            as a directory, used as the base for a relative reference.
        terminal_origin: This request's own ``scheme://host[:port]``.

    Returns:
        The ``Location`` to emit, or ``None`` when the redirect must be refused.
    """
    if not location:
        return location

    try:
        parsed = urlparse(location)
    except ValueError:
        # A Location this process cannot even parse (a malformed IPv6 authority,
        # say) is one it cannot classify, and an unclassifiable destination on
        # the console's origin is exactly what the refusal below exists for.
        return None
    if parsed.scheme or parsed.netloc:
        if _same_origin(parsed, backend_url):
            # Re-base onto the panel namespace, keeping query and fragment.
            remainder = urlunsplit(("", "", parsed.path, parsed.query, parsed.fragment))
            return panel_prefix + remainder
        if _same_origin(parsed, terminal_origin):
            return location
        return None

    if location.startswith("/"):
        return panel_prefix + location
    return urljoin(base_path, location)


def _relay_redirect(
    resp: httpx.Response,
    panel_prefix: str,
    backend_url: str,
    *,
    body: bytes = b"",
    base_path: str,
    terminal_origin: str,
) -> Response:
    """Relay a backend redirect to the browser rather than following it upstream.

    Following a redirect server-side is what would carry the operator secret off
    loopback, and what would let a backend aim this process at a host the
    browser cannot reach — so a ``30x`` is never followed here. It is handed back
    to the browser instead, with its ``Location`` re-based into the panel's
    namespace (:func:`_rewrite_redirect_location`) and its headers filtered like
    any other (:func:`_filter_response_headers`).

    A redirect off to a third origin is neither followed nor relayed: the
    browser is answered ``502`` with no ``Location`` at all, so the console's
    origin never carries an attacker-chosen destination.
    """
    new_location: str | None = None
    if "location" in resp.headers:
        new_location = _rewrite_redirect_location(
            resp.headers["location"],
            backend_url,
            panel_prefix,
            base_path=base_path,
            terminal_origin=terminal_origin,
        )
        if new_location is None:
            logger.warning(
                "Refusing off-site panel redirect from %s to %s",
                backend_url,
                resp.headers["location"],
            )
            return Response(
                content="Panel redirected off-site",
                status_code=502,
                headers={"cache-control": _DEFAULT_NO_CACHE},
                media_type="text/plain",
            )

    resp_headers = _filter_response_headers(resp.headers)
    if new_location is not None:
        resp_headers["location"] = new_location
    resp_headers.setdefault("cache-control", _DEFAULT_NO_CACHE)
    return Response(content=body, status_code=resp.status_code, headers=resp_headers)


def _panel_json_rewrite_paths(request: Request, panel_id: str) -> tuple[str, ...]:
    """The panel's configured ``rewrite_json_paths`` suffixes (usually empty)."""
    for cp in getattr(request.app.state, "custom_panels", []):
        if cp.get("id") == panel_id:
            paths = cp.get("rewriteJsonPaths") or ()
            if isinstance(paths, (list, tuple)):
                return tuple(str(p).rstrip("/") for p in paths if p)
            return (str(paths).rstrip("/"),)
    return ()


def _rewrite_content(body: str, panel_id: str, outer_prefix: str = "") -> str:
    """Rewrite root-absolute paths inside string delimiters for proxied content.

    Only touches known prefixes inside ``"``, ``'``, or backtick delimiters.
    CDN URLs (``https://…``), protocol-relative (``//…``), and data URIs
    are unaffected because the pattern requires a delimiter immediately
    before the ``/``.

    Args:
        body: The response body to rewrite.
        panel_id: The panel ID this content was proxied from.
        outer_prefix: The per-user mount prefix (``/u/<user>`` or ``""``, see
            ``compute_url_prefix()``). Prepended so a panel's internal
            assets/APIs resolve under the outer prefix too, not just
            ``/panel/<id>``. Empty prefix ⇒ unchanged (pre-refactor) output.
    """
    prefix = f"{outer_prefix}/panel/{panel_id}"

    for path in _REWRITE_PREFIXES:
        # Match path inside string delimiters: "/static/..." → "/panel/id/static/..."
        # The lookbehind ensures we only match after a quote character.
        body = re.sub(
            r"""(?<=["'`])""" + re.escape(path),
            prefix + path,
            body,
        )

    # Rewrite bare '/api' (without trailing slash) inside delimiters.
    body = re.sub(r"""(?<=["'`])/api(?=["'`])""", f"{prefix}/api", body)

    # Rewrite href="/" to href="/panel/{id}/"
    body = body.replace('href="/"', f'href="{prefix}/"')
    body = body.replace("href='/'", f"href='{prefix}/'")

    return body


#: The hub's own copy of the shared design system — the same directory
#: ``_app_setup.mount_shared_static()`` serves at ``/design-system``.
_DESIGN_SYSTEM_DIR = Path(__file__).resolve().parents[2] / "design_system" / "static"

#: Text asset suffixes that get the same root-absolute rewrite proxied
#: HTML/JS/CSS receives, mapped to the content type to serve them as.
_DS_TEXT_TYPES = {
    ".css": "text/css",
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".html": "text/html",
    ".json": "application/json",
    ".map": "application/json",
}


@router.api_route(
    "/panel/{panel_id}/design-system/{asset_path:path}",
    methods=["GET", "HEAD"],
)
async def proxy_panel_design_system(panel_id: str, asset_path: str, request: Request):
    """Serve the HUB's design-system assets to an embedded panel.

    MUST stay declared above :func:`proxy_panel` — that route's
    ``{path:path}`` is a catch-all and Starlette matches in declaration
    order, so moving this below it makes it unreachable.

    Every panel's HTML loads ``/design-system/css/tokens.css`` root-absolute,
    which :func:`_rewrite_content` turns into
    ``/panel/<id>/design-system/css/tokens.css``. Left to the generic proxy
    that request reaches the *sidecar*, which serves whatever design-system
    copy its own build shipped. A sidecar running an older release — a
    container image built before a token change, say — then renders a
    different palette than the terminal hosting it, and the theme the hub
    broadcasts resolves to the wrong colors inside that one frame.

    Intercepting here makes the shared design system single-sourced from the
    hub for every embedded panel, so theme consistency does not depend on
    each sidecar's build being in step. Sidecars still serve their own
    ``/design-system`` on their standalone URLs; only the embedded path is
    redirected to the hub's copy.

    Falls through to the sidecar (404 here → the generic proxy never runs, so
    a genuinely missing asset 404s) rather than masking a bad path.
    """
    if not _DESIGN_SYSTEM_DIR.is_dir():  # pragma: no cover - packaging guard
        return Response(content="design system unavailable", status_code=404)

    # Contain the path: reject anything that escapes the static root via
    # traversal or an absolute path before touching the filesystem.
    candidate = (_DESIGN_SYSTEM_DIR / asset_path).resolve()
    if not candidate.is_relative_to(_DESIGN_SYSTEM_DIR.resolve()) or not candidate.is_file():
        return Response(content="Not found", status_code=404)

    headers = {"cache-control": _DEFAULT_NO_CACHE}
    media_type = _DS_TEXT_TYPES.get(candidate.suffix.lower())

    if media_type is None:
        guessed, _ = mimetypes.guess_type(candidate.name)
        return Response(
            content=candidate.read_bytes(),
            headers=headers,
            media_type=guessed or "application/octet-stream",
        )

    # Text assets carry root-absolute self-references (e.g.
    # osprey-theme-switcher.js dynamically imports '/design-system/js/
    # theme-manager.js'). Apply the proxy's own rewrite so those resolve back
    # into this panel's namespace — and therefore back to the hub's copy.
    text = _rewrite_content(candidate.read_text(encoding="utf-8"), panel_id, compute_url_prefix())
    return Response(content=text, headers=headers, media_type=media_type)


@router.api_route(
    "/panel/{panel_id}/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
)
async def proxy_panel(panel_id: str, path: str, request: Request):
    """Forward a request to the companion panel server.

    Inbound operator credentials are stripped and, for a config-declared
    loopback backend only, the operator secret is re-issued — see this
    module's docstring for why the proxy is the boundary where that happens.
    """
    backend_url = _resolve_panel_url(request, panel_id)
    if not backend_url:
        return Response(
            content=f"Panel '{panel_id}' is not available",
            status_code=404,
        )

    outer_prefix = compute_url_prefix()

    # Build the target URL.
    target = f"{backend_url.rstrip('/')}/{path}"
    if request.url.query:
        target = f"{target}?{request.url.query}"

    client: httpx.AsyncClient = request.app.state.proxy_client

    # Forward headers, dropping host (httpx sets it), hop-by-hop, and
    # accept-encoding. httpx must negotiate content-codings it can itself
    # decode (it transparently decompresses gzip/br); forwarding the
    # browser's Accept-Encoding list (which includes zstd) lets upstreams
    # pick zstd, which httpx cannot decode — it passes the compressed bytes
    # through unchanged while the proxy strips the content-encoding header
    # below, so the browser renders raw compressed garbage.
    #
    # The credential headers go too (see _STRIPPED_REQUEST_HEADERS): the
    # browser attaches the operator's proof to a panel request because the
    # panel shares the terminal's origin, and the backend on the other side of
    # this hop is not entitled to it.
    #
    # ``Origin`` stops here as well. It names the page that made the request —
    # the terminal — and the terminal's own gate has already held it against
    # the terminal's origin before this route ran. Relayed onward it reaches a
    # backend whose gate compares it against *that backend's* address
    # (``localhost:10071`` for the bluesky sidecar), which the terminal's origin
    # never equals: every write from a proxied panel would be refused as
    # cross-origin while every read succeeds. On this hop the proxy is a new
    # client, and a client that sends no ``Origin`` is what a gated backend
    # admits on the operator secret injected below.
    fwd_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in HOP_BY_HOP
        and k.lower() not in ("host", "accept-encoding", "origin")
        and not _is_stripped_header(k)
    }
    fwd_headers["x-forwarded-prefix"] = f"{outer_prefix}/panel/{panel_id}"

    # Re-issue the operator secret, but only toward a backend that is both
    # config-declared and on loopback. This runs after the strip above, so a
    # header the browser sent can never masquerade as an injected one.
    secret_header = _terminal_secret_header(request, panel_id, backend_url)
    fwd_headers.update(secret_header)

    # The event-dispatcher dashboard endpoints are bearer-gated. Inject the
    # dispatcher token server-side for the EVENTS panel only, so the browser
    # never holds it (and other panels are unaffected). The web-terminal process
    # picks up EVENT_DISPATCHER_TOKEN from the project .env via load_dotenv.
    #
    # Gated exactly like the operator secret above — config-declared AND
    # loopback — rather than on the id string. Config origin alone keeps a
    # runtime registration from squatting the id, but it would still send the
    # deployment's dispatcher token over the network to whatever host an
    # off-box `events` panel names; a shared bearer token leaving the machine is
    # the same failure as the operator secret leaving it.
    if (
        panel_id == "events"
        and _panel_is_config_defined(request, "events")
        and _backend_is_loopback(backend_url)
    ):
        token = os.environ.get("EVENT_DISPATCHER_TOKEN", "")
        if token:
            fwd_headers["authorization"] = f"Bearer {token}"

    outer_panel_prefix = f"{outer_prefix}/panel/{panel_id}"

    # The base a relative ``Location`` is resolved against, and the origin a
    # relayed one is allowed to name. The panel prefix is spelled as a directory
    # so the root route (``/panel/<id>``, no trailing slash) does not resolve a
    # relative redirect one segment too high — into a different panel id.
    relative_base = f"{outer_panel_prefix}/{path}"
    terminal_origin = f"{request.url.scheme}://{request.url.netloc}"

    try:
        body = await request.body()

        # SSE: stream the response.
        if "text/event-stream" in request.headers.get("accept", ""):
            # The follow is suppressed for the same reasons as the standard
            # path below — a redirect would carry an injected secret off
            # loopback, and would steer the proxy's own reach even without one.
            # An SSE endpoint answering with a redirect is relayed to the
            # browser instead of streamed.
            upstream = await client.send(
                client.build_request(
                    method=request.method,
                    url=target,
                    headers=fwd_headers,
                    content=body if body else None,
                    # SSE streams idle between events — disable the read
                    # timeout so quiet periods don't kill the connection.
                    timeout=httpx.Timeout(None, connect=5.0),
                ),
                stream=True,
                follow_redirects=False,
            )

            if _is_redirect(upstream):
                await upstream.aclose()
                return _relay_redirect(
                    upstream,
                    outer_panel_prefix,
                    backend_url,
                    base_path=relative_base,
                    terminal_origin=terminal_origin,
                )

            async def _stream():
                try:
                    async for chunk in upstream.aiter_bytes():
                        yield chunk
                finally:
                    await upstream.aclose()

            resp_headers = _filter_response_headers(upstream.headers)
            resp_headers.setdefault("cache-control", _DEFAULT_NO_CACHE)
            return StreamingResponse(
                _stream(),
                status_code=upstream.status_code,
                headers=resp_headers,
                media_type="text/event-stream",
            )

        # Standard request. No proxied request follows a redirect, whether or
        # not it carries the injected secret: the secret must not ride one off
        # loopback, and the proxy's reach into the container network must not be
        # steerable by a backend either (see _request_no_redirect). Redirects are
        # relayed to the browser, which re-requests through the gate.
        resp = await _request_no_redirect(
            client,
            method=request.method,
            url=target,
            headers=fwd_headers,
            content=body if body else None,
        )
        if _is_redirect(resp):
            return _relay_redirect(
                resp,
                outer_panel_prefix,
                backend_url,
                body=resp.content,
                base_path=relative_base,
                terminal_origin=terminal_origin,
            )
    except httpx.RequestError as exc:
        logger.warning("Proxy error for panel %s: %s", panel_id, exc)
        return Response(
            content=f"Upstream connection failed: {exc}",
            status_code=502,
        )

    # Filter response headers (see _filter_response_headers); fill in the
    # no-cache default when the upstream made no caching decision of its own
    # (see _DEFAULT_NO_CACHE).
    resp_headers = _filter_response_headers(resp.headers)
    resp_headers.setdefault("cache-control", _DEFAULT_NO_CACHE)

    content_type = resp.headers.get("content-type", "")
    base_type = content_type.split(";")[0].strip().lower()

    # Rewrite root-absolute paths in HTML/JS/CSS — skip vendor assets
    # (large, immutable, no OSPREY paths to rewrite).
    if base_type in _REWRITABLE_TYPES and "/vendor/" not in path:
        text = resp.text
        text = _rewrite_content(text, panel_id, outer_prefix)
        return Response(
            content=text,
            status_code=resp.status_code,
            headers=resp_headers,
            media_type=content_type,
        )

    # JSON is normally passed through untouched (rewriting arbitrary API
    # payloads would corrupt data). But some backends bootstrap their web UI
    # from a JSON config endpoint that carries root-absolute paths (e.g. an
    # ``api_url`` the SPA uses as its API base) — served through this proxy,
    # the UI would then call the *browser* origin and silently break. Panels
    # opt specific endpoints in via ``web.panels.<id>.rewrite_json_paths``
    # (a list of path suffixes); only those responses get the same
    # known-prefix literal rewrite HTML/JS/CSS receive.
    if base_type == "application/json":
        stripped = path.rstrip("/")
        if any(stripped.endswith(sfx) for sfx in _panel_json_rewrite_paths(request, panel_id)):
            return Response(
                content=_rewrite_content(resp.text, panel_id, outer_prefix),
                status_code=resp.status_code,
                headers=resp_headers,
                media_type=content_type,
            )

    # JSON, images, binary — pass through unchanged.
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=resp_headers,
        media_type=content_type if content_type else None,
    )


@router.api_route(
    "/panel/{panel_id}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
)
async def proxy_panel_root(panel_id: str, request: Request):
    """Proxy the root path of a companion panel (no trailing path segment)."""
    return await proxy_panel(panel_id, "", request)


@router.websocket("/panel/{panel_id}/{path:path}")
async def proxy_panel_ws(panel_id: str, path: str, websocket: WebSocket):
    """Forward a WebSocket connection to the companion panel server.

    The upstream handshake is built from scratch rather than by copying the
    browser's — so the operator's cookie and secret are absent by construction,
    matching what :func:`proxy_panel` strips on the HTTP path. The only header
    added is the operator secret, and only for a backend
    :func:`_terminal_secret_header` vouches for.
    """
    backend_url = _resolve_panel_url(websocket, panel_id)
    if not backend_url:
        await websocket.close(code=4004, reason=f"Panel '{panel_id}' not available")
        return

    # Convert http(s) backend URL to ws(s)
    ws_url = backend_url.rstrip("/").replace("http://", "ws://").replace("https://", "wss://")
    target = f"{ws_url}/{path}"

    try:
        import websockets
    except ImportError:
        logger.error("websockets package not installed — cannot proxy WebSocket")
        await websocket.close(code=4500, reason="WebSocket proxy unavailable")
        return

    # Nothing from the browser's handshake is forwarded; this is the whole
    # outbound header set, and it is empty unless the backend earns the secret.
    upstream_headers = _terminal_secret_header(websocket, panel_id, backend_url)

    await websocket.accept()

    # websockets' opening handshake follows 30x redirects (up to
    # WEBSOCKETS_MAX_REDIRECTS) and re-sends ``additional_headers`` verbatim to
    # each target — the ws↔ws analogue of the httpx redirect leak the HTTP path
    # guards against. When the secret is being carried, refuse to follow any
    # redirect so it cannot ride one off the loopback host the gate vouched for.
    connector = websockets.connect(target, additional_headers=upstream_headers or None)
    if upstream_headers:
        # ``process_redirect`` returns the new URI to follow a redirect, or the
        # exception to refuse it; returning the exception unchanged declines
        # every redirect and surfaces it instead of chasing it with the secret.
        connector.process_redirect = lambda exc: exc

    try:
        async with connector as upstream:

            async def client_to_upstream():
                # receive() (not receive_text()) — binary-protocol panels
                # (e.g. the noVNC/RFB phoebus stream) send bytes frames,
                # which receive_text() rejects, killing the relay.
                try:
                    while True:
                        msg = await websocket.receive()
                        if msg.get("type") == "websocket.disconnect":
                            break
                        data = msg.get("bytes")
                        if data is None:
                            data = msg.get("text")
                        if data is not None:
                            await upstream.send(data)
                except WebSocketDisconnect:
                    pass

            async def upstream_to_client():
                try:
                    async for msg in upstream:
                        if isinstance(msg, str):
                            await websocket.send_text(msg)
                        else:
                            await websocket.send_bytes(msg)
                except websockets.ConnectionClosed:
                    pass

            # Run both directions concurrently; when either side closes,
            # cancel the other.
            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(client_to_upstream()),
                    asyncio.create_task(upstream_to_client()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()

    except Exception as exc:
        logger.warning("WebSocket proxy error for panel %s: %s", panel_id, exc)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
