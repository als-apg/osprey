"""Reverse proxy for companion panel servers running inside the container.

Companion servers (artifact gallery, ARIEL, channel-finder, lattice, OKF) and
the notebook sidecar bind to 127.0.0.1 inside the Docker container.  The browser cannot reach them
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
deployment. The notebook sidecar also carries a credential of its own, minted
per launch and injected by :func:`_panel_auth_header` on the same gate; a scoped
credential in place of the operator's own is still the follow-up for the rest.
"""

from __future__ import annotations

import asyncio
import html
import ipaddress
import json
import logging
import os
import re
import secrets
from pathlib import Path
from urllib.parse import ParseResult, urljoin, urlparse, urlunsplit

import httpx
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse
from starlette.concurrency import run_in_threadpool
from starlette.websockets import WebSocketState
from websockets.asyncio.client import connect as _WebSocketConnect

from osprey.apps.manifest import is_write_method
from osprey.dispatch import DISPATCHER_MCP_PATH
from osprey.interfaces._media_types import STATIC_TEXT_TYPES as _STATIC_TEXT_TYPES
from osprey.interfaces._media_types import asset_media_type as _asset_media_type
from osprey.interfaces.common_middleware import (
    ASSET_CORS_HEADERS,
    EXTERNAL_ORIGIN_ENV,
    FACILITY_TIMEZONE_ATTRIBUTE,
    FIRST_PARTY_PANEL_STATIC,
    MOUNT_SEGMENT_RE,
    PANEL_ASSET_SUBPATHS,
    PANEL_CAP_SEGMENT,
    STORAGE_SCOPE_ATTRIBUTE,
    compute_url_prefix,
    resolve_storage_scope,
)
from osprey.interfaces.common_middleware import (
    _normalize_origin as _normalize_external_origin,
)
from osprey.interfaces.opaque_origin import document_headers
from osprey.interfaces.web_auth import get_web_credentials
from osprey.interfaces.web_terminal.panel_cap import opaque_panel_kind, verify_request_cap
from osprey.interfaces.web_terminal.routes.websocket import (
    _login_tier,
    _rendered_config,
    _section_of,
)
from osprey.profiles.web_panels import SIDECAR_PANELS
from osprey.registry.web import FRAMEWORK_WEB_SERVERS, panel_url_state_attr
from osprey.utils.http_proxy import HOP_BY_HOP
from osprey.utils.launch_token import LAUNCH_TOKEN_HEADER
from osprey.utils.owner_header import OWNER_HEADER, owner_from_header
from osprey_connectors.config import get_facility_timezone
from osprey_connectors.identity import acting_identity

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
#
# Sidecar panels have no registry key — the terminal starts them itself — so
# their state attribute is derived from the panel id directly.
_PANEL_STATE_MAP = {
    definition.panel_id: panel_url_state_attr(key)
    for key, definition in FRAMEWORK_WEB_SERVERS.items()
} | {panel_id: panel_url_state_attr(panel_id) for panel_id in SIDECAR_PANELS}


def _resolve_panel_url(scope: Request | WebSocket, panel_id: str) -> str | None:
    """Map a panel ID to its internal server URL, or ``None`` if unavailable.

    Args:
        scope: The request or websocket being proxied; only ``app.state`` is
            read, so both connection kinds answer the same way.
        panel_id: The panel id from the proxied URL.
    """
    attr = _PANEL_STATE_MAP.get(panel_id)
    if attr:
        return getattr(scope.app.state, attr, None)

    # Custom panels: look up by ID in the custom panels list.
    for cp in getattr(scope.app.state, "custom_panels", []):
        if cp.get("id") == panel_id:
            url = cp.get("url", "")
            return url or None

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


def configured_agent_routes(custom_panels: list[dict]) -> frozenset[tuple[str, str]]:
    """Compose the exact ``(method, /panel/<id><path>)`` pairs the panels declare.

    Only a config-declared panel addressed at loopback contributes: a runtime
    registration or an off-box backend declares nothing, whatever its dict holds.

    Args:
        custom_panels: The custom-panel dicts :func:`_load_panel_config` forwards,
            each with ``agentRoutes`` as ``{method, path}`` entries whose path is
            backend-root-relative.
    """
    pairs: set[tuple[str, str]] = set()
    for panel in custom_panels:
        if not panel.get("configDefined") or not _backend_is_loopback(panel.get("url", "")):
            continue
        for route in panel.get("agentRoutes") or []:
            pairs.add((route["method"].upper(), f"/panel/{panel['id']}{route['path']}"))
    return frozenset(pairs)


def _login_arms_writes(config_path: Path | None) -> bool:
    """Whether the acting login's ``control_system.writes`` tier arms writes here.

    The same per-login tier the header chip shows (``_login_tier``): the
    deployment's tier, then the login's own rung. ``asked`` and ``allowed``
    arm; ``never`` and ``unknown`` do not. The owner's per-machine switch is
    not read here: the write itself meets it downstream, where the position
    decides whether a person is asked. Blocking (it reads the render); call it
    off the loop.
    """
    section = _section_of(_rendered_config(config_path))
    return _login_tier(section, acting_identity()) in ("asked", "allowed")


async def _unarmed_application(scope: Request | WebSocket, panel_id: str) -> str | None:
    """The application ``panel_id`` serves, when this login does not arm it.

    ``None`` for a panel that names no application — a built-in panel, a
    runtime registration, a config-defined panel of the non-application shape —
    and for an application panel on a login whose write tier arms writes.
    A disabled panel block never reaches ``custom_panels``, so its application
    is never armed here.
    """
    for panel in getattr(scope.app.state, "custom_panels", []):
        if panel.get("id") == panel_id:
            application = panel.get("app")
            if not application:
                return None
            config_path = getattr(scope.app.state, "config_path", None)
            if await run_in_threadpool(_login_arms_writes, config_path):
                return None
            return str(application)
    return None


def _unarmed_refusal(scope: Request | WebSocket, application: str, what: str) -> str:
    """The refusal naming the login and the application."""
    login = str(getattr(scope.app.state, "terminal_user", "") or "").strip()
    who = f"login '{login}'" if login else "this login"
    return f"{what} refused: {who} does not arm application '{application}' for writes"


class _RedirectRefusingConnect(_WebSocketConnect):
    """An upstream websocket handshake that follows no redirect.

    A handshake carrying the operator secret or a panel credential follows no
    redirect: ``websockets`` re-sends ``additional_headers`` verbatim to every
    redirect target, so following one could carry the credential off the host
    the gate vouched for. The redirect surfaces as the exception it raised.
    """

    def process_redirect(self, exc: Exception) -> Exception | str:
        """Decline every redirect by returning the exception unchanged."""
        return exc


#: The panel id the event dispatcher is published under.
#:
#: Two things ride on this id and on nothing else about the request: the
#: dispatcher bearer this proxy injects, and the hop to the dispatcher's MCP
#: transport below. Spelled once so the two cannot come to disagree about which
#: panel they mean.
_EVENTS_PANEL_ID = "events"


def _is_dispatcher_mcp_hop(panel_id: str, path: str) -> bool:
    """Whether this request is the hop to the event dispatcher's MCP transport.

    That hop carries a JSON-RPC conversation over streamable HTTP rather than a
    page: the body is the message, so nothing on it may be rewritten, and the
    upstream's own content-type is what tells the client whether it was handed
    one reply or a stream.

    The path is matched against :data:`~osprey.dispatch.DISPATCHER_MCP_PATH`
    rather than a literal, and the same constant is rendered into the
    dispatcher's compose as ``FASTMCP_STREAMABLE_HTTP_PATH``. Two further
    spellings of it are literals — the panel-tier route the terminal's own gate
    admits, and the entry URL an agent's client holds — and tests hold both to
    this constant, because a spelling that drifted from it would 404 the agent
    at one end or cost it the no-rewrite guarantee at the other.

    Args:
        panel_id: The panel id from the proxied URL.
        path: The path segment below the panel id, without a leading slash.
    """
    return panel_id == _EVENTS_PANEL_ID and f"/{path}" == DISPATCHER_MCP_PATH


#: Request headers that are never forwarded to a panel backend.
#:
#: Three of them carry the operator's identity at the terminal: the browser
#: session cookie, an ``Authorization`` header, and the
#: ``X-Osprey-Terminal-Secret`` the multi-user reverse proxy stamps on. The
#: browser attaches them to panel requests because a panel is served from the
#: terminal's own origin, not because the panel is entitled to them — and a
#: backend handed any of the three could replay it against the terminal as the
#: operator.
#:
#: ``X-Osprey-Owner`` is not a credential and is dropped for the opposite
#: reason. A backend that reads it files queued work under that name and gates
#: that work's writes by that user's chip, so a value chosen by the browser —
#: which sits inside a session the agent itself can drive — would let one
#: session name another user as the author of its work, and narrow or widen
#: what that work may do. Nothing inbound is evidence of who is driving:
#: :func:`_owner_header` mints that from this process's own identity instead.
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
        OWNER_HEADER.lower(),
        LAUNCH_TOKEN_HEADER.lower(),
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
TERMINAL_SECRET_HEADER = "X-Osprey-Terminal-Secret"  # noqa: S105 -- names a header, holds no secret


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


def _backend_earns_credentials(scope: Request | WebSocket, panel_id: str, backend_url: str) -> bool:
    """Whether a credential this process holds may be sent to this backend.

    Both halves are required. A backend addressed at loopback but registered at
    runtime is not enough: runtime registration is a path the agent's own
    sandbox can reach, and a loopback listener it registered would be handed a
    credential out of this process. A config-declared backend pointing off-box
    is not enough either: the credential would leave the machine to a host the
    operator's browser never authenticated to, over a hop that may not even be
    TLS.

    Every injection site — the operator secret, the event dispatcher's bearer
    and a sidecar's own launch credential — asks this one question, so the two
    halves cannot drift apart between them.

    Args:
        scope: The request or websocket being proxied.
        panel_id: The panel id from the proxied URL.
        backend_url: The resolved backend URL the request is being forwarded to.
    """
    return _panel_origin_is_declared(scope, panel_id) and _backend_is_loopback(backend_url)


def _terminal_secret_header(
    scope: Request | WebSocket, panel_id: str, backend_url: str
) -> dict[str, str]:
    """The terminal-secret header to send upstream, or ``{}`` for no injection.

    When the backend does not earn it (:func:`_backend_earns_credentials`) — or
    the process cannot produce credentials at all — nothing is injected and the
    panel simply loads unauthenticated. That is the failure an operator can see
    and act on; the alternative failure, a leaked secret, is silent.

    Args:
        scope: The request or websocket being proxied.
        panel_id: The panel id from the proxied URL.
        backend_url: The resolved backend URL the request is being forwarded to.

    Returns:
        A single-entry mapping ready to merge into the outbound headers, or an
        empty mapping when this backend must not receive the secret.
    """
    if not _backend_earns_credentials(scope, panel_id, backend_url):
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


def _panel_auth_header(
    scope: Request | WebSocket, panel_id: str, backend_url: str
) -> dict[str, str]:
    """The panel's own launch credential to send upstream, or ``{}``.

    A sidecar this process starts is handed a credential of its own and refuses
    anything that arrives without it. The launcher publishes that credential as
    a ready-made header mapping on ``app.state.panel_auth_headers``, keyed by
    panel id, and it is injected on this hop so the browser never holds it. An
    application panel's launch token is held the same way.

    Gated exactly as the operator secret is
    (:func:`_backend_earns_credentials`): a credential handed to a listener the
    agent's own sandbox registered, or sent to a host the operator never
    authenticated to, fails silently, so an absent header — the panel loads
    unauthenticated and says so — is the one to prefer.

    Args:
        scope: The request or websocket being proxied.
        panel_id: The panel id from the proxied URL.
        backend_url: The resolved backend URL the request is being forwarded to.

    Returns:
        The headers to merge into the outbound set, or an empty mapping when
        this backend must not receive them.
    """
    if not _backend_earns_credentials(scope, panel_id, backend_url):
        return {}
    published = getattr(scope.app.state, "panel_auth_headers", None) or {}
    return dict(published.get(panel_id, {}))


def _owner_header(scope: Request | WebSocket, panel_id: str, backend_url: str) -> dict[str, str]:
    """Who this process says is driving, or ``{}`` for a backend not told.

    Minted from :func:`~osprey_connectors.identity.acting_identity` — the account
    this container runs as — and never from anything that arrived. The value
    decides whose chip gates the writes a queued plan makes, so an inbound
    header is a claim rather than evidence: the strip above and this mint are
    the two halves of one boundary, and a request leaves this hop carrying the
    name this process vouches for whatever it arrived with.

    The identity ladder answers ``unknown`` rather than nothing when no marker
    names a human, and that answer is forwarded like any other. It is the name
    this terminal's own control-context record is written under, so a lane
    looking up the chip that governs the plan has to be handed the same one.

    A name this end cannot spell as a header — outside the allowlist
    :func:`~osprey.utils.owner_header.owner_from_header` applies at the reading
    end — is sent as no header at all, and the work is recorded owner-less.
    That is the one failure this path may have: the header value goes out over
    a client that encodes it as ASCII and raises on anything else, so an
    account the guard would refuse anyway would otherwise fail every proxied
    request for that user rather than costing them their attribution.

    Gated by :func:`_backend_earns_credentials`, the question the credentials
    ask, though the reason differs — the name is an attribution, not a secret.
    A backend registered at runtime is a listener the agent's own sandbox can
    create, and an off-box backend is a host nobody at this console
    authenticated to; neither is owed the account name of the human driving.
    The backends that read the header are declared and on loopback, and they
    are the same backends this hop hands a credential to, so the attribution
    reaches exactly as far as the authority it describes.

    Args:
        scope: The request being proxied.
        panel_id: The panel id from the proxied URL.
        backend_url: The resolved backend URL the request is being forwarded to.

    Returns:
        A single-entry mapping ready to merge into the outbound headers, or an
        empty mapping when this backend is told nothing about who is driving.
    """
    if not _backend_earns_credentials(scope, panel_id, backend_url):
        return {}
    owner = owner_from_header(acting_identity())
    return {OWNER_HEADER: owner} if owner else {}


async def _request_no_redirect(
    client: httpx.AsyncClient,
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    content: bytes | None,
    timeout: httpx.Timeout | None = None,
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

    *timeout* is the panel's own bound (:func:`_panel_timeout`); ``None`` sends
    the request on the shared client's default.
    """
    if timeout is None:
        return await client.request(
            method=method,
            url=url,
            headers=headers,
            content=content,
            follow_redirects=False,
        )
    return await client.request(
        method=method,
        url=url,
        headers=headers,
        content=content,
        follow_redirects=False,
        timeout=timeout,
    )


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
    return not (parsed.scheme and other.scheme and parsed.scheme.lower() != other.scheme.lower())


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

    def _already_prefixed(path: str) -> bool:
        # A prefix-AWARE backend — one that honors the ``X-Forwarded-Prefix``
        # this proxy sends on every request — emits Locations that already
        # live inside the panel namespace. Prepending again would double the
        # prefix (``/u/a/panel/x/u/a/panel/x/...``) and 404 on the terminal,
        # so the re-base is idempotent: a path already carrying the prefix is
        # relayed as it stands. Segment-boundary comparison, so a panel id
        # that is a prefix of another (``x`` vs ``x-2``) is never conflated.
        return path == panel_prefix or path.startswith((panel_prefix + "/", panel_prefix + "?"))

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
            if _already_prefixed(remainder):
                return remainder
            return panel_prefix + remainder
        if _same_origin(parsed, terminal_origin):
            return location
        return None

    if location.startswith("/"):
        if _already_prefixed(location):
            return location
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


def _panel_timeout(request: Request, panel_id: str, shared: httpx.Timeout) -> httpx.Timeout | None:
    """The timeout of this panel's proxied requests, or ``None`` for the shared default.

    A config-defined panel that writes ``read_timeout_s`` replaces the read bound
    of the shared client's timeout and keeps its connect, write and pool bounds;
    ``null`` lifts the read bound. A panel that writes nothing, and every runtime
    registration (a timeout is configuration), stays on *shared*.

    Args:
        request: The request being proxied.
        panel_id: The panel id from the proxied URL.
        shared: The shared proxy client's timeout.
    """
    for cp in getattr(request.app.state, "custom_panels", []):
        if cp.get("id") == panel_id:
            if not cp.get("configDefined") or "readTimeoutS" not in cp:
                return None
            return httpx.Timeout(
                connect=shared.connect,
                read=cp["readTimeoutS"],
                write=shared.write,
                pool=shared.pool,
            )
    return None


def _path_rewrite_prefix(path: object) -> tuple[str, ...]:
    """The rewrite prefix a panel's configured ``path`` contributes, if any.

    A panel's document is served at ``/panel/<id><path>``, so a literal that
    begins with ``path`` has exactly one correct destination under the panel's
    namespace. Two rules keep the derived prefix to that:

    - a path of ``/`` is every literal, so deriving from it would rewrite every
      quote-delimited ``/…`` in the body — the default path contributes nothing;
    - a query string and a fragment are not part of a prefix any literal can
      begin with, so only the path component counts (a noVNC path such as
      ``/vnc.html?path=…&autoconnect=1`` derives ``/vnc.html``).

    Args:
        path: The panel's configured ``path``; anything but a string derives
            nothing.
    """
    if not isinstance(path, str):
        return ()
    component = urlparse(path).path
    if not component.startswith("/") or component == "/":
        return ()
    return (component,)


def _panel_rewrite_prefixes(request: Request, panel_id: str) -> tuple[str, ...]:
    """The prefixes this panel adds to ``_REWRITE_PREFIXES`` (usually none).

    Only a config-defined panel contributes: a runtime registration's ``path``
    is a free string its caller chooses, and a rewrite prefix is configuration.
    The panel's declared ``rewrite_prefixes`` follow the one derived from its
    ``path``, trailing slash kept, since ``/x`` and ``/x/`` are different prefixes.
    """
    for cp in getattr(request.app.state, "custom_panels", []):
        if cp.get("id") == panel_id:
            if not cp.get("configDefined"):
                return ()
            declared = cp.get("rewritePrefixes") or ()
            if isinstance(declared, (list, tuple)):
                declared = tuple(str(p) for p in declared if p)
            else:
                declared = (str(declared),)
            derived = _path_rewrite_prefix(cp.get("path"))
            return tuple(dict.fromkeys((*derived, *declared)))
    return ()


def _rewrite_content(
    body: str,
    panel_id: str,
    outer_prefix: str,
    extra_prefixes: tuple[str, ...] = (),
    *,
    panel_base: str | None = None,
) -> str:
    """Rewrite root-absolute paths inside string delimiters for proxied content.

    Only touches known prefixes inside ``"``, ``'``, or backtick delimiters.
    CDN URLs (``https://…``), protocol-relative (``//…``), and data URIs
    are unaffected because the pattern requires a delimiter immediately
    before the ``/``.

    Every literal is rebased at most once, in a single pass over the body. A
    rebased literal still starts with ``/`` when the outer prefix is empty
    (``/panel/<id>/...``), so a later prefix that matches the rebased text —
    a panel whose ``path`` derives ``/panel/`` — would otherwise rebase it a
    second time into ``/panel/<id>/panel/<id>/...``. Which of several matching
    prefixes wins does not matter: each one is a leading slice of the literal,
    and the rebase only puts the panel prefix in front of it.

    Args:
        body: The response body to rewrite.
        panel_id: The panel ID this content was proxied from.
        outer_prefix: The per-user mount prefix (``/u/<user>`` or ``""``, see
            ``compute_url_prefix()``). Prepended so a panel's internal
            assets/APIs resolve under the outer prefix too, not just
            ``/panel/<id>``. An empty prefix yields bare ``/panel/<id>/...``.
        extra_prefixes: The root-absolute prefixes this one panel adds to
            ``_REWRITE_PREFIXES``; none by default.
        panel_base: The full browser path to rewrite into instead of
            ``<outer_prefix>/panel/<panel_id>`` — a C-CAP path, so a capped
            panel's own absolute paths stay under its cap. The hub's own
            assets (:data:`PANEL_ASSET_SUBPATHS`) are the exception: they are
            not the panel's bytes and a cap path would relay them to its
            backend, so they keep pointing at their C-ASSETS route
            ``<outer_prefix>/panel/<panel_id>/<sub>/`` — already exempt and
            CORS-readable, so nothing new is exempted.
    """
    prefix = panel_base or f"{outer_prefix}/panel/{panel_id}"
    if panel_base:
        # First, so the broader "/static/" prefix below cannot claim
        # "/static/fonts/" for the cap.
        for sub in PANEL_ASSET_SUBPATHS:
            body = re.sub(
                r"""(?<=["'`])""" + re.escape(f"/{sub}/"),
                f"{outer_prefix}/panel/{panel_id}/{sub}/",
                body,
            )

    # One alternation, one pass: "/static/..." → "/panel/id/static/...". The
    # lookbehind matches only right after a quote character; bare '/api' (no
    # trailing slash) is matched only when the delimiter closes it.
    alternatives = [
        re.escape(path) for path in dict.fromkeys((*_REWRITE_PREFIXES, *extra_prefixes))
    ]
    alternatives.append(r"""/api(?=["'`])""")
    pattern = r"""(?<=["'`])(?:""" + "|".join(alternatives) + ")"
    body = re.sub(pattern, lambda match: prefix + match.group(0), body)

    # Rewrite href="/" to href="/panel/{id}/"
    body = body.replace('href="/"', f'href="{prefix}/"')
    return body.replace("href='/'", f"href='{prefix}/'")


#: The hub's own copy of the shared design system — the same directory
#: ``_app_setup.mount_shared_static()`` serves at ``/design-system``.
_DESIGN_SYSTEM_DIR = Path(__file__).resolve().parents[2] / "design_system" / "static"

#: The hub's shared fonts — the same directory ``_app_setup`` mounts at
#: ``/static/fonts`` — served to an embedded panel by
#: :func:`proxy_panel_shared_fonts`.
_SHARED_FONTS_DIR = Path(__file__).resolve().parents[2] / "shared_fonts"

#: The hub's own ``web_terminal/static`` tree, served to an embedded panel by
#: :func:`proxy_panel_terminal_static`.
_TERMINAL_STATIC_DIR = Path(__file__).resolve().parents[1] / "static"

#: The only suffixes :func:`proxy_panel_terminal_static` will serve.
#:
#: That route exists for one caller — the Lab bar's module graph and the
#: stylesheet it anchors against — so it serves what a module graph is made of
#: and refuses the rest of the tree. Without this, ``index.html`` and
#: ``session.html`` (raw Jinja sources) answer as ``text/html`` under a
#: ``/panel/`` path. That is not an exposure today: ``/static/index.html``
#: already serves the identical bytes to an unauthenticated caller, because
#: ``/static`` is one of the exempt mount prefixes. It is simply reach this
#: route has no use for, and a route that cannot serve a document cannot later
#: be talked into serving one.
#:
#: Deliberately NOT applied to the design-system route, which serves whole
#: pages (``theme-lab.html``) on purpose.
_TERMINAL_STATIC_SUFFIXES = frozenset({".js", ".mjs", ".css", ".map", ".svg", ".woff2", ".png"})


def _contained_asset(root: Path, asset_path: str) -> Path | None:
    """Resolve *asset_path* under *root*, or ``None`` when it is not a file there.

    The one containment implementation both panel-scoped static routes use.
    ``asset_path`` is attacker-controlled — it is whatever the browser put after
    the route prefix — so the resolved path is compared against the resolved
    root before anything is read. That refuses three shapes at once: a
    traversal (``../../secrets``, and its percent-encoded form, which ASGI has
    already decoded by the time it arrives), an absolute path (``/`` joined onto
    a ``Path`` discards the left-hand side entirely), and a symlink inside the
    root pointing outside it (``resolve()`` follows links, so the escape shows
    up in the comparison).

    ``is_file()`` is the second half: a directory answers nothing, so neither
    route can be talked into a listing, and a missing asset 404s instead of
    raising.

    The whole resolution runs inside one ``try``. ``resolve()`` raises rather
    than answering for a path the OS cannot even look at — a NUL byte
    (``js/api.js%00.txt``, which ASGI delivers decoded) is a ``ValueError`` from
    ``lstat`` before any containment check runs, and an over-long name or a
    symlink loop is an ``OSError``. Every one of those is the same answer this
    function already gives for a path that is not a file under the root, so
    they are answered the same way: ``None``, and a 404 from both callers.
    """
    try:
        candidate = (root / asset_path).resolve()
        if not candidate.is_relative_to(root.resolve()) or not candidate.is_file():
            return None
    except (ValueError, OSError):
        return None
    return candidate


@router.api_route(
    "/panel/{panel_id}/design-system/{asset_path:path}",
    methods=["GET", "HEAD"],
)
async def proxy_panel_design_system(panel_id: str, asset_path: str, _request: Request):
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

    A C-ASSETS path: exempt from the auth gate for ``GET``/``HEAD``
    (:func:`~osprey.interfaces.common_middleware.is_panel_asset_path`) and
    answered with ``Access-Control-Allow-Origin: *``, so an opaque agent
    document embedded under this panel can still import the theme modules.
    """
    if not _DESIGN_SYSTEM_DIR.is_dir():  # pragma: no cover - packaging guard
        return Response(content="design system unavailable", status_code=404)

    # The id is interpolated into the rewritten body below (and into a
    # ``re.sub`` replacement string), and this route answers anyone: it is a
    # C-ASSETS path, exempt from the gate. A segment outside the mount class —
    # a quote, a backslash — is refused before it can reach a served module.
    if MOUNT_SEGMENT_RE.fullmatch(panel_id) is None:
        return Response(content="Not found", status_code=404)

    candidate = _contained_asset(_DESIGN_SYSTEM_DIR, asset_path)
    if candidate is None:
        return Response(content="Not found", status_code=404)

    headers = {"cache-control": _DEFAULT_NO_CACHE, **ASSET_CORS_HEADERS}
    media_type = _asset_media_type(candidate)

    if candidate.suffix.lower() not in _STATIC_TEXT_TYPES:
        return Response(content=candidate.read_bytes(), headers=headers, media_type=media_type)

    # Text assets carry root-absolute self-references (e.g.
    # osprey-theme-switcher.js dynamically imports '/design-system/js/
    # theme-manager.js'). Apply the proxy's own rewrite so those resolve back
    # into this panel's namespace — and therefore back to the hub's copy.
    # The fixed prefixes only: this is the hub's tree, not the panel's upstream,
    # so nothing derived from the panel's configuration applies to it.
    text = _rewrite_content(candidate.read_text(encoding="utf-8"), panel_id, compute_url_prefix())
    return Response(content=text, headers=headers, media_type=media_type)


@router.api_route(
    "/panel/{panel_id}/terminal-static/{asset_path:path}",
    methods=["GET", "HEAD"],
)
async def proxy_panel_terminal_static(
    panel_id: str,  # noqa: ARG001 - route path parameter; terminal assets are served verbatim
    asset_path: str,
):
    """Serve the TERMINAL's own ``static/`` tree to an embedded panel.

    MUST stay declared above :func:`proxy_panel` for the same reason the
    design-system route does: that route's ``{path:path}`` is a catch-all and
    Starlette matches in declaration order.

    The hub's chip modules have to run on a panel page. The notebook panel is
    JupyterLab, served through this proxy under ``/panel/jupyter/``, so the
    browser's origin is the hub — but every path the Lab page emits is relative
    to JupyterLab's own ``base_url``, which is ``/panel/jupyter/``. A
    ``/static/js/...`` reference from that page therefore resolves into the
    *sidecar's* namespace and 404s, and :func:`_rewrite_content` rewrites the
    literal to ``/panel/jupyter/static/...`` for good measure. This route is the
    address that lands back on the hub's own files from inside a panel's path.

    Unlike the design-system route, bodies are served **verbatim**. These
    modules address the hub's API root-absolutely through ``withPrefix()`` in
    ``api.js``, which resolves ``/api/...`` against the deployment prefix at
    runtime; rewriting those literals into ``/panel/jupyter/api/...`` would aim
    the chip's reads at the sidecar instead of the terminal that answers them.
    Relative imports (``./api.js``) need no rewrite — they already resolve
    inside this route's own directory.

    The tier is operator, like every path under ``/panel/`` but the dispatcher
    MCP hop, and ``PANEL_TIER_ROUTES`` is untouched. Unlike its design-system
    neighbour this route is not a C-ASSETS path
    (:func:`~osprey.interfaces.common_middleware.is_panel_asset_path`),
    so it stays behind the auth gate. That is the right level — the page
    doing the importing is one the operator is already logged in to.

    Only :data:`_TERMINAL_STATIC_SUFFIXES` are served; the rest of the tree —
    the two Jinja page sources among it — answers 404 here.
    """
    if not _TERMINAL_STATIC_DIR.is_dir():  # pragma: no cover - packaging guard
        return Response(content="terminal static assets unavailable", status_code=404)

    candidate = _contained_asset(_TERMINAL_STATIC_DIR, asset_path)
    if candidate is None or candidate.suffix.lower() not in _TERMINAL_STATIC_SUFFIXES:
        return Response(content="Not found", status_code=404)

    return Response(
        content=candidate.read_bytes(),
        headers={"cache-control": _DEFAULT_NO_CACHE},
        media_type=_asset_media_type(candidate),
    )


#: The one panel whose page carries the hub's control-target bar.
#:
#: The notebook panel is a page an operator runs code from, so it is the one
#: panel where "which machine does this land on, and may it write?" has to be
#: answerable on the page itself once the page is on its own — popped out of
#: the hub, or opened at its own address. Every other panel's HTML is relayed
#: untouched.
#:
#: The tag goes in regardless of how the page is opened; the bar module mounts
#: only when its document is a top-level window, so a Lab page framed inside
#: the hub keeps the header chip as its one picker. That test is the page's to
#: make: a request does not say whether it came from a frame.
_LAB_BAR_PANEL_ID = "jupyter"

#: The bar's entry module, relative to the terminal-static route's root.
_LAB_BAR_MODULE = "js/control-target-lab-bar.js"

#: First path segments that are JupyterLab's own application documents — the
#: only pages the bar goes on.
#:
#: An allow-list, not an exclusion list, because a content type is not a page:
#: jupyter-server serves a user's own ``.html`` file as ``text/html``
#: (``files/handlers.py`` guesses by suffix), so ``files/report.html``, the
#: ``view/`` wrapper JupyterLab opens it in and an ``nbconvert/html/`` export
#: all look exactly like the Lab page to a content-type gate. The bar inside
#: one of those would be a second chip in a viewer frame, a second event
#: stream against the browser's connection cap, and the hub's stylesheet
#: cascading into a document somebody else wrote. Default-deny is what keeps a
#: page nobody has thought of yet out of that list.
_LAB_BAR_PAGES = frozenset({"lab", "tree", "notebooks", "doc", "consoles", "edit"})

#: The opening ``<head>`` tag, wherever the backend put its attributes.
#:
#: ``\b`` refuses ``<header>``. The FIRST match wins, which for JupyterLab is
#: the real head — its template opens ``<!doctype html><html lang="en"><head>``
#: — but would not be for a backend that put the string in a comment or a
#: script ahead of it. Worth re-reading if this ever serves a second sidecar.
_HEAD_OPEN_RE = re.compile(r"<head\b[^>]*>", re.IGNORECASE)


def _script_safe(value: object) -> str:
    """*value* as JSON that cannot end the ``<script>`` carrying it.

    ``json.dumps`` escapes quotes and backslashes but leaves ``<`` alone, and
    inside a script the HTML parser looks for ``</script`` before the
    JavaScript parser sees anything. ``\\u003c`` is a valid JSON escape, so
    both the prefix global and the import map parse identically to the
    unescaped form.

    The prefix is deployment configuration (``/u/<user>``), not operator input,
    so this is depth rather than a fix for a live hole — but every string
    interpolated into a script tag is escaped for that tag on principle, and
    both tags below are built from the same one.
    """
    return json.dumps(value).replace("<", "\\u003c")


def _control_target_bar_markup(
    panel_prefix: str, outer_prefix: str, nonce: str | None = None
) -> str:
    """The three tags that put the control-target bar on an embedded page.

    Args:
        panel_prefix: ``{outer_prefix}/panel/{panel_id}``, i.e. the sidecar's
            own ``base_url`` without the trailing slash.
        outer_prefix: the per-user mount prefix (``compute_url_prefix()``).
        nonce: the page's script nonce (:func:`_lab_page_csp`). Every tag
            carries it — an inline import map is governed by ``script-src``
            exactly like an inline script — so the bar runs under the policy
            and nothing the page did not get from the proxy does.

    Three tags, in the order they have to run:

    1. **The prefix global.** ``withPrefix()`` in ``api.js`` reads
       ``window.__OSPREY_PREFIX__`` and prepends it to every root-absolute
       ``/api/...`` the chip asks for. It is the OUTER prefix, never the panel
       path: the chip reads the *hub's* ``/api/terminal/posture``, and a prefix
       of ``/panel/jupyter`` would aim every read at JupyterLab, which answers
       none of them.
    2. **A scoped import map**, for the root-absolute module specifiers in the
       bar's import closure, both under ``/design-system/`` (``confirm-skip.js``
       imports ``storage-scope.js``, ``lab-theme-follow.js`` imports
       ``tokens.js``). The hub's own page retargets that prefix with a
       document-wide import map; here the mapping is confined to a
       ``scopes`` entry keyed on the bar's own directory, so nothing the panel
       itself imports can be redirected by it. With an empty prefix the mapping
       is the identity, which is what the hub's map is there too.
    3. **The module.** Its ``src`` is fully formed:
       ``/terminal-static/`` is deliberately absent from
       :data:`_REWRITE_PREFIXES`, so a root-absolute literal would never be
       rewritten into the panel's namespace for us.

    A module script is deferred by definition, so all three land before the
    page's own scripts run and none of them blocks the parse.
    """
    module_src = f"{panel_prefix}/terminal-static/{_LAB_BAR_MODULE}"
    import_map = {
        "scopes": {
            f"{panel_prefix}/terminal-static/js/": {
                "/design-system/": f"{outer_prefix}/design-system/",
            }
        }
    }
    attr = "" if nonce is None else f' nonce="{html.escape(nonce, quote=True)}"'
    return (
        f"<script{attr}>window.__OSPREY_PREFIX__ = {_script_safe(outer_prefix)};</script>"
        f'<script{attr} type="importmap">{_script_safe(import_map)}</script>'
        f'<script{attr} type="module" src="{html.escape(module_src, quote=True)}"></script>'
    )


def _is_lab_application_page(text: str, panel_id: str, path: str, base_type: str) -> bool:
    """The four conditions that make a response one of JupyterLab's own pages.

    The notebook panel (:data:`_LAB_BAR_PANEL_ID`), a path that is one of
    Lab's application routes (:func:`_is_lab_application_route`), an HTML
    body, and a document with a ``<head>``. One predicate for both decisions
    taken on that page — the bar goes in, and every ``<script>`` on it is
    trusted with the policy's nonce — so the two can never disagree about which
    documents they cover.
    """
    if panel_id != _LAB_BAR_PANEL_ID or base_type != "text/html":
        return False
    if not _is_lab_application_route(path):
        return False
    return _HEAD_OPEN_RE.search(text) is not None


#: The route shape jupyterlab_server renders Lab's ``index.html`` template on,
#: after ``lab`` or ``doc``: an optional named workspace, then an optional
#: ``tree/`` path (``jupyterlab_server.handlers.MASTER_URL_PATTERN``).
_LAB_APP_ROUTE_REST_RE = re.compile(r"(?:/workspaces/[A-Za-z0-9_-]+)?(?:/tree(?:/.*)?)?/?")


def _is_lab_application_route(path: str) -> bool:
    """Whether *path* is a route that answers with Lab's own template.

    ``lab`` and ``doc`` are matched by the full route shape, not the first
    segment: the same prefix also carries ``lab/extensions/`` (a file handler
    that serves any file type out of the extension directories) and
    ``lab/api/`` (settings, themes, workspaces), neither of which is the
    template. The other entries of :data:`_LAB_BAR_PAGES` are template-only
    pages and match by first segment.
    """
    head, sep, rest = path.lstrip("/").partition("/")
    if head not in _LAB_BAR_PAGES:
        return False
    if head in {"lab", "doc"}:
        return _LAB_APP_ROUTE_REST_RE.fullmatch(sep + rest) is not None
    return True


#: Where, under the notebook panel's own prefix, Lab loads script from.
#:
#: ``static/`` is the application bundle and its webpack chunks,
#: ``lab/extensions/`` the prebuilt (federated) extensions — plotly, the
#: widget manager — and ``terminal-static/`` the control-target bar. Each is a
#: tree the image ships; none is a store anything at run time writes into.
#: ``lab/extensions/`` holds to that only because the sidecar pins
#: ``labextensions_path`` to the image's install tree
#: (``jupyter_sidecar.JupyterSidecar._argv``): jupyter-server's default also
#: searches ``$JUPYTER_DATA_DIR``, which is on the agent-writable shared root.
#: ``files/`` — the user's and the agent's notebooks and HTML — is the tree
#: that must never be in this list, which is why the policy names paths and
#: not the origin.
_LAB_SCRIPT_TREES = ("static/", "lab/extensions/", "terminal-static/")

#: Every ``<script`` opening tag, attributes and all.
_SCRIPT_OPEN_RE = re.compile(r"<script\b[^>]*>", re.IGNORECASE)


def _lab_page_csp(nonce: str, origin: str, outer_prefix: str) -> str:
    """The ``Content-Security-Policy`` for one JupyterLab application page.

    Kernel output renders inside this document, on the console's origin, with
    the operator's cookie; output written by an agent must not be able to run
    there. So ``script-src`` admits exactly two things: the parser-inserted
    tags the proxy relayed or injected, by **per-response nonce**, and Lab's
    static trees plus the hub's design system, by **fully qualified path**
    (``<origin><prefix>/panel/jupyter/static/`` …). The origin itself is not a
    source: it also serves ``/panel/jupyter/files/`` and the artifact store,
    both agent-writable, so ``'self'`` would admit a script an agent saved.
    No ``'unsafe-inline'`` (a ``<script>`` in an HTML output) and no
    ``'strict-dynamic'`` (it would discard the path list in favour of whatever
    a nonced script chose to load).

    ``'unsafe-eval'`` is in, as the q-0001 fallback: JupyterLab's own settings
    registry validates every plugin's settings with Ajv, which compiles each
    schema with ``new Function`` — without it the settings plugin fails to
    activate and Lab comes up unthemed and unconfigured (JupyterLab 4.6). It
    lets a script that is already running evaluate a string; it admits no new
    script. The Lab renderers that evaluate *output* —
    ``@jupyterlab/javascript-extension`` (``application/javascript``) and
    ``@jupyterlab/vega5-extension`` (Vega compiles spec expressions with
    ``new Function``) — are disabled in
    ``jupyter_sidecar.LAB_DISABLED_EXTENSIONS``, so kernel output has no path
    into it.

    ``object-src 'none'`` and ``base-uri 'none'`` close the two classic
    script-src bypasses; ``frame-ancestors 'self'`` keeps the frame rule
    jupyter-server's own policy carried (the hub frames this page from the same
    origin); ``form-action 'none'`` because a form post is the one
    page-initiated write a script policy does not stop, and Lab posts none.

    Args:
        nonce: this response's nonce, used once.
        origin: the browser-facing origin, as :func:`_browser_origin` resolves it.
        outer_prefix: the per-user mount prefix (``compute_url_prefix()``).
    """
    panel_root = f"{origin}{outer_prefix}/panel/{_LAB_BAR_PANEL_ID}/"
    sources = [f"'nonce-{nonce}'", "'unsafe-eval'"]
    sources += [f"{panel_root}{tree}" for tree in _LAB_SCRIPT_TREES]
    sources.append(f"{origin}{outer_prefix}/design-system/")
    return "; ".join(
        (
            f"script-src {' '.join(sources)}",
            "object-src 'none'",
            "base-uri 'none'",
            "frame-ancestors 'self'",
            "form-action 'none'",
        )
    )


def _browser_origin(request: Request) -> str:
    """The origin a browser reaches this app at, resolved as the Origin check resolves it.

    :data:`EXTERNAL_ORIGIN_ENV` wins where it is set — behind nginx the
    request's own ``Host`` is a container name no browser has seen — and
    otherwise the request's scheme and ``Host``. The same order as
    ``WebAuthMiddleware._resolve_external_origin``, so a page whose script
    sources name one origin is never accepted under another; normalized the
    way that check compares (lower case, default port dropped).
    """
    configured = os.environ.get(EXTERNAL_ORIGIN_ENV, "").strip()
    if configured:
        return _normalize_external_origin(configured).rstrip("/")
    host = request.headers.get("host", "").strip() or request.url.netloc
    return _normalize_external_origin(f"{request.url.scheme}://{host}")


def _nonce_scripts(text: str, nonce: str) -> str:
    """Put *nonce* on every ``<script>`` tag of a JupyterLab application page.

    Called only on a page :func:`_is_lab_application_page` admits, and before
    the bar goes in. Such a page is jupyter-server's rendering of Lab's own
    ``index.html`` template: its tags are the config data block, the deferred
    main bundle and the inline token-stripper, and every value the template
    interpolates goes through ``tojson`` or ``escape``, so no ``<script`` in it
    was written by a notebook, a kernel or an agent. Output reaches this
    document later, through the DOM, where the parser-inserted nonce does not
    follow it.

    Every tag rather than a list of the two that execute, so a Lab upgrade
    that adds or reorders a script degrades to "still runs" rather than to a
    blank page — the template is the thing being trusted either way.
    """
    attr = f' nonce="{html.escape(nonce, quote=True)}"'
    return _SCRIPT_OPEN_RE.sub(lambda m: m.group(0)[:7] + attr + m.group(0)[7:], text)


def _inject_control_target_bar(
    text: str,
    panel_id: str,
    path: str,
    base_type: str,
    outer_prefix: str,
    nonce: str | None = None,
) -> str:
    """Put the control-target bar into the notebook panel's HTML, and nothing else's.

    Called after :func:`_rewrite_content`, so what goes in is what comes out —
    a rewrite pass over this markup would send the bar's own URLs into the
    panel's namespace.

    Four conditions, all narrow on purpose: the notebook panel
    (:data:`_LAB_BAR_PANEL_ID`), one of JupyterLab's own application pages
    (:data:`_LAB_BAR_PAGES`), an HTML body, and a document with a ``<head>`` to
    inject into. Anything else is relayed byte-for-byte — a JSON API answer, a
    stylesheet, another panel's page, an HTML fragment that is not a document,
    and above all **a user's own ``.html`` file**, which the sidecar serves as
    ``text/html`` from ``files/`` exactly as it serves the Lab page.

    The insert is at the top of ``<head>``, which is the earliest point the
    prefix global can be set, and therefore the only point that is certainly
    before the module graph it configures. It costs the page a few hundred
    bytes ahead of its own ``<meta charset>``, which stays far inside the
    1024 bytes the encoding sniff is allowed to read.

    **No header is set here.** The page's ``Content-Security-Policy`` is
    :func:`_lab_page_csp`, set by :func:`proxy_panel` on the same predicate
    (:func:`_is_lab_application_page`); *nonce* is that policy's, and every
    injected tag carries it. Without one the tags are emitted bare, which only
    a caller that sets no script policy may do.
    """
    if not _is_lab_application_page(text, panel_id, path, base_type):
        return text

    head = _HEAD_OPEN_RE.search(text)
    if head is None:  # unreachable: the predicate found it
        return text

    panel_prefix = f"{outer_prefix}/panel/{panel_id}"
    markup = _control_target_bar_markup(panel_prefix, outer_prefix, nonce)
    return f"{text[: head.end()]}{markup}{text[head.end() :]}"


#: The opening of the document's root tag.
#:
#: The lookahead refuses a longer tag name such as ``<htmlx>`` or
#: ``<html-embed>``. The FIRST match wins; as with :data:`_HEAD_OPEN_RE`, a
#: backend that put the string in a comment ahead of the real tag would be
#: stamped there instead.
_HTML_OPEN_RE = re.compile(r"<html(?=[\s/>])", re.IGNORECASE)


def _stamp_storage_scope(text: str, base_type: str, scope: str) -> str:
    """Stamp the mount's storage scope on a relayed document's ``<html>`` tag.

    Why the proxy: every panel page reaches the browser through this hop, and
    only the hub knows whose mount it is. A companion server may be shared by
    the whole roster, so it cannot name the person it renders for.

    Why first: an HTML parser keeps the first of two same-named attributes, so
    a backend that spelled its own scope cannot outrank the hub's.

    Why an empty scope relays byte for byte: single-user serving stays exactly
    as it is, so the absent attribute keeps meaning "use the bare key"
    (``storage-scope.js``).

    What the stamp is not: it separates preferences, not secrets. Any script
    on the origin can read every key.

    Args:
        text: The relayed body, already rewritten for this deployment.
        base_type: The response's media type without parameters.
        scope: The mount's storage scope, ``""`` when there is no mount user.

    Returns:
        ``text`` with the attribute as the root tag's first attribute, or
        ``text`` unchanged when there is no scope, the body is not HTML, or it
        has no root tag.
    """
    if not scope or base_type != "text/html":
        return text
    match = _HTML_OPEN_RE.search(text)
    if match is None:
        return text
    stamp = f' {STORAGE_SCOPE_ATTRIBUTE}="{html.escape(scope, quote=True)}"'
    return f"{text[: match.end()]}{stamp}{text[match.end() :]}"


def _stamp_facility_timezone(text: str, base_type: str, zone: str) -> str:
    """Stamp the facility time zone on a relayed document's ``<html>`` tag.

    Why the proxy: a panel's own server may not know the zone. The Bluesky
    sidecar reads no configuration, and some companions serve a static file.
    Every panel page reaches the browser through this hop, and the hub resolves
    the zone the agent is told.

    Why a document's own stamp stands: the zone is one deployment's
    configuration, so a companion that stamps its own names the same zone, and
    the root tag carries the attribute once.

    Why it is always present: the resolver degrades to ``UTC`` rather than
    failing, so every relayed page names a zone.

    Args:
        text: The relayed body, already rewritten for this deployment.
        base_type: The response's media type without parameters.
        zone: The facility's IANA zone name.

    Returns:
        ``text`` with the attribute as the root tag's first attribute, or
        ``text`` unchanged when there is no zone, the body is not HTML, it has
        no root tag, or the root tag already names a zone.
    """
    if not zone or base_type != "text/html":
        return text
    match = _HTML_OPEN_RE.search(text)
    if match is None:
        return text
    tag_end = text.find(">", match.end())
    tag = text[match.end() :] if tag_end == -1 else text[match.end() : tag_end]
    if f"{FACILITY_TIMEZONE_ATTRIBUTE}=" in tag.lower():
        return text
    stamp = f' {FACILITY_TIMEZONE_ATTRIBUTE}="{html.escape(zone, quote=True)}"'
    return f"{text[: match.end()]}{stamp}{text[match.end() :]}"


@router.api_route(
    "/panel/{panel_id}/static/fonts/{asset_path:path}",
    methods=["GET", "HEAD"],
)
async def proxy_panel_shared_fonts(
    panel_id: str,  # noqa: ARG001 - route path parameter; fonts are served verbatim
    asset_path: str,
):
    """Serve the HUB's shared fonts to an embedded panel.

    MUST stay declared above :func:`proxy_panel` — that route's
    ``{path:path}`` is a catch-all and Starlette matches in declaration
    order, so moving this below it makes it unreachable.

    Every panel links ``/static/fonts/fonts.css`` root-absolute, which
    :func:`_rewrite_content` turns into ``/panel/<id>/static/fonts/fonts.css``.
    Left to the generic proxy that request reaches the panel's own backend,
    and a URL-backed panel does not ship OSPREY's fonts, so the link would
    404 and the panel would render in system fonts. Answered here, every
    embedded panel gets the hub's typeface.

    ``fonts.css`` names its font files relatively, so it is served verbatim.
    A missing font 404s here and never reaches the backend.
    """
    if not _SHARED_FONTS_DIR.is_dir():  # pragma: no cover - packaging guard
        return Response(content="shared fonts unavailable", status_code=404)

    candidate = _contained_asset(_SHARED_FONTS_DIR, asset_path)
    if candidate is None:
        return Response(content="Not found", status_code=404)

    return Response(
        content=candidate.read_bytes(),
        headers={"cache-control": _DEFAULT_NO_CACHE, **ASSET_CORS_HEADERS},
        media_type=_asset_media_type(candidate),
    )


#: Where each first-party panel's ``/static`` tree lives in this package — the
#: directory that panel's own app mounts at ``/static``. The hub and these apps
#: ship in one package, so the hub can answer the panel's module tree itself.
#: Every panel id in
#: :data:`~osprey.interfaces.common_middleware.FIRST_PARTY_PANEL_STATIC` must
#: have an entry here.
_FIRST_PARTY_STATIC_ROOTS = {
    "artifacts": Path(__file__).resolve().parents[2] / "artifacts" / "static",
}

#: Cache-Control for a versioned vendor bundle, the policy the panel's own app
#: applies to ``/static/…/vendor/`` (``NoCacheStaticMiddleware``).
_VENDOR_CACHE = "public, max-age=31536000, immutable"


def _first_party_static_route(panel_id: str, subpath: str, root: Path):
    """Build the handler serving ``/panel/<panel_id>/<subpath>/…`` from *root*.

    *root* is the directory the panel's own app serves at ``/<subpath>``. The
    body is what the proxy's catch-all used to relay from that app: text
    types get the same root-absolute rewrite into this panel's namespace, so
    ``import '/static/js/state.js'`` still lands back on this route.
    """

    async def serve(asset_path: str) -> Response:
        if not root.is_dir():  # pragma: no cover - packaging guard
            return Response(content="panel static assets unavailable", status_code=404)

        candidate = _contained_asset(root, asset_path)
        if candidate is None:
            return Response(content="Not found", status_code=404)

        vendored = "vendor" in Path(asset_path).parts
        headers = {
            "cache-control": _VENDOR_CACHE if vendored else _DEFAULT_NO_CACHE,
            **ASSET_CORS_HEADERS,
        }
        media_type = _asset_media_type(candidate)
        if media_type not in _REWRITABLE_TYPES:
            return Response(content=candidate.read_bytes(), headers=headers, media_type=media_type)
        text = _rewrite_content(
            candidate.read_text(encoding="utf-8"), panel_id, compute_url_prefix()
        )
        return Response(content=text, headers=headers, media_type=media_type)

    serve.__name__ = f"proxy_panel_first_party_{panel_id}_{subpath}".replace("/", "_").replace(
        "-", "_"
    )
    serve.__doc__ = (
        f"Serve the {panel_id} panel's own ``/{subpath}`` tree from the hub's package.\n\n"
        "A C-ASSETS path: declared above ``proxy_panel``, exempt from the auth gate for "
        "GET/HEAD, answered with ``Access-Control-Allow-Origin: *`` and no credentials."
    )
    return serve


# The static module trees of config-defined first-party panels (C-ASSETS),
# one literal route per panel and sub-path. Literal, never ``/panel/{id}/…``:
# a runtime or discovered panel's ``/panel/<id>/static/…`` must still reach the
# catch-all below (and stay gated). Registered HERE, above ``proxy_panel``,
# because Starlette matches in declaration order.
for _panel_id, _subpaths in FIRST_PARTY_PANEL_STATIC.items():
    for _subpath in _subpaths:
        _root = _FIRST_PARTY_STATIC_ROOTS[_panel_id] / _subpath.removeprefix("static/")
        router.add_api_route(
            f"/panel/{_panel_id}/{_subpath}/{{asset_path:path}}",
            _first_party_static_route(_panel_id, _subpath, _root),
            methods=["GET", "HEAD"],
        )
del _panel_id, _subpaths, _subpath, _root


_PROXY_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]


def _cap_base(panel_id: str, cap: str) -> str:
    """The browser path a C-CAP request for ``panel_id`` is served under."""
    return f"{compute_url_prefix()}/panel/{panel_id}/{PANEL_CAP_SEGMENT}/{cap}"


def _mark_opaque_response(response: Response) -> Response:
    """Give a response carrying an opaque panel's bytes its sandbox headers.

    Every non-inert type carries ``opaque_headers()``: the sandbox CSP is
    *appended* beside any policy the upstream wrote, since a browser enforces
    every CSP it receives. Inert types get ``nosniff``.
    """
    for name, value in document_headers(response.headers.get("content-type")).items():
        if name.lower() == "content-security-policy":
            response.headers.append(name, value)
        else:
            response.headers[name] = value
    return response


def _mark_cap_response(response: Response) -> Response:
    """Give a response served under a cap path its C-CAP headers.

    ``Access-Control-Allow-Origin: *`` and never credentials (the upstream's own
    CORS headers were already dropped by :func:`_filter_response_headers`), so
    the panel's opaque document can read its own files, plus the sandbox
    headers of :func:`_mark_opaque_response`.
    """
    response.headers.update(ASSET_CORS_HEADERS)
    return _mark_opaque_response(response)


@router.api_route(
    f"/panel/{{panel_id}}/{PANEL_CAP_SEGMENT}/{{cap}}/{{path:path}}", methods=_PROXY_METHODS
)
async def proxy_panel_cap(panel_id: str, cap: str, path: str, request: Request):
    """Relay a request under a runtime panel's C-CAP path to that panel only.

    Declared above :func:`proxy_panel`, which would otherwise relay
    ``~c/<cap>/…`` to the backend as a path. The auth gate exempts this path
    (:func:`~osprey.interfaces.common_middleware.is_panel_cap_path`), so the cap
    is validated here instead of the cookie: forged, foreign-login, expired or
    other-panel → ``403``. Only a runtime-registered panel is ever capped, and a
    runtime backend earns no credential, so nothing but the request itself goes
    upstream. The root-path rewrite targets the cap prefix, so the panel's own
    absolute paths keep resolving under its cap.
    """
    if not verify_request_cap(request, panel_id, cap, "proxy"):
        return Response(content="Forbidden", status_code=403, media_type="text/plain")
    base = _cap_base(panel_id, cap)
    response = await _forward_to_panel(panel_id, path, request, panel_base=base)
    return _mark_cap_response(response)


@router.api_route("/panel/{panel_id}/{path:path}", methods=_PROXY_METHODS)
async def proxy_panel(panel_id: str, path: str, request: Request):
    """Forward a request to the companion panel server.

    Inbound operator credentials are stripped and, for a config-declared
    loopback backend only, the operator secret is re-issued — see this
    module's docstring for why the proxy is the boundary where that happens.

    A runtime-registered panel's bytes are agent-writable whoever serves them,
    so on this cookie-gated path too its documents carry ``opaque_headers()``
    (decided by where the panel came from, never by a flag it declares);
    config-defined and first-party panels keep their headers.
    """
    response = await _forward_to_panel(panel_id, path, request)
    if opaque_panel_kind(request, panel_id) == "proxy":
        return _mark_opaque_response(response)
    return response


async def _forward_to_panel(
    panel_id: str, path: str, request: Request, *, panel_base: str | None = None
) -> Response:
    """Relay one request to ``panel_id``'s backend and shape its answer.

    Args:
        panel_id: The panel id from the proxied URL.
        path: The path under the panel's namespace.
        request: The inbound request.
        panel_base: The browser path the panel is served under, when it is not
            ``<prefix>/panel/<id>`` — a C-CAP path. Redirects, the forwarded
            prefix and the root-path rewrite all target it.
    """
    backend_url = _resolve_panel_url(request, panel_id)
    if not backend_url:
        return Response(
            content=f"Panel '{panel_id}' is not available",
            status_code=404,
        )

    # An application panel on a login that does not arm it is read-only here:
    # a write method is refused before anything leaves the terminal.
    method = request.method.upper()
    unarmed_application = await _unarmed_application(request, panel_id)
    if unarmed_application is not None and is_write_method(method):
        return Response(
            content=_unarmed_refusal(request, unarmed_application, method),
            status_code=403,
        )

    outer_prefix = compute_url_prefix()
    panel_prefix = panel_base or f"{outer_prefix}/panel/{panel_id}"

    # Build the target URL. The dispatcher's MCP transport is addressed from
    # DISPATCHER_MCP_PATH, not from the inbound spelling: this hop and the
    # dispatcher's compose environment read that constant, while the route the
    # terminal's gate admits and the entry URL an agent's client holds spell the
    # same path literally and are held to the constant by tests. A hop that
    # rebuilt the path from the request would answer to none of them.
    dispatcher_mcp = _is_dispatcher_mcp_hop(panel_id, path)
    if dispatcher_mcp:
        target = f"{backend_url.rstrip('/')}{DISPATCHER_MCP_PATH}"
    else:
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
    fwd_headers["x-forwarded-prefix"] = panel_prefix

    # Re-issue the operator secret, but only toward a backend that is both
    # config-declared and on loopback. This runs after the strip above, so a
    # header the browser sent can never masquerade as an injected one.
    secret_header = _terminal_secret_header(request, panel_id, backend_url)
    fwd_headers.update(secret_header)

    # The event-dispatcher dashboard endpoints and its MCP transport are both
    # bearer-gated. Inject the dispatcher token server-side for the EVENTS panel
    # only, so the browser never holds it (and other panels are unaffected). The
    # token is read from the credential holder, which took it out of this
    # process's environment at population — so no child this process spawns
    # inherits it, and this process is the single point that can open either.
    #
    # Gated on the one question every injection site in this function asks —
    # :func:`_backend_earns_credentials` — rather than on the id string. A
    # declared origin alone keeps a runtime registration from squatting the id,
    # but it would still send the deployment's dispatcher token over the network
    # to whatever host an off-box `events` panel names; a shared bearer token
    # leaving the machine is the same failure as the operator secret leaving it.
    if panel_id == _EVENTS_PANEL_ID and _backend_earns_credentials(request, panel_id, backend_url):
        token = get_web_credentials(request.app).dispatcher_token
        if token:
            fwd_headers["authorization"] = f"Bearer {token}"

    # A launched sidecar's own credential or an application's launch token, on
    # the same gate as the secret above. A safe method still reaches an
    # application this login does not arm; its launch token does not go with it.
    panel_auth = _panel_auth_header(request, panel_id, backend_url)
    if panel_auth and unarmed_application is None:
        fwd_headers.update(panel_auth)

    # Who is driving, for the backends that file work under it and gate that
    # work by the driver's own chip. Minted in this same run, after the strip,
    # so an inbound claim can neither survive nor sit beside the minted value.
    fwd_headers.update(_owner_header(request, panel_id, backend_url))

    outer_panel_prefix = panel_prefix

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
                    content=body or None,
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
            # This branch is chosen by what the *request* was willing to accept,
            # so what comes back is not always a stream: an MCP call advertises
            # both shapes and is answered with a single ``application/json``
            # reply as often as with a stream. The upstream's own content-type
            # therefore stands, and ``text/event-stream`` is the answer only for
            # an upstream that named no type — which is what a plain SSE
            # endpoint is.
            return StreamingResponse(
                _stream(),
                status_code=upstream.status_code,
                headers=resp_headers,
                media_type=upstream.headers.get("content-type") or "text/event-stream",
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
            content=body or None,
            timeout=_panel_timeout(request, panel_id, client.timeout),
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

    # The MCP hop's answer is a JSON-RPC envelope, which is parsed rather than
    # rendered: a path rewritten inside it is a value the client reads back
    # wrong, not a link that resolves elsewhere. It is relayed byte for byte,
    # before either rewrite below can consider it — including the opt-in JSON
    # rewrite, which a panel's own configuration could otherwise aim at this
    # path. Nothing on this hop is addressed by a browser, so there is no
    # root-absolute path here that would need moving under the panel prefix.
    if dispatcher_mcp:
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=resp_headers,
            media_type=content_type or None,
        )

    # Rewrite root-absolute paths in HTML/JS/CSS — skip vendor assets
    # (large, immutable, no OSPREY paths to rewrite).
    if base_type in _REWRITABLE_TYPES and "/vendor/" not in path:
        text = resp.text
        text = _rewrite_content(
            text,
            panel_id,
            outer_prefix,
            _panel_rewrite_prefixes(request, panel_id),
            panel_base=panel_base,
        )
        # A JupyterLab application page runs under the proxy's script policy
        # (_lab_page_csp): Lab's own tags and the bar's carry this response's
        # nonce, and the upstream policy is replaced rather than relayed. Every
        # other response, notably files/ with jupyter-server's sandbox, keeps
        # the policy its upstream wrote.
        lab_page_nonce = None
        if _is_lab_application_page(text, panel_id, path, base_type):
            lab_page_nonce = secrets.token_urlsafe(18)
            text = _nonce_scripts(text, lab_page_nonce)
            for key in [k for k in resp_headers if k.lower() == "content-security-policy"]:
                del resp_headers[key]
            resp_headers["content-security-policy"] = _lab_page_csp(
                lab_page_nonce, _browser_origin(request), outer_prefix
            )
            # A nonce is per response only if the page is never reused: no
            # cache, and none of the upstream's validators, which describe the
            # un-nonced body and would let a 304 hand back an old nonce.
            for key in [
                k
                for k in resp_headers
                if k.lower() in {"cache-control", "etag", "last-modified", "expires"}
            ]:
                del resp_headers[key]
            resp_headers["cache-control"] = "no-store"
        # After the rewrite, never before: everything injected here is already
        # fully formed for this deployment, and a second pass over it would
        # aim the bar's own URLs into the panel's namespace.
        text = _inject_control_target_bar(
            text, panel_id, path, base_type, outer_prefix, lab_page_nonce
        )
        text = _stamp_storage_scope(
            text,
            base_type,
            resolve_storage_scope(getattr(request.app.state, "terminal_user", "")),
        )
        # Resolved per request, like the storage scope, by the call the hub's own pages use.
        if base_type == "text/html":
            text = _stamp_facility_timezone(text, base_type, get_facility_timezone().key)
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
                content=_rewrite_content(
                    resp.text,
                    panel_id,
                    outer_prefix,
                    _panel_rewrite_prefixes(request, panel_id),
                    panel_base=panel_base,
                ),
                status_code=resp.status_code,
                headers=resp_headers,
                media_type=content_type,
            )

    # JSON, images, binary — pass through unchanged.
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=resp_headers,
        media_type=content_type or None,
    )


@router.api_route(
    "/panel/{panel_id}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
)
async def proxy_panel_root(panel_id: str, request: Request):
    """Proxy the root path of a companion panel (no trailing path segment)."""
    return await proxy_panel(panel_id, "", request)


#: Seconds the upstream WebSocket handshake may take before the proxy gives up
#: on it. ``websockets`` allows ten by default, and a panel is entitled to hold
#: a handshake longer than that on purpose: jupyter-server answers a
#: kernel-channel upgrade only once the kernel behind it has replied to a
#: kernel-info request, and gives that reply a full minute
#: (``MappingKernelManager.kernel_info_timeout``). A kernel joining a bound
#: session imports the framework before it answers, so on a loaded host its
#: reply lands after ten seconds — and it was this proxy, not the panel, that
#: hung up, closing the browser normally before its first frame while the
#: kernel came up fine behind it. The budget is the panel's, plus room for the
#: upgrade itself; a handshake still unanswered after this is not going to be.
UPSTREAM_OPEN_TIMEOUT = 90.0


@router.websocket(f"/panel/{{panel_id}}/{PANEL_CAP_SEGMENT}/{{cap}}/{{path:path}}")
async def proxy_panel_ws_cap(panel_id: str, cap: str, path: str, websocket: WebSocket):
    """Relay a websocket under a runtime panel's C-CAP path to that panel's backend.

    The one ``Origin: null`` handshake the console accepts: the auth gate
    exempts the cap path, so the cap is validated here instead of the cookie
    and the Origin, and the socket reaches only this panel's own backend. A
    forged, foreign-login, expired or other-panel cap is refused at the
    handshake (``403``). Declared above :func:`proxy_panel_ws`.
    """
    if not verify_request_cap(websocket, panel_id, cap, "proxy"):
        await websocket.close(code=1008, reason="invalid capability")
        return
    await proxy_panel_ws(panel_id, path, websocket)


@router.websocket("/panel/{panel_id}/{path:path}")
async def proxy_panel_ws(panel_id: str, path: str, websocket: WebSocket):
    """Forward a WebSocket connection to the companion panel server.

    The upstream handshake is built from scratch rather than by copying the
    browser's — so the operator's cookie and secret are absent by construction,
    matching what :func:`proxy_panel` strips on the HTTP path. The headers added
    are the operator secret and a launched sidecar's own credential, each only
    for a backend its own gate vouches for.

    One field of the browser's handshake is relayed: its subprotocol offer.
    The upstream picks from it, and the browser's own handshake is answered
    with that pick — so the upstream connects first and the browser is
    accepted second. A browser whose non-empty offer is answered with nothing
    treats the handshake as failed.
    """
    backend_url = _resolve_panel_url(websocket, panel_id)
    if not backend_url:
        await websocket.close(code=4004, reason=f"Panel '{panel_id}' not available")
        return

    # A socket to an application panel is a write path: refused on a login
    # that does not arm the application, before any credential is attached.
    application = await _unarmed_application(websocket, panel_id)
    if application is not None:
        # A close reason is capped at 123 bytes on the wire.
        reason = _unarmed_refusal(websocket, application, "WebSocket")
        reason = reason.encode()[:123].decode(errors="ignore")
        await websocket.close(code=4003, reason=reason)
        return

    # Convert http(s) backend URL to ws(s)
    ws_url = backend_url.rstrip("/").replace("http://", "ws://").replace("https://", "wss://")
    target = f"{ws_url}/{path}"
    # The query belongs to the upstream handshake, not to this route: a backend
    # that keys a socket off a query parameter sees nothing without it. Mirrors
    # what the HTTP leg forwards.
    if websocket.url.query:
        target = f"{target}?{websocket.url.query}"

    try:
        import websockets
    except ImportError:
        logger.error("websockets package not installed — cannot proxy WebSocket")
        await websocket.close(code=4500, reason="WebSocket proxy unavailable")
        return

    # No browser header is forwarded; this is the whole outbound header set,
    # and it is empty unless the backend earns the secret or a credential.
    upstream_headers = _terminal_secret_header(websocket, panel_id, backend_url)
    upstream_headers.update(_panel_auth_header(websocket, panel_id, backend_url))

    # The browser's ``Sec-WebSocket-Protocol`` offer, as the ASGI server parsed it.
    offered = list(websocket.scope.get("subprotocols") or [])

    # websockets' opening handshake follows 30x redirects (up to
    # WEBSOCKETS_MAX_REDIRECTS) and re-sends ``additional_headers`` verbatim to
    # each target — the ws↔ws analogue of the httpx redirect leak the HTTP path
    # guards against. When the secret is being carried, refuse to follow any
    # redirect so it cannot ride one off the loopback host the gate vouched for.
    connect_type = _RedirectRefusingConnect if upstream_headers else websockets.connect
    connector = connect_type(
        target,
        additional_headers=upstream_headers or None,
        subprotocols=offered or None,
        open_timeout=UPSTREAM_OPEN_TIMEOUT,
    )

    client_gone = False
    try:
        async with connector as upstream:
            # Answer the browser with the upstream's pick — ``None`` when the
            # two sides negotiated nothing.
            try:
                await websocket.accept(subprotocol=getattr(upstream, "subprotocol", None))
            except (WebSocketDisconnect, OSError):
                # The browser left while the upstream was connecting. Leaving
                # the ``async with`` closes the upstream; nothing to relay.
                client_gone = True
                logger.debug("WebSocket client for panel %s left before accept", panel_id)
                return

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
        if not client_gone:
            try:
                if websocket.application_state is WebSocketState.CONNECTING:
                    # The upstream handshake failed before the browser's was
                    # answered: accept, then close normally (1000) rather than
                    # refuse the handshake with a 403. A browser that offered a
                    # subprotocol still fails its own handshake, as before.
                    await websocket.accept()
                await websocket.close()
            except Exception:
                pass
