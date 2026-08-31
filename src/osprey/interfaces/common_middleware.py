"""Shared middleware for OSPREY FastAPI applications.

Two of the four classes here are about presentation — caching headers and
turning an unhandled exception into structured JSON. The other two are the
HTTP surface's safety layers, one at each end of the stack:

* :class:`WebAuthMiddleware` is the gate, installed outermost. It is what
  stands between a browser (or a process the agent spawned inside its own
  sandbox) and every OSPREY interface's HTTP and websocket surface, and it
  files every ``401``/``403`` it answers in the unified audit ledger.
* :class:`HttpAuditMiddleware` is installed innermost, and files every
  *admitted* state-changing request. Together they make the ledger able to
  answer both halves of the operator's question: what was turned away, and
  what got through.

Neither emitter can cost the request it describes — see
:func:`_emit_audit_record`.

**The two halves are not symmetric about websockets, deliberately.** A
*refused* handshake is filed by the gate on the ``web_auth`` surface (with
method :data:`WEBSOCKET_METHOD`, which a websocket does not otherwise have).
An *admitted* upgrade is not filed here at all: :class:`HttpAuditMiddleware`
only wraps ``http`` scopes, because :data:`SAFE_METHODS` is an HTTP notion and
a socket has no method to discriminate on. What flows through an accepted
socket — terminal input, chat turns — is audited where it is understood, by
the spawn and posture emitters in
:mod:`osprey.interfaces.web_terminal.routes.websocket`, never on
``http_mutation``. So: refused upgrades on ``web_auth``, admitted ones on the
spawn/posture surfaces, and nothing websocket-shaped in between.
"""

import html
import json
import logging
import os
from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, NamedTuple
from urllib.parse import parse_qs, parse_qsl, urlencode

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from osprey.audit.envelope import DECISION_ALLOWED, DECISION_REFUSED, POSTURE_SOURCE_APP
from osprey.interfaces.web_auth import Tier, WebCredentials, classify, get_web_credentials

if TYPE_CHECKING:  # pragma: no cover - typing only
    from osprey.audit.dedup import RecordedDecision

logger = logging.getLogger("osprey.interfaces.middleware")

__all__ = [
    "AUDITED_REFUSAL_STATUSES",
    "AUDIT_ACCOUNT_HEADER",
    "AUDIT_ACCOUNT_KEY",
    "AUDIT_EXPECTED_ACCOUNT_KEY",
    "AUDIT_OIDC_SUBJECT_KEY",
    "AUDIT_ROLE_HEADER",
    "AUDIT_ROLE_SOURCE_HEADER",
    "AUDIT_SUBJECT_HEADER",
    "EXEMPT_PATHS",
    "EXTERNAL_ORIGIN_ENV",
    "HTTP_MUTATION_POSTURE",
    "HTTP_MUTATION_SURFACE",
    "MAX_BODY_PEEK_BYTES",
    "MAX_FORWARDED_VALUE_CHARS",
    "MAX_SESSION_COOKIE_CANDIDATES",
    "OPERATOR_SECRET_HEADER",
    "REASON_MUTATION",
    "REASON_MUTATION_UNANSWERED",
    "REASON_ROUTE_REFUSED",
    "REFUSAL_REASONS",
    "SAFE_METHODS",
    "SESSION_COOKIE_BASE",
    "STATIC_MOUNT_PREFIXES",
    "TERMINAL_USER_ENV",
    "TOKEN_EXCHANGE_PATHS",
    "UNSAFE_FORWARDED_VALUE",
    "WEBSOCKET_REFUSAL_CODE",
    "WEB_AUTH_POSTURE",
    "WEB_AUTH_SURFACE",
    "WEB_PORT_ENV",
    "ExceptionLoggingMiddleware",
    "ForwardedIdentity",
    "HttpAuditMiddleware",
    "NoCacheStaticMiddleware",
    "WebAuthMiddleware",
    "apply_url_prefix",
    "compute_url_prefix",
    "forwarded_identity",
    "is_exempt_path",
    "read_cookie_candidates",
    "read_cookies",
    "session_cookie_name",
]


class NoCacheStaticMiddleware(BaseHTTPMiddleware):
    """Control browser caching for static assets and API responses.

    Vendor assets (versioned filenames like plotly-3.3.1.min.js) are cached
    aggressively — they never change without a filename bump.  All other
    static/API paths are uncached to avoid stale code after updates.
    """

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        path = request.url.path
        if "/vendor/" in path and path.startswith("/static/"):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        elif path.startswith("/static/") or path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response


class ExceptionLoggingMiddleware(BaseHTTPMiddleware):
    """Catch unhandled exceptions — log traceback + return structured JSON 500."""

    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as exc:
            logger.error(
                "Unhandled exception on %s %s", request.method, request.url.path, exc_info=True
            )
            return JSONResponse(
                status_code=500,
                content={"error": str(exc), "path": request.url.path},
            )


#: The header a non-browser operator client carries the operator secret in. In
#: the multi-user shape nginx forwards the deployment's value under this name;
#: in the single-user shape it is what a script or a probe uses instead of a
#: cookie. Lower-cased, because ASGI header names arrive lower-cased.
OPERATOR_SECRET_HEADER = "x-osprey-terminal-secret"

#: The stem of the browser session cookie's name. The settled port is appended
#: (see :func:`session_cookie_name`) so two OSPREY servers on the same host —
#: which share an origin as far as a cookie is concerned, since cookies ignore
#: ports — do not overwrite each other's session.
SESSION_COOKIE_BASE = "osprey_terminal_session"

#: Where the settled port is published for :func:`session_cookie_name`. Read,
#: never popped: it is not a credential and the cookie-exchange route and this
#: middleware must derive the same name from it.
WEB_PORT_ENV = "OSPREY_WEB_PORT"

#: The origin browsers actually reach this app at, when something in front of
#: the app rewrites it. Set by an nginx deployment, where the app's own ``Host``
#: header is the internal service name and would never match the ``Origin`` a
#: browser sends. Unset in the single-user shape, where the two agree.
EXTERNAL_ORIGIN_ENV = "OSPREY_TERMINAL_EXTERNAL_ORIGIN"

#: The multi-user deployment's per-container user name, which is also the URL
#: path prefix nginx mounts that container at. Read, never popped: it is not a
#: credential, and every layer that builds a browser-facing URL must derive the
#: same prefix from it.
TERMINAL_USER_ENV = "OSPREY_TERMINAL_USER"

#: How many same-named session cookies the gate will weigh before giving up.
#: A page on a neighbouring host under the same registrable domain can set a
#: ``Domain``-scoped cookie of the same name, and a browser then sends both, in
#: an order the app does not control — so reading only the first hands that page
#: a lockout. Trying every copy is not the answer either: each costs a
#: constant-time comparison, so an unbounded list is unbounded work per request.
#:
#: What this bound buys is precise, and it is less than closing the lockout: it
#: caps the per-request work and raises the bar from one planted cookie to
#: :data:`MAX_SESSION_COOKIE_CANDIDATES` of them. The primitive is unchanged —
#: a page that can set one cookie of this name can set three — so an attacker
#: with that position still locks the operator out, just less cheaply. Three is
#: past what a legitimate browser ever sends (the app sets exactly one).
#:
#: Closing it properly means removing the shadowing primitive rather than
#: out-counting it: a ``__Host-`` name prefix, which a browser refuses to honour
#: unless the cookie is ``Secure``, ``Path=/`` and ``Domain``-less, on the https
#: shape only — paired with the same derivation for the cookie name nginx is
#: rendered with, so the two stay in step. Left as follow-up.
MAX_SESSION_COOKIE_CANDIDATES = 3

#: Methods that cannot change state, and so are not subject to the Origin check.
#: A cross-site ``GET`` can be *made* by a browser, but the attacker cannot read
#: the answer; a cross-site ``POST`` needs no answer to do damage.
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

#: Paths reachable with no credential at all, matched exactly.
#:
#: ``/health`` is the container healthcheck and the load balancer's probe, which
#: run before any credential exists. ``/static/session.html`` is the second page
#: a one-time token may be exchanged at — a browser that has no cookie yet has
#: to be able to land there, or there is no way to ever get one. Being exempt
#: does not skip the exchange: :class:`WebAuthMiddleware` reads a ``?token=``
#: before it short-circuits on this set, so a login URL pointing here still
#: trades the secret for a cookie rather than leaving it in the address bar.
#: Both are exact strings: ``/health/`` and ``/healthz`` are not exempt.
EXEMPT_PATHS: frozenset[str] = frozenset({"/health", "/static/session.html"})

#: Paths where a ``?token=`` query parameter may carry the operator secret to
#: bootstrap a session cookie, matched exactly. These are the two pages a login
#: URL can point a cookie-less browser at; :class:`WebAuthMiddleware` answers
#: the request itself, minting a session and redirecting to the token-stripped
#: URL, so every interface that installs the gate performs the exchange.
#:
#: The scope is deliberately narrow. A secret in a URL leaks — into access
#: logs, browser history and the ``Referer`` of the next request — so ``?token=``
#: authenticates **only a GET to one of these paths**, never an arbitrary route
#: or a mutating method. It authenticates exactly once, to hand out the cookie
#: that takes over from then on; it is not a universal credential. (Both entries
#: are also :data:`EXEMPT_PATHS` or unprotected navigation targets, so widening
#: this set is the only way to grant ``?token=`` reach — a route added elsewhere
#: cannot accidentally accept a URL-borne secret.)
TOKEN_EXCHANGE_PATHS: frozenset[str] = frozenset({"/", "/static/session.html"})

#: The static mounts :func:`osprey.interfaces._app_setup.configure_interface_app`
#: installs, in its order. Held as mount *paths* rather than as ready-made
#: prefixes so a test can walk ``app.routes`` and check the two lists agree — a
#: mount added there without an entry here would serve assets behind a gate that
#: the browser cannot pass while fetching them.
#:
#: These are CSS, JavaScript and fonts, served to a browser that is by
#: definition not yet authenticated on its first paint. Nothing here reads
#: process state or accepts input; the credential-bearing surface is ``/api`` and
#: the websockets, and none of it is under these prefixes.
STATIC_MOUNT_PREFIXES: tuple[str, ...] = ("/static/fonts", "/design-system", "/static")

#: How much of a request body the panel-tier check will read before giving up
#: and calling the body unknowable. A panel registration is a few hundred bytes;
#: anything past this is not one, and reading it would let an unauthenticated
#: caller make the gate buffer arbitrary memory. Giving up fails *closed* — an
#: over-long body is treated as carrying a ``url``, which is operator-tier.
MAX_BODY_PEEK_BYTES = 64 * 1024

#: Close code for a websocket refused on a server that does not offer the
#: ``websocket.http.response`` extension. 1008 (policy violation) is honest
#: here: the credential was evaluated and rejected. It is the fallback only —
#: the extension gives the client a real 401 it can read, and uvicorn's
#: websockets implementation offers it.
WEBSOCKET_REFUSAL_CODE = 1008

_MISSING_DETAIL = "authentication required"
_INVALID_DETAIL = "invalid credential"
_TIER_DETAIL = "this route requires operator credentials"
_ORIGIN_DETAIL = "cross-origin request refused"
_UNAVAILABLE_DETAIL = "authentication is not available; see the service log"

# --------------------------------------------------------------------------- #
# The audit vocabulary of the two HTTP emitters
# --------------------------------------------------------------------------- #

#: The surface :class:`WebAuthMiddleware` names on its own records: the gate,
#: not the route behind it. One surface for the whole gate rather than one per
#: refusal branch, so "who was turned away from this container, and why" is a
#: single file rather than a join across four.
WEB_AUTH_SURFACE: str = "web_auth"

#: The posture the gate stamps on its records. The gate governs *entry*: it
#: holds no posture store, spawns no session, and by the time it decides,
#: nothing has been written. ``sandbox`` is the honest reading of "this surface
#: changed nothing", and it is paired with
#: :data:`~osprey.audit.envelope.POSTURE_SOURCE_APP` so no reader mistakes it
#: for evidence about what the caller would have been allowed to do inside.
#: Same reasoning, same value, as the auth sidecar's own stamp.
WEB_AUTH_POSTURE: str = "sandbox"

#: The surface :class:`HttpAuditMiddleware` names: an admitted state-changing
#: HTTP request. Distinct from :data:`WEB_AUTH_SURFACE` because the two answer
#: different questions and an operator reads them for different reasons — one
#: file of things that were turned away, one of things that got through.
HTTP_MUTATION_SURFACE: str = "http_mutation"

#: The posture the mutation layer stamps. ``writes``, because that is what this
#: surface *is*: a request reaching this layer has passed the gate and is about
#: to change state, and stamping ``sandbox`` on it would describe the opposite
#: of what happened. As above, ``posture_source=app`` says the value is the
#: app's own stamp and not a session's posture — an interface app has none.
HTTP_MUTATION_POSTURE: str = "writes"

#: The statuses :meth:`WebAuthMiddleware._refuse` files a record for.
#:
#: ``401`` and ``403`` are decisions *about the caller* — a credential was
#: evaluated, or an origin was compared, and the answer was no. ``503`` is the
#: gate saying it cannot run at all: it names no decision about the request, it
#: is already an ``ERROR`` on the service log, and filing it here would put a
#: deployment fault in the column an operator reads for caller behaviour.
AUDITED_REFUSAL_STATUSES: frozenset[int] = frozenset({401, 403})

#: The machine-ish reason each refusal detail is filed under.
#:
#: Keyed on the detail constants rather than on the status, because the status
#: is two values and the causes are four: an absent credential and an invalid
#: one are both ``401`` and are not the same event to an operator watching a
#: deployment. Keyed on the constants rather than re-spelled, so a reworded
#: message cannot silently drop its records into the fallback.
REFUSAL_REASONS: dict[str, str] = {
    _MISSING_DETAIL: "missing_credential",
    _INVALID_DETAIL: "invalid_credential",
    _TIER_DETAIL: "tier_refused",
    _ORIGIN_DETAIL: "origin_mismatch",
}

#: Where a refusal lands when its detail is not in :data:`REFUSAL_REASONS`.
#: Unreachable today (the gate's whole audited vocabulary is mapped, and a test
#: pins that); it exists so that a fifth refusal added without a mapping is
#: recorded uselessly rather than not at all.
REASON_UNCATEGORIZED: str = "refused"

#: An admitted state-changing request that the route answered successfully.
REASON_MUTATION: str = "mutation"

#: An admitted state-changing request that the *route* then refused. The gate
#: let it in; something further in did not. Recorded as a refusal rather than
#: as an admission, because a layer that stamped everything it admitted
#: ``allowed`` would overwrite the decision the inner layer actually made.
REASON_ROUTE_REFUSED: str = "route_refused"

#: An admitted state-changing request the route never answered — it raised
#: before producing a status. The *decision* stays ``allowed``, which is the
#: truth (the gate let it in and it ran), but the reason says on its own that
#: no answer was produced. Given its own word rather than folded into
#: :data:`REASON_MUTATION` because ``decision``/``reason`` is the pair an
#: operator scans, and "admitted and succeeded" versus "admitted and blew up"
#: should not require also parsing ``detail`` for ``status=none``.
REASON_MUTATION_UNANSWERED: str = "mutation_unanswered"

#: Names *who proved* the login behind an authorized request, as nginx forwards
#: it from the auth sidecar's ``auth_request`` answer. In a password session
#: that is the roster username; in an OIDC session it is the provider's
#: subject, which names a person and not a roster card — see
#: :data:`AUDIT_ACCOUNT_HEADER` for the card itself.
#:
#: Spelled here rather than imported from
#: :mod:`osprey.services.auth_sidecar.identity_headers`, which owns it: an
#: interface app has no business pulling a service package into its import
#: closure for four string constants. ``tests/interfaces/test_http_audit_emitters.py``
#: pins the four spellings against each other, the same trade the rest of the
#: audit work makes for a cross-package constant.
AUDIT_SUBJECT_HEADER: str = "X-Osprey-Auth-Subject"

#: Names the roster account an authorized request is on — the card ``/verify``
#: checked the session against, and the one this container can compare with the
#: user it believes it is serving. Coincides with :data:`AUDIT_SUBJECT_HEADER`
#: in a password session, where the roster username *is* the proof, and
#: diverges in an OIDC session, where a shared card may be opened by any of
#: several people. Absent from an older sidecar image that does not emit it.
AUDIT_ACCOUNT_HEADER: str = "X-Osprey-Auth-Account"

#: Names the role that account holds. Absent means *no* role, never a default
#: one — the deny-safe reading the sidecar documents.
AUDIT_ROLE_HEADER: str = "X-Osprey-Auth-Role"

#: Names where that role came from — ``roster`` or ``claim``, the sidecar's
#: closed vocabulary. Decoded under the same bound as the other two, so a
#: value the ledger would refuse cannot reach a screen instead. The ledger
#: reads it and does not record it: what a login granted is the sidecar's
#: own record, and this gate has never re-stated it.
AUDIT_ROLE_SOURCE_HEADER: str = "X-Osprey-Auth-Role-Source"

#: The ``detail`` key naming the *account* a request arrived authorized as, as
#: forwarded by nginx from the sidecar's answer.
#:
#: Spelled ``account`` and not ``subject`` on purpose. The envelope's top-level
#: ``subject`` on these two surfaces is the **route** (``"POST /api/config"``),
#: while the auth sidecar's own emitter puts the **account** in its top-level
#: ``subject``. One word would therefore have meant two things across the
#: epic's surfaces — and both of them inside a single HTTP record, one level
#: apart. ``account`` collides with neither.
AUDIT_ACCOUNT_KEY: str = "account"

#: The ``detail`` key naming this container's own user, present **only** when
#: the forwarded account is not it. Its presence *is* the mismatch signal.
AUDIT_EXPECTED_ACCOUNT_KEY: str = "expected_account"

#: The ``detail`` key naming *who proved* the login, when that is not the
#: account itself — the OIDC subject on a shared card.
#:
#: Spelled ``oidc_subject`` and deliberately **not** ``identity``. The audit
#: writer owns ``identity`` for the ledger's directory component, so a key of
#: that name inside ``detail`` would put one word on two different things in
#: the same record — the folder a record is filed under, and a person named
#: inside it. ``subject`` was unavailable for the reason
#: :data:`AUDIT_ACCOUNT_KEY` records: the envelope's top-level ``subject`` on
#: these surfaces is the route.
AUDIT_OIDC_SUBJECT_KEY: str = "oidc_subject"

#: The longest forwarded identity value that may be recorded verbatim.
#:
#: Comfortably above anything real — an OIDC ``sub`` is a short URL-safe
#: identifier and a role is constrained to ``USERNAME_CHARSET_RE`` by the
#: render-time lint — and far below
#: :data:`~osprey.audit.envelope.MAX_DETAIL_CHARS`, which is the point. The
#: envelope truncates ``detail`` silently, so an unbounded caller-chosen value
#: could fill it and push :data:`AUDIT_EXPECTED_ACCOUNT_KEY` off the end,
#: making a forged header read in the ledger as an ordinary admitted request.
#: Bounding the value at the emitter means every conditional key the emitter
#: appends survives, whatever a caller sends.
MAX_FORWARDED_VALUE_CHARS: int = 128

#: What a forwarded subject or role is recorded as when it could not have come
#: from the sidecar. The headers arrive over the wire and nothing upstream of
#: this gate is obliged to have produced them, so a value outside the sidecar's
#: printable-ASCII contract is *named* as unusable rather than copied into a
#: record an operator will read as an identifier.
UNSAFE_FORWARDED_VALUE: str = "<unsafe>"

#: The method an audit record names for a websocket, which has none. Matches
#: the spelling :meth:`WebAuthMiddleware._log_origin_refusal` already uses, so
#: a log line and a ledger line about the same refusal read the same.
WEBSOCKET_METHOD: str = "WEBSOCKET"

#: The detail value standing in for a status that never arrived — the route
#: raised before it answered. Spelled rather than omitted: a record whose
#: ``status`` key is missing reads like an older record, while one that says
#: ``none`` says what happened.
_STATUS_NONE = "none"

#: ``scheme`` in a websocket scope is ``ws``/``wss``; the ``Origin`` a browser
#: sends for that handshake is the *page's*, which is ``http``/``https``.
_ORIGIN_SCHEME = {"ws": "http", "wss": "https"}

#: The port each scheme leaves out when an origin is serialized.
_DEFAULT_ORIGIN_PORTS = {"http://": ":80", "https://": ":443"}


def _normalize_origin(origin: str) -> str:
    """Put an origin in the serialization a browser would have written.

    A browser lower-cases the scheme and host and **elides the scheme's default
    port** (RFC 6454 / WHATWG URL): a page served from ``http://host:80`` sends
    ``Origin: http://host``. Nothing on the app's side does that. The deployment
    renderer writes ``http://<fqdn>:<nginx_port>`` with the port always explicit
    when TLS is off, so ``nginx_port: 80`` declares an origin no browser will
    ever send; the single-user ``Host`` fallback carries whatever port the
    client happened to type. Comparing the two raw refuses every mutating
    request while the interface still loads, which reads as a broken app rather
    than as a configuration mismatch.

    Normalizing here rather than at each producer is deliberate: this is the one
    place both sides meet, and it also covers a hand-written
    :data:`EXTERNAL_ORIGIN_ENV`. Only the *default* port is dropped, and only
    for its own scheme — ``http://host:443`` and ``https://host:80`` keep
    theirs, and a genuinely different port still fails to match.
    """
    value = origin.strip().lower()
    for scheme, default_port in _DEFAULT_ORIGIN_PORTS.items():
        if value.startswith(scheme) and value.endswith(default_port):
            return value[: -len(default_port)]
    return value


def session_cookie_name(port: int | str | None = None) -> str:
    """Return the browser session cookie's name for a given server port.

    Args:
        port: The settled port. ``None`` reads :data:`WEB_PORT_ENV`, which is
            how both this middleware and the token-exchange route reach the same
            name without being told it.

    Returns:
        ``osprey_terminal_session_<port>``, or the bare
        :data:`SESSION_COOKIE_BASE` when no port is known or the value is not a
        plain number. The fallback matters more than it looks: a cookie *name*
        is a token, and a stray value from the environment could otherwise
        produce a name no browser will store, leaving the exchange silently
        unable to hand out sessions.
    """
    if port is None:
        port = os.environ.get(WEB_PORT_ENV, "")
    text = str(port).strip()
    if not text.isdigit():
        return SESSION_COOKIE_BASE
    return f"{SESSION_COOKIE_BASE}_{text}"


def compute_url_prefix() -> str:
    """Compute the per-container URL path prefix for multi-user deployments.

    Multi-user deployments run one container per user behind a shared nginx
    front door, each mounted at ``/u/<user>/``. nginx **strips** that prefix
    before proxying, so the app itself only ever sees bare paths — which means
    every URL the app hands *back* to a browser (a redirect ``Location``, an
    injected asset URL) has to put the prefix on again by hand.

    This lives here, next to the gate, because the gate is the first component
    to need it: the ``?token=`` exchange answers with a ``Location``, and a bare
    one would send the browser to the front door's root instead of back into the
    user's own mount. The Web Terminal's routes and templates import both
    halves from here too, so there is one implementation rather than two.

    Returns:
        ``"/u/<user>"`` when :data:`TERMINAL_USER_ENV` is set and non-empty;
        otherwise ``""``, which makes every application of it a no-op and
        preserves single-origin/dev behaviour exactly.
    """
    user = os.environ.get(TERMINAL_USER_ENV, "").strip()
    return f"/u/{user}" if user else ""


def apply_url_prefix(prefix: str, path: str) -> str:
    """Prepend ``prefix`` to ``path`` unless it must pass through unchanged.

    An already-absolute URL (``http://``, ``https://``, protocol-relative
    ``//``) is never touched so an external URL is never corrupted, and an
    empty prefix is a byte-identical no-op (single-origin/dev behavior).

    The browser-side ``withPrefix()`` in ``static/js/api.js`` additionally
    guards non-root-absolute and already-prefixed inputs, which server-side
    callers never produce — that asymmetry is deliberate, not drift.
    """
    if not prefix or path.startswith(("http://", "https://", "//")):
        return path
    return f"{prefix}{path}"


def is_exempt_path(path: str) -> bool:
    """Whether ``path`` is reachable without a credential.

    Exactly :data:`EXEMPT_PATHS`, plus anything under a
    :data:`STATIC_MOUNT_PREFIXES` mount. A prefix matches the mount path itself
    or a path below it — ``/static`` and ``/static/app.css`` are exempt,
    ``/static-secrets`` is not, because a bare ``startswith`` would hand every
    path that merely begins with those characters an unauthenticated route.

    A path carrying a ``..`` segment is never exempt, checked here rather than
    left to the router. ``/static/../api/config`` begins with ``/static/`` and
    would otherwise clear the gate as a static asset, even though the segment it
    resolves to is an operator route. Downstream normalisation happens to 404 it
    today, but this module's guarantee is that the gate's *own* matching cannot
    be talked past — it must not depend on a later component remembering to
    normalise. Both a percent-encoded ``..`` and a bare one are covered: the
    ASGI ``path`` is already percent-decoded, so ``%2e%2e`` arrives here as
    ``..``.
    """
    if ".." in path.split("/"):
        return False
    if path in EXEMPT_PATHS:
        return True
    return any(path == prefix or path.startswith(f"{prefix}/") for prefix in STATIC_MOUNT_PREFIXES)


def _scope_headers(scope: Scope) -> dict[str, str]:
    """Decode an ASGI scope's headers into a lower-cased mapping.

    ``latin-1`` because that is what ASGI specifies and because it cannot fail:
    a header here is unauthenticated input that may hold any byte at all,
    including a NUL, and a decode error would turn a forged header into a 500
    where a refusal belongs.

    Repeats keep the first value, except ``cookie``, whose repeats are joined
    the way a single header would have carried them — HTTP/2 clients are allowed
    to split the cookie header, and dropping the second half would make a valid
    session look absent.
    """
    headers: dict[str, str] = {}
    for raw_name, raw_value in scope.get("headers") or ():
        name = raw_name.decode("latin-1").lower()
        value = raw_value.decode("latin-1")
        if name not in headers:
            headers[name] = value
        elif name == "cookie":
            headers[name] = _join_cookie_headers((headers[name], value))
    return headers


# --------------------------------------------------------------------------- #
# The audit emitters both HTTP layers share
# --------------------------------------------------------------------------- #


def _audit_subject(scope: Scope) -> str:
    """What an audit record on this connection says was acted on.

    ``"<METHOD> <path>"`` — the route, which is an identifier, never a body or
    a query string, which are values. A websocket has no method, so it is
    named :data:`WEBSOCKET_METHOD`, matching what the origin-refusal log line
    has always called one.
    """
    if scope.get("type") == "websocket":
        method = WEBSOCKET_METHOD
    else:
        method = (scope.get("method") or "?").upper()
    return f"{method} {scope.get('path') or '?'}"


def _recordable(value: str) -> bool:
    """Whether a forwarded identity value may be recorded as an identifier.

    Printable ASCII, no space anywhere, and at most
    :data:`MAX_FORWARDED_VALUE_CHARS` characters. The first two are the
    sidecar's own contract for all four identity headers (see
    :mod:`osprey.services.auth_sidecar.identity_headers`, which refuses to mint
    or emit anything outside printable ASCII, or anything space-padded); two
    rules are added for this ledger. A value carrying an *interior* space
    cannot be written into a space-separated ``key=value`` detail without
    becoming two fields. And a value long enough to fill ``detail`` would push
    the keys appended after it past the envelope's silent truncation — so the
    one input a forger fully controls would be the one that erases the mismatch
    marker from the record. Nothing real is lost by either: an OIDC ``sub`` is
    a short URL-safe identifier, a roster account and a role name are both
    constrained to ``USERNAME_CHARSET_RE`` by the render-time lint, and a role
    source is one of two framework constants.

    A value that fails this did not come from the sidecar, or cannot be written
    down unambiguously; either way, copying it verbatim would put a forged or
    unreadable string into the column an operator reads as an identifier.
    """
    return (
        bool(value)
        and len(value) <= MAX_FORWARDED_VALUE_CHARS
        and all("\x20" < char <= "\x7e" for char in value)
    )


class ForwardedIdentity(NamedTuple):
    """What nginx forwarded about the login behind one request.

    A named tuple rather than four positional values because ``subject`` and
    ``account`` answer two questions that read alike and are not the same one:
    *who proved the login* and *whose roster card the request is on*. Every
    reader here says which it meant.

    Attributes:
        subject: Who proved the login — the OIDC provider's subject, or the
            roster username in a password session.
        account: The roster account the request is on. ``None`` from a sidecar
            image that predates the header; readers fall back to ``subject``.
        role: The privilege that account holds, or ``None`` for no role.
        role_source: ``roster`` or ``claim``, present only beside a role.
    """

    subject: str | None
    account: str | None
    role: str | None
    role_source: str | None


def forwarded_identity(headers: Mapping[str, str]) -> ForwardedIdentity:
    """The :class:`ForwardedIdentity` nginx forwarded on this request, if any.

    Each field is ``None`` when its header is absent or empty — the sidecar
    omits a header it has no value for, so absence is a state of its own and
    must not be flattened into a blank string. ``account`` is additionally
    ``None`` against a sidecar image built before that header existed, which is
    why a reader comparing accounts has to keep a subject fallback. A value
    that fails :func:`_recordable` becomes :data:`UNSAFE_FORWARDED_VALUE`:
    present, and named as unusable.

    Public because it is the ONE decoder for these headers: the audit emitters
    record the account and the role it returns, and the web terminal's page
    shows the role and where it came from. A second reader with its own bound
    would let a value the ledger refuses reach the screen, or the reverse.
    ``headers`` is read by lower-cased name, so a plain dict of lower-cased
    keys and Starlette's case-insensitive ``Headers`` both work.
    """
    resolved: list[str | None] = []
    for header in (
        AUDIT_SUBJECT_HEADER,
        AUDIT_ACCOUNT_HEADER,
        AUDIT_ROLE_HEADER,
        AUDIT_ROLE_SOURCE_HEADER,
    ):
        raw = (headers.get(header.lower()) or "").strip()
        if not raw:
            resolved.append(None)
        elif _recordable(raw):
            resolved.append(raw)
        else:
            resolved.append(UNSAFE_FORWARDED_VALUE)
    return ForwardedIdentity(*resolved)


def _container_user() -> str:
    """The single user this container was rendered for, or ``""``.

    Empty in every shape that is not the multi-user deployment — the laptop,
    a framework service, the dispatch worker — where there is no per-user
    container and so nothing for a forwarded subject to disagree with.
    """
    return os.environ.get(TERMINAL_USER_ENV, "").strip()


def _audit_detail(scope: Scope, headers: dict[str, str], **extra: str) -> tuple[str, str | None]:
    """The record's ``detail`` string and the role to put in its own field.

    ``detail`` is a space-separated run of ``key=value`` identifiers, which is
    what makes it greppable without being parsed. Three keys come from the
    forwarded identity, and each says something by being there — or by not:

    * :data:`AUDIT_ACCOUNT_KEY` — the roster **account** the request is on,
      present whenever nginx named an identity at all. Not spelled ``subject``:
      on these surfaces the envelope's own ``subject`` field is the route.
    * :data:`AUDIT_EXPECTED_ACCOUNT_KEY` — this container's own user, present
      **only** when the forwarded account is not it. Its presence *is* the
      mismatch: one user's authorization arrived at another user's container.
    * :data:`AUDIT_OIDC_SUBJECT_KEY` — *who proved* the login, present only
      when that is not the account itself. A password session, where the roster
      username is the proof, therefore gains no key at all; an OIDC session
      records the provider's subject beside the card it opened.

    **The comparison is on the account, never on the subject.** Under
    ``auth.method: oidc`` the subject is the IdP's assertion about a person —
    an opaque id or an email — while ``OSPREY_TERMINAL_USER`` is the roster
    name of the card, so comparing those two disagreed by construction and
    marked every request of every card as a mismatch. Only
    :data:`AUDIT_ACCOUNT_HEADER` names something this container can be the
    container *for*. When that header is absent — a deployment pinning a
    sidecar image built before it existed — the subject is compared exactly as
    it was before, so such a deployment keeps its old behaviour rather than
    losing the check altogether.

    The mismatch key is written *before* the account it disagrees with. Each
    forwarded value is bounded at :data:`MAX_FORWARDED_VALUE_CHARS`, so nothing
    a caller sends can reach the envelope's ``detail`` truncation today;
    writing the marker first means that stays true however the ``extra`` keys
    grow, and putting the subject *last* keeps the same true of the account.

    The mismatch is recorded and logged, never enforced. Refusing on it would
    be a behaviour change smuggled in under an audit heading, and the gate has
    never treated these headers as a credential.
    """
    forwarded = forwarded_identity(headers)
    subject, role = forwarded.subject, forwarded.role
    # The account when the sidecar named one, else the subject — which is what
    # the account header carries in a password session anyway, and is the whole
    # of the fallback against an older image.
    account = forwarded.account if forwarded.account is not None else subject
    parts = [f"{key}={value}" for key, value in extra.items()]
    if account is not None:
        container_user = _container_user()
        if container_user and account != container_user:
            parts.append(f"{AUDIT_EXPECTED_ACCOUNT_KEY}={container_user}")
            logger.warning(
                "Forwarded account %r does not name this container's user %r on %s; "
                "recorded, not refused",
                account,
                container_user,
                _audit_subject(scope),
            )
        parts.append(f"{AUDIT_ACCOUNT_KEY}={account}")
        if subject is not None and subject != account:
            parts.append(f"{AUDIT_OIDC_SUBJECT_KEY}={subject}")
    return " ".join(parts), role


def _emit_audit_record(build: Callable[[], dict[str, Any] | None]) -> None:
    """File one record in the unified ledger. Never raises, never blocks.

    Both HTTP emitters funnel through here for the same reason the writer has
    its own never-raises boundary: an unwritable audit zone must degrade the
    trail rather than turn a served request into a 500, and the gate in
    particular must be able to refuse a caller even when it cannot say so in
    the ledger.

    **The fields are built by** ``build``, **not passed in**, so that the whole
    of an emitter's work happens inside this boundary — header decoding, detail
    assembly and the mismatch ``WARNING`` included, not just the write. A
    builder run by its caller would leave those outside: on the gate a raise
    there turns a ``401`` into a propagating exception, and on the mutation
    layer, where the call is made from a ``finally``, it both masks the route's
    own exception and turns an already-served ``200`` into a ``500``. Those are
    the two outcomes the module docstring rules out, so the boundary has to
    enclose everything, not the last step. Returning ``None`` from ``build``
    means "nothing to file" and writes nothing.

    The writer is imported *inside* the call rather than at module scope. This
    module is in the import closure of every interface app and of the auth
    sidecar's own harness, and both are held to what they may drag in; the
    lookup is a ``sys.modules`` hit after the first request.
    """
    try:
        fields = build()
        if fields is None:
            return
        from osprey.audit.writer import record

        # ``actor`` is the writer's to fill: it resolves the same ladder the
        # ledger directory is keyed on, so the two cannot drift.
        record(**fields)
    except Exception:  # noqa: BLE001 - the audit trail degrades; the request does not.
        logger.warning("Could not file an HTTP audit record", exc_info=True)


def _join_cookie_headers(values: Iterable[str]) -> str:
    """Concatenate repeated ``Cookie`` headers into the one header they mean.

    HTTP/2 clients are allowed to split the cookie header, and the halves mean
    exactly what a single header carrying them in order would. Spelled once so
    that no reader of an untrusted request can rejoin them differently — see
    :func:`read_cookie_candidates`.
    """
    return "; ".join(values)


def read_cookie_candidates(header_values: Iterable[str], name: str) -> list[str]:
    """Return the values a request's ``Cookie`` headers carry under ``name``.

    The raw-headers form of :func:`read_cookies`, for a caller holding the list
    of headers a request arrived with rather than the single joined header
    :func:`_scope_headers` hands the gate. Rejoining is not optional and not the
    caller's to get right: a reader that took only the first header would miss a
    session an HTTP/2 browser really sent, and a reader that rejoined them its
    own way could offer a different set of candidates than the gate accepted —
    which is how a credential ends up admitted on every request and revoked by
    no logout.
    """
    return read_cookies(_join_cookie_headers(header_values), name)


def read_cookies(header: str | None, name: str) -> list[str]:
    """Return the values a ``Cookie`` header carries under ``name``, in order.

    Hand-parsed rather than handed to :mod:`http.cookies`, which raises on input
    it dislikes and silently drops the rest of a header after a segment it
    cannot parse. A cookie header is written by whoever is calling, so the only
    acceptable behaviour on malformed input is "this credential is absent" —
    never an exception, and never losing a well-formed cookie because a
    neighbouring one was junk.

    **A list rather than the first match**, capped at
    :data:`MAX_SESSION_COOKIE_CANDIDATES`. A browser can be made to send two
    cookies of the same name — a page on a sibling host under the same
    registrable domain sets a ``Domain``-scoped one, and the browser then sends
    it alongside the app's own host-scoped cookie, in an order the app does not
    control. Taking only the first would let that page shadow a live session
    permanently, which is a lockout an operator has no way to diagnose. The cap
    is what keeps the alternative honest: each candidate costs one constant-time
    comparison, so the work per request stays bounded by a constant instead of
    by the length of an attacker-written header.

    **Public, and deliberately so.** The logout route
    (``web_terminal/routes/websocket.py``) has to revoke exactly the candidates
    this gate would accept, so it reaches this primitive — through
    :func:`read_cookie_candidates`, which rejoins the raw headers first — rather
    than deriving its own. Two readers of the same untrusted header that
    disagree about even one shape is how a credential ends up admitted by the
    gate and never revoked by logout — so what this returns, repeats included,
    is a contract an external caller depends on rather than an internal detail.
    Note that it takes a header that has already been *joined*, as the gate's is;
    a caller holding the headers a request arrived with wants
    :func:`read_cookie_candidates`.
    """
    if not header:
        return []
    values: list[str] = []
    for segment in header.split(";"):
        key, separator, value = segment.partition("=")
        if not separator or key.strip() != name:
            continue
        candidate = value.strip()
        if len(candidate) >= 2 and candidate.startswith('"') and candidate.endswith('"'):
            candidate = candidate[1:-1]
        if candidate:
            values.append(candidate)
        if len(values) >= MAX_SESSION_COOKIE_CANDIDATES:
            break
    return values


def _read_bearer(header: str | None) -> str | None:
    """Return the token from an ``Authorization: Bearer <token>`` header.

    ``None`` for an absent header, a non-``Bearer`` scheme, or an empty token,
    so that a caller sending ``Authorization: Bearer`` with nothing after it is
    treated as carrying no credential rather than as offering an empty one.
    """
    if not header:
        return None
    scheme, separator, token = header.partition(" ")
    if not separator or scheme.strip().lower() != "bearer":
        return None
    return token.strip() or None


def _exchange_query_token(scope: Scope) -> str | None:
    """Return a ``?token=`` operator secret eligible to bootstrap a session.

    ``None`` unless the request is a **GET** to one of
    :data:`TOKEN_EXCHANGE_PATHS` carrying a non-empty ``token`` query parameter.
    Every one of those conditions is a security boundary, not a convenience:

    * **GET only.** A secret that reaches the server in a URL is written to
      access logs and browser history; permitting it on a mutating method would
      let a logged or shared URL replay a state change.
    * **Exchange paths only.** ``?token=`` is not a universal credential — it
      bootstraps the cookie on exactly the two login-landing pages and nowhere
      else.
    * **Never a websocket.** A websocket handshake synthesises a ``GET`` method,
      so the scope ``type`` is checked explicitly to keep a URL-borne secret from
      opening a live channel.

    A caller that clears these gates presents the token as its one credential,
    and the middleware judges it with a single :func:`WebCredentials.verify_operator`.
    """
    if scope.get("type") != "http":
        return None
    if (scope.get("method") or "").upper() != "GET":
        return None
    if (scope.get("path") or "") not in TOKEN_EXCHANGE_PATHS:
        return None
    raw = scope.get("query_string") or b""
    # ``latin-1`` cannot raise on the arbitrary bytes an unauthenticated query
    # string may hold; ``parse_qs`` drops blank values, so ``?token=`` with no
    # value yields no key and is correctly treated as carrying no credential.
    params = parse_qs(raw.decode("latin-1"))
    values = params.get("token")
    if not values:
        return None
    return values[0].strip() or None


def _body_has_url(body: bytes | None) -> bool:
    """Whether a JSON request body carries a top-level ``url`` key.

    ``True`` for anything that cannot be answered — an unreadable body, one that
    is not JSON, one that is not an object. This is the fail-closed direction
    that :func:`osprey.interfaces.web_auth.classify` documents: a body the gate
    cannot read must be treated as the URL-backed registration that only an
    operator may make, never as the harmless one.
    """
    if body is None:
        return True
    try:
        parsed = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return True
    return not isinstance(parsed, dict) or "url" in parsed


def _recovery_html() -> str:
    """The one sentence on the refusal page that says how to get back in.

    Which sentence is true depends on the shape, and the shape is readable from
    :data:`TERMINAL_USER_ENV` — set only by the multi-user deployment, which
    mounts one container per user behind nginx.

    * Set — the container's ``osprey web`` prints no ``Open:`` line at all: it
      suppresses the announcement whenever a bind host is declared, because
      nginx owns the way in. Telling the reader to re-open a line nothing
      printed is a dead end, and a browser genuinely does reach this page in
      that shape (``auth.method: none``, or a roster entry with
      ``login: false``, leaves the app's own gate as the outermost one). The
      way in there is ``osprey users login-url``.
    * Unset — single-user, where the launcher did print the line and re-opening
      it is exactly right.

    Both branches are fixed strings. The user name is not interpolated: it is
    not a credential, but the reader already has it in their own URL, and a
    static page is one less thing that can echo a request back at a browser.
    """
    if os.environ.get(TERMINAL_USER_ENV, "").strip():
        return (
            "<p>Ask whoever runs this deployment for a fresh login link, or run "
            "<code>osprey users login-url &lt;user&gt;</code> on the deployment "
            "host. Opening that link hands this browser a fresh session.</p>"
        )
    return (
        "<p>Re-open the login link the command that started this server printed — "
        "the line beginning <code>Open:</code>, whose URL carries a "
        "<code>?token=</code>. Opening it hands this browser a fresh session.</p>"
    )


def _refusal_page(status: int, detail: str) -> str:
    """Render the HTML a *navigating* browser gets instead of the JSON refusal.

    Deliberately one self-contained string with no asset references. The static
    mounts are unauthenticated and would load, but a refusal page that depends
    on them is a refusal page that renders blank the moment a mount is missing
    — and this page exists precisely for the operator whose session is already
    broken. Nothing here is dynamic except the status, the detail and which of
    :func:`_recovery_html`'s two fixed sentences applies; the first two are this
    module's own constants, and ``detail`` is escaped anyway, because a page
    that interpolates a value into markup should not depend on where the value
    came from staying true.
    """
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>OSPREY — sign in required</title><style>"
        "body{margin:0;min-height:100vh;display:flex;align-items:center;"
        "justify-content:center;background:#14161a;color:#e6e8eb;"
        "font:16px/1.6 ui-sans-serif,system-ui,-apple-system,sans-serif}"
        "main{max-width:34rem;padding:2rem}"
        "h1{font-size:1.35rem;margin:0 0 .75rem}"
        "p{margin:0 0 .75rem;color:#aab2bd}"
        "code{background:#1e2228;border-radius:4px;padding:.1rem .35rem;color:#e6e8eb}"
        "small{color:#6d7580}"
        "</style></head><body><main>"
        "<h1>This session is not signed in</h1>"
        "<p>OSPREY needs a browser session before it will serve this page.</p>"
        f"{_recovery_html()}"
        f"<p><small>{status} — {html.escape(detail)}</small></p>"
        "</main></body></html>\n"
    )


async def _peek_body(receive: Receive) -> tuple[bytes | None, Receive]:
    """Read the request body, returning it and a ``receive`` that replays it.

    An ASGI body can only be read once, so a gate that consumes it to make a
    decision has to give the application back a stream indistinguishable from
    the one it took. The replacement replays the buffered messages verbatim —
    including a ``http.disconnect`` that arrived mid-body — before delegating to
    the original callable, so a handler that streams, or that checks
    ``more_body``, sees exactly what it would have seen.

    Returns:
        ``(body, receive)`` where ``body`` is ``None`` when the request exceeded
        :data:`MAX_BODY_PEEK_BYTES` or ended early — the "unknowable" answer
        that :func:`_body_has_url` turns into the safe tier.
    """
    buffered: list[Message] = []
    size = 0
    truncated = False
    while True:
        message = await receive()
        buffered.append(message)
        if message.get("type") != "http.request":
            truncated = True
            break
        size += len(message.get("body") or b"")
        if size > MAX_BODY_PEEK_BYTES:
            truncated = True
            break
        if not message.get("more_body", False):
            break

    # A copy, not ``buffered`` itself: the join below reads that list, and a
    # replay that consumed it would leave the body it returned empty.
    pending = list(buffered)

    async def replay() -> Message:
        if pending:
            return pending.pop(0)
        return await receive()

    if truncated:
        return None, replay
    body = b"".join(message.get("body") or b"" for message in buffered)
    return body, replay


class WebAuthMiddleware:
    """Pure-ASGI gate refusing every unauthenticated request to an interface app.

    OSPREY's interfaces run in the same process as the agent's sandbox and are
    reachable over loopback, so before this existed anything the agent spawned
    could drive the operator's terminal API — read the chat, rewrite the safety
    configuration, repoint a panel proxy. This middleware is the one place that
    stops it, which is why it is structural rather than a dependency each route
    declares: a route added without the dependency would be open, and "someone
    forgets" is not an acceptable failure mode for the only thing standing
    between a sandboxed process and the console.

    **Pure ASGI, not** :class:`~starlette.middleware.base.BaseHTTPMiddleware`.
    That base class dispatches ``http`` scopes only and passes everything else
    through untouched, so both terminal websockets — the ones that carry
    keystrokes into a shell — would complete their handshake behind a gate that
    believes it is refusing them.

    Three credentials are accepted, and **exactly one is checked per request**,
    chosen by which the caller presented rather than by which one verifies:

    * ``X-Osprey-Terminal-Secret`` → the operator secret. Full access.
    * the session cookie (:func:`session_cookie_name`) → a live browser session.
      Full access.
    * ``Authorization: Bearer`` → the panel token. Allowed only where
      :func:`osprey.interfaces.web_auth.classify` returns :data:`Tier.PANEL`.

    Ahead of all three sits the **token exchange**: a ``?token=`` on a **GET**
    to a :data:`TOKEN_EXCHANGE_PATHS` page carries the operator secret,
    presented once by a cookie-less browser following the login URL. The gate
    answers it itself — ``303`` to the token-stripped URL with a session cookie
    attached — rather than admitting it to a handler. Doing it here is what
    makes the login URL work on *every* interface that installs this gate
    instead of only the one that remembered to write an exchange route, and it
    means the secret leaves the address bar on the first response, before the
    page renders and before any subresource can carry it in a ``Referer``.
    Scoped tightly (GET + exchange paths only) because a secret in a URL leaks,
    so it can never authenticate a mutating request or an arbitrary route.

    **The Origin check applies to every credential, not only the cookie.** It
    was once scoped to the cookie path on the reasoning that no browser sends
    the operator header — which is false in the multi-user shape, where nginx
    injects that header on every authenticated browser request, so scoping the
    check to cookies switched app-level CSRF off for exactly the deployment that
    has real users. A request that carries an ``Origin`` at all is a browser
    request whatever else it carries, so a present-and-foreign ``Origin`` on a
    mutating or websocket request is refused regardless of credential.
    Non-browser callers (a probe, a script, an in-process companion) send no
    ``Origin`` and are unaffected; the cookie path additionally refuses an
    *absent* ``Origin`` that no ``Sec-Fetch-Site: same-origin`` vouches for,
    because a cookie is the one credential a browser attaches to a request the
    operator did not intend to make.

    Checking only the presented credential is what keeps the cost of a request
    at one :func:`secrets.compare_digest` and no I/O. It also means a caller
    that sends a *wrong* operator header is refused rather than falling through
    to its cookie — deliberate: falling through would make the number of
    comparisons, and so the time, depend on how many credentials were wrong.

    Refusals are shaped per protocol: HTTP answers ``401``/``403`` with a JSON
    ``{"detail": ...}`` — or, when the caller's ``Accept`` asks for HTML (a
    browser *navigating*, rather than a script fetching), a small HTML page
    saying the same thing, so a single-user operator whose session expired lands
    on a readable explanation instead of raw JSON. A websocket is refused **at
    the handshake** with the JSON status and body through the
    ``websocket.http.response`` ASGI extension, so the client sees an HTTP error
    it can read instead of a socket that opens and immediately closes.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        cookie_name: str | None = None,
        external_origin: str | None = None,
    ) -> None:
        """Wrap ``app``.

        Args:
            app: The next ASGI application in the stack.
            cookie_name: The browser session cookie's name — both the one the
                gate looks for and the one the token exchange sets. Defaults to
                deriving it per request from :data:`WEB_PORT_ENV`, because an
                app is often constructed before the port it will be served on
                has settled; pass a value to pin it (a companion app that must
                accept the hub's cookie, or a test).
            external_origin: The origin browsers reach this app at. Defaults to
                reading :data:`EXTERNAL_ORIGIN_ENV` per request, falling back to
                the request's own ``Host``.
        """
        self.app = app
        self._cookie_name = cookie_name
        self._external_origin = external_origin

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Authenticate one connection, answer its token exchange, or refuse it."""
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        path = scope.get("path") or ""
        exempt = is_exempt_path(path)

        # Read before the exempt short-circuit, because one exchange page
        # (``/static/session.html``) is itself exempt: a browser with no cookie
        # has to be able to reach it, and it is also where a login URL may point.
        # Skipping the token there would leave the secret in the address bar of
        # the very page whose job is to trade it for a cookie.
        query_token = _exchange_query_token(scope)
        if exempt and query_token is None:
            await self.app(scope, receive, send)
            return

        headers = _scope_headers(scope)
        try:
            credentials = get_web_credentials(scope.get("app"))
        except RuntimeError:
            # The container shape declared a bind host but supplied no operator
            # secret, so the holder cannot populate and re-raises on every call.
            # This already fails closed — the raise is before any downstream
            # call — but letting it surface as an unhandled ASGI 500 makes the
            # one refusal that means "this gate is not operational" the one
            # refusal shaped like a crash. A 503 says it plainly, and matches
            # the auth sidecar's unconfigured answer.
            if exempt:
                # An exempt page needs no credential, so an unpopulatable holder
                # must not take it down; the exchange simply does not happen and
                # the page renders as it would have with no token at all.
                await self.app(scope, receive, send)
                return
            logger.error(
                "web credentials are unavailable; refusing all authentication "
                "(the deployment declared a bind host but supplied no operator secret)",
            )
            await self._refuse(scope, send, 503, _UNAVAILABLE_DETAIL, headers)
            return

        # A cookie-less browser following a login URL presents the operator
        # secret as ``?token=`` on a GET to an exchange path. It is judged on
        # that token alone — refused if wrong rather than falling through to
        # another credential, the same deny-path discipline the header secret
        # below follows — and, when valid, answered here with the session cookie
        # and a redirect that drops the token. This runs BEFORE the credential
        # ladder so a valid login URL always gets the secret out of the address
        # bar, even on a request that some other credential would also have
        # admitted. Scoped to GET + exchange paths by :func:`_exchange_query_token`,
        # so a URL-borne secret can never authenticate a mutating request or an
        # arbitrary route.
        if query_token is not None:
            if credentials.verify_operator(query_token):
                await self._exchange(scope, send, credentials, headers)
            elif exempt:
                # No credential is needed here anyway: decline the exchange and
                # let the page render, rather than inventing a refusal for a
                # route that is open by design.
                await self.app(scope, receive, send)
            else:
                await self._refuse(scope, send, 401, _INVALID_DETAIL, headers)
            return

        secret = (headers.get(OPERATOR_SECRET_HEADER) or "").strip()
        if secret:
            if not credentials.verify_operator(secret):
                await self._refuse(scope, send, 401, _INVALID_DETAIL, headers)
                return
            if self._origin_refused(scope, headers, strict=False):
                await self._refuse(scope, send, 403, _ORIGIN_DETAIL, headers)
                return
            await self.app(scope, receive, send)
            return

        cookie_name = self._cookie_name or session_cookie_name()
        session_ids = read_cookies(headers.get("cookie"), cookie_name)
        if session_ids:
            await self._authenticate_session(
                scope, receive, send, credentials, headers, session_ids
            )
            return

        token = _read_bearer(headers.get("authorization"))
        if token is not None:
            await self._authenticate_panel(scope, receive, send, credentials, headers, token)
            return

        await self._refuse(scope, send, 401, _MISSING_DETAIL, headers)

    async def _authenticate_session(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        credentials: WebCredentials,
        headers: dict[str, str],
        session_ids: list[str],
    ) -> None:
        """Admit a cookie-bearing request, subject to the strict Origin check.

        ``session_ids`` is every value the ``Cookie`` header carried under the
        session name, bounded by :data:`MAX_SESSION_COOKIE_CANDIDATES`; any one
        of them being live admits the request, so a shadowing cookie set by a
        sibling host cannot lock the operator out. The check is *strict* here —
        an absent ``Origin`` still has to be vouched for by ``Sec-Fetch-Site`` —
        because the cookie is the credential a browser attaches on its own.
        """
        if not any(credentials.verify_session(session_id) for session_id in session_ids):
            await self._refuse(scope, send, 401, _INVALID_DETAIL, headers)
            return
        if self._origin_refused(scope, headers, strict=True):
            await self._refuse(scope, send, 403, _ORIGIN_DETAIL, headers)
            return
        await self.app(scope, receive, send)

    async def _authenticate_panel(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        credentials: WebCredentials,
        headers: dict[str, str],
        token: str,
    ) -> None:
        """Admit a panel-token request, but only to a panel-tier HTTP route.

        No websocket is ever panel-tier, and that is enforced here rather than
        left to the route table. A websocket carries a live channel — the
        terminal's sockets carry keystrokes into a shell — so the panel token,
        which exists for the narrow set of arrangement calls an in-process
        companion makes, must never open one. Deciding this structurally means a
        future ``("GET", "/api/...")`` panel-tier route cannot accidentally
        become reachable over a websocket handshake that synthesises ``GET``.
        """
        if not credentials.verify_panel(token):
            await self._refuse(scope, send, 401, _INVALID_DETAIL, headers)
            return

        if scope["type"] == "websocket":
            await self._refuse(scope, send, 401, _TIER_DETAIL, headers)
            return

        if self._origin_refused(scope, headers, strict=False):
            await self._refuse(scope, send, 403, _ORIGIN_DETAIL, headers)
            return

        method = scope.get("method") or ""
        path = scope.get("path") or ""

        body_has_url = True
        if (method.upper(), path) == ("POST", "/api/panels/register"):
            body, receive = await _peek_body(receive)
            body_has_url = _body_has_url(body)

        if classify(method, path, body_has_url) is not Tier.PANEL:
            await self._refuse(scope, send, 401, _TIER_DETAIL, headers)
            return
        await self.app(scope, receive, send)

    def _origin_refused(self, scope: Scope, headers: dict[str, str], *, strict: bool) -> bool:
        """Whether this request must be refused for its ``Origin``.

        Called from :meth:`__call__` for every credential rather than from the
        cookie path alone. ``strict`` is the one thing that differs between
        them:

        * ``strict=False`` (operator header, panel token) — an ``Origin`` that
          is present and foreign is refused; an absent one is not. Those two
          credentials are also sent by non-browser callers, which send no
          ``Origin`` at all, so demanding one would refuse every script and
          probe. In the multi-user shape the operator header arrives on genuine
          browser requests too — nginx injects it — and those *do* carry an
          ``Origin``, which is exactly the case this catches.
        * ``strict=True`` (session cookie) — an absent ``Origin`` must still be
          vouched for by ``Sec-Fetch-Site: same-origin``, because the cookie is
          attached by the browser whether or not the operator meant to send it.
        """
        if not self._origin_check_applies(scope):
            return False
        if "origin" not in headers and not strict:
            return False
        if self._origin_matches(scope, headers):
            return False
        self._log_origin_refusal(scope, headers)
        return True

    def _log_origin_refusal(self, scope: Scope, headers: dict[str, str]) -> None:
        """Say, once, which two origins disagreed and on what request.

        This is the only refusal whose cause lives in deployment configuration
        rather than in what the caller sent, and without a line here it is
        indistinguishable from a working deployment: the page loads (safe
        methods are exempt from the check), every write is refused, and the
        JSON detail says only that the request was cross-origin. Neither origin
        is a credential and neither is echoed to the browser, so both can be
        named. No rate limiting: this fires only on requests already refused.
        """
        origin = headers.get("origin")
        received = (
            repr(origin)
            if origin is not None
            else f"absent (Sec-Fetch-Site: {headers.get('sec-fetch-site') or 'absent'})"
        )
        method = WEBSOCKET_METHOD if scope["type"] == "websocket" else (scope.get("method") or "?")
        logger.warning(
            "Refusing %s %s: Origin %s does not match this app's own origin %r. "
            "If the two differ only in how this deployment is addressed, %s is "
            "what the app was told to expect.",
            method,
            scope.get("path") or "?",
            received,
            self._resolve_external_origin(scope, headers),
            EXTERNAL_ORIGIN_ENV,
        )

    def _origin_check_applies(self, scope: Scope) -> bool:
        """Whether this request is of a shape whose Origin needs checking.

        Websockets always, because a websocket handshake is a cross-site request
        a page can make with the operator's cookie attached and then talk
        through. Otherwise only the unsafe methods: a cross-site ``GET`` cannot
        be read back by the page that made it.
        """
        if scope["type"] == "websocket":
            return True
        return (scope.get("method") or "").upper() not in SAFE_METHODS

    def _origin_matches(self, scope: Scope, headers: dict[str, str]) -> bool:
        """Whether the request's Origin is this app's own.

        The comparison is whole-string, case-insensitive, and made after both
        sides are put in the browser's own serialization by
        :func:`_normalize_origin` — an origin's scheme and host are
        case-insensitive, and a scheme's default port is not part of how a
        browser writes one. Nothing else about it may differ. Never a prefix
        match: ``https://osprey.example.com.attacker.test`` starts with the real
        origin.

        With no ``Origin`` header the decision falls to ``Sec-Fetch-Site``,
        where only ``same-origin`` passes. When neither header is present the
        answer is no: a browser sends one or both on every request this check
        applies to, so their joint absence means the cookie is being replayed by
        something that is not the browser it was issued to.
        """
        expected = self._resolve_external_origin(scope, headers)
        origin = headers.get("origin")
        if origin is None:
            return headers.get("sec-fetch-site", "").strip().lower() == "same-origin"
        return bool(expected.strip()) and _normalize_origin(origin) == _normalize_origin(expected)

    def _resolve_external_origin(self, scope: Scope, headers: dict[str, str]) -> str:
        """The origin a browser reaches this app at.

        :data:`EXTERNAL_ORIGIN_ENV` wins where it is set, which is the reverse
        proxy shape: nginx terminates ``https://host/u/<user>`` and forwards to
        a container whose own ``Host`` is a service name no browser has ever
        seen, so deriving the origin there would refuse every real request.

        Otherwise it is derived from the request's own ``Host`` — correct for
        the single-user shape, where the browser and the app agree about the
        address because there is nothing in between. Deriving it per request
        rather than pinning one at startup is what lets the same process answer
        on ``localhost`` and ``127.0.0.1`` without being told both.

        An empty answer (no configured origin, no ``Host``) refuses every
        request carrying an ``Origin``, which is the fail-closed direction.
        """
        configured = self._external_origin
        if configured is None:
            configured = os.environ.get(EXTERNAL_ORIGIN_ENV, "")
        configured = configured.strip()
        if configured:
            return configured
        host = headers.get("host", "").strip()
        if not host:
            return ""
        scheme = scope.get("scheme") or "http"
        return f"{_ORIGIN_SCHEME.get(scheme, scheme)}://{host}"

    async def _exchange(
        self,
        scope: Scope,
        send: Send,
        credentials: WebCredentials,
        headers: dict[str, str],
    ) -> None:
        """Trade a verified ``?token=`` for a session cookie and redirect.

        Answered by the gate itself, not passed downstream: the exchange is the
        same on every interface, and a handler-level implementation is one each
        app has to remember to write. Six properties are load-bearing:

        * **303, not 302.** The exchange is a GET, and 303 states plainly that
          the follow-up is a GET too.
        * **The token is dropped from the redirect target**, and only the token
          — a sibling query parameter the login URL carried survives, so a
          deep link still lands where it meant to. The secret leaves the address
          bar, browser history and every later ``Referer`` on this first
          response, before the page renders and before it can fetch a
          subresource.
        * **The multi-user prefix is put back on.** nginx strips ``/u/<user>``
          before proxying, so the path this app sees is bare and a relative
          ``Location`` built from it would send the browser to the front door's
          root. See :func:`compute_url_prefix`.
        * **``HttpOnly``, ``SameSite=Lax``, ``Path=/``.** ``Lax`` rather than
          ``Strict`` because the login link is a top-level cross-site
          navigation and ``Strict`` would withhold the cookie that was just
          set; the Origin check, not ``SameSite``, is what refuses forged
          cross-site requests.
        * **``Secure`` is derived, never assumed.** See :meth:`_cookie_is_secure`.
        * **``Max-Age`` is the configured session lifetime**, which makes the
          cookie persistent rather than a browser-session cookie that dies with
          the window: it survives a browser restart, and an operator who closes
          the control-room browser is not logged out by that alone. Stating the
          same number the server holds keeps the two expiries from disagreeing —
          a longer cookie would be presented after the server had already
          forgotten the session, a shorter one would drop a session still good.
          The browser's copy is a convenience, not the authority: the server
          refuses the session after its own deadline either way.
        """
        session_id = credentials.create_session()
        cookie = self._session_cookie(scope, headers, session_id, credentials.session_ttl_seconds)
        location = self._exchange_location(scope)
        await send(
            {
                "type": "http.response.start",
                "status": 303,
                "headers": [
                    (b"location", location.encode("latin-1")),
                    (b"content-length", b"0"),
                    (b"set-cookie", cookie.encode("latin-1")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": b""})

    def _exchange_location(self, scope: Scope) -> str:
        """The token-stripped, prefix-restored URL the exchange redirects to."""
        path = scope.get("path") or "/"
        raw = (scope.get("query_string") or b"").decode("latin-1")
        # ``keep_blank_values`` so ``?theme=`` survives as itself rather than
        # vanishing; every ``token`` key is dropped, including a duplicate.
        remaining = [
            (key, value) for key, value in parse_qsl(raw, keep_blank_values=True) if key != "token"
        ]
        location = apply_url_prefix(compute_url_prefix(), path)
        if remaining:
            location = f"{location}?{urlencode(remaining)}"
        return location

    def _session_cookie(
        self, scope: Scope, headers: dict[str, str], session_id: str, max_age: int
    ) -> str:
        """Render the ``Set-Cookie`` value the exchange hands the browser.

        ``max_age`` is the session lifetime in seconds the server will honour,
        written out as ``Max-Age`` so the browser expires the cookie on the same
        schedule the server expires the session.
        """
        name = self._cookie_name or session_cookie_name()
        attributes = [
            f"{name}={session_id}",
            "Path=/",
            "HttpOnly",
            "SameSite=Lax",
            f"Max-Age={max_age}",
        ]
        if self._cookie_is_secure(scope, headers):
            attributes.append("Secure")
        return "; ".join(attributes)

    def _cookie_is_secure(self, scope: Scope, headers: dict[str, str]) -> bool:
        """Whether the session cookie may be marked ``Secure``.

        Derived from the origin the *browser* reaches this app at — the same
        answer :meth:`_resolve_external_origin` gives the Origin check — rather
        than hard-coded either way, because both hard-codings are wrong for one
        of the two shapes. Always setting it would make the browser refuse the
        cookie over the single-user loopback ``http://`` origin, breaking the
        exchange the instant it is set; never setting it leaves a real
        deployment's session cookie eligible to be sent over plain HTTP, which
        is the one thing ``Secure`` exists to prevent.

        In the multi-user shape nginx terminates TLS and forwards over plain
        HTTP, so ``scope["scheme"]`` says ``http`` and only
        :data:`EXTERNAL_ORIGIN_ENV` knows the truth — which is exactly what
        :meth:`_resolve_external_origin` consults first.
        """
        return self._resolve_external_origin(scope, headers).strip().lower().startswith("https://")

    def _audit_refusal(
        self, scope: Scope, status: int, detail: str, headers: dict[str, str]
    ) -> None:
        """File one ``401``/``403`` in the unified ledger.

        The record's *actor* is this container's own audit identity, filled in
        by the writer from :func:`~osprey.utils.identity.acting_identity` (this
        layer names none) — never the forwarded account. The two are different
        questions: the actor says which deployment identity's ledger this
        belongs in (and which file it can be written to), while the account is
        a claim that arrived on the request and is recorded as such, inside
        ``detail`` — with the provider's subject beside it as ``oidc_subject=``
        only where it differs. Letting a header choose the actor would let a
        caller file records under somebody else's name.

        ``session`` is ``None`` and stays that way: no posture-store key exists
        for a request that never reached a session, and inventing one would
        join this record to a session that never existed.

        Every line of this — the status filter, the header decode, the detail
        assembly — runs inside :func:`_emit_audit_record`'s never-raises
        boundary, because a refusal this method could not describe must still
        be a refusal the caller receives.
        """

        def build() -> dict[str, Any] | None:
            if status not in AUDITED_REFUSAL_STATUSES:
                return None
            record_detail, role = _audit_detail(scope, headers)
            return {
                "surface": WEB_AUTH_SURFACE,
                "posture": WEB_AUTH_POSTURE,
                "posture_source": POSTURE_SOURCE_APP,
                "session": None,
                "subject": _audit_subject(scope),
                "decision": DECISION_REFUSED,
                "reason": REFUSAL_REASONS.get(detail, REASON_UNCATEGORIZED),
                "detail": record_detail or None,
                "role": role,
            }

        _emit_audit_record(build)

    async def _refuse(
        self,
        scope: Scope,
        send: Send,
        status: int,
        detail: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Answer one refused connection, in the shape its protocol expects.

        A websocket is refused with a real HTTP response where the server offers
        the ``websocket.http.response`` extension — uvicorn's does — because a
        client that is told ``401`` can report an expired session, while one
        that is told only "closed" cannot tell that from a crash. The
        :data:`WEBSOCKET_REFUSAL_CODE` close is the fallback for a server that
        does not advertise the extension, and it never accepts the handshake
        first: an accepted socket is a socket that briefly existed.

        An HTTP refusal is JSON unless the caller's ``Accept`` asks for HTML,
        which distinguishes a browser *navigating* to a page from a script
        fetching data. That distinction is what stops a single-user operator
        whose session expired from being shown raw JSON: there is no perimeter
        in that shape to redirect them to a login page, so this gate is the only
        thing that can explain what happened. The status is identical either
        way, and so is the meaning — only the rendering differs, and no
        machine-readable client sees the change (``fetch``/``XHR`` default to
        ``Accept: */*``).

        **Every refusal is filed here**, before the answer is written, because
        this is the one place all of them converge: eight call sites across the
        credential ladder, the token exchange and the tier check reach it, and
        an emitter at any of them would be one somebody forgets to add to the
        ninth. Only :data:`AUDITED_REFUSAL_STATUSES` are filed — a ``503`` is
        the gate reporting its own unavailability, not a decision about this
        caller.
        """
        self._audit_refusal(scope, status, detail, headers or {})
        wants_html = (
            scope["type"] == "http" and "text/html" in (headers or {}).get("accept", "").lower()
        )
        if wants_html:
            payload = _refusal_page(status, detail).encode("utf-8")
            content_type = b"text/html; charset=utf-8"
        else:
            payload = json.dumps({"detail": detail}).encode("utf-8")
            content_type = b"application/json"
        response_headers = [
            (b"content-type", content_type),
            (b"content-length", str(len(payload)).encode("ascii")),
        ]
        if scope["type"] == "websocket":
            extensions = scope.get("extensions") or {}
            if "websocket.http.response" in extensions:
                await send(
                    {
                        "type": "websocket.http.response.start",
                        "status": status,
                        "headers": response_headers,
                    }
                )
                await send({"type": "websocket.http.response.body", "body": payload})
            else:
                await send({"type": "websocket.close", "code": WEBSOCKET_REFUSAL_CODE})
            return
        await send({"type": "http.response.start", "status": status, "headers": response_headers})
        await send({"type": "http.response.body", "body": payload})


class HttpAuditMiddleware:
    """Pure-ASGI layer recording every state-changing request that got through.

    :class:`WebAuthMiddleware` records what the gate turned away. This records
    the other half — what it let in — and it is the half an operator actually
    reconstructs an incident from: a refused request changed nothing, while an
    admitted ``POST /api/config`` changed the safety configuration of a running
    terminal and leaves no other trace that it was this account which did it.

    **Innermost, by installation order.** ``configure_interface_app`` adds this
    first, which makes it the last layer before the router. Two consequences,
    both wanted: it never sees a request the gate refused (so a refusal is
    recorded once, by the gate, and never twice), and the status it records is
    the one the route actually produced rather than one a later layer rewrote.

    **The discriminator is** :data:`SAFE_METHODS`, and nothing else. Method is
    the one property of a request that is knowable before the route is chosen
    and that says whether state can change; a path allowlist would need an
    entry per interface and would be wrong the day a route was added. ``GET``,
    ``HEAD`` and ``OPTIONS`` pass through with no wrapper at all, which matters
    because they are the whole static-asset path.

    **Landing in every interface app is the point.** The terminal, ARIEL,
    channel finder, health, artifacts and the lattice dashboard all call
    ``configure_interface_app``, so a mutation route added to any of them is
    audited the day it is written, without its author knowing this exists.

    What it deliberately does not do is decide anything: it never refuses, and
    it never inspects a body. It is an observer with a byte of state.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Wrap ``app``.

        Args:
            app: The next ASGI application in the stack — in practice the
                router, since this layer is installed innermost.
        """
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Pass one connection through, recording it if it can change state.

        A safe method, a websocket and a lifespan message are handed straight
        on: not wrapping them is what keeps this layer off the hot path that
        serves every CSS file.

        The record is written in a ``finally``, so a route that raises before
        answering is still recorded — it was admitted, it ran, and whether it
        finished is exactly what the ledger should be able to say. The
        exception itself is not swallowed; ``ExceptionLoggingMiddleware`` sits
        outside this layer and still turns it into the 500 it always did.

        The mutation runs inside one :func:`~osprey.audit.dedup.decision_scope`,
        and the marker is read *inside* it — in the same ``finally``, which sits
        within the ``with`` block for exactly that reason. The scope resets the
        marker on the way out, so a read one line later would always see
        ``None`` and this layer would record over every inner decision. See
        :meth:`_record` for what it does with the answer.
        """
        if scope.get("type") != "http" or (scope.get("method") or "").upper() in SAFE_METHODS:
            await self.app(scope, receive, send)
            return

        # Imported here, not at module scope, for the reason `_emit_audit_record`
        # gives: `osprey.audit.dedup` pulls the writer in with it, and this module
        # sits in the import closure of every interface app and of the auth
        # sidecar. A `sys.modules` hit per mutation, and none at all for the safe
        # methods that returned above.
        from osprey.audit.dedup import decision_scope, recorded_decision

        status: list[int] = []

        async def audited_send(message: Message) -> None:
            if message["type"] == "http.response.start":
                status.append(message["status"])
            await send(message)

        with decision_scope():
            try:
                await self.app(scope, receive, audited_send)
            finally:
                self._record(
                    scope,
                    status[0] if status else None,
                    inner=recorded_decision(),
                )

    def _record(
        self, scope: Scope, status: int | None, *, inner: "RecordedDecision | None" = None
    ) -> None:
        """File one admitted mutation, under the decision the response implies.

        A ``4xx``/``5xx`` is recorded as a refusal with
        :data:`REASON_ROUTE_REFUSED`, not as an admission: the gate admitted
        the request but something inside declined it, and a layer that stamped
        everything it admitted ``allowed`` would overwrite the inner decision
        with a less true one. That is the same invariant
        :mod:`osprey.audit.dedup` generalises to the cases a status cannot
        show — a guard that refuses *and* returns a successful response — via
        a marker the inner recorder sets and the outer layer defers to.

        **The pairing is wired, not merely described.** *inner* is the marker
        an inner recorder left, read by :meth:`__call__` inside the scope it
        opened. When it is set, this layer files nothing: the route decided,
        and its record is the truer one. No route in any interface app writes
        to the unified ledger *today* — so *inner* is always ``None`` on the
        running path — but the mechanism is live rather than a recipe, because
        the recipe's one plausible misreading (read the marker after the scope
        closes) yields two records, the second of them ``allowed`` on a refused
        request, and nothing would have failed to say so.

        **An inner recorder must be on the task this layer awaits.** The
        marker is a ``ContextVar``, and Starlette runs a request's middleware
        chain and an ``async def`` route in one task, so a mark set there
        carries back to here. A ``def`` route does not qualify: Starlette runs
        it through ``run_in_threadpool``, which *copies* the context, and the
        mark dies with the worker thread — this layer then sees ``None`` and
        records over the route's decision. A sync route (and any recorder
        behind ``asyncio.create_task`` or ``to_thread``) must become
        ``async def``, or record through a seam outside this layer, before it
        can be an inner recorder. Note that several protected-write refusal
        helpers are sync functions; that is fine as long as every route calling
        them is ``async def``, since a sync helper called inline stays on the
        awaiting task.

        **The marker is not re-asserted outward.** ``configure_interface_app``
        installs this layer innermost, so nothing wraps it that could defer to
        it. The MCP middleware, which *can* be wrapped, does re-assert.

        A marker whose record did not land (``stored=False``) is still deferred
        to on a ``2xx``: this layer's own record would say ``allowed``, and
        ``allowed`` over a refusal is the outcome the marker exists to prevent.
        On a ``4xx``/``5xx`` it is not, because this layer's record says
        ``refused`` too — filing it turns a silence back into a duplicate-free
        single line.

        A missing status means the route raised before answering. It is
        recorded as an admission — the request was let in and ran — under
        :data:`REASON_MUTATION_UNANSWERED`, with ``status=none`` in ``detail``
        saying the same thing a second way. It is not guessed into a refusal
        the route never made, and it does not share a reason with the requests
        that succeeded.

        The whole body runs inside :func:`_emit_audit_record`'s never-raises
        boundary. That matters more here than on the gate: this method is
        called from a ``finally``, so a raise would both mask the route's own
        exception and turn a response the client has already received into a
        ``500``.
        """
        refused = status is not None and status >= 400
        if inner is not None and (inner.stored or not refused):
            # Deliberately plain ``scope`` lookups and lazy %-formatting: this
            # branch runs outside `_emit_audit_record`'s never-raises boundary,
            # from a `finally`, where anything that raised would mask the
            # route's own exception.
            logger.debug(
                "Deferring the audit record for %s %s to the route (%s/%s)",
                scope.get("method"),
                scope.get("path"),
                inner.decision,
                inner.reason,
            )
            return

        def build() -> dict[str, Any]:
            record_detail, role = _audit_detail(
                scope,
                _scope_headers(scope),
                status=str(status) if status is not None else _STATUS_NONE,
            )
            if refused:
                reason = REASON_ROUTE_REFUSED
            elif status is None:
                reason = REASON_MUTATION_UNANSWERED
            else:
                reason = REASON_MUTATION
            return {
                "surface": HTTP_MUTATION_SURFACE,
                "posture": HTTP_MUTATION_POSTURE,
                "posture_source": POSTURE_SOURCE_APP,
                "session": None,
                "subject": _audit_subject(scope),
                "decision": DECISION_REFUSED if refused else DECISION_ALLOWED,
                "reason": reason,
                "detail": record_detail or None,
                "role": role,
            }

        _emit_audit_record(build)
