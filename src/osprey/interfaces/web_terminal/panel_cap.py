"""C-CAP: capability paths that let an opaque panel load its own files.

An agent-writable panel — a discovered bundle served at ``/panel-static/<id>/``
or a runtime-registered panel proxied at ``/panel/<id>/`` — runs at an OPAQUE
origin, so its requests carry ``Origin: null`` and no SameSite=Lax session
cookie. Behind the cookie gate it could then load nothing past its first
document: not its module scripts, not a JSON file sitting next to it.

The hub therefore frames such a panel at a capability path,
``<prefix>/panel-static/<id>/~c/<cap>/…`` or ``<prefix>/panel/<id>/~c/<cap>/…``.
``cap`` is an HMAC, under a key derived from this process's operator secret,
over the login that asked for the frame, the panel id and an expiry at most
:data:`CAP_TTL_SECONDS` away. A request under a cap path is admitted by the cap
instead of the cookie (the auth gate and nginx's ``auth_request`` exempt
exactly these paths, see
:func:`osprey.interfaces.common_middleware.is_panel_cap_path`), and the route
that answers it reads only that panel's bytes, or relays only to that panel's
backend. Relative URLs inside the bundle inherit the cap, so multi-file bundles
keep working; an absolute console path does not, and stays behind the gate.

A cap is refused (``403``) when it is forged, names another panel, has expired
or promises a lifetime longer than :data:`CAP_TTL_SECONDS`, or was minted for a
login this process no longer holds: a browser session that has expired or been
revoked, or a login of another deployment (whose key differs). The hub mints a
fresh one whenever it builds the frame's URL, so a reload re-mints.

What a cap grants is deliberately small: read access to bytes the agent could
already write, at an origin that can reach nothing else. It is never a console
credential — no operator route accepts it, and the routes that do accept it
send no credential upstream.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import time
from typing import Any, Literal

from fastapi import Request
from starlette.requests import HTTPConnection

from osprey.interfaces.common_middleware import (
    OPERATOR_SECRET_HEADER,
    PANEL_CAP_SEGMENT,
    apply_url_prefix,
    compute_url_prefix,
    read_cookies,
    session_cookie_name,
)
from osprey.interfaces.web_auth import WebCredentials, _digest, get_web_credentials
from osprey.profiles.web_panels import BUILTIN_PANELS, SIDECAR_PANELS
from osprey.registry.web import FRAMEWORK_WEB_SERVERS

__all__ = [
    "CAP_TTL_SECONDS",
    "OPERATOR_LOGIN",
    "PanelKind",
    "cap_url",
    "login_for_request",
    "mint_cap",
    "opaque_panel_kind",
    "verify_cap",
    "verify_request_cap",
]

#: The longest a cap may live. The contract's ceiling: a frame left open longer
#: than this is re-minted by a reload.
CAP_TTL_SECONDS = 12 * 60 * 60

#: The login a request admitted by the operator secret is minted for. That is
#: the shape behind the multi-user perimeter, where nginx injects the per-user
#: secret on every authenticated browser request and strips the cookie; each
#: user's container holds its own secret, so a cap minted in one user's
#: container fails the HMAC in every other.
OPERATOR_LOGIN = "operator"

#: Which serving route a panel's cap path belongs to.
PanelKind = Literal["static", "proxy"]

_KEY_LABEL = b"osprey-panel-cap/v1"
_LOGIN_TAG_BYTES = 12
_SIG_BYTES = 24

#: Panel ids the hub itself serves or launches: never agent-writable, never capped.
_FIRST_PARTY_IDS = frozenset(
    {*BUILTIN_PANELS, *SIDECAR_PANELS}
    | {definition.panel_id for definition in FRAMEWORK_WEB_SERVERS.values()}
)


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _key(credentials: WebCredentials) -> bytes:
    """The cap key: derived from the operator secret, never the secret itself."""
    return hmac.new(
        credentials.operator_secret.encode("utf-8"), _KEY_LABEL, hashlib.sha256
    ).digest()


def _login_tag(key: bytes, login: str) -> str:
    mac = hmac.new(key, b"login\x00" + login.encode("utf-8"), hashlib.sha256).digest()
    return _b64(mac[:_LOGIN_TAG_BYTES])


def _signature(key: bytes, panel_id: str, tag: str, expiry: int) -> str:
    message = f"cap\x00{panel_id}\x00{tag}\x00{expiry}".encode()
    return _b64(hmac.new(key, message, hashlib.sha256).digest()[:_SIG_BYTES])


def mint_cap(
    credentials: WebCredentials,
    login: str,
    panel_id: str,
    *,
    now: float | None = None,
    ttl: float = CAP_TTL_SECONDS,
) -> str:
    """Mint the cap segment for ``login`` reading ``panel_id``.

    Args:
        credentials: This process's credential holder (the key's source).
        login: :data:`OPERATOR_LOGIN`, or the session digest of a browser login.
        panel_id: The one panel this cap reads.
        now: The clock to mint against; the current time when omitted.
        ttl: Seconds until the cap expires. A test may pass a negative value to
            mint an already-expired cap; the verifier refuses any cap whose
            expiry lies more than :data:`CAP_TTL_SECONDS` ahead.

    Returns:
        ``<expiry-hex>.<login-tag>.<signature>`` — URL-safe, one path segment.
    """
    key = _key(credentials)
    expiry = int((time.time() if now is None else now) + ttl)
    tag = _login_tag(key, login)
    return f"{expiry:x}.{tag}.{_signature(key, panel_id, tag, expiry)}"


def _login_is_live(credentials: WebCredentials, key: bytes, tag: str, now: float) -> bool:
    """Whether ``tag`` names a login this process still holds."""
    if hmac.compare_digest(tag, _login_tag(key, OPERATOR_LOGIN)):
        return True
    # A copy: the holder's map is mutated under its own lock by other threads.
    sessions = dict(credentials.sessions)
    live = False
    for digest, deadline in sessions.items():
        if deadline > now and hmac.compare_digest(tag, _login_tag(key, digest)):
            live = True
    return live


def verify_cap(
    credentials: WebCredentials, cap: str, panel_id: str, *, now: float | None = None
) -> bool:
    """Whether ``cap`` admits a read of ``panel_id`` now.

    False for a malformed or forged cap, one minted for another panel, one
    past its expiry or promising more than :data:`CAP_TTL_SECONDS`, and one
    whose login this process no longer holds.
    """
    now = time.time() if now is None else now
    parts = cap.split(".")
    if len(parts) != 3:
        return False
    expiry_hex, tag, signature = parts
    try:
        expiry = int(expiry_hex, 16)
    except ValueError:
        return False
    if f"{expiry:x}" != expiry_hex:
        return False
    key = _key(credentials)
    if not secrets.compare_digest(
        signature.encode("ascii", "replace"),
        _signature(key, panel_id, tag, expiry).encode("ascii"),
    ):
        return False
    if expiry <= now or expiry - now > CAP_TTL_SECONDS:
        return False
    return _login_is_live(credentials, key, tag, now)


def login_for_request(request: Request) -> str | None:
    """The login an authenticated request was made by, or ``None``.

    The gate has already admitted the request; this names which credential did,
    the same way the gate reads it: the operator secret header first, then the
    session cookie. A panel-token caller (an in-process companion) and a
    process with no credentials name no login, and get no cap.
    """
    try:
        credentials = get_web_credentials(request.app)
    except RuntimeError:
        return None
    secret = (request.headers.get(OPERATOR_SECRET_HEADER) or "").strip()
    if secret:
        return OPERATOR_LOGIN if credentials.identify_operator(secret) is not None else None
    for session_id in read_cookies(request.headers.get("cookie"), session_cookie_name()):
        if credentials.verify_session(session_id):
            return _digest(session_id)
    return None


def _custom_panel(request: HTTPConnection, panel_id: str) -> dict[str, Any] | None:
    for cp in getattr(request.app.state, "custom_panels", []):
        if isinstance(cp, dict) and cp.get("id") == panel_id:
            return cp
    return None


def opaque_panel_kind(request: HTTPConnection, panel_id: str) -> PanelKind | None:
    """Whether ``panel_id`` is an opaque panel, and which route serves it.

    Decided by where the bytes come from, never by a flag the panel declares:
    a discovered bundle (``"static"``) or a runtime registration (``"proxy"``).
    Built-in, framework and config-defined panels are ``None`` — they are the
    deployment's own and are never capped.
    """
    if panel_id in _FIRST_PARTY_IDS:
        return None
    cp = _custom_panel(request, panel_id)
    if cp is None or cp.get("configDefined"):
        return None
    if cp.get("discovered"):
        dirs = getattr(request.app.state, "discovered_panel_dirs", {}) or {}
        return "static" if panel_id in dirs else None
    return "proxy"


def cap_url(request: Request, panel_id: str) -> str | None:
    """The browser URL an opaque panel is framed at, minted for this request's login.

    Ends in ``/`` so the bundle's relative URLs resolve under the cap. ``None``
    for a panel that is not opaque or a request that names no login.
    """
    kind = opaque_panel_kind(request, panel_id)
    if kind is None:
        return None
    login = login_for_request(request)
    if login is None:
        return None
    cap = mint_cap(get_web_credentials(request.app), login, panel_id)
    root = "panel-static" if kind == "static" else "panel"
    return apply_url_prefix(compute_url_prefix(), f"/{root}/{panel_id}/{PANEL_CAP_SEGMENT}/{cap}/")


def verify_request_cap(conn: HTTPConnection, panel_id: str, cap: str, kind: PanelKind) -> bool:
    """Whether a cap-path request may be served: the panel is ``kind`` and the cap valid.

    Args:
        conn: The request or websocket under the cap path.
        panel_id: The panel id from the path.
        cap: The cap segment from the path.
        kind: The route answering — ``"static"`` for ``/panel-static``,
            ``"proxy"`` for ``/panel``. A cap never crosses from one to the other.
    """
    if opaque_panel_kind(conn, panel_id) != kind:
        return False
    try:
        credentials = get_web_credentials(conn.app)
    except RuntimeError:
        return False
    return verify_cap(credentials, cap, panel_id)
