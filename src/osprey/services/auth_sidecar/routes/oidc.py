"""OIDC login and callback for the auth sidecar.

Two routes under the public ``/auth/`` prefix nginx proxies here:
:data:`LOGIN_PATH` starts the handshake for one roster user and
:data:`CALLBACK_PATH` finishes it. Both exist only when the deployment runs
``auth.method: oidc``; in password mode they answer 404, because the factory
includes every route module regardless of method and "this endpoint is not part
of this deployment" is the honest answer.

**The clicked card is the username.** The landing page offers one card per
roster user, so the login route is addressed per user and the callback's job is
not "who is this person" but "is this person the user whose card was clicked".
The identity the IdP asserts is compared against
:meth:`~osprey.services.auth_sidecar.app.AuthSettings.oidc_subject` for *that*
user and nothing else: an identity that maps to no roster user, or to a
different one, is refused with 403. Searching the roster for whichever user the
subject happens to match would turn a mis-click into someone else's terminal,
which is exactly the isolation this sidecar exists to establish.

**Which user was clicked is server-side state.** It travels in the pinned
Starlette session cookie (:data:`PENDING_FLOW_SESSION_KEY`) alongside Authlib's
own ``state``/``nonce``, keyed by the same ``state`` value — never as a query
parameter on the ``redirect_uri``, which the browser could edit between the two
legs of the handshake and thereby pick which roster user a successful IdP login
unlocks. The session cookie is signed with the state secret, so its contents are
the sidecar's own word.

**The identity comes from the verified ID token.** Authlib validates the token's
signature, issuer, audience and nonce before
:meth:`authorize_access_token` returns it as ``token["userinfo"]``; this module
reads the configured claim from there and never calls the userinfo endpoint. The
audience is on that list only because :func:`_claims_options` asks for it —
Authlib checks ``aud`` when the caller names it and not otherwise, so removing
that argument would silently reduce the audience check to the OIDC ``azp`` rule.
A deployment whose IdP carries the mapped claim only at that endpoint therefore
locks out with a log line naming the claim, rather than authorising on a second,
weaker trust path.

**Nothing about a failure reaches the browser but its category.** Tokens,
client secrets, and the claim values examined while validating a login never
enter a response or a log line; refusals name the user and the reason class
only. The one identity value that is retained is the accepted subject: a
successful login stores it in the (signed, not encrypted) session cookie and
records it on the success log line, because it is an opaque account identifier
rather than a credential and a later verify subrequest reports it back.
"""

from __future__ import annotations

import logging
import secrets
from typing import Any

import httpx
from authlib.integrations.starlette_client import OAuth, OAuthError
from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import RedirectResponse

from ..app import AuthSettings, get_revocation_store, get_session_codec, get_settings
from ..exceptions import InvalidSessionError
from ..return_to import safe_return_to
from ..sessions import SESSION_COOKIE_NAME, SessionState

logger = logging.getLogger(__name__)

router = APIRouter()

LOGIN_PATH = "/auth/oidc/login"
"""Starts the handshake for one roster user: ``?user=<name>&next=<return-to>``."""

CALLBACK_PATH = "/auth/oidc/callback"
"""Where the IdP sends the browser back. Registered with the IdP as
:attr:`~osprey.services.auth_sidecar.app.AuthSettings.external_origin` + this
path — both legs must agree on it byte for byte, and the token exchange repeats
it, so it is derived from the deployment's own origin rather than from the
inbound request (which arrives from nginx over loopback and names the wrong
host)."""

CLIENT_NAME = "osprey_oidc"
"""Authlib client name. It namespaces Authlib's own session entries
(``_state_osprey_oidc_<state>``), so it is part of the state cookie's shape."""

PENDING_FLOW_SESSION_KEY = "osprey_oidc_pending"
"""State-cookie key holding the in-flight handshake: ``{"state", "user", "next"}``.

One entry, not a map: Authlib's Starlette integration drops every previous
``_state_*`` entry each time it stores a new one, so a browser can only ever
have one handshake it could complete. A second entry here would outlive the
Authlib data it is paired with and could only ever fail the state check."""

DEFAULT_SCOPE = "openid profile email"
"""Requested scopes.

``openid`` is what makes this OIDC rather than bare OAuth2 (it is also what
makes Authlib generate and check a nonce). ``profile`` and ``email`` are asked
for because the claim a facility maps onto a roster user is commonly
``preferred_username`` or ``email``, and a claim that was never requested is
simply absent from the token — which this module reads as "deny"."""


def _discovery_url(issuer: str) -> str:
    """The OIDC discovery document for ``issuer``."""
    return f"{issuer.rstrip('/')}/.well-known/openid-configuration"


def _require_oidc_mode(settings: AuthSettings) -> None:
    """Refuse with 404 unless this deployment authenticates with OIDC.

    Raises:
        HTTPException: 404 in password mode. Not 403: the endpoint is not part
            of this deployment's surface at all, and saying "forbidden" would
            imply an identity was evaluated.
    """
    if settings.method != "oidc":
        raise HTTPException(status_code=404, detail="this deployment does not use OIDC login")


def _oauth_client(request: Request) -> Any:
    """The app's one Authlib client, built on first use and cached.

    Cached on ``app.state`` rather than at module scope: two apps in one process
    (every test that builds a second one) would otherwise share a client
    configured for the first app's issuer and credentials. Building it lazily
    also keeps the discovery fetch off the factory's path — the sidecar starts
    and answers its healthcheck whether or not the IdP is reachable — and the
    cached client holds the fetched metadata, so discovery happens once rather
    than per login.

    Tests replace it by assigning ``app.state.oidc_client`` before the first
    request.
    """
    client = getattr(request.app.state, "oidc_client", None)
    if client is not None:
        return client

    settings = get_settings(request)
    registry = OAuth()
    registry.register(
        name=CLIENT_NAME,
        server_metadata_url=_discovery_url(settings.oidc_issuer or ""),
        client_id=settings.oidc_client_id,
        client_secret=settings.oidc_client_secret,
        client_kwargs={"scope": DEFAULT_SCOPE},
    )
    client = registry.create_client(CLIENT_NAME)
    request.app.state.oidc_client = client
    return client


def _same_value(left: str, right: str) -> bool:
    """Whether two text values are equal, compared in constant time.

    As UTF-8 bytes, never as ``str``: :func:`secrets.compare_digest` raises
    ``TypeError`` the moment either side carries a non-ASCII character, and both
    things compared here can. A ``state`` parameter is unauthenticated input, so
    one accented character in it would be an unhandled 500 rather than a
    refusal; a mapped identity like ``jörg@example.org`` would raise on the
    comparison that was about to *succeed*, locking that operator out for good.
    """
    return secrets.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


def _claims_options(settings: AuthSettings) -> dict[str, dict[str, list[str]]]:
    """Which ID-token claims Authlib must check, and against what.

    **``aud`` is only checked when it is named here.** Authlib's own default
    supplies ``iss`` and nothing else, so without this an ID token minted for a
    *different* relying party validates as long as it carries an ``azp`` naming
    this client — the OIDC ``azp`` rule is then the only thing standing in the
    way, and an IdP emits ``azp`` at its discretion. Naming ``aud`` makes the
    audience check this deployment's own rather than a side effect of a claim
    that may never arrive.

    **``iss`` is repeated here because this mapping replaces Authlib's default
    rather than extending it.** Omitting it would trade the audience hole for an
    issuer one.

    Both spellings of the issuer are accepted. OIDC Discovery requires the
    published ``issuer`` to equal the prefix its document was fetched from, and
    :func:`_discovery_url` strips one trailing slash off that prefix — so a
    deployment whose configured issuer differs from the IdP's published one by
    exactly that character is correctly configured, and locking it out would be
    this function inventing a requirement neither party has.

    Args:
        settings: The deployment's settings. Both values it reads are
            deployment-wide requirements in OIDC mode, so the configuration
            guard has already refused every request that could reach here
            without them.

    Returns:
        A ``claims_options`` mapping for
        :meth:`authorize_access_token`.
    """
    issuer = (settings.oidc_issuer or "").strip()
    trimmed = issuer.rstrip("/")
    return {
        "iss": {"values": [issuer] if issuer == trimmed else [issuer, trimmed]},
        "aud": {"values": [settings.oidc_client_id]},
    }


def _current_session(request: Request) -> SessionState:
    """The browser's auth session, or a fresh one.

    An unreadable cookie — tampered, stale-keyed, or from a retired payload
    version — is not an error here: it carries no authorisation, so the login
    it is about to be replaced by starts from an empty session.

    **A revoked session id is not inherited.** Logout revokes by id and verify
    refuses every cookie carrying a revoked one, so a callback that kept the id
    would re-sign a cookie verify then rejects — and the next handshake would
    inherit it again, locking the browser out of the terminal it just
    authenticated for until the entry expired. Retiring the id here is what makes
    a logout followed by a fresh login mean what it says.
    """
    codec = get_session_codec(request)
    raw = request.cookies.get(SESSION_COOKIE_NAME)
    if not raw:
        return codec.new_state()
    try:
        state = codec.decode(raw)
    except InvalidSessionError:
        logger.info("discarding an unreadable auth session cookie during oidc login")
        return codec.new_state()

    if get_revocation_store(request).is_revoked(state.session_id):
        logger.info("starting a new session: the cookie presented at oidc login had been revoked")
        return codec.new_state()
    return state


@router.get(LOGIN_PATH)
async def oidc_login(
    request: Request,
    user: str = Query(..., description="Roster user whose card was clicked"),
    return_to: str | None = Query(None, alias="next", description="Same-origin path to return to"),
) -> Response:
    """Redirect to the IdP to authenticate as ``user``.

    Args:
        request: The inbound request.
        user: The roster user being logged in.
        return_to: Where to send the browser after a successful callback.

    Returns:
        A redirect to the IdP's authorization endpoint.

    Raises:
        HTTPException: 404 when this deployment is not in OIDC mode or ``user``
            is not on the roster; 403 when ``user`` has no mapped IdP identity
            (the handshake could only end in a refusal, so it does not start);
            502 when the issuer's discovery document cannot be fetched or read.
    """
    settings = get_settings(request)
    _require_oidc_mode(settings)

    if not settings.knows_user(user):
        raise HTTPException(status_code=404, detail="no such user")
    if settings.oidc_subject(user) is None:
        logger.warning("oidc login refused for %r: no IdP identity is mapped to that user", user)
        raise HTTPException(status_code=403, detail="this user has no mapped identity provider")

    target = safe_return_to(return_to, user, flow="oidc login")
    client = _oauth_client(request)
    redirect_uri = f"{settings.external_origin}{CALLBACK_PATH}"

    # Deliberately not `authorize_redirect`, which is these three steps with the
    # state value kept inside it. The state is what binds this handshake to the
    # user whose card was clicked, so this route needs it in hand.
    try:
        authorization = await client.create_authorization_url(redirect_uri)
    except httpx.HTTPError:
        logger.warning("oidc login failed: the issuer's discovery document is unreachable")
        raise HTTPException(
            status_code=502, detail="the identity provider could not be reached"
        ) from None
    except Exception as exc:
        # Reached, not defensive: a discovery document that is served but wrong
        # — HTML from a captive portal, JSON with no authorization_endpoint —
        # raises out of Authlib rather than out of httpx, and a mistyped issuer
        # is where that shows up first. Same reasoning as the callback's broad
        # arm, and the same discipline: class name only, never the response.
        logger.warning(
            "oidc login failed: the issuer's discovery document is unusable (%s)",
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=502, detail="the identity provider's configuration could not be read"
        ) from None

    await client.save_authorize_data(request, redirect_uri=redirect_uri, **authorization)
    request.session[PENDING_FLOW_SESSION_KEY] = {
        "state": authorization["state"],
        "user": user,
        "next": target,
    }
    return RedirectResponse(authorization["url"], status_code=302)


@router.get(CALLBACK_PATH)
async def oidc_callback(request: Request) -> Response:
    """Finish the handshake and unlock the user whose card was clicked.

    Args:
        request: The inbound request, carrying the IdP's ``code`` and ``state``.

    Returns:
        A redirect to the validated return-to, carrying the re-issued auth
        session cookie.

    Raises:
        HTTPException: 404 outside OIDC mode; 400 when no handshake is in flight
            for this browser, when the returned ``state`` does not match the one
            that started it, or when the IdP reports an error; 403 when the
            asserted identity is not the one mapped to the clicked user; 502
            when the IdP is unreachable or its response does not validate.
    """
    settings = get_settings(request)
    _require_oidc_mode(settings)

    # Popped before anything is validated, so a callback that fails any check
    # below still consumes the pending flow. The trade is deliberate: it costs a
    # cancelled login (any page that can reach this URL can make the operator
    # click their card again), and it buys a handshake that cannot be replayed,
    # probed repeatedly, or left half-open in the cookie. A login is cheap to
    # restart; a reusable one is not cheap to hold.
    pending = request.session.pop(PENDING_FLOW_SESSION_KEY, None)
    if not isinstance(pending, dict) or not pending.get("state") or not pending.get("user"):
        logger.warning("oidc callback rejected: no login is in flight for this browser")
        raise HTTPException(status_code=400, detail="no login is in progress")

    # Checked here as well as inside Authlib, and against our own record of the
    # state: Authlib proves the callback belongs to a handshake this browser
    # started, while this proves it belongs to the handshake that carried this
    # user. Without it a browser holding two half-finished flows could complete
    # one of them under the other's username.
    returned_state = request.query_params.get("state") or ""
    if not _same_value(str(pending["state"]), returned_state):
        logger.warning("oidc callback rejected: returned state does not match the login in flight")
        raise HTTPException(status_code=400, detail="login state mismatch")

    user = str(pending["user"])
    expected_subject = settings.oidc_subject(user)
    if expected_subject is None:
        # Unreachable through the login route, which refuses first; re-checked
        # because this is the decision that must never fall through to "some
        # other user's subject matched".
        logger.warning("oidc callback refused for %r: no IdP identity is mapped to that user", user)
        raise HTTPException(status_code=403, detail="this user has no mapped identity provider")

    client = _oauth_client(request)
    try:
        token = await client.authorize_access_token(
            request, claims_options=_claims_options(settings)
        )
    except OAuthError as exc:
        # The IdP reported an error, or Authlib's own state check failed.
        logger.warning("oidc callback rejected by the authorization step: %s", type(exc).__name__)
        raise HTTPException(
            status_code=400, detail="the identity provider rejected the login"
        ) from None
    except httpx.HTTPError:
        logger.warning(
            "oidc callback failed: the identity provider's token endpoint is unreachable"
        )
        raise HTTPException(
            status_code=502, detail="the identity provider could not be reached"
        ) from None
    except Exception as exc:  # ID-token validation: signature, issuer, audience, nonce, claims.
        # Deliberately broad. The concrete types come from whichever JWT library
        # the installed Authlib delegates to, and pinning them here would turn a
        # dependency bump into a 500 on a token that should simply be refused.
        # Only the class name is logged: an exception from claim validation can
        # carry claim material, and none of it belongs in the service log.
        logger.warning(
            "oidc callback rejected: the ID token did not validate (%s)", type(exc).__name__
        )
        raise HTTPException(
            status_code=502, detail="the identity provider's response could not be validated"
        ) from None

    claims = token.get("userinfo") or {}
    asserted = claims.get(settings.oidc_claim)
    if not isinstance(asserted, str) or not asserted:
        logger.warning(
            "oidc callback refused for %r: the ID token carries no usable %r claim",
            user,
            settings.oidc_claim,
        )
        raise HTTPException(status_code=403, detail="the identity provider asserted no identity")

    if not _same_value(asserted, expected_subject):
        # No search of the roster for a user this identity *would* match: the
        # card that was clicked is the only user this login can unlock.
        logger.warning(
            "oidc callback refused for %r: the asserted identity is mapped to a different user "
            "or to none",
            user,
        )
        raise HTTPException(status_code=403, detail="this identity is not permitted for this user")

    codec = get_session_codec(request)
    now = codec.now()
    # No generation tag: that is the password mode's rotation signal, computed
    # from a stored hash which OIDC has none of. An OIDC entry is bounded by its
    # expiry and by logout alone. It carries the asserted subject instead — an
    # opaque account identifier, not a credential — so a later verify subrequest
    # can name which provider account is behind this unlocked user without
    # re-contacting the IdP. `expected_subject` and `asserted` are byte-equal
    # here (the constant-time check above just proved it); the configured value
    # is stored so the cookie carries the deployment's own canonical spelling.
    session = _current_session(request).with_user(
        user,
        expires_at=now + settings.session_lifetime,
        generation_tag="",
        oidc_subject=expected_subject,
    )

    response = RedirectResponse(
        safe_return_to(pending.get("next"), user, flow="oidc login"), status_code=303
    )
    response.set_cookie(
        SESSION_COOKIE_NAME,
        codec.encode(session),
        httponly=True,
        samesite="lax",
        secure=settings.tls_enabled,
        # Not the state cookie's "/auth": this one has to reach the terminals it
        # authorises, whose verify subrequest nginx issues from "/u/<user>/".
        path="/",
    )
    # The subject is a claim value, but an opaque account identifier rather than
    # a credential (the cookie already carries it, signed not encrypted), so it
    # is recorded on this success line. `%r` quotes it, so a value carrying a
    # newline cannot forge a second log line.
    logger.info("oidc login succeeded for %r (subject %r)", user, expected_subject)
    return response
