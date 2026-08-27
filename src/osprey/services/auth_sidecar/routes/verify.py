"""The ``auth_request`` target: one bare 200 or 401 per protected request.

nginx calls this once for every request under ``/u/<user>/`` — page loads, API
calls, websocket handshakes alike — and turns any non-2xx answer into a denial.
It is therefore on the hot path of the whole deployment and says as little as
possible: a bodyless 200 or a bare 401, with no redirect and no
``WWW-Authenticate``. The only things an authorized answer may carry are the two
identity headers of
:mod:`~osprey.services.auth_sidecar.identity_headers` — the account behind the
request and the role it holds, both opaque identifiers and never a credential —
and each only when the session holds one.
Where an unauthenticated browser should be *sent* is
nginx's decision (``error_page 401``, content-negotiated), not this route's; a
redirect issued here would be followed by the subrequest instead of the user.

**The username is a render-time literal.** Each roster user has its own internal
``location = /_osprey_auth/<user>`` whose ``proxy_pass`` carries
``?user=<user>`` baked in at render time, so nothing about the client's request
— path, headers, cookies — selects which identity is being authorized. That is
why this route reads a query parameter and never reconstructs the user from
``$request_uri`` or ``X-Original-URI``: those are attacker-influenceable, and a
path-confusion trick against them would authorize the wrong terminal. A query
carrying the parameter more than once is refused outright rather than resolved,
so the answer can never depend on which of two values a parser happens to pick.

**Every denial looks the same.** An unknown user, an unmapped one, a missing
cookie, a forged cookie and a rotated password all produce the identical bare
401. The reason is recorded at debug level, by category, for the operator
reading the sidecar's log — never the cookie, the generation tag, or the stored
hash. Cookies are signed, not encrypted, so their contents are the caller's
session; nothing about it belongs in a log line.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Cookie, Depends, Query, Response

from ..app import AuthSettings, get_revocation_store, get_session_codec, get_settings
from ..exceptions import InvalidSessionError
from ..identity_headers import ROLE_HEADER, SUBJECT_HEADER, is_header_safe
from ..passwords import verify_generation_tag
from ..revocation import RevocationStore
from ..sessions import SESSION_COOKIE_NAME, SessionCodec, UnlockedUser
from .recheck import session_role, session_subject

logger = logging.getLogger(__name__)

router = APIRouter()

VERIFY_PATH = "/verify"
"""Rendered into every per-user internal location's ``proxy_pass``.

Deliberately *not* under the public ``/auth/`` prefix: everything below that
prefix is reachable without a session (it is where sessions come from), and this
endpoint must stay reachable only through nginx's ``internal`` auth locations.
"""


def _deny(user: str, reason: str) -> Response:
    """Refuse, recording why for the log and nothing for the client.

    Args:
        user: The username the subrequest asked about, logged with ``%r`` so a
            value carrying newlines cannot forge a log line.
        reason: A fixed category string. Never interpolate a cookie, a tag or a
            hash into it.

    Returns:
        A bare 401 — the same response for every reason, so the client cannot
        tell an unknown user from a bad cookie.
    """
    logger.debug("verify denied for user %r: %s", user, reason)
    return Response(status_code=401)


@router.get(VERIFY_PATH)
async def verify(
    settings: Annotated[AuthSettings, Depends(get_settings)],
    codec: Annotated[SessionCodec, Depends(get_session_codec)],
    revocations: Annotated[RevocationStore, Depends(get_revocation_store)],
    user: Annotated[list[str] | None, Query()] = None,
    session_cookie: Annotated[str | None, Cookie(alias=SESSION_COOKIE_NAME)] = None,
) -> Response:
    """Authorize one ``auth_request`` subrequest for ``user``.

    All four conditions must hold, and any one of them failing gives the same
    answer:

    1. the request names exactly one roster user;
    2. the session cookie verifies, is not over-age, and is well-formed;
    3. that user is in the cookie's unlocked set and has not passed its expiry;
    4. the session id has not been revoked by a logout;

    plus, in password mode, that the entry's credential-generation tag still
    matches the digest of the user's *current* stored hash. That last check is
    what makes ``osprey users passwd`` take effect immediately: rotating the
    password changes the stored hash, so every session minted against the old
    one stops verifying — without server-side session state, and surviving the
    container recreate that rotation performs (which is also what clears the
    revocation store).

    The parameters are missing-tolerant on purpose. ``user`` defaults to absent
    and the cookie to ``None`` rather than being required, because FastAPI would
    otherwise answer a malformed subrequest with a 422 carrying a validation
    body — a different status and a different shape from every other refusal,
    which is both an information leak and something nginx would surface
    differently.

    ``user`` is read as a *list* so that a repeated parameter is refused rather
    than resolved. A single-valued ``str`` would silently take the first value
    and authorize against it, making the answer depend on which end of
    ``?user=bob&user=alice`` the parser picks. nginx's exact-match ``internal``
    locations cannot produce that query today — the username is a render-time
    literal and the render gate rejects a roster name carrying ``&`` or ``?`` —
    but this endpoint is the deployment's single authorization decision, and it
    should not rest its safety on another layer's invariant holding forever.

    Args:
        settings: The deployment's frozen settings.
        codec: The app's one session codec, which also holds the one clock every
            expiry in the session path is stamped from and checked against.
        revocations: The app's one revocation store, written by logout.
        user: The roster username, supplied by nginx as a render-time literal.
            Exactly one value authorizes; none and several are both refused.
        session_cookie: The signed session cookie, if the browser sent one.

    Returns:
        An empty 200 when the request is authorized, a bare 401 otherwise. An
        authorized answer additionally carries the identity headers the session
        can fill — the account behind the request and the role it holds (see
        :func:`_authorized`).
    """
    requested = user or []
    if len(requested) != 1:
        # Zero values means something other than a rendered nginx location
        # called this; several means the query was assembled by something that
        # is not this deployment's nginx at all. The count goes in the log, the
        # values never do — they are the one part of this request a client could
        # have chosen.
        return _deny("", f"expected exactly one user parameter, got {len(requested)}")

    username = requested[0]
    if not username:
        return _deny(username, "empty user parameter")

    if not settings.knows_user(username):
        return _deny(username, "user is not on the roster")

    if not session_cookie:
        return _deny(username, "no session cookie")

    try:
        state = codec.decode(session_cookie)
    except InvalidSessionError:
        # Tampering, a malformed or unknown-version payload, and an over-age
        # cookie are one case here: the cookie cannot be trusted, so there is no
        # session. The exception's message stays out of the log — it is derived
        # from the cookie the caller sent.
        return _deny(username, "session cookie rejected")

    now = codec.now()
    if not state.is_unlocked(username, now):
        # Covers both "this browser never unlocked that user" and "the entry
        # lapsed": `decode` has already dropped expired entries.
        return _deny(username, "user is not unlocked in this session")

    if revocations.is_revoked(state.session_id):
        return _deny(username, "session was revoked by a logout")

    if settings.method != "oidc":
        # Password mode. Written as "not OIDC" rather than "is password" so the
        # stricter branch is the default one: the guard admits only the two
        # supported methods, and anything else reaching here must get the tag
        # check, not skip it.
        #
        # In OIDC mode the entry's tag is not consulted at all, whatever it
        # holds. There is no stored hash to compare it against, and only a
        # holder of the signing secret could have put a tag there in the first
        # place — a session that survives that check has already proved more
        # than the tag could.
        stored = settings.password_hash(username)
        if stored is None:
            # A roster user with no provisioned credential. An individual
            # denial, not a deployment-wide fault: the other users still
            # authenticate, so this is a 401 rather than the guard's 503.
            return _deny(username, "user has no stored credential")

        entry = state.entry(username)
        if entry is None or not verify_generation_tag(entry.generation_tag, stored):
            # `is_unlocked` above means the entry exists; the check is here so
            # the tag comparison cannot be reached with `None`. A tag that no
            # longer matches means the password was rotated under this session.
            return _deny(username, "credential generation tag does not match")

    return _authorized(username, state.entry(username), settings)


def _subject_for(username: str, entry: UnlockedUser | None, settings: AuthSettings) -> str:
    """Which account this authorized session names, by method.

    Delegated to :func:`~osprey.services.auth_sidecar.routes.recheck.session_subject`
    so that "which method names which account" has exactly one definition. The
    login routes ask the same table what a *fresh* login may carry; this is the
    same table asked what an already-minted session names, and a rule that lived
    in both places would drift the first time one of them grew a method.

    Args:
        username: The roster user this subrequest authorized. Verified against
            the roster, and byte-identical to the entry's own username.
        entry: The unlocked entry, or ``None`` on the defensive path.
        settings: The deployment's frozen settings, read for the method.

    Returns:
        The subject to report, or ``""`` when this session names no account.
    """
    return session_subject(method=settings.method, username=username, entry=entry)


def _authorized(username: str, entry: UnlockedUser | None, settings: AuthSettings) -> Response:
    """The bare 200, carrying whichever identity headers the session can fill.

    ``is_unlocked`` has already established that ``entry`` exists, so ``None`` is
    only a defensive guard.

    Two headers may ride the 200. ``X-Osprey-Auth-Subject`` names the account —
    the provider subject under OIDC, the roster username otherwise — so a later
    layer can tell who is behind the request without re-reading the cookie.
    ``X-Osprey-Auth-Role`` names the role that account holds. Each is *omitted*
    rather than emitted empty when the session has nothing to put in it, so a
    present header always means a known value and no consumer has to tell blank
    from absent: presence of the subject means a known account, and absence of
    the role means no privileges.

    **The role is the one the login granted, not the roster's current answer.**
    It is read off the session and never re-derived here, so a retired role
    lapses with the session rather than at the moment the config is edited: an
    operator who removes a ``role:`` (or an OIDC claim binding) takes it away
    from the next login, while sessions already outstanding keep carrying it
    until they expire or are revoked. Deliberate — re-deriving on every
    subrequest would turn a config edit into a live privilege change on browsers
    mid-session, and it would put a second reading of the identity matrix on the
    hot path. If retiring a role must retire outstanding sessions too, the
    generation tag is the mechanism to copy: it already revokes on a credential
    change, and it does so by invalidating the session rather than by rewriting
    what it says.

    **A changed METHOD is a different question, and it does not lapse.** The
    paragraph above is about a role that is no longer *granted*; switching
    ``auth.method`` from ``password`` to ``oidc`` is about a proof this
    deployment no longer *accepts*. The signing secret survives a re-render, so
    those password cookies stay readable, and under ``oidc`` the generation-tag
    check that a password rotation would have tripped is skipped. Both halves of
    the identity therefore go through the matrix rather than off the entry:
    :func:`~osprey.services.auth_sidecar.routes.recheck.session_subject` already
    named nobody for such a session, and
    :func:`~osprey.services.auth_sidecar.routes.recheck.session_role` now grants
    it no role either.

    **An uncarryable value denies.** Both values are checked against
    :func:`~osprey.services.auth_sidecar.identity_headers.is_header_safe` even
    though the session codec refuses to store one that fails — the roster
    username reaches here without passing through the codec at all, so a
    deployment rendered past its lint with a non-ASCII username would otherwise
    have its identity mangled across the boundary or fail the response encoding
    on the hot path. Denying instead is the closed answer, and it is the same
    bare 401 as every other refusal.

    Args:
        username: The roster user this subrequest authorized.
        entry: The unlocked entry backing the decision.
        settings: The deployment's frozen settings.

    Returns:
        A 200 carrying zero, one or both identity headers — or a 401 if an
        identity this deployment cannot carry was about to be reported.
    """
    headers: dict[str, str] = {}
    subject = _subject_for(username, entry, settings)
    if subject:
        if not is_header_safe(subject):
            return _deny(username, "the session subject cannot be carried in an identity header")
        headers[SUBJECT_HEADER] = subject

    role = session_role(method=settings.method, entry=entry)
    if role:
        if not is_header_safe(role):
            return _deny(username, "the session role cannot be carried in an identity header")
        headers[ROLE_HEADER] = role

    return Response(status_code=200, headers=headers)
