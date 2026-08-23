"""The ``auth_request`` target: one bare 200 or 401 per protected request.

nginx calls this once for every request under ``/u/<user>/`` — page loads, API
calls, websocket handshakes alike — and turns any non-2xx answer into a denial.
It is therefore on the hot path of the whole deployment and says as little as
possible: a bodyless 200 or a bare 401, with no redirect and no
``WWW-Authenticate``. The only thing an authorized answer may carry is the
``X-Osprey-Auth-Subject`` header naming the OIDC account behind the request (an
opaque identifier, never a credential), and only when the session holds one.
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
from ..passwords import verify_generation_tag
from ..revocation import RevocationStore
from ..sessions import SESSION_COOKIE_NAME, SessionCodec, UnlockedUser

logger = logging.getLogger(__name__)

router = APIRouter()

VERIFY_PATH = "/verify"
"""Rendered into every per-user internal location's ``proxy_pass``.

Deliberately *not* under the public ``/auth/`` prefix: everything below that
prefix is reachable without a session (it is where sessions come from), and this
endpoint must stay reachable only through nginx's ``internal`` auth locations.
"""

SUBJECT_HEADER = "X-Osprey-Auth-Subject"
"""Response header carrying the OIDC subject on an authorized OIDC request.

Set only when the unlocked entry holds a non-empty subject, so a password
session (which has none) and a pre-subject OIDC session leave it absent rather
than blank. An OIDC ``sub`` is ASCII by specification, which is what this header
carries; a subject drawn from a non-ASCII claim spelling meets the latin-1 wire
encoding of HTTP header values and is out of scope until nginx forwarding (a
later phase) settles that cross-boundary encoding — nginx does not forward this
header yet.
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
        authorized OIDC request additionally carries ``X-Osprey-Auth-Subject``
        naming the provider account; a password session, and an OIDC session
        minted before the subject was carried, omit that header (see
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

    return _authorized(state.entry(username))


def _authorized(entry: UnlockedUser | None) -> Response:
    """The bare 200, carrying the OIDC subject when the session holds one.

    ``is_unlocked`` has already established that ``entry`` exists, so ``None`` is
    only a defensive guard; either way the response is a 200.

    An OIDC session carries the provider's ``sub`` in
    :attr:`~osprey.services.auth_sidecar.sessions.UnlockedUser.oidc_subject`, and
    it is returned as ``X-Osprey-Auth-Subject`` so a later layer can tell which
    provider account is behind the request without re-reading the cookie. A
    password session, and any OIDC session minted before the subject was carried,
    holds an empty subject; the header is then *omitted* rather than emitted
    empty, so its presence always means a known account and downstream code need
    not distinguish an empty value from an absent one.

    The subject is not forwarded anywhere by nginx yet — that is a later phase —
    so this header is presently observable only on the internal auth subrequest.
    """
    subject = entry.oidc_subject if entry is not None else ""
    if subject:
        return Response(status_code=200, headers={SUBJECT_HEADER: subject})
    return Response(status_code=200)
