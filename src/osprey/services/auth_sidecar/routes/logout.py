"""Logout: surrender one unlocked user without disturbing the others.

``GET /auth/logout?user=<name>`` under the public ``/auth/`` prefix, reached
either from the in-app logout control (which navigates here after its own PTY
teardown POST) or directly. It composes with that teardown rather than replacing
it: the app tears down the terminal, this retires the session's claim on it.

**One user, not the browser.** A control-room machine may have several roster
users unlocked in one browser at once, so logging out of alice must leave bob
exactly as he was. That shapes everything below.

**Why the session id is rotated.** Revocation is recorded per session *id* and
:func:`~osprey.services.auth_sidecar.routes.verify.verify` refuses any cookie
whose id appears in the store — for every user it carries, not just the one that
logged out. Reissuing the surrendered cookie's id would therefore log bob out as
a side effect of alice leaving. So the response carries the surviving users under
a **fresh** id while the old id goes into the revocation store: the cookie the
browser handed in is dead, and the one it gets back is not. The issued-at stamp
is preserved, so this cannot be used to extend a session's overall age by
logging out repeatedly.

**Revoked until the entry would have lapsed anyway.** The store is given the
logged-out user's own ``expires_at``. That is exactly long enough: the only
thing a captured pre-logout cookie could do is authorize *that* user, and the
entry inside it stops authorizing at that same moment on its own expiry check.

**Every logout looks the same from outside.** Logging out a user who was never
unlocked, or one who is not on the roster at all, produces the same 303 to the
landing page and the same reissued cookie as a real logout — no error, no detail,
nothing that reveals whether that user was unlocked in this browser. The
distinction is recorded in the log, for the operator, at debug level.

Reached by a top-level ``GET`` navigation on purpose (the in-app control is a
link, and ``SameSite=Lax`` sends the cookie on exactly that). The cost is that
another site can force a logout; the benefit is that logout works from a plain
link with no script. A forced logout ends a session early — a nuisance, never an
escalation — so the trade is taken deliberately.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Annotated

from fastapi import APIRouter, Cookie, Depends, Query, Response
from fastapi.responses import RedirectResponse

from ..app import AuthSettings, get_revocation_store, get_session_codec, get_settings
from ..exceptions import InvalidSessionError
from ..revocation import RevocationStore
from ..sessions import (
    SESSION_COOKIE_NAME,
    SessionCodec,
    SessionState,
)

logger = logging.getLogger(__name__)

router = APIRouter()

LOGOUT_PATH = "/auth/logout"
"""Public path nginx proxies here verbatim (``location /auth/``)."""

LANDING_PATH = "/"
"""Where a logged-out browser is sent.

A same-origin absolute path, not
:attr:`~osprey.services.auth_sidecar.app.AuthSettings.external_origin` + "/":
the origin is only required in OIDC mode (it builds the ``redirect_uri``), and a
password-mode deployment that leaves it unset must still redirect somewhere real.
nginx serves the landing page at exactly this path.
"""


def _redirect() -> RedirectResponse:
    """The one response every logout produces, successful or not.

    303 rather than 302: the browser must follow it with a ``GET`` regardless of
    how it arrived, and the landing page is a different resource rather than a
    relocation of this one. ``no-store`` for the same reason every login
    response carries it: nothing in the auth flow may ever be replayed from a
    cache, and a replayed logout would skip the revocation while looking like
    it worked.
    """
    return RedirectResponse(LANDING_PATH, status_code=303, headers={"Cache-Control": "no-store"})


def _set_session_cookie(
    response: Response, codec: SessionCodec, state: SessionState, settings: AuthSettings
) -> None:
    """Attach the reissued session cookie.

    Attributes are the same set the login and OIDC callback routes issue — a
    cookie that differed in any of them would be a *second* cookie to the
    browser, leaving the original in place and logged out of nothing.

    ``Path=/`` because this cookie has to reach the terminals it authorises,
    whose verify subrequest nginx issues from ``/u/<user>/``; no ``Max-Age``,
    so it is a session cookie whose real lifetime is the signed expiry inside it.
    """
    response.set_cookie(
        SESSION_COOKIE_NAME,
        codec.encode(state),
        httponly=True,
        samesite="lax",
        secure=settings.tls_enabled,
        path="/",
    )


@router.get(LOGOUT_PATH)
async def logout(
    settings: Annotated[AuthSettings, Depends(get_settings)],
    codec: Annotated[SessionCodec, Depends(get_session_codec)],
    revocations: Annotated[RevocationStore, Depends(get_revocation_store)],
    user: Annotated[list[str] | None, Query()] = None,
    session_cookie: Annotated[str | None, Cookie(alias=SESSION_COOKIE_NAME)] = None,
) -> Response:
    """Lock ``user`` again in this browser's session.

    Both parameters are optional so that a malformed request redirects like every
    other logout instead of drawing FastAPI's 422 validation body — a different
    status, a different shape, and a description of this endpoint's parameters
    that nothing needs. ``user`` is read as a list for the same reason
    :func:`~osprey.services.auth_sidecar.routes.verify.verify` does: a repeated
    parameter is refused outright rather than resolved to whichever value a
    parser happens to pick first.

    Args:
        settings: The deployment's frozen settings.
        codec: The app's one session codec, and the one clock in the session path.
        revocations: The app's one revocation store — the same object verify
            reads, or this would revoke something nothing checks.
        user: The roster user to log out. Exactly one value acts; none and
            several are no-ops.
        session_cookie: The browser's session cookie, if it sent one.

    Returns:
        A 303 to the landing page, carrying a reissued cookie whenever there was
        a readable session to reissue.
    """
    response = _redirect()

    requested = user or []
    if len(requested) != 1 or not requested[0]:
        # The count, never the values: they are caller-supplied. A request that
        # names no user (or several) has not asked for anything specific, so
        # nothing is revoked and no cookie is disturbed.
        logger.debug("logout ignored: expected exactly one user parameter, got %d", len(requested))
        return response

    username = requested[0]

    if not session_cookie:
        logger.debug("logout for %r: no session cookie to act on", username)
        return response

    try:
        state = codec.decode(session_cookie)
    except InvalidSessionError:
        # Tampered, stale-keyed or over-age. It carries no authorisation, so
        # there is nothing to revoke and nothing worth reissuing; the browser
        # keeps a cookie that every verify already refuses, and the next login
        # replaces it. The exception message stays out of the log — it is
        # derived from the value the caller sent.
        logger.debug("logout for %r: session cookie is unreadable", username)
        return response

    entry = state.entry(username)
    if entry is not None:
        # Only now is there something to retire. `decode` has already dropped
        # expired entries, so reaching here means the user really was unlocked.
        revocations.revoke(state.session_id, entry.expires_at)
        logger.info("logout: %r surrendered an unlocked session", username)
    else:
        logger.debug("logout for %r: that user was not unlocked in this session", username)

    # Rotated id, preserved issued-at, and the same cookie whether or not
    # anything was revoked — so a logout for a user who was never unlocked is
    # indistinguishable from one that ended a real session.
    rotated = replace(state.without_user(username), session_id=codec.new_state().session_id)
    _set_session_cookie(response, codec, rotated, settings)
    return response
