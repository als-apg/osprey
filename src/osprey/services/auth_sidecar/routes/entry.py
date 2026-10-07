"""The one card-less way in: sign in without naming a card.

Two public paths. :data:`ENTRY_PATH` serves both methods: under ``password``
``GET`` renders a username-and-password form and ``POST`` evaluates it; under
``oidc`` ``GET`` hands the browser to :data:`OIDC_ENTRY_PATH`, where a handshake
that names no card starts.

**A person names themselves, and the service opens exactly the cards the card
login would open for that credential.** For every roster card ``c``, the
card-less login mints an entry for ``c`` if and only if the card login for
``c``, driven by the same credential or the same token, would mint one, and the
entry is field for field the same: username, opener, admitted identity,
subject, generation tag, role and role source. Only ``expires_at`` can differ,
because it is stamped from the clock. Nothing new decides who opens what: under
password login the card form's own branch order is mirrored
(:func:`password_cards`), and under OIDC the callback's own per-card decision
is called. One card is a redirect to that terminal, several are a list of them,
none is a refusal.

**Why a module of its own.** :mod:`~osprey.services.auth_sidecar.routes.login`
is built on "the clicked card is the username", and this route is the one place
where no card was clicked. It reuses that module's password helpers rather than
re-spelling them — the origin check, the form readers, the page frame, the
refusal messages and the username bound — so both surfaces refuse, throttle
and render the same way.

**One throttle window per name.** The form keys the login throttle on the typed
name's bounded spelling: the same bucket that name's own card and its opener
field on a shared card use, so no surface can shop for a fresh window. An
unknown name, a name with no credential and a wrong password produce one
status, one page, one message, one throttle charge and one ledger category. The
one distinguishable answer, "no terminal", is reachable only after the
credential verified.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import RedirectResponse

from osprey.interfaces.common_middleware import url_mount_prefix

from .. import audit
from ..app import (
    ACCESS_ROSTER,
    AuthSettings,
    get_attempt_throttle,
    get_revocation_store,
    get_session_codec,
    get_settings,
)
from ..passwords import PasswordCheck, check_password, generation_tag
from ..revocation import RevocationStore
from ..sessions import SESSION_COOKIE_NAME, SessionCodec, SessionState
from ..throttle import AttemptThrottle
from .login import (
    _DEFAULT_THEME_FAMILY,
    _NO_STORE_HEADERS,
    DENIAL_MESSAGE,
    FIELD_PASSWORD,
    MAX_USERNAME_LENGTH,
    THROTTLE_MESSAGE,
    _cross_site_post,
    _current_session,
    _expected_origin,
    _form_values,
    _only,
    _prefers_html,
    _submission_origin,
    _templates,
    _theme_blocks,
)
from .recheck import LoginGrant, RecheckRefused, recheck_login, roster_roles

logger = logging.getLogger(__name__)

router = APIRouter()

ENTRY_PATH = "/auth/enter"
"""Public path of the card-less sign-in, under both methods."""

OIDC_ENTRY_PATH = "/auth/oidc/enter"
"""Where an OIDC deployment's ``GET`` is sent instead of a password form.

Spelled out rather than imported from
:mod:`~osprey.services.auth_sidecar.routes.oidc`, for the reason
:data:`~osprey.services.auth_sidecar.routes.login.OIDC_LOGIN_PATH` is: reading
one string from that module would pull Authlib into this one's import. The two
constants are checked against each other by the tests instead.
"""

TEMPLATE_NAME = "enter.html"
"""Rendered from the package's ``templates/`` directory."""

FIELD_USERNAME = "username"
"""The form's username field, where the person types their own roster name."""

NO_CARD_MESSAGE = "No terminal for this account."
"""The refusal a verified person no card admits is told."""


@dataclass(frozen=True)
class OpenedCard:
    """One card a card-less login unlocked, as the response presents it.

    Attributes:
        name: The roster card, which is also its terminal's mount segment.
        own: Whether the card is the person's own by roster identity, rather
            than one shared with them. Listed first, and untagged.
    """

    name: str
    own: bool


def password_cards(settings: AuthSettings, person: str) -> tuple[OpenedCard, ...]:
    """Every card a password login as ``person`` opens, own cards first.

    Mirrors the card form's branch order in
    :func:`~osprey.services.auth_sidecar.routes.login.login_submit`, which is
    the definition: a card whose rule carries ``roster`` takes the credential
    of whoever typed their name into its opener field, and only otherwise does
    ``self`` accept the card owner's own. So a card carrying ``roster`` opens
    for ``person`` as its opener — their own card included — and any other
    card opens only when it is ``person``'s own and ``self`` admits them.
    ``user:`` and ``domain:`` principals stay inert, as they are on the card
    form.

    Every card is evaluated, in roster order and with no early break.

    Args:
        settings: The deployment's frozen settings.
        person: The unclamped name whose credential verified.

    Returns:
        The opened cards, own cards first and otherwise in roster order.
    """
    opened: list[OpenedCard] = []
    for card in settings.users:
        if ACCESS_ROSTER in settings.access(card):
            opened.append(OpenedCard(name=card, own=card == person))
        elif card == person and settings.owner_admitted(card):
            opened.append(OpenedCard(name=card, own=True))
    return tuple(sorted(opened, key=lambda card: not card.own))


def _href(card: OpenedCard) -> str:
    """The terminal a card's link and redirect point at."""
    return f"{url_mount_prefix(card.name)}/"


def render(
    request: Request,
    *,
    shape: str,
    status_code: int,
    username: str = "",
    error: str | None = None,
    cards: Sequence[OpenedCard] = (),
    headers: dict[str, str] | None = None,
) -> Response:
    """Render the card-less page in one of its three shapes.

    Args:
        request: The inbound request, which Starlette's template response needs.
        shape: ``"form"``, ``"cards"`` or ``"none"``.
        status_code: The response status.
        username: The bounded name to prefill on a re-rendered form.
        error: The message to show above the form, or ``None``.
        cards: The unlocked cards, for the ``cards`` shape.
        headers: Extra response headers, merged over the no-store default.

    Returns:
        The rendered page.
    """
    settings = get_settings(request)
    return _templates.TemplateResponse(
        request,
        TEMPLATE_NAME,
        {
            "shape": shape,
            "username": username,
            "error": error,
            "enter_path": ENTRY_PATH,
            "cards": [{"name": card.name, "href": _href(card), "own": card.own} for card in cards],
            "theme_blocks": _theme_blocks(settings.web_theme or _DEFAULT_THEME_FAMILY),
            "app_name": settings.web_app_name,
        },
        status_code=status_code,
        headers={**_NO_STORE_HEADERS, **(headers or {})},
    )


def opened_response(
    request: Request,
    settings: AuthSettings,
    codec: SessionCodec,
    session: SessionState,
    cards: Sequence[OpenedCard],
) -> Response:
    """The answer to a card-less login that opened at least one card.

    One card is a 303 to that terminal; several are the list page, every link
    on which is already unlocked. Either way the session cookie carries the
    attribute set every login issues. Never called with no card: the caller
    refuses instead.

    Args:
        request: The inbound request.
        settings: The deployment's frozen settings.
        codec: The app's session codec.
        session: The session with every opened card minted into it.
        cards: The opened cards, own cards first.

    Returns:
        The redirect or the list page, carrying the session cookie.
    """
    response: Response
    if len(cards) == 1:
        response = RedirectResponse(_href(cards[0]), status_code=303, headers=_NO_STORE_HEADERS)
    else:
        response = render(request, shape="cards", status_code=200, cards=cards)
    response.set_cookie(
        SESSION_COOKIE_NAME,
        codec.encode(session),
        httponly=True,
        samesite="lax",
        secure=settings.tls_enabled,
        # The cookie reaches the terminals it authorises, as every login's does.
        path="/",
    )
    return response


def refuse_no_card(request: Request, *, subject: str) -> Response:
    """File and answer a login no card admits.

    Args:
        request: The inbound request.
        subject: The ledger subject the refusal files under, and the name the
            log line gives.

    Returns:
        The ``none`` page with 403.
    """
    logger.warning("sign-in for %r opened no card: no card on this deployment admits it", subject)
    audit.record_login_refusal(user=subject, reason=audit.REASON_NO_CARD)
    return render(request, shape="none", status_code=403)


@router.get(ENTRY_PATH)
async def enter_page(
    request: Request,
    settings: Annotated[AuthSettings, Depends(get_settings)],
) -> Response:
    """Show the card-less sign-in, or hand an OIDC browser to its handshake.

    Args:
        request: The inbound request.
        settings: The deployment's frozen settings.

    Returns:
        The sign-in form under password login; a 302 to :data:`OIDC_ENTRY_PATH`
        under OIDC.

    Raises:
        HTTPException: 404 under any other method.
    """
    if settings.method == "oidc":
        return RedirectResponse(OIDC_ENTRY_PATH, status_code=302, headers=_NO_STORE_HEADERS)
    if settings.method != "password":
        raise HTTPException(
            status_code=404,
            detail="this deployment does not use password login",
            headers=_NO_STORE_HEADERS,
        )
    return render(request, shape="form", status_code=200)


@router.post(ENTRY_PATH)
async def enter_submit(
    request: Request,
    settings: Annotated[AuthSettings, Depends(get_settings)],
    codec: Annotated[SessionCodec, Depends(get_session_codec)],
    throttle: Annotated[AttemptThrottle, Depends(get_attempt_throttle)],
    revocations: Annotated[RevocationStore, Depends(get_revocation_store)],
) -> Response:
    """Evaluate one card-less password attempt and unlock every card it opens.

    The order is the card form's: the origin check before the form is read, the
    throttle before the credential, and only an evaluated attempt records a
    failure. After the credential verified, every card :func:`password_cards`
    names is re-checked against the identity matrix exactly as the card form
    re-checks it; a card the matrix refuses is filed under its own category
    and left out, which is the card form's answer for that card alone.

    Args:
        request: The inbound request, carrying the submitted form.
        settings: The deployment's frozen settings.
        codec: The app's session codec and clock.
        throttle: The app's one login-attempt throttle.
        revocations: The app's revocation store.

    Returns:
        A 303 to the one opened terminal, or the list page when several opened,
        carrying the re-issued session cookie; the form again with 401 on a
        refused credential, or with 429 and ``Retry-After`` inside an open
        window; the ``none`` page with 403 when the credential verified and no
        card admits it; a 303 back to the form when a browser named no user.

    Raises:
        HTTPException: 404 when this deployment serves no password login; 400
            when the form arrives from another origin, or does not name exactly
            one username and the caller did not ask for HTML.
    """
    if settings.method != "password":
        raise HTTPException(
            status_code=404,
            detail="this deployment does not use password login",
            headers=_NO_STORE_HEADERS,
        )

    if _cross_site_post(request, settings):
        # Before the form is read: a cross-site POST is not an attempt, so it
        # may not touch the throttle.
        logger.warning(
            "login POST refused: submitted from %r, which is not this deployment (%r)",
            _submission_origin(request),
            _expected_origin(request, settings),
        )
        raise HTTPException(
            status_code=400,
            detail="this form was not submitted from this deployment",
            headers=_NO_STORE_HEADERS,
        )

    form = await request.form()
    submitted = _form_values(form, FIELD_USERNAME)
    name = _only(submitted)
    if name is None:
        named = len(submitted)
        logger.warning("sign-in submitted without exactly one username field (got %d)", named)
        if named < 2 and _prefers_html(request):
            return RedirectResponse(ENTRY_PATH, status_code=303, headers=_NO_STORE_HEADERS)
        raise HTTPException(
            status_code=400,
            detail="the sign-in form must name exactly one username",
            headers=_NO_STORE_HEADERS,
        )

    # The bounded spelling is what is echoed, logged and throttled; the
    # unclamped one is what the credential is looked up under, so a prefix of
    # an over-long name can never resolve to a real roster user.
    oversize = len(name) > MAX_USERNAME_LENGTH
    shown = name[:MAX_USERNAME_LENGTH]
    if oversize:
        logger.warning("sign-in submitted an impossible username of %d characters", len(name))

    delay = throttle.retry_after(shown)
    if delay > 0:
        logger.info(
            "sign-in attempt for %r refused unevaluated: the attempt window is still open", shown
        )
        return render(
            request,
            shape="form",
            status_code=429,
            error=THROTTLE_MESSAGE,
            username=shown,
            headers={"Retry-After": str(math.ceil(delay))},
        )

    password = _only(_form_values(form, FIELD_PASSWORD)) or ""
    stored = None if oversize else settings.password_hash(name)
    outcome = None if stored is None else check_password(password, stored)
    if stored is None or outcome is not PasswordCheck.MATCH:
        throttle.record_failure(shown)
        unevaluable = outcome is PasswordCheck.UNEVALUABLE
        if unevaluable:
            logger.error(
                "sign-in refused for %r: the stored credential cannot be evaluated; "
                "replace it with `osprey users passwd %s`",
                shown,
                shown,
            )
        else:
            logger.warning("sign-in refused for %r: the submitted credential did not verify", shown)
        audit.record_login_refusal(
            user=shown,
            reason=(
                audit.REASON_CREDENTIAL_UNEVALUABLE if unevaluable else audit.REASON_BAD_CREDENTIAL
            ),
        )
        return render(request, shape="form", status_code=401, error=DENIAL_MESSAGE, username=shown)

    throttle.record_success(shown)

    table = roster_roles(request)
    admitted: list[tuple[OpenedCard, LoginGrant]] = []
    for card in password_cards(settings, name):
        try:
            grant = recheck_login(method=settings.method, user=card.name, roster_roles=table)
        except RecheckRefused as refused:
            # The card form files this record and answers 403 for this card;
            # here the same decision drops this card alone.
            logger.warning("sign-in for %r left out %r: %s", name, card.name, refused.reason)
            audit.record_login_refusal(user=card.name, reason=refused.reason)
            continue
        admitted.append((card, grant))

    if not admitted:
        # The throttle stays cleared: the credential did verify, and holding the
        # window open would penalise the one person whose password was right.
        return refuse_no_card(request, subject=shown)

    now = codec.now()
    session = _current_session(request, codec, revocations)
    openers: list[str] = []
    for card, grant in admitted:
        # The card form's two shapes: a card carrying `roster` is opened by the
        # name typed into its opener field, any other card by its own user.
        opener = name if ACCESS_ROSTER in settings.access(card.name) else ""
        session = session.with_user(
            grant.subject,
            expires_at=now + settings.session_lifetime,
            # The person's own hash: the one the card form checks on both of its
            # branches for this credential.
            generation_tag=generation_tag(stored),
            role=grant.role,
            role_source=grant.role_source,
            opener=opener,
        )
        openers.append(opener)

    response = opened_response(
        request, settings, codec, session, tuple(card for card, _ in admitted)
    )
    for (card, grant), opener in zip(admitted, openers, strict=True):
        logger.info("password sign-in for %r opened %r", name, card.name)
        audit.record_login_success(
            user=card.name,
            method=settings.method,
            role=grant.role,
            detail=f"opener={opener}" if opener else None,
        )
    return response
