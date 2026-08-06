"""Password-mode login page and credential POST for the auth sidecar.

One public path, :data:`LOGIN_PATH`, under the ``/auth/`` prefix nginx proxies
here: ``GET`` renders the prompt and ``POST`` evaluates the credential. It is
where every denied browser navigation lands — nginx's ``error_page 401`` sends
``/auth/login?user=<name>&next=<path>`` — and it is the only place a
password-mode session comes from.

**The clicked card is the username.** The landing page offers one card per roster
user, so this page is addressed per user and asks for a password and nothing
else. There is no username field: a name typed here could only disagree with the
one the link named, and the page would then either lie about who is being
unlocked or refuse an operator who typed their own name correctly.

**Every method's unauthenticated browser arrives here.** nginx's 401 handler
knows nothing about how this deployment authenticates, so it emits the same
``/auth/login`` redirect under OIDC. A password form is unanswerable there —
there are no passwords to enter — so ``GET`` in OIDC mode redirects onward to
:data:`~osprey.services.auth_sidecar.routes.oidc.LOGIN_PATH`, carrying the same
user and the same return-to. That path is this module's alone (the OIDC routes
live under ``/auth/oidc/``), so the dispatch has to happen here or nowhere.

**The throttle runs before the credential, never after.** A refused attempt
costs a dict lookup; an evaluated one costs an scrypt derivation. Consulting
:meth:`~osprey.services.auth_sidecar.throttle.AttemptThrottle.retry_after` first
is what makes parallel guessing throughput-bounded rather than merely delayed,
and it is why a 429 never calls ``record_failure``: counting attempts the
sidecar declined to evaluate would let a flood ratchet the window against the
operator it is supposed to protect.

**Every refusal is the same refusal.** A user with no provisioned credential, a
user who is not on the roster at all, and a wrong password produce one status,
one page, one message, and one identical call into the throttle. What the deny
path deliberately does *not* do is spend an scrypt derivation on a username that
has no stored hash: the roster is public (the landing page lists it), so
equalising that work would buy no secrecy, while turning every POST naming an
invented user into a key derivation an attacker can request without ever meeting
a throttle window. The residual is a timing difference, and it is real —
measured at roughly 32.3 ms for a credentialed roster user against 2.9 ms for an
uncredentialed one and 2.3 ms for a name that is not on the roster. It
distinguishes "this user has a password provisioned" from "this user does not",
which the landing page already discloses, and it is accepted rather than
overlooked.

``GET`` does answer 404 for a user who is not on the roster — the same answer the
OIDC login route gives, and the honest one, since no credential is evaluated
there and nginx only ever generates links for users that exist. Roster
membership is public by design; what must not vary is the answer to *"was that
the right password"*, which is what the POST path holds identical.

**Nothing a client sends is assumed to be small.** nginx will pass through a
return-to as long as a request line allows, and a form field has no bound at
all. Both are capped —
:data:`~osprey.services.auth_sidecar.return_to.MAX_RETURN_TO_LENGTH` for the
return-to, :data:`MAX_USERNAME_LENGTH` for the name — before either can reach a
``Location`` header, where an over-long value becomes a 502 from the proxy
rather than a login, or the rendered page, where an unauthenticated,
pre-throttle request would otherwise choose the response's size.

**A cross-site POST is not a login.** ``SameSite=Lax`` keeps the session cookie
off cross-site *sub*requests, but it does not stop a browser from storing the
``Set-Cookie`` a top-level cross-site form POST comes back with. Unchecked, a
page anywhere could log an operator into an attacker's roster user — and since a
login *adds* to the unlocked set rather than replacing it, the operator would
keep their own terminals and have no reason to notice which one they are typing
into. So before anything else happens on ``POST``, the submitting origin
(``Origin``, or ``Referer`` where an older browser omits it on a same-origin
form) is compared against this deployment's own, and anything else — including a
request that declares no origin at all — is refused. The refusal is the *same*
400 a malformed form already produces: a distinct status would tell an attacker
exactly which control fired, and same-shaped refusals are this route's whole
deny-path discipline.

**One consequence worth stating for anyone driving this endpoint from a
script**: a POST with neither header is refused, so a test or health probe must
send an ``Origin`` naming the deployment. Real browsers do it unprompted, and a
scripted client has no victim's cookie jar for a minted session to act on, so
nothing legitimate is lost by requiring it.
"""

from __future__ import annotations

import logging
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote, urlsplit

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.datastructures import FormData

from ..app import (
    AuthSettings,
    get_attempt_throttle,
    get_revocation_store,
    get_session_codec,
    get_settings,
)
from ..exceptions import InvalidSessionError
from ..passwords import generation_tag, verify_password
from ..return_to import safe_return_to
from ..revocation import RevocationStore
from ..sessions import SESSION_COOKIE_NAME, SessionCodec, SessionState
from ..throttle import AttemptThrottle

logger = logging.getLogger(__name__)

router = APIRouter()

LOGIN_PATH = "/auth/login"
"""Public path nginx's ``error_page 401`` redirects browsers to, verbatim."""

OIDC_LOGIN_PATH = "/auth/oidc/login"
"""Where an OIDC deployment's ``GET`` is sent instead of a password form.

Spelled out rather than imported from
:mod:`~osprey.services.auth_sidecar.routes.oidc`: reading one string from that
module would pull Authlib and its HTTP stack into this one's import, on a
deployment that may have no OIDC in it at all. The two constants are checked
against each other by the tests instead.
"""

TEMPLATE_NAME = "login.html"
"""Rendered from the package's ``templates/`` directory."""

FIELD_USER = "user"
FIELD_NEXT = "next"
FIELD_PASSWORD = "password"
"""Form field names, matching the hidden inputs and the password input in
``login.html``. The browser test and the composed-stack test drive these."""

DENIAL_MESSAGE = "That password was not accepted. Please try again."
"""The one refusal message. Names no user, no roster and no reason: an unknown
user and a wrong password must read identically."""

THROTTLE_MESSAGE = "Too many attempts. Wait a moment and try again."
"""Shown with the 429. Carries no count and no window length beyond the
``Retry-After`` header, which is the machine-readable half."""

MAX_LOCATION_LENGTH = 2048
"""Longest ``Location`` header this route will emit, in characters.

The return-to cap alone does not bound this one. The OIDC hand-off
percent-encodes the return-to into a query parameter, and encoding a path made
of separators triples its length — so a value comfortably inside
:data:`~osprey.services.auth_sidecar.return_to.MAX_RETURN_TO_LENGTH` can still
leave here as a header wider than the proxy buffer nginx reads a response's
headers into, which the operator meets as a 502. The check is therefore made on
the string that actually goes out.
"""

MAX_USERNAME_LENGTH = 128
"""Longest username this route will echo or key a throttle window on.

A roster username is a directory segment in ``/u/<user>/`` and an environment
variable suffix; nothing near this length can be one. The bound exists because
the submitted value reaches a rendered page and a throttle key on a path that is
unauthenticated and — for the page — reached before any credential is evaluated,
so without it an anonymous request would choose how large a response the sidecar
builds.
"""

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"

_templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
"""Autoescaping Jinja2 environment over the package's templates."""

_NO_STORE_HEADERS = {"Cache-Control": "no-store"}
"""On every rendered page.

A login prompt in a shared control room must not come back out of a browser's
back-forward cache with a previous operator's username on it, and a 401 page
must never be served from cache in place of the terminal it was standing in for.
"""

_THEME_VARIABLES = (
    "--bg-primary",
    "--bg-panel",
    "--bg-elevated",
    "--text-primary",
    "--text-secondary",
    "--text-muted",
    "--color-accent",
    "--color-error",
    "--border-default",
    "--border-accent",
    "--shadow-dropdown",
)
"""The design-system custom properties ``login.html`` uses, read by name.

Read by name rather than copied as a block for the reason the landing page does
the same: the page is standalone and wants only what it uses, and a renamed
token must surface as a fallback-with-a-warning rather than as a deployment
quietly off-palette.
"""

_DEFAULT_THEME_FAMILY = "main"
"""Which family's dark/light pair to bake in.

The sidecar has no theme setting of its own — it authenticates, it does not
render terminals — so the login page wears the framework default rather than
inventing a configuration surface for one page.
"""

_SAFE_CSS_VALUE_RE = re.compile(r"[#A-Za-z0-9 ,.()%/_-]+")
"""What a baked custom-property value may look like.

Values come from the framework's own token tree, so this is belt and braces on a
string that lands inside a ``<style>`` element — where HTML escaping is no
defence against a ``</style>`` sequence, and would corrupt legitimate values.

Applied with :meth:`~re.Pattern.fullmatch` rather than ``^…$`` and
:meth:`~re.Pattern.match`: ``$`` also matches *before* a trailing newline, so an
anchored ``match`` would accept a value carrying one — and a newline is exactly
what would let a value break out of the declaration it was baked into.
"""


@lru_cache(maxsize=1)
def _theme_blocks() -> tuple[dict[str, Any], ...]:
    """The design-system token values to bake into the page, resolved once.

    Two blocks, in the shape ``login.html`` and the landing page share: the
    default family's dark values unconditionally, and its light values behind a
    ``prefers-color-scheme`` query, so the login page follows the viewer's OS the
    way the terminals behind it do.

    Returns:
        Block dicts, or an empty tuple when the token tree cannot be read or
        carries a value this module will not put in a stylesheet. Empty is not a
        failure the operator has to act on: every ``var()`` in the template
        carries a literal fallback, so the page still renders in the default
        dark palette.
    """
    try:
        from osprey.interfaces.design_system.theme_config import (
            load_theme_registry,
            theme_css_variables,
        )

        _, defaults = load_theme_registry()
        family = defaults[_DEFAULT_THEME_FAMILY]
        wanted = (
            (None, "dark", family["dark"]),
            ("(prefers-color-scheme: light)", "light", family["light"]),
        )
        blocks = []
        for media, color_scheme, theme_id in wanted:
            values = theme_css_variables(theme_id, _THEME_VARIABLES)
            for name, value in values.items():
                if not _SAFE_CSS_VALUE_RE.fullmatch(value):
                    raise ValueError(f"{name} in theme {theme_id!r} is not a safe CSS value")
            blocks.append(
                {
                    "media": media,
                    "color_scheme": color_scheme,
                    "variables": tuple(values.items()),
                }
            )
    except (ImportError, OSError, KeyError, ValueError) as exc:
        # KeyError also covers MissingThemeVariableError, which is one — a
        # renamed or dangling token. None of these stop a login: the page falls
        # back to the palette baked into its own stylesheet.
        logger.warning(
            "login page is using its built-in palette: the design-system tokens could not "
            "be resolved (%s)",
            type(exc).__name__,
        )
        return ()
    return tuple(blocks)


def _only(values: list[str] | None) -> str | None:
    """The single non-empty string in ``values``, or ``None``.

    Several values are refused rather than resolved, for the reason
    :func:`~osprey.services.auth_sidecar.routes.verify.verify` refuses them: a
    login that took the first of ``?user=bob&user=alice`` would let the answer
    depend on which end a parser happens to pick.
    """
    if not values or len(values) != 1:
        return None
    return values[0] or None


def _page(
    request: Request,
    *,
    user: str,
    target: str,
    status_code: int = 200,
    error: str | None = None,
    headers: dict[str, str] | None = None,
) -> Response:
    """Render the password prompt.

    The same template for the first prompt, a refusal and a throttled attempt —
    only ``status_code`` and ``error`` differ, so no refusal can be told from
    another by its shape.

    Args:
        request: The inbound request, which Starlette's template response needs.
        user: The roster user named by the link or the form, echoed into the
            prompt. Escaped by the template, never trusted as markup.
        target: The already-validated return-to, carried in a hidden field.
        status_code: 200 for the prompt, 401 for a refusal, 429 when throttled.
        error: The message to show, or ``None`` for a first prompt.
        headers: Extra response headers, merged over the no-store default.

    Returns:
        The rendered page.
    """
    return _templates.TemplateResponse(
        request,
        TEMPLATE_NAME,
        {
            "user": user,
            "next": target,
            "error": error,
            "login_path": LOGIN_PATH,
            "theme_blocks": _theme_blocks(),
        },
        status_code=status_code,
        headers={**_NO_STORE_HEADERS, **(headers or {})},
    )


def _current_session(
    request: Request, codec: SessionCodec, revocations: RevocationStore
) -> SessionState:
    """The browser's auth session, or a fresh one.

    An unreadable cookie — tampered, stale-keyed, or from a retired payload
    version — is not an error here: it carries no authorisation, so the login it
    is about to be replaced by starts from an empty session. A *readable* one is
    kept so that unlocking a second user leaves the first one unlocked, which is
    the whole point of a set-valued session.

    **A revoked session id is not inherited.** Logout revokes by id, and verify
    refuses every cookie carrying a revoked one — so a login that kept the id
    would hand back a freshly signed cookie that verify then rejects, and the
    next login would inherit it again. The browser would be locked out of a
    terminal it just authenticated for, until the entry expired, with nothing in
    the response saying why. It is reachable as an ordinary logout/login race,
    so the id is retired here rather than carried.
    """
    raw = request.cookies.get(SESSION_COOKIE_NAME)
    if not raw:
        return codec.new_state()
    try:
        state = codec.decode(raw)
    except InvalidSessionError:
        logger.info("discarding an unreadable auth session cookie during login")
        return codec.new_state()

    if revocations.is_revoked(state.session_id):
        logger.info("starting a new session: the cookie presented at login had been revoked")
        return codec.new_state()
    return state


def _oidc_destination(user: str, target: str) -> str:
    """The OIDC login URL this route hands a browser off to."""
    return (
        f"{OIDC_LOGIN_PATH}?{FIELD_USER}={quote(user, safe='')}"
        f"&{FIELD_NEXT}={quote(target, safe='')}"
    )


def _expected_origin(request: Request, settings: AuthSettings) -> str | None:
    """The one origin a login form may legitimately have been submitted from.

    The deployment's declared
    :attr:`~osprey.services.auth_sidecar.app.AuthSettings.external_origin` is the
    authority wherever it exists. It is derived at render time from the
    deployment's own configuration — the same value the OIDC ``redirect_uri`` is
    built from — so it is the deployment's own word about where it lives, not a
    client's.

    It is optional in password mode, though, and a deployment without one still
    has to be able to log people in. There the origin the request was *addressed
    to* stands in (``Host``, with ``X-Forwarded-Proto`` for the scheme nginx
    received). That is weaker — ``Host`` is client-supplied — but not weak
    against the attack this check exists for: a cross-site form makes the
    victim's browser set ``Host`` from the URL it is posting *to* while
    ``Origin`` names the page it is posting *from*, and an attacker can choose
    neither. A client that forges both is not a browser and has no victim's
    cookie jar to act on.

    Args:
        request: The inbound request, as nginx forwarded it.
        settings: The deployment's frozen settings.

    Returns:
        A lowercased ``scheme://host``, or ``None`` when neither source yields
        one — in which case the caller refuses, because a check with nothing to
        compare against is not a check.
    """
    if settings.external_origin:
        return settings.external_origin.rstrip("/").lower()

    forwarded = request.headers.get("x-forwarded-proto") or ""
    # A proxy chain appends rather than replaces; the first entry is the scheme
    # the browser used.
    scheme = (forwarded.split(",")[0].strip() or request.url.scheme).lower()
    host = (request.headers.get("host") or request.url.netloc).strip().lower()
    return f"{scheme}://{host}" if host else None


def _submission_origin(request: Request) -> str | None:
    """Where this POST says it was submitted from, or ``None`` if it will not say.

    ``Origin`` first: browsers send it on every cross-origin POST and, for over a
    decade, on same-origin ones too, and it cannot be set by page script.
    ``Referer`` is the fallback for the older same-origin behaviour, read for its
    origin alone — the path is neither needed nor trusted.

    A literal ``null`` origin (a sandboxed iframe, some redirect chains) counts as
    "will not say" rather than as an origin: it matches no deployment, and
    treating it as a value would only obscure why the refusal happened.
    """
    origin = (request.headers.get("origin") or "").strip()
    if origin and origin.lower() != "null":
        return origin.rstrip("/").lower()

    referer = (request.headers.get("referer") or "").strip()
    if referer:
        parts = urlsplit(referer)
        if parts.scheme and parts.netloc:
            return f"{parts.scheme}://{parts.netloc}".lower()
    return None


def _cross_site_post(request: Request, settings: AuthSettings) -> bool:
    """Whether this POST may not be treated as a submission of *this* login form.

    Fail closed on both halves. A POST that declares no origin at all is refused
    rather than admitted: every browser this login page can be used from sends
    one, so the only callers a permissive branch would serve are scripted clients
    — which have no victim's cookie jar and therefore nothing legitimate to gain
    from the sidecar minting a session. A deployment that can name no origin of
    its own is refused for the same reason: a comparison with nothing on one side
    is not a comparison.
    """
    submitted = _submission_origin(request)
    expected = _expected_origin(request, settings)
    return submitted is None or expected is None or submitted != expected


def _form_values(form: FormData, field: str) -> list[str]:
    """Every string value submitted under ``field``.

    Non-string parts (an upload posted under a field this form has no file
    input for) are dropped, so a multipart body cannot make a value look like a
    username.
    """
    return [value for value in form.getlist(field) if isinstance(value, str)]


@router.get(LOGIN_PATH)
async def login_page(
    request: Request,
    settings: Annotated[AuthSettings, Depends(get_settings)],
    user: Annotated[list[str] | None, Query()] = None,
    return_to: Annotated[list[str] | None, Query(alias=FIELD_NEXT)] = None,
) -> Response:
    """Show the password prompt for one roster user.

    Both parameters are optional and list-typed on purpose: required ones would
    make a malformed link draw FastAPI's 422 validation body — a different
    status, a different shape, and a description of this endpoint's parameters
    that nothing needs — and a scalar ``user`` would silently resolve a repeated
    parameter instead of refusing it.

    ``next`` may legitimately arrive empty: nginx sanitises the original path
    through an allowlist and emits nothing when it does not survive, in which
    case the login returns the browser to ``/u/<user>/``.

    An over-long ``user`` is refused here rather than carried: no roster name
    approaches :data:`MAX_USERNAME_LENGTH`, and both onward paths from this
    handler — the rendered page and the OIDC hand-off's percent-encoded
    ``Location`` — would otherwise let an anonymous link decide how much this
    service emits.

    Args:
        request: The inbound request.
        settings: The deployment's frozen settings.
        user: The roster user whose card was clicked. Exactly one value.
        return_to: Where to send the browser afterwards, if the link carried a
            usable one.

    Returns:
        The rendered prompt in password mode, or a redirect to the OIDC login
        route on a deployment that authenticates that way — nginx's 401 handler
        sends every method's browsers here, and a password form is unanswerable
        under OIDC.

    Raises:
        HTTPException: 400 when the link does not name exactly one user; 404
            when this deployment serves no password login, or when ``user`` is
            not on the roster.
    """
    username = _only(user)
    if username is None:
        # Counts and lengths, never values: this is the one part of the request a
        # client chose freely.
        logger.warning(
            "login page requested without exactly one user parameter (got %d)", len(user or [])
        )
        raise HTTPException(
            status_code=400,
            detail="the login link must name exactly one user",
            headers=_NO_STORE_HEADERS,
        )

    if len(username) > MAX_USERNAME_LENGTH:
        logger.warning(
            "login page requested for an impossible username of %d characters", len(username)
        )
        raise HTTPException(
            status_code=400, detail="that is not a username", headers=_NO_STORE_HEADERS
        )

    target = safe_return_to(_only(return_to), username)

    if settings.method == "oidc":
        # The cross-method hand-off. Roster membership and identity mapping are
        # the OIDC route's decisions, so this redirects without pre-judging
        # either — it only ensures the browser reaches a login it can complete.
        destination = _oidc_destination(username, target)
        if len(destination) > MAX_LOCATION_LENGTH:
            # Percent-encoding a path of separators triples it, so a return-to
            # inside its own cap can still leave here as a header the proxy will
            # not carry. Measured on the value that actually goes out, and the
            # answer is the same as for any unusable return-to: the user's own
            # terminal.
            destination = _oidc_destination(username, f"/u/{username}/")
        return RedirectResponse(destination, status_code=302, headers=_NO_STORE_HEADERS)

    if settings.method != "password":
        # Unreachable behind the configuration guard, which admits only the two
        # supported methods. Written as the stricter default anyway: an
        # unexpected method must not fall through into a password prompt.
        raise HTTPException(
            status_code=404,
            detail="this deployment does not use password login",
            headers=_NO_STORE_HEADERS,
        )

    if not settings.knows_user(username):
        # No credential is evaluated on this path and the roster is public — the
        # landing page lists it — so the honest answer is that there is no such
        # login page, matching what the OIDC login route says. The POST path,
        # where a credential *is* evaluated, deliberately does not distinguish.
        raise HTTPException(status_code=404, detail="no such user", headers=_NO_STORE_HEADERS)

    return _page(request, user=username, target=target)


@router.post(LOGIN_PATH)
async def login_submit(
    request: Request,
    settings: Annotated[AuthSettings, Depends(get_settings)],
    codec: Annotated[SessionCodec, Depends(get_session_codec)],
    throttle: Annotated[AttemptThrottle, Depends(get_attempt_throttle)],
    revocations: Annotated[RevocationStore, Depends(get_revocation_store)],
) -> Response:
    """Evaluate one password attempt and, on success, unlock that user.

    The form is read off the request rather than declared as FastAPI ``Form``
    parameters, for the reason the query parameters above are optional: a
    missing or repeated field must produce this route's own refusal, never a 422
    validation body. It also makes the "exactly one value" rule explicit for the
    username, which is the field whose ambiguity would matter.

    Order is load-bearing, twice over. The ``Origin`` check comes before
    everything — a cross-site form must not be able to burn an operator's
    throttle window, let alone reach a credential — and the throttle is consulted
    before the credential is looked at, so a refused attempt costs a dict lookup
    instead of a key derivation and never grows the window. Only an attempt this
    route actually evaluated records a failure.

    On success the user joins — rather than replaces — the browser's unlocked
    set, carrying an expiry stamped from the codec's clock and the
    credential-generation tag of the hash it was checked against. The tag is
    what makes a later ``osprey deploy passwd`` retire this session: verify
    recomputes it from the *current* stored hash and refuses when they differ.
    The username goes into the entry exactly as submitted, because verify
    exact-matches it against the roster — normalising it here would open the
    chain rather than close it.

    Args:
        request: The inbound request, carrying the submitted form.
        settings: The deployment's frozen settings.
        codec: The app's one session codec, and the one clock every expiry in
            the session path is stamped from.
        throttle: The app's one login-attempt throttle.
        revocations: The app's one revocation store, so a session id logout has
            retired is not carried into the cookie this login issues.

    Returns:
        A 303 to the validated return-to carrying the re-issued session cookie;
        the prompt again with 401 and one fixed message when the credential is
        refused; the prompt with 429 and ``Retry-After`` when the attempt
        arrived inside an open window.

    Raises:
        HTTPException: 400 when the form arrives from another origin or does not
            name exactly one user; 404 when this deployment serves no password
            login.
    """
    if settings.method != "password":
        raise HTTPException(
            status_code=404,
            detail="this deployment does not use password login",
            headers=_NO_STORE_HEADERS,
        )

    if _cross_site_post(request, settings):
        # Before the form is even read: this request is not a login attempt at
        # all, so it may not touch the throttle, and the refusal is the shape a
        # malformed form already produces rather than a new one to probe with.
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
    submitted_users = _form_values(form, FIELD_USER)
    username = _only(submitted_users)
    if username is None:
        logger.warning(
            "login submitted without exactly one user field (got %d)", len(submitted_users)
        )
        raise HTTPException(
            status_code=400,
            detail="the login form must name exactly one user",
            headers=_NO_STORE_HEADERS,
        )

    # An over-long name cannot be a roster user, and it must not be *looked up*
    # as one either: a prefix of it could collide with a real roster name and
    # unlock that user's credential under a name the browser never sent. So it is
    # refused on the ordinary deny path below (`stored` forced to None), while
    # the bounded form is what gets echoed and what keys the throttle window —
    # neither the page nor the throttle's memory is the caller's to size.
    oversize = len(username) > MAX_USERNAME_LENGTH
    shown = username[:MAX_USERNAME_LENGTH] if oversize else username
    if oversize:
        logger.warning("login submitted an impossible username of %d characters", len(username))

    target = safe_return_to(_only(_form_values(form, FIELD_NEXT)), shown)
    # A missing, repeated or non-string password is simply not a password: it
    # takes the ordinary refusal path rather than a distinguishable one, and
    # `verify_password` answers False for an empty string without deriving a key.
    password = _only(_form_values(form, FIELD_PASSWORD)) or ""

    delay = throttle.retry_after(shown)
    if delay > 0:
        # Before evaluation, and without recording a failure: counting attempts
        # that were never evaluated is how a flood of parallel guesses would
        # ratchet the window against the operator it protects.
        logger.info(
            "login attempt for %r refused unevaluated: the attempt window is still open", username
        )
        return _page(
            request,
            user=shown,
            target=target,
            status_code=429,
            error=THROTTLE_MESSAGE,
            headers={"Retry-After": str(math.ceil(delay))},
        )

    stored = None if oversize else settings.password_hash(username)
    # `stored is None` covers a roster user with no provisioned credential, a
    # username that is not on the roster at all, and one too long to be either.
    # None derives a key — see the module docstring — and all three leave through
    # the identical refusal below, including the same call into the throttle.
    if stored is None or not verify_password(password, stored):
        throttle.record_failure(shown)
        logger.warning("login refused for %r: the submitted credential did not verify", shown)
        return _page(request, user=shown, target=target, status_code=401, error=DENIAL_MESSAGE)

    # `shown` here too, not `username`: on this path they are the same string —
    # an over-long name never gets a stored hash — and keying both throttle calls
    # the same way is what makes "the window this attempt was checked against" and
    # "the window it clears" provably the same entry.
    throttle.record_success(shown)
    now = codec.now()
    session = _current_session(request, codec, revocations).with_user(
        username,
        expires_at=now + settings.session_lifetime,
        generation_tag=generation_tag(stored),
    )

    response = RedirectResponse(target, status_code=303, headers=_NO_STORE_HEADERS)
    response.set_cookie(
        SESSION_COOKIE_NAME,
        codec.encode(session),
        httponly=True,
        samesite="lax",
        secure=settings.tls_enabled,
        # Not the state cookie's "/auth": this one has to reach the terminals it
        # authorises, whose verify subrequest nginx issues from "/u/<user>/".
        # No Max-Age either — a browser-session cookie whose real lifetime is the
        # signed expiry inside it, which is the only one the sidecar can enforce.
        path="/",
    )
    logger.info("password login succeeded for %r", username)
    return response
