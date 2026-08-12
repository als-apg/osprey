"""Auth sidecar FastAPI application factory.

One process per multi-user deployment, sitting beside nginx: it answers the
``auth_request`` subrequest for every ``/u/<user>/`` location and serves the
login flow at ``/auth/*``. Served as ``uvicorn
osprey.services.auth_sidecar.app:create_app --factory``.

**Everything it needs arrives as environment.** The compose service reads the
non-secret settings from the rendered ``modules.web_terminals.auth`` stanza and
the secrets from ``.env.auth`` (0600, sidecar-only ``env_file``); nothing is
read from ``config.yml`` here and no credential ever enters a rendered
artifact. :class:`AuthSettings` is the single parser for that surface — routes
read ``request.app.state.settings`` (or depend on :func:`get_settings`) rather
than reaching into :data:`os.environ` themselves. The shared per-process
objects are built here for the same reason — the session codec
(:func:`get_session_codec`), the revocation store
(:func:`get_revocation_store`) and the login-attempt throttle
(:func:`get_attempt_throttle`). Each only means anything if every route touches
the same instance: four differently-parameterised codecs (or four clocks) let
expiries disagree, a second revocation store is a logout nothing checks, and a
second throttle counts every attempt as the first. Anything else shared across
routes belongs here too, on the same pattern — built once in the factory,
reached through an accessor.

Per-user values are keyed by
:func:`~osprey.deployment.web_terminals.personas.env_var_suffix` — the one
definition of the username→env-var mapping, shared with credential
provisioning and lint so a password can never be keyed one way at mint time and
another at verify time.

**Fail closed, but still answer the healthcheck.** A misconfigured sidecar must
lock people out rather than let them in, so :func:`create_app` never raises:
the app starts, ``/health`` reports ``configured: false``, and every other path
— routes that do not exist yet included, over HTTP and websockets alike — is
refused by :class:`ConfigurationGuard` before it runs. That ordering is
deliberate: a route author cannot forget to declare the check, because the
check does not live in the routes.

**Session middleware is pinned here, not defaulted.** Authlib's Starlette
client keeps its OIDC ``state``/``nonce`` in a Starlette session, whose defaults
(14-day lifetime, ``path=/``, no ``Secure``) are all wrong for a
short-lived cross-site handshake artifact. The pins live next to the settings
that feed them; see :data:`STATE_COOKIE_NAME` and friends. It is installed only
when a real state secret was minted — never under a stand-in key, so a missing
secret surfaces as a loud error at the first line that touches a session rather
than as cookies signed by a key that dies with the process.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.sessions import SessionMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from osprey.deployment.web_terminals.personas import env_var_suffix, env_var_suffix_collisions

from .revocation import RevocationStore
from .sessions import SessionCodec
from .throttle import AttemptThrottle

logger = logging.getLogger(__name__)

SERVICE_NAME = "auth-sidecar"
"""Reported by ``/health``; matches the compose service's name suffix."""

HEALTH_PATH = "/health"
"""The one path :class:`ConfigurationGuard` lets through unconfigured.

nginx never proxies it — it exists for the container healthcheck and for
``osprey up``'s post-up smoke check, both of which talk to the sidecar's own
port directly.
"""

SUPPORTED_METHODS = ("password", "oidc")
"""Methods this process can actually serve.

``none`` is a supported *deployment* posture (see ``render.py``'s
``SUPPORTED_AUTH_METHODS``) but not a supported sidecar mode: with auth off
there is no sidecar in the compose file at all, so a sidecar that finds
``OSPREY_AUTH_METHOD=none`` has been mis-wired and refuses everything.
"""

# --- OIDC state cookie (Starlette SessionMiddleware) -------------------------
# Pinned, never configurable: this cookie carries only the in-flight OIDC
# handshake state, and every attribute below is a security property of that
# handshake rather than a facility preference.

STATE_COOKIE_NAME = "osprey_auth_state"
"""Distinct from the auth session cookie, so logout/rotation never disturb it."""

STATE_COOKIE_PATH = "/auth"
"""The only prefix the handshake runs under — the cookie is never sent to
``/u/<user>/`` or to the verify subrequest."""

STATE_COOKIE_MAX_AGE = 300
"""Seconds. An abandoned handshake expires in minutes, not Starlette's 14 days."""

STATE_COOKIE_SAME_SITE: Literal["lax", "strict", "none"] = "lax"
"""``lax`` (not ``strict``): the IdP redirects back with a top-level GET, which
``strict`` would strip the cookie from."""

# --- Environment variable names ---------------------------------------------

ENV_METHOD = "OSPREY_AUTH_METHOD"
ENV_SESSION_SECRET = "OSPREY_AUTH_SESSION_SECRET"
ENV_STATE_SECRET = "OSPREY_AUTH_STATE_SECRET"
ENV_SESSION_LIFETIME = "OSPREY_AUTH_SESSION_LIFETIME"
ENV_TLS_ENABLED = "OSPREY_AUTH_TLS_ENABLED"
ENV_EXTERNAL_ORIGIN = "OSPREY_AUTH_EXTERNAL_ORIGIN"
ENV_USERS = "OSPREY_AUTH_USERS"

ENV_PW_HASH_PREFIX = "OSPREY_AUTH_PW_HASH_"
"""Per-user stored hash: ``OSPREY_AUTH_PW_HASH_<SUFFIX>``."""

ENV_OIDC_ISSUER = "OSPREY_AUTH_OIDC_ISSUER"
ENV_OIDC_CLIENT_ID_ENV = "OSPREY_AUTH_OIDC_CLIENT_ID_ENV"
ENV_OIDC_CLIENT_SECRET_ENV = "OSPREY_AUTH_OIDC_CLIENT_SECRET_ENV"
ENV_OIDC_CLAIM = "OSPREY_AUTH_OIDC_CLAIM"
ENV_OIDC_SUBJECT_PREFIX = "OSPREY_AUTH_OIDC_SUBJECT_"
"""Per-user expected IdP identity: ``OSPREY_AUTH_OIDC_SUBJECT_<SUFFIX>``."""

DEFAULT_OIDC_CLIENT_ID_ENV = "OSPREY_AUTH_OIDC_CLIENT_ID"
DEFAULT_OIDC_CLIENT_SECRET_ENV = "OSPREY_AUTH_OIDC_CLIENT_SECRET"
DEFAULT_OIDC_CLAIM = "sub"
DEFAULT_SESSION_LIFETIME = 12 * 60 * 60

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})

# Route modules included when they exist, in this order. Each is expected to
# expose a module-level ``router``. A name that has not landed yet is skipped —
# and *only* that case is skipped: an ImportError raised from inside a module
# that does exist propagates, so a broken route can never degrade into a
# silently missing one.
_ROUTE_MODULES = ("verify", "login", "logout", "oidc")
_ROUTES_PACKAGE = f"{__package__}.routes"


def _flag(raw: str | None, default: bool = False) -> bool:
    """Read an env flag; anything unrecognised falls back to ``default``."""
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in _TRUE_VALUES


def _is_unrecognized_flag(raw: str | None) -> bool:
    """Whether ``raw`` is a non-empty value :func:`_flag` does not understand.

    ``OSPREY_AUTH_TLS_ENABLED=maybe`` reads as false, which silently downgrades
    the ``Secure`` attribute on both cookies — the deployment believes it is
    serving TLS and the cookies say otherwise. The value is still treated as
    false (guessing "they probably meant yes" would be worse), but the caller
    warns loudly rather than letting the typo pass unremarked.
    """
    if raw is None or not raw.strip():
        return False
    return raw.strip().lower() not in (_TRUE_VALUES | _FALSE_VALUES)


def _positive_int(raw: str | None, default: int) -> int:
    """Read a positive int from the environment, falling back to ``default``."""
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _roster(raw: str | None) -> tuple[str, ...]:
    """Parse the comma-separated roster, dropping blanks and repeats.

    Order is preserved: it is the roster's declaration order, which the login
    page reuses.
    """
    seen: list[str] = []
    for chunk in (raw or "").split(","):
        name = chunk.strip()
        if name and name not in seen:
            seen.append(name)
    return tuple(seen)


@dataclass(frozen=True)
class AuthSettings:
    """The sidecar's whole configuration surface, parsed once at startup.

    Built by :meth:`from_env` and cached on ``app.state.settings``. Frozen: the
    values are read at process start and a change to any of them (a rotated
    password above all) reaches the sidecar only through a container
    force-recreate, because compose bakes ``env_file`` content at container
    *creation*. Reading the environment lazily per request would suggest a
    liveness this deployment does not have.

    **Every secret-bearing field is excluded from the generated ``repr``.** A
    dataclass repr is one ``logger.debug("settings=%s", settings)``, one
    unhandled traceback frame, or one debugger session away from printing the
    session signing key into a log aggregator. The values are still readable as
    attributes by code that needs them; they simply never render by accident.
    The variable *names* (``oidc_client_id_var``) and the OIDC subject map stay
    visible — those are configuration, not credentials, and are exactly what an
    operator needs to see when diagnosing a lockout.

    Attributes:
        method: ``password`` or ``oidc``. Any other value — including the
            ``none`` posture, which means no sidecar should be running at all —
            leaves the app unconfigured and refusing.
        session_secret: Signing key for the auth session cookie. Empty when
            unset, which is a refusal condition.
        state_secret: Signing key for the OIDC state cookie, kept separate from
            ``session_secret`` so the short-lived handshake artifact and the
            long-lived session never share a key.
        session_lifetime: How long an unlocked-user entry stays valid, seconds.
        tls_enabled: Whether the public origin is HTTPS. Drives the ``Secure``
            attribute on both cookies.
        external_origin: Scheme://host[:port] the browser reaches nginx on,
            derived once at render time and shared with the landing page. The
            OIDC ``redirect_uri`` is built from it.
        users: Roster usernames, in declaration order.
        password_hashes: ``{username: stored hash}`` for the roster users that
            have one. A roster user with no entry is absent, never empty-string
            keyed — :meth:`password_hash` returns ``None`` and the caller denies.
        oidc_issuer: Discovery issuer URL, or ``None``.
        oidc_client_id: The OIDC client id itself, resolved through the
            indirection described in :meth:`from_env`.
        oidc_client_secret: Likewise the client secret.
        oidc_client_id_var: The env-var name the client id was read from — the
            resolved name, not the default, so a refusal message points at the
            variable this deployment actually uses.
        oidc_client_secret_var: Likewise for the client secret.
        oidc_claim: Which ID-token claim carries the identity to map onto a
            roster user.
        oidc_subjects: ``{username: expected claim value}`` from the roster.
    """

    method: str = "none"
    session_secret: str = field(default="", repr=False)
    state_secret: str = field(default="", repr=False)
    session_lifetime: int = DEFAULT_SESSION_LIFETIME
    tls_enabled: bool = False
    external_origin: str = ""
    users: tuple[str, ...] = ()
    password_hashes: Mapping[str, str] = field(default_factory=dict, repr=False)
    oidc_issuer: str | None = None
    oidc_client_id: str = ""
    oidc_client_secret: str = field(default="", repr=False)
    oidc_client_id_var: str = DEFAULT_OIDC_CLIENT_ID_ENV
    oidc_client_secret_var: str = DEFAULT_OIDC_CLIENT_SECRET_ENV
    oidc_claim: str = DEFAULT_OIDC_CLAIM
    oidc_subjects: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> AuthSettings:
        """Parse settings out of ``env`` (default :data:`os.environ`).

        Never raises: a malformed value falls back to its default and shows up
        as a refusal in :meth:`missing_requirements`, because a sidecar that
        dies on import is a sidecar whose healthcheck cannot explain why
        everyone is locked out.

        The OIDC client credentials are read through one level of indirection:
        the config names the *variable* holding each credential
        (``OSPREY_AUTH_OIDC_CLIENT_ID_ENV``, defaulting to
        :data:`DEFAULT_OIDC_CLIENT_ID_ENV`), and the credential itself lives in
        ``.env.auth`` under that name. A facility that already ships its IdP
        client id under its own variable name therefore does not have to
        duplicate it.

        Args:
            env: Environment mapping to read. Defaults to the process
                environment.

        Returns:
            The parsed settings.
        """
        source: Mapping[str, str] = os.environ if env is None else env

        users = _roster(source.get(ENV_USERS))
        password_hashes: dict[str, str] = {}
        oidc_subjects: dict[str, str] = {}
        for user in users:
            suffix = env_var_suffix(user)
            stored = (source.get(f"{ENV_PW_HASH_PREFIX}{suffix}") or "").strip()
            if stored:
                password_hashes[user] = stored
            subject = (source.get(f"{ENV_OIDC_SUBJECT_PREFIX}{suffix}") or "").strip()
            if subject:
                oidc_subjects[user] = subject

        client_id_var = (
            source.get(ENV_OIDC_CLIENT_ID_ENV) or ""
        ).strip() or DEFAULT_OIDC_CLIENT_ID_ENV
        client_secret_var = (
            source.get(ENV_OIDC_CLIENT_SECRET_ENV) or ""
        ).strip() or DEFAULT_OIDC_CLIENT_SECRET_ENV

        return cls(
            method=(source.get(ENV_METHOD) or "none").strip().lower() or "none",
            session_secret=(source.get(ENV_SESSION_SECRET) or "").strip(),
            state_secret=(source.get(ENV_STATE_SECRET) or "").strip(),
            session_lifetime=_positive_int(
                source.get(ENV_SESSION_LIFETIME), DEFAULT_SESSION_LIFETIME
            ),
            tls_enabled=_flag(source.get(ENV_TLS_ENABLED)),
            external_origin=(source.get(ENV_EXTERNAL_ORIGIN) or "").strip().rstrip("/"),
            users=users,
            password_hashes=password_hashes,
            oidc_issuer=(source.get(ENV_OIDC_ISSUER) or "").strip() or None,
            oidc_client_id=(source.get(client_id_var) or "").strip(),
            oidc_client_secret=(source.get(client_secret_var) or "").strip(),
            oidc_client_id_var=client_id_var,
            oidc_client_secret_var=client_secret_var,
            oidc_claim=(source.get(ENV_OIDC_CLAIM) or "").strip() or DEFAULT_OIDC_CLAIM,
            oidc_subjects=oidc_subjects,
        )

    def missing_requirements(self) -> tuple[str, ...]:
        """Names of the settings that must be present before auth can be served.

        Deliberately *not* a per-user check: a roster user with no password
        hash, or with no mapped OIDC subject, is denied individually by the
        verify/callback routes. Only deployment-wide gaps — the ones that would
        make every answer meaningless — belong here, since anything named here
        takes the whole sidecar down to 503.

        The two OIDC client credentials are reported under their *resolved*
        variable names (:attr:`oidc_client_id_var`), not the defaults: a
        deployment that pointed the config at ``FACILITY_IDP_CLIENT`` must be
        told to set ``FACILITY_IDP_CLIENT``, or the message sends an operator to
        set a variable nothing reads.

        The two session-codec preconditions — a non-blank signing secret and a
        positive lifetime — are checked here rather than trusted from
        :meth:`from_env`'s normalization. ``configured`` is what
        :func:`create_app` consults before building the
        :class:`~osprey.services.auth_sidecar.sessions.SessionCodec`, and that
        codec raises on either. Checking the values themselves makes
        "servable implies constructible" true of any :class:`AuthSettings`,
        however it was built, instead of true only of ones that came through
        the parser — otherwise a directly-constructed
        ``AuthSettings(method="password", session_secret="   ")`` reports
        servable and then kills the factory, taking the healthcheck that would
        have explained it down too.

        Returns:
            Env-var names (never values), sorted for a stable log line. Empty
            when the sidecar can serve.
        """
        missing: set[str] = set()
        if self.method not in SUPPORTED_METHODS:
            missing.add(ENV_METHOD)
        if not self.session_secret.strip():
            missing.add(ENV_SESSION_SECRET)
        if self.session_lifetime <= 0:
            missing.add(ENV_SESSION_LIFETIME)
        if self.method == "oidc":
            if not self.state_secret.strip():
                missing.add(ENV_STATE_SECRET)
            if not self.oidc_issuer:
                missing.add(ENV_OIDC_ISSUER)
            if not self.external_origin:
                missing.add(ENV_EXTERNAL_ORIGIN)
            if not self.oidc_client_id:
                missing.add(self.oidc_client_id_var)
            if not self.oidc_client_secret:
                missing.add(self.oidc_client_secret_var)
        return tuple(sorted(missing))

    def roster_collisions(self) -> dict[str, list[str]]:
        """Roster usernames that key the same per-user env vars.

        ``alice-b`` and ``alice_b`` both map to the suffix ``ALICE_B``, so they
        would share one ``OSPREY_AUTH_PW_HASH_ALICE_B`` entry — one operator's
        password opening another's terminal, which is precisely the isolation
        this service exists to establish. There is no safe way to serve such a
        roster, so a collision refuses the whole sidecar rather than guessing
        which user the credential belongs to.

        Returns:
            ``{suffix: [colliding usernames]}``, empty when the roster is
            unambiguous. Detection is
            :func:`~osprey.deployment.web_terminals.personas.env_var_suffix_collisions`,
            shared with lint and credential provisioning.
        """
        return env_var_suffix_collisions(self.users)

    @property
    def configured(self) -> bool:
        """Whether the sidecar can serve anything beyond ``/health``."""
        return not self.missing_requirements() and not self.roster_collisions()

    def password_hash(self, user: str) -> str | None:
        """The stored hash for ``user``, or ``None`` if the roster has none."""
        return self.password_hashes.get(user)

    def oidc_subject(self, user: str) -> str | None:
        """The IdP identity mapped to ``user``, or ``None`` if unmapped."""
        return self.oidc_subjects.get(user)

    def knows_user(self, user: str) -> bool:
        """Whether ``user`` is on this deployment's roster."""
        return user in self.users


def get_settings(request: Request) -> AuthSettings:
    """The running app's settings — usable directly or as a FastAPI dependency."""
    settings: AuthSettings = request.app.state.settings
    return settings


def _require_wired(request: Request, attribute: str, description: str) -> Any:
    """Fetch a shared object built by :func:`create_app`, or fail loudly.

    The single implementation behind every shared-object accessor below, so the
    three cannot drift into three different behaviors for the same condition.

    Raises:
        RuntimeError: If the object was never built, which happens only when the
            settings are unservable. Route code never sees this — the
            configuration guard has already refused the request with 503 — so it
            is an assertion about wiring, not a case for callers to handle.
    """
    value = getattr(request.app.state, attribute, None)
    if value is None:
        raise RuntimeError(
            f"auth sidecar has no {description}; the configuration guard should have "
            "refused this request"
        )
    return value


def get_session_codec(request: Request) -> SessionCodec:
    """The app's one session codec — usable directly or as a FastAPI dependency.

    Every route that touches a session cookie goes through this rather than
    constructing its own: one signing secret, one maximum age, and above all one
    clock, so an expiry stamped by the login route is checked against the same
    time source by verify.

    Raises:
        RuntimeError: See :func:`_require_wired`.
    """
    codec: SessionCodec = _require_wired(request, "session_codec", "session codec")
    return codec


def get_revocation_store(request: Request) -> RevocationStore:
    """The app's one revocation store — usable directly or as a dependency.

    Logout writes to it and verify reads it, so they must be the same object:
    two stores would mean a logout that revokes a session nothing checks. The
    store keeps its own ``time.time`` clock, matching the absolute epoch
    expiries carried in session cookies — deliberately not the codec's clock,
    which tests may have moved.

    Raises:
        RuntimeError: See :func:`_require_wired`.
    """
    store: RevocationStore = _require_wired(request, "revocation_store", "revocation store")
    return store


def get_attempt_throttle(request: Request) -> AttemptThrottle:
    """The app's one login-attempt throttle — usable directly or as a dependency.

    A throttle is only a throttle if every attempt lands in the same one; a
    per-route or per-request instance would count each attempt as the first and
    never close a window. It runs on ``time.monotonic`` by design (see
    :mod:`~osprey.services.auth_sidecar.throttle`), so a wall-clock step cannot
    open a window early — another reason it does not share the codec's clock.

    Raises:
        RuntimeError: See :func:`_require_wired`.
    """
    throttle: AttemptThrottle = _require_wired(request, "attempt_throttle", "attempt throttle")
    return throttle


WEBSOCKET_REFUSAL_CODE = 1011
"""Close code for a websocket refused by the guard (RFC 6455 "internal error").

The honest code for "this gate cannot decide right now" — the same thing the
503 says on the HTTP side. 1008 (policy violation) would claim the request was
evaluated and rejected, which is not what happened.

What a real client sees is not this number: a close sent before the handshake
is accepted is turned by uvicorn into an HTTP 403 rejection of the upgrade, so
the socket never opens at all. The code is what the ASGI test client observes,
and the reason it is pinned in tests — treat it as the refusal's identity
in-process, not as a promise about the wire.
"""


class ConfigurationGuard:
    """Pure-ASGI middleware refusing every path but ``/health`` when unconfigured.

    Placed outside the routes on purpose. The alternative — a dependency each
    route declares — fails open the first time someone adds a route and forgets
    it, and "someone forgets" is not an acceptable failure mode for the only
    thing standing between a browser and another operator's terminal.

    **Pure ASGI, not** :class:`~starlette.middleware.base.BaseHTTPMiddleware`.
    That base class only dispatches ``http`` scopes and passes everything else
    straight through, so a websocket route on an unconfigured sidecar would
    complete its handshake and start exchanging frames past a gate that believes
    it is refusing all traffic. The sidecar serves no websocket route today,
    which is exactly why this must be structural: the failure would arrive with
    whoever adds the first one, long after this reasoning is out of view.

    Refusals are shaped per protocol:

    * ``http`` → 503 with a JSON body. nginx turns any non-2xx subrequest answer
      into a denial, and 503 says "this gate is not operational" rather than
      implying credentials were evaluated and rejected.
    * ``websocket`` → closed with :data:`WEBSOCKET_REFUSAL_CODE`, without ever
      accepting the handshake.
    * anything else (``lifespan``) → passed through untouched. Startup and
      shutdown must run whatever the configuration, or an unconfigured app
      could not even boot to serve its healthcheck.
    """

    def __init__(self, app: ASGIApp, settings: AuthSettings) -> None:
        """Wrap ``app``, refusing traffic while ``settings`` are unservable.

        Args:
            app: The next ASGI application in the stack.
            settings: The app's settings. Servability is evaluated once, here:
                the settings are frozen and read at process start, and this
                check sits in front of every request, so re-deriving it per
                request would re-scan the roster for collisions on the hot path
                to reach an answer that cannot have changed.
        """
        self.app = app
        self.settings = settings
        self._servable = settings.configured

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Refuse, or hand the scope on."""
        if (
            self._servable
            or scope["type"] not in ("http", "websocket")
            or scope.get("path") == HEALTH_PATH
        ):
            await self.app(scope, receive, send)
            return

        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": WEBSOCKET_REFUSAL_CODE})
            return

        response = JSONResponse(
            status_code=503,
            content={
                "detail": (
                    "auth sidecar is not configured; refusing all authentication "
                    "(see the service log for the missing settings)"
                )
            },
        )
        await response(scope, receive, send)


def _include_available_routers(app: FastAPI) -> None:
    """Include each route module in :data:`_ROUTE_MODULES` that exists.

    The extension point for the verify/login/logout/oidc tasks: land
    ``routes/<name>.py`` exposing a ``router``, and this factory picks it up
    with no edit here. A module that has not landed is skipped; an ImportError
    from *within* a module that has landed propagates, so a route broken by a
    bad import fails loudly at startup instead of quietly not existing.
    """
    for name in _ROUTE_MODULES:
        module_path = f"{_ROUTES_PACKAGE}.{name}"
        try:
            module = import_module(module_path)
        except ModuleNotFoundError as exc:
            if exc.name not in (module_path, _ROUTES_PACKAGE):
                raise
            continue
        app.include_router(module.router)


def create_app(env: Mapping[str, str] | None = None) -> FastAPI:
    """Create the auth sidecar's FastAPI application.

    Args:
        env: Environment mapping to configure from. Defaults to the process
            environment; tests pass an explicit mapping.

    Returns:
        A configured app, constructible whatever the environment holds. When a
        required setting is missing it still starts and still answers
        ``/health`` — every other path is refused with 503.
    """
    settings = AuthSettings.from_env(env)

    app = FastAPI(
        title="OSPREY Auth Sidecar",
        description="Authenticates multi-user web-terminal access at the nginx perimeter",
        version="1.0.0",
        # Deliberately unlike the sibling interface factories: a security
        # perimeter publishes no schema of its own routes. `app.openapi()` still
        # builds one in-process for tests; only the served endpoints are gone.
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.settings = settings
    # Evaluated once: the settings are frozen, so the answer cannot change, and
    # the healthcheck is polled continuously — no reason to re-scan the roster
    # for collisions on every poll.
    servable = settings.configured
    # Built once, here, and shared by every route through `get_session_codec`.
    # Only when the settings are servable: `SessionCodec` refuses an empty
    # secret (correctly — an unsigned cookie is a forgeable one), and the app
    # has to start regardless so the healthcheck can report why it will not
    # serve. `configured` implies a non-empty session secret, so this
    # construction cannot raise.
    app.state.session_codec = (
        SessionCodec(settings.session_secret, max_age=settings.session_lifetime)
        if servable
        else None
    )
    # The other two shared stores, same pattern and same reason. Each is
    # per-process state that only means anything if every route touches the one
    # instance: a second revocation store is a logout nothing checks, a second
    # throttle counts every attempt as the first. Both keep their own clocks
    # (`time.time` for the store, to match the absolute expiries in session
    # cookies; `time.monotonic` for the throttle, so a clock step cannot open a
    # window early) — deliberately not the codec's.
    app.state.revocation_store = RevocationStore() if servable else None
    app.state.attempt_throttle = AttemptThrottle() if servable else None

    @app.get(HEALTH_PATH)
    async def health() -> dict[str, Any]:
        """Liveness probe. Answers 200 even unconfigured, and names no secret."""
        return {
            "status": "ok",
            "service": SERVICE_NAME,
            "method": settings.method,
            "configured": servable,
        }

    _include_available_routers(app)

    # Installed only against a real minted secret. There is no ephemeral
    # fallback key: one would let a session-touching route quietly issue state
    # cookies signed by a secret that dies with the process and that no other
    # replica or restart can verify — the failure would surface as unexplainable
    # intermittent OIDC handshake errors. Without the secret the middleware is
    # simply absent, so touching `request.session` raises at the call site and
    # names the real problem.
    if settings.state_secret:
        app.add_middleware(
            SessionMiddleware,
            secret_key=settings.state_secret,
            session_cookie=STATE_COOKIE_NAME,
            path=STATE_COOKIE_PATH,
            max_age=STATE_COOKIE_MAX_AGE,
            same_site=STATE_COOKIE_SAME_SITE,
            https_only=settings.tls_enabled,
        )
    # Added last, so it is the outermost layer: a refused request reaches
    # neither the session middleware nor any route.
    app.add_middleware(ConfigurationGuard, settings=settings)

    _log_configuration(settings, env)

    return app


def _log_configuration(settings: AuthSettings, env: Mapping[str, str] | None) -> None:
    """Emit the startup breadcrumbs an operator needs to diagnose a lockout.

    Every message names variables, never values. Called once from
    :func:`create_app`, after the app is assembled.
    """
    missing = settings.missing_requirements()
    collisions = settings.roster_collisions()

    if missing:
        logger.error(
            "auth sidecar is not configured and will refuse every request except %s: "
            "unset or invalid %s",
            HEALTH_PATH,
            ", ".join(missing),
        )
    if collisions:
        # Named in full: the operator has to rename one of the users, and the
        # suffix alone would not tell them which two names collided.
        logger.error(
            "auth sidecar refuses to serve an ambiguous roster: %s. Two usernames that "
            "map to one env-var suffix would share a single credential; rename one.",
            "; ".join(
                f"{', '.join(names)} all key {ENV_PW_HASH_PREFIX}{suffix}"
                for suffix, names in sorted(collisions.items())
            ),
        )
    if settings.configured and not settings.users:
        logger.warning(
            "auth sidecar started with an empty roster (%s is unset or blank): no user "
            "can be authenticated, so every terminal stays locked",
            ENV_USERS,
        )

    source: Mapping[str, str] = os.environ if env is None else env
    if _is_unrecognized_flag(source.get(ENV_TLS_ENABLED)):
        logger.warning(
            "%s is set to a value that is neither true nor false; reading it as false, "
            "so cookies will be issued without the Secure attribute",
            ENV_TLS_ENABLED,
        )

    origin_is_https = settings.external_origin.startswith("https://")
    if settings.external_origin and origin_is_https != settings.tls_enabled:
        # Both directions are real failures, and they fail differently enough
        # that one shared message would misdiagnose whichever one it did not
        # describe.
        if origin_is_https:
            # The browser reaches this deployment over HTTPS, but every cookie
            # the sidecar issues will lack Secure, so any incidental plain-HTTP
            # request to the same host leaks them.
            logger.warning(
                "%s is an https origin but %s is false: cookies will be issued without "
                "the Secure attribute on a TLS deployment — one of the two settings is wrong",
                ENV_EXTERNAL_ORIGIN,
                ENV_TLS_ENABLED,
            )
        else:
            # Worse than a leak: a browser silently discards a Secure cookie
            # sent over http, so every login appears to succeed and nothing is
            # ever unlocked. Nothing in the response says why.
            logger.warning(
                "%s is a cleartext origin but %s is true: browsers discard Secure cookies "
                "on http, so logins will appear to succeed and unlock nothing — one of the "
                "two settings is wrong",
                ENV_EXTERNAL_ORIGIN,
                ENV_TLS_ENABLED,
            )
    elif not settings.tls_enabled:
        logger.warning(
            "auth sidecar is serving on a cleartext origin: session cookies will be "
            "issued without the Secure attribute"
        )
