"""Process-wide credentials for the OSPREY web surfaces.

Every OSPREY web interface — the web terminal, the panel hosts, the companion
apps that share its process — is reachable over HTTP, and until this module
existed none of them asked the caller for anything. That made the agent's own
sandbox a foothold: code the agent ran could reach the very server that spawned
it and drive the operator's UI. This module holds the three credentials that
close that path and the one place they are decided, so that no surface invents
its own answer:

* the **operator secret**, which authorises everything an operator can do;
* the **panel token**, a deliberately weaker credential for the narrow set of
  panel-arrangement calls in-process companions legitimately make;
* an in-memory **browser-session map**, ``{session_id: expiry}``, holding the
  sessions handed out when a browser exchanges a one-time URL token for a
  cookie.

**Population happens once per process and is idempotent.** The first caller
builds the credentials under :data:`_POPULATION_LOCK`; every later caller — on
any thread, from any app in the process — gets the same object back. That is
what lets an in-process companion app inherit the parent's credentials instead
of minting a second set that the browser holding the first one would fail
against.

**The environment read is a ``pop``, not a ``get``, and that is load-bearing.**
The agent SDK layers its own environment *over* ``os.environ`` when it spawns a
child — it builds the CLI's environment as ``{**os.environ, **options.env}`` —
so a secret left in ``os.environ`` when the server starts serving is handed
straight to the sandboxed process this module exists to keep out, and a launch
helper that merely declines to *copy* the name changes nothing there. Popping
it means the value lives in this module's memory and is no longer inherited by
a child this process spawns. An empty or whitespace-only value counts as absent: an unset compose
variable interpolates to the empty string, and treating that as a credential
would authorise every caller who also sends nothing.

**Population runs once, but the carriers are closed on every launch.** A
launcher that is about to spawn or become the server re-publishes the operator
secret (:func:`mint_and_announce`) precisely so the child inherits it; in the
*direct-serve* shape — the default ``osprey web``, and the per-user container's
``CMD`` — that "child" is this same, already-populated process, where no second
:func:`_populate` and therefore no second pop would ever run.
:func:`close_env_carriers` is what closes the window again: every interface app
calls it at construction (see
:func:`osprey.interfaces._app_setup.configure_interface_app`), so whichever way
the process was started, neither carrier is in ``os.environ`` by the time it is
answering requests and spawning agents.

**What the pop does not reach.** A process that received the secret through
``execve`` — the ``--reload`` worker, the ``--detach`` child, the per-user
container — also keeps the exec-time environment block the kernel recorded for
it. On Linux that block is readable at ``/proc/<pid>/environ`` by any process
running as the same UID, which includes the agent this module is keeping the
secret from; ``os.environ.pop`` edits the interpreter's mapping and the
``environ(7)`` array, not that snapshot. So the pop closes the *inheritance*
path — the one an SDK overlay travels — and does not make the value unreadable
to a same-UID reader on those shapes. Closing that residual needs either a
privilege split (a container user the agent cannot become) or a handoff that
never touches the environment at all (an inherited file descriptor, a 0600
file); both are deliberately out of scope here.
"""

from __future__ import annotations

import os
import secrets
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "BIND_HOST_ENV",
    "DEFAULT_SESSION_TTL_SECONDS",
    "OPERATOR_SECRET_ENV",
    "PANEL_REGISTER_ROUTE",
    "PANEL_TIER_ROUTES",
    "PANEL_TOKEN_ENV",
    "Tier",
    "WebCredentials",
    "classify",
    "close_env_carriers",
    "get_web_credentials",
    "mint_and_announce",
    "mint_secret",
    "reset_web_credentials",
]

#: The operator secret's environment carrier. In the container shape the deploy
#: ``.env`` supplies it per user and nginx forwards the same value as a header;
#: in the single-user shape nothing sets it and this module mints one.
OPERATOR_SECRET_ENV = "OSPREY_TERMINAL_SECRET"

#: The panel token's environment carrier.
PANEL_TOKEN_ENV = "OSPREY_PANEL_TOKEN"

#: The multi-user shape's tell, read (never popped — ``osprey web`` reads it
#: too) exactly as :func:`osprey.cli.web_cmd.resolve_bind_host` reads it: any
#: truthy value means "a deployment declared this, nginx owns the perimeter".
BIND_HOST_ENV = "OSPREY_TERMINAL_BIND_HOST"

#: How long a browser session stays valid. Twelve hours outlives a working day
#: at the console without leaving a forgotten tab authorised indefinitely; the
#: cookie carrying it is a session cookie, so a closed browser drops it sooner.
DEFAULT_SESSION_TTL_SECONDS = 12 * 60 * 60

#: Compared against when a session lookup misses, purely so a miss costs the
#: same byte-comparison work as a hit — it is a cost-equalizer, never a
#: credential, and :meth:`WebCredentials.verify_session` gates its answer on
#: the map lookup so that a candidate equal to this value still refuses. Sized
#: like a minted id (``token_urlsafe(32)`` yields 43 characters) rather than
#: like the candidate, which is attacker-controlled and would otherwise make
#: the decoy's length a channel of its own.
_SESSION_DECOY = "\x00" * 43


def mint_secret() -> str:
    """Mint a fresh credential.

    The same recipe as :func:`osprey.deployment.service_tokens._default_token`,
    deliberately: 256 bits drawn from :mod:`secrets`, URL-safe so the value
    survives being pasted into a query string by ``mint_and_announce`` and into
    a ``.env`` by the deploy path without escaping.
    """
    return secrets.token_urlsafe(32)


@dataclass(eq=False)
class WebCredentials:
    """The credentials one OSPREY process serves its web surfaces with.

    Ordinarily obtained through :func:`get_web_credentials` rather than
    constructed: the module-level holder is what makes the credentials the
    *process's*, shared by every app in it. Constructing one directly is for
    tests that want a known secret without touching the environment.

    The verification methods are what request handling calls, so none of them
    touch the disk or the network. The two credential comparisons are
    constant-time; the session lookup is not, for the reason
    :meth:`verify_session` gives.
    """

    operator_secret: str
    panel_token: str

    #: Live browser sessions, ``{session_id: monotonic deadline}``. Monotonic
    #: rather than wall-clock so a machine that adjusts its clock — an NTP step,
    #: a laptop resuming — cannot retroactively extend or void a session.
    sessions: dict[str, float] = field(default_factory=dict)

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __repr__(self) -> str:
        """Describe the holder without printing either secret.

        The default dataclass ``__repr__`` would put both credentials into
        every log line, traceback frame and debugger view that touches this
        object — which is the one place a credential popped out of the
        environment must not end up.
        """
        return f"WebCredentials(sessions={len(self.sessions)})"

    def verify_operator(self, candidate: str | None) -> bool:
        """Whether ``candidate`` is the operator secret, compared in constant time."""
        return _same_secret(candidate, self.operator_secret)

    def verify_panel(self, candidate: str | None) -> bool:
        """Whether ``candidate`` is the panel token, compared in constant time.

        A true answer authorises only the panel-arrangement tier, never an
        operator action; the route table decides which tier a request needs and
        this method never widens it.
        """
        return _same_secret(candidate, self.panel_token)

    def create_session(self, ttl_seconds: float = DEFAULT_SESSION_TTL_SECONDS) -> str:
        """Mint a browser session id, record its expiry, and return it.

        Expired entries are dropped here rather than on a timer: sessions are
        only ever created by a browser exchanging a token, so this is the one
        call whose frequency tracks the map's growth, and it means the process
        carries no reaper thread for a dictionary that holds a handful of keys.
        """
        session_id = mint_secret()
        with self._lock:
            self._purge_expired(time.monotonic())
            self.sessions[session_id] = time.monotonic() + ttl_seconds
        return session_id

    def revoke_session(self, session_id: str) -> bool:
        """Drop one browser session, returning whether it had been live.

        Logging out has to invalidate the cookie server-side as well as clear
        it in the browser: a cookie value that has already left the process
        cannot be un-sent, so the only thing that can refuse it afterwards is
        the map here no longer holding it.
        """
        with self._lock:
            return self.sessions.pop(session_id, None) is not None

    def verify_session(self, session_id: str | None) -> bool:
        """Whether ``session_id`` names a live browser session.

        One dictionary lookup and one :func:`secrets.compare_digest` per call,
        with no I/O — this runs on every cookie-bearing request, so anything
        that read a file or opened a socket would land on the hot path of every
        page load and every websocket frame.

        **The map lookup is what decides the answer.** The comparison below
        cannot: on a hit it weighs the candidate against itself, so it is true
        by construction. Its job is the other half — it runs on the miss path
        too, against a fixed-length decoy, so that a rejected id costs the same
        visible work as an accepted one and the timing of a refusal says
        nothing about how much of a guess was right. The decoy is never
        allowed to *be* the answer: ``hit`` gates the return, because a
        candidate that happens to equal the decoy would otherwise compare equal
        to it and authenticate against an empty map.

        Bytes rather than ``str``, because a cookie is unauthenticated input
        and ``compare_digest`` raises ``TypeError`` on a non-ASCII ``str`` —
        one accented character in a forged cookie would be an unhandled 500
        instead of a refusal.

        **This is not constant time, and that is accepted rather than
        overlooked.** ``session_id in self.sessions`` hashes the candidate and,
        on a bucket hit, falls back to ``str.__eq__``, which stops at the first
        differing byte; the ``compare_digest`` above equalises the cost of the
        *comparison* but cannot equalise the cost of the *lookup*. What that
        timing could reveal is that a candidate shares a prefix with a live
        session id — and a session id is 256 bits from :mod:`secrets`, minted
        server-side and handed only to a browser that already authenticated, so
        a prefix oracle buys an attacker nothing reachable: there is no
        structure to extend, no offline check, and the id expires with the
        session. The credentials an attacker would actually target — the
        operator secret and the panel token — are compared by
        :func:`_same_secret`, which is constant time.
        """
        if not session_id:
            return False
        now = time.monotonic()
        with self._lock:
            self._purge_expired(now)
            hit = session_id in self.sessions
            canonical = session_id if hit else _SESSION_DECOY
            matched = secrets.compare_digest(session_id.encode("utf-8"), canonical.encode("utf-8"))
        return hit and matched

    def _purge_expired(self, now: float) -> None:
        """Drop every session whose deadline has passed. Caller holds ``_lock``."""
        expired = [sid for sid, deadline in self.sessions.items() if deadline <= now]
        for sid in expired:
            del self.sessions[sid]


def _same_secret(candidate: str | None, expected: str) -> bool:
    """Compare a caller-supplied credential against the held one in constant time.

    UTF-8 bytes rather than ``str`` for the same reason
    :meth:`WebCredentials.verify_session` uses them: the candidate arrives in a
    header an attacker writes, and ``secrets.compare_digest`` raises on a
    non-ASCII ``str``, which would turn a refusal into a 500.
    """
    if not candidate:
        return False
    return secrets.compare_digest(candidate.encode("utf-8"), expected.encode("utf-8"))


_POPULATION_LOCK = threading.Lock()
_CREDENTIALS: WebCredentials | None = None


def _populate() -> WebCredentials:
    """Build this process's credentials from the environment, minting what is absent.

    Raises:
        RuntimeError: in the container shape, when no operator secret was
            supplied. Minting one there would be worse than failing: nginx
            forwards the value the deploy ``.env`` pinned, so a locally minted
            secret would never match the header the reverse proxy sends and
            every request would be refused with no indication why.
    """
    # Both carriers are popped BEFORE the check below, because the check can
    # raise and the process that catches it goes on serving and spawning
    # children. A panel token left in ``os.environ`` at that point is exactly
    # the inheritance this module exists to prevent, so the fatal path must not
    # be the one path that leaks it.
    operator_secret = os.environ.pop(OPERATOR_SECRET_ENV, "").strip()
    supplied_panel_token = os.environ.pop(PANEL_TOKEN_ENV, "").strip()

    if not operator_secret:
        if os.environ.get(BIND_HOST_ENV):
            raise RuntimeError(
                f"{OPERATOR_SECRET_ENV} is empty or unset, but {BIND_HOST_ENV} is declared, "
                "so this process is running behind the multi-user reverse proxy. The secret "
                f"must be supplied by the deployment: set {OPERATOR_SECRET_ENV} for this "
                "service in the compose environment (the deploy .env holds the per-user "
                "value). Refusing to mint one here, because nginx forwards the deployment's "
                "value and a locally minted secret would refuse every request."
            )
        operator_secret = mint_secret()

    return WebCredentials(
        operator_secret=operator_secret,
        panel_token=supplied_panel_token or mint_secret(),
    )


def get_web_credentials(app: Any = None) -> WebCredentials:
    """Return this process's credentials, populating them on first use.

    Pass the Starlette/FastAPI ``app`` a request is being served by and the
    result is cached on ``app.state.web_credentials``, which is where the auth
    middleware reads it from. Every app in the process resolves to the *same*
    object: a companion app mounted alongside the terminal must accept the
    cookie the terminal handed the browser, and it can only do that by sharing
    the credentials rather than minting its own.

    Called with no ``app`` it returns the same process-wide holder, for callers
    that need the credentials outside a request — the CLI printing the one-time
    URL, or a test.

    Population is guarded by a lock and runs at most once, so two threads
    racing to serve the first two requests cannot end up with different
    secrets. A failure to populate is not cached: the container-shape
    ``RuntimeError`` re-raises on every call, because the condition that caused
    it — a declared bind host with no supplied secret — is still true.
    """
    if app is not None:
        state = getattr(app, "state", None)
        existing = getattr(state, "web_credentials", None)
        if isinstance(existing, WebCredentials):
            return existing

    global _CREDENTIALS
    with _POPULATION_LOCK:
        if _CREDENTIALS is None:
            _CREDENTIALS = _populate()
        credentials = _CREDENTIALS

    if app is not None:
        state = getattr(app, "state", None)
        if state is not None:
            state.web_credentials = credentials
    return credentials


def close_env_carriers() -> None:
    """Remove both credential carriers from ``os.environ`` if either is still there.

    :func:`_populate` pops them, but it runs at most once per process, while
    :func:`mint_and_announce` re-publishes the operator secret on *every*
    launch — deliberately, so a ``--reload`` worker or ``--detach`` child
    inherits it. On the direct-serve path the launcher becomes the server
    without a new process, so that re-published value would otherwise sit in
    ``os.environ`` for the server's whole life and be handed to every agent the
    SDK spawns, which builds its child environment as
    ``{**os.environ, **options.env}``.

    Called at every interface app's construction, from
    :func:`osprey.interfaces._app_setup.configure_interface_app`, immediately
    after the credentials have been resolved — so the carriers are closed
    whichever launch shape this process took, including the per-user
    container's ``osprey web``. Idempotent, and cheap enough to be
    unconditional: the credentials live in this module's memory and nothing
    here reads the environment back.

    **Call :func:`get_web_credentials` first.** This does not settle anything;
    running it before population would discard a deployment-supplied secret and
    leave the holder minting a local one that nginx would never match.
    """
    os.environ.pop(OPERATOR_SECRET_ENV, None)
    os.environ.pop(PANEL_TOKEN_ENV, None)


def reset_web_credentials() -> None:
    """Forget the process credentials so the next call re-populates them.

    For tests. A test that exercises the environment-driven population path has
    to start from an unpopulated holder, and without this the first test in a
    worker would decide the secrets for every test after it. Note that it does
    not restore the environment variables population popped, and does not clear
    ``app.state.web_credentials`` on an app built before the reset — an app
    outliving the reset keeps the credentials it cached.
    """
    global _CREDENTIALS
    with _POPULATION_LOCK:
        _CREDENTIALS = None


def mint_and_announce(host: str, port: int, *, path: str = "/") -> str:
    """Settle this process's operator secret and return the URL that carries it.

    Every OSPREY serving entry point calls this once, just before it binds, and
    prints what comes back. The returned URL is the operator's only way in:
    nothing else prints the secret, and a browser that has not been handed this
    URL cannot reach the surface at all.

    Args:
        host: The address being bound, used verbatim — a caller that binds
            ``0.0.0.0`` announces ``0.0.0.0``, because only the caller knows
            which name its operator should reach it by. An IPv6 literal is
            wrapped in brackets, since ``http://::1:8080/`` has no unambiguous
            parse.
        port: The **settled** port, not the requested one. Entry points that
            fall back to a free port must call this after that fallback, or
            they announce a token for a socket nothing is listening on.
        path: The page the token is exchanged at, defaulting to the app root.
            A bare path, with no query string of its own — the ``?token=`` is
            appended, so a path already carrying a query would produce two.

    Returns:
        ``http://<host>:<port><path>?token=<operator secret>``.

    **The token in the URL is the operator secret itself**, not a second
    credential the holder tracks alongside it. That is forced by how the
    serving process is started: under ``--reload`` and ``--detach`` the process
    that prints this URL is not the process that answers it, and the only thing
    that crosses that boundary is the environment. A distinct URL token would
    have to be carried over in an environment variable of its own, at which
    point it is one more secret in the child's environment buying nothing — it
    is exchangeable for a full operator session either way. Nothing is written
    to disk, and nothing is registered: the child recognises the token because
    :func:`get_web_credentials` there pops the same secret out of the
    environment, and :meth:`WebCredentials.verify_operator` compares it in
    constant time.

    The token is exchanged **once** by the browser — the exchange answers with
    a session cookie and a redirect to the clean URL, so the secret leaves the
    address bar — but it is not burned by that exchange. It stays valid for the
    life of the process, because the operator who reloads the tab, opens a
    second window, or comes back after the cookie expires has no other way to
    re-enter, and re-minting would invalidate the session already handed out.

    **Setting the environment variable here contradicts the ``pop`` in
    :func:`_populate` on purpose, and the window it opens is closed again
    before the process serves anything.** The pop closes the agent-visible
    window in the *serving* process: by the time that process is answering
    requests and spawning the agent's sandbox, the secret exists only in this
    module's memory. This function runs in the *CLI parent*, before any serving
    app is constructed and before any agent exists, and re-publishes the value
    for exactly one purpose — the process it is about to spawn or become. Under
    ``--reload`` and ``--detach`` that is a fresh process, which pops the
    carrier at its own first :func:`get_web_credentials`. In *direct-serve*
    mode the launcher **becomes** the server in the same, already-populated
    process, so no fresh :func:`_populate` runs and nothing there would pop the
    value this function just re-published; :func:`close_env_carriers`, called
    from every interface app's construction, is what removes it on that path.
    Relying instead on :mod:`osprey.agent_runner.clean_env` to omit the name
    would not work: the agent SDK overlays its ``env`` onto ``os.environ``
    rather than replacing it, so an omitted key is simply inherited. The rule
    that keeps all of this consistent is: **call this from a launcher,
    immediately before serving or spawning, never from inside a process that is
    already serving.** A serving process that calls it puts its own secret back
    into the environment its agent inherits — until the next app construction,
    if any — which is the escalation this module exists to prevent.

    Idempotent per process: the secret is settled by the process-wide holder,
    so a second call — a second entry point in the same process, a retry after
    a port fallback — announces the same token and the same URL rather than
    invalidating the first.

    Raises:
        RuntimeError: in the container shape with no supplied secret, from
            :func:`get_web_credentials`. The guard is untouched here: a caller
            in that shape must be given the deployment's secret, and this
            function then announces *that* value, which is the one nginx
            forwards. Host-side entry points normally leave
            :data:`BIND_HOST_ENV` unset and never see it.
    """
    credentials = get_web_credentials()

    # Deliberate re-publication for the child/worker about to be started; see
    # the contradiction paragraph above for why this is not the leak the pop
    # exists to prevent.
    os.environ[OPERATOR_SECRET_ENV] = credentials.operator_secret

    if not path.startswith("/"):
        path = f"/{path}"
    authority = f"[{host}]" if ":" in host and not host.startswith("[") else host
    # A minted secret is already URL-safe, so this is a no-op for one. A
    # deployment-supplied secret is not: it comes out of a ``.env`` and may
    # hold anything, and an unescaped ``&`` or ``#`` there would truncate the
    # token the browser sends back into something that authenticates nobody.
    token = urllib.parse.quote(credentials.operator_secret, safe="")
    return f"http://{authority}:{port}{path}?token={token}"


class Tier(Enum):
    """Which credential a request needs before it is allowed to reach a route.

    Two members and no ordering: the middleware branches on identity, and a
    numeric or comparable tier would invite a ``>=`` test that quietly treats a
    new member as sufficient. :data:`Tier.OPERATOR` is the default answer for
    anything :func:`classify` does not recognise, so adding a route to the tree
    without touching this module leaves it protected rather than exposed.
    """

    #: The narrow, low-privilege tier: panel arrangement and activity reporting.
    #: Reachable with the panel token *or* with any operator credential.
    PANEL = "panel"

    #: Everything else. Reachable only with the operator secret or a live
    #: browser session — never with the panel token.
    OPERATOR = "operator"


#: The exact ``(method, path)`` pairs an in-process companion may drive with the
#: panel token alone.
#:
#: Every entry is a real route in this tree: the first six live in
#: :mod:`osprey.interfaces.web_terminal.routes.panels` and
#: :mod:`osprey.interfaces.web_terminal.routes.agent_activity`, and
#: ``POST /api/focus`` is the artifacts app's focus setter
#: (:mod:`osprey.interfaces.artifacts.app`). Because one process serves several
#: apps and the table is shared by all of them, ``POST /api/focus`` is
#: panel-tier everywhere — on an app that has no such route the request clears
#: the gate and is then answered with a 404 by routing, which grants nothing.
#:
#: What is deliberately *absent* matters as much as what is present:
#: ``GET /api/panel-focus`` (a real route) and ``POST /api/panel-layout`` (which
#: writes the persisted layout) are operator-only, as is every config, scaffold,
#: memory, chat, feedback, file, session and proxy route, and both websockets.
PANEL_TIER_ROUTES: frozenset[tuple[str, str]] = frozenset(
    {
        ("GET", "/api/panels"),
        ("POST", "/api/panel-focus"),
        ("POST", "/api/panel-visibility"),
        ("POST", "/api/panel-close"),
        ("POST", "/api/panel-arrange"),
        ("POST", "/api/agent-activity"),
        ("POST", "/api/focus"),
    }
)

#: The one route whose tier depends on its body, kept out of
#: :data:`PANEL_TIER_ROUTES` so the flat table stays a pure lookup. Registering
#: a panel *with* a ``url`` decides where the terminal's own proxy forwards
#: operator traffic, which is an operator act; registering without one cannot
#: repoint anything.
PANEL_REGISTER_ROUTE: tuple[str, str] = ("POST", "/api/panels/register")


def classify(method: str | None, path: str | None, body_has_url: bool) -> Tier:
    """Return the credential tier a request must satisfy.

    Args:
        method: The HTTP method from the ASGI scope. Websocket scopes carry no
            method, so the middleware supplies one for them; either way the
            value is upper-cased before the lookup. A missing or empty method
            yields :data:`Tier.OPERATOR`.
        path: The request path from the ASGI scope, **bare**. In the
            multi-user shape nginx strips the ``/u/<user>`` prefix before the
            app sees the request, so a prefixed path is not something this
            function accepts — it is something it refuses, by not matching.
            The path must also carry no query string; ``scope["path"]`` never
            does.
        body_has_url: Whether the JSON body of a
            ``POST /api/panels/register`` carries a ``url`` key. Ignored for
            every other route. **Pass ``True`` whenever the answer is not
            known** — an unreadable or unparseable body must be treated as the
            dangerous case, not the harmless one. Required rather than
            defaulted: a ``False`` default would let a caller that forgot the
            argument grant URL-backed registration to the weak credential, and
            a ``True`` default would be a parameter whose name contradicts its
            value. Making the caller decide is the only spelling with no wrong
            way to hold it.

    Returns:
        :data:`Tier.PANEL` for the handful of routes in
        :data:`PANEL_TIER_ROUTES`, and for a ``url``-free panel registration.
        :data:`Tier.OPERATOR` for everything else.

    The match is **exact** on both fields, and that is the security property
    rather than an implementation shortcut. A prefix match on ``/api/panel``
    would hand the weak credential ``/api/panel-layout``; a prefix match on
    ``/api/panels`` would hand it every registration, ``url`` and all. So a
    trailing slash, a different case, an unexpected extra segment or a
    surviving ``/u/<user>`` prefix all fall through to
    :data:`Tier.OPERATOR`. That direction of failure costs a companion at most
    a refusal on a spelling no client uses; the other direction would be a
    privilege escalation.
    """
    if not method or not path:
        return Tier.OPERATOR

    route = (method.upper(), path)
    if route in PANEL_TIER_ROUTES:
        return Tier.PANEL
    if route == PANEL_REGISTER_ROUTE and not body_has_url:
        return Tier.PANEL
    return Tier.OPERATOR
