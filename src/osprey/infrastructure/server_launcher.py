"""Generic auto-launcher for OSPREY companion servers.

Provides a reusable ``ServerLauncher`` that starts a uvicorn server
in a daemon thread on first demand.  Server definitions live in
``registry.web`` — this module uses ``importlib`` to resolve factories
at call time so the infrastructure layer never imports from interfaces/.
"""

from __future__ import annotations

import errno
import importlib
import logging
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from functools import partial
from pathlib import Path

from osprey.registry.web import (
    FRAMEWORK_WEB_SERVERS,
    WebServerDefinition,
    resolve_web_server_address,
    web_server_config_section,
)
from osprey.utils.workspace import load_osprey_config

logger = logging.getLogger("osprey.infrastructure.server_launcher")

# When a port is genuinely unbindable on first check, it may be a predecessor
# that is still shutting down after a restart. Wait a bounded grace period for
# it to release the port so we can bind (own) it ourselves rather than trusting
# a possibly-dying external responder. See issue #327. A socket left in
# TIME_WAIT is *bindable* (the probe mirrors uvicorn's ``SO_REUSEADDR``), so it
# never reaches this loop and never delays startup.
_PORT_RELEASE_GRACE_ATTEMPTS = 5
_PORT_RELEASE_GRACE_INTERVAL = 0.5

# After any refusal verdict in _adopt_or_refuse — the listener could not be
# attributed to this deployment, whatever the probes said — we cannot bind the
# port now, but it may free up later (a lazy caller like artifact_store
# re-invokes ensure_running on every save). Re-probe at most this often so a
# panel can self-heal without paying the probe cost on every call.
_HELD_PORT_RETRY_COOLDOWN = 30.0

#: The route the adoption probes ask, and it must NOT be an exempt one.
#: ``/health`` and the rest of
#: :data:`osprey.interfaces.common_middleware.EXEMPT_PATHS` are reachable with
#: no credential at all — any OSPREY checkout on the machine answers ``/health``
#: with a 200 — so they carry no information about *whose* server is listening.
#: ``/`` is not exempt, and the auth gate answers it before routing, so the
#: probe reads the gate's verdict rather than a panel's route table: a panel
#: that happens not to serve an index still answers 401 unauthenticated and a
#: non-401 (404) credentialed, which is exactly the discrimination this needs.
_ADOPTION_PROBE_PATH = "/"

#: Seconds either adoption probe waits. Both run only on the already-degraded
#: path where the port could not be bound, never on a clean start.
_ADOPTION_PROBE_TIMEOUT = 1.0

#: The header the credentialed probe carries the operator secret in. Mirrors
#: :data:`osprey.interfaces.common_middleware.OPERATOR_SECRET_HEADER` rather
#: than importing it, because infrastructure/ does not import interfaces/; the
#: two spellings are pinned equal by a test.
_OPERATOR_SECRET_HEADER = "x-osprey-terminal-secret"

#: The environment carrier for this process's operator secret. Mirrors
#: :data:`osprey.interfaces.web_auth.OPERATOR_SECRET_ENV`, pinned by the same
#: test. Only a *carrier*: see :meth:`ServerLauncher._operator_secret`.
_OPERATOR_SECRET_ENV = "OSPREY_TERMINAL_SECRET"


def _loopback_for(host: str) -> str:
    """Map a wildcard bind host to a loopback address reachable as a *client*.

    A server bound to a wildcard (``0.0.0.0`` / ``::`` / ``""``) also accepts
    connections on the corresponding loopback, but the wildcard itself is not a
    valid client destination on macOS/BSD. Non-wildcard hosts pass through.
    """
    if host in ("0.0.0.0", ""):
        return "127.0.0.1"
    if host == "::":
        return "::1"
    return host


def _probe_url(host: str, port: int, path: str) -> str:
    """The URL a probe of *host:port* opens, with the client-side rules applied.

    Two of them, and both belong to every probe this module makes rather than to
    any one of them: a wildcard bind host is addressed via its loopback (see
    :func:`_loopback_for`, because the wildcard is not a valid destination), and
    an IPv6 literal is bracketed so the ``:port`` suffix is not read as part of
    the address.
    """
    probe_host = _loopback_for(host)
    netloc = f"[{probe_host}]" if ":" in probe_host else probe_host
    return f"http://{netloc}:{port}{path}"


def _describe_status(status: int | None) -> str:
    """Render one adoption-probe outcome for the operator-facing log line."""
    return "no answer" if status is None else f"HTTP {status}"


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """A redirect handler that refuses to follow, so a 3xx is its own status.

    The adoption verdict is about the listener *on this port*, and the default
    opener would destroy exactly that: it follows a 3xx transparently and
    reports the final hop's status as if the port had answered it. Worse, it
    re-sends every header it was given — including
    :data:`_OPERATOR_SECRET_HEADER` — to whatever host the ``Location`` names,
    which would hand this process's operator secret to a destination chosen by
    the very listener under suspicion.

    Returning ``None`` from ``redirect_request`` makes urllib fall through to
    ``HTTPDefaultErrorHandler``, which raises ``HTTPError`` carrying the 3xx
    code — so :meth:`ServerLauncher._probe_status` reports the redirect itself.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        """Never follow: the listener's own first response is the verdict."""
        return None


def _probe_opener(*extra: urllib.request.BaseHandler) -> urllib.request.OpenerDirector:
    """Build the opener the adoption probes use, redirect-following removed.

    ``build_opener`` drops a default handler when a supplied one subclasses it,
    so passing :class:`_NoRedirectHandler` replaces the redirect-following
    default rather than stacking on top of it. *extra* handlers let a test
    substitute the transport without rebuilding the rest of the chain.
    """
    return urllib.request.build_opener(_NoRedirectHandler(), *extra)


class ServerLauncher:
    """Double-checked-locking launcher for a uvicorn companion server.

    Args:
        name: Human-readable server name (for logging).
        config_reader: Callable returning ``(host, port)`` from config.
        auto_launch_checker: Callable returning ``True`` if auto-launch is enabled.
        app_factory: Callable returning a ASGI app instance. Receives
            ``workspace_root`` kwarg if ``pass_workspace`` is True.
        pass_workspace: If True, resolve and pass ``workspace_root`` to the app factory.
        release_grace_attempts: Times to re-probe a held port, waiting for a
            shutting-down predecessor to release it before deferring/warning.
        release_grace_interval: Seconds between those re-probes.
        held_port_retry_cooldown: After a port is found held by a non-responder,
            seconds to wait before ``ensure_running`` re-probes again.
    """

    def __init__(
        self,
        name: str,
        config_reader: Callable[[], tuple[str, int]],
        auto_launch_checker: Callable[[], bool],
        app_factory: Callable[..., object],
        pass_workspace: bool = False,
        release_grace_attempts: int = _PORT_RELEASE_GRACE_ATTEMPTS,
        release_grace_interval: float = _PORT_RELEASE_GRACE_INTERVAL,
        held_port_retry_cooldown: float = _HELD_PORT_RETRY_COOLDOWN,
    ) -> None:
        self._name = name
        self._config_reader = config_reader
        self._auto_launch_checker = auto_launch_checker
        self._app_factory = app_factory
        self._pass_workspace = pass_workspace
        self._release_grace_attempts = release_grace_attempts
        self._release_grace_interval = release_grace_interval
        self._held_port_retry_cooldown = held_port_retry_cooldown
        self._launched = False
        # Monotonic deadline before which ensure_running() short-circuits after
        # a refusal verdict (see ensure_running).
        self._retry_not_before = 0.0
        # Set once _adopt_or_refuse has refused. The grace window exists for a
        # shutting-down predecessor, and a predecessor gets exactly one; after a
        # refusal the holder is established, so later calls skip straight to the
        # verdict instead of sleeping through the window on every save. Reset by
        # _launch_in_thread: a committed launch ends the episode this describes.
        self._refused_once = False
        # The (unauthenticated status, credentialed outcome) pair of the last
        # refusal, so an unchanged verdict logs at info rather than warning.
        # Reset alongside _refused_once, and for the same reason.
        self._last_refusal: tuple[int | None, str] | None = None
        self._lock = threading.Lock()

    def _port_is_bindable(self, host: str, port: int) -> bool:
        """Return True if this process could bind *host:port* right now.

        This is the ownership verdict, and it is the only question that
        matters before a launch: not "can something be reached there?" but
        "will our own ``bind()`` succeed?". It is asked by performing exactly
        that bind, so probe and server cannot disagree:

        * the address family follows the SPELLING of the host string, not a
          name lookup: a host containing ``:`` is probed as ``AF_INET6``, every
          other as ``AF_INET``. A hostname such as ``localhost`` therefore gets
          an IPv4-only probe even where it resolves to both families, so a
          listener holding only its IPv6 address is invisible here; that
          conflict surfaces at launch instead, as the "thread exited before
          health check passed" warning from :meth:`_launch_in_thread`;
        * the EXACT ``(host, port)`` the server will bind is probed — a
          wildcard host binds the wildcard, never a substituted loopback,
          because a wildcard bind conflicts with listeners a loopback probe
          would miss;
        * ``SO_REUSEADDR`` is set to mirror uvicorn's listener, so a socket
          left in ``TIME_WAIT`` reads as bindable for the probe exactly as it
          is for the server;
        * only ``EADDRINUSE`` and ``EACCES`` mean the port is taken. Any other
          ``OSError`` says the probe itself could not answer — that is logged
          as a diagnostic and does NOT count as taken, so a probe defect can
          never silently suppress a launch.

        A listener that answers a connect but does not block this bind (a
        Docker Desktop host-loopback pass-through, say) is a foreign host-side
        listener, not an owner of this port: see :meth:`_port_answers_connect`.
        """
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        try:
            with socket.socket(family, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((host, port))
            return True
        except OSError as exc:
            if exc.errno in (errno.EADDRINUSE, errno.EACCES):
                return False
            logger.warning(
                "%s bind probe of %s:%s was inconclusive (%s); treating the port as free",
                self._name,
                host,
                port,
                exc,
            )
            return True

    def _port_answers_connect(self, host: str, port: int) -> bool:
        """Return True if a TCP connect to *host:port* is accepted.

        Diagnostics only — never a verdict. A connect can succeed against a
        listener that does not block our bind at all (the Docker Desktop
        host-loopback pass-through), and it can fail against a listener that
        does. Ownership is decided by :meth:`_port_is_bindable`; this probe
        exists to explain what an operator can otherwise see with ``curl``.
        A wildcard bind host is probed via loopback, which a listener on the
        wildcard also accepts.
        """
        try:
            with socket.create_connection((_loopback_for(host), port), timeout=1):
                return True
        except OSError:
            return False

    def _is_running(self, host: str, port: int) -> bool:
        """Check the /health endpoint (quick, no dependencies)."""
        try:
            req = urllib.request.Request(_probe_url(host, port, "/health"), method="GET")
            with urllib.request.urlopen(req, timeout=1) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _operator_secret(self) -> str | None:
        """Return this process's operator secret, or None if it holds none.

        Two sources, in this order, because the environment variable is a
        *carrier* rather than the store: ``osprey web`` publishes the minted
        secret into ``OSPREY_TERMINAL_SECRET`` so whatever it spawns or becomes
        inherits one value, and the receiving process then POPS it into a
        process-wide holder at app construction (``web_auth._populate``, and
        ``close_env_carriers`` on the direct-serve path) precisely so no agent
        subprocess inherits it. Reading only the environment would therefore
        find nothing in the hub that has been serving for any length of time —
        the one process for which adoption is worth deciding.

        So: the carrier first (a process that has not populated yet, which is
        also the only case where the two could differ in timing), then the
        in-process holder — read with ``peek_web_credentials``, which returns
        what is already there and NEVER populates.

        Populating here would be a bug, not a slower path. The most frequent
        caller of ``ensure_*_server`` is the MCP server the agent spawns, on
        every artifact save, and its environment deliberately carries a
        re-introduced panel token and no operator secret at all. In that process
        ``get_web_credentials`` would MINT a fresh operator secret and panel
        token that nothing else in the deployment recognises — a fabricated
        identity offered to the credentialed probe, and a misdiagnosed warning
        when the probe then fails — and would pop the panel token out of
        ``os.environ`` on the way, racing the panel-auth latch and stripping the
        carrier from every child spawned afterwards. "This process holds no
        operator identity" is the honest answer here, and the caller refuses on
        it.

        The holder is reached by a call-time import, the same deferral
        :func:`_make_app_factory` uses, so this module still has no import-time
        dependency on interfaces/.
        """
        carried = (os.environ.get(_OPERATOR_SECRET_ENV) or "").strip()
        if carried:
            return carried
        try:
            from osprey.interfaces.web_auth import peek_web_credentials

            credentials = peek_web_credentials()
        except Exception as exc:
            logger.debug("%s could not read this process's operator secret (%s)", self._name, exc)
            return None
        if credentials is None:
            return None
        return (credentials.operator_secret or "").strip() or None

    def _probe_status(self, host: str, port: int, secret: str | None = None) -> int | None:
        """GET :data:`_ADOPTION_PROBE_PATH` on *host:port*; return its HTTP status.

        Returns None when nothing answered with HTTP at all (connection
        refused, timeout, a non-HTTP process holding the port) — distinct from
        any status, because "did not answer" and "answered 401" support
        opposite conclusions about what is listening.

        Redirects are NOT followed (see :class:`_NoRedirectHandler`): the
        verdict must be the listener's own first response, and following a 3xx
        would both mask it and re-send the secret header to a destination that
        listener chose.

        Args:
            host: The configured bind host; a wildcard is probed via loopback.
            port: The configured port.
            secret: When given, carried in :data:`_OPERATOR_SECRET_HEADER`.
                The value itself is never logged.
        """
        url = _probe_url(host, port, _ADOPTION_PROBE_PATH)
        headers = {_OPERATOR_SECRET_HEADER: secret} if secret else {}
        req = urllib.request.Request(url, method="GET", headers=headers)
        try:
            with _probe_opener().open(req, timeout=_ADOPTION_PROBE_TIMEOUT) as resp:
                return int(resp.status)
        except urllib.error.HTTPError as exc:
            # A 401 is the informative answer here, and urllib raises it.
            return int(exc.code)
        except Exception:
            return None

    def _adopt_or_refuse(self, host: str, port: int) -> None:
        """Decide what a port we could not bind is, and act on the verdict.

        Called only after the release grace window has passed, so something is
        holding this port for good. The question is whether that something is
        this deployment's own panel — in which case there is nothing to launch
        and we should stop trying — or a stranger, in which case adopting it
        would hand the operator a panel backed by a server this process cannot
        talk to.

        ``/health`` cannot answer that question and takes no part in it: it is
        exempt from the auth gate by design, so every OSPREY checkout on the
        machine answers it 200. Credentials can, and the evidence is two-sided:

        * unauthenticated probe 401 **and** credentialed probe non-401 — the
          listener demands a credential and accepts *ours*, so it is our own
          panel under a shared ``OSPREY_TERMINAL_SECRET``. Adopt it: latch, so
          a per-call caller stops re-probing;
        * anything else — a 200 to the unauthenticated probe (an open server
          that gates nothing), a 401 to both (a server holding a different
          secret), no answer at all, or no operator secret in this process to
          probe with — is not attributable. Refuse: do not latch, warn naming
          the listener and both probe outcomes, and set the cooldown so the
          launcher self-heals when the port frees.

        The credentialed probe runs ONLY when the unauthenticated one answered
        401. That is both a saving and a rule: a listener that just served an
        ungated 200 has demonstrated it does not check credentials, and this
        process must not hand its operator secret to it to find that out twice.

        For the same reason the local secret is *looked up* only in that branch.
        Consulting the credential holder is not free — it is the one question
        whose asking can change the answer (see :meth:`_operator_secret`) — so a
        listener that answered 200, 503 or nothing is refused without this
        process ever asking who it is.

        Two refusals carry their own advice rather than the generic one, because
        the generic advice ("a foreign server, or one holding a different
        secret — stop it") would be a fabrication for them: a 503 is an OSPREY
        gate that could not populate its own credentials, so the fault named is
        that service's configuration; and a listener this process holds no
        identity to question at all is equally consistent with our own panel.

        Repeat refusals are demoted to ``info``: the default topology re-enters
        this path on every artifact save once the cooldown expires, and an
        unchanged verdict is not news. A *changed* outcome pair warns again.
        """
        unauthenticated = self._probe_status(host, port)
        no_identity = False

        if unauthenticated is None:
            credentialed_outcome = "not attempted (nothing answered the first probe)"
            credentialed = None
        elif unauthenticated == 503:
            credentialed_outcome = (
                "not attempted — the listener's auth gate is unavailable, HTTP 503"
            )
            credentialed = None
        elif unauthenticated != 401:
            credentialed_outcome = (
                f"not attempted (the listener answered HTTP {unauthenticated} "
                "without demanding a credential)"
            )
            credentialed = None
        else:
            secret = self._operator_secret()
            if secret is None:
                no_identity = True
                credentialed_outcome = "not attempted (this process holds no operator secret)"
                credentialed = None
            else:
                credentialed = self._probe_status(host, port, secret)
                credentialed_outcome = _describe_status(credentialed)

        if unauthenticated == 401 and credentialed is not None and credentialed != 401:
            logger.info(
                "%s: adopting the server already listening at %s:%s — it refused an "
                "unauthenticated %s (401) and accepted this process's operator secret "
                "(%s), so it is this deployment's own panel. Not launching a second one.",
                self._name,
                host,
                port,
                _ADOPTION_PROBE_PATH,
                credentialed_outcome,
            )
            self._launched = True
            return

        # Warn once per DISTINCT verdict. The first refusal, and any later one
        # that says something new, is operator-facing news; a repeat of the same
        # pair on the next save is not, and warning on every save would train the
        # operator to ignore the channel that carries the real change.
        outcomes = (unauthenticated, credentialed_outcome)
        emit = logger.info if outcomes == self._last_refusal else logger.warning

        if no_identity:
            # The listener gates, and this process cannot say whose gate it is.
            # It must not guess out loud: from here the listener is equally
            # consistent with this deployment's own hub-owned panel (in which
            # case the panel is fine and there is nothing to stop) and with a
            # stranger's server. Advice either way would be a fabrication.
            emit(
                "%s: %s:%s is held by a listener that demands a credential "
                "(unauthenticated probe of %s: HTTP 401), but this process holds no "
                "operator secret: none in the %s carrier, and no populated credential "
                "holder. Without an operator identity it cannot attribute that "
                "listener to this deployment or to a stranger. Not launching a second "
                "server on that port, and not guessing what the listener serves. "
                "Will retry on a later call.",
                self._name,
                host,
                port,
                _ADOPTION_PROBE_PATH,
                _OPERATOR_SECRET_ENV,
            )
        elif unauthenticated == 503:
            # A 503 to the gate's own route is an OSPREY auth gate that could
            # not populate its credentials — it is answering for nobody, so it
            # cannot be attributed either way, and the generic advice would be
            # wrong twice over: this is very likely not a "foreign" server, and
            # telling the operator to stop it hides the configuration fault
            # that is the actual thing to fix.
            emit(
                "%s: %s:%s is held by a listener whose auth gate answered HTTP 503 to an "
                "unauthenticated probe of %s, which is an OSPREY gate that could not populate "
                "its own credentials. A gate answering for nobody cannot be attributed to this "
                "deployment or to a stranger, and this process will not offer its operator "
                "secret to it to find out. Check that service's %s and bind-host "
                "configuration. Not launching a second server on that port. Will retry on a "
                "later call.",
                self._name,
                host,
                port,
                _ADOPTION_PROBE_PATH,
                _OPERATOR_SECRET_ENV,
            )
        else:
            emit(
                "%s: %s:%s is held by a listener this process cannot attribute to itself. "
                "Unauthenticated probe of %s: %s; credentialed probe: %s. Only a 401 to the "
                "first and a non-401 to the second identify it as this deployment's own panel; "
                "anything else is a foreign server, or one holding a different "
                "%s than this process, so credentialed calls between them would be refused. "
                "The panel will be unbacked (502) until the port is free. Stop the other "
                "server and let this process own it, or point both at the same deployment. "
                "Will retry on a later call.",
                self._name,
                host,
                port,
                _ADOPTION_PROBE_PATH,
                _describe_status(unauthenticated),
                credentialed_outcome,
                _OPERATOR_SECRET_ENV,
            )

        self._last_refusal = outcomes
        self._refused_once = True
        self._retry_not_before = time.monotonic() + self._held_port_retry_cooldown

    def _launch_in_thread(self, host: str, port: int) -> None:
        """Start uvicorn in a daemon thread."""

        def _run() -> None:
            try:
                import uvicorn

                if self._pass_workspace:
                    from osprey.utils.workspace import resolve_shared_data_root

                    # Launched servers are daemons serving the shared store (they
                    # may be auto-launched from a session-scoped MCP process on
                    # first artifact save).
                    app = self._app_factory(workspace_root=resolve_shared_data_root())
                else:
                    app = self._app_factory()
                uvicorn.run(app, host=host, port=port, log_level="warning")
            except Exception:
                logger.exception("%s thread crashed", self._name)
                self._launched = False

        # Committing to a launch ends the held-port episode the refusal memory
        # describes: the port was ours to take, so whatever was holding it is
        # gone. Both fields are scoped to ONE continuous episode — carrying them
        # across a launch would deny the next episode its grace window (a fresh
        # predecessor is exactly what that window is for) and demote its first
        # refusal to ``info`` for matching a verdict about a listener that no
        # longer exists.
        self._refused_once = False
        self._last_refusal = None

        t = threading.Thread(target=_run, daemon=True, name=self._name.lower().replace(" ", "-"))
        t.start()
        logger.info("%s launched at http://%s:%s", self._name, host, port)

        # Brief health-check to verify *our* server came up. Liveness is checked
        # first: if the thread has exited (e.g. the bind failed in a TOCTOU race
        # with another process), a /health 200 would be a foreign responder, not
        # ours — trusting it would recreate the #327 false positive.
        for _attempt in range(3):
            time.sleep(0.5)
            if not t.is_alive():
                logger.warning("%s thread exited before health check passed", self._name)
                self._launched = False
                return
            if self._is_running(host, port):
                logger.info("%s health check passed", self._name)
                self._launched = True
                return

        # Thread still alive but /health never answered — mark launched to avoid
        # busy-retry loops; the crash handler in _run() resets _launched if the
        # thread later dies.
        logger.warning(
            "%s health check failed after launch — server may not be reachable at %s:%s",
            self._name,
            host,
            port,
        )
        self._launched = True

    def ensure_running(self) -> None:
        """Ensure the server is running; launch if needed.

        Safe to call multiple times — no-op after first launch.

        Ownership is decided by whether we can actually bind the port, not by
        a bare ``/health`` 200 and not by whether something answers a connect.
        A ``/health`` 200 from a stale or foreign responder is a false
        positive: acting on it makes the manager skip the launch and leave the
        panel unbacked (proxy 502) after a restart. And a connect that is
        answered by a listener which does not block our bind (a Docker Desktop
        host-loopback pass-through) is not an owner at all. Instead:

        * port bindable          -> launch and own it, whatever answers a
          connect there;
        * unbindable then freed  -> a shutting-down predecessor; waited out,
          then launched. A ``TIME_WAIT`` remnant is bindable and never reaches
          this branch, so it no longer delays startup, and the wait is offered
          once per held-port episode: after a refusal the holder is
          established, so later calls go straight from the bind check to the
          verdict — until a launch is committed, which ends the episode and
          restores the window for the next predecessor;
        * unbindable throughout  -> adopt the listener only if a credentialed
          probe pair attributes it to this deployment (latched); otherwise
          refuse it, warn (repeat refusals with an unchanged verdict drop to
          ``info``), and retry on a later call so a lazily-relaunched panel can
          self-heal once the port frees. See :meth:`_adopt_or_refuse`;
          ``/health`` takes no part in that verdict, because it is exempt from
          the auth gate and so answers 200 for any OSPREY process at all.
        """
        if not self._auto_launch_checker():
            return

        if self._launched:
            return

        with self._lock:
            if self._launched:
                return

            # A recent "held by a non-responder" outcome throttles re-probing so
            # a per-call caller (e.g. artifact_store on every save) does not pay
            # the grace cost repeatedly while still recovering when the port frees.
            if time.monotonic() < self._retry_not_before:
                return

            host, port = self._config_reader()

            # We can bind it -> the port is ours to take. Something may still
            # answer a connect there (a host-side listener the container's bind
            # does not contend with); say so, then launch anyway.
            if self._port_is_bindable(host, port):
                if self._port_answers_connect(host, port):
                    logger.info(
                        "%s: %s:%s answers a TCP connect but does not block our bind — a "
                        "foreign host-side listener (e.g. a Docker Desktop loopback "
                        "pass-through), not an owner of this port. Launching.",
                        self._name,
                        host,
                        port,
                    )
                self._launch_in_thread(host, port)
                return

            # Unbindable on first check. Give a shutting-down predecessor a
            # bounded grace period to release the port so we can bind it
            # ourselves. Only a genuinely unbindable port waits here — and only
            # the first time: once we have refused this port, the holder is
            # established rather than departing, and a per-save caller must not
            # pay the whole window (holding self._lock) again on every call. A
            # port that has since been freed still launches, via the bindable
            # check above that every call makes first.
            if not self._refused_once:
                for _attempt in range(self._release_grace_attempts):
                    time.sleep(self._release_grace_interval)
                    if self._port_is_bindable(host, port):
                        self._launch_in_thread(host, port)
                        return

            # Still unbindable after the grace window: something holds this port
            # for good. Whether we stand down for it is a question about WHOSE
            # server it is, and only a credential can answer that — see
            # _adopt_or_refuse. Credentialed probes are paid only here, on this
            # already-degraded path; a clean start never reaches them.
            self._adopt_or_refuse(host, port)


# ---------------------------------------------------------------------------
# Generic helpers — build ServerLauncher callbacks from WebServerDefinition
# ---------------------------------------------------------------------------


def _make_auto_launch_checker(defn: WebServerDefinition) -> Callable[[], bool]:
    """Return a callable that checks whether auto-launch is enabled.

    Navigation into the server's config section goes through
    ``registry.web.web_server_config_section``, the same one
    ``resolve_web_server_address`` uses, so ``auto_launch`` and the port it
    guards can never be read from different depths — and so an ``auto_launch``
    written at the depth this server does not read raises instead of quietly
    reading back as the default and launching a panel the operator switched off.
    """

    def _checker() -> bool:
        config = load_osprey_config()
        top = config.get(defn.config_key, {})
        if defn.require_section and not top:
            return False
        section = web_server_config_section(defn, config)
        return bool(section.get("auto_launch", defn.auto_launch_default))

    return _checker


def _resolve_dotted(config: dict, dotted: str) -> object:
    """Traverse a dotted path like ``"ariel.web.port"`` into *config*."""
    obj: object = config
    for key in dotted.split("."):
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)  # type: ignore[union-attr]
    return obj


def _make_app_factory(defn: WebServerDefinition) -> Callable[..., object]:
    """Return a callable that dynamically imports and invokes the factory."""
    module_path, attr_name = defn.factory_path.rsplit(":", 1)

    def _factory(workspace_root: Path | None = None) -> object:
        try:
            mod = importlib.import_module(module_path)
        except ImportError as err:
            if defn.import_error_message:
                raise ImportError(defn.import_error_message) from err
            raise
        create_app = getattr(mod, attr_name)

        kwargs: dict[str, object] = {}
        if defn.pass_workspace:
            kwargs["workspace_root"] = workspace_root
        if defn.factory_config_kwargs:
            config = load_osprey_config()
            for kwarg_name, dotted_path in defn.factory_config_kwargs.items():
                kwargs[kwarg_name] = _resolve_dotted(config, dotted_path)
        return create_app(**kwargs)

    return _factory


# ---------------------------------------------------------------------------
# Build launchers from the catalog
# ---------------------------------------------------------------------------

_auto_launch_checkers: dict[str, Callable[[], bool]] = {
    key: _make_auto_launch_checker(defn) for key, defn in FRAMEWORK_WEB_SERVERS.items()
}

_launchers: dict[str, ServerLauncher] = {
    key: ServerLauncher(
        name=defn.name,
        # One derivation of (host, port) for every producer and consumer of a
        # companion server's address — see registry.web.resolve_web_server_address.
        # The launcher must not resolve it differently from the callers that
        # publish the URL, or a panel points at a port nothing listens on.
        config_reader=partial(resolve_web_server_address, key),
        auto_launch_checker=_auto_launch_checkers[key],
        app_factory=_make_app_factory(defn),
        pass_workspace=defn.pass_workspace,
    )
    for key, defn in FRAMEWORK_WEB_SERVERS.items()
}


def ensure_web_server(key: str) -> None:
    """Ensure the web server identified by *key* is running."""
    _launchers[key].ensure_running()


def is_auto_launch_enabled(key: str) -> bool:
    """Return True if the web server identified by *key* is configured to auto-launch.

    ``ensure_web_server`` already applies this check, but it applies it silently:
    a caller cannot tell a skipped launch from a completed one. Callers that
    advertise a server to the outside world — the web terminal publishes each
    panel's URL, and panel availability is computed from that URL alone — must
    ask first, so a suppressed server is presented as unavailable rather than as
    a live panel that 502s.
    """
    return _auto_launch_checkers[key]()


# Backward-compatible named aliases (used by web_terminal/app.py, artifact_store.py)
def ensure_artifact_server() -> None:
    """Ensure the artifact server is running; launch if needed."""
    ensure_web_server("artifact")


def ensure_ariel_server() -> None:
    """Ensure the ARIEL server is running; launch if needed."""
    ensure_web_server("ariel")


def ensure_channel_finder_server() -> None:
    """Ensure the Channel Finder web server is running; launch if needed."""
    ensure_web_server("channel_finder")


def ensure_lattice_dashboard_server() -> None:
    """Ensure the lattice dashboard server is running; launch if needed."""
    ensure_web_server("lattice_dashboard")


def ensure_okf_server() -> None:
    """Ensure the OKF knowledge panel server is running; launch if needed."""
    ensure_web_server("okf")


def ensure_system_health_server() -> None:
    """Ensure the system-health dashboard server is running; launch if needed."""
    ensure_web_server("system_health")
