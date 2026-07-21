"""Generic auto-launcher for OSPREY companion servers.

Provides a reusable ``ServerLauncher`` that starts a uvicorn server
in a daemon thread on first demand.  Server definitions live in
``registry.web`` — this module uses ``importlib`` to resolve factories
at call time so the infrastructure layer never imports from interfaces/.
"""

from __future__ import annotations

import importlib
import logging
import socket
import threading
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

from osprey.registry.web import FRAMEWORK_WEB_SERVERS, WebServerDefinition
from osprey.utils.workspace import load_osprey_config

logger = logging.getLogger("osprey.infrastructure.server_launcher")

# When a port is already held on first check, it may be a predecessor that is
# still shutting down after a restart. Wait a bounded grace period for it to
# release the port so we can bind (own) it ourselves rather than trusting a
# possibly-dying external responder. See issue #327.
_PORT_RELEASE_GRACE_ATTEMPTS = 5
_PORT_RELEASE_GRACE_INTERVAL = 0.5

# When a port stays held by a process that does not answer /health, we cannot
# bind it now, but it may free up later (a lazy caller like artifact_store
# re-invokes ensure_running on every save). Re-probe at most this often so a
# panel can self-heal without paying the grace cost on every call.
_HELD_PORT_RETRY_COOLDOWN = 30.0


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
        # a "port held by a non-responder" outcome (see ensure_running).
        self._retry_not_before = 0.0
        self._lock = threading.Lock()

    def _port_has_listener(self, host: str, port: int) -> bool:
        """Return True if a live TCP listener is bound to *host:port*.

        This is a truer "is the port taken?" signal than a ``/health`` 200:
        it answers the ownership question directly instead of conflating
        "the port is bound" with "the bound thing is a healthy HTTP server".
        A wildcard bind host is probed via loopback, which the same listener
        also accepts.
        """
        try:
            with socket.create_connection((_loopback_for(host), port), timeout=1):
                return True
        except OSError:
            return False

    def _is_running(self, host: str, port: int) -> bool:
        """Check the /health endpoint (quick, no dependencies)."""
        probe_host = _loopback_for(host)
        netloc = f"[{probe_host}]" if ":" in probe_host else probe_host
        try:
            url = f"http://{netloc}:{port}/health"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=1) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _launch_in_thread(self, host: str, port: int) -> None:
        """Start uvicorn in a daemon thread."""

        def _run() -> None:
            try:
                import uvicorn

                if self._pass_workspace:
                    from osprey.utils.workspace import resolve_workspace_root

                    app = self._app_factory(workspace_root=resolve_workspace_root())
                else:
                    app = self._app_factory()
                uvicorn.run(app, host=host, port=port, log_level="warning")
            except Exception:
                logger.exception("%s thread crashed", self._name)
                self._launched = False

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

        Ownership is decided by whether a live TCP listener holds the port,
        not by a bare ``/health`` 200. A ``/health`` 200 from a stale or
        foreign responder is a false positive: it previously made the manager
        skip the launch and leave the panel unbacked (proxy 502) after a
        restart (issue #327). Instead:

        * port free            -> launch and own it;
        * port held then freed -> a shutting-down predecessor; waited out,
          then launched;
        * port held throughout  -> defer to a legitimate external server if it
          serves ``/health`` (latched); else warn and retry on a later call so a
          lazily-relaunched panel can self-heal once the port frees.
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

            # Nothing listening -> the port is ours to take.
            if not self._port_has_listener(host, port):
                self._launch_in_thread(host, port)
                return

            # Held on first check. Give a shutting-down predecessor a bounded
            # grace period to release the port so we can bind it ourselves.
            for _attempt in range(self._release_grace_attempts):
                time.sleep(self._release_grace_interval)
                if not self._port_has_listener(host, port):
                    self._launch_in_thread(host, port)
                    return

            # Still held after the grace window. A live /health responder is a
            # legitimate server owned by another process — defer to it (latched).
            if self._is_running(host, port):
                logger.info(
                    "%s already served by another process at %s:%s — deferring launch",
                    self._name,
                    host,
                    port,
                )
                self._launched = True
            else:
                # Held by something that does not answer /health; we cannot bind
                # now. Do not latch — throttle and retry so the panel recovers if
                # the port frees later.
                logger.warning(
                    "%s port %s:%s is held by a process that does not answer /health; "
                    "the panel will be unbacked (502) until the port is free. "
                    "Will retry on a later call.",
                    self._name,
                    host,
                    port,
                )
                self._retry_not_before = time.monotonic() + self._held_port_retry_cooldown


# ---------------------------------------------------------------------------
# Generic helpers — build ServerLauncher callbacks from WebServerDefinition
# ---------------------------------------------------------------------------


def _make_config_reader(defn: WebServerDefinition) -> Callable[[], tuple[str, int]]:
    """Return a callable that reads (host, port) from config for *defn*.

    Port can be overridden via the environment variable
    ``defn.port_env_var`` (``OSPREY_{CONFIG_KEY}_PORT``), e.g.
    ``OSPREY_ARTIFACT_SERVER_PORT=8186``.  This is needed for
    ``--network host`` deployments where multiple containers share the
    host network and must avoid port collisions — the multi-user compose
    render sets these per user from the same ``port_env_var`` derivation.
    """

    def _reader() -> tuple[str, int]:
        import os

        config = load_osprey_config()
        section = config.get(defn.config_key, {})
        if defn.config_web_subkey:
            section = section.get(defn.config_web_subkey, {})
        host = section.get("host", defn.host_default)
        port = section.get("port", defn.port_default)

        # Environment override — useful for host-network multi-container deploys
        env_val = os.environ.get(defn.port_env_var)
        if env_val:
            port = int(env_val)

        return host, port

    return _reader


def _make_auto_launch_checker(defn: WebServerDefinition) -> Callable[[], bool]:
    """Return a callable that checks whether auto-launch is enabled."""

    def _checker() -> bool:
        config = load_osprey_config()
        top = config.get(defn.config_key, {})
        if defn.require_section and not top:
            return False
        section = top.get(defn.config_web_subkey, {}) if defn.config_web_subkey else top
        return section.get("auto_launch", defn.auto_launch_default)

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

_launchers: dict[str, ServerLauncher] = {
    key: ServerLauncher(
        name=defn.name,
        config_reader=_make_config_reader(defn),
        auto_launch_checker=_make_auto_launch_checker(defn),
        app_factory=_make_app_factory(defn),
        pass_workspace=defn.pass_workspace,
    )
    for key, defn in FRAMEWORK_WEB_SERVERS.items()
}


def ensure_web_server(key: str) -> None:
    """Ensure the web server identified by *key* is running."""
    _launchers[key].ensure_running()


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
