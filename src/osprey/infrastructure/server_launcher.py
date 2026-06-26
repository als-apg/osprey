"""Generic auto-launcher for OSPREY companion servers.

Provides a reusable ``ServerLauncher`` that starts a uvicorn server
in a daemon thread on first demand.  Server definitions live in
``registry.web`` — this module uses ``importlib`` to resolve factories
at call time so the infrastructure layer never imports from interfaces/.
"""

from __future__ import annotations

import importlib
import logging
import threading
import urllib.request
from collections.abc import Callable
from pathlib import Path

from osprey.registry.web import FRAMEWORK_WEB_SERVERS, WebServerDefinition
from osprey.utils.workspace import load_osprey_config

logger = logging.getLogger("osprey.infrastructure.server_launcher")


class ServerLauncher:
    """Double-checked-locking launcher for a uvicorn companion server.

    Args:
        name: Human-readable server name (for logging).
        config_reader: Callable returning ``(host, port)`` from config.
        auto_launch_checker: Callable returning ``True`` if auto-launch is enabled.
        app_factory: Callable returning a ASGI app instance. Receives
            ``workspace_root`` kwarg if ``pass_workspace`` is True.
        pass_workspace: If True, resolve and pass ``workspace_root`` to the app factory.
    """

    def __init__(
        self,
        name: str,
        config_reader: Callable[[], tuple[str, int]],
        auto_launch_checker: Callable[[], bool],
        app_factory: Callable[..., object],
        pass_workspace: bool = False,
    ) -> None:
        self._name = name
        self._config_reader = config_reader
        self._auto_launch_checker = auto_launch_checker
        self._app_factory = app_factory
        self._pass_workspace = pass_workspace
        self._launched = False
        self._lock = threading.Lock()

    def _is_running(self, host: str, port: int) -> bool:
        """Check the /health endpoint (quick, no dependencies)."""
        try:
            url = f"http://{host}:{port}/health"
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

        # Brief health-check to verify server came up
        import time

        for _attempt in range(3):
            time.sleep(0.5)
            if self._is_running(host, port):
                logger.info("%s health check passed", self._name)
                self._launched = True
                return

        if not t.is_alive():
            # Thread already exited (crashed) — don't mark as launched
            logger.warning("%s thread exited before health check passed", self._name)
            self._launched = False
        else:
            logger.warning(
                "%s health check failed after launch — server may not be reachable at %s:%s",
                self._name,
                host,
                port,
            )
            # Still mark as launched to avoid busy-retry loops; the crash handler
            # in _run() will reset _launched if the thread actually crashes.
            self._launched = True

    def ensure_running(self) -> None:
        """Ensure the server is running; launch if needed.

        Safe to call multiple times — no-op after first launch.
        """
        if not self._auto_launch_checker():
            return

        if self._launched:
            return

        with self._lock:
            if self._launched:
                return

            host, port = self._config_reader()

            if self._is_running(host, port):
                self._launched = True
                return

            self._launch_in_thread(host, port)


# ---------------------------------------------------------------------------
# Generic helpers — build ServerLauncher callbacks from WebServerDefinition
# ---------------------------------------------------------------------------


def _make_config_reader(defn: WebServerDefinition) -> Callable[[], tuple[str, int]]:
    """Return a callable that reads (host, port) from config for *defn*.

    Port can be overridden via environment variable
    ``OSPREY_{CONFIG_KEY}_PORT`` (upper-cased), e.g.
    ``OSPREY_ARTIFACT_SERVER_PORT=8186``.  This is needed for
    ``--network host`` deployments where multiple containers share the
    host network and must avoid port collisions.
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
        env_key = f"OSPREY_{defn.config_key.upper()}_PORT"
        env_val = os.environ.get(env_key)
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
    """Traverse a dotted path like ``"tuning.api_url"`` into *config*."""
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


def ensure_tuning_server() -> None:
    """Ensure the tuning panel server is running; launch if needed."""
    ensure_web_server("tuning")


def ensure_channel_finder_server() -> None:
    """Ensure the Channel Finder web server is running; launch if needed."""
    ensure_web_server("channel_finder")


def ensure_lattice_dashboard_server() -> None:
    """Ensure the lattice dashboard server is running; launch if needed."""
    ensure_web_server("lattice_dashboard")
