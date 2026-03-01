"""Generic auto-launcher for OSPREY companion servers.

Provides a reusable ``ServerLauncher`` that starts a uvicorn server
in a daemon thread on first demand. Used by the Artifact Gallery and
ARIEL logbook server launchers.
"""

from __future__ import annotations

import logging
import threading
import urllib.request
from collections.abc import Callable
from pathlib import Path

from osprey.utils.workspace import load_osprey_config

logger = logging.getLogger("osprey.mcp_server.server_launcher")


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


def _artifact_config() -> tuple[str, int]:
    config = load_osprey_config()
    art = config.get("artifact_server", {})
    return art.get("host", "127.0.0.1"), art.get("port", 8086)


def _artifact_auto_launch() -> bool:
    config = load_osprey_config()
    return config.get("artifact_server", {}).get("auto_launch", True)


def _artifact_app_factory(workspace_root: Path | None = None) -> object:
    from osprey.interfaces.artifacts.app import create_app

    return create_app(workspace_root=workspace_root)


_artifact_launcher = ServerLauncher(
    name="Artifact gallery",
    config_reader=_artifact_config,
    auto_launch_checker=_artifact_auto_launch,
    app_factory=_artifact_app_factory,
    pass_workspace=True,
)


def _ariel_config() -> tuple[str, int]:
    config = load_osprey_config()
    ariel = config.get("ariel", {}).get("web", {})
    return ariel.get("host", "127.0.0.1"), ariel.get("port", 8085)


def _ariel_auto_launch() -> bool:
    config = load_osprey_config()
    return config.get("ariel", {}).get("web", {}).get("auto_launch", True)


def _ariel_app_factory() -> object:
    from osprey.interfaces.ariel.app import create_app

    return create_app()


_ariel_launcher = ServerLauncher(
    name="ARIEL server",
    config_reader=_ariel_config,
    auto_launch_checker=_ariel_auto_launch,
    app_factory=_ariel_app_factory,
)


def ensure_artifact_server() -> None:
    """Ensure the artifact server is running; launch if needed."""
    _artifact_launcher.ensure_running()


def ensure_ariel_server() -> None:
    """Ensure the ARIEL server is running; launch if needed."""
    _ariel_launcher.ensure_running()


def _tuning_config() -> tuple[str, int]:
    config = load_osprey_config()
    tuning_web = config.get("tuning", {}).get("web", {})
    return tuning_web.get("host", "127.0.0.1"), tuning_web.get("port", 8090)


def _tuning_auto_launch() -> bool:
    config = load_osprey_config()
    return config.get("tuning", {}).get("web", {}).get("auto_launch", True)


def _tuning_app_factory() -> object:
    from osprey.interfaces.tuning.app import create_app

    config = load_osprey_config()
    api_url = config.get("tuning", {}).get("api_url")
    return create_app(tuning_api_url=api_url)


_tuning_launcher = ServerLauncher(
    name="Tuning panel",
    config_reader=_tuning_config,
    auto_launch_checker=_tuning_auto_launch,
    app_factory=_tuning_app_factory,
)


def ensure_tuning_server() -> None:
    """Ensure the tuning panel server is running; launch if needed."""
    _tuning_launcher.ensure_running()


def _deplot_config() -> tuple[str, int]:
    config = load_osprey_config()
    deplot = config.get("deplot", {})
    return deplot.get("host", "127.0.0.1"), deplot.get("port", 8095)


def _deplot_auto_launch() -> bool:
    config = load_osprey_config()
    deplot = config.get("deplot", {})
    if not deplot:
        return False  # No deplot section → don't launch
    return deplot.get("auto_launch", True)


def _deplot_app_factory() -> object:
    try:
        from osprey.services.deplot.server import create_app

        return create_app()
    except ImportError as err:
        raise ImportError(
            "DePlot dependencies not installed. Install with: uv sync --extra graph"
        ) from err


_deplot_launcher = ServerLauncher(
    name="DePlot service",
    config_reader=_deplot_config,
    auto_launch_checker=_deplot_auto_launch,
    app_factory=_deplot_app_factory,
)


def ensure_deplot_server() -> None:
    """Ensure the DePlot service is running; launch if needed."""
    _deplot_launcher.ensure_running()


def _channel_finder_config() -> tuple[str, int]:
    config = load_osprey_config()
    cf = config.get("channel_finder", {}).get("web", {})
    return cf.get("host", "127.0.0.1"), cf.get("port", 8092)


def _channel_finder_auto_launch() -> bool:
    config = load_osprey_config()
    cf = config.get("channel_finder", {})
    if not cf:
        return False  # No channel_finder section → don't launch
    return cf.get("web", {}).get("auto_launch", True)


def _channel_finder_app_factory() -> object:
    from osprey.interfaces.channel_finder.app import create_app

    return create_app()


_channel_finder_launcher = ServerLauncher(
    name="Channel Finder",
    config_reader=_channel_finder_config,
    auto_launch_checker=_channel_finder_auto_launch,
    app_factory=_channel_finder_app_factory,
)


def ensure_channel_finder_server() -> None:
    """Ensure the Channel Finder web server is running; launch if needed."""
    _channel_finder_launcher.ensure_running()
