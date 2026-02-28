"""Agentsview subprocess manager for the OSPREY Web Terminal.

Launches ``agentsview`` (Go binary) as a subprocess.  agentsview
indexes Claude Code session JSONL files from ``~/.claude/projects/``
(its built-in default) and serves a rich analytics UI.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
import urllib.request

from osprey.mcp_server.common import load_osprey_config

logger = logging.getLogger("osprey.interfaces.agentsview.launcher")


class AgentsviewLauncher:
    """Double-checked-locking launcher for the agentsview subprocess."""

    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None
        self._launched = False
        self._lock = threading.Lock()

    @staticmethod
    def _is_running(host: str, port: int) -> bool:
        """Check the /api/v1/stats endpoint (agentsview liveness probe)."""
        try:
            url = f"http://{host}:{port}/api/v1/stats"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=1) as resp:
                return resp.status == 200
        except Exception:
            return False

    def ensure_running(self, host: str, port: int) -> None:
        """Ensure agentsview is running; launch if needed.

        Safe to call multiple times — re-launches if the process died.
        """
        if self._launched and self._process is not None and self._process.poll() is None:
            return

        with self._lock:
            # Re-check under lock: process may have been restarted by another thread
            if self._launched and self._process is not None and self._process.poll() is None:
                return
            # Reset state so we attempt a fresh launch
            self._launched = False

            if self._is_running(host, port):
                self._launched = True
                return

            binary = shutil.which("agentsview")
            if not binary:
                raise FileNotFoundError(
                    "agentsview not found. "
                    "Install with: curl -fsSL https://agentsview.io/install.sh | bash"
                )

            cmd = [binary, "-host", host, "-port", str(port), "-no-browser"]

            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._launched = True
            logger.info(
                "agentsview launched (pid=%s) at http://%s:%s",
                self._process.pid,
                host,
                port,
            )

            # Wait for the server to become ready
            for _ in range(30):
                if self._is_running(host, port):
                    logger.info("agentsview ready")
                    return
                # Check if process died
                if self._process.poll() is not None:
                    logger.error("agentsview exited early (rc=%s)", self._process.returncode)
                    self._process = None
                    self._launched = False
                    return
                time.sleep(1)

    def stop(self) -> None:
        """Terminate the agentsview subprocess gracefully."""
        if self._process is None:
            return

        self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        logger.info("agentsview stopped")
        self._process = None
        self._launched = False


# ---------------------------------------------------------------------------
# Module-level helpers (same pattern as cui/launcher.py)
# ---------------------------------------------------------------------------


def _agentsview_config() -> tuple[str, int]:
    config = load_osprey_config()
    av = config.get("agentsview", {})
    return av.get("host", "127.0.0.1"), av.get("port", 8096)


def _agentsview_auto_launch() -> bool:
    config = load_osprey_config()
    return config.get("agentsview", {}).get("auto_launch", True)


_agentsview_launcher = AgentsviewLauncher()


def ensure_agentsview() -> None:
    """Ensure the agentsview server is running; launch if needed."""
    if not _agentsview_auto_launch():
        return
    host, port = _agentsview_config()
    _agentsview_launcher.ensure_running(host, port)


def stop_agentsview() -> None:
    """Stop the agentsview subprocess."""
    _agentsview_launcher.stop()
