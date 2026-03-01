"""CUI subprocess manager for the OSPREY Web Terminal.

Launches ``cui-server`` (npm package) as a subprocess, analogous to
``ServerLauncher`` but using ``subprocess.Popen`` instead of uvicorn
daemon threads.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
import urllib.request

from osprey.utils.workspace import load_osprey_config

logger = logging.getLogger("osprey.interfaces.cui.launcher")


class CUIProcessLauncher:
    """Double-checked-locking launcher for the CUI subprocess."""

    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None
        self._launched = False
        self._lock = threading.Lock()

    @staticmethod
    def _resolve_command(host: str, port: int) -> list[str]:
        """Build the command list for launching cui-server.

        Prefers a globally-installed ``cui-server`` binary; falls back
        to ``npx cui-server`` if available.

        Raises:
            FileNotFoundError: Neither ``cui-server`` nor ``npx`` found.
        """
        cui_bin = shutil.which("cui-server")
        if cui_bin:
            return [cui_bin, "--port", str(port), "--host", host, "--skip-auth-token"]

        npx_bin = shutil.which("npx")
        if npx_bin:
            return [npx_bin, "cui-server", "--port", str(port), "--host", host, "--skip-auth-token"]

        raise FileNotFoundError("cui-server not found. Install with: npm install -g cui-server")

    @staticmethod
    def _is_running(host: str, port: int) -> bool:
        """Check the /health endpoint (quick, no dependencies)."""
        try:
            url = f"http://{host}:{port}/health"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=1) as resp:
                return resp.status == 200
        except Exception:
            return False

    def ensure_running(self, host: str, port: int, cwd: str | None = None) -> None:
        """Ensure CUI server is running; launch if needed.

        Safe to call multiple times — no-op after first launch.
        """
        if self._launched:
            return

        with self._lock:
            if self._launched:
                return

            if self._is_running(host, port):
                self._launched = True
                return

            cmd = self._resolve_command(host, port)
            self._process = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._launched = True
            logger.info(
                "CUI server launched (pid=%s) at http://%s:%s", self._process.pid, host, port
            )

            # Wait for the server to become ready (npx may need to download first)
            for _ in range(30):
                if self._is_running(host, port):
                    logger.info("CUI server ready")
                    return
                # Check if process died
                if self._process.poll() is not None:
                    logger.error("CUI server exited early (rc=%s)", self._process.returncode)
                    self._process = None
                    self._launched = False
                    return
                time.sleep(1)

    def stop(self) -> None:
        """Terminate the CUI subprocess gracefully."""
        if self._process is None:
            return

        self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        logger.info("CUI server stopped")
        self._process = None
        self._launched = False


# ---------------------------------------------------------------------------
# Module-level helpers (same pattern as server_launcher.py)
# ---------------------------------------------------------------------------


def _cui_config() -> tuple[str, int]:
    config = load_osprey_config()
    cui = config.get("cui_server", {})
    return cui.get("host", "127.0.0.1"), cui.get("port", 3001)


def _cui_auto_launch() -> bool:
    config = load_osprey_config()
    return config.get("cui_server", {}).get("auto_launch", True)


_cui_launcher = CUIProcessLauncher()


def ensure_cui_server(cwd: str | None = None) -> None:
    """Ensure the CUI server is running; launch if needed."""
    if not _cui_auto_launch():
        return
    host, port = _cui_config()
    _cui_launcher.ensure_running(host, port, cwd=cwd)


def stop_cui_server() -> None:
    """Stop the CUI server subprocess."""
    _cui_launcher.stop()
