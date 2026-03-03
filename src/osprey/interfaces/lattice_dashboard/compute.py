"""Compute manager — subprocess orchestration for figure workers.

Spawns workers as subprocesses, monitors completion, updates state,
and broadcasts SSE events when figures are ready.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from osprey.interfaces.lattice_dashboard.state import (
    ALL_FIGURES,
    FAST_FIGURES,
    VERIFICATION_FIGURES,
    LatticeState,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger("osprey.lattice_dashboard.compute")


class ComputeManager:
    """Manages subprocess workers for lattice figure computation.

    Args:
        state: LatticeState instance for reading/writing state.
        broadcaster: Object with a ``broadcast(data)`` method for SSE push.
    """

    def __init__(self, state: LatticeState, broadcaster: Any) -> None:
        self._state = state
        self._broadcaster = broadcaster
        self._processes: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    def refresh_fast(self) -> list[str]:
        """Cancel running fast workers and recompute all 4 fast figures."""
        launched = []
        for name in FAST_FIGURES:
            self._launch_worker(name)
            launched.append(name)
        return launched

    def refresh_verification(self) -> list[str]:
        """Launch DA + LMA verification workers."""
        launched = []
        for name in VERIFICATION_FIGURES:
            self._launch_worker(name)
            launched.append(name)
        return launched

    def refresh_one(self, name: str) -> None:
        """Launch a single figure worker."""
        if name not in ALL_FIGURES:
            raise ValueError(f"Unknown figure: {name}. Must be one of {ALL_FIGURES}")
        self._launch_worker(name)

    def _launch_worker(self, name: str) -> None:
        """Spawn a subprocess for the named worker."""
        # Cancel if already running
        with self._lock:
            proc = self._processes.get(name)
            if proc is not None and proc.poll() is None:
                logger.info("Cancelling running %s worker (pid=%d)", name, proc.pid)
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()

        state_path = self._state.state_path
        output_path = self._state.figures_dir / f"{name}.json"

        worker_module = f"osprey.interfaces.lattice_dashboard.workers.{name}"
        cmd = [sys.executable, "-m", worker_module, str(state_path), str(output_path)]

        logger.info("Launching %s worker: %s", name, " ".join(cmd))
        self._state.mark_computing(name)
        self._broadcaster.broadcast({"type": "figure_status", "name": name, "status": "computing"})

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except Exception as exc:
            error_msg = f"Failed to launch worker: {exc}"
            logger.exception("Worker launch failed for %s", name)
            self._state.mark_error(name, error_msg)
            self._broadcaster.broadcast({"type": "figure_error", "name": name, "error": error_msg})
            return

        with self._lock:
            self._processes[name] = proc

        # Monitor in background thread
        t = threading.Thread(
            target=self._monitor_worker,
            args=(name, proc, output_path),
            daemon=True,
            name=f"lattice-worker-{name}",
        )
        t.start()

    def _monitor_worker(self, name: str, proc: subprocess.Popen, output_path: Path) -> None:
        """Wait for worker to complete and update state accordingly."""
        try:
            stdout, stderr = proc.communicate(timeout=300)  # 5 min max
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            error_msg = "Worker timed out after 300s"
            logger.warning("%s worker timed out", name)
            self._state.mark_error(name, error_msg)
            self._broadcaster.broadcast({"type": "figure_error", "name": name, "error": error_msg})
            return

        if proc.returncode != 0:
            stderr_text = stderr.decode("utf-8", errors="replace")[-500:]
            error_msg = f"Worker exited with code {proc.returncode}: {stderr_text}"
            logger.warning("%s worker failed: %s", name, error_msg)
            self._state.mark_error(name, error_msg)
            self._broadcaster.broadcast({"type": "figure_error", "name": name, "error": error_msg})
            return

        if not output_path.exists():
            error_msg = "Worker completed but no output file produced"
            logger.warning("%s: %s", name, error_msg)
            self._state.mark_error(name, error_msg)
            self._broadcaster.broadcast({"type": "figure_error", "name": name, "error": error_msg})
            return

        logger.info("%s worker completed successfully", name)
        self._state.mark_ready(name)
        self._broadcaster.broadcast({"type": "figure_ready", "name": name})

    def cancel_all(self) -> None:
        """Terminate all running workers."""
        with self._lock:
            for name, proc in self._processes.items():
                if proc.poll() is None:
                    logger.info("Terminating %s worker", name)
                    proc.terminate()
