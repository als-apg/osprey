"""Monitoring stack subprocess manager for the OSPREY Web Terminal.

Launches OTEL Collector, Prometheus, and Grafana as subprocesses,
analogous to ``CUIProcessLauncher`` in ``osprey.interfaces.cui.launcher``.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from osprey.interfaces.monitoring.config_generator import DATA_BASE, generate_configs
from osprey.interfaces.monitoring.installer import resolve_binary
from osprey.mcp_server.common import load_osprey_config

logger = logging.getLogger(__name__)


class MonitoringStackLauncher:
    """Manages OTEL Collector + Prometheus + Grafana as subprocesses."""

    def __init__(self) -> None:
        self._processes: dict[str, subprocess.Popen] = {}
        self._launched = False
        self._lock = threading.Lock()

    @staticmethod
    def _is_healthy(url: str) -> bool:
        """Check a health endpoint (quick, no dependencies)."""
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _wait_for_health(self, name: str, url: str, process: subprocess.Popen) -> bool:
        """Poll a health endpoint until ready (up to 30s)."""
        for _ in range(30):
            if self._is_healthy(url):
                logger.info("%s ready", name)
                return True
            if process.poll() is not None:
                logger.error("%s exited early (rc=%s)", name, process.returncode)
                return False
            time.sleep(1)
        logger.warning("%s did not become healthy within 30s", name)
        return False

    def _launch_prometheus(self, config: dict) -> bool:
        """Launch Prometheus."""
        binary = resolve_binary("prometheus")
        if not binary:
            logger.warning("prometheus binary not found — run `osprey monitoring install`")
            return False

        prom = config.get("prometheus", {})
        host = prom.get("host", "127.0.0.1")
        port = prom.get("port", 9090)

        health_url = f"http://{host}:{port}/-/ready"
        if self._is_healthy(health_url):
            logger.info("Prometheus already running at %s:%s", host, port)
            return True

        config_path = DATA_BASE / "config" / "prometheus.yml"
        data_dir = DATA_BASE / "prometheus" / "data"
        log_file = DATA_BASE / "logs" / "prometheus.log"

        cmd = [
            str(binary),
            f"--config.file={config_path}",
            f"--storage.tsdb.path={data_dir}",
            f"--web.listen-address={host}:{port}",
            "--storage.tsdb.retention.time=7d",
        ]

        with open(log_file, "a") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=lf)

        self._processes["prometheus"] = proc
        logger.info("Prometheus launched (pid=%s) at http://%s:%s", proc.pid, host, port)
        return self._wait_for_health("Prometheus", health_url, proc)

    def _launch_otel_collector(self, config: dict) -> bool:
        """Launch the OTEL Collector."""
        binary = resolve_binary("otelcol-contrib")
        if not binary:
            logger.warning("otelcol-contrib binary not found — run `osprey monitoring install`")
            return False

        otel = config.get("otel_collector", {})
        host = otel.get("host", "127.0.0.1")

        health_url = f"http://{host}:13133"
        if self._is_healthy(health_url):
            logger.info("OTEL Collector already running")
            return True

        config_path = DATA_BASE / "config" / "otel-collector.yaml"
        log_file = DATA_BASE / "logs" / "otel-collector.log"

        cmd = [str(binary), "--config", str(config_path)]

        with open(log_file, "a") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=lf)

        self._processes["otel_collector"] = proc
        logger.info("OTEL Collector launched (pid=%s)", proc.pid)
        return self._wait_for_health("OTEL Collector", health_url, proc)

    def _launch_grafana(self, config: dict) -> bool:
        """Launch Grafana."""
        binary = resolve_binary("grafana")
        if not binary:
            # Also try grafana-server
            binary = resolve_binary("grafana-server")
        if not binary:
            logger.warning("grafana binary not found — run `osprey monitoring install`")
            return False

        grafana = config.get("grafana", {})
        host = grafana.get("host", "127.0.0.1")
        port = grafana.get("port", 8091)

        health_url = f"http://{host}:{port}/api/health"
        if self._is_healthy(health_url):
            logger.info("Grafana already running at %s:%s", host, port)
            return True

        config_path = DATA_BASE / "config" / "grafana.ini"
        log_file = DATA_BASE / "logs" / "grafana.log"

        # Grafana needs a homepath for its public/ assets
        grafana_home = DATA_BASE / "grafana-home"
        if not grafana_home.exists():
            grafana_home = Path("/usr/share/grafana")  # system default

        cmd = [
            str(binary),
            "server",
            f"--config={config_path}",
            f"--homepath={grafana_home}",
        ]

        with open(log_file, "a") as lf:
            proc = subprocess.Popen(cmd, stdout=lf, stderr=lf)

        self._processes["grafana"] = proc
        logger.info("Grafana launched (pid=%s) at http://%s:%s", proc.pid, host, port)
        return self._wait_for_health("Grafana", health_url, proc)

    def ensure_running(self, monitoring_config: dict) -> None:
        """Ensure all monitoring services are running; launch if needed.

        Safe to call multiple times — no-op after first launch.
        Launch order: Prometheus -> OTEL Collector -> Grafana.
        """
        if self._launched:
            return

        with self._lock:
            if self._launched:
                return

            # Generate configs before launching
            generate_configs(monitoring_config)

            # Launch in dependency order
            prom_ok = self._launch_prometheus(monitoring_config)
            otel_ok = self._launch_otel_collector(monitoring_config)
            grafana_ok = self._launch_grafana(monitoring_config)

            if prom_ok or otel_ok or grafana_ok:
                self._launched = True

            if not (prom_ok and otel_ok and grafana_ok):
                logger.warning(
                    "Monitoring stack partially launched: "
                    "prometheus=%s otel=%s grafana=%s",
                    prom_ok,
                    otel_ok,
                    grafana_ok,
                )

    def stop(self) -> None:
        """Terminate all monitoring subprocesses gracefully."""
        for name, proc in list(self._processes.items()):
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                logger.info("%s stopped", name)
            except Exception:
                logger.warning("Error stopping %s", name, exc_info=True)

        self._processes.clear()
        self._launched = False

    def status(self) -> dict[str, bool]:
        """Return health status of each monitoring service."""
        config = _monitoring_config()
        otel = config.get("otel_collector", {})
        prom = config.get("prometheus", {})
        grafana = config.get("grafana", {})

        otel_host = otel.get("host", "127.0.0.1")
        prom_host = prom.get("host", "127.0.0.1")
        prom_port = prom.get("port", 9090)
        grafana_host = grafana.get("host", "127.0.0.1")
        grafana_port = grafana.get("port", 3000)

        return {
            "prometheus": self._is_healthy(f"http://{prom_host}:{prom_port}/-/ready"),
            "otel_collector": self._is_healthy(f"http://{otel_host}:13133"),
            "grafana": self._is_healthy(f"http://{grafana_host}:{grafana_port}/api/health"),
        }


# ---------------------------------------------------------------------------
# Module-level helpers (same pattern as server_launcher.py / cui launcher)
# ---------------------------------------------------------------------------


def _monitoring_config() -> dict:
    config = load_osprey_config()
    return config.get("monitoring", {})


def _monitoring_auto_launch() -> bool:
    return _monitoring_config().get("auto_launch", True)


_monitoring_launcher = MonitoringStackLauncher()


def ensure_monitoring_stack() -> None:
    """Ensure the monitoring stack is running; launch if needed."""
    if not _monitoring_auto_launch():
        return
    _monitoring_launcher.ensure_running(_monitoring_config())


def stop_monitoring_stack() -> None:
    """Stop all monitoring subprocesses."""
    _monitoring_launcher.stop()
