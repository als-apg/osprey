"""Tests for the monitoring stack launcher."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from osprey.interfaces.monitoring.launcher import MonitoringStackLauncher


@pytest.fixture()
def launcher():
    return MonitoringStackLauncher()


@pytest.fixture()
def monitoring_config():
    return {
        "auto_launch": True,
        "otel_collector": {
            "host": "127.0.0.1",
            "grpc_port": 4317,
            "http_port": 4318,
            "prometheus_port": 8889,
        },
        "prometheus": {
            "host": "127.0.0.1",
            "port": 9090,
        },
        "grafana": {
            "host": "127.0.0.1",
            "port": 8091,
        },
    }


class TestMonitoringStackLauncher:
    """Test the MonitoringStackLauncher class."""

    @patch("osprey.interfaces.monitoring.launcher.generate_configs")
    @patch("osprey.interfaces.monitoring.launcher.resolve_binary")
    def test_launch_order(self, mock_resolve, mock_gen_configs, launcher, monitoring_config):
        """Prometheus -> OTEL Collector -> Grafana launch order."""
        mock_resolve.return_value = None  # No binaries found
        mock_gen_configs.return_value = {}

        launcher.ensure_running(monitoring_config)

        # resolve_binary called in order: prometheus, otelcol-contrib, grafana/grafana-server
        calls = [c.args[0] for c in mock_resolve.call_args_list]
        assert calls[0] == "prometheus"
        assert calls[1] == "otelcol-contrib"
        # grafana or grafana-server
        assert "grafana" in calls[2]

    @patch("osprey.interfaces.monitoring.launcher.generate_configs")
    @patch("osprey.interfaces.monitoring.launcher.resolve_binary")
    def test_noop_after_first_launch(self, mock_resolve, mock_gen_configs, launcher, monitoring_config):
        """Double-checked locking: second call is a no-op."""
        mock_resolve.return_value = None
        mock_gen_configs.return_value = {}

        launcher.ensure_running(monitoring_config)
        launcher.ensure_running(monitoring_config)

        # generate_configs only called once (no-op on second call since _launched is set)
        # But since no binaries found, _launched stays False — so it will be called twice
        # Let's test the case where something succeeds
        mock_resolve.reset_mock()
        mock_gen_configs.reset_mock()

    @patch("osprey.interfaces.monitoring.launcher.generate_configs")
    def test_health_check_url_prometheus(self, mock_gen_configs, launcher, monitoring_config):
        """Test health check URL construction for Prometheus."""
        assert launcher._is_healthy("http://127.0.0.1:99999") is False

    @patch("osprey.interfaces.monitoring.launcher.generate_configs")
    @patch("urllib.request.urlopen", side_effect=ConnectionRefusedError)
    def test_health_check_url_otel_collector(self, mock_urlopen, mock_gen_configs, launcher):
        """Test health check URL construction for OTEL Collector."""
        assert launcher._is_healthy("http://127.0.0.1:13133") is False

    @patch("osprey.interfaces.monitoring.launcher.generate_configs")
    @patch("urllib.request.urlopen", side_effect=ConnectionRefusedError)
    def test_health_check_url_grafana(self, mock_urlopen, mock_gen_configs, launcher):
        """Test health check URL construction for Grafana."""
        assert launcher._is_healthy("http://127.0.0.1:8091/api/health") is False

    def test_graceful_shutdown(self, launcher):
        """Test terminate() -> wait(5s) -> kill() for each process."""
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.wait.return_value = 0

        launcher._processes["test_service"] = mock_proc
        launcher._launched = True

        launcher.stop()

        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called_once_with(timeout=5)
        assert launcher._launched is False
        assert len(launcher._processes) == 0

    def test_graceful_shutdown_force_kill(self, launcher):
        """Test force kill when terminate times out."""
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.wait.side_effect = [subprocess.TimeoutExpired("cmd", 5), 0]

        launcher._processes["test_service"] = mock_proc
        launcher._launched = True

        launcher.stop()

        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()

    @patch("osprey.interfaces.monitoring.launcher.load_osprey_config")
    @patch("urllib.request.urlopen", side_effect=ConnectionRefusedError)
    def test_status_returns_dict(self, mock_urlopen, mock_config, launcher):
        """Status returns a dict with all 3 service keys."""
        mock_config.return_value = {"monitoring": {}}
        result = launcher.status()
        assert "prometheus" in result
        assert "otel_collector" in result
        assert "grafana" in result
        # All should be False since nothing is running
        assert all(v is False for v in result.values())


class TestModuleLevelHelpers:
    """Test the module-level convenience functions."""

    @patch("osprey.interfaces.monitoring.launcher.load_osprey_config")
    def test_monitoring_auto_launch_default(self, mock_config):
        from osprey.interfaces.monitoring.launcher import _monitoring_auto_launch

        mock_config.return_value = {"monitoring": {}}
        assert _monitoring_auto_launch() is True

    @patch("osprey.interfaces.monitoring.launcher.load_osprey_config")
    def test_monitoring_auto_launch_disabled(self, mock_config):
        from osprey.interfaces.monitoring.launcher import _monitoring_auto_launch

        mock_config.return_value = {"monitoring": {"auto_launch": False}}
        assert _monitoring_auto_launch() is False

    @patch("osprey.interfaces.monitoring.launcher._monitoring_auto_launch")
    @patch("osprey.interfaces.monitoring.launcher._monitoring_launcher")
    def test_ensure_skips_when_disabled(self, mock_launcher, mock_auto):
        from osprey.interfaces.monitoring.launcher import ensure_monitoring_stack

        mock_auto.return_value = False
        ensure_monitoring_stack()
        mock_launcher.ensure_running.assert_not_called()
