"""Tests for the monitoring config generator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from osprey.interfaces.monitoring.config_generator import generate_configs


@pytest.fixture()
def monitoring_config():
    """Default monitoring config for tests."""
    return {
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


@pytest.fixture()
def mock_data_base(tmp_path):
    """Redirect config/data dirs to tmp_path for test isolation."""
    config_dir = tmp_path / "config"
    with (
        patch("osprey.interfaces.monitoring.config_generator.CONFIG_DIR", config_dir),
        patch("osprey.interfaces.monitoring.config_generator.DATA_BASE", tmp_path),
    ):
        yield tmp_path


class TestGenerateConfigs:
    """Test Jinja2 config rendering."""

    def test_renders_all_configs(self, monitoring_config, mock_data_base):
        result = generate_configs(monitoring_config)
        assert "otel_collector" in result
        assert "prometheus" in result
        assert "grafana" in result

    def test_otel_collector_yaml_contains_ports(self, monitoring_config, mock_data_base):
        result = generate_configs(monitoring_config)
        content = result["otel_collector"].read_text()
        assert "4317" in content
        assert "4318" in content
        assert "8889" in content

    def test_otel_collector_has_hostmetrics_receiver(self, monitoring_config, mock_data_base):
        result = generate_configs(monitoring_config)
        content = result["otel_collector"].read_text()
        assert "hostmetrics:" in content
        assert "cpu:" in content
        assert "memory:" in content
        assert "disk:" in content
        assert "load:" in content
        assert "network:" in content

    def test_otel_collector_hostmetrics_in_pipeline(self, monitoring_config, mock_data_base):
        result = generate_configs(monitoring_config)
        content = result["otel_collector"].read_text()
        assert "hostmetrics" in content
        # hostmetrics should be in the metrics pipeline receivers
        assert "receivers: [otlp, hostmetrics]" in content

    def test_prometheus_yaml_has_scrape_targets(self, monitoring_config, mock_data_base):
        result = generate_configs(monitoring_config)
        content = result["prometheus"].read_text()
        assert "otel-collector" in content
        assert "8889" in content
        assert "9090" in content

    def test_grafana_ini_has_allow_embedding(self, monitoring_config, mock_data_base):
        result = generate_configs(monitoring_config)
        content = result["grafana"].read_text()
        assert "allow_embedding = true" in content

    def test_grafana_ini_has_anonymous_auth(self, monitoring_config, mock_data_base):
        result = generate_configs(monitoring_config)
        content = result["grafana"].read_text()
        assert "[auth.anonymous]" in content
        assert "enabled = true" in content
        assert "org_role = Viewer" in content

    def test_grafana_ini_disables_unnecessary_features(self, monitoring_config, mock_data_base):
        result = generate_configs(monitoring_config)
        content = result["grafana"].read_text()
        assert "[explore]" in content
        assert "[alerting]" in content
        assert "enabled = false" in content

    def test_grafana_ini_sets_home_dashboard(self, monitoring_config, mock_data_base):
        result = generate_configs(monitoring_config)
        content = result["grafana"].read_text()
        assert "default_home_dashboard_path" in content
        assert "claude-code-overview.json" in content

    def test_grafana_port_configurable(self, mock_data_base):
        config = {
            "grafana": {"host": "0.0.0.0", "port": 4000},
        }
        result = generate_configs(config)
        content = result["grafana"].read_text()
        assert "http_port = 4000" in content
        assert "http_addr = 0.0.0.0" in content

    def test_provisioning_datasource_references_prometheus(
        self, monitoring_config, mock_data_base
    ):
        result = generate_configs(monitoring_config)
        content = result["grafana_datasource"].read_text()
        assert "http://127.0.0.1:9090" in content
        assert "type: prometheus" in content

    def test_provisioning_dashboard_provider(self, monitoring_config, mock_data_base):
        result = generate_configs(monitoring_config)
        content = result["grafana_dashboard_provider"].read_text()
        assert "OSPREY" in content
        assert "dashboards" in content

    def test_creates_data_directories(self, monitoring_config, mock_data_base):
        generate_configs(monitoring_config)
        assert (mock_data_base / "prometheus" / "data").is_dir()
        assert (mock_data_base / "grafana" / "data").is_dir()
        assert (mock_data_base / "grafana" / "provisioning" / "datasources").is_dir()
        assert (mock_data_base / "logs").is_dir()

    def test_default_values_when_config_empty(self, mock_data_base):
        """Empty config should use defaults for all values."""
        result = generate_configs({})
        assert result["otel_collector"].exists()
        content = result["otel_collector"].read_text()
        # Defaults
        assert "127.0.0.1:4317" in content
        assert "127.0.0.1:4318" in content

    def test_dashboard_json_copied(self, monitoring_config, mock_data_base):
        result = generate_configs(monitoring_config)
        if "grafana_dashboard" in result:
            content = result["grafana_dashboard"].read_text()
            assert "Claude Code Overview" in content
