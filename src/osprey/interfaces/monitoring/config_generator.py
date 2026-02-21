"""Render Jinja2 config templates for the OSPREY monitoring stack.

Generates OTEL Collector, Prometheus, and Grafana configs from the
``monitoring:`` section of config.yml into ``~/.osprey/monitoring/config/``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"
CONFIG_DIR = Path.home() / ".osprey" / "monitoring" / "config"
DATA_BASE = Path.home() / ".osprey" / "monitoring"


def generate_configs(monitoring_config: dict) -> dict[str, Path]:
    """Render all monitoring config templates.

    Args:
        monitoring_config: The ``monitoring:`` section from config.yml.

    Returns:
        Mapping of config name to rendered file path.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure data directories exist
    for subdir in (
        "prometheus/data",
        "grafana/data",
        "grafana/provisioning/datasources",
        "grafana/provisioning/dashboards",
        "grafana/dashboards",
        "logs",
    ):
        (DATA_BASE / subdir).mkdir(parents=True, exist_ok=True)

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        keep_trailing_newline=True,
    )

    # Build template context
    otel = monitoring_config.get("otel_collector", {})
    prom = monitoring_config.get("prometheus", {})
    grafana = monitoring_config.get("grafana", {})

    context = {
        "otel_host": otel.get("host", "127.0.0.1"),
        "otel_grpc_port": otel.get("grpc_port", 4317),
        "otel_http_port": otel.get("http_port", 4318),
        "otel_prometheus_port": otel.get("prometheus_port", 8889),
        "prometheus_host": prom.get("host", "127.0.0.1"),
        "prometheus_port": prom.get("port", 9090),
        "grafana_host": grafana.get("host", "127.0.0.1"),
        "grafana_port": grafana.get("port", 8091),
        "data_base": str(DATA_BASE),
    }

    rendered: dict[str, Path] = {}

    # OTEL Collector config
    otel_cfg = CONFIG_DIR / "otel-collector.yaml"
    otel_cfg.write_text(env.get_template("otel-collector.yaml.j2").render(context))
    rendered["otel_collector"] = otel_cfg

    # Prometheus config
    prom_cfg = CONFIG_DIR / "prometheus.yml"
    prom_cfg.write_text(env.get_template("prometheus.yml.j2").render(context))
    rendered["prometheus"] = prom_cfg

    # Grafana config
    grafana_cfg = CONFIG_DIR / "grafana.ini"
    grafana_cfg.write_text(env.get_template("grafana.ini.j2").render(context))
    rendered["grafana"] = grafana_cfg

    # Grafana provisioning — datasource
    ds_path = DATA_BASE / "grafana" / "provisioning" / "datasources" / "prometheus.yaml"
    ds_path.write_text(
        env.get_template("grafana/provisioning/datasources/prometheus.yaml.j2").render(context)
    )
    rendered["grafana_datasource"] = ds_path

    # Grafana provisioning — dashboard provider
    db_path = DATA_BASE / "grafana" / "provisioning" / "dashboards" / "osprey.yaml"
    db_path.write_text(
        env.get_template("grafana/provisioning/dashboards/osprey.yaml.j2").render(context)
    )
    rendered["grafana_dashboard_provider"] = db_path

    # Copy pre-built dashboard JSON
    src_dashboard = TEMPLATES_DIR / "grafana" / "dashboards" / "claude-code-overview.json"
    dst_dashboard = DATA_BASE / "grafana" / "dashboards" / "claude-code-overview.json"
    if src_dashboard.exists():
        dst_dashboard.write_text(src_dashboard.read_text())
        rendered["grafana_dashboard"] = dst_dashboard

    logger.info("Generated monitoring configs in %s", CONFIG_DIR)
    return rendered
