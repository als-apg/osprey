"""Tests for the ``_inject_scan_panels`` build step in ``osprey.cli.build_cmd``.

Covers the three responsibilities of the scan-panels-injection step: copying
the bundled ``templates/services/scan_panels/`` compose template, writing the
``services.scan_panels`` config + registering it in ``deployed_services``
(additively), and registering the three ``web.panels.<id>`` entries
(``scan-plan``, ``scan-results``, ``scan-health``) with the sidecar-root
``url`` + per-panel ``path``/``label`` (+ ``health_endpoint`` for
``scan-health``) — mirroring ``_inject_dispatch``'s ``events`` panel
registration, including the "explicit override wins" precedence.
"""

from __future__ import annotations

from pathlib import Path

from ruamel.yaml import YAML

from osprey.cli.build_cmd import _inject_scan_panels
from osprey.cli.build_profile import ScanPanelsConfig


def _write_config(project_path: Path, *, extra: dict | None = None) -> None:
    """Write a minimal config.yml with a pre-existing deployed service."""
    yaml = YAML()
    config: dict = {
        "deployed_services": ["postgresql"],
        "services": {"postgresql": {}},
    }
    if extra:
        config.update(extra)
    with open(project_path / "config.yml", "w") as fh:
        yaml.dump(config, fh)


def _read_config(project_path: Path) -> dict:
    """Reload config.yml as a plain dict."""
    yaml = YAML()
    with open(project_path / "config.yml") as fh:
        return yaml.load(fh)


def test_inject_scan_panels_copies_template_dir(tmp_path: Path) -> None:
    """The bundled compose template dir is copied into services/scan_panels."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_scan_panels(ScanPanelsConfig(), project_path=project_path)

    dest = project_path / "services" / "scan_panels"
    assert (dest / "docker-compose.yml.j2").is_file()
    assert (dest / "Dockerfile").is_file()


def test_inject_scan_panels_writes_service_config(tmp_path: Path) -> None:
    """services.scan_panels is written with path + port, and deployed_services
    is additive — keeps existing services, appends scan_panels."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_scan_panels(ScanPanelsConfig(port=8095), project_path=project_path)

    config = _read_config(project_path)
    sp = config["services"]["scan_panels"]
    assert sp["path"] == "./services/scan_panels"
    assert sp["port"] == 8095
    assert "image" not in sp

    deployed = [str(s) for s in config["deployed_services"]]
    assert "postgresql" in deployed
    assert "scan_panels" in deployed


def test_inject_scan_panels_deployed_services_idempotent(tmp_path: Path) -> None:
    """Re-running the injector does not duplicate the deployed_services entry."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_scan_panels(ScanPanelsConfig(), project_path=project_path)
    _inject_scan_panels(ScanPanelsConfig(), project_path=project_path)

    config = _read_config(project_path)
    deployed = [str(s) for s in config["deployed_services"]]
    assert deployed.count("scan_panels") == 1


def test_inject_scan_panels_registers_three_web_panels(tmp_path: Path) -> None:
    """The three web.panels.<id> entries are registered with the sidecar-root
    url + per-panel path/label, and scan-health additionally gets a
    health_endpoint. The url points at the sidecar ROOT (not a panel-specific
    sub-path) — the panel's static mount is selected via `path`."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_scan_panels(ScanPanelsConfig(port=8095), project_path=project_path)

    panels = _read_config(project_path)["web"]["panels"]

    plan = panels["scan-plan"]
    assert plan["url"] == "${SCAN_PANELS_URL:-http://localhost:8095}"
    assert plan["path"] == "/plan/"
    assert plan["label"] == "SCAN PLAN"
    assert "health_endpoint" not in plan

    results = panels["scan-results"]
    assert results["url"] == "${SCAN_PANELS_URL:-http://localhost:8095}"
    assert results["path"] == "/results/"
    assert results["label"] == "SCAN RESULTS"
    assert "health_endpoint" not in results

    health = panels["scan-health"]
    assert health["url"] == "${SCAN_PANELS_URL:-http://localhost:8095}"
    assert health["path"] == "/health-panel/"
    assert health["label"] == "SCAN HEALTH"
    assert health["health_endpoint"] == "/health"


def test_inject_scan_panels_derives_url_from_custom_port(tmp_path: Path) -> None:
    """A non-default scan_panels.port is reflected in the derived panel urls."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_scan_panels(ScanPanelsConfig(port=9999), project_path=project_path)

    panels = _read_config(project_path)["web"]["panels"]
    assert panels["scan-plan"]["url"] == "${SCAN_PANELS_URL:-http://localhost:9999}"
    assert panels["scan-results"]["url"] == "${SCAN_PANELS_URL:-http://localhost:9999}"
    assert panels["scan-health"]["url"] == "${SCAN_PANELS_URL:-http://localhost:9999}"


def test_inject_scan_panels_explicit_url_override_wins(tmp_path: Path) -> None:
    """A pre-existing web.panels.<id>.url (e.g. a facility config override
    merged earlier in the build) is not clobbered by the derived default."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(
        project_path,
        extra={
            "web": {
                "panels": {
                    "scan-plan": {"url": "http://custom-host:1234"},
                }
            }
        },
    )

    _inject_scan_panels(ScanPanelsConfig(port=8095), project_path=project_path)

    panels = _read_config(project_path)["web"]["panels"]
    # Explicit override preserved.
    assert panels["scan-plan"]["url"] == "http://custom-host:1234"
    # But path/label are still filled in via setdefault (were absent before).
    assert panels["scan-plan"]["path"] == "/plan/"
    assert panels["scan-plan"]["label"] == "SCAN PLAN"
    # Untouched panels still get their derived default.
    assert panels["scan-results"]["url"] == "${SCAN_PANELS_URL:-http://localhost:8095}"


def test_inject_scan_panels_explicit_path_label_override_wins(tmp_path: Path) -> None:
    """A pre-existing path/label is not clobbered by the injector's setdefault."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(
        project_path,
        extra={
            "web": {
                "panels": {
                    "scan-health": {"path": "/custom-health/", "label": "CUSTOM"},
                }
            }
        },
    )

    _inject_scan_panels(ScanPanelsConfig(port=8095), project_path=project_path)

    health = _read_config(project_path)["web"]["panels"]["scan-health"]
    assert health["path"] == "/custom-health/"
    assert health["label"] == "CUSTOM"
    # Derived url and health_endpoint are still filled in.
    assert health["url"] == "${SCAN_PANELS_URL:-http://localhost:8095}"
    assert health["health_endpoint"] == "/health"


def test_inject_scan_panels_missing_config_yml_is_noop(tmp_path: Path) -> None:
    """Missing config.yml is a warned no-op (mirrors _inject_bluesky), not a crash."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    # No config.yml written.

    _inject_scan_panels(ScanPanelsConfig(), project_path=project_path)  # must not raise

    assert not (project_path / "config.yml").exists()
    # Template is still copied before the config.yml check.
    assert (project_path / "services" / "scan_panels" / "Dockerfile").is_file()
