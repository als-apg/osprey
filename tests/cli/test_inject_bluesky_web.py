"""Tests for the ``_inject_bluesky_web`` build step in ``osprey.cli.build_cmd``.

Covers the three responsibilities of the bluesky-web-injection step: copying
the bundled ``templates/services/bluesky_web/`` compose template, writing the
``services.bluesky_web`` config + registering it in ``deployed_services``
(additively), and registering the ``web.panels.bluesky`` entry with the
sidecar-root ``url`` + per-panel ``path``/``label`` — mirroring
``_inject_dispatch``'s ``events`` panel registration, including the
"explicit override wins" precedence.

WHICH panel ids get registered is ``test_panel_registration.py``'s subject —
that is where a change to the id set is meant to fail first.
"""

from __future__ import annotations

from pathlib import Path

from ruamel.yaml import YAML

from osprey.cli.build_cmd import _inject_bluesky_web
from osprey.cli.build_profile import BlueskyWebConfig
from osprey.port_layout import default_port

#: The sidecar's layout slot — what the injection writes when nothing moves it.
_WEB_PORT = default_port("bluesky_web")


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


def test_inject_bluesky_web_copies_template_dir(tmp_path: Path) -> None:
    """The bundled compose template dir is copied into services/bluesky_web."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_bluesky_web(BlueskyWebConfig(), project_path=project_path)

    dest = project_path / "services" / "bluesky_web"
    assert (dest / "docker-compose.yml.j2").is_file()
    assert (dest / "Dockerfile").is_file()


def test_inject_bluesky_web_writes_service_config(tmp_path: Path) -> None:
    """services.bluesky_web is written with path + port, and deployed_services
    is additive — keeps existing services, appends bluesky_web."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_bluesky_web(BlueskyWebConfig(port=_WEB_PORT), project_path=project_path)

    config = _read_config(project_path)
    sp = config["services"]["bluesky_web"]
    assert sp["path"] == "./services/bluesky_web"
    assert sp["port"] == _WEB_PORT
    assert "image" not in sp

    deployed = [str(s) for s in config["deployed_services"]]
    assert "postgresql" in deployed
    assert "bluesky_web" in deployed


def test_inject_bluesky_web_deployed_services_idempotent(tmp_path: Path) -> None:
    """Re-running the injector does not duplicate the deployed_services entry."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_bluesky_web(BlueskyWebConfig(), project_path=project_path)
    _inject_bluesky_web(BlueskyWebConfig(), project_path=project_path)

    config = _read_config(project_path)
    deployed = [str(s) for s in config["deployed_services"]]
    assert deployed.count("bluesky_web") == 1


def test_inject_bluesky_web_registers_the_web_panels(tmp_path: Path) -> None:
    """The web.panels.bluesky entry is registered with the sidecar-root url +
    its path/label. The url points at the sidecar ROOT (not a panel-specific
    sub-path) — the panel's static mount is selected via `path`.

    The registered id set is covered by ``test_panel_registration.py``; this
    file's subject is the injector's compose/config wiring."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_bluesky_web(BlueskyWebConfig(port=_WEB_PORT), project_path=project_path)

    panels = _read_config(project_path)["web"]["panels"]

    bluesky = panels["bluesky"]
    assert bluesky["url"] == f"${{BLUESKY_WEB_URL:-http://localhost:{_WEB_PORT}}}"
    assert bluesky["path"] == "/bluesky/"
    assert bluesky["label"] == "BLUESKY"
    assert "health_endpoint" not in bluesky


def test_inject_bluesky_web_derives_url_from_custom_port(tmp_path: Path) -> None:
    """A non-default bluesky_web.port is reflected in the derived panel urls."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(project_path)

    _inject_bluesky_web(BlueskyWebConfig(port=9999), project_path=project_path)

    panels = _read_config(project_path)["web"]["panels"]
    assert panels["bluesky"]["url"] == "${BLUESKY_WEB_URL:-http://localhost:9999}"


def test_inject_bluesky_web_explicit_url_override_wins(tmp_path: Path) -> None:
    """A pre-existing web.panels.<id>.url (e.g. a facility config override
    merged earlier in the build) is not clobbered by the derived default."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(
        project_path,
        extra={
            "web": {
                "panels": {
                    "bluesky": {"url": "http://custom-host:1234"},
                }
            }
        },
    )

    _inject_bluesky_web(BlueskyWebConfig(port=_WEB_PORT), project_path=project_path)

    bluesky = _read_config(project_path)["web"]["panels"]["bluesky"]
    # Explicit override preserved.
    assert bluesky["url"] == "http://custom-host:1234"
    # But path/label are still filled in via setdefault (were absent before).
    assert bluesky["path"] == "/bluesky/"
    assert bluesky["label"] == "BLUESKY"


def test_inject_bluesky_web_explicit_path_label_override_wins(tmp_path: Path) -> None:
    """A pre-existing path/label is not clobbered by the injector's setdefault."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    _write_config(
        project_path,
        extra={
            "web": {
                "panels": {
                    "bluesky": {"path": "/custom-results/", "label": "CUSTOM"},
                }
            }
        },
    )

    _inject_bluesky_web(BlueskyWebConfig(port=_WEB_PORT), project_path=project_path)

    bluesky = _read_config(project_path)["web"]["panels"]["bluesky"]
    assert bluesky["path"] == "/custom-results/"
    assert bluesky["label"] == "CUSTOM"
    # Derived url is still filled in.
    assert bluesky["url"] == f"${{BLUESKY_WEB_URL:-http://localhost:{_WEB_PORT}}}"


def test_inject_bluesky_web_missing_config_yml_is_noop(tmp_path: Path) -> None:
    """Missing config.yml is a warned no-op (mirrors _inject_bluesky), not a crash."""
    project_path = tmp_path / "project"
    project_path.mkdir()
    # No config.yml written.

    _inject_bluesky_web(BlueskyWebConfig(), project_path=project_path)  # must not raise

    assert not (project_path / "config.yml").exists()
    # Template is still copied before the config.yml check.
    assert (project_path / "services" / "bluesky_web" / "Dockerfile").is_file()
