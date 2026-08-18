"""The BLUESKY panel's registration in a built project.

``_inject_bluesky_panels`` writes the ``web.panels.bluesky`` entry into a built
``config.yml``, pointing the web-terminal proxy at the sidecar and selecting
the panel's static mount via ``path``. This module covers that registration
end to end: the entry a fresh build writes, the profile validation that lets
``bluesky`` stay url-less (its URL is derived post-build from the sidecar
port), and that the registered path actually serves the panel's HTML.

The rest of ``_inject_bluesky_panels`` (template copy, service config,
deployed_services, url derivation, override precedence) is covered by
``test_inject_bluesky_panels.py``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
from ruamel.yaml import YAML

from osprey.cli.build_injectors import _inject_bluesky_panels
from osprey.cli.build_profile import BlueskyPanelsConfig, BuildProfile


def _write_config(project_path: Path, *, panels: dict | None = None) -> None:
    """Write a minimal config.yml, optionally pre-seeded with web.panels entries."""
    yaml = YAML()
    config: dict = {"deployed_services": [], "services": {}}
    if panels is not None:
        config["web"] = {"panels": panels}
    with open(project_path / "config.yml", "w") as fh:
        yaml.dump(config, fh)


def _read_panels(project_path: Path) -> dict:
    yaml = YAML()
    with open(project_path / "config.yml") as fh:
        return yaml.load(fh)["web"]["panels"]


@pytest.fixture
def project_path(tmp_path: Path) -> Path:
    path = tmp_path / "project"
    path.mkdir()
    return path


def test_injector_registers_the_bluesky_panel(project_path: Path) -> None:
    """The entry carries the derived url, the panel's mount path, and its label."""
    _write_config(project_path)

    _inject_bluesky_panels(BlueskyPanelsConfig(port=8095), project_path=project_path)

    bluesky = _read_panels(project_path)["bluesky"]
    assert bluesky["url"] == "${BLUESKY_PANELS_URL:-http://localhost:8095}"
    assert bluesky["path"] == "/bluesky/"
    assert bluesky["label"] == "BLUESKY"


def test_injector_registers_exactly_one_panel(project_path: Path) -> None:
    """One sidecar, one tab: a fresh build writes ``bluesky`` and nothing else.

    A second entry here would put two rail entries in front of the same
    bundle on every new project and on every rebuild.
    """
    _write_config(project_path)

    _inject_bluesky_panels(BlueskyPanelsConfig(), project_path=project_path)

    assert set(_read_panels(project_path)) == {"bluesky"}


def test_profile_listing_bluesky_validates_without_warning(tmp_path: Path) -> None:
    """``bluesky`` is url-less-legal (its URL is derived post-build) — and silent."""
    profile = BuildProfile(
        name="modern",
        web_panels=["bluesky"],
        bluesky_panels=BlueskyPanelsConfig(),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        profile.validate(tmp_path)


def test_the_registered_path_actually_serves_the_panel(project_path: Path) -> None:
    """The operator-visible half: the registered path returns the panel's HTML.

    A mount can be registered against an empty directory and still 404 (see
    ``tests/interfaces/bluesky_panels/test_health.py``), which would leave this
    config.yml pointing a tab at nothing.
    """
    from fastapi.testclient import TestClient

    from osprey.interfaces.bluesky_panels.app import app

    _write_config(project_path)
    _inject_bluesky_panels(BlueskyPanelsConfig(), project_path=project_path)

    panels = _read_panels(project_path)

    with TestClient(app) as client:
        response = client.get(panels["bluesky"]["path"])
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
