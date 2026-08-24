"""Startup behaviour of the web app under the read-only render-zone marker.

In a privilege-split container the render zone (``config.yml`` + ``.claude/``)
is root-owned and the root entrypoint performs the artifact regen and the
scaffold restore before dropping to the non-root app user. The server then runs
as that user and must not attempt either write. ``OSPREY_RENDER_ZONE_READONLY=1``
is the marker that says so.

Two halves, both driving the real app factory through the ``TestClient``
lifespan:

* marker set — no write is attempted (no ``regen_if_drift``, which writes even
  on its no-op path, and no scaffold restore), a warning names what would have
  changed, and ``app.state.render_zone_readonly`` is True;
* marker absent — today's behaviour is pinned byte-for-byte.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from osprey.cli.templates.manager import TemplateManager
from osprey.interfaces.web_terminal.app import create_app


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A minimal project dir with the ambient config env cleared."""
    (tmp_path / "_agent_data").mkdir(exist_ok=True)
    monkeypatch.delenv("CONFIG_FILE", raising=False)
    monkeypatch.delenv("OSPREY_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def startup_spies(monkeypatch):
    """Record every render-zone write the lifespan could attempt.

    ``restore_scaffold_bodies`` is imported *inside* the lifespan, so the patch
    has to land on the source module rather than on ``app``'s namespace.
    """
    calls: dict[str, list] = {"regen_if_drift": [], "dry_run": [], "restore": []}

    monkeypatch.setattr(
        TemplateManager,
        "regen_if_drift",
        lambda self, pd: calls["regen_if_drift"].append(Path(pd)) or [],
    )
    monkeypatch.setattr(
        TemplateManager,
        "regenerate_claude_code",
        lambda self, pd, dry_run=False, **kw: (
            calls["dry_run"].append((Path(pd), dry_run))
            or {"changed": ["settings.json", ".mcp.json"], "unchanged": []}
        ),
    )
    monkeypatch.setattr(
        "osprey.interfaces.web_terminal.scaffold_gallery_service.restore_scaffold_bodies",
        lambda pd: calls["restore"].append(Path(pd)),
    )
    return calls


def _run_lifespan(project):
    with patch(
        "osprey.interfaces.web_terminal.app._load_web_config",
        return_value={"watch_dir": str(project / "_agent_data")},
    ):
        app = create_app(shell_command="echo", project_dir=str(project))
        with TestClient(app):
            pass
    return app


def test_render_zone_readonly_marker_attempts_no_render_zone_write(
    project, startup_spies, monkeypatch, caplog
):
    """Marker set ⇒ dry-run preview only, no restore, no drift-stamping regen."""
    monkeypatch.setenv("OSPREY_RENDER_ZONE_READONLY", "1")

    with caplog.at_level(logging.WARNING):
        app = _run_lifespan(project)

    assert app.state.render_zone_readonly is True
    # regen_if_drift writes even when nothing drifted (it stamps settings.json
    # with os.utime), so it must not be reached at all.
    assert startup_spies["regen_if_drift"] == []
    assert startup_spies["restore"] == []
    assert startup_spies["dry_run"] == [(project.resolve(), True)]

    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING
        and "OSPREY_RENDER_ZONE_READONLY" in record.getMessage()
    ]
    assert warnings, "read-only render zone must warn that regen was skipped"
    # The warning has to name what is out of sync — nothing else will report it.
    assert "settings.json" in warnings[0] and ".mcp.json" in warnings[0]


def test_render_zone_readonly_marker_fails_open_when_preview_raises(
    project, startup_spies, monkeypatch
):
    """A failing dry-run preview must not stop the server coming up."""
    monkeypatch.setenv("OSPREY_RENDER_ZONE_READONLY", "1")

    def boom(self, pd, dry_run=False, **kw):
        raise RuntimeError("preview exploded")

    monkeypatch.setattr(TemplateManager, "regenerate_claude_code", boom)

    app = _run_lifespan(project)

    assert app.state.render_zone_readonly is True
    assert startup_spies["regen_if_drift"] == []


@pytest.mark.parametrize("marker", ["", "0", "true", "yes"])
def test_render_zone_readonly_only_the_exact_marker_value_counts(
    project, startup_spies, monkeypatch, marker
):
    """Only ``1`` flips the posture; anything else keeps the writing behaviour."""
    monkeypatch.setenv("OSPREY_RENDER_ZONE_READONLY", marker)

    app = _run_lifespan(project)

    assert app.state.render_zone_readonly is False
    assert startup_spies["regen_if_drift"] == [project.resolve()]
    assert startup_spies["dry_run"] == []


def test_marker_absent_regenerates_and_restores_as_before(project, startup_spies, monkeypatch):
    """No marker ⇒ today's startup behaviour, unchanged."""
    monkeypatch.delenv("OSPREY_RENDER_ZONE_READONLY", raising=False)

    app = _run_lifespan(project)

    assert app.state.render_zone_readonly is False
    assert startup_spies["regen_if_drift"] == [project.resolve()]
    assert startup_spies["restore"] == [Path(app.state.project_cwd)]
    # No dry-run preview is taken on the writable path.
    assert startup_spies["dry_run"] == []


def test_marker_absent_state_flag_is_readable_via_getattr(project, startup_spies, monkeypatch):
    """Downstream routes read the flag defensively; it is always present."""
    monkeypatch.delenv("OSPREY_RENDER_ZONE_READONLY", raising=False)

    app = _run_lifespan(project)

    assert getattr(app.state, "render_zone_readonly", False) is False
