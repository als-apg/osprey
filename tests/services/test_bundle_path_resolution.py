"""``facility_knowledge.bundle_path`` resolves identically at all three sites.

The key is read by three consumers — the ``osprey_facility_knowledge`` MCP
server, the ``osprey knowledge`` CLI, and the OKF knowledge panel. They used to
normalise it three different ways, so the shipped relative value
(``data/facility_knowledge``) pointed the CLI at ``<cwd>/data/facility_knowledge``
while the other two opened ``<project>/data/facility_knowledge``.

Every test here runs from a foreign CWD — that is the condition under which the
old divergence was observable at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

CONFIG_KEY = "facility_knowledge.bundle_path"

#: The relative value shipped in control_assistant/config.yml.j2.
SHIPPED_RELATIVE = "data/facility_knowledge"


def _write_concept(bundle_root: Path, concept_id: str) -> None:
    """Create a minimal valid OKF concept document inside *bundle_root*."""
    path = bundle_root / f"{concept_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\ntype: concept\ntitle: X\ndescription: d\n---\n\nbody\n",
        encoding="utf-8",
    )


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A project with config.yml + a bundle, entered from an unrelated CWD.

    Yields ``(project_dir, bundle_dir)``. ``OSPREY_CONFIG`` points at the
    project's config.yml — the documented deployment mechanism — and the process
    CWD is a sibling directory that contains no bundle.
    """
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "config.yml").write_text("", encoding="utf-8")

    bundle_dir = project_dir / SHIPPED_RELATIVE
    _write_concept(bundle_dir, "devices/bpm")

    foreign_cwd = tmp_path / "elsewhere"
    foreign_cwd.mkdir()
    monkeypatch.chdir(foreign_cwd)
    monkeypatch.setenv("OSPREY_CONFIG", str(project_dir / "config.yml"))

    return project_dir, bundle_dir


# ---------------------------------------------------------------------------
# One call per consumer, each through that consumer's own public entry point
# ---------------------------------------------------------------------------


def _mcp_server_resolution(raw: str, config_dir: Path) -> Path:
    """Resolve *raw* the way the facility-knowledge MCP server does at startup."""
    from osprey.mcp_server.facility_knowledge.server import _resolve_bundle_path

    return _resolve_bundle_path({"facility_knowledge": {"bundle_path": raw}}, config_dir)


def _cli_resolution(raw: str, monkeypatch) -> Path:
    """Resolve *raw* the way ``osprey knowledge <cmd>`` does with no BUNDLE arg."""
    import osprey.utils.config as config_module
    from osprey.cli.knowledge_cmd import _resolve_bundle

    monkeypatch.setattr(
        config_module,
        "get_config_value",
        lambda key, default=None: raw if key == CONFIG_KEY else default,
    )
    return _resolve_bundle(None)


def _panel_resolution(raw: str) -> Path:
    """Resolve *raw* the way the OKF panel does — via the real app factory.

    Returns the root of the bundle the panel actually opened, so a resolution
    that silently drops the panel into guarded mode fails loudly here.
    """
    from osprey.interfaces.okf_panel.app import create_app

    app = create_app(raw)
    assert app.state.bundle is not None, "panel fell into guarded mode — path did not resolve"
    return Path(app.state.bundle.root)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_shipped_relative_path_agrees_across_all_three_sites(project, monkeypatch):
    """The shipped relative value resolves to the project bundle everywhere."""
    project_dir, bundle_dir = project

    mcp = _mcp_server_resolution(SHIPPED_RELATIVE, project_dir)
    cli = _cli_resolution(SHIPPED_RELATIVE, monkeypatch)
    panel = _panel_resolution(SHIPPED_RELATIVE)

    assert mcp == cli == panel == bundle_dir.resolve()
    assert Path.cwd() not in bundle_dir.parents, "fixture must place the CWD outside the project"


def test_cli_does_not_resolve_relative_to_the_cwd(project, monkeypatch):
    """Regression: the CLI must not be a CWD-relative outlier.

    A bundle sitting at ``<cwd>/data/facility_knowledge`` must NOT win over the
    one the config points at — that decoy is exactly what a CWD-relative
    resolver opens.
    """
    _project_dir, bundle_dir = project

    decoy = Path.cwd() / SHIPPED_RELATIVE
    _write_concept(decoy, "devices/decoy")

    resolved = _cli_resolution(SHIPPED_RELATIVE, monkeypatch)

    assert resolved == bundle_dir.resolve()
    assert resolved != decoy.resolve()


def test_tilde_path_is_expanded_at_all_three_sites(project, monkeypatch, tmp_path):
    """``~/...`` is expanded rather than treated as a literal directory name."""
    project_dir, _bundle_dir = project

    home = tmp_path / "home"
    _write_concept(home / "kb", "refs/glossary")
    monkeypatch.setenv("HOME", str(home))

    raw = "~/kb"
    mcp = _mcp_server_resolution(raw, project_dir)
    cli = _cli_resolution(raw, monkeypatch)
    panel = _panel_resolution(raw)

    assert mcp == cli == panel == home / "kb"


def test_absolute_path_passes_through_at_all_three_sites(project, monkeypatch, tmp_path):
    """An absolute value is used as given, ignoring both CWD and config dir."""
    project_dir, _bundle_dir = project

    elsewhere = tmp_path / "shared-bundle"
    _write_concept(elsewhere, "devices/quad")

    raw = str(elsewhere)
    mcp = _mcp_server_resolution(raw, project_dir)
    cli = _cli_resolution(raw, monkeypatch)
    panel = _panel_resolution(raw)

    assert mcp == cli == panel == elsewhere


def test_resolver_is_the_single_rule():
    """All three sites route through the shared helper, not private copies."""
    import osprey.cli.knowledge_cmd as knowledge_cmd
    import osprey.interfaces.okf_panel.app as panel_app
    import osprey.mcp_server.facility_knowledge.server as mcp_server

    shared_import = "from osprey.services.facility_knowledge.bundle_path import resolve_bundle_path"

    assert not hasattr(panel_app, "_normalize_bundle_path")
    for module in (knowledge_cmd, panel_app, mcp_server):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert shared_import in source, f"{module.__name__} does not use the shared resolver"
