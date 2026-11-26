"""The demo template's facility collection passes CANOPY's check.

The template ships one facility collection (``web/canopy/collections/
demo.json``), OSPREY's own content: ``default: true``, the
``facility/overview`` and ``sr/orbit-feedback`` screens from the template's own
library, and three named tools (Orbit, RF thermal, Septum pulse) that open the
preset's trace, correlation and waveform default views.

The image build judges that directory against the library beside it
(``python -I -m canopy_mcp.collections_cli data/canopy/collections
--library data/canopy/library``), so a fresh project must pass that check
without anyone touching anything. This module renders the template's declared
sources the way a build assembles a project's data tree (``copy_template_data``) and
runs the same command against the real ``canopy_mcp`` under this interpreter.
These tests reach CANOPY only through a subprocess, so they carry the
``canopy`` mark explicitly.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from osprey import repo_layout
from osprey.cli.templates.manager import TemplateManager
from osprey.cli.templates.scaffolding import copy_template_data
from osprey.integrations.canopy import LIBRARY_SUBPATH

COLLECTIONS_SUBPATH = Path("data/canopy/collections")
TEMPLATE_COLLECTION = "demo.json"


def _source_of(render: Path) -> Path:
    """Where the template keeps the declared source the render lands at *render*."""
    return next(
        Path(source.source_rel)
        for source in repo_layout.rendered_sources()
        if Path(source.render_rel) == render
    )


SOURCE_COLLECTIONS = _source_of(COLLECTIONS_SUBPATH)
SOURCE_LIBRARY = Path(LIBRARY_SUBPATH)
LIBRARY_RENDER = next(
    Path(source.render_rel)
    for source in repo_layout.rendered_sources()
    if Path(source.source_rel) == SOURCE_LIBRARY
)


@pytest.fixture(scope="module")
def template_app() -> Path:
    root = Path(TemplateManager().template_root) / "apps" / "demo"
    assert (root / SOURCE_COLLECTIONS).is_dir(), root
    return root


@pytest.fixture
def project(template_app: Path, tmp_path: Path) -> Path:
    """A project data tree rendered from the template's declared sources, as a build lays it out."""
    project = tmp_path / "project"
    project.mkdir()
    copy_template_data(
        Path(TemplateManager().template_root),
        project,
        "unused",
        "demo",
        {},
        repo_root=template_app,
    )
    assert (project / COLLECTIONS_SUBPATH / TEMPLATE_COLLECTION).is_file()
    return project


def _check(project: Path) -> subprocess.CompletedProcess[str]:
    """CANOPY's collection check, run from the project as the image build runs it."""
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "canopy_mcp.collections_cli",
            str(COLLECTIONS_SUBPATH),
            "--library",
            str(LIBRARY_RENDER),
        ],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=120,
    )


class TestWhatTheTemplateShips:
    def test_one_collection_and_nothing_else(self, template_app):
        directory = template_app / SOURCE_COLLECTIONS
        assert sorted(p.name for p in directory.iterdir()) == [TEMPLATE_COLLECTION]

    def test_the_collection_names_its_screens_and_tools(self, template_app):
        doc = json.loads((template_app / SOURCE_COLLECTIONS / TEMPLATE_COLLECTION).read_bytes())
        assert doc["name"] == "Demo"
        assert doc["default"] is True
        assert doc["description"] == (
            "The facility overview and orbit feedback screens with Orbit, RF thermal "
            "and Septum pulse each opened on its own view."
        )
        # Each tool member carries its query: a query-less member keys as
        # ``tool:<id>``, the bare launcher's key, and the two would share a tile.
        assert doc["members"] == [
            {"screen": "library/facility/overview"},
            {"screen": "library/sr/orbit-feedback"},
            {
                "tool": "trace",
                "name": "Orbit",
                "query": "pv=SR:DIAG:BPM:03:POSITION:X&pv=SR:DIAG:BPM:03:POSITION:Y&range=-1h",
            },
            {
                "tool": "correlation",
                "name": "RF thermal",
                "query": (
                    "x=SR:RF:CAVITY:01:TEMPERATURE:RB&y=SR:RF:CAVITY:01:POWER:REV"
                    "&dt=1&n=1000&lookback=1h&view=scatter"
                ),
            },
            {
                "tool": "waveform",
                "name": "Septum pulse",
                "query": "pv=SR:MAG:SEPTUM:01:CURRENT:WAVEFORM&persist=10",
            },
        ]

    def test_every_screen_member_is_a_screen_the_template_library_holds(self, template_app):
        doc = json.loads((template_app / SOURCE_COLLECTIONS / TEMPLATE_COLLECTION).read_bytes())
        for member in doc["members"]:
            if "screen" in member:
                screen_id = member["screen"].removeprefix("library/")
                path = template_app / SOURCE_LIBRARY / "screens" / f"{screen_id}.screen.json"
                assert path.is_file(), path


class TestTheImageCheck:
    @pytest.mark.canopy
    def test_canopys_check_passes_on_the_scaffolded_template(self, project):
        result = _check(project)
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout == "collections: 1 checked, 0 refused\n"
        assert result.stderr == ""

    @pytest.mark.canopy
    def test_canopys_check_refuses_a_member_the_library_does_not_list(self, project):
        """The check is live: it judges screen members against the shipped library."""
        path = project / COLLECTIONS_SUBPATH / TEMPLATE_COLLECTION
        doc = json.loads(path.read_bytes())
        doc["members"][0] = {"screen": "library/sr/not-in-the-library"}
        path.write_text(json.dumps(doc), encoding="utf-8")
        result = _check(project)
        assert result.returncode == 3, result.stdout + result.stderr
        assert result.stdout == "collections: 1 checked, 1 refused\n"
        assert result.stderr.startswith(f"canopy_mcp.collections: refused: {TEMPLATE_COLLECTION}: ")
        assert "is not a screen the library lists" in result.stderr
