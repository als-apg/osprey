"""The demo template's facility collection passes CANOPY's check.

The template ships one facility collection (``web/canopy/collections/
demo.json``), OSPREY's own content: ``default: true``, the
``facility/overview`` and ``sr/orbit-feedback`` screens from the template's own
library, and three named tools (Orbit, RF thermal, Septum pulse) that open the
preset's trace, correlation and waveform default views.

The image build judges that directory against the library beside it, from
the deployment directory it has just copied in
(:func:`osprey.integrations.canopy.image_check.collections_command`:
``python -I -m canopy_mcp.collections_cli web/canopy/collections --library
web/canopy/library``), so a fresh project must pass that check without anyone
touching anything. This module lays the template's collections and library at
their declared source paths, as a deployment holds them, and runs the command
the image check composes, in its environment, against the real ``canopy_mcp``
under this interpreter. These tests reach CANOPY only through a subprocess, so
they carry the ``canopy`` mark explicitly.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from osprey import repo_layout
from osprey.cli.templates.manager import TemplateManager
from osprey.integrations.canopy import image_check

#: The declared source paths of the facility collections and the screen library.
SOURCE_COLLECTIONS = Path(repo_layout.child_display_path("web", "canopy-collections"))
SOURCE_LIBRARY = Path(repo_layout.child_display_path("web", "canopy-library"))
TEMPLATE_COLLECTION = "demo.json"


@pytest.fixture(scope="module")
def template_app() -> Path:
    root = Path(TemplateManager().template_root) / "apps" / "demo"
    assert (root / SOURCE_COLLECTIONS).is_dir(), root
    return root


@pytest.fixture
def deployment(template_app: Path, tmp_path: Path) -> Path:
    """A deployment directory holding the template's collections and library as sources."""
    deployment = tmp_path / "deployment"
    for source in (SOURCE_COLLECTIONS, SOURCE_LIBRARY):
        shutil.copytree(template_app / source, deployment / source)
    assert (deployment / SOURCE_COLLECTIONS / TEMPLATE_COLLECTION).is_file()
    return deployment


def _check(deployment: Path) -> subprocess.CompletedProcess[str]:
    """CANOPY's collection check, run from the deployment as the image build runs it."""
    return subprocess.run(
        image_check.collections_command(),
        cwd=deployment,
        env=image_check.check_environment(),
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
    def test_canopys_check_passes_on_the_scaffolded_template(self, deployment):
        result = _check(deployment)
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout == "collections: 1 checked, 0 refused\n"
        assert result.stderr == ""

    @pytest.mark.canopy
    def test_canopys_check_refuses_a_member_the_library_does_not_list(self, deployment):
        """The check is live: it judges screen members against the shipped library."""
        path = deployment / SOURCE_COLLECTIONS / TEMPLATE_COLLECTION
        doc = json.loads(path.read_bytes())
        doc["members"][0] = {"screen": "library/sr/not-in-the-library"}
        path.write_text(json.dumps(doc), encoding="utf-8")
        result = _check(deployment)
        assert result.returncode == 3, result.stdout + result.stderr
        assert result.stdout == "collections: 1 checked, 1 refused\n"
        assert result.stderr.startswith(f"canopy_mcp.collections: refused: {TEMPLATE_COLLECTION}: ")
        assert "is not a screen the library lists" in result.stderr
