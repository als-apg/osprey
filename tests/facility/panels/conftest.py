"""The panels fixture, built once per session; see ``_fixture.py``."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.facility.panels._fixture import PANELS_DIR, write_fixture_facility


@pytest.fixture(scope="session")
def panel_facility(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The fixture's built facility file."""
    return write_fixture_facility(tmp_path_factory.mktemp("panels") / "facility.json")


@pytest.fixture(scope="session")
def panel_tables() -> Path:
    """The fixture's rule tables directory."""
    return PANELS_DIR


@pytest.fixture(scope="session")
def panel_graph(panel_facility: Path, panel_tables: Path):
    """The graph over the fixture; read-only, so one per session."""
    from osprey.facility.panels import load_graph

    return load_graph(panel_facility, panel_tables)


@pytest.fixture(scope="session")
def ring_graph(tmp_path_factory: pytest.TempPathFactory, panel_tables: Path):
    """The fixture graph with its ring a periodic model ``XRM`` 14 m long: every
    XR01/XR02 device belongs to it, so ``XR``'s roll-up draws its sectors as arcs."""
    from osprey.facility.panels import load_graph

    def change(document) -> None:
        document["models"] = [
            *document.get("models", []),
            {"name": "XRM", "engine": "none", "length": 14.0, "periodic": True},
        ]
        for device in document["devices"]:
            if device["id"].startswith(("XR01:", "XR02:")):
                device["model"] = "XRM"

    facility = write_fixture_facility(tmp_path_factory.mktemp("ring") / "facility.json", change)
    return load_graph(facility, panel_tables)
