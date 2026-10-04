"""Stage the scenarios of a render's simulator view, as the build writes them."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from osprey.facility.sources import read_yaml
from osprey.facility.views import view_bytes
from osprey.facility.views.simulator import SCENARIOS_FILE, SCENARIOS_SCHEMA


def write_scenarios_view(render: Path, scenarios: Mapping[str, Mapping[str, Any]]) -> Path:
    """Write ``<render>/data/simulator/scenarios.json`` listing ``scenarios``.

    Args:
        render: The render's root.
        scenarios: Scenario name -> its blocks (``description``, ``archiver``,
            ``logbook``, ...), carried verbatim.

    Returns:
        The file written.
    """
    document = {
        "schema": SCENARIOS_SCHEMA,
        "scenarios": [{"name": name, **dict(scenarios[name])} for name in sorted(scenarios)],
    }
    path = render / "data" / "simulator" / SCENARIOS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(view_bytes(document))
    return path


def facility_scenarios(directory: Path) -> dict[str, dict[str, Any]]:
    """The scenarios of a facility's ``scenarios/`` directory, by file stem."""
    return {
        path.stem: read_yaml(path.read_text(encoding="utf-8")) or {}
        for path in sorted(directory.glob("*.yaml"))
    }
