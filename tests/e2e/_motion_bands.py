"""The settle band of each reading a deployed facility serves.

A served reading is the value it holds plus the motion its seed declares; this
reads those seeds from the simulator view the build rendered -- the bytes the
container serves -- and maps each address through the product's own rule,
:func:`osprey.facility.motion.settle_band`. A reading whose seed declares no
motion gets 0.0.

Imports nothing that serves Channel Access, so any deploy-backed suite may call
it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path


def served_settle_bands(repo: Path, addresses: Iterable[str]) -> dict[str, float]:
    """Each address's settle band, from the seeds the deployment at ``repo`` serves.

    Args:
        repo: The deployment repo, whose render holds the simulator view.
        addresses: The served addresses to look up.

    Returns:
        ``{address: band}``, 0.0 for an address whose seed declares no motion.
    """
    from osprey.facility.motion import settle_band
    from osprey_connectors.simulation.view import SEEDS_FILE, SimulatorView

    seeds_json = SimulatorView.path_for_project(repo) / SEEDS_FILE
    assert seeds_json.is_file(), f"the deployment at {repo} rendered no seeds at {seeds_json}"
    seeds = json.loads(seeds_json.read_text(encoding="utf-8"))["seeds"]
    return {address: settle_band(seeds.get(address)) for address in addresses}
