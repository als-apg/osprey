"""The hierarchical channel-finder index: the facility file as a tree of tree levels.

``data/channel_finder/hierarchical.json`` carries
``"schema": "osprey.facility.channel_finder/1"``, a ``hierarchy`` whose levels
are the facility's place level words, then ``class``, ``device`` and ``leaf``,
and a ``tree`` in which every channel sits at the same depth: an absent level
is the node ``-``. Each leaf's ``_channel_part`` is the full address, so the
hierarchical loader reads every channel back as exactly its address.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> Any:
    from osprey.services.channel_finder.databases.hierarchical import (
        HierarchicalChannelDatabase,
    )

    return HierarchicalChannelDatabase(str(path))


def _leaf(address: str) -> dict[str, Any]:
    return {"_channel_part": address}


# --- the loader --------------------------------------------------------------------


def test_a_bare_placeholder_pattern_loads_each_address_byte_for_byte(tmp_path: Path) -> None:
    addresses = (
        "SR04U___GDS1PS_AC00",
        "SR04U___GDS1PS_AM00",
        "SR01C___QF1____AM00",
        "BTS:HCM1:AC",
    )
    document = {
        "hierarchy": {
            "levels": [{"name": name, "type": "tree"} for name in ("class", "device", "leaf")],
            "naming_pattern": "{class}{device}{leaf}",
        },
        "tree": {
            "-": {
                "_channel_part": "",
                "_description": "no class",
                "-": {
                    "_channel_part": "",
                    "_description": "no device",
                    **{address: _leaf(address) for address in addresses},
                },
            }
        },
    }
    path = tmp_path / "hierarchical.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    database = _load(path)

    assert {row["channel"] for row in database.get_all_channels()} == set(addresses)
    for address in addresses:
        assert database.channel_map[address]["path"] == {"class": "", "device": "", "leaf": address}
