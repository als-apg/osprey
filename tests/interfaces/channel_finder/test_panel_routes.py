"""The server's panel links spell a route exactly as the panel's route.js does.

The example links are the shared fixture route.test.mjs also reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osprey.services.pv_directory.facts import panel_link

_FIXTURE = Path(__file__).parent / "fixtures" / "panel_routes.json"
_ENTRIES = json.loads(_FIXTURE.read_text(encoding="utf-8"))
_NAMED = [e for e in _ENTRIES if e["route"]["view"] in ("pv", "ioc")]


@pytest.mark.parametrize("entry", _NAMED, ids=[e["hash"] for e in _NAMED])
def test_panel_link_spells_every_fixture_hash(entry: dict) -> None:
    route = entry["route"]
    link = panel_link(route["view"], route["params"]["name"])
    assert link["panel"] == "/panel/channel-finder/" + entry["hash"]


@pytest.mark.parametrize("char", list("!*'()/%: "))
def test_the_fixture_names_carry_every_hard_character(char: str) -> None:
    assert any(char in e["route"]["params"]["name"] for e in _NAMED)
