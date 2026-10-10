"""The hello-world tutorial names only channels its facility file serves.

The mock connector serves exactly the channels of the built facility file and
refuses any other address, so an address the tutorial asks the reader to read
or write must be a channel of hello-world's facility tree. The one address the
tutorial shows being refused is the exception, and must stay outside it.
"""

from __future__ import annotations

import re
from pathlib import Path

from osprey.facility.build import build_facility

REPO_ROOT = Path(__file__).resolve().parents[2]
TUTORIAL = REPO_ROOT / "docs/source/getting-started/hello-world-tutorial.rst"
HELLO_WORLD = REPO_ROOT / "src/osprey/templates/facilities/hello_world"

#: A colon-separated channel address, as the tutorial spells one.
ADDRESS = re.compile(r"\b[A-Z][A-Z0-9_]*(?::[A-Z0-9_]+)+\b")

#: The mock connector's refusal for an address outside the facility file.
REFUSAL = re.compile(rf"({ADDRESS.pattern}) is not in build/facility\.json")


def _channels() -> set[str]:
    document = build_facility(HELLO_WORLD, project_name="hello")
    return {channel["id"] for channel in document["channels"]}


def _tutorial() -> str:
    return TUTORIAL.read_text(encoding="utf-8")


def _refused_examples() -> set[str]:
    return set(REFUSAL.findall(_tutorial()))


def test_the_tutorial_shows_the_refusal_of_an_unknown_address() -> None:
    refused = _refused_examples()
    assert refused, "the tutorial shows no '<address> is not in build/facility.json' refusal"
    assert not refused & _channels()


def test_every_address_the_tutorial_names_is_a_hello_world_channel() -> None:
    named = set(ADDRESS.findall(_tutorial())) - _refused_examples()
    assert named, "the tutorial names no channel address"
    assert sorted(named - _channels()) == []
