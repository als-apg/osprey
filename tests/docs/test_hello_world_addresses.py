"""The getting-started tutorials name only channels their preset's facility serves.

The connector serves exactly the channels of the built facility file and
refuses any other address, so an address a tutorial asks the reader to read or
write must be a channel of that preset's facility tree: hello-world's own small
facility, or the example facility the control-assistant preset ships. A family
written with a ``*`` segment (``SR:MAG:HCM:*:CURRENT:SP``) must match at least
one channel. The one address the hello-world tutorial shows being refused is
the exception, and must stay outside it.

A file a tutorial pulls in with ``literalinclude`` is read as part of the page,
so the shipped ``limits.yaml`` the hello-world page shows is held to the same
rule.
"""

from __future__ import annotations

import re
from fnmatch import fnmatchcase
from functools import cache
from pathlib import Path

import pytest

from osprey.facility.build import build_facility

REPO_ROOT = Path(__file__).resolve().parents[2]
GETTING_STARTED = REPO_ROOT / "docs/source/getting-started"
FACILITIES = REPO_ROOT / "src/osprey/templates/facilities"

#: Each tutorial page, and the facility tree its preset ships.
TUTORIALS = {
    "hello-world-tutorial.rst": "hello_world",
    "control-assistant.rst": "example",
}

#: A colon-separated channel address, as the tutorials spell one; a ``*``
#: segment names a family.
ADDRESS = re.compile(r"(?<![\w*])[A-Z][A-Z0-9_]*(?::(?:[A-Z0-9_]+|\*))+(?![\w*])")

#: The mock connector's refusal for an address outside the facility file.
REFUSAL = re.compile(rf"({ADDRESS.pattern}) is not a channel of this facility")

LITERALINCLUDE = re.compile(r"^\.\. literalinclude:: (\S+)", re.MULTILINE)


@cache
def _channels(facility: str) -> frozenset[str]:
    document = build_facility(FACILITIES / facility, project_name=facility)
    return frozenset(channel["id"] for channel in document["channels"])


def _page(name: str) -> str:
    """The page's text, with every file it ``literalinclude``s appended."""
    path = GETTING_STARTED / name
    text = path.read_text(encoding="utf-8")
    included = [
        (path.parent / target).resolve().read_text(encoding="utf-8")
        for target in LITERALINCLUDE.findall(text)
    ]
    return "\n".join([text, *included])


def _refused_examples(text: str) -> set[str]:
    return set(REFUSAL.findall(text))


def test_the_hello_world_tutorial_shows_the_refusal_of_an_unknown_address() -> None:
    refused = _refused_examples(_page("hello-world-tutorial.rst"))
    assert refused, "the tutorial shows no '<address> is not a channel of this facility' refusal"
    assert not refused & _channels("hello_world")


@pytest.mark.parametrize(("page", "facility"), sorted(TUTORIALS.items()))
def test_every_address_a_tutorial_names_is_a_channel_of_its_facility(
    page: str, facility: str
) -> None:
    text = _page(page)
    named = set(ADDRESS.findall(text)) - _refused_examples(text)
    assert named, f"{page} names no channel address"
    channels = _channels(facility)
    unknown = sorted(
        address
        for address in named
        if not any(fnmatchcase(channel, address) for channel in channels)
    )
    assert unknown == [], f"{page} names addresses {facility} does not serve"
