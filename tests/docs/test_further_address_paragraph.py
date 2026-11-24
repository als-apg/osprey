"""The how-to's paragraph on a further address names what it states, from the description's list.

``docs/source/how-to/describe-your-facility.rst`` says what an address under a
wiring record's ``also`` states for itself; :data:`osprey_facility.ADDRESS_SLOTS`
is that list, and every other slot is a record fact the build stops on.
"""

from __future__ import annotations

from pathlib import Path

from osprey_facility import ADDRESS_SLOTS, RECORD_FACTS

_PAGE = Path(__file__).resolve().parents[2] / "docs/source/how-to/describe-your-facility.rst"

#: The paragraph's opening words.
_OPENING = "Each address under ``also`` is a channel"


def _states() -> str:
    """The paragraph's first sentence: what a further address states for itself."""
    text = " ".join(_PAGE.read_text(encoding="utf-8").split())
    start = text.index(_OPENING)
    return text[start : text.index(". ", start) + 1]


def test_the_paragraph_names_every_address_slot() -> None:
    states = _states()

    named = [slot for slot in ADDRESS_SLOTS if slot != "provenance"]
    assert [slot for slot in named if f"``{slot}``" not in states] == []
    assert [fact for fact in RECORD_FACTS if f"``{fact}``" in states] == []
