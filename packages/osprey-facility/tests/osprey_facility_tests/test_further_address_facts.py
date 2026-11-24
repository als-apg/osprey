"""A further address of a wiring record states only its own slots and reads the rest from its record.

:data:`osprey_facility.ADDRESS_SLOTS` and :data:`osprey_facility.RECORD_FACTS`
split every channel slot of the published schema between the two.
"""

from __future__ import annotations

import json

from osprey_facility import ADDRESS_KIND, ADDRESS_SLOTS, RECORD_FACTS
from osprey_facility_tests._harness import PACKAGE_SRC

SCHEMA = PACKAGE_SRC / "osprey_facility" / "schema" / "_generated" / "facility.schema.json"


def test_every_channel_slot_is_an_address_slot_or_a_record_fact() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    slots = set(schema["$defs"]["Channel"]["properties"])

    assert not set(ADDRESS_SLOTS) & set(RECORD_FACTS)
    assert set(ADDRESS_SLOTS) | set(RECORD_FACTS) == slots


def test_the_kind_is_a_part_of_the_record_facts() -> None:
    assert set(ADDRESS_KIND) <= set(RECORD_FACTS)
