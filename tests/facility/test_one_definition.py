"""Facts of the facility build that are defined in exactly one place."""

from __future__ import annotations

from osprey.facility.combine import SET_VALUED
from osprey.facility.sources import COMPUTED_SLOTS, KIND_CLASS, RECORD_FILES, SET_VALUED_SLOTS


def test_combine_sorts_exactly_the_set_valued_rows_a_layer_may_state() -> None:
    for kind in RECORD_FILES.values():
        cls = KIND_CLASS[kind]
        assert {(cls, slot) for slot in SET_VALUED[kind]} == {
            row for row in SET_VALUED_SLOTS if row[0] == cls and row[1] not in COMPUTED_SLOTS[kind]
        }, kind
    assert SET_VALUED == {
        "place": frozenset({"names"}),
        "device": frozenset({"names"}),
        "channel": frozenset({"names", "tags", "former_addresses", "endpoint_of"}),
        "group": frozenset({"names", "members"}),
    }
