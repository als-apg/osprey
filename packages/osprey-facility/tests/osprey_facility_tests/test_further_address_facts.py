"""A further address of a wiring record states only its own slots and reads the rest from its record.

:data:`osprey_facility.ADDRESS_SLOTS` and :data:`osprey_facility.RECORD_FACTS`
split every channel slot of the published schema between the two, and
:meth:`FacilityRecords.channel` reads a further address through that split.
"""

from __future__ import annotations

import json
from typing import Any

from osprey_facility import ADDRESS_KIND, ADDRESS_SLOTS, RECORD_FACTS
from osprey_facility.records import FacilityRecords
from osprey_facility_tests._harness import PACKAGE_SRC

SCHEMA = PACKAGE_SRC / "osprey_facility" / "schema" / "_generated" / "facility.schema.json"

RECORD, FURTHER, ALONE = "QD:SP", "RACK3:UNIT2:SP", "QF:SP"


def _document() -> dict[str, Any]:
    """Three channels: a setpoint, a further address of it, and a channel no record names."""
    return {
        "channels": [
            {
                "id": RECORD,
                "role": "setpoint",
                "value_type": "float",
                "pair": "QD:RB",
                "unit": "A",
                "tolerance": {"absolute": 0.5},
                "timeout_s": 2.0,
                "signal": "current_setpoint",
                "label": "QD current",
                "names": ["QD"],
                "on": {"device": "SR/QD"},
            },
            {
                "id": FURTHER,
                "role": "setpoint",
                "value_type": "float",
                "label": "Rack 3 unit 2",
                "names": ["U2"],
                "description": "the rack's second unit",
                "on": {"device": "RACK3"},
                "x_note": "kept",
            },
            {"id": ALONE, "role": "setpoint", "pair": "QF:RB", "unit": "A"},
        ],
        "models": [
            {"name": "SR", "wiring": [{"id": RECORD, "address": RECORD, "also": [FURTHER]}]},
        ],
    }


def test_every_channel_slot_is_an_address_slot_or_a_record_fact() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    slots = set(schema["$defs"]["Channel"]["properties"])

    assert not set(ADDRESS_SLOTS) & set(RECORD_FACTS)
    assert set(ADDRESS_SLOTS) | set(RECORD_FACTS) == slots


def test_the_kind_is_a_part_of_the_record_facts() -> None:
    assert set(ADDRESS_KIND) <= set(RECORD_FACTS)


def test_a_further_address_reads_every_record_fact_from_its_record() -> None:
    records = FacilityRecords(_document())

    read = records.channel(FURTHER)

    assert read is not None
    record = records.record("channel", RECORD)
    assert {key: read.get(key) for key in RECORD_FACTS} == {
        key: record.get(key) for key in RECORD_FACTS
    }
    assert read["pair"] == "QD:RB"
    assert read["tolerance"] == {"absolute": 0.5}


def test_a_further_address_keeps_its_own_label_names_and_place() -> None:
    read = FacilityRecords(_document()).channel(FURTHER)

    assert read is not None
    assert read["id"] == FURTHER
    assert read["label"] == "Rack 3 unit 2"
    assert read["names"] == ["U2"]
    assert read["on"] == {"device": "RACK3"}
    assert read["description"] == "the rack's second unit"
    assert read["x_note"] == "kept"


def test_an_address_no_record_names_reads_as_its_own_row() -> None:
    records = FacilityRecords(_document())

    assert records.channel(ALONE) is records.record("channel", ALONE)
    assert records.channel(RECORD) is records.record("channel", RECORD)
    assert records.channel("NO:SUCH") is None


def test_wiring_model_names_the_model_of_a_further_address() -> None:
    records = FacilityRecords(_document())

    assert records.wiring_model(FURTHER) == "SR"
    assert records.wiring_model(RECORD) == "SR"
    assert records.wiring_model(ALONE) is None
