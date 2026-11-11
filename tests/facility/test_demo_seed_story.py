"""What the example facility's seeds say about the machine at rest.

The storage ring stores beam in top-off: the current refills every period and
decays between refills, and the lifetime readback reads the decay time that
produces it. The booster carries booster-scale current, no magnet or vacuum
supply reads faulted, and the ring is in machine studies: no beam goes to
users, as ``SR:BEAM:USER_MODE`` says.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from osprey.facility.sources import read_yaml
from osprey_connectors.simulation import topoff

TREE = Path(__file__).resolve().parents[2] / "src/osprey/templates/facilities/example"
DCCT = "SR:DIAG:DCCT:01:CURRENT:RB"
LIFETIME = "SR:DIAG:DCCT:01:LIFETIME:RB"
BOOSTER_DCCT = "BR:DIAG:DCCT:01:CURRENT:RB"
MODE = "SR:OPS:MODE:RB"

#: The archive's hot and tail sampling cadences, in seconds.
ARCHIVE_CADENCES_S = (10.0, 60.0)


def seeds() -> dict[str, dict[str, Any]]:
    """The committed seeds, keyed by address."""
    document: dict[str, dict[str, Any]] = read_yaml(
        (TREE / "seeds.yaml").read_text(encoding="utf-8")
    )
    return document


def channels() -> dict[str, dict[str, Any]]:
    """The committed channel records, keyed by id."""
    text = (TREE / "records/channels.yaml").read_text(encoding="utf-8")
    return {record["id"]: record for record in read_yaml(text)}


def test_the_stored_current_is_a_top_off_sawtooth_about_500_ma() -> None:
    seed = seeds()[DCCT]
    assert "nominal" not in seed
    record = seed["topoff"]
    assert channels()[DCCT]["unit"] == "mA"
    held = topoff.held_value(seed)
    assert held is not None
    assert 499.5 < held < 500.0
    tau, period = topoff.tau_period(record)
    sawtooth = record["ceiling"] * (1 - math.exp(-(period - 1e-6) / tau))
    assert 0.5 < sawtooth < 0.7


def test_the_refill_period_never_locks_to_the_archive_cadence() -> None:
    period = seeds()[DCCT]["topoff"]["period_s"]
    for cadence in ARCHIVE_CADENCES_S:
        assert (cadence / period) % 1 and (period / cadence) % 1, cadence


def test_the_lifetime_reads_the_decay_time_of_the_stored_current() -> None:
    every = seeds()
    seed = every[LIFETIME]
    assert seed == {"topoff_lifetime": {"of": DCCT}}
    assert channels()[LIFETIME]["unit"] == "h"
    assert topoff.held_value(seed, every.get) == every[DCCT]["topoff"]["lifetime_h"]


def test_the_booster_carries_booster_scale_current() -> None:
    seed = seeds()[BOOSTER_DCCT]
    assert channels()[BOOSTER_DCCT]["unit"] == "mA"
    assert 1.0 <= seed["nominal"] <= 1.5
    assert 0 < seed["noise"]["absolute"] <= 0.02 * seed["nominal"]


def test_no_magnet_or_vacuum_supply_reads_faulted() -> None:
    flags = {
        address: seed["nominal"]
        for address, seed in seeds().items()
        if address.endswith(":STATUS:FAULT")
    }
    ring = [a for a in flags if a.startswith(("SR:MAG:", "SR:VAC:"))]
    assert len(ring) == 300
    assert {flags[a] for a in flags} == {"FALSE"}


def test_the_ring_is_in_machine_studies() -> None:
    assert seeds()[MODE] == {"nominal": "Machine studies"}
    assert seeds()["SR:BEAM:USER_MODE"] == {"nominal": "FALSE"}
    assert channels()[MODE]["options"][0] == "User operation"
