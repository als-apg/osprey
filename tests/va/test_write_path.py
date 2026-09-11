"""What decides that a co-hosted setpoint is written into the lattice.

The pyat-coupled partition and the ``subfield`` vocabulary decide it; the
address text does not. A facility whose namespace spells its setpoints
``..._SP``, ``...:CTRL`` or anything else has the same setpoints as one
spelling them ``...:SP``, and this file pins that: the serving layer reads
the manifest's own declaration and never the address string.

Pure python -- no server, no lattice, no Channel Access.
"""

from __future__ import annotations

import pytest

from osprey.services.virtual_accelerator.manifest import (
    PARTITION_PYAT_COUPLED,
    PARTITION_SP_ECHO,
)
from osprey.services.virtual_accelerator.serving.pvdb import build_serving_pvdb
from osprey.services.virtual_accelerator.serving.write_path import (
    MODE_ECHO,
    MODE_PHYSICS,
    CohostWritePath,
    physics_setpoint_addresses,
)

#: A facility that separates its address levels with ``_``, not ``:``.
MAGNET_SP = "ZZEXP_MAG_Q1_CURRENT_SP"
MAGNET_RB = "ZZEXP_MAG_Q1_CURRENT_RB"
VALVE_SP = "ZZEXP_VAC_V1_POSITION_SP"
VALVE_RB = "ZZEXP_VAC_V1_POSITION_RB"


def _channel(address: str, *, subfield: str, partition: str, field: str, family: str) -> dict:
    return {
        "address": address,
        "ring": "ZZEXP",
        "system": "MAG",
        "family": family,
        "device": "Q1",
        "field": field,
        "subfield": subfield,
        "partition": partition,
        "record_type": "ai",
        "noise": False,
    }


@pytest.fixture
def records():
    """A manifest with no ``:`` in any address and a normal SP/RB vocabulary."""
    return build_serving_pvdb(
        [
            _channel(
                MAGNET_SP,
                subfield="SP",
                partition=PARTITION_PYAT_COUPLED,
                field="CURRENT",
                family="QUAD",
            ),
            _channel(
                MAGNET_RB,
                subfield="RB",
                partition=PARTITION_PYAT_COUPLED,
                field="CURRENT",
                family="QUAD",
            ),
            _channel(
                VALVE_SP,
                subfield="SP",
                partition=PARTITION_SP_ECHO,
                field="POSITION",
                family="VALVE",
            ),
            _channel(
                VALVE_RB,
                subfield="RB",
                partition=PARTITION_SP_ECHO,
                field="POSITION",
                family="VALVE",
            ),
        ],
        async_setpoints=True,
    )


class TestSetpointsComeFromTheManifestNotTheAddress:
    def test_the_served_database_states_its_physics_setpoints(self, records):
        assert records.physics_setpoints == {MAGNET_SP}

    def test_the_write_path_reads_that_set(self, records):
        assert physics_setpoint_addresses(records) == {MAGNET_SP}

    def test_a_setpoint_not_spelled_sp_still_routes_through_physics(self, records):
        path = CohostWritePath(
            records,
            enqueue=lambda values, done=None, reset=False: None,
            physics_setpoints=physics_setpoint_addresses(records),
        )

        assert path.routes[MAGNET_SP].mode == MODE_PHYSICS
        assert path.routes[MAGNET_SP].readback == MAGNET_RB

    def test_an_echo_setpoint_is_still_only_an_echo(self, records):
        path = CohostWritePath(
            records,
            enqueue=lambda values, done=None, reset=False: None,
            physics_setpoints=physics_setpoint_addresses(records),
        )

        assert path.routes[VALVE_SP].mode == MODE_ECHO
        assert path.routes[VALVE_SP].readback == VALVE_RB

    def test_a_pyat_coupled_readback_is_not_a_setpoint(self, records):
        assert MAGNET_RB not in records.physics_setpoints
        assert MAGNET_RB not in records.setpoint_readbacks
