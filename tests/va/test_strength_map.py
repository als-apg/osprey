"""Unit tests for the virtual accelerator's nominal-current baseline.

:class:`~osprey.services.virtual_accelerator.lattice.strengths.StrengthMap`
scales every magnet strength by ``I / I_nom``, so the set of channels it holds
a nominal current for decides which writes reach the lattice at all. That set
comes from the manifest -- the pyat-coupled partition's setpoint half -- and
not from the address text, which is a facility's own spelling and carries no
promise that a magnet current is written ``:CURRENT:SP``.

Two facts are pinned here: a setpoint the demo tree's grammar does not name is
still in the map, and a channel the manifest puts outside the pyat-coupled
partition is not -- the sp-echo transport-line magnets share family and device
tokens with the ring's, so a looser filter would hand a ring device the wrong
baseline.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from osprey.services.virtual_accelerator.lattice import strengths
from osprey.services.virtual_accelerator.lattice.strengths import StrengthMap
from osprey.services.virtual_accelerator.manifest import (
    PARTITION_PYAT_COUPLED,
    PARTITION_SP_ECHO,
    READBACK_SUBFIELD,
    RECORD_TYPE_ANALOG,
    SETPOINT_SUBFIELD,
    pyat_coupled_setpoint_addresses,
)

#: A magnet current setpoint spelled the way no six-token colon grammar reads:
#: the facility uses slashes and a dotted subfield, so nothing about the text
#: ends in ``:CURRENT:SP``.
_RING_SETPOINT = "ZZSM/MAG/QF01/CURRENT.SETPOINT"
_RING_READBACK = "ZZSM/MAG/QF01/CURRENT.READBACK"
#: Same family and device tokens, one partition away: a transport-line magnet
#: that no lattice element backs.
_ECHO_SETPOINT = "ZZTL/MAG/QF01/CURRENT.SETPOINT"


def _channel(address: str, subfield: str, partition: str) -> dict[str, Any]:
    """One manifest channel for the ``QF`` device ``01`` of a made-up facility."""
    return {
        "address": address,
        "ring": "ZZSM",
        "system": "MAG",
        "family": "QF",
        "device": "01",
        "field": "CURRENT",
        "subfield": subfield,
        "partition": partition,
        "record_type": RECORD_TYPE_ANALOG,
        "noise": False,
    }


_CHANNELS = [
    _channel(_RING_SETPOINT, SETPOINT_SUBFIELD, PARTITION_PYAT_COUPLED),
    _channel(_RING_READBACK, READBACK_SUBFIELD, PARTITION_PYAT_COUPLED),
    _channel(_ECHO_SETPOINT, SETPOINT_SUBFIELD, PARTITION_SP_ECHO),
]

_MACHINE_JSON = {
    _RING_SETPOINT: {"value": 137.5, "units": "A"},
    _RING_READBACK: {"value": 137.5, "units": "A"},
    _ECHO_SETPOINT: {"value": 42.0, "units": "A"},
}


@pytest.fixture
def foreign_map(monkeypatch: pytest.MonkeyPatch) -> StrengthMap:
    """A map built from a facility whose addresses carry no colon grammar."""
    monkeypatch.setattr(
        strengths, "load_machine_json_channels", lambda: dict(_MACHINE_JSON), raising=True
    )
    # No ring: every assertion below is about the nominal-current baseline, and
    # baked strengths are snapshotted from whatever elements the ring holds.
    return StrengthMap(cast(Any, []), channels=_CHANNELS)


class TestTheNominalCurrentBaselineComesFromTheManifest:
    def test_a_setpoint_no_address_grammar_names_still_has_its_nominal(
        self, foreign_map: StrengthMap
    ) -> None:
        assert foreign_map.i_nom(_RING_SETPOINT) == 137.5

    def test_a_channel_outside_the_pyat_coupled_partition_is_not_in_the_map(
        self, foreign_map: StrengthMap
    ) -> None:
        with pytest.raises(KeyError):
            foreign_map.i_nom(_ECHO_SETPOINT)

    def test_a_readback_is_not_in_the_map(self, foreign_map: StrengthMap) -> None:
        with pytest.raises(KeyError):
            foreign_map.i_nom(_RING_READBACK)


class TestADeviceResolvesThroughTheManifestRoster:
    """``i_nom_for`` answers "what is this device's baseline" without an address.

    ``apply`` and the model's current readback hold a family and a device id,
    never an address. Formatting one from them would put a second spelling of
    the demo tree's grammar in the module every write goes through; the
    manifest already carries the family and device of each channel, so the
    device -> setpoint map is derived from the same rows the baseline is.
    """

    def test_a_device_resolves_when_no_address_grammar_names_it(
        self, foreign_map: StrengthMap
    ) -> None:
        assert foreign_map.i_nom_for("QF", "01") == 137.5

    def test_it_agrees_with_the_address_accessor_across_the_bundled_manifest(self) -> None:
        from osprey.services.virtual_accelerator.manifest import build_manifest

        channels = build_manifest()["channels"]
        strength_map = StrengthMap(cast(Any, []), channels=channels)
        coupled_setpoints = pyat_coupled_setpoint_addresses(channels)
        by_address = {channel["address"]: channel for channel in channels}

        assert coupled_setpoints
        for address in sorted(coupled_setpoints):
            channel = by_address[address]
            assert strength_map.i_nom_for(channel["family"], channel["device"]) == (
                strength_map.i_nom(address)
            )

    def test_an_unknown_pair_raises_naming_both_tokens(self, foreign_map: StrengthMap) -> None:
        with pytest.raises(ValueError) as excinfo:
            foreign_map.i_nom_for("ZZ", "99")
        assert "ZZ" in str(excinfo.value)
        assert "99" in str(excinfo.value)

    def test_a_device_outside_the_pyat_coupled_partition_does_not_resolve(
        self, foreign_map: StrengthMap
    ) -> None:
        # ``_ECHO_SETPOINT`` carries the same family and device tokens as the
        # ring's magnet; only the partition tells them apart.
        with pytest.raises(ValueError):
            StrengthMap(
                cast(Any, []),
                channels=[_channel(_ECHO_SETPOINT, SETPOINT_SUBFIELD, PARTITION_SP_ECHO)],
            ).i_nom_for("QF", "01")
