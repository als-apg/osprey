"""Tests for the MML device grain in ``ttl_generator.mml_source``.

``devices_for_family`` turns one family view into the corpus's devices: one per
``(system, family, index)`` for a family that carries a channel. Each rule the
proposal states for a device (identity, source name, raw type, position and its
ordinal stand-in, status/sector/device/elementIndex, section code, prose) is
pinned here on a small normalised body and a hand-built mapping.
"""

from __future__ import annotations

import math

import pytest

from osprey.services.facility_knowledge.ttl_generator.mml_source import devices_for_family
from osprey.services.facility_knowledge.ttl_generator.model import (
    DEVICE_IRI_PREFIX,
    Address,
    Device,
)
from osprey.services.mml.family import FamilyView
from osprey.services.mml.mapping.schema import Facility, Family, Mapping, System

FACILITY = "quokka"


def _mapping(
    *,
    system_raw: str = "RING",
    system_name: str = "SR",
    system_description: str | None = "The storage ring",
    family_raw: str = "BPMx",
    rename: str | None = None,
    family_description: str | None = "Horizontal BPMs",
    facility: str | None = FACILITY,
) -> Mapping:
    return Mapping(
        facility=Facility(token=facility, title=None, description=None, provenance="human"),
        systems={
            system_raw: System(
                raw=system_raw,
                name=system_name,
                description=system_description,
                provenance="human",
            )
        },
        section_order=(system_raw,),
        families={
            family_raw: Family(
                raw=family_raw,
                rename=rename,
                branch=None,
                class_="BPM",
                aliases=(),
                description=family_description,
                provenance="human",
                channels=1,
                fields={},
            )
        },
    )


def _view(body: dict, system: str = "RING", name: str = "BPMx") -> FamilyView:
    return FamilyView(system, name, body)


def _two_device_body(**arrays) -> dict:
    return {"Monitor": {"ChannelNames": ["SR:BPM1:X", "SR:BPM2:X"]}, **arrays}


def _extras(device: Device) -> dict:
    return dict(device.extra_properties)


class TestGrain:
    """One device per index of a channel-bearing family."""

    def test_one_device_per_index(self):
        """A two-device family yields two devices, in index order."""
        devices = devices_for_family(_view(_two_device_body()), _mapping())
        assert [device.device for device in devices] == ["1", "2"]

    def test_zero_channel_family_yields_no_devices(self):
        """A family whose every slot is blank has no devices."""
        body = {"DeviceList": [[1, 1], [1, 2]], "Monitor": {"ChannelNames": [None, " "]}}
        assert devices_for_family(_view(body), _mapping()) == []

    def test_family_without_fields_yields_no_devices(self):
        """A family with a DeviceList but no channel field has no devices."""
        body = {"DeviceList": [[1, 1], [1, 2]]}
        assert devices_for_family(_view(body), _mapping()) == []

    def test_second_call_is_identical(self):
        """The function is deterministic."""
        view = _view(_two_device_body(Position=[1.0, "NaN"], Status=[1, 0]))
        assert devices_for_family(view, _mapping()) == devices_for_family(view, _mapping())


class TestIdentity:
    """Mapped tokens build the address, IRI and identifiers."""

    def test_address_tokens_are_mapped(self):
        """ring and system are the mapped system; family is the renamed family."""
        mapping = _mapping(rename="BPMX")
        device = devices_for_family(_view(_two_device_body()), mapping)[0]
        address = Address(device.ring, device.system, device.family, device.device, "", "")
        assert address == Address("SR", "SR", "BPMX", "1", "", "")

    def test_iri_local_name_is_system_family_index(self):
        """The device IRI ends in ``{facility}_{system}_{family}_{index}``."""
        device = devices_for_family(_view(_two_device_body()), _mapping())[1]
        assert device.iri == f"{DEVICE_IRI_PREFIX}{FACILITY}_SR_BPMx_2"

    def test_device_id_and_section_id_carry_facility_and_system(self):
        """deviceId and sourceSectionId name the facility and mapped system."""
        device = devices_for_family(_view(_two_device_body()), _mapping())[0]
        assert device.device_id == f"narad:device:{FACILITY}:SR:BPMx_1"
        assert device.source_section_id == f"narad:section:{FACILITY}:sr"

    def test_section_code_is_mapped_system(self):
        """section_code is the mapped system name, not the raw token."""
        device = devices_for_family(_view(_two_device_body()), _mapping())[0]
        assert device.section_code == "SR"

    def test_descriptions_come_from_the_mapping(self):
        """System and ring prose are the system's; family prose is the family's."""
        device = devices_for_family(_view(_two_device_body()), _mapping())[0]
        assert device.system_description == "The storage ring"
        assert device.ring_description == "The storage ring"
        assert device.family_description == "Horizontal BPMs"

    def test_absent_descriptions_stay_none(self):
        """A mapping with no prose yields no prose."""
        mapping = _mapping(system_description=None, family_description=None)
        device = devices_for_family(_view(_two_device_body()), mapping)[0]
        assert device.system_description is None
        assert device.ring_description is None
        assert device.family_description is None

    def test_bindings_are_not_attached_here(self):
        """Devices leave the binding list empty."""
        device = devices_for_family(_view(_two_device_body()), _mapping())[0]
        assert device.binding_iris == ()

    def test_ordinals_are_one_based_family_positions(self):
        """Provisional ordinals count devices from 1 within the family."""
        devices = devices_for_family(_view(_two_device_body()), _mapping())
        assert [d.ordinal_in_section for d in devices] == [1, 2]
        assert [d.ordinal_in_facility for d in devices] == [1, 2]

    def test_unmapped_system_is_refused(self):
        """A view whose system the mapping does not name raises ValueError."""
        with pytest.raises(ValueError, match="BOOST"):
            devices_for_family(_view(_two_device_body(), system="BOOST"), _mapping())

    def test_unmapped_family_is_refused(self):
        """A view whose family the mapping does not name raises ValueError."""
        with pytest.raises(ValueError, match="HCM"):
            devices_for_family(_view(_two_device_body(), name="HCM"), _mapping())

    def test_missing_facility_token_is_refused(self):
        """An undecided facility token cannot mint an IRI."""
        with pytest.raises(ValueError, match="facility"):
            devices_for_family(_view(_two_device_body()), _mapping(facility=None))


class TestSourceNameAndRawType:
    """CommonNames and DeviceType fill in per device when aligned."""

    def test_common_names_when_aligned(self):
        """An aligned CommonNames slot is the source name."""
        body = _two_device_body(CommonNames=["BPM(1,2)", "BPM(1,3)"])
        devices = devices_for_family(_view(body), _mapping())
        assert [d.source_name for d in devices] == ["BPM(1,2)", "BPM(1,3)"]

    def test_blank_common_name_slot_falls_back(self):
        """A blank or null slot falls back to ``{family}{index}`` for that device only."""
        body = _two_device_body(CommonNames=["", "BPM(1,3)"])
        devices = devices_for_family(_view(body), _mapping(rename="BPMX"))
        assert [d.source_name for d in devices] == ["BPMX1", "BPM(1,3)"]
        body = _two_device_body(CommonNames=[None, "BPM(1,3)"])
        assert devices_for_family(_view(body), _mapping())[0].source_name == "BPMx1"

    def test_misaligned_common_names_fall_back(self):
        """A CommonNames list of the wrong length is not used."""
        body = _two_device_body(CommonNames=["only-one"])
        devices = devices_for_family(_view(body), _mapping())
        assert [d.source_name for d in devices] == ["BPMx1", "BPMx2"]

    def test_device_type_when_aligned(self):
        """An aligned, non-empty DeviceType slot is the raw type; a blank one falls back."""
        body = _two_device_body(DeviceType=["Bergoz", ""])
        devices = devices_for_family(_view(body), _mapping(rename="BPMX"))
        assert [d.raw_type for d in devices] == ["Bergoz", "BPMx"]

    def test_raw_type_defaults_to_the_export_family_token(self):
        """Without DeviceType the raw type is the export's own family word, not the rename."""
        devices = devices_for_family(_view(_two_device_body()), _mapping(rename="BPMX"))
        assert {d.raw_type for d in devices} == {"BPMx"}
        assert {d.family for d in devices} == {"BPMX"}


class TestPosition:
    """Position is used only when finite; otherwise the ordinal stands in."""

    def test_nan_slot_uses_ordinal_for_that_device_only(self):
        """``[1.0, 'NaN', 3.0]`` keeps 1.0 and 3.0 and stands in 2.0 for the middle."""
        body = {
            "Monitor": {"ChannelNames": ["A", "B", "C"]},
            "Position": [1.0, "NaN", 3.0],
        }
        devices = devices_for_family(_view(body), _mapping())
        assert [d.s_position_m for d in devices] == [1.0, 2.0, 3.0]
        assert "positionSource" not in _extras(devices[0])
        assert _extras(devices[1])["positionSource"] == "ordinal"
        assert "positionSource" not in _extras(devices[2])

    @pytest.mark.parametrize("slot", [None, "Inf", "-Inf", float("nan"), float("inf"), "12.5"])
    def test_non_finite_or_non_number_slot_is_a_stand_in(self, slot):
        """null, string and non-finite slots never reach s_position_m."""
        body = _two_device_body(Position=[slot, 4.0])
        devices = devices_for_family(_view(body), _mapping())
        assert devices[0].s_position_m == 1.0
        assert math.isfinite(devices[0].s_position_m)
        assert _extras(devices[0])["positionSource"] == "ordinal"
        assert devices[1].s_position_m == 4.0

    def test_boolean_slot_is_a_stand_in(self):
        """A JSON boolean is not a position."""
        body = _two_device_body(Position=[True, 4.0])
        device = devices_for_family(_view(body), _mapping())[0]
        assert device.s_position_m == 1.0
        assert _extras(device)["positionSource"] == "ordinal"

    def test_integer_position_becomes_float(self):
        """An integer slot is kept as a float."""
        device = devices_for_family(_view(_two_device_body(Position=[7, 8])), _mapping())[0]
        assert device.s_position_m == 7.0
        assert isinstance(device.s_position_m, float)

    def test_scalar_position_on_one_device_family(self):
        """A finite scalar on a one-device family is its position."""
        body = {"Monitor": {"ChannelNames": ["A"]}, "Position": 12.25}
        device = devices_for_family(_view(body), _mapping())[0]
        assert device.s_position_m == 12.25
        assert "positionSource" not in _extras(device)

    def test_non_finite_scalar_on_one_device_family(self):
        """A 'NaN' scalar on a one-device family stands in the ordinal."""
        body = {"Monitor": {"ChannelNames": ["A"]}, "Position": "NaN"}
        device = devices_for_family(_view(body), _mapping())[0]
        assert device.s_position_m == 1.0
        assert _extras(device)["positionSource"] == "ordinal"

    def test_scalar_position_on_multi_device_family_is_ignored(self):
        """A scalar does not align with several devices; every device stands in."""
        devices = devices_for_family(_view(_two_device_body(Position=5.0)), _mapping())
        assert [d.s_position_m for d in devices] == [1.0, 2.0]
        assert all(_extras(d)["positionSource"] == "ordinal" for d in devices)

    def test_absent_position_uses_ordinals(self):
        """No Position at all stands in the ordinal on every device."""
        devices = devices_for_family(_view(_two_device_body()), _mapping())
        assert [d.s_position_m for d in devices] == [1.0, 2.0]
        assert all(_extras(d)["positionSource"] == "ordinal" for d in devices)

    def test_position_from_setup_block(self):
        """Arrays read from a setup block count as aligned."""
        body = _two_device_body(setup={"Position": [0.5, 1.5]})
        devices = devices_for_family(_view(body), _mapping())
        assert [d.s_position_m for d in devices] == [0.5, 1.5]


class TestExtraProperties:
    """status, sector, device and elementIndex ride as sorted extra properties."""

    def test_device_list_rows_give_sector_and_device(self):
        """``DeviceList [[1, 2], [1, 3]]`` yields sector 1 and devices 2, 3."""
        body = {
            "DeviceList": [[1, 2], [1, 3]],
            "Monitor": {"ChannelNames": ["A", "B"]},
            "Position": [1.0, 2.0],
        }
        devices = devices_for_family(_view(body), _mapping())
        assert [(_extras(d)["sector"], _extras(d)["device"]) for d in devices] == [(1, 2), (1, 3)]
        assert all(type(_extras(d)["sector"]) is int for d in devices)

    def test_flat_device_list_pair_on_one_device_family(self):
        """A flat ``[sector, device]`` pair is the one device's row."""
        body = {"DeviceList": [4, 7], "Monitor": {"ChannelNames": ["A"]}}
        device = devices_for_family(_view(body), _mapping())[0]
        assert (_extras(device)["sector"], _extras(device)["device"]) == (4, 7)

    def test_integer_valued_float_device_list_is_cast(self):
        """A .mat export's float rows become ints."""
        body = {"DeviceList": [[1.0, 2.0], [1.0, 3.0]], "Monitor": {"ChannelNames": ["A", "B"]}}
        device = devices_for_family(_view(body), _mapping())[1]
        assert (_extras(device)["sector"], _extras(device)["device"]) == (1, 3)
        assert type(_extras(device)["device"]) is int

    def test_misaligned_device_list_gives_no_sector(self):
        """A DeviceList that does not align with the devices is not used."""
        body = {"DeviceList": "junk", "Monitor": {"ChannelNames": ["A", "B"]}}
        devices = devices_for_family(_view(body), _mapping())
        assert all("sector" not in _extras(d) and "device" not in _extras(d) for d in devices)

    def test_status_is_int_never_bool(self):
        """Aligned Status slots, booleans included, are carried as ints."""
        body = _two_device_body(Status=[True, 0])
        devices = devices_for_family(_view(body), _mapping())
        statuses = [_extras(d)["status"] for d in devices]
        assert statuses == [1, 0]
        assert all(type(value) is int for value in statuses)

    def test_non_numeric_status_slot_is_skipped(self):
        """A null Status slot carries no status for that device."""
        devices = devices_for_family(_view(_two_device_body(Status=[None, 1])), _mapping())
        assert "status" not in _extras(devices[0])
        assert _extras(devices[1])["status"] == 1

    def test_misaligned_status_is_not_used(self):
        """A Status list of the wrong length carries no status."""
        devices = devices_for_family(_view(_two_device_body(Status=[1])), _mapping())
        assert all("status" not in _extras(d) for d in devices)

    def test_element_index_when_aligned(self):
        """An aligned ElementList gives elementIndex."""
        devices = devices_for_family(_view(_two_device_body(ElementList=[3, 9])), _mapping())
        assert [_extras(d)["elementIndex"] for d in devices] == [3, 9]

    def test_extra_properties_are_sorted_by_key(self):
        """Every property present is carried, sorted by key."""
        body = {
            "DeviceList": [[1, 2], [1, 3]],
            "ElementList": [5, 6],
            "Status": [1, 1],
            "Position": ["NaN", 2.0],
            "Monitor": {"ChannelNames": ["A", "B"]},
        }
        device = devices_for_family(_view(body), _mapping())[0]
        keys = [key for key, _value in device.extra_properties]
        assert keys == ["device", "elementIndex", "positionSource", "sector", "status"]
        assert keys == sorted(keys)
