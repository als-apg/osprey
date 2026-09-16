"""Tests for the MML binding grain and model builder in ``ttl_generator.mml_source``.

``bindings_for_family`` turns one family view into the corpus's channel bindings
and signal groups; ``build_graph_model`` walks every system in ``section_order``
and assembles the complete, directed :class:`GraphModel`. Each binding rule the
proposal states (one binding per non-blank slot per key, protocol and subfield
per key, broadcast, partial lists, prose from the mapping, the scalar-only
``HWUnits``/``DataType`` rule) and each model refusal (undirected group,
duplicate device IRI) is pinned here on small normalised bodies.
"""

from __future__ import annotations

import pytest

from osprey.services.facility_knowledge.ttl_generator.mml_source import (
    bindings_for_family,
    build_graph_model,
)
from osprey.services.facility_knowledge.ttl_generator.model import (
    BINDING_IRI_PREFIX,
    CONFIDENCE,
    DEVICE_IRI_PREFIX,
    ChannelBinding,
    GraphModel,
)
from osprey.services.mml.family import FamilyView
from osprey.services.mml.mapping.schema import (
    Direction,
    Facility,
    Family,
    Field,
    Mapping,
    System,
)

FACILITY = "quokka"


def _family(raw: str, *, rename: str | None = None, fields: dict[str, str] | None = None):
    return Family(
        raw=raw,
        rename=rename,
        branch=None,
        class_="BPM",
        aliases=(raw,),
        description=f"{raw} family",
        provenance="stated",
        channels=1,
        fields={
            name: Field(description=text, provenance="stated")
            for name, text in (fields or {}).items()
        },
    )


def _mapping(
    *,
    systems: dict[str, str] | None = None,
    families: dict[str, Family] | None = None,
    directions: dict[str, str | None] | None = None,
) -> Mapping:
    systems = systems if systems is not None else {"RING": "SR"}
    families = (
        families
        if families is not None
        else {"BPMx": _family("BPMx", fields={"Monitor": "Horizontal position"})}
    )
    return Mapping(
        facility=Facility(token=FACILITY, title=None, description=None, provenance="stated"),
        systems={
            raw: System(raw=raw, name=name, description=f"{name} system", provenance="stated")
            for raw, name in systems.items()
        },
        section_order=tuple(systems.values()),
        families=families,
        directions={
            key: Direction(direction=value, provenance="stated", override=False)
            for key, value in (directions or {}).items()
        },
    )


def _view(body: dict, system: str = "RING", name: str = "BPMx") -> FamilyView:
    return FamilyView(system, name, body)


def _two_devices(field: dict) -> dict:
    return {"DeviceList": [[1, 1], [1, 2]], "Monitor": field}


def _extras(binding: ChannelBinding) -> dict:
    return dict(binding.extra_properties)


class TestBindingGrain:
    """One binding per non-blank slot per channel key."""

    def test_one_binding_per_slot(self):
        """Two devices with two PVs yield two bindings, device order."""
        view = _view(_two_devices({"ChannelNames": ["SR:BPM1:X", "SR:BPM2:X"]}))
        bindings, _ = bindings_for_family(view, _mapping())
        assert [b.full_pv for b in bindings] == ["SR:BPM1:X", "SR:BPM2:X"]
        assert [b.address.device for b in bindings] == ["1", "2"]

    def test_full_pv_is_exact_stripped_string(self):
        """The PV is the slot stripped, never rebuilt from the address."""
        view = _view(_two_devices({"ChannelNames": ["  SR:BPM1:X ", "sr{bpm}2-x"]}))
        bindings, _ = bindings_for_family(view, _mapping())
        assert [b.full_pv for b in bindings] == ["SR:BPM1:X", "sr{bpm}2-x"]
        assert all(b.full_pv != b.address.text for b in bindings)

    def test_blank_slots_bind_nothing(self):
        """A blank or null slot is skipped; its neighbours keep their index."""
        body = {
            "DeviceList": [[1, 1], [1, 2], [1, 3]],
            "Monitor": {"ChannelNames": ["A", "  ", "C"]},
        }
        bindings, _ = bindings_for_family(_view(body), _mapping())
        assert [(b.address.device, b.full_pv) for b in bindings] == [("1", "A"), ("3", "C")]

    def test_identity_and_iris(self):
        """Address, local name, IRIs and signal key use mapped tokens."""
        mapping = _mapping(
            systems={"RING": "SR"},
            families={"BPMx": _family("BPMx", rename="BPMX", fields={"Monitor": "pos"})},
        )
        view = _view(_two_devices({"ChannelNames": ["P1", "P2"]}))
        binding = bindings_for_family(view, mapping)[0][0]
        assert (binding.address.ring, binding.address.system) == ("SR", "SR")
        assert binding.address.family == "BPMX"
        assert (binding.address.field, binding.address.subfield) == ("Monitor", "val")
        assert binding.iri == f"{BINDING_IRI_PREFIX}narad_endpoint_{FACILITY}_SR_BPMX_1_Monitor_val"
        assert binding.device_iri == f"{DEVICE_IRI_PREFIX}{FACILITY}_SR_BPMX_1"
        assert binding.device_key == ("SR", "SR", "BPMX", "1")
        assert binding.signal_key == ("BPMX", "Monitor", "val")
        assert binding.signal_name == "bpmx_monitor_val"
        assert binding.confidence == CONFIDENCE

    def test_zero_channel_family_yields_nothing(self):
        """A family with no non-blank slot yields no binding and no group."""
        view = _view({"Monitor": {"ChannelNames": []}})
        assert bindings_for_family(view, _mapping()) == ([], [])


class TestProtocol:
    """Protocol and subfield follow the channel key."""

    def test_channel_names_is_ca_val(self):
        """``ChannelNames`` binds as ``ca`` with subfield ``val``."""
        view = _view(_two_devices({"ChannelNames": ["A", "B"]}))
        binding = bindings_for_family(view, _mapping())[0][0]
        assert (binding.protocol, binding.address.subfield) == ("ca", "val")

    def test_tango_names_is_tango_tango(self):
        """``TangoNames`` binds as ``tango`` with subfield ``tango``."""
        view = _view(_two_devices({"TangoNames": ["sr/bpm/1", "sr/bpm/2"]}))
        binding = bindings_for_family(view, _mapping())[0][0]
        assert (binding.protocol, binding.address.subfield) == ("tango", "tango")

    def test_dual_key_field_binds_twice_per_slot(self):
        """A field carrying both keys yields two bindings per slot and two groups."""
        view = _view(_two_devices({"ChannelNames": ["A1", "A2"], "TangoNames": ["t/1", "t/2"]}))
        bindings, groups = bindings_for_family(view, _mapping())
        assert [(b.address.device, b.address.subfield, b.protocol) for b in bindings] == [
            ("1", "val", "ca"),
            ("1", "tango", "tango"),
            ("2", "val", "ca"),
            ("2", "tango", "tango"),
        ]
        assert {g.key: g.members for g in groups} == {
            ("BPMx", "Monitor", "val"): ("A1", "A2"),
            ("BPMx", "Monitor", "tango"): ("t/1", "t/2"),
        }


class TestListShapes:
    """Broadcast, empty and partial channel lists."""

    def test_one_row_list_broadcasts(self):
        """A 1-row list on two devices binds each device to the same PV, marked."""
        view = _view(_two_devices({"ChannelNames": ["SR:DCCT"]}))
        bindings, groups = bindings_for_family(view, _mapping())
        assert [(b.address.device, b.full_pv) for b in bindings] == [
            ("1", "SR:DCCT"),
            ("2", "SR:DCCT"),
        ]
        assert all(_extras(b)["broadcast"] == 1 for b in bindings)
        assert groups[0].members == ("SR:DCCT", "SR:DCCT")

    def test_one_device_family_is_not_broadcast(self):
        """A 1-row list on a one-device family carries no broadcast mark."""
        view = _view({"DeviceList": [[1, 1]], "Monitor": {"ChannelNames": ["X"]}})
        bindings, _ = bindings_for_family(view, _mapping())
        assert "broadcast" not in _extras(bindings[0])

    def test_broadcast_marks_only_the_broadcast_key(self):
        """In a dual-key field only the key whose list broadcasts is marked."""
        view = _view(_two_devices({"ChannelNames": ["A1", "A2"], "TangoNames": ["t/all"]}))
        bindings, _ = bindings_for_family(view, _mapping())
        marks = {(b.address.device, b.address.subfield): _extras(b) for b in bindings}
        assert "broadcast" not in marks[("1", "val")]
        assert marks[("1", "tango")]["broadcast"] == 1

    def test_zero_length_list_binds_nothing(self):
        """An empty list beside a populated field binds nothing for itself."""
        body = {
            "DeviceList": [[1, 1], [1, 2]],
            "Monitor": {"ChannelNames": ["A", "B"]},
            "Setpoint": {"ChannelNames": []},
        }
        bindings, groups = bindings_for_family(_view(body), _mapping())
        assert {b.address.field for b in bindings} == {"Monitor"}
        assert [g.field for g in groups] == ["Monitor"]

    def test_partial_list_binds_the_slots_it_has(self):
        """A 2-slot list on a three-device family binds devices 1 and 2."""
        body = {
            "DeviceList": [[1, 1], [1, 2], [1, 3]],
            "Monitor": {"ChannelNames": ["A", "B"]},
        }
        bindings, _ = bindings_for_family(_view(body), _mapping())
        assert [(b.address.device, b.full_pv) for b in bindings] == [("1", "A"), ("2", "B")]
        assert all("broadcast" not in _extras(b) for b in bindings)

    def test_list_longer_than_devices_is_refused(self):
        """A slot with no device to own it is refused rather than dropped."""
        body = {
            "DeviceList": [[1, 1], [1, 2]],
            "Monitor": {"ChannelNames": ["A", "B", "C"]},
        }
        with pytest.raises(ValueError, match="Monitor"):
            bindings_for_family(_view(body), _mapping())


class TestProse:
    """Every binding carries the mapping's field description."""

    def test_description_from_mapping_field(self):
        """``description`` is the mapping's field prose, not the export's."""
        field = {"ChannelNames": ["A", "B"], "Description": "exported text"}
        bindings, _ = bindings_for_family(_view(_two_devices(field)), _mapping())
        assert {b.description for b in bindings} == {"Horizontal position"}

    def test_field_absent_from_mapping_has_no_description(self):
        """A field the mapping does not describe carries no prose."""
        mapping = _mapping(families={"BPMx": _family("BPMx")})
        bindings, _ = bindings_for_family(_view(_two_devices({"ChannelNames": ["A"]})), mapping)
        assert all(b.description is None for b in bindings)


class TestUnitsAndDataType:
    """``HWUnits`` and ``DataType`` ride on a binding only as scalars."""

    @pytest.mark.parametrize("key", ["HWUnits", "DataType"])
    def test_empty_list_emits_nothing(self, key):
        """``[]`` (678 ALS fields) gives no property and no refusal."""
        field = {"ChannelNames": ["A", "B"], key: []}
        bindings, _ = bindings_for_family(_view(_two_devices(field)), _mapping())
        assert all(key not in _extras(b) for b in bindings)

    @pytest.mark.parametrize("key", ["HWUnits", "DataType"])
    def test_scalar_string_rides_on_every_binding(self, key):
        """A non-empty string is copied to every binding of the field."""
        field = {"ChannelNames": ["A", "B"], key: "Amps"}
        bindings, _ = bindings_for_family(_view(_two_devices(field)), _mapping())
        assert [_extras(b)[key] for b in bindings] == ["Amps", "Amps"]

    @pytest.mark.parametrize("key", ["HWUnits", "DataType"])
    def test_per_device_list_is_indexed(self, key):
        """A list of length ``n_devices`` gives slot i to device i."""
        field = {"ChannelNames": ["A", "B"], key: ["A", "B"]}
        bindings, _ = bindings_for_family(_view(_two_devices(field)), _mapping())
        assert [_extras(b)[key] for b in bindings] == ["A", "B"]

    @pytest.mark.parametrize(
        "value",
        ["", "   ", None, 3, ["A"], ["A", "B", "C"], [["A"], ["B"]], {"x": "A"}],
    )
    def test_other_shapes_emit_nothing(self, value):
        """Blank, null, numeric, misaligned and nested values give no property."""
        field = {"ChannelNames": ["A", "B"], "HWUnits": value}
        bindings, _ = bindings_for_family(_view(_two_devices(field)), _mapping())
        assert all("HWUnits" not in _extras(b) for b in bindings)

    def test_blank_slot_in_per_device_list_skips_that_device(self):
        """A blank slot in an aligned list leaves that one binding without the key."""
        field = {"ChannelNames": ["A", "B"], "HWUnits": ["", "mm"]}
        bindings, _ = bindings_for_family(_view(_two_devices(field)), _mapping())
        assert "HWUnits" not in _extras(bindings[0])
        assert _extras(bindings[1])["HWUnits"] == "mm"

    def test_element_list_never_rides_on_bindings(self):
        """``ElementList`` is a device property only."""
        body = {**_two_devices({"ChannelNames": ["A", "B"]}), "ElementList": [4, 7]}
        bindings, _ = bindings_for_family(_view(body), _mapping())
        assert all(set(_extras(b)) <= {"broadcast", "HWUnits", "DataType"} for b in bindings)
        assert all("elementIndex" not in _extras(b) for b in bindings)


class TestSignalGroups:
    """One group per ``(mapped family, field, subfield)``."""

    def test_group_per_field_and_subfield(self):
        """Two fields yield two groups whose members follow binding order."""
        body = {
            "DeviceList": [[1, 1], [1, 2]],
            "Monitor": {"ChannelNames": ["M1", "M2"]},
            "Setpoint": {"ChannelNames": ["S1", "S2"]},
        }
        bindings, groups = bindings_for_family(_view(body), _mapping())
        assert [g.key for g in groups] == [("BPMx", "Monitor", "val"), ("BPMx", "Setpoint", "val")]
        assert groups[0].members == ("M1", "M2")
        assert all(g.direction is None for g in groups)
        assert groups[0].name == "bpmx_monitor_val"


def _ao(**systems) -> dict:
    return {"_import_order": list(systems), "_exports": {}, **systems}


def _bpm_body(prefix: str, n: int = 2) -> dict:
    return {
        "DeviceList": [[1, i + 1] for i in range(n)],
        "Monitor": {"ChannelNames": [f"{prefix}:BPM{i + 1}:X" for i in range(n)]},
    }


class TestBuildGraphModel:
    """The complete model across systems."""

    def _two_system_mapping(self, **overrides) -> Mapping:
        defaults = {
            "systems": {"RING": "SR", "BOOST": "BR"},
            "families": {
                "BPMx": _family("BPMx", fields={"Monitor": "pos"}),
                "HCM": _family("HCM", fields={"Setpoint": "kick"}),
            },
            "directions": {"BPMx.Monitor": "read", "HCM.Setpoint": "write"},
        }
        defaults.update(overrides)
        return _mapping(**defaults)

    def _ao(self) -> dict:
        return _ao(
            BOOST={"BPMx": _bpm_body("BR")},
            RING={
                "BPMx": _bpm_body("SR", 3),
                "HCM": {"DeviceList": [[1, 1]], "Setpoint": {"ChannelNames": ["SR:HCM1:SP"]}},
                "Empty": {"Monitor": {"ChannelNames": []}},
            },
        )

    def test_returns_directed_model(self):
        """Every group carries the mapping's direction, on both subfields."""
        model = build_graph_model(self._ao(), self._two_system_mapping(), ["SR", "BR"])
        assert isinstance(model, GraphModel)
        assert model.facility == FACILITY
        assert {g.key: g.direction for g in model.signal_groups} == {
            ("BPMx", "Monitor", "val"): "read",
            ("HCM", "Setpoint", "val"): "write",
        }

    def test_walks_section_order(self):
        """Devices follow ``section_order``, then family order, then index."""
        model = build_graph_model(self._ao(), self._two_system_mapping(), ["SR", "BR"])
        assert [d.iri.removeprefix(DEVICE_IRI_PREFIX) for d in model.devices] == [
            f"{FACILITY}_SR_BPMx_1",
            f"{FACILITY}_SR_BPMx_2",
            f"{FACILITY}_SR_BPMx_3",
            f"{FACILITY}_SR_HCM_1",
            f"{FACILITY}_BR_BPMx_1",
            f"{FACILITY}_BR_BPMx_2",
        ]
        reordered = build_graph_model(self._ao(), self._two_system_mapping(), ["BR", "SR"])
        assert reordered.devices[0].ring == "BR"

    def test_ordinals_restamped(self):
        """Section and facility ordinals count across families and systems."""
        model = build_graph_model(self._ao(), self._two_system_mapping(), ["SR", "BR"])
        assert [d.ordinal_in_section for d in model.devices] == [1, 2, 3, 4, 1, 2]
        assert [d.ordinal_in_facility for d in model.devices] == [1, 2, 3, 4, 5, 6]

    def test_binding_iris_restamped_onto_devices(self):
        """Each device lists exactly its own bindings' IRIs, in binding order."""
        model = build_graph_model(self._ao(), self._two_system_mapping(), ["SR", "BR"])
        for device in model.devices:
            own = tuple(b.iri for b in model.bindings if b.device_iri == device.iri)
            assert device.binding_iris == own
            assert own
        assert [b.iri for b in model.bindings] == [
            iri for device in model.devices for iri in device.binding_iris
        ]
        assert len(model.bindings) == 6

    def test_groups_merge_across_systems(self):
        """The same mapped family in two systems shares one group."""
        model = build_graph_model(self._ao(), self._two_system_mapping(), ["SR", "BR"])
        groups = model.signal_groups_by_key()
        assert groups[("BPMx", "Monitor", "val")].members == (
            "SR:BPM1:X",
            "SR:BPM2:X",
            "SR:BPM3:X",
            "BR:BPM1:X",
            "BR:BPM2:X",
        )
        assert [g.key for g in model.signal_groups] == sorted(groups)

    def test_rename_translates_direction_keys(self):
        """A raw-keyed direction reaches the group under its renamed token."""
        mapping = self._two_system_mapping(
            families={
                "BPMx": _family("BPMx", rename="BPMX"),
                "HCM": _family("HCM"),
            }
        )
        model = build_graph_model(self._ao(), mapping, ["SR", "BR"])
        assert model.signal_groups_by_key()[("BPMX", "Monitor", "val")].direction == "read"

    def test_dual_key_direction_covers_both_subfields(self):
        """One ``<family>.<field>`` direction directs the val and tango groups."""
        ao = _ao(RING={"BPMx": _two_devices({"ChannelNames": ["A", "B"], "TangoNames": ["t"]})})
        mapping = _mapping(directions={"BPMx.Monitor": "read"})
        model = build_graph_model(ao, mapping, ["SR"])
        assert {g.key: g.direction for g in model.signal_groups} == {
            ("BPMx", "Monitor", "val"): "read",
            ("BPMx", "Monitor", "tango"): "read",
        }

    def test_undirected_group_refused(self):
        """A group whose direction is null names itself in the refusal."""
        mapping = self._two_system_mapping(
            directions={"BPMx.Monitor": "read", "HCM.Setpoint": None}
        )
        with pytest.raises(ValueError, match="HCM.*Setpoint"):
            build_graph_model(self._ao(), mapping, ["SR", "BR"])

    def test_missing_direction_refused(self):
        """A group with no directions key at all is refused."""
        mapping = self._two_system_mapping(directions={"HCM.Setpoint": "write"})
        with pytest.raises(ValueError, match="BPMx.*Monitor"):
            build_graph_model(self._ao(), mapping, ["SR", "BR"])

    def test_duplicate_device_iri_refused(self):
        """Two raw families renamed to one token in one system collide."""
        ao = _ao(RING={"BPMx": _bpm_body("A"), "BPMy": _bpm_body("B")})
        mapping = _mapping(
            families={
                "BPMx": _family("BPMx", rename="BPM"),
                "BPMy": _family("BPMy", rename="BPM"),
            },
            directions={"BPMx.Monitor": "read", "BPMy.Monitor": "read"},
        )
        with pytest.raises(ValueError, match="device IRI"):
            build_graph_model(ao, mapping, ["SR"])

    def test_unknown_section_name_refused(self):
        """A ``section_order`` entry naming no mapped system is refused."""
        with pytest.raises(ValueError, match="XR"):
            build_graph_model(self._ao(), self._two_system_mapping(), ["SR", "BR", "XR"])

    def test_system_outside_section_order_refused(self):
        """An ``ao`` system the order omits is refused rather than dropped."""
        with pytest.raises(ValueError, match="BOOST"):
            build_graph_model(self._ao(), self._two_system_mapping(), ["SR"])

    def test_deterministic(self):
        """Two builds from the same inputs are equal."""
        mapping = self._two_system_mapping()
        assert build_graph_model(self._ao(), mapping, ["SR", "BR"]) == build_graph_model(
            self._ao(), mapping, ["SR", "BR"]
        )

    def test_hwunits_three_shapes_through_model(self):
        """``[]``, ``'Amps'`` and ``['A', 'B']`` all build without an extras refusal."""
        ao = _ao(
            RING={
                "BPMx": {
                    "DeviceList": [[1, 1], [1, 2]],
                    "Monitor": {"ChannelNames": ["M1", "M2"], "HWUnits": []},
                    "Setpoint": {"ChannelNames": ["S1", "S2"], "HWUnits": "Amps"},
                    "Readback": {"ChannelNames": ["R1", "R2"], "HWUnits": ["A", "B"]},
                }
            }
        )
        mapping = _mapping(
            directions={"BPMx.Monitor": "read", "BPMx.Setpoint": "write", "BPMx.Readback": "read"}
        )
        model = build_graph_model(ao, mapping, ["SR"])
        units = {b.full_pv: dict(b.extra_properties).get("HWUnits") for b in model.bindings}
        assert units == {"M1": None, "M2": None, "S1": "Amps", "S2": "Amps", "R1": "A", "R2": "B"}
