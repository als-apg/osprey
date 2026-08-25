"""Tests for the demo-machine graph data model (ttl_generator.model).

Covers the six-token address grammar, the deterministic derivations
(source names, ordinals, IRIs, signal groups) on a small synthetic map, and a
census of the real tier-3 channel database that the generator will run on.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from osprey.services.facility_knowledge.ttl_generator.model import (
    BINDING_IRI_PREFIX,
    CONFIDENCE,
    DEVICE_IRI_PREFIX,
    FACILITY,
    NARAD_SEM_NS,
    PROTOCOL,
    Address,
    HierarchyDescriptions,
    build_model,
    parse_address,
    resolve_hierarchy_descriptions,
)

#: The demo machine the generator actually runs on.
TIER3_DB_PATH = (
    Path(__file__).resolve().parents[3]
    / "src/osprey/templates/apps/control_assistant/data"
    / "channel_databases/tiers/tier3/hierarchical.json"
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

#: Two rings, three families (one of them a gauge-style ``SR01`` device token),
#: and a family that appears in both rings so ordinals have to be ring-scoped.
_SYNTHETIC_ADDRESSES = (
    "SR:MAG:DIPOLE:01:CURRENT:SP",
    "SR:MAG:DIPOLE:01:CURRENT:RB",
    "SR:MAG:DIPOLE:02:CURRENT:SP",
    "SR:MAG:DIPOLE:02:CURRENT:RB",
    "SR:VAC:GAUGE:SR01:PRESSURE:RB",
    "SR:VAC:GAUGE:SR02:PRESSURE:RB",
    "BR:MAG:DIPOLE:01:CURRENT:SP",
    "BR:MAG:DIPOLE:01:CURRENT:RB",
)


@pytest.fixture
def synthetic_map() -> dict[str, dict]:
    """A small expanded channel map in the shape the hierarchical DB produces."""
    return {
        addr: {
            "channel": addr,
            "path": dict(
                zip(
                    ("ring", "system", "family", "device", "field", "subfield"),
                    addr.split(":"),
                    strict=True,
                )
            ),
        }
        for addr in _SYNTHETIC_ADDRESSES
    }


@pytest.fixture(scope="module")
def tier3_channel_map() -> dict[str, dict]:
    """The real tier-3 hierarchical channel database, expanded."""
    from osprey.services.channel_finder.databases.hierarchical import (
        HierarchicalChannelDatabase,
    )

    assert TIER3_DB_PATH.is_file(), f"tier-3 channel database missing at {TIER3_DB_PATH}"
    return HierarchicalChannelDatabase(str(TIER3_DB_PATH)).channel_map


@pytest.fixture(scope="module")
def tier3_raw() -> dict:
    """The raw tier-3 database JSON — the unexpanded tree and its level list."""
    assert TIER3_DB_PATH.is_file(), f"tier-3 channel database missing at {TIER3_DB_PATH}"
    return json.loads(TIER3_DB_PATH.read_text())


@pytest.fixture
def synthetic_levels() -> list[dict[str, str]]:
    """The six-level grammar in the shape the database file spells it."""
    return [
        {"name": "ring", "type": "tree"},
        {"name": "system", "type": "tree"},
        {"name": "family", "type": "tree"},
        {"name": "device", "type": "instances"},
        {"name": "field", "type": "tree"},
        {"name": "subfield", "type": "tree"},
    ]


@pytest.fixture
def synthetic_tree() -> dict:
    """A two-ring tree with prose at every level except DEVICE.

    ``SR:MAG:QF`` carries a description at every level; ``BR`` deliberately
    leaves the system and the subfield undescribed, and re-uses the same
    ``CURRENT/SP`` tokens with different prose.
    """
    return {
        "_comment": "ignored: not a level token",
        "SR": {
            "_description": "Storage Ring",
            "MAG": {
                "_description": "SR magnets",
                "QF": {
                    "_description": "SR focusing quadrupoles",
                    "DEVICE": {
                        "_expansion": {"_type": "range", "_range": [1, 4]},
                        "CURRENT": {
                            "_description": "SR current",
                            "SP": {"_description": "SR setpoint"},
                            "RB": {},
                        },
                    },
                },
            },
        },
        "BR": {
            "_description": "Booster Ring",
            "MAG": {
                "QF": {
                    "_description": "BR focusing quadrupoles",
                    "DEVICE": {
                        "CURRENT": {
                            "_description": "BR current",
                            "SP": {},
                        },
                    },
                },
            },
        },
    }


# ---------------------------------------------------------------------------
# Address grammar
# ---------------------------------------------------------------------------


class TestParseAddress:
    """The six-token colon grammar."""

    def test_parses_a_valid_address(self):
        """Every token lands in its named slot."""
        assert parse_address("SR:MAG:DIPOLE:01:CURRENT:SP") == Address(
            ring="SR",
            system="MAG",
            family="DIPOLE",
            device="01",
            field="CURRENT",
            subfield="SP",
        )

    def test_parses_a_gauge_style_device_token(self):
        """The SR:VAC:GAUGE family uses ``SR01``-style device tokens."""
        parsed = parse_address("SR:VAC:GAUGE:SR01:PRESSURE:RB")
        assert parsed.device == "SR01"
        assert parsed.device_key == ("SR", "VAC", "GAUGE", "SR01")
        assert parsed.signal_key == ("GAUGE", "PRESSURE", "RB")

    def test_text_round_trips(self):
        """``Address.text`` rebuilds the address it was parsed from."""
        addr = "BTS:MAG:QF:03:CURRENT:RB"
        assert parse_address(addr).text == addr

    def test_rejects_too_few_tokens(self):
        """Five tokens is not an address."""
        with pytest.raises(ValueError, match="5 colon-separated token"):
            parse_address("SR:MAG:DIPOLE:01:CURRENT")

    def test_rejects_too_many_tokens(self):
        """Seven tokens is not an address either."""
        with pytest.raises(ValueError, match="7 colon-separated token"):
            parse_address("SR:MAG:DIPOLE:01:CURRENT:SP:EXTRA")

    def test_rejects_an_empty_token(self):
        """An empty token is named in the error message."""
        with pytest.raises(ValueError, match="empty family token"):
            parse_address("SR:MAG::01:CURRENT:SP")

    def test_rejects_an_empty_string(self):
        """The degenerate case still produces a legible error."""
        with pytest.raises(ValueError, match="1 colon-separated token"):
            parse_address("")


# ---------------------------------------------------------------------------
# build_model on a synthetic map
# ---------------------------------------------------------------------------


class TestBuildModelDevices:
    """Device identity, naming and ordinals."""

    def test_one_device_per_identity_tuple(self, synthetic_map):
        """Devices are the distinct (RING, SYSTEM, FAMILY, DEVICE) tuples."""
        model = build_model(synthetic_map)
        assert model.device_addresses() == (
            "SR:MAG:DIPOLE:01",
            "SR:MAG:DIPOLE:02",
            "SR:VAC:GAUGE:SR01",
            "SR:VAC:GAUGE:SR02",
            "BR:MAG:DIPOLE:01",
        )

    def test_source_name_is_family_plus_device_token(self, synthetic_map):
        """FAMILY + DEVICE, including the gauge-style token."""
        by_address = {d.address: d for d in build_model(synthetic_map).devices}
        assert by_address["SR:MAG:DIPOLE:01"].source_name == "DIPOLE01"
        assert by_address["SR:VAC:GAUGE:SR01"].source_name == "GAUGESR01"

    def test_section_code_and_raw_type(self, synthetic_map):
        """sectionCode is the ring token, rawType is the family token."""
        device = build_model(synthetic_map).devices[0]
        assert device.section_code == "SR"
        assert device.raw_type == "DIPOLE"

    def test_ordinals_are_ring_scoped_and_facility_wide(self, synthetic_map):
        """ordinalInSection restarts per ring; ordinalInFacility does not."""
        by_address = {d.address: d for d in build_model(synthetic_map).devices}
        section = {a: d.ordinal_in_section for a, d in by_address.items()}
        facility = {a: d.ordinal_in_facility for a, d in by_address.items()}
        assert section == {
            "SR:MAG:DIPOLE:01": 1,
            "SR:MAG:DIPOLE:02": 2,
            "SR:VAC:GAUGE:SR01": 3,
            "SR:VAC:GAUGE:SR02": 4,
            "BR:MAG:DIPOLE:01": 1,
        }
        assert facility == {
            "SR:MAG:DIPOLE:01": 1,
            "SR:MAG:DIPOLE:02": 2,
            "SR:VAC:GAUGE:SR01": 3,
            "SR:VAC:GAUGE:SR02": 4,
            "BR:MAG:DIPOLE:01": 5,
        }

    def test_s_position_tracks_the_section_ordinal(self, synthetic_map):
        """sPositionM is the in-ring ordinal as a float."""
        for device in build_model(synthetic_map).devices:
            assert device.s_position_m == float(device.ordinal_in_section)
            assert isinstance(device.s_position_m, float)

    def test_device_carries_its_binding_iris(self, synthetic_map):
        """hasBinding targets are attached to the device, in binding order."""
        by_address = {d.address: d for d in build_model(synthetic_map).devices}
        assert by_address["SR:MAG:DIPOLE:01"].binding_iris == (
            f"{BINDING_IRI_PREFIX}narad_endpoint_demo_SR_DIPOLE01_CURRENT_RB",
            f"{BINDING_IRI_PREFIX}narad_endpoint_demo_SR_DIPOLE01_CURRENT_SP",
        )

    def test_natural_device_ordering(self):
        """Device tokens sort numerically, so 2 comes before 10."""
        addrs = [f"SR:VAC:GAUGE:SR{n:02d}:PRESSURE:RB" for n in (10, 2, 1)]
        model = build_model({a: {} for a in addrs})
        assert model.device_addresses() == (
            "SR:VAC:GAUGE:SR01",
            "SR:VAC:GAUGE:SR02",
            "SR:VAC:GAUGE:SR10",
        )


class TestBuildModelIdentifiers:
    """IRIs and NARAD identifier literals, spelled exactly."""

    def test_device_identifiers(self, synthetic_map):
        """Device IRI, deviceId and sourceSectionId follow the NARAD convention."""
        by_address = {d.address: d for d in build_model(synthetic_map).devices}
        device = by_address["SR:MAG:DIPOLE:01"]
        assert device.iri == "https://narad.example.org/device/demo_SR_DIPOLE01"
        assert device.device_id == "narad:device:demo:SR:DIPOLE01"
        assert device.source_section_id == "narad:section:demo:sr"

    def test_gauge_device_identifiers(self, synthetic_map):
        """The gauge-style device token flows through unchanged."""
        by_address = {d.address: d for d in build_model(synthetic_map).devices}
        device = by_address["SR:VAC:GAUGE:SR01"]
        assert device.iri == "https://narad.example.org/device/demo_SR_GAUGESR01"
        assert device.device_id == "narad:device:demo:SR:GAUGESR01"

    def test_binding_identifiers(self, synthetic_map):
        """Binding IRI, bindingId, fullPv, protocol and confidence."""
        by_pv = {b.full_pv: b for b in build_model(synthetic_map).bindings}
        binding = by_pv["SR:MAG:DIPOLE:01:CURRENT:SP"]
        assert binding.iri == (
            "https://narad.example.org/binding/narad_endpoint_demo_SR_DIPOLE01_CURRENT_SP"
        )
        assert binding.binding_id == "narad:binding:narad:endpoint:demo:SR:DIPOLE01:CURRENT_SP"
        assert binding.full_pv == "SR:MAG:DIPOLE:01:CURRENT:SP"
        assert binding.protocol == PROTOCOL == "ca"
        assert binding.confidence == CONFIDENCE == "high"
        assert binding.device_iri == "https://narad.example.org/device/demo_SR_DIPOLE01"

    def test_prefix_constants_are_the_narad_namespaces(self):
        """The namespace constants are the shared NARAD namespaces."""
        assert FACILITY == "demo"
        assert DEVICE_IRI_PREFIX == "https://narad.example.org/device/"
        assert BINDING_IRI_PREFIX == "https://narad.example.org/binding/"
        assert NARAD_SEM_NS == "https://narad.example.org/schema/shared_semantics/"


class TestBuildModelSignalGroups:
    """SemanticSignal individuals minted per (FAMILY, FIELD, SUBFIELD)."""

    def test_groups_and_members(self, synthetic_map):
        """One group per combination, carrying its member addresses."""
        groups = {g.name: g for g in build_model(synthetic_map).signal_groups}
        assert set(groups) == {"dipole_current_sp", "dipole_current_rb", "gauge_pressure_rb"}
        assert groups["dipole_current_sp"].members == (
            "SR:MAG:DIPOLE:01:CURRENT:SP",
            "SR:MAG:DIPOLE:02:CURRENT:SP",
            "BR:MAG:DIPOLE:01:CURRENT:SP",
        )
        assert groups["gauge_pressure_rb"].members == (
            "SR:VAC:GAUGE:SR01:PRESSURE:RB",
            "SR:VAC:GAUGE:SR02:PRESSURE:RB",
        )

    def test_group_is_family_scoped_not_ring_scoped(self, synthetic_map):
        """A family present in two rings shares one signal group."""
        groups = {g.name: g for g in build_model(synthetic_map).signal_groups}
        rings = {m.split(":")[0] for m in groups["dipole_current_sp"].members}
        assert rings == {"SR", "BR"}

    def test_signal_iri_and_key(self, synthetic_map):
        """Signals live in the narad_sem namespace and keep their key."""
        groups = {g.name: g for g in build_model(synthetic_map).signal_groups}
        group = groups["dipole_current_sp"]
        assert group.iri == f"{NARAD_SEM_NS}dipole_current_sp"
        assert group.key == ("DIPOLE", "CURRENT", "SP")

    def test_hyphenated_family_folds_to_underscore(self):
        """``ION-PUMP`` becomes a legal Turtle local name."""
        model = build_model({"SR:VAC:ION-PUMP:01:PRESSURE:RB": {}})
        assert model.signal_groups[0].name == "ion_pump_pressure_rb"
        assert model.signal_groups[0].iri == f"{NARAD_SEM_NS}ion_pump_pressure_rb"

    def test_direction_is_undecided_here(self, synthetic_map):
        """Direction is a later pass's job; the slot starts empty."""
        assert all(g.direction is None for g in build_model(synthetic_map).signal_groups)

    def test_with_directions_returns_a_new_model(self, synthetic_map):
        """A later pass can stamp group-constant directions without mutating."""
        model = build_model(synthetic_map)
        stamped = model.with_directions({("DIPOLE", "CURRENT", "SP"): "writes"})
        assert stamped.signal_groups_by_key()[("DIPOLE", "CURRENT", "SP")].direction == "writes"
        assert stamped.signal_groups_by_key()[("GAUGE", "PRESSURE", "RB")].direction is None
        assert all(g.direction is None for g in model.signal_groups)

    def test_bindings_reference_their_signal(self, synthetic_map):
        """Every binding names the signal group it belongs to."""
        model = build_model(synthetic_map)
        by_key = model.signal_groups_by_key()
        for binding in model.bindings:
            group = by_key[binding.signal_key]
            assert binding.signal_name == group.name
            assert binding.signal_iri == group.iri
            assert binding.full_pv in group.members


class TestBuildModelDeterminism:
    """Same input, same output — regardless of dict order."""

    def test_shuffled_input_produces_an_identical_model(self, synthetic_map):
        """Ordinals, ordering and every derived field are insertion-order free."""
        baseline = build_model(synthetic_map)
        for seed in range(5):
            items = list(synthetic_map.items())
            random.Random(seed).shuffle(items)
            assert build_model(dict(items)) == baseline

    def test_binding_count_matches_the_input(self, synthetic_map):
        """Exactly one binding per channel address."""
        assert len(build_model(synthetic_map).bindings) == len(synthetic_map)

    def test_invalid_address_is_rejected(self):
        """A malformed key fails the whole build rather than being skipped."""
        with pytest.raises(ValueError, match="colon-separated token"):
            build_model({"SR:MAG:DIPOLE:01:CURRENT:SP": {}, "NOT-AN-ADDRESS": {}})


# ---------------------------------------------------------------------------
# The real tier-3 database
# ---------------------------------------------------------------------------


class TestRealTier3Database:
    """Census of the demo machine the generator actually runs on."""

    def test_model_census(self, tier3_channel_map):
        """2908 bindings, 512 devices, 28 ring/system/family combos, 3 rings."""
        model = build_model(tier3_channel_map)

        assert len(model.bindings) == 2908
        assert len(model.bindings) == len(tier3_channel_map)

        expected_devices = {tuple(addr.split(":")[:4]) for addr in tier3_channel_map}
        assert len(model.devices) == len(expected_devices)
        assert {d.key for d in model.devices} == expected_devices

        combos = {(d.ring, d.system, d.family) for d in model.devices}
        assert len(combos) == 28

        assert {d.ring for d in model.devices} == {"SR", "BR", "BTS"}

    def test_ordinals_are_dense_per_ring(self, tier3_channel_map):
        """Every ring's ordinals run 1..N with no gaps or repeats."""
        model = build_model(tier3_channel_map)
        per_ring: dict[str, list[int]] = {}
        for device in model.devices:
            per_ring.setdefault(device.ring, []).append(device.ordinal_in_section)
        for ring, ordinals in per_ring.items():
            assert sorted(ordinals) == list(range(1, len(ordinals) + 1)), ring

    def test_facility_ordinals_are_dense_and_ring_ordered(self, tier3_channel_map):
        """Facility ordinals run 1..N in SR, BR, BTS order."""
        model = build_model(tier3_channel_map)
        assert [d.ordinal_in_facility for d in model.devices] == list(
            range(1, len(model.devices) + 1)
        )
        ring_first_seen = []
        for device in model.devices:
            if device.ring not in ring_first_seen:
                ring_first_seen.append(device.ring)
        assert ring_first_seen == ["SR", "BR", "BTS"]

    def test_identifiers_are_unique(self, tier3_channel_map):
        """No two devices or bindings collide on an IRI or an id literal."""
        model = build_model(tier3_channel_map)
        assert len({d.iri for d in model.devices}) == len(model.devices)
        assert len({d.device_id for d in model.devices}) == len(model.devices)
        assert len({b.iri for b in model.bindings}) == len(model.bindings)
        assert len({b.binding_id for b in model.bindings}) == len(model.bindings)

    def test_every_binding_belongs_to_a_device_and_a_group(self, tier3_channel_map):
        """The three populations are mutually consistent."""
        model = build_model(tier3_channel_map)
        device_iris = {d.iri for d in model.devices}
        group_keys = {g.key for g in model.signal_groups}
        assert {b.device_iri for b in model.bindings} == device_iris
        assert {b.signal_key for b in model.bindings} == group_keys
        total_members = sum(len(g.members) for g in model.signal_groups)
        assert total_members == len(model.bindings)
        total_device_bindings = sum(len(d.binding_iris) for d in model.devices)
        assert total_device_bindings == len(model.bindings)

    def test_gauge_family_keeps_its_device_tokens(self, tier3_channel_map):
        """SR:VAC:GAUGE uses SR01..SR12, so its source names read GAUGESR01."""
        model = build_model(tier3_channel_map)
        gauges = [d for d in model.devices if d.family == "GAUGE"]
        assert len(gauges) == 12
        assert {d.source_name for d in gauges} == {f"GAUGESR{n:02d}" for n in range(1, 13)}


# ---------------------------------------------------------------------------
# Hierarchy descriptions
# ---------------------------------------------------------------------------


class TestResolveHierarchyDescriptions:
    """The prose the raw hierarchical tree carries, keyed by level path."""

    def test_key_shapes_are_the_level_path(self, synthetic_tree, synthetic_levels):
        """Each map is keyed by the path down to its own level, DEVICE excluded."""
        descriptions = resolve_hierarchy_descriptions(synthetic_tree, synthetic_levels)

        assert descriptions.ring == {
            ("SR",): "Storage Ring",
            ("BR",): "Booster Ring",
        }
        assert descriptions.system == {("SR", "MAG"): "SR magnets"}
        assert descriptions.family == {
            ("SR", "MAG", "QF"): "SR focusing quadrupoles",
            ("BR", "MAG", "QF"): "BR focusing quadrupoles",
        }
        assert descriptions.field == {
            ("SR", "MAG", "QF", "CURRENT"): "SR current",
            ("BR", "MAG", "QF", "CURRENT"): "BR current",
        }
        assert descriptions.subfield == {("SR", "MAG", "QF", "CURRENT", "SP"): "SR setpoint"}

    def test_missing_descriptions_are_simply_absent(self, synthetic_tree, synthetic_levels):
        """A node without ``_description`` contributes no key at all."""
        descriptions = resolve_hierarchy_descriptions(synthetic_tree, synthetic_levels)

        assert ("BR", "MAG") not in descriptions.system
        assert ("SR", "MAG", "QF", "CURRENT", "RB") not in descriptions.subfield
        assert ("BR", "MAG", "QF", "CURRENT", "SP") not in descriptions.subfield

    def test_underscore_keys_are_never_tokens(self, synthetic_tree, synthetic_levels):
        """``_comment``, ``_description`` and ``_expansion`` are metadata, not levels."""
        descriptions = resolve_hierarchy_descriptions(synthetic_tree, synthetic_levels)

        every_token = {
            token
            for level in (
                descriptions.ring,
                descriptions.system,
                descriptions.family,
                descriptions.field,
                descriptions.subfield,
            )
            for key in level
            for token in key
        }
        assert not any(token.startswith("_") for token in every_token)
        assert "DEVICE" not in every_token

    def test_blank_and_non_string_prose_is_dropped(self, synthetic_levels):
        """Only non-empty text counts; surrounding whitespace is trimmed."""
        tree = {
            "SR": {"_description": "  Storage Ring  "},
            "BR": {"_description": "   "},
            "BTS": {"_description": None},
        }
        descriptions = resolve_hierarchy_descriptions(tree, synthetic_levels)

        assert descriptions.ring == {("SR",): "Storage Ring"}

    def test_empty_tree_gives_five_empty_maps(self, synthetic_levels):
        """No tree, no prose — and no exception."""
        descriptions = resolve_hierarchy_descriptions({}, synthetic_levels)

        assert descriptions == HierarchyDescriptions(
            ring={}, system={}, family={}, field={}, subfield={}
        )

    def test_keys_are_sorted(self, synthetic_tree, synthetic_levels):
        """Iteration order is the sorted key order, not the JSON's."""
        descriptions = resolve_hierarchy_descriptions(synthetic_tree, synthetic_levels)

        assert list(descriptions.ring) == [("BR",), ("SR",)]
        assert list(descriptions.family) == [("BR", "MAG", "QF"), ("SR", "MAG", "QF")]

    def test_is_pure_and_repeatable(self, synthetic_tree, synthetic_levels):
        """Two runs agree, and neither one touches the input tree."""
        before = json.dumps(synthetic_tree, sort_keys=True)
        first = resolve_hierarchy_descriptions(synthetic_tree, synthetic_levels)
        second = resolve_hierarchy_descriptions(synthetic_tree, synthetic_levels)

        assert first == second
        assert json.dumps(synthetic_tree, sort_keys=True) == before

    def test_rejects_a_different_level_grammar(self, synthetic_tree, synthetic_levels):
        """The five maps name the six-token grammar, so a foreign one is refused."""
        renamed = [dict(level) for level in synthetic_levels]
        renamed[1]["name"] = "subsystem"

        with pytest.raises(ValueError, match="level"):
            resolve_hierarchy_descriptions(synthetic_tree, renamed)

    def test_rejects_a_device_level_that_is_not_instances(self, synthetic_tree, synthetic_levels):
        """A tree-typed DEVICE level would put device tokens in the keys."""
        retyped = [dict(level) for level in synthetic_levels]
        retyped[3]["type"] = "tree"

        with pytest.raises(ValueError, match="level"):
            resolve_hierarchy_descriptions(synthetic_tree, retyped)


class TestHierarchyDescriptionsOnTier3:
    """The real demo machine's 247 descriptions."""

    @pytest.fixture(scope="class")
    def descriptions(self, tier3_raw) -> HierarchyDescriptions:
        return resolve_hierarchy_descriptions(tier3_raw["tree"], tier3_raw["hierarchy"]["levels"])

    def test_census(self, descriptions):
        """3 rings, 8 systems, 28 families, 66 fields, 142 subfields."""
        assert len(descriptions.ring) == 3
        assert len(descriptions.system) == 8
        assert len(descriptions.family) == 28
        assert len(descriptions.field) == 66
        assert len(descriptions.subfield) == 142

    def test_every_description_in_the_file_is_accounted_for(self, tier3_raw, descriptions):
        """Nothing is dropped: the five maps hold every ``_description`` there is."""

        def count(node) -> int:
            total = 0
            for key, value in node.items():
                if key == "_description":
                    total += 1
                elif isinstance(value, dict):
                    total += count(value)
            return total

        resolved = sum(
            len(level)
            for level in (
                descriptions.ring,
                descriptions.system,
                descriptions.family,
                descriptions.field,
                descriptions.subfield,
            )
        )
        assert resolved == count(tier3_raw["tree"]) == 247

    def test_the_same_subfield_reads_differently_per_ring(self, descriptions):
        """QF CURRENT SP means one thing in the storage ring and another in the booster.

        This collision is why the prose has to hang off the binding-level path
        and not off the ``(FAMILY, FIELD, SUBFIELD)`` signal group.
        """
        storage_ring = descriptions.subfield[("SR", "MAG", "QF", "CURRENT", "SP")]
        booster = descriptions.subfield[("BR", "MAG", "QF", "CURRENT", "SP")]

        assert storage_ring != booster
        assert storage_ring and booster

    def test_the_device_level_carries_no_prose(self, descriptions):
        """DEVICE nodes hold ``_expansion`` only, so no key ever runs six tokens."""
        assert all(len(key) == 5 for key in descriptions.subfield)
        assert all(len(key) == 4 for key in descriptions.field)

    def test_keys_describe_addresses_the_machine_really_has(self, descriptions, tier3_channel_map):
        """Every resolved path is a prefix of at least one expanded channel."""
        addresses = [tuple(addr.split(":")) for addr in tier3_channel_map]
        ring_system_family_field_subfield = {
            (ring, system, family, field, subfield)
            for ring, system, family, _device, field, subfield in addresses
        }
        for width, level in (
            (1, descriptions.ring),
            (2, descriptions.system),
            (3, descriptions.family),
        ):
            prefixes = {address[:width] for address in ring_system_family_field_subfield}
            assert set(level) <= prefixes

        field_paths = {(r, s, f, fl) for r, s, f, fl, _sf in ring_system_family_field_subfield}
        assert set(descriptions.field) <= field_paths
        assert set(descriptions.subfield) <= ring_system_family_field_subfield


# ---------------------------------------------------------------------------
# Description attachment
# ---------------------------------------------------------------------------


class TestBuildModelDescriptions:
    """Prose reaches the model by exact join, or not at all."""

    @pytest.fixture
    def described_map(self) -> dict[str, dict]:
        """A channel map whose addresses live in ``synthetic_tree``.

        ``SR:MAG:QF:01:CURRENT:SP`` is described at every level; ``CURRENT:RB``
        has no subfield prose and ``BR`` has no system prose, so the partial
        cases are covered too.
        """
        return {
            addr: {"channel": addr}
            for addr in (
                "SR:MAG:QF:01:CURRENT:SP",
                "SR:MAG:QF:01:CURRENT:RB",
                "BR:MAG:QF:01:CURRENT:SP",
            )
        }

    @pytest.fixture
    def hierarchy(self, synthetic_tree, synthetic_levels) -> HierarchyDescriptions:
        return resolve_hierarchy_descriptions(synthetic_tree, synthetic_levels)

    def test_no_inputs_leaves_every_field_none(self, described_map):
        """Keys-only behaviour is preserved when both kwargs are omitted."""
        model = build_model(described_map)

        for binding in model.bindings:
            assert binding.description is None
            assert binding.field_description is None
            assert binding.subfield_description is None
        for device in model.devices:
            assert device.family_description is None
            assert device.system_description is None
            assert device.ring_description is None

    def test_explicit_none_matches_omitting_the_inputs(self, described_map):
        """Passing ``None`` explicitly builds exactly the keys-only model."""
        assert build_model(
            described_map, hierarchy_descriptions=None, binding_descriptions=None
        ) == build_model(described_map)

    def test_bindings_carry_their_own_three_texts(self, described_map, hierarchy):
        """Binding prose comes from the address; field/subfield from its path."""
        model = build_model(
            described_map,
            hierarchy_descriptions=hierarchy,
            binding_descriptions={"SR:MAG:QF:01:CURRENT:SP": "Focusing quad 1 setpoint"},
        )
        by_pv = {binding.full_pv: binding for binding in model.bindings}

        described = by_pv["SR:MAG:QF:01:CURRENT:SP"]
        assert described.description == "Focusing quad 1 setpoint"
        assert described.field_description == "SR current"
        assert described.subfield_description == "SR setpoint"

    def test_a_missing_key_leaves_none_rather_than_falling_back(self, described_map, hierarchy):
        """Exact joins only: no tie-break, no borrowing a sibling's prose."""
        model = build_model(
            described_map,
            hierarchy_descriptions=hierarchy,
            binding_descriptions={"SR:MAG:QF:01:CURRENT:SP": "Focusing quad 1 setpoint"},
        )
        by_pv = {binding.full_pv: binding for binding in model.bindings}

        readback = by_pv["SR:MAG:QF:01:CURRENT:RB"]
        assert readback.description is None  # absent from binding_descriptions
        assert readback.field_description == "SR current"
        assert readback.subfield_description is None  # RB carries no prose

    def test_the_join_is_ring_qualified(self, described_map, hierarchy):
        """The same FIELD token reads differently in each ring."""
        model = build_model(described_map, hierarchy_descriptions=hierarchy)
        by_pv = {binding.full_pv: binding for binding in model.bindings}

        assert by_pv["SR:MAG:QF:01:CURRENT:SP"].field_description == "SR current"
        assert by_pv["BR:MAG:QF:01:CURRENT:SP"].field_description == "BR current"

    def test_devices_carry_family_system_and_ring_texts(self, described_map, hierarchy):
        """Each (RING, SYSTEM, FAMILY) path is unique, so device prose resolves exactly."""
        model = build_model(described_map, hierarchy_descriptions=hierarchy)
        by_address = {device.address: device for device in model.devices}

        storage_ring = by_address["SR:MAG:QF:01"]
        assert storage_ring.family_description == "SR focusing quadrupoles"
        assert storage_ring.system_description == "SR magnets"
        assert storage_ring.ring_description == "Storage Ring"

        booster = by_address["BR:MAG:QF:01"]
        assert booster.family_description == "BR focusing quadrupoles"
        assert booster.system_description is None  # BR:MAG carries no prose
        assert booster.ring_description == "Booster Ring"

    def test_blank_prose_is_not_a_description(self, described_map, hierarchy):
        """Whitespace-only binding text is dropped, and text is trimmed."""
        model = build_model(
            described_map,
            hierarchy_descriptions=hierarchy,
            binding_descriptions={
                "SR:MAG:QF:01:CURRENT:SP": "   ",
                "SR:MAG:QF:01:CURRENT:RB": "  Focusing quad 1 readback  ",
            },
        )
        by_pv = {binding.full_pv: binding for binding in model.bindings}

        assert by_pv["SR:MAG:QF:01:CURRENT:SP"].description is None
        assert by_pv["SR:MAG:QF:01:CURRENT:RB"].description == "Focusing quad 1 readback"

    def test_descriptions_do_not_disturb_the_ordering(self, described_map, hierarchy):
        """Prose is carried, never sorted on."""
        plain = build_model(described_map)
        described = build_model(
            described_map,
            hierarchy_descriptions=hierarchy,
            binding_descriptions={"SR:MAG:QF:01:CURRENT:SP": "Focusing quad 1 setpoint"},
        )

        assert described.binding_addresses() == plain.binding_addresses()
        assert described.device_addresses() == plain.device_addresses()
        assert [g.key for g in described.signal_groups] == [g.key for g in plain.signal_groups]

    def test_signal_groups_carry_no_prose(self, described_map, hierarchy):
        """SemanticSignal is keyed without a ring, so it takes no ring-qualified text."""
        model = build_model(
            described_map,
            hierarchy_descriptions=hierarchy,
            binding_descriptions={"SR:MAG:QF:01:CURRENT:SP": "Focusing quad 1 setpoint"},
        )
        group = model.signal_groups[0]

        assert not [name for name in vars(group) if "description" in name]


class TestDescriptionsOnTier3:
    """The demo machine resolves every text it has, on every node that takes one."""

    @pytest.fixture(scope="class")
    def hierarchy(self, tier3_raw) -> HierarchyDescriptions:
        return resolve_hierarchy_descriptions(tier3_raw["tree"], tier3_raw["hierarchy"]["levels"])

    @pytest.fixture(scope="class")
    def binding_texts(self) -> dict[str, str]:
        """The per-channel prose from the tier-3 ``in_context`` database."""
        path = TIER3_DB_PATH.with_name("in_context.json")
        assert path.is_file(), f"tier-3 description database missing at {path}"
        payload = json.loads(path.read_text())
        return {entry["address"]: entry["description"] for entry in payload["channels"]}

    @pytest.fixture(scope="class")
    def model(self, tier3_channel_map, hierarchy, binding_texts):
        return build_model(
            tier3_channel_map,
            hierarchy_descriptions=hierarchy,
            binding_descriptions=binding_texts,
        )

    def test_every_binding_resolves_all_three_texts(self, model):
        """2908 bindings, none of them missing prose."""
        assert len(model.bindings) == 2908
        assert all(binding.description for binding in model.bindings)
        assert all(binding.field_description for binding in model.bindings)
        assert all(binding.subfield_description for binding in model.bindings)

    def test_every_device_resolves_all_three_texts(self, model):
        """512 devices, none of them missing prose."""
        assert len(model.devices) == 512
        assert all(device.family_description for device in model.devices)
        assert all(device.system_description for device in model.devices)
        assert all(device.ring_description for device in model.devices)

    def test_binding_prose_is_the_in_context_text_verbatim(self, model, binding_texts):
        """The per-binding join is on the full six-token address."""
        assert {b.full_pv: b.description for b in model.bindings} == binding_texts

    def test_device_texts_have_the_expected_spread(self, model):
        """28 family, 8 system and 3 ring texts across the device population."""
        assert len({device.family_description for device in model.devices}) == 28
        assert len({device.system_description for device in model.devices}) == 8
        assert len({device.ring_description for device in model.devices}) == 3

    def test_no_hierarchy_text_is_left_behind(self, model, hierarchy):
        """All 247 resolved strings reach a node: three levels on devices, two on bindings."""
        assert {d.ring_description for d in model.devices} == set(hierarchy.ring.values())
        assert {d.system_description for d in model.devices} == set(hierarchy.system.values())
        assert {d.family_description for d in model.devices} == set(hierarchy.family.values())
        assert {b.field_description for b in model.bindings} == set(hierarchy.field.values())
        assert {b.subfield_description for b in model.bindings} == set(hierarchy.subfield.values())

    def test_the_ring_collision_survives_on_the_binding(self, model):
        """QF CURRENT SP keeps one text per ring — the reason prose hangs off bindings."""
        texts = {
            binding.full_pv: binding.subfield_description
            for binding in model.bindings
            if binding.address.family == "QF"
            and binding.address.field == "CURRENT"
            and binding.address.subfield == "SP"
        }
        storage_ring = texts["SR:MAG:QF:01:CURRENT:SP"]
        booster = texts["BR:MAG:QF:01:CURRENT:SP"]

        assert storage_ring != booster
