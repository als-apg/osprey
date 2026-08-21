"""Tests for the demo-machine graph data model (ttl_generator.model).

Covers the six-token address grammar, the deterministic derivations
(source names, ordinals, IRIs, signal groups) on a small synthetic map, and a
census of the real tier-3 channel database that the generator will run on.
"""

from __future__ import annotations

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
    build_model,
    parse_address,
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

    repo_root = Path(__file__).resolve().parents[3]
    db_path = (
        repo_root
        / "src/osprey/templates/apps/control_assistant/data"
        / "channel_databases/tiers/tier3/hierarchical.json"
    )
    assert db_path.is_file(), f"tier-3 channel database missing at {db_path}"
    return HierarchicalChannelDatabase(str(db_path)).channel_map


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
        """The namespace constants match the shipped ALS corpus."""
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
