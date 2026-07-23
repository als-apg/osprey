"""Drift gates binding the committed tier-DB artifacts to the canonical
generator and the declarative spec.

These are the standing regression guards for the one-facility tier-DB
unification: they fail loudly the moment a committed channel-database file
drifts from what ``generate_from_spec`` would produce, or from the
``ALS_U_AR`` + ``CHANNEL_SCHEMA`` declaration the generator is grown from.

Four separable assertions (PROPOSAL.md rev 4, Goal 2):

* (a) **Idempotency** -- running the documented regeneration command over the
  committed artifacts is a byte-identical no-op (tier 3 x3 paradigms AND the
  derived tier-1 in_context).
* (b) **Cross-paradigm identity** -- the three committed tier-3 paradigms
  expose one identical address set.
* (c) **Tier-1 is the declared filter of tier 3** -- committed
  ``tier1/in_context.json`` equals ``apply_tier1_filter(tier3)``
  entry-for-entry.
* (d) **Spec-family expansion equality** -- committed tier-3 leaves with ring
  ``SR`` AND a family declared in ``ALS_U_AR`` equal the from-scratch
  ``CHANNEL_SCHEMA`` expansion, both in address set and in per-leaf
  ``DataType``/``HWUnits``.

BR/BTS rings and non-spec SR systems (``RF``/``VAC``/``DIAG:DCCT`` ...) are
hand-authored content: they are covered by (a)-(c) only, never by (d).

Pattern follows ``tests/templates/test_machine_json_lattice.py``: derive every
expectation from the spec, never pin a raw channel-count literal.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from osprey.services.channel_finder.databases.hierarchical import HierarchicalChannelDatabase
from osprey.services.channel_finder.databases.middle_layer import MiddleLayerDatabase
from osprey.services.channel_finder.databases.template import ChannelDatabase
from osprey.services.channel_finder.tools.generate_from_spec import (
    TIER3_FILENAMES,
    address_set,
    apply_tier1_filter,
    generate,
    verify_cross_paradigm_identity,
)
from osprey.simulation.channel_schema import CHANNEL_SCHEMA
from osprey.simulation.facility_spec import ALS_U_AR

REPO_ROOT = Path(__file__).resolve().parents[3]
_CHANNEL_DB_DIR = REPO_ROOT / "src/osprey/templates/apps/control_assistant/data/channel_databases"
COMMITTED_TIER3_DIR = _CHANNEL_DB_DIR / "tiers/tier3"
COMMITTED_TIER1_DIR = _CHANNEL_DB_DIR / "tiers/tier1"

RING = "SR"
SPEC_FAMILIES = frozenset(f.name for f in ALS_U_AR.families)


def _load_channels(path: Path) -> list[dict[str, str]]:
    return json.loads(path.read_text())["channels"]


# ---------------------------------------------------------------------------
# (a) Idempotency: regenerating the committed artifacts changes nothing.
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_regenerating_committed_artifacts_is_a_byte_identical_noop(self, tmp_path: Path):
        # Mirror the committed ``tiers/{tier1,tier3}`` layout, regenerate, and
        # assert every file came back byte-for-byte -- generate() over the
        # committed tree must be a fixpoint.
        tiers = tmp_path / "tiers"
        shutil.copytree(COMMITTED_TIER3_DIR, tiers / "tier3")
        shutil.copytree(COMMITTED_TIER1_DIR, tiers / "tier1")

        generate(tiers / "tier3")

        committed_root = COMMITTED_TIER3_DIR.parent
        regenerated = sorted(tiers.rglob("*.json"))
        assert regenerated, "no regenerated files found"
        for produced in regenerated:
            rel = produced.relative_to(tiers)
            committed = (committed_root / rel).read_bytes()
            assert produced.read_bytes() == committed, f"{rel} drifted from committed bytes"


# ---------------------------------------------------------------------------
# (b) Cross-paradigm identity of the committed tier-3 databases.
# ---------------------------------------------------------------------------


class TestCrossParadigmIdentity:
    def test_committed_tier3_paradigms_share_one_address_set(self):
        verify_cross_paradigm_identity(
            COMMITTED_TIER3_DIR / TIER3_FILENAMES["hierarchical"],
            COMMITTED_TIER3_DIR / TIER3_FILENAMES["in_context"],
            COMMITTED_TIER3_DIR / TIER3_FILENAMES["middle_layer"],
        )

    def test_address_sets_are_actually_equal_and_nonempty(self):
        hier = address_set(
            COMMITTED_TIER3_DIR / TIER3_FILENAMES["hierarchical"], HierarchicalChannelDatabase
        )
        ctx = address_set(COMMITTED_TIER3_DIR / TIER3_FILENAMES["in_context"], ChannelDatabase)
        ml = address_set(COMMITTED_TIER3_DIR / TIER3_FILENAMES["middle_layer"], MiddleLayerDatabase)
        assert hier == ctx == ml
        assert len(hier) > 0


# ---------------------------------------------------------------------------
# (c) Committed tier 1 is exactly the declared filter of committed tier 3.
# ---------------------------------------------------------------------------


class TestTier1IsFilterOfTier3:
    def test_committed_tier1_equals_filter_of_committed_tier3(self):
        tier3_channels = _load_channels(COMMITTED_TIER3_DIR / "in_context.json")
        committed_tier1 = _load_channels(COMMITTED_TIER1_DIR / "in_context.json")

        filtered = apply_tier1_filter(tier3_channels)

        # Count is derived from both sides, never pinned.
        assert len(filtered) == len(committed_tier1)
        assert {c["address"]: c for c in filtered} == {c["address"]: c for c in committed_tier1}


# ---------------------------------------------------------------------------
# (d) SR spec-family tier-3 leaves == from-scratch CHANNEL_SCHEMA expansion.
# ---------------------------------------------------------------------------


def _from_scratch_spec_leaves() -> dict[str, tuple[str, str]]:
    """Expand ``ALS_U_AR`` + ``CHANNEL_SCHEMA`` from scratch to the expected
    ``{address: (DataType, HWUnits)}`` for every SR spec-family leaf."""
    expected: dict[str, tuple[str, str]] = {}
    for family in ALS_U_AR.families:
        schema = CHANNEL_SCHEMA[family.name]
        for device_id in range(1, family.count + 1):
            for field, subs in schema.fields.items():
                for subfield, subschema in subs.items():
                    address = (
                        f"{RING}:{schema.system}:{family.name}:{device_id:02d}:{field}:{subfield}"
                    )
                    expected[address] = (subschema.data_type, subschema.hw_units)
    return expected


def _committed_sr_spec_leaves() -> dict[str, tuple[str, str]]:
    """Committed tier-3 ``{address: (DataType, HWUnits)}`` restricted to ring
    ``SR`` AND a spec-declared family (via the middle_layer paradigm, which
    carries the per-leaf metadata)."""
    ml = MiddleLayerDatabase(str(COMMITTED_TIER3_DIR / "middle_layer.json"))
    leaves: dict[str, tuple[str, str]] = {}
    for channel in ml.get_all_channels():
        address = channel["address"]
        parts = address.split(":")
        if parts[0] == RING and parts[2] in SPEC_FAMILIES:
            leaves[address] = (channel["DataType"], channel["HWUnits"])
    return leaves


class TestSpecFamilyExpansionEquality:
    def test_committed_sr_spec_addresses_equal_from_scratch_expansion(self):
        expected = _from_scratch_spec_leaves()
        committed = _committed_sr_spec_leaves()
        assert set(committed) == set(expected)

    def test_committed_sr_spec_leaf_metadata_matches_schema(self):
        expected = _from_scratch_spec_leaves()
        committed = _committed_sr_spec_leaves()
        # Per-leaf DataType/HWUnits agreement (not just the address set).
        assert committed == expected

    def test_expansion_covers_only_spec_families_on_the_sr_ring(self):
        # Guard the (d) scope itself: BR/BTS and non-spec SR families must be
        # excluded, so the from-scratch set never leaks a hand-authored family.
        committed = _committed_sr_spec_leaves()
        families = {addr.split(":")[2] for addr in committed}
        assert families == SPEC_FAMILIES
        assert all(addr.split(":")[0] == RING for addr in committed)
