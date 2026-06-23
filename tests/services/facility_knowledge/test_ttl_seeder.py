"""Tests for the TTL-based OKF stub seeder.

All tests that actually invoke rdflib are guarded by ``pytest.importorskip``
so the suite remains green when the ``knowledge`` extra is not installed.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Canonical als-ontology TTL path relative to the repository root.
_ALS_GTB_TTL = (
    Path(__file__).parent.parent.parent.parent  # repo root
    / "../../../../als-ontology/data/rdf/als_gtb.ttl"
).resolve()

# Minimal synthetic TTL for unit tests (does not require the als-ontology repo).
_MINI_TTL = textwrap.dedent("""\
    @prefix narad_p: <https://narad.example.org/property/> .
    @prefix narad_sem: <https://narad.example.org/schema/shared_semantics/> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

    <https://narad.example.org/device/tst_SEC_QF1> a narad_sem:Quadrupole ;
        narad_p:deviceId "narad:device:tst:SEC:QF1" ;
        narad_p:sectionCode "SEC" ;
        narad_p:sourceName "QF1" ;
        narad_p:hasBinding
            <https://narad.example.org/binding/tst_SEC_QF1_Monitor>,
            <https://narad.example.org/binding/tst_SEC_QF1_Setpoint> .

    <https://narad.example.org/device/tst_SEC_QD1> a narad_sem:Quadrupole ;
        narad_p:deviceId "narad:device:tst:SEC:QD1" ;
        narad_p:sectionCode "SEC" ;
        narad_p:sourceName "QD1" .

    <https://narad.example.org/binding/tst_SEC_QF1_Monitor> a narad_sem:ChannelBinding ;
        narad_p:fullPv "SEC:QF1:CurrentRBV" ;
        narad_p:protocol "ca" ;
        narad_p:confidence "high" ;
        narad_p:readsSignal narad_sem:current_readback .

    <https://narad.example.org/binding/tst_SEC_QF1_Setpoint> a narad_sem:ChannelBinding ;
        narad_p:fullPv "SEC:QF1:Setpoint" ;
        narad_p:protocol "ca" ;
        narad_p:confidence "high" ;
        narad_p:writesSignal narad_sem:current_setpoint .
""")


@pytest.fixture
def mini_ttl(tmp_path: Path) -> Path:
    """Write the synthetic TTL to a temp file and return its path."""
    p = tmp_path / "mini.ttl"
    p.write_text(_MINI_TTL, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Import-level guard — no rdflib → all rdflib-touching tests skip
# ---------------------------------------------------------------------------


def _require_rdflib():
    """Skip current test if rdflib is not installed."""
    return pytest.importorskip("rdflib", reason="rdflib not installed (knowledge extra required)")


# ---------------------------------------------------------------------------
# Module-import tests (no rdflib required)
# ---------------------------------------------------------------------------


class TestModuleImport:
    """Importing the seeder must never trigger rdflib at module level."""

    def test_seeder_importable_without_rdflib(self):
        """The seeder package is importable regardless of whether rdflib is installed."""
        from osprey.services.facility_knowledge.seeder import (  # noqa: F401
            DeviceStub,
            seed_from_ttl,
        )

    def test_seed_from_ttl_returns_empty_for_none(self):
        """seed_from_ttl(None) returns [] without touching rdflib."""
        from osprey.services.facility_knowledge.seeder import seed_from_ttl

        result = seed_from_ttl(None)
        assert result == []

    def test_seed_from_ttl_returns_empty_for_empty_string(self):
        """seed_from_ttl('') returns [] without touching rdflib."""
        from osprey.services.facility_knowledge.seeder import seed_from_ttl

        result = seed_from_ttl("")
        assert result == []


# ---------------------------------------------------------------------------
# Synthetic-TTL unit tests (require rdflib)
# ---------------------------------------------------------------------------


class TestSeedFromTTLSynthetic:
    """Unit tests using a small hand-crafted TTL — no dependency on als-ontology."""

    def test_returns_two_stubs(self, mini_ttl):
        _require_rdflib()
        from osprey.services.facility_knowledge.seeder import seed_from_ttl

        stubs = seed_from_ttl(mini_ttl)
        assert len(stubs) == 2

    def test_resource_iri_verbatim(self, mini_ttl):
        _require_rdflib()
        from osprey.services.facility_knowledge.seeder import seed_from_ttl

        stubs = seed_from_ttl(mini_ttl)
        iris = {s.resource for s in stubs}
        assert "https://narad.example.org/device/tst_SEC_QF1" in iris
        assert "https://narad.example.org/device/tst_SEC_QD1" in iris

    def test_device_class_is_leaf_class(self, mini_ttl):
        _require_rdflib()
        from osprey.services.facility_knowledge.seeder import seed_from_ttl

        stubs = seed_from_ttl(mini_ttl)
        for stub in stubs:
            assert stub.device_class == "Quadrupole"

    def test_title_uses_section_source_and_class(self, mini_ttl):
        _require_rdflib()
        from osprey.services.facility_knowledge.seeder import seed_from_ttl

        stubs = {s.resource: s for s in seed_from_ttl(mini_ttl)}
        qf1 = stubs["https://narad.example.org/device/tst_SEC_QF1"]
        assert qf1.title == "SEC:QF1 (Quadrupole)"

    def test_channels_sorted_by_name(self, mini_ttl):
        _require_rdflib()
        from osprey.services.facility_knowledge.seeder import seed_from_ttl

        stubs = {s.resource: s for s in seed_from_ttl(mini_ttl)}
        qf1 = stubs["https://narad.example.org/device/tst_SEC_QF1"]
        assert len(qf1.channels) == 2
        names = [c.channel for c in qf1.channels]
        assert names == sorted(names)

    def test_channel_pv_and_direction(self, mini_ttl):
        _require_rdflib()
        from osprey.services.facility_knowledge.seeder import seed_from_ttl

        stubs = {s.resource: s for s in seed_from_ttl(mini_ttl)}
        qf1 = stubs["https://narad.example.org/device/tst_SEC_QF1"]
        ch = {c.channel: c for c in qf1.channels}

        assert ch["Monitor"].pv == "SEC:QF1:CurrentRBV"
        assert ch["Monitor"].direction == "reads"
        assert ch["Monitor"].signal == "current_readback"

        assert ch["Setpoint"].pv == "SEC:QF1:Setpoint"
        assert ch["Setpoint"].direction == "writes"
        assert ch["Setpoint"].signal == "current_setpoint"

    def test_device_with_no_bindings_has_empty_channels(self, mini_ttl):
        _require_rdflib()
        from osprey.services.facility_knowledge.seeder import seed_from_ttl

        stubs = {s.resource: s for s in seed_from_ttl(mini_ttl)}
        qd1 = stubs["https://narad.example.org/device/tst_SEC_QD1"]
        assert qd1.channels == []

    def test_body_contains_frontmatter_and_schema(self, mini_ttl):
        _require_rdflib()
        from osprey.services.facility_knowledge.seeder import seed_from_ttl

        stubs = {s.resource: s for s in seed_from_ttl(mini_ttl)}
        qf1 = stubs["https://narad.example.org/device/tst_SEC_QF1"]
        body = qf1.body

        assert body.startswith("---\n")
        assert "type: device_stub" in body
        assert "resource: https://narad.example.org/device/tst_SEC_QF1" in body
        assert "device_class: Quadrupole" in body
        assert "# Schema" in body
        assert "Monitor" in body
        assert "Setpoint" in body

    def test_result_sorted_by_resource(self, mini_ttl):
        _require_rdflib()
        from osprey.services.facility_knowledge.seeder import seed_from_ttl

        stubs = seed_from_ttl(mini_ttl)
        iris = [s.resource for s in stubs]
        assert iris == sorted(iris)


# ---------------------------------------------------------------------------
# Idempotency test
# ---------------------------------------------------------------------------


class TestIdempotency:
    """Re-running seed_from_ttl on unchanged input must produce byte-for-byte
    identical output."""

    def test_two_runs_produce_identical_bodies(self, mini_ttl):
        _require_rdflib()
        from osprey.services.facility_knowledge.seeder import seed_from_ttl

        run1 = {s.resource: s.body for s in seed_from_ttl(mini_ttl)}
        run2 = {s.resource: s.body for s in seed_from_ttl(mini_ttl)}

        assert run1 == run2


# ---------------------------------------------------------------------------
# Integration test against real als-ontology TTL (optional)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _ALS_GTB_TTL.exists(),
    reason="als-ontology repo not present alongside osprey",
)
class TestSeedFromALSGTB:
    """Integration tests against the real als_gtb.ttl (skipped when absent)."""

    def test_yields_device_stubs(self):
        _require_rdflib()
        from osprey.services.facility_knowledge.seeder import seed_from_ttl

        stubs = seed_from_ttl(_ALS_GTB_TTL)
        assert len(stubs) > 0

    def test_all_resources_are_device_iris(self):
        _require_rdflib()
        from osprey.services.facility_knowledge.seeder import seed_from_ttl

        stubs = seed_from_ttl(_ALS_GTB_TTL)
        for stub in stubs:
            assert stub.resource.startswith("https://narad.example.org/device/"), (
                f"Unexpected IRI: {stub.resource}"
            )

    def test_resource_matches_device_iri_verbatim(self):
        """resource field must equal the canonical device/… IRI verbatim."""
        _require_rdflib()
        from osprey.services.facility_knowledge.seeder import seed_from_ttl

        stubs = seed_from_ttl(_ALS_GTB_TTL)
        # BC1 is always present in the GTL section.
        bc1 = next(
            (s for s in stubs if s.resource == "https://narad.example.org/device/als_GTL_BC1"),
            None,
        )
        assert bc1 is not None, "Expected device als_GTL_BC1 not found"
        assert bc1.resource == "https://narad.example.org/device/als_GTL_BC1"

    def test_device_class_is_real_leaf_class(self):
        _require_rdflib()
        from osprey.services.facility_knowledge.seeder import seed_from_ttl

        stubs = seed_from_ttl(_ALS_GTB_TTL)
        bc1 = next(s for s in stubs if s.resource.endswith("als_GTL_BC1"))
        assert bc1.device_class == "BuckingCoil"

    def test_idempotent_on_als_gtb(self):
        """Two back-to-back runs produce zero byte diff."""
        _require_rdflib()
        from osprey.services.facility_knowledge.seeder import seed_from_ttl

        run1 = {s.resource: s.body for s in seed_from_ttl(_ALS_GTB_TTL)}
        run2 = {s.resource: s.body for s in seed_from_ttl(_ALS_GTB_TTL)}
        assert run1 == run2

    def test_stubs_sorted_by_resource(self):
        _require_rdflib()
        from osprey.services.facility_knowledge.seeder import seed_from_ttl

        stubs = seed_from_ttl(_ALS_GTB_TTL)
        iris = [s.resource for s in stubs]
        assert iris == sorted(iris)
