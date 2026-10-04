"""Tests for the TTL-based OKF stub seeder.

All tests that actually invoke rdflib are guarded by ``pytest.importorskip``,
so they report a clear skip if rdflib is not importable — it is a core
dependency, so that means a broken environment.
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

_QF1 = "https://narad.example.org/device/tst_device_SEC_x2F_QF1"
_QD1 = "https://narad.example.org/device/tst_device_SEC_x2F_QD1"


def _mini_facility() -> dict:
    """A small facility file: one place, two quadrupoles, channels on one of them.

    Two of the channels end in the same token, so a channel name cut from the
    end of an address would give both rows one name.
    """
    return {
        "schema": "osprey.facility.facility/1",
        "identity": {"code": "tst"},
        "places": [{"id": "SEC", "level": "sector"}],
        "devices": [
            {"id": "SEC/QF1", "class": "Quadrupole", "place": "SEC", "names": ["QF1"]},
            {"id": "SEC/QD1", "class": "Quadrupole", "place": "SEC", "names": ["QD1"]},
        ],
        "channels": [
            {
                "id": "SEC:QF1:CURRENT:RB",
                "on": {"device": "SEC/QF1"},
                "signal": "current_readback",
            },
            {
                "id": "SEC:QF1:CURRENT:SP",
                "role": "setpoint",
                "on": {"device": "SEC/QF1"},
                "signal": "current_setpoint",
            },
            {
                "id": "SEC:QF1:VOLTAGE:RB",
                "on": {"device": "SEC/QF1"},
                "signal": "voltage_readback",
            },
        ],
    }


# A device whose binding IRI is not one the graph view mints: the channel name
# falls back to the binding's ``bindingId``.
_FOREIGN_TTL = textwrap.dedent("""\
    @prefix narad_p: <https://narad.example.org/property/> .
    @prefix narad_sem: <https://narad.example.org/schema/shared_semantics/> .

    <https://narad.example.org/device/tst_SEC_QF1> a narad_sem:Quadrupole ;
        narad_p:deviceId "SEC/QF1" ;
        narad_p:facility "tst" ;
        narad_p:hasBinding <https://narad.example.org/binding/tst_SEC_QF1_Monitor> .

    <https://narad.example.org/binding/tst_SEC_QF1_Monitor> a narad_sem:ChannelBinding ;
        narad_p:bindingId "SEC:QF1:Monitor" ;
        narad_p:fullPv "SEC:QF1:Monitor" ;
        narad_p:readsSignal narad_sem:current_readback .
""")


@pytest.fixture
def mini_ttl(tmp_path: Path) -> Path:
    """Write the graph view of the small facility file and return its path."""
    from osprey.facility.views.graph import graph_text

    p = tmp_path / "mini.ttl"
    p.write_text(graph_text(_mini_facility()), encoding="utf-8")
    return p


@pytest.fixture
def foreign_ttl(tmp_path: Path) -> Path:
    """Write the foreign-IRI corpus and return its path."""
    p = tmp_path / "foreign.ttl"
    p.write_text(_FOREIGN_TTL, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Import-level guard — no rdflib → all rdflib-touching tests skip
# ---------------------------------------------------------------------------


def _require_rdflib():
    """Skip current test if rdflib is not importable."""
    return pytest.importorskip(
        "rdflib", reason="rdflib not importable (core dependency; broken environment)"
    )


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
        assert _QF1 in iris
        assert _QD1 in iris

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
        qf1 = stubs[_QF1]
        assert qf1.title == "SEC:QF1 (Quadrupole)"

    def test_channels_sorted_by_name(self, mini_ttl):
        _require_rdflib()
        from osprey.services.facility_knowledge.seeder import seed_from_ttl

        stubs = {s.resource: s for s in seed_from_ttl(mini_ttl)}
        qf1 = stubs[_QF1]
        assert len(qf1.channels) == 3
        names = [c.channel for c in qf1.channels]
        assert names == sorted(names)

    def test_channel_cells_are_unique_per_device(self, mini_ttl):
        """Each channel's cell is its address, so two channels never share one."""
        _require_rdflib()
        from osprey.services.facility_knowledge.seeder import seed_from_ttl

        stubs = {s.resource: s for s in seed_from_ttl(mini_ttl)}
        names = [c.channel for c in stubs[_QF1].channels]
        assert names == ["SEC:QF1:CURRENT:RB", "SEC:QF1:CURRENT:SP", "SEC:QF1:VOLTAGE:RB"]

    def test_a_foreign_binding_iri_names_the_channel_by_its_binding_id(self, foreign_ttl):
        _require_rdflib()
        from osprey.services.facility_knowledge.seeder import seed_from_ttl

        (stub,) = seed_from_ttl(foreign_ttl)
        assert [c.channel for c in stub.channels] == ["SEC:QF1:Monitor"]

    def test_channel_pv_and_direction(self, mini_ttl):
        _require_rdflib()
        from osprey.services.facility_knowledge.seeder import seed_from_ttl

        stubs = {s.resource: s for s in seed_from_ttl(mini_ttl)}
        qf1 = stubs[_QF1]
        ch = {c.channel: c for c in qf1.channels}

        assert ch["SEC:QF1:CURRENT:RB"].pv == "SEC:QF1:CURRENT:RB"
        assert ch["SEC:QF1:CURRENT:RB"].direction == "reads"
        assert ch["SEC:QF1:CURRENT:RB"].signal == "current_readback"

        assert ch["SEC:QF1:CURRENT:SP"].pv == "SEC:QF1:CURRENT:SP"
        assert ch["SEC:QF1:CURRENT:SP"].direction == "writes"
        assert ch["SEC:QF1:CURRENT:SP"].signal == "current_setpoint"

    def test_device_with_no_bindings_has_empty_channels(self, mini_ttl):
        _require_rdflib()
        from osprey.services.facility_knowledge.seeder import seed_from_ttl

        stubs = {s.resource: s for s in seed_from_ttl(mini_ttl)}
        qd1 = stubs[_QD1]
        assert qd1.channels == []

    def test_body_contains_frontmatter_and_schema(self, mini_ttl):
        _require_rdflib()
        from osprey.services.facility_knowledge.seeder import seed_from_ttl

        stubs = {s.resource: s for s in seed_from_ttl(mini_ttl)}
        qf1 = stubs[_QF1]
        body = qf1.body

        assert body.startswith("---\n")
        assert "type: device_stub" in body
        assert f"resource: {_QF1}" in body
        assert "device_class: Quadrupole" in body
        assert "# Schema" in body
        assert "| Channel | PV | Signal | Direction |\n" in body
        assert "SEC:QF1:CURRENT:SP" in body

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
