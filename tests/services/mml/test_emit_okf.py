"""Tests for the OKF knowledge-bundle emitter.

``write_okf_bundle`` writes one ``facility.md`` and one family page per mapped
``(system, family)`` that carries channels, then regenerates the bundle indexes.
The cases pin the page set, the front matter every page carries, the facility
body's omission of lines the accelerator data lacks, the family field table,
that the bundle passes ``osprey knowledge validate`` semantics and advertises no
dangling concept, and that a second write changes no byte. The pages are
written from judged views, so a reviewer's answer reaches the device count a
family page states.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from osprey.services.facility_knowledge.okf.bundle import OKFBundle
from osprey.services.facility_knowledge.okf.document import OKFDocument
from osprey.services.facility_knowledge.okf.index import validate_index
from osprey.services.mml.emit import EmitContext, build_context
from osprey.services.mml.emit.okf import write_okf_bundle
from osprey.services.mml.mapping.schema import (
    Direction,
    Facility,
    Family,
    FamilyJudgments,
    Field,
    Mapping,
    System,
)
from osprey.services.mml.systems import EXPORTS_KEY, IMPORT_ORDER_KEY


def _family(raw: str, *, rename: str | None = None, description: str | None = None) -> Family:
    return Family(
        raw=raw,
        rename=rename,
        branch=None,
        class_="BPM",
        aliases=(),
        description=description if description is not None else f"The {raw} family.",
        provenance="human",
        channels=2,
        fields={
            "Monitor": Field(description="Readback of the monitor.", provenance="human"),
            "Setpoint": Field(description="Setpoint | commanded value.", provenance="derived"),
        },
    )


def _mapping(
    *,
    families: dict[str, Family] | None = None,
    judgments: dict[str, FamilyJudgments] | None = None,
    **facility,
) -> Mapping:
    fac = {"token": "quokka", "title": "Quokka", "description": "The Quokka facility."}
    fac.update(facility)
    return Mapping(
        facility=Facility(provenance="human", **fac),
        systems={
            "RING": System(raw="RING", name="SR", description="Storage ring.", provenance="human"),
            "BOOST": System(raw="BOOST", name="BR", description="Booster.", provenance="human"),
        },
        section_order=("SR", "BR"),
        families=families
        if families is not None
        else {
            "BPMx": _family("BPMx", rename="BPMX"),
            "HCM": _family("HCM"),
            "Empty": _family("Empty"),
        },
        directions={
            "BPMx.Monitor": Direction(direction="read", provenance="human", override=False),
            "HCM.Monitor": Direction(direction="read", provenance="derived", override=False),
            "HCM.Setpoint": Direction(direction="write", provenance="derived", override=False),
        },
        judgments=judgments or {},
    )


def _ao() -> dict:
    return {
        EXPORTS_KEY: {"RING": {"exporter": "mml_export 1.0.0"}},
        IMPORT_ORDER_KEY: ["BOOST", "RING"],
        "BOOST": {
            "BPMx": {"Monitor": {"ChannelNames": ["BR:BPM1:X"]}},
        },
        "RING": {
            "HCM": {
                "DeviceList": [[1, 1], [1, 2]],
                "Monitor": {"ChannelNames": ["SR:HCM1:AM", "SR:HCM2:AM"]},
                "Setpoint": {"ChannelNames": ["SR:HCM1:SP", "SR:HCM2:SP"]},
            },
            "BPMx": {
                "Monitor": {"ChannelNames": ["SR:BPM1:X", None], "TangoNames": ["a/b/c", "d/e/f"]}
            },
            "Empty": {"Monitor": {"ChannelNames": [None]}},
            "_private": {"Monitor": {"ChannelNames": ["X"]}},
        },
    }


def _ad() -> dict:
    return {
        "RING": {
            "Machine": "Quokka",
            "SubMachine": "RING",
            "OperationalMode": "User optics",
            "Energy": 2.4,
            "Circumference": 92.0,
            "HarmonicNumber": 150,
            "MCF": 0.0012,
            "OpsData": {"LatticeFile": "quokka_ring_user"},
        },
        "BOOST": {"Machine": "Quokka", "SubMachine": "BOOSTER"},
    }


@pytest.fixture
def ctx(tmp_path: Path) -> EmitContext:
    ao_path = tmp_path / "ao.json"
    mapping_path = tmp_path / "mapping.yaml"
    ao_path.write_bytes(b'{"RING": {}}\n')
    mapping_path.write_bytes(b"facility:\n  token: quokka\n")
    return build_context(ao_path, mapping_path, _ao())


def _emit(tmp_path: Path, ctx: EmitContext, **kwargs) -> Path:
    bundle = tmp_path / "bundle"
    write_okf_bundle(
        kwargs.get("ao", _ao()),
        kwargs.get("ad", _ad()),
        kwargs.get("mapping", _mapping()),
        ctx,
        bundle,
    )
    return bundle


def _doc(path: Path) -> OKFDocument:
    return OKFDocument.parse(path.read_text(encoding="utf-8"))


class TestPageSet:
    def test_one_page_per_mapped_family_with_channels(self, tmp_path: Path, ctx) -> None:
        bundle = _emit(tmp_path, ctx)

        pages = sorted(p.relative_to(bundle).as_posix() for p in bundle.rglob("*.md"))

        assert pages == [
            "facility.md",
            "families/BR-BPMX.md",
            "families/SR-BPMX.md",
            "families/SR-HCM.md",
            "families/index.md",
            "index.md",
        ]

    def test_returns_every_written_path(self, tmp_path: Path, ctx) -> None:
        bundle = tmp_path / "bundle"

        written = write_okf_bundle(_ao(), _ad(), _mapping(), ctx, bundle)

        assert set(written) == set(bundle.rglob("*.md"))

    def test_family_missing_from_mapping_raises(self, tmp_path: Path, ctx) -> None:
        ao = _ao()
        ao["RING"]["Unmapped"] = {"Monitor": {"ChannelNames": ["X"]}}

        with pytest.raises(ValueError, match="Unmapped"):
            _emit(tmp_path, ctx, ao=ao)

    def test_system_left_out_of_section_order_raises(self, tmp_path: Path, ctx) -> None:
        ao = _ao()
        ao["LINAC"] = {"HCM": {"Monitor": {"ChannelNames": ["X"]}}}

        with pytest.raises(ValueError, match="LINAC"):
            _emit(tmp_path, ctx, ao=ao)


class TestFrontMatter:
    def test_facility_front_matter(self, tmp_path: Path, ctx) -> None:
        bundle = _emit(tmp_path, ctx)

        fm = _doc(bundle / "facility.md").frontmatter

        assert fm == {
            "type": "Facility",
            "title": "Quokka",
            "description": "The Quokka facility.",
            "exporter": "mml_export 1.0.0",
            "ao_sha256": ctx.ao_sha256,
            "mapping_sha256": ctx.mapping_sha256,
        }

    def test_family_front_matter(self, tmp_path: Path, ctx) -> None:
        bundle = _emit(tmp_path, ctx)

        fm = _doc(bundle / "families" / "SR-HCM.md").frontmatter

        assert fm == {
            "type": "DeviceFamily",
            "title": "SR HCM",
            "description": "The HCM family.",
            "provenance": "human",
            "exporter": "mml_export 1.0.0",
            "ao_sha256": ctx.ao_sha256,
            "mapping_sha256": ctx.mapping_sha256,
        }

    def test_null_facility_description_raises(self, tmp_path: Path, ctx) -> None:
        with pytest.raises(ValueError, match="facility.description"):
            _emit(tmp_path, ctx, mapping=_mapping(description=None))

    def test_facility_title_falls_back_to_token(self, tmp_path: Path, ctx) -> None:
        bundle = _emit(tmp_path, ctx, mapping=_mapping(title=None))

        assert _doc(bundle / "facility.md").frontmatter["title"] == "quokka"

    def test_every_page_round_trips(self, tmp_path: Path, ctx) -> None:
        bundle = _emit(tmp_path, ctx)

        for page in bundle.rglob("*.md"):
            if page.name == "index.md":
                continue
            text = page.read_text(encoding="utf-8")
            assert OKFDocument.parse(text).serialize() == text, page


class TestFacilityBody:
    def test_lists_ad_facts(self, tmp_path: Path, ctx) -> None:
        body = _doc(_emit(tmp_path, ctx) / "facility.md").body

        assert "- Machine: Quokka" in body
        assert "- Sub-machines: SR, BR" in body
        assert "## SR" in body
        assert "- Energy (GeV): 2.4" in body
        assert "- Circumference (m): 92" in body
        assert "- Harmonic number: 150" in body
        assert "- MCF: 0.0012" in body
        assert "- Mode: User optics" in body
        assert "- Lattice file: quokka_ring_user" in body

    def test_sub_machines_follow_section_order(self, tmp_path: Path, ctx) -> None:
        mapping = _mapping()
        reordered = Mapping(
            facility=mapping.facility,
            systems=mapping.systems,
            section_order=("BR", "SR"),
            families=mapping.families,
            directions=mapping.directions,
        )

        body = _doc(_emit(tmp_path, ctx, mapping=reordered) / "facility.md").body

        assert "- Sub-machines: BR, SR" in body
        assert body.index("## BR") < body.index("## SR")

    def test_injection_energy_and_lattice_model_are_facts(self, tmp_path: Path, ctx) -> None:
        """``InjectionEnergy`` is a labelled fact; ``ATModel`` names the lattice when no file does."""
        ad = _ad()
        ad["RING"] = {
            **ad["RING"],
            "InjectionEnergy": 0.1,
            "ATModel": "quokka_ring_model",
            "OpsData": {},
        }

        body = _doc(_emit(tmp_path, ctx, ad=ad) / "facility.md").body

        assert "- Injection energy (GeV): 0.1" in body
        assert "- Lattice file: quokka_ring_model" in body

    def test_a_lattice_file_wins_over_the_model_name(self, tmp_path: Path, ctx) -> None:
        """When both are present the lattice file is the fact, the model name a fallback."""
        ad = _ad()
        ad["RING"] = {**ad["RING"], "ATModel": "quokka_ring_model"}

        body = _doc(_emit(tmp_path, ctx, ad=ad) / "facility.md").body

        assert "- Lattice file: quokka_ring_user" in body
        assert "quokka_ring_model" not in body

    def test_lines_omitted_when_ad_lacks_them(self, tmp_path: Path, ctx) -> None:
        body = _doc(_emit(tmp_path, ctx) / "facility.md").body
        booster = body[body.index("## BR") :]

        assert "Energy" not in booster
        assert "Lattice file" not in booster

    def test_every_system_has_a_section_with_its_description(self, tmp_path: Path, ctx) -> None:
        """Each sub-machine opens a section led by the mapping's prose, AD or not."""
        body = _doc(_emit(tmp_path, ctx, ad={}) / "facility.md").body

        assert "- Sub-machines: SR, BR" in body
        assert "## SR\n\nStorage ring." in body
        assert "## BR\n\nBooster." in body
        assert "Machine" not in body
        assert "Energy" not in body
        assert "The Quokka facility." in body


class TestFamilyBody:
    def test_field_table_has_keys_and_directions(self, tmp_path: Path, ctx) -> None:
        body = _doc(_emit(tmp_path, ctx) / "families" / "SR-HCM.md").body

        assert "| Monitor | ChannelNames | read | 2 | Readback of the monitor. |" in body
        assert r"| Setpoint | ChannelNames | write | 2 | Setpoint \| commanded value. |" in body

    def test_both_channel_keys_and_blank_slots(self, tmp_path: Path, ctx) -> None:
        body = _doc(_emit(tmp_path, ctx) / "families" / "SR-BPMX.md").body

        assert "| Monitor | ChannelNames, TangoNames | read | 3 |" in body

    def test_undirected_field_says_so(self, tmp_path: Path, ctx) -> None:
        ao = _ao()
        ao["RING"]["HCM"]["Readback"] = {"ChannelNames": ["A", "B"]}

        body = _doc(_emit(tmp_path, ctx, ao=ao) / "families" / "SR-HCM.md").body

        assert "| Readback | ChannelNames | unset | 2 |" in body


class TestBundle:
    def test_knowledge_validate_semantics(self, tmp_path: Path, ctx) -> None:
        bundle = _emit(tmp_path, ctx)

        for md in sorted(bundle.rglob("*.md")):
            if md.name == "index.md":
                validate_index(md, bundle_root=bundle)
            else:
                OKFDocument.parse(md.read_text(encoding="utf-8")).validate("authoring")

    def test_indexes_list_pages(self, tmp_path: Path, ctx) -> None:
        bundle = _emit(tmp_path, ctx)

        root = (bundle / "index.md").read_text(encoding="utf-8")
        families = (bundle / "families" / "index.md").read_text(encoding="utf-8")

        assert "# Facility\n\n* [Quokka](/facility.md)" in root
        assert "# Subdirectories\n\n* [families](/families/)" in root
        headings = [line for line in families.splitlines() if line.startswith("# ")]
        assert headings == ["# DeviceFamily"]
        assert families.count("(/families/") == 3

    def test_every_concept_resolves(self, tmp_path: Path, ctx) -> None:
        bundle = _emit(tmp_path, ctx)

        concepts = OKFBundle(bundle).list_concepts()

        assert {c.concept_id for c in concepts} == {
            "facility",
            "families/SR-HCM",
            "families/SR-BPMX",
            "families/BR-BPMX",
        }
        for concept in concepts:
            assert (bundle / f"{concept.concept_id}.md").is_file()

    def test_second_write_is_byte_identical(self, tmp_path: Path, ctx) -> None:
        bundle = _emit(tmp_path, ctx)
        before = {p: p.read_bytes() for p in bundle.rglob("*") if p.is_file()}

        _emit(tmp_path, ctx)

        assert {p: p.read_bytes() for p in bundle.rglob("*") if p.is_file()} == before


class TestJudgedViews:
    """A family page counts the devices the reviewer settled on, not the exported ones."""

    def test_a_dropped_device_is_not_counted_on_the_family_page(self, tmp_path: Path, ctx) -> None:
        """A TUNE exports three devices and binds two, so the answered page says two."""
        ao = _ao()
        ao["RING"]["TUNE"] = {
            "DeviceList": [[1, 1], [1, 2], [1, 3]],
            "Position": 0,
            "Monitor": {"ChannelNames": ["SR:TUNE:X", "SR:TUNE:Y"]},
        }
        mapping = _mapping(
            families={
                "BPMx": _family("BPMx", rename="BPMX"),
                "HCM": _family("HCM"),
                "Empty": _family("Empty"),
                "TUNE": _family("TUNE"),
            },
            judgments={"TUNE": FamilyJudgments(unbound_devices={3: "drop"})},
        )

        body = _doc(_emit(tmp_path, ctx, ao=ao, mapping=mapping) / "families" / "SR-TUNE.md").body

        assert "- Devices: 2\n- Channels: 2" in body
