"""The MML lattice reader gives every injector section its beam order.

``scripts/facility_demo/_mml.py`` reads ``gtblattice.m`` (GTL, LN, LTB) and
``btslattice.m`` (BTS) into ordered, occurrence-indexed elements with type,
length, normalized K and bend angle. The tests read the MML files when an MML
checkout is present and otherwise the checked-in excerpts, which must then
parse exactly like MML.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCES = REPO_ROOT / "scripts/facility_demo"
EXCERPTS = Path(__file__).resolve().parent / "fixtures/mml"


def _module(name: str, filename: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SOURCES / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # import-time required because the section parametrizes below read the loaded
    # module, and decorators are evaluated at import; scripts/ is not a package, and
    # dataclasses resolve their annotations through the module's sys.modules entry.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mml = _module("facility_demo_mml", "_mml.py")
elements = _module("facility_demo_elements", "_elements.py")

#: Each lattice file and its checked-in excerpt.
EXCERPT_OF = {
    "GTB/gtblattice.m": "gtblattice_excerpt.m",
    "BTS/btslattice.m": "btslattice_excerpt.m",
}

HAVE_MML = mml.MML_ROOT is not None and all(
    (mml.MML_ROOT / original).is_file() for original in EXCERPT_OF
)
needs_mml = pytest.mark.skipif(not HAVE_MML, reason="no MML checkout")


def _text(section: str, *, prefer_mml: bool = True) -> str:
    original = mml.LATTICES[section][0]
    path = mml.MML_ROOT / original if prefer_mml and HAVE_MML else EXCERPTS / EXCERPT_OF[original]
    return path.read_text(encoding="latin-1")


def _section(section: str, *, prefer_mml: bool = True):
    return mml.parse(_text(section, prefer_mml=prefer_mml), section)


def _keys(section: str) -> list[str]:
    return [element.deck_element for element in _section(section).elements]


def _by_key(section: str) -> dict:
    return {element.deck_element: element for element in _section(section).elements}


# --- MML and the excerpt agree ------------------------------------------------------


def _shape(parsed) -> list[tuple]:
    return [(element.deck_element, element.definition) for element in parsed.elements] + [
        ("energy", parsed.energy)
    ]


@needs_mml
@pytest.mark.parametrize("section", sorted(mml.LATTICES))
def test_the_excerpt_parses_like_mml(section: str) -> None:
    assert _shape(_section(section, prefer_mml=False)) == _shape(_section(section))


@pytest.mark.parametrize("section", sorted(mml.LATTICES))
def test_the_excerpt_parses_on_its_own(section: str) -> None:
    parsed = _section(section, prefer_mml=False)
    assert parsed.name == section
    assert len(parsed.elements) > 20


# --- beam order -------------------------------------------------------------------


def test_every_section_is_parsed() -> None:
    assert set(mml.LATTICES) == {"GTL", "LN", "LTB", "BTS"}
    assert set(mml.LATTICES) <= set(elements.SECTIONS)


@pytest.mark.parametrize("section", ["GTL", "LN"])
def test_tv1_sits_between_bpm1_and_bpm2(section: str) -> None:
    keys = _keys(section)
    tv = keys.index(f"{section}:TV#1")
    assert keys.index(f"{section}:BPM#1") < tv < keys.index(f"{section}:BPM#2")


def test_gtl_runs_in_mml_order() -> None:
    counted = [key for key, element in _by_key("GTL").items() if element.type != "drift"]
    assert counted == [
        "GTL:START#1",
        "GTL:HCM#1",
        "GTL:VCM#1",
        "GTL:HCM#2",
        "GTL:VCM#2",
        "GTL:BPM#1",
        "GTL:HCM#3",
        "GTL:VCM#3",
        "GTL:TV#1",
        "GTL:BPM#2",
        "GTL:HCM#4",
        "GTL:VCM#4",
        "GTL:TV#2",
        "GTL:END#1",
    ]


def test_the_linac_carries_its_triplet_between_tv1_and_the_correctors() -> None:
    keys = [key for key, element in _by_key("LN").items() if element.type != "drift"]
    tv = keys.index("LN:TV#1")
    assert keys[tv + 1 : tv + 4] == ["LN:LNQ11#1", "LN:LNQ12#1", "LN:LNQ13#1"]
    # AS, the accelerating section, is a drift of its own name, once per structure.
    all_keys = _keys("LN")
    assert all_keys.index("LN:LNQ13#1") < all_keys.index("LN:AS#2") < all_keys.index("LN:BPM#2")


def test_ltb_concatenates_ltb1_and_ltb2() -> None:
    keys = _keys("LTB")
    assert keys[0] == "LTB:START#1"
    assert keys[-1] == "LTB:END#1"
    assert keys.index("LTB:BS#1") < keys.index("LTB:LTB_BS_VCM3#1") < keys.index("LTB:VCM#3")
    # Occurrences count across both arrays: LTB1 has two BPMs, LTB2 the next five.
    assert keys.index("LTB:BPM#2") < keys.index("LTB:BS#1") < keys.index("LTB:BPM#3")
    assert "LTB:BPM#7" not in keys and "LTB:BPM#6" in keys


def test_srtv1_parses_into_the_bts_line() -> None:
    bts = _section("BTS")
    assert bts.elements[-1].deck_element == "BTS:SRTV1#1"
    assert bts.elements[-1].family == "TV"
    assert bts.elements[-1].type == "marker"


@needs_mml
def test_srtv1_comes_from_btslattice_line_151() -> None:
    assert _by_key("BTS")["BTS:SRTV1#1"].line == 151


def test_a_repeated_drift_is_occurrence_indexed() -> None:
    keys = _keys("BTS")
    assert keys.index("BTS:L50#1") < keys.index("BTS:SRBMP3#1") < keys.index("BTS:L50#2")


def test_occurrences_number_from_one_in_beam_order() -> None:
    for section in mml.LATTICES:
        seen: dict[str, int] = {}
        for element in _section(section).elements:
            seen[element.name] = seen.get(element.name, 0) + 1
            assert element.occurrence == seen[element.name], element
            assert element.section == section


def test_deck_element_keys_are_unique() -> None:
    for section in mml.LATTICES:
        keys = _keys(section)
        assert len(keys) == len(set(keys)), section


# --- element values ---------------------------------------------------------------


def test_the_energies_are_mmls_defaults() -> None:
    for section in ("GTL", "LN", "LTB"):
        assert _section(section).energy == pytest.approx(0.05e9)
    assert _section("BTS").energy == pytest.approx(1.9e9)


def test_getbrho_matches_mml() -> None:
    e0 = 0.51099906e-3
    assert mml.getbrho(0.05) == pytest.approx(10 / 2.99792458 * math.sqrt((0.05 + e0) ** 2 - e0**2))
    assert mml.getbrho(1.9) == pytest.approx(6.3394, abs=1e-3)


def test_gtb_quadrupoles_are_normalized_by_brho() -> None:
    brho = mml.getbrho(0.05)
    ltb = _by_key("LTB")
    q11 = ltb["LTB:Q11#1"]
    assert q11.type == "quadrupole"
    assert q11.length == pytest.approx(0.15)
    assert q11.k == pytest.approx(-2.0 * 2.44 / 7.2 / brho)
    assert ltb["LTB:BR1QF2#1"].k == pytest.approx(0.462 / brho)
    assert _by_key("LN")["LN:LNQ12#1"].k == pytest.approx(-1.2635 / brho)
    assert _by_key("LN")["LN:LNQ12#1"].length == pytest.approx(0.12)


def test_bts_quadrupoles_keep_mmls_k() -> None:
    bts = _by_key("BTS")
    assert bts["BTS:Q7#1"].k == pytest.approx(1.7455)
    assert bts["BTS:Q21#1"].k == pytest.approx(-0.6129)
    assert bts["BTS:Q1#1"].length == pytest.approx(0.3)


def test_bends_carry_angle_and_edges() -> None:
    ltb = _by_key("LTB")
    b1 = ltb["LTB:B1#1"]
    assert (b1.type, b1.family, b1.kind) == ("rbend", "BEND", "bend")
    assert b1.length == pytest.approx(0.515)
    assert b1.angle == pytest.approx(math.radians(40))
    assert b1.definition.entrance_angle == pytest.approx(math.radians(18))
    assert b1.definition.exit_angle == pytest.approx(math.radians(18))
    assert ltb["LTB:BX#1"].angle == 0.0
    sik = _by_key("BTS")["BTS:SIK#1"]
    assert sik.angle == pytest.approx(10.07 * math.pi / 180)
    assert sik.definition.exit_angle == 0.0


def test_a_redefined_corrector_takes_its_sections_length() -> None:
    assert _by_key("GTL")["GTL:HCM#1"].length == pytest.approx(6 * 0.0254)
    assert _by_key("LTB")["LTB:HCM#1"].length == pytest.approx(0.052)
    assert _by_key("BTS")["BTS:HCM#1"].length == pytest.approx(6 * 0.0254)


def test_drift_lengths_evaluate_their_expressions() -> None:
    assert _by_key("LN")["LN:LN_TV1_Q11#1"].length == pytest.approx(0.146 - 0.12 / 2)
    assert _by_key("BTS")["BTS:L3#1"].length == pytest.approx(0.798989 - 6 * 0.0254 / 2)
    assert _by_key("LTB")["LTB:LTB_HCM1_VCM1#1"].length == pytest.approx(0.178 - 0.052)


def test_kinds_classify_like_deck_coverage() -> None:
    gtl = _by_key("GTL")
    assert gtl["GTL:HCM#1"].kind == "corrector"
    assert gtl["GTL:BPM#1"].kind == "monitor"
    assert gtl["GTL:TV#1"].kind is None
    assert gtl["GTL:GTL_START_HCM1#1"].kind is None
    assert _by_key("LN")["LN:LNQ11#1"].kind == "quad"
    assert _by_key("BTS")["BTS:SRBMP4#1"].kind == "bend"


# --- the element table resolves --------------------------------------------------


def test_every_table_deck_element_resolves_to_exactly_one_element() -> None:
    parsed: dict[str, int] = {}
    for section in mml.LATTICES:
        for key in _keys(section):
            parsed[key] = parsed.get(key, 0) + 1
    named = {row.deck_element for row in elements.load_elements() if row.deck_element}
    named |= set(elements.FIXED_ELEMENTS)
    assert sorted(key for key in named if parsed.get(key) != 1) == []


# --- errors -----------------------------------------------------------------------


def test_an_unknown_section_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown section"):
        mml.parse("", "SR")


def test_a_missing_array_is_refused() -> None:
    text = "Energy = 1e9;\nSTART = marker('START', 'IdentityPass');\n"
    with pytest.raises(ValueError, match="BTS"):
        mml.parse(text, "BTS")


def test_an_undefined_array_token_is_refused() -> None:
    text = "Energy = 1e9;\nSTART = marker('START', 'IdentityPass');\nBTS = [START NOPE];\n"
    with pytest.raises(ValueError, match="line 3: array token NOPE"):
        mml.parse(text, "BTS")


def test_an_unbound_length_is_refused_with_its_line() -> None:
    text = "Energy = 1e9;\n\nHCM = corrector('HCM', Lcm, [0 0], 'CorrectorPass');\n"
    with pytest.raises(ValueError, match="line 3: cannot evaluate 'Lcm'"):
        mml.execute(text)


def test_control_flow_and_unknown_calls_are_skipped() -> None:
    text = "\n".join(
        [
            "function varargout = btslattice(varargin)",
            "if nargin >=1",
            "    Energy = varargin{1};",
            "else",
            "    Energy = 2e9;",
            "end",
            "L = 0.5;  % a length",
            "D = drift('D', L/2, ...",
            "    'DriftPass');",
            "TABLE = [",
            "  0.0 1.0",
            "  ];",
            "BTS = [D ...",
            "D];",
            "buildlat(BTS);",
            "L0 = findspos(THERING, 1);",
        ]
    )
    parsed = mml.parse(text, "BTS")
    assert parsed.energy == 2e9
    assert [(e.deck_element, e.length, e.line) for e in parsed.elements] == [
        ("BTS:D#1", 0.25, 13),
        ("BTS:D#2", 0.25, 14),
    ]
