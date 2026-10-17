"""The example facility's injector transfer-line decks ``decks/LTB.json`` and ``decks/BTS.json``.

``scripts/facility_demo/_transfer.py`` writes each deck from the MML lattice:
every quadrupole ``K`` and bend angle is MML's, the deck energies are the
demo's (150 MeV, 2.0 GeV), the launch Twiss comes from MML's
``updateatindex.m``, the four septa are plain fixed bends, and a nominal
particle crosses each line in one pass. The tests read the MML checkout when
present and otherwise the checked-in excerpts, which parse exactly like MML.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType

import pytest

at = pytest.importorskip("at")

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCES = REPO_ROOT / "scripts/facility_demo"
DECKS = REPO_ROOT / "src/osprey/templates/facilities/example/decks"
EXCERPTS = Path(__file__).resolve().parent / "fixtures/mml"

#: Each lattice file and its checked-in excerpt.
EXCERPT_OF = {
    "GTB/gtblattice.m": "gtblattice_excerpt.m",
    "BTS/btslattice.m": "btslattice_excerpt.m",
}

#: Each line's ``updateatindex.m``, relative to the MML root.
UPDATEATINDEX = {"LTB": "GTB/updateatindex.m", "BTS": "BTS/updateatindex.m"}


def _module(name: str, filename: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    # ``_transfer`` imports its siblings by bare name, as generate.py does.
    spec = importlib.util.spec_from_file_location(name, SOURCES / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # import-time required because the section parametrizes below read the loaded module, and decorators are evaluated at import; scripts/ is not a package, so the
    # module is registered for its dataclasses and its siblings are found on a
    # sys.path entry held only while it executes.
    sys.modules[name] = module
    added = str(SOURCES) not in sys.path
    if added:
        sys.path.insert(0, str(SOURCES))
    try:
        spec.loader.exec_module(module)
    finally:
        if added:
            sys.path.remove(str(SOURCES))
    return module


transfer = _module("facility_demo_transfer", "_transfer.py")
elements_table = _module("facility_demo_elements", "_elements.py")
mml = transfer._mml

HAVE_MML = mml.MML_ROOT is not None and all(
    (mml.MML_ROOT / original).is_file() for original in EXCERPT_OF
)
needs_mml = pytest.mark.skipif(not HAVE_MML, reason="no MML checkout")


def _section(name: str):
    original = mml.LATTICES[name][0]
    path = mml.MML_ROOT / original if HAVE_MML else EXCERPTS / EXCERPT_OF[original]
    return mml.parse(path.read_text(encoding="latin-1"), name)


def _lattice(name: str):
    return at.load_lattice(str(DECKS / f"{name}.json"))


def _by_name(name: str) -> dict:
    return {element.FamName: element for element in _lattice(name) if element.FamName}


# --- the committed decks are the generator's ----------------------------------------


@pytest.mark.parametrize("name", transfer.SECTIONS)
def test_the_committed_deck_is_the_generators_output(name: str) -> None:
    assert transfer.deck_text(_section(name)) == (DECKS / f"{name}.json").read_text("utf-8")


@needs_mml
def test_write_decks_writes_both_decks(tmp_path: Path) -> None:
    written = transfer.write_decks(tmp_path)
    assert [path.name for path in written] == ["LTB.json", "BTS.json"]
    for path in written:
        assert path.read_bytes() == (DECKS / path.name).read_bytes()


# --- energies and optics --------------------------------------------------------------


def test_the_deck_energies_are_the_demos() -> None:
    assert _lattice("LTB").energy == pytest.approx(150.0e6)
    assert _lattice("BTS").energy == pytest.approx(2.0e9)


@pytest.mark.parametrize("name", transfer.SECTIONS)
def test_every_quadrupole_k_equals_mml(name: str) -> None:
    section = _section(name)
    deck = _by_name(name)
    quads = [element for element in section.elements if element.type == "quadrupole"]
    assert quads
    for element in quads:
        quad = deck[transfer.element_name(element.deck_element, section)]
        assert isinstance(quad, at.Quadrupole), element.deck_element
        assert quad.K == element.k, element.deck_element
        assert quad.PolynomB[1] == element.k, element.deck_element
        assert quad.Length == element.length, element.deck_element


@pytest.mark.parametrize("name", transfer.SECTIONS)
def test_every_bend_angle_equals_mml(name: str) -> None:
    section = _section(name)
    deck = _by_name(name)
    bends = [element for element in section.elements if element.kind == "bend"]
    assert bends
    for element in bends:
        bend = deck[transfer.element_name(element.deck_element, section)]
        assert isinstance(bend, at.Dipole), element.deck_element
        definition = element.definition
        assert bend.BendingAngle == definition.angle, element.deck_element
        assert bend.EntranceAngle == definition.entrance_angle, element.deck_element
        assert bend.ExitAngle == definition.exit_angle, element.deck_element
        assert bend.K == definition.k, element.deck_element
        assert bend.Length == definition.length, element.deck_element


@pytest.mark.parametrize("name", transfer.SECTIONS)
def test_the_deck_holds_the_mml_line_in_order(name: str) -> None:
    section = _section(name)
    lattice = _lattice(name)
    names = [element.FamName for element in lattice]
    expected = [transfer.element_name(e.deck_element, section) for e in section.elements]
    assert names[: len(expected)] == expected
    assert names[0] == transfer.START_MARKER
    assert names[-1] == transfer.END_MARKER
    assert lattice.circumference == pytest.approx(sum(e.length for e in section.elements))


@pytest.mark.parametrize("name", transfer.SECTIONS)
def test_monitors_and_counted_elements_are_uniquely_named(name: str) -> None:
    section = _section(name)
    lattice = _lattice(name)
    kinds = Counter(element.kind for element in section.elements)
    counts = Counter(type(element).__name__ for element in lattice)
    assert counts["Monitor"] == kinds["monitor"]
    assert counts["Corrector"] == kinds["corrector"]
    assert counts["Quadrupole"] == kinds["quad"]
    assert counts["Dipole"] == kinds["bend"]
    named = [e.FamName for e in lattice if not isinstance(e, at.Drift)]
    assert len(named) == len(set(named))


def test_element_name_follows_the_occurrence() -> None:
    section = _section("LTB")
    assert transfer.element_name("LTB:BPM#3", section) == "BPM3"
    assert transfer.element_name("LTB:Q11#1", section) == "Q11"
    with pytest.raises(ValueError, match="LTB:BPM#99"):
        transfer.element_name("LTB:BPM#99", section)


def test_every_table_deck_element_and_fixed_element_is_in_a_deck() -> None:
    sections = {name: _section(name) for name in transfer.SECTIONS}
    keys = {row.deck_element for row in elements_table.load_elements() if row.deck_element}
    keys |= set(elements_table.FIXED_ELEMENTS)
    for key in sorted(keys):
        name = key.split(":", 1)[0]
        if name not in sections:
            continue
        deck = _by_name(name)
        assert transfer.element_name(key, sections[name]) in deck, key


def test_the_four_septa_are_fixed_bends() -> None:
    deck = _by_name("BTS")
    for septum in ("SEN", "SEK", "SIK", "SIN"):
        assert isinstance(deck[septum], at.Dipole), septum
        assert f"BTS:{septum}#1" in elements_table.FIXED_ELEMENTS


# --- launch Twiss ---------------------------------------------------------------------


def _mml_twiss(name: str) -> dict[str, list[float]]:
    text = (mml.MML_ROOT / UPDATEATINDEX[name]).read_text(encoding="latin-1")
    found = {}
    for key, field in (("alpha", "alpha"), ("beta", "beta"), ("dispersion", "Dispersion")):
        match = re.search(rf"TwissData\.{field}\s*=\s*\[([^\]]*)\]", text)
        assert match, (name, field)
        found[key] = [float(value) for value in match.group(1).split()]
    return found


@needs_mml
@pytest.mark.parametrize("name", transfer.SECTIONS)
def test_twiss_in_is_mmls_updateatindex(name: str) -> None:
    assert transfer.twiss_in(name) == _mml_twiss(name)


def test_the_ltb_beta_choice_is_stated() -> None:
    doc = transfer.__doc__ or ""
    assert "???" in doc
    assert "12 m" in doc
    assert transfer.twiss_in("LTB")["beta"] == [12.0, 12.0]


# --- a nominal single pass ------------------------------------------------------------


@pytest.mark.parametrize("name", transfer.SECTIONS)
def test_a_nominal_single_pass_loses_nothing(name: str) -> None:
    from osprey.simulation.engines.pyat import prepare
    from osprey.simulation.engines.pyat_single_pass import SinglePassSimulator

    deck = DECKS / f"{name}.json"
    prepared = prepare(
        deck, {"pyat": {"solve": "single_pass", "twiss_in": transfer.twiss_in(name)}}
    )
    solution = SinglePassSimulator(at.load_lattice(str(deck)), twiss_in=prepared.twiss_in).solve()
    assert len(solution) == 6
    for x, y in solution.values():
        assert (x, y) == (0.0, 0.0)
    for beta_x, beta_y in solution.beta.values():
        assert 0.0 < beta_x < 1.0e3
        assert 0.0 < beta_y < 1.0e3
