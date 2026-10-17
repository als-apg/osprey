"""The injector element table is a complete, well-formed spec of the injector channels.

``scripts/facility_demo/injector_elements.tsv`` maps every GTLView/BTSView binding
and every non-panel injector channel to a demo name in the six-token grammar.
Every name uses only the grammar's alphabets and no legacy MML form; every role
has a template, and every injector BPM reads in millimetres. Every deck element
a row names resolves to exactly one occurrence of the MML lattice arrays (LTB1 +
LTB2 of ``gtblattice.m``, BTS of ``btslattice.m``), and every corrector,
quadrupole, bend and BPM occurrence there is bound by a row or listed as fixed.
The arrays are read from MML when it is present, else from the checked-in
excerpt, which must then agree with MML.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from collections import defaultdict
from functools import cache
from pathlib import Path
from types import ModuleType

import pytest

from osprey.facility.sources import read_yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ELEMENTS_SOURCE = REPO_ROOT / "scripts/facility_demo/_elements.py"
EXAMPLE = REPO_ROOT / "src/osprey/templates/facilities/example"
EXCERPTS = Path(__file__).resolve().parent / "fixtures/mml"

#: The machine folder of an MML checkout (the one holding ``GTB/`` and ``BTS/``),
#: when ``OSPREY_MML_MACHINE_ROOT`` names one.
MML_ROOT = (
    Path(os.environ["OSPREY_MML_MACHINE_ROOT"])
    if os.environ.get("OSPREY_MML_MACHINE_ROOT")
    else None
)

#: Each deck section: its MML file, its excerpt, and the arrays it concatenates.
LATTICES = {
    "LTB": ("GTB/gtblattice.m", "gtblattice_excerpt.m", ("LTB1", "LTB2")),
    "BTS": ("BTS/btslattice.m", "btslattice_excerpt.m", ("BTS",)),
}

#: The element kinds deck coverage counts, by MML constructor.
COUNTED_CONSTRUCTORS = {
    "corrector": "corrector",
    "quadrupole": "quad",
    "rbend": "bend",
    "sbend": "bend",
}

#: The injector sections, whose BPMs read in millimetres.
INJECTOR_SECTIONS = ("GTL", "LN", "LTB", "BTS")

#: The septa: the legacy setpoint of each and the deck element it is bound to.
SEPTA = {
    "SR01S___SEK____AC01": "BTS:SIK#1",
    "SR01S___SEN____AC00": "BTS:SIN#1",
    "BR2_____SEN____AC00": "BTS:SEN#1",
    "BR2_____SEK____AC01": "BTS:SEK#1",
}


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("facility_demo_elements", ELEMENTS_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # import-time required because the rows below feed a parametrize decorator, which
    # is evaluated at import, and scripts/ is not a package: dataclasses resolve
    # their annotations through the module's sys.modules entry.
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


elements = _module()
ROWS = elements.load_elements()


# --- the light MML scan -----------------------------------------------------------

_DEFINITION = re.compile(r"^\s*(\w+)\s*=\s*(\w+)\(\s*'(\w+)'")
_ARRAY = re.compile(r"^\s*(\w+)\s*=\s*\[")
_TOKEN = re.compile(r"[A-Za-z_]\w*")


def _code(line: str) -> str:
    return line.split("%", 1)[0]


def scan(text: str, arrays: tuple[str, ...]) -> list[tuple[str, int, str | None]]:
    """``(token, n, kind)`` per array element: its n-th occurrence and counted kind.

    The kind is ``corrector``, ``quad``, ``bend``, ``monitor`` (a ``BPM`` marker),
    or ``None`` for every other element (drifts, other markers, apertures).
    """
    lines = text.splitlines()
    kinds: dict[str, str | None] = {}
    bodies: dict[str, list[str]] = {}
    index = 0
    while index < len(lines):
        line = _code(lines[index])
        definition = _DEFINITION.match(line)
        array = _ARRAY.match(line)
        if array and array.group(1) in arrays:
            body = [line.split("[", 1)[1]]
            while "]" not in body[-1]:
                index += 1
                body.append(_code(lines[index]))
            body[-1] = body[-1].split("]", 1)[0]
            bodies[array.group(1)] = body
        elif definition:
            name, constructor, family_name = definition.groups()
            if constructor == "marker":
                kinds[name] = "monitor" if family_name == "BPM" else None
            else:
                kinds[name] = COUNTED_CONSTRUCTORS.get(constructor)
        index += 1
    assert set(bodies) == set(arrays), f"arrays not found: {set(arrays) - set(bodies)}"
    seen: dict[str, int] = defaultdict(int)
    occurrences = []
    for array_name in arrays:
        for token in _TOKEN.findall(" ".join(bodies[array_name])):
            seen[token] += 1
            occurrences.append((token, seen[token], kinds.get(token)))
    return occurrences


def _lattice_text(section: str, *, prefer_mml: bool) -> str:
    original, excerpt, _ = LATTICES[section]
    source = MML_ROOT / original if MML_ROOT is not None else None
    path = source if prefer_mml and source is not None and source.is_file() else EXCERPTS / excerpt
    return path.read_text(encoding="latin-1")


@cache
def occurrences() -> dict[str, list[tuple[str, int, str | None]]]:
    """Every section's scan, from MML when present, else from the excerpt."""
    return {
        section: scan(_lattice_text(section, prefer_mml=True), LATTICES[section][2])
        for section in LATTICES
    }


def deck_elements() -> set[str]:
    return {
        f"{sec}:{token}#{n}" for sec, scanned in occurrences().items() for token, n, _ in scanned
    }


def counted_elements() -> set[str]:
    return {
        f"{sec}:{token}#{n}"
        for sec, scanned in occurrences().items()
        for token, n, kind in scanned
        if kind is not None
    }


@pytest.mark.skipif(MML_ROOT is None or not MML_ROOT.is_dir(), reason="no MML checkout")
@pytest.mark.parametrize("section", sorted(LATTICES))
def test_the_excerpt_scans_like_mml(section: str) -> None:
    arrays = LATTICES[section][2]
    assert scan(_lattice_text(section, prefer_mml=False), arrays) == scan(
        _lattice_text(section, prefer_mml=True), arrays
    )


def test_the_scan_finds_the_known_shape() -> None:
    counted = counted_elements()
    for element in (
        "LTB:HCM#2",
        "LTB:VCM#7",
        "LTB:BPM#6",
        "LTB:BX#1",
        "LTB:BRKI#1",
        "BTS:HCM#9",
        "BTS:VCM#9",
        "BTS:BPM#6",
        "BTS:Q7#1",
        "BTS:SRBMP4#1",
    ):
        assert element in counted
    for element in ("LTB:HCM#3", "LTB:BPM#7", "BTS:HCM#10", "BTS:BPM#7"):
        assert element not in counted
    assert "BTS:SRTV1#1" in deck_elements()


# --- the table's shape --------------------------------------------------------------


def test_the_table_has_rows() -> None:
    assert len(ROWS) > 400


def test_a_malformed_header_is_refused(tmp_path: Path) -> None:
    table = tmp_path / "t.tsv"
    table.write_text("legacy\trole\tdemo_name\tdeck_element\tnominal\n")
    with pytest.raises(ValueError, match="header"):
        elements.load_elements(table)


def test_a_short_row_is_refused(tmp_path: Path) -> None:
    table = tmp_path / "t.tsv"
    table.write_text(
        "\t".join(elements.COLUMNS) + "\n# banner\nX\thcm_sp\tGTL:MAG:HCM:01:CURRENT:SP\n"
    )
    with pytest.raises(ValueError, match="row 2"):
        elements.load_elements(table)


@pytest.mark.parametrize("row", ROWS, ids=lambda row: f"{row.legacy_pv}|{row.role}")
def test_every_demo_name_follows_the_grammar(row) -> None:
    name = row.demo_name
    match = elements.DEMO_NAME.match(name)
    assert match, name
    assert match["sec"] in elements.SECTIONS
    assert match["sys"] in elements.SYSTEMS
    assert match["family"] in elements.FAMILIES
    assert match["field"] in elements.FIELDS
    assert match["sub"] in elements.SUBFIELDS
    assert not re.match(r"^[A-Z]{2,4}\d*_{2,}", name)
    assert not name.endswith((":Setpoint", ":Monitor"))


def test_every_role_has_a_template() -> None:
    assert {row.role for row in ROWS} <= set(elements.ROLE_TEMPLATES)


def test_every_template_is_well_formed() -> None:
    for role, template in elements.ROLE_TEMPLATES.items():
        assert template.site in elements.SITES, role
        assert template.value_type in ("float", "bool", "int", "waveform"), role
        assert template.writable == (template.site == "sp"), role
        assert (template.shape is not None) == (template.value_type == "waveform"), role
        if template.tolerance is not None:
            assert template.writable and template.value_type == "float", role


def test_every_injector_bpm_reads_in_millimetres() -> None:
    bpm_rows = [
        row
        for row in ROWS
        if row.template.device_class == "BeamPositionMonitor"
        and row.template.signal.startswith("position_")
    ]
    assert {row.section for row in bpm_rows} == set(INJECTOR_SECTIONS)
    assert {row.template.unit for row in bpm_rows} == {"mm"}


def test_the_subfield_agrees_with_the_role() -> None:
    for row in ROWS:
        sub = row.demo_name.rsplit(":", 1)[1]
        assert (sub == "SP") == row.template.writable, row
        if row.template.site == "rb":
            assert sub == "RB", row


def test_every_setpoint_has_its_readback() -> None:
    names = {row.demo_name for row in ROWS}
    for row in ROWS:
        if row.template.writable:
            assert row.demo_name[: -len("SP")] + "RB" in names, row.demo_name


def test_rows_sharing_a_name_agree() -> None:
    by_name: dict[str, set[tuple]] = defaultdict(set)
    for row in ROWS:
        by_name[row.demo_name].add((row.role, row.deck_element, row.nominal))
    assert {name: seen for name, seen in by_name.items() if len(seen) > 1} == {}


def test_a_legacy_name_carries_each_role_and_site_once() -> None:
    roles: dict[tuple[str, str], int] = defaultdict(int)
    sites: dict[tuple[str, str], int] = defaultdict(int)
    for row in ROWS:
        if row.legacy_pv is not None:
            roles[row.legacy_pv, row.role] += 1
            sites[row.legacy_pv, row.template.site] += 1
    assert [key for key, count in roles.items() if count > 1] == []
    assert [key for key, count in sites.items() if count > 1] == []


def test_every_nominal_reads_by_its_type() -> None:
    for row in ROWS:
        if row.template.value_type == "waveform":
            assert row.nominal is None, row
            continue
        assert row.nominal is not None, row
        value = row.nominal_value()
        if row.template.value_type == "bool":
            assert value in ("TRUE", "FALSE"), row


# --- the devices ----------------------------------------------------------------------


def _devices() -> dict[str, set[str]]:
    """Each device prefix (the first four tokens) and its field:subfield tails."""
    devices: dict[str, set[str]] = defaultdict(set)
    for row in ROWS:
        tokens = row.demo_name.split(":")
        devices[":".join(tokens[:4])].add(":".join(tokens[4:]))
    return devices


def test_every_injector_bpm_has_position_sum_and_valid() -> None:
    bpms = {device: tails for device, tails in _devices().items() if ":DIAG:BPM:" in device}
    assert {device.split(":")[0] for device in bpms} == set(INJECTOR_SECTIONS)
    for device, tails in bpms.items():
        assert {"POSITION:X", "POSITION:Y", "SUM:RB", "STATUS:VALID"} <= tails, device


def test_every_screen_has_paddle_and_lamp() -> None:
    screens = {device: tails for device, tails in _devices().items() if ":DIAG:TV:" in device}
    assert len(screens) == 21
    for device, tails in screens.items():
        assert tails == {"PADDLE:SP", "PADDLE:RB", "LAMP:RB"}, device


def test_every_magnet_setpoint_has_an_on_status() -> None:
    for device, tails in _devices().items():
        if ":MAG:" in device and "CURRENT:SP" in tails:
            assert "STATUS:ON" in tails, device


def test_the_frozen_bts_channels_are_rows() -> None:
    channels = read_yaml((EXAMPLE / "records/channels.yaml").read_text(encoding="utf-8"))
    frozen = {channel["id"] for channel in channels if channel["id"].startswith("BTS:")}
    assert len(frozen) >= 72
    assert frozen <= {row.demo_name for row in ROWS}


def test_nominals_agree_with_the_committed_seeds() -> None:
    seeds = read_yaml((EXAMPLE / "seeds.yaml").read_text(encoding="utf-8"))
    for row in ROWS:
        seed = seeds.get(row.demo_name) or {}
        if "nominal" not in seed:
            continue
        if row.template.value_type == "bool":
            assert str(seed["nominal"]) == row.nominal, row
        else:
            assert float(seed["nominal"]) == pytest.approx(float(row.nominal)), row


# --- the deck ---------------------------------------------------------------------------


def test_deck_elements_are_occurrence_indexed_or_blank() -> None:
    for row in ROWS:
        if row.deck_element is not None:
            assert elements.DECK_ELEMENT.match(row.deck_element), row
            assert row.section in elements.SECTIONS
    for element in elements.FIXED_ELEMENTS:
        assert elements.DECK_ELEMENT.match(element), element


def test_every_deck_element_resolves_to_one_occurrence() -> None:
    named = {row.deck_element for row in ROWS if row.deck_element} | set(elements.FIXED_ELEMENTS)
    resolved = deck_elements()
    assert sorted(named - resolved) == []


def test_every_counted_occurrence_is_bound_or_fixed() -> None:
    bound = {row.deck_element for row in ROWS if row.deck_element}
    assert sorted(counted_elements() - bound - set(elements.FIXED_ELEMENTS)) == []
    for element in (
        "LTB:BX#1",
        "LTB:BS#1",
        "LTB:BR1QD1#1",
        "LTB:BR1QF2#1",
        "LTB:BRKI#1",
        "BTS:SRBMP3#1",
        "BTS:SRBMP4#1",
    ):
        assert element in bound | set(elements.FIXED_ELEMENTS)


def test_one_device_per_deck_element() -> None:
    devices: dict[str, set[str]] = defaultdict(set)
    for row in ROWS:
        if row.deck_element:
            devices[row.deck_element].add(row.demo_name.rsplit(":", 2)[0])
    assert {element: names for element, names in devices.items() if len(names) > 1} == {}


def test_only_the_septa_are_both_bound_and_fixed() -> None:
    bound = {row.deck_element for row in ROWS if row.deck_element}
    assert bound & set(elements.FIXED_ELEMENTS) == set(SEPTA.values())


@pytest.mark.parametrize(("legacy", "element"), sorted(SEPTA.items()))
def test_each_septum_is_bound_explicitly_and_unwired(legacy: str, element: str) -> None:
    (row,) = [row for row in ROWS if row.legacy_pv == legacy]
    assert row.deck_element == element
    assert element in elements.FIXED_ELEMENTS
    assert not row.wired
    assert row.template.device_class == "Septum"


@pytest.mark.parametrize("family", ["HCM", "VCM"])
def test_bts_corrector_06_has_an_explicit_row(family: str) -> None:
    rows = [row for row in ROWS if row.demo_name == f"BTS:MAG:{family}:06:CURRENT:SP"]
    assert len(rows) == 1
    element = f"BTS:{family}#6"
    assert rows[0].deck_element == element or element in elements.FIXED_ELEMENTS
    assert rows[0].legacy_pv is None


@pytest.mark.parametrize("family", ["HCM", "VCM"])
@pytest.mark.parametrize("number", [7, 8, 9])
def test_bts_correctors_07_to_09_follow_the_panel(family: str, number: int) -> None:
    (row,) = [row for row in ROWS if row.legacy_pv == f"BTS:{family}{number}:Setpoint"]
    assert row.demo_name == f"BTS:MAG:{family}:{number:02d}:CURRENT:SP"
    assert row.deck_element == f"BTS:{family}#{number}"


def test_all_paddles_out_is_an_int_in_bts() -> None:
    (row,) = [row for row in ROWS if row.legacy_pv == "CCD:Paddles:Status"]
    assert row.section == "BTS"
    assert row.template.value_type == "int"
    assert row.nominal_value() == 0


def test_the_sr_injection_screen_sits_in_bts() -> None:
    rows = [row for row in ROWS if (row.legacy_pv or "").startswith("SR01C___TV1")]
    assert len(rows) == 3
    assert {row.section for row in rows} == {"BTS"}
