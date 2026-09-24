"""Tests for the ``PROFILE.md`` renderer.

``render_profile(census, votes)`` turns the import census and the direction
votes into Markdown: a facility section with the totals, the PN_LOCAL-illegal
system tokens and every shared PV, then one section per sub-machine carrying
every FR2 item. The synthetic export here has two systems that share a PV,
disagree on one direction and leave another undecided, and pends no judgment;
a second export pends one of each kind, so both sides of every judgment block
are pinned.

The ``Virtual accelerator`` heading is read from the one committed 2.0 export,
``tests/fixtures/mml/synthetic``, so the section is pinned against a real
export rather than a hand-typed one; the two shapes that fixture does not carry
(an extra field, sibling units that disagree) are made by editing the loaded
copy, never the committed file.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from osprey.services.mml.census import take_census
from osprey.services.mml.directions import vote_directions
from osprey.services.mml.normalize import normalize_family
from osprey.services.mml.profile import (
    HAZARD_HEADINGS,
    SYSTEM_HEADINGS,
    VA_HEADINGS,
    render_profile,
)
from tests.templates.mml_export_contract import EXPORTER_VERSION

SYNTHETIC = Path(__file__).resolve().parents[2] / "fixtures" / "mml" / "synthetic"


def _ao() -> dict:
    return {
        "_import_order": ["SR", "BR"],
        "_exports": {"SR": {"exporter": "1.0"}},
        "SR": {
            "BPM": {
                "FamilyName": "BPM",
                "Description": "Beam position monitors",
                "MemberOf": ["BPM", "Diagnostics"],
                "DeviceList": [[1, 1], [1, 2]],
                "Status": [1, 0],
                "Position": [1.5, "NaN"],
                "DeviceType": ["BPM", ""],
                "X": {
                    "MemberOf": ["BPM", "Monitor"],
                    "ChannelNames": ["SR:BPM1:X", "SR:BPM2:X"],
                    "HWUnits": [],
                    "Range": ["-Inf", "Inf"],
                },
                "Gain": {"ChannelNames": ["SR:BPM:GAIN"]},
            },
            "HCM": {
                "FamilyName": "HCM",
                "DeviceList": [[1, 1], [1, 2]],
                "Setpoint": {
                    "MemberOf": ["Setpoint"],
                    "ChannelNames": ["SR:HCM1:SP", "SR:HCM2:SP"],
                },
                "Readback": {
                    "ChannelNames": ["SHARED:RB", None],
                    "TangoNames": ["sr/hcm/1", "sr/hcm/2"],
                },
            },
            "Empty": {"FamilyName": "Empty", "CommonNames": ["a"]},
        },
        "BR": {
            "BPM": {
                "FamilyName": "BPM",
                "setup": {"DeviceList": [[1, 1]]},
                "X": {"MemberOf": ["Setpoint"], "ChannelNames": ["BR:BPM1:X"]},
            },
            "HCM": {
                "FamilyName": "HCM",
                "Readback": {"ChannelNames": ["SHARED:RB"]},
            },
        },
    }


def _ad() -> dict:
    return {
        "SR": {"Machine": "Quokka", "SubMachine": "SR", "Energy": 1.9},
        "BR": {"Machine": "Quokka", "SubMachine": "BR"},
    }


def _render() -> str:
    ao = _ao()
    return render_profile(take_census(ao, _ad()), vote_directions(ao))


def _pending_ao() -> dict:
    """One system pending every judgment kind, one pending nothing."""
    return {
        "_import_order": ["SR", "LTB"],
        "SR": {
            "DCCT": {
                "FamilyName": "DCCT",
                "DeviceList": [[1, 1]],
                "Monitor": {
                    "ChannelNames": ["SR:DCCT:AveI-I", "SR:DCCT:Lifetime-I"],
                },
            },
            "TUNE": {
                "FamilyName": "TUNE",
                "DeviceList": [[1, 1], [1, 2], [1, 3]],
                "Monitor": {"ChannelNames": ["SR:TUNE:Vx-I", "SR:TUNE:Vy-I"]},
            },
            "QM": {
                "FamilyName": "QM",
                "DeviceList": [[1, 1], [1, 2]],
                "Monitor": {"ChannelNames": ["p", "p"]},
                "Setpoint": {"ChannelNames": ["q", "q"]},
            },
        },
        "LTB": {
            "BPM": {
                "FamilyName": "BPM",
                "DeviceList": [[1, 1]],
                "X": {"ChannelNames": ["LTB:BPM1:X"]},
            }
        },
    }


def _pending_render() -> str:
    ao = _pending_ao()
    return render_profile(take_census(ao, None), vote_directions(ao))


def _synthetic(suffix: str) -> dict:
    """Return one committed file of the synthetic 2.0 export."""
    return json.loads((SYNTHETIC / f"quokka.sr.{suffix}.json").read_text())


def _synthetic_ao(raw: dict) -> dict:
    """Return the synthetic AO normalised and keyed by system, as a merge leaves it."""
    return {
        "SR": {
            name: (normalize_family(body) if isinstance(body, dict) else body)
            for name, body in raw.items()
            if name != "_export"
        },
        "_import_order": ["SR"],
    }


def _va_render(*, ao: dict | None = None, va: dict | None = None, **extra: object) -> str:
    """Render the profile of the synthetic 2.0 export, with both siblings."""
    merged = _synthetic_ao(ao if ao is not None else _synthetic("ao"))
    census = take_census(
        merged,
        {"SR": _synthetic("ad")},
        va={"SR": va if va is not None else _synthetic("va")},
        response={"SR": _synthetic("response")},
        **extra,
    )
    return render_profile(census, vote_directions(merged))


def _section(text: str, system: str) -> str:
    start = text.index(f"## System `{system}`")
    rest = text[start + 1 :]
    end = rest.find("\n## ")
    return rest if end == -1 else rest[:end]


def _va(text: str, system: str = "SR") -> str:
    """Slice the ``Virtual accelerator`` block out of one section."""
    return _section(text, system).split("### Virtual accelerator")[1]


def _judgments(text: str, system: str | None = None) -> str:
    """Slice the ``Judgment required`` block out of one section."""
    section = text[: text.index("## System `")] if system is None else _section(text, system)
    return section.split("### Judgment required")[1].split("### ")[0]


class TestHeadings:
    """Every FR2 item has a heading in every system section."""

    def test_title_and_facility_section(self):
        """The profile opens with a title and a facility section."""
        text = _render()
        assert text.startswith("# MML import profile\n")
        assert "\n## Facility\n" in text
        assert "\n### Totals\n" in text
        assert "\n### Shared PVs\n" in text
        assert "\n### System tokens not PN_LOCAL\n" in text

    def test_every_system_has_every_heading(self):
        """Both systems carry every system and hazard heading, in import order."""
        text = _render()
        assert text.index("## System `SR`") < text.index("## System `BR`")
        for system in ("SR", "BR"):
            section = _section(text, system)
            for heading in SYSTEM_HEADINGS:
                assert f"\n### {heading}\n" in section, (system, heading)
            for heading in HAZARD_HEADINGS:
                assert f"\n#### {heading}\n" in section, (system, heading)

    def test_fr2_headings_are_named(self):
        """The heading tables name every FR2 item."""
        assert SYSTEM_HEADINGS == (
            "Families",
            "Fields and channel keys",
            "Device counts",
            "Disabled devices",
            "MemberOf census",
            "Direction votes",
            "Judgment required",
            "Descriptions",
            "Hazards",
            "Position and DeviceType coverage",
            "Families with arrays from setup",
            "AD scalars",
            "Virtual accelerator",
        )
        assert VA_HEADINGS == (
            "Export facts",
            "Refused families",
            "Response",
            "Family coverage",
            "Sampled fields",
            "Other systems",
        )
        assert HAZARD_HEADINGS == (
            "Function handles",
            "Typo keys",
            "Non-finite ranges",
            "HWUnits and DataType shapes",
            "Case-duplicate families",
            "System token not PN_LOCAL",
            "Dual-key fields",
            "Empty channel lists",
            "Partial channel lists",
            "Broadcast rows",
            "Families with zero channels",
            "Shared PVs",
        )


class TestContent:
    """The census facts appear in the right section."""

    def test_shared_pv_owner_lines(self):
        """The shared PV lists every owner, in the facility and both systems."""
        text = _render()
        owner_sr = "  - `(SR, HCM, Readback, 1)`"
        owner_br = "  - `(BR, HCM, Readback, 1)`"
        facility = text[: text.index("## System `SR`")]
        assert "- `SHARED:RB`" in facility
        assert owner_sr in facility and owner_br in facility
        for system in ("SR", "BR"):
            section = _section(text, system)
            assert "- `SHARED:RB`" in section
            assert owner_sr in section and owner_br in section

    def test_shared_pv_owners_say_the_index_is_one_based(self):
        """Every ordinal the page prints is the 1-based one the mapping takes."""
        text = _render()

        assert "Owners are `(system, family, field, index)`; the index is 1-based." in text

    def test_fields_devices_and_disabled(self):
        """Fields show their keys, device counts their source, Status 0 its index."""
        sr = _section(_render(), "SR")
        assert "| HCM | Readback | ChannelNames, TangoNames |" in sr
        disabled = sr.split("### Disabled devices")[1].split("### MemberOf census")[0]
        assert "| Family | Disabled devices (1-based ordinal) |" in disabled
        assert "| BPM | 2 |" in disabled  # the 0-based index 1
        br = _section(_render(), "BR")
        assert "| HCM | 1 | fallback |" in br
        assert "- BPM" in br.split("### Families with arrays from setup")[1]

    def test_direction_disagreement_and_undecided(self):
        """BPM.X disagrees across systems; BPM.Gain is undecided; Readback votes read."""
        text = _render()
        sr = _section(text, "SR")
        votes = sr.split("### Direction votes")[1].split("### Descriptions")[0]
        assert "- `BPM.X`: BR write, SR read" in votes.split("#### Disagreements")[1]
        undecided = votes.split("#### Undecided")[1]
        for pair in ("`BPM.Gain`", "`BPM.X`"):
            assert f"- {pair}" in undecided
        assert "`HCM.Readback`" not in undecided
        assert "| HCM | Setpoint | write | write | memberof |" in votes
        br_votes = _section(text, "BR").split("#### Undecided")[1]
        assert "- `BPM.Gain`" not in br_votes

    def test_descriptions_hazards_coverage_ad(self):
        """Description coverage, hazards, coverage counts and AD scalars render."""
        sr = _section(_render(), "SR")
        assert "- BPM: Beam position monitors" in sr
        assert "- HCM" in sr.split("#### Families without descriptions")[1]
        hazards = sr.split("### Hazards")[1]
        assert "- Empty" in hazards.split("#### Families with zero channels")[1]
        assert "- `BPM.X.HWUnits`: empty" in hazards
        assert "- `HCM.Readback`" in hazards.split("#### Dual-key fields")[1]
        assert "| BPM | 1 | 1 | 1 | 1 |" in sr
        assert "| `Energy` | 1.9 |" in sr

    def test_empty_lists_say_none(self):
        """An empty hazard list is written as 'None.' rather than omitted."""
        br = _section(_render(), "BR")
        assert "#### Function handles\n\nNone.\n" in br

    def test_illegal_system_token(self):
        """A system token failing PN_LOCAL is listed facility-wide and in its section."""
        ao = {"S-R": {"BPM": {"X": {"ChannelNames": ["A"]}}}}
        text = render_profile(take_census(ao, None), vote_directions(ao))
        facility = text[: text.index("## System `")]
        assert "- `S-R`" in facility.split("### System tokens not PN_LOCAL")[1]
        assert "- `S-R`" in _section(text, "S-R").split("#### System token not PN_LOCAL")[1]

    def test_pipes_are_escaped(self):
        """Table cells escape pipes and newlines so a row stays one row."""
        ao = {"SR": {"BPM": {"Description": "a|b\nc", "X": {"ChannelNames": ["A"]}}}}
        text = render_profile(take_census(ao, None), vote_directions(ao))
        assert "a\\|b c" in text


class TestJudgmentRequired:
    """The judgments a reviewer still owes, per system and facility-wide."""

    def test_a_row_beyond_devices_names_its_field_key_signal_and_answers(self):
        """Each extra row is one question, with the three answers it takes."""
        block = _judgments(_pending_render(), "SR")
        assert (
            "  - row beyond devices `Monitor` `ChannelNames` `SR:DCCT:Lifetime-I`:"
            " `drop | device | field: <Name>`"
        ) in block
        assert "- `DCCT`, 1 device\n" in block

    def test_a_device_answer_says_broadcast_fields_reach_the_new_device(self):
        """The consequence of `device` is stated where the answer is offered."""
        assert (
            "  - `device` adds a device to the family, which every broadcast field also reaches."
        ) in _judgments(_pending_render(), "SR")

    def test_an_unbound_device_prints_its_export_ordinal(self):
        """The ordinal is the export one, the same key the mapping takes."""
        block = _judgments(_pending_render(), "SR")
        assert "- `TUNE`, 3 devices\n" in block
        assert "  - unbound device ordinal 3 `[1, 3]`: `drop | keep`" in block
        assert "Ordinals are the 1-based export ordinals the `judgments:` block takes." in block

    def test_an_unbound_device_without_a_device_list_row_prints_the_ordinal_alone(self):
        """A row the export did not state numerically leaves the ordinal bare."""
        ao = _pending_ao()
        ao["SR"]["TUNE"]["DeviceList"] = [[1, 1], [1, 2], ["", ""]]
        text = render_profile(take_census(ao, None), vote_directions(ao))
        assert "  - unbound device ordinal 3: `drop | keep`" in _judgments(text, "SR")

    def test_a_supply_group_names_its_members_rows_pvs_and_answers(self):
        """One group per index set, each fact on its own line under the answer."""
        block = _judgments(_pending_render(), "SR")
        assert "  - supply group 1: `keep_all | {1: <owning ordinal>}`" in block
        assert "    - members 1 `[1, 1]`, 2 `[1, 2]`" in block
        assert "    - PVs `p`, `q`" in block

    def test_a_supply_group_states_what_an_owner_answer_strands(self):
        """Both members of `Monitor [p, p]`/`Setpoint [q, q]` carry nothing else."""
        assert (
            "    - 2 of its 2 members carry no channel outside this group; an owner answer"
            " leaves 1 of them bound by nothing when the owner is one of them, else 2."
        ) in _judgments(_pending_render(), "SR")

    def test_a_group_stranding_nobody_drops_the_consequence_clause(self):
        """With nothing stranded the owner clause says nothing, so it is not printed."""
        ao = _pending_ao()
        ao["SR"]["QM"]["Setpoint"]["ChannelNames"] = ["q", "r"]
        text = render_profile(take_census(ao, None), vote_directions(ao))
        assert ("    - 0 of its 2 members carry no channel outside this group.") in _judgments(
            text, "SR"
        )
        assert "bound by nothing" not in _judgments(text, "SR")

    def test_the_facility_roll_up_prefixes_every_family_with_its_system(self):
        """The same families appear facility-wide, system-first."""
        facility = _judgments(_pending_render())
        for name in ("`SR.DCCT`, 1 device", "`SR.TUNE`, 3 devices", "`SR.QM`, 2 devices"):
            assert f"- {name}\n" in facility

    def test_a_system_pending_nothing_says_none(self):
        """LTB asks nothing, and says so rather than omitting the block."""
        assert _judgments(_pending_render(), "LTB").strip() == "None."

    def test_a_pending_free_export_says_none_in_every_block(self):
        """An export whose grain is decidable by rule pends nothing anywhere."""
        text = _render()
        assert _judgments(text).strip() == "None."
        for system in ("SR", "BR"):
            assert _judgments(text, system).strip() == "None."


class TestVirtualAccelerator:
    """The ``Virtual accelerator`` heading the VA MAP card's EXPORT box reads."""

    def test_every_va_heading_is_present_under_the_system_heading(self):
        """A system carrying a block carries every block heading, in order."""
        block = _va(_va_render())
        found = [heading for heading in VA_HEADINGS if f"\n#### {heading}\n" in block]
        assert found == list(VA_HEADINGS)

    def test_the_export_facts_are_one_label_and_value_each(self):
        """Deck, size, energy, calibration and nominal counts, one line each."""
        block = _va(_va_render())
        for row in (
            f"| Exporter | `{EXPORTER_VERSION}` |",
            "| Deck | `quokka_sr_deck` |",
            "| Elements | 43 |",
            "| Energy (GeV) | 2 |",
            "| Calibrations | linear 18, table 4 |",
            "| Nominals | 15 (4 synthetic) |",
        ):
            assert row in block, row

    def test_an_uncounted_cavity_is_unstated_rather_than_zero(self):
        """Nothing loads the deck at import yet, and the page says so."""
        assert "| Cavities | unstated |" in _va(_va_render())

    def test_a_counted_cavity_replaces_the_unstated_word(self):
        """Once a deck is counted the number is printed instead."""
        assert "| Cavities | 3 |" in _va(_va_render(ring_facts={"SR": {"cavities": 3}}))

    def test_the_deck_named_is_the_matlab_deck_not_the_served_copy(self):
        """The reviewer is sent back to the deck MATLAB exported from."""
        block = _va(_va_render())
        assert "`quokka_sr_deck`" in block
        assert "data/mml/lattice" not in block and "SR.mat" not in block

    def test_every_refused_family_is_a_warning_line_with_its_reason(self):
        """A refusal is what sends the reviewer back to MATLAB, so it carries the reason."""
        block = _va(_va_render())
        assert (
            "- ⚠ `SEPTUM`: SEPTUM.Monitor: getpvmodel answered the nominal in Physics units,"
            " not the hardware units it was asked in."
        ) in block
        assert "- ⚠ `TUNE`: Family TUNE lists no devices" in block
        assert "- ⚠ `Version`: Invalid input argument" in block

    def test_a_family_that_is_only_a_refusal_is_still_a_coverage_row(self):
        """It binds nothing, so the card can group it with every other such family."""
        block = _va(_va_render())
        assert "| Version | 0 | unstated | 0 | 0 | no | none | none | none |" in block
        assert "- ⚠ `Version`:" in block

    def test_a_response_block_states_its_sides_origin_size_and_timestamp(self):
        """The card's export box names where the matrix came from and how big it is."""
        assert "| BPMx | HC | model | 4 | 4 | 2026-09-17T09:00:00 |" in _va(_va_render())

    def test_family_coverage_states_the_verdict_inputs(self):
        """AT coverage per family: device rows covered and elements behind them."""
        block = _va(_va_render())
        assert "| HC | 4 | HCM | 4 | 7 | no | none | none | none |" in block
        assert "| BEND | 4 | BEND | 4 | 4 | yes | none | none | none |" in block

    def test_a_family_outside_the_deck_states_no_type_rather_than_a_blank(self):
        """BSOFT has no AT block; the cell says so instead of reading empty."""
        assert "| BSOFT | 2 | unstated | 0 | 0 | yes | none | none | none |" in _va(_va_render())

    def test_hooks_are_listed_against_the_family_that_carries_them(self):
        """A SpecialFunction or parameter-group hook is informational, and named."""
        block = _va(_va_render())
        assert "SpecialFunctionSet: qk_setidgap; ATParameterGroup: BendingAngle" in block

    def test_no_verdict_is_rendered_here(self):
        """Verdicts are the mapping's; the profile only states what they read."""
        block = _va(_va_render())
        assert "couple" not in block and "latch" not in block

    def test_a_sampled_field_states_its_calibration_nominal_and_units(self):
        """One row per sampled field, with the grid the calibration was built on."""
        block = _va(_va_render())
        assert "| SEPTUM | Monitor | linear | fallback | Physics | synthetic | Volt |" in block
        assert "| QF | Setpoint | linear | range | Hardware | sampled | 1/m^2 |" in block
        assert "| QF | Monitor | linear | range | unstated | unstated | 1/m^2 |" in block

    def test_an_extra_field_and_disagreeing_sibling_units_are_reported(self):
        """Both are verdict inputs the committed fixture does not carry."""
        raw = _synthetic("ao")
        raw["QF"]["Monitor"]["PhysicsUnits"] = "Radian"
        va = _synthetic("va")
        va["families"]["QF"]["fields"] = [*va["families"]["QF"]["fields"], "Desired"]
        block = _va(_va_render(ao=raw, va=va))
        assert "| QF | 4 | K | 4 | 4 | no | Desired | Setpoint 1/m^2, Monitor Radian | none |" in (
            block
        )

    def test_a_system_without_a_block_says_so_in_one_line(self):
        """A 1.0 export states nothing about a virtual accelerator."""
        assert _va(_render(), "SR").strip() == "no 2.0 export"
        assert _va(_render(), "BR").strip() == "no 2.0 export"

    def test_another_systems_block_is_one_line_under_the_mapped_system(self):
        """The VA lane reads one system's block; the rest are a glance, not a section."""
        block = _va(_two_system_render(_synthetic("va")))
        assert "- `LTB`: 17 families, 43 elements, 2 GeV" in block
        assert _va(_two_system_render(None), "SR").split("#### Other systems")[1].strip() == (
            "- `LTB`: no 2.0 export"
        )

    def test_a_single_family_block_is_counted_in_the_singular(self):
        """One family reads 'family', not 'families'."""
        va = copy.deepcopy(_synthetic("va"))
        va["families"] = {"QF": va["families"]["QF"]}
        assert "- `LTB`: 1 family, 43 elements, 2 GeV" in _va(_two_system_render(va))


def _two_system_render(ltb_va: dict | None) -> str:
    """Render two systems, only the second of which may carry a block."""
    merged = _synthetic_ao(_synthetic("ao"))
    merged["LTB"] = {"BPM": {"FamilyName": "BPM", "X": {"ChannelNames": ["LTB:BPM1:X"]}}}
    merged["_import_order"] = ["SR", "LTB"]
    va = {"SR": _synthetic("va")}
    if ltb_va is not None:
        va["LTB"] = ltb_va
    census = take_census(merged, {"SR": _synthetic("ad")}, va=va)
    return render_profile(census, vote_directions(merged))


class TestDeterminism:
    """Rendering is a pure function of its inputs."""

    def test_second_render_is_byte_identical(self):
        """Two renders of the same census are byte-identical."""
        assert _render().encode() == _render().encode()
        assert _pending_render().encode() == _pending_render().encode()
        assert _va_render().encode() == _va_render().encode()

    def test_vote_dict_order_does_not_matter(self):
        """Reversing the votes dict does not change the output."""
        ao = _ao()
        census = take_census(ao, _ad())
        votes = vote_directions(ao)
        reversed_votes = dict(reversed(list(votes.items())))
        assert render_profile(census, votes) == render_profile(census, reversed_votes)

    def test_ends_with_single_newline(self):
        """The document ends with exactly one newline."""
        text = _render()
        assert text.endswith("\n") and not text.endswith("\n\n")
