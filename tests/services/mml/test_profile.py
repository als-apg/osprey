"""Tests for the ``PROFILE.md`` renderer.

``render_profile(census, votes)`` turns the import census and the direction
votes into Markdown: a facility section with the totals, the PN_LOCAL-illegal
system tokens and every shared PV, then one section per sub-machine carrying
every FR2 item. The synthetic export here has two systems that share a PV,
disagree on one direction and leave another undecided.
"""

from __future__ import annotations

from osprey.services.mml.census import take_census
from osprey.services.mml.directions import vote_directions
from osprey.services.mml.profile import HAZARD_HEADINGS, SYSTEM_HEADINGS, render_profile


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


def _section(text: str, system: str) -> str:
    start = text.index(f"## System `{system}`")
    rest = text[start + 1 :]
    end = rest.find("\n## ")
    return rest if end == -1 else rest[:end]


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
            "Descriptions",
            "Hazards",
            "Position and DeviceType coverage",
            "Families with arrays from setup",
            "AD scalars",
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
        owner_sr = "  - `(SR, HCM, Readback, 0)`"
        owner_br = "  - `(BR, HCM, Readback, 0)`"
        facility = text[: text.index("## System `SR`")]
        assert "- `SHARED:RB`" in facility
        assert owner_sr in facility and owner_br in facility
        for system in ("SR", "BR"):
            section = _section(text, system)
            assert "- `SHARED:RB`" in section
            assert owner_sr in section and owner_br in section

    def test_shared_pv_owners_say_the_index_is_zero_based(self):
        """Every corpus token is 1-based, so the raw slot index says which it is."""
        text = _render()

        assert "Owners are `(system, family, field, index)`; the index is 0-based." in text

    def test_fields_devices_and_disabled(self):
        """Fields show their keys, device counts their source, Status 0 its index."""
        sr = _section(_render(), "SR")
        assert "| HCM | Readback | ChannelNames, TangoNames |" in sr
        assert "| BPM | 1 |" in sr  # disabled device index 1
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


class TestDeterminism:
    """Rendering is a pure function of its inputs."""

    def test_second_render_is_byte_identical(self):
        """Two renders of the same census are byte-identical."""
        assert _render().encode() == _render().encode()

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
