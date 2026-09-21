"""Tests for the MML direction voter.

``vote_directions`` proposes a read or write direction for every channel-bearing
``(family, field)`` of a merged ``ao`` dict. ``MemberOf`` tags decide first, the
field name's suffix second, and a field neither speaks for stays undecided. The
vote is the union over sub-machines, so a disagreement between systems is kept
visible in ``per_system`` instead of being settled silently.
"""

from __future__ import annotations

import pytest

from osprey.services.mml.directions import (
    GRAMMAR_SUFFIXES,
    MEMBEROF_WORDS,
    Vote,
    vote_directions,
)


def _field(member_of=None, channels=("SR:X",)) -> dict:
    body: dict = {"ChannelNames": list(channels)}
    if member_of is not None:
        body["MemberOf"] = member_of
    return body


def _ao(**systems: dict) -> dict:
    return dict(systems)


class TestRealFacilityShapedCases:
    """A real facility's field shapes vote as the MML convention reads them."""

    @pytest.mark.parametrize(
        ("field", "member_of", "direction", "source"),
        [
            ("On", ["SR", "Boolean Monitor"], "read", "memberof"),
            ("OnControl", ["SR", "Boolean Control"], "write", "memberof"),
            ("OnMonitor", ["SR", "Multi-Bit Boolean Monitor"], "read", "memberof"),
            ("Reset", None, "write", "grammar"),
            ("RampRate", None, None, "undecided"),
            ("Voltage", ["SR", "Monitor"], "read", "memberof"),
        ],
    )
    def test_case(self, field, member_of, direction, source):
        """Each field shaped like a real facility's gets the expected direction and source."""
        ao = _ao(SR={"PS": {"DeviceList": [[1, 1]], field: _field(member_of)}})
        vote = vote_directions(ao)[("PS", field)]
        assert vote == Vote(direction=direction, per_system={"SR": direction}, source=source)


class TestMemberOf:
    """Tags are matched by whitespace-split word and win over the field name."""

    @pytest.mark.parametrize(
        ("tags", "direction"),
        [
            (["Setpoint"], "write"),
            (["Control"], "write"),
            (["MachineConfig", "Save/Restore"], "write"),
            (["Monitor"], "read"),
            (["Monitor", "Setpoint"], None),
            (["Boolean Monitor", "Boolean Control"], None),
        ],
    )
    def test_words(self, tags, direction):
        """Direction words decide; both kinds present leaves the field undecided."""
        ao = _ao(SR={"F": {"Value": _field(tags)}})
        assert vote_directions(ao)[("F", "Value")].direction == direction

    def test_word_is_matched_whole_not_as_substring(self):
        """``Monitoring`` is not the word ``Monitor``; the name grammar decides."""
        ao = _ao(SR={"F": {"Setpoint": _field(["Monitoring"])}})
        vote = vote_directions(ao)[("F", "Setpoint")]
        assert (vote.direction, vote.source) == ("write", "grammar")

    def test_tags_win_over_field_name(self):
        """A ``Monitor`` tag on a field named ``...Setpoint`` reads."""
        ao = _ao(SR={"F": {"Setpoint": _field(["Monitor"])}})
        vote = vote_directions(ao)[("F", "Setpoint")]
        assert (vote.direction, vote.source) == ("read", "memberof")

    def test_conflicting_tags_do_not_fall_through_to_grammar(self):
        """Tags naming both directions stay undecided even when the name has a suffix."""
        ao = _ao(SR={"F": {"Monitor": _field(["Monitor", "Setpoint"])}})
        assert vote_directions(ao)[("F", "Monitor")].direction is None

    def test_tags_without_direction_words_fall_through(self):
        """Non-direction tags leave the field-name grammar to decide."""
        ao = _ao(SR={"F": {"AmpRB": _field(["HCM", "Save/Restore"])}})
        vote = vote_directions(ao)[("F", "AmpRB")]
        assert (vote.direction, vote.source) == ("read", "grammar")

    def test_bare_string_and_blank_slots_are_accepted(self):
        """A bare string tag and ``None`` slots are read without error."""
        ao = _ao(SR={"F": {"A": _field("Boolean Monitor"), "B": _field([None, "Setpoint", ""])}})
        votes = vote_directions(ao)
        assert votes[("F", "A")].direction == "read"
        assert votes[("F", "B")].direction == "write"

    def test_family_level_tags_do_not_vote(self):
        """Only the field's own ``MemberOf`` counts, never the family's."""
        ao = _ao(SR={"F": {"MemberOf": ["Monitor"], "Current": _field()}})
        assert vote_directions(ao)[("F", "Current")].direction is None

    def test_word_table_is_documented(self):
        """The word table names both directions and has a docstring source."""
        assert set(MEMBEROF_WORDS) == {"read", "write"}
        assert "Monitor" in MEMBEROF_WORDS["read"]
        assert {"Setpoint", "Control", "MachineConfig"} <= set(MEMBEROF_WORDS["write"])


class TestGrammar:
    """The field name decides by suffix only."""

    @pytest.mark.parametrize(
        ("field", "direction"),
        [
            ("Monitor", "read"),
            ("CurrentRB", "read"),
            ("GapRBV", "read"),
            ("Readback", "read"),
            ("Setpoint", "write"),
            ("GapSP", "write"),
            ("OnControl", "write"),
            ("Reset", "write"),
            ("On", None),
            ("Voltage", None),
            ("MonitorGain", None),
        ],
    )
    def test_suffix(self, field, direction):
        """Suffixes decide; bare ``On`` and prefix-only matches do not."""
        ao = _ao(SR={"F": {field: _field()}})
        vote = vote_directions(ao)[("F", field)]
        assert vote.direction == direction
        assert vote.source == ("grammar" if direction else "undecided")

    def test_suffix_table_never_names_on(self):
        """``On`` is the MML monitor, so no suffix table entry is bare ``On``."""
        assert all("On" not in suffixes for suffixes in GRAMMAR_SUFFIXES.values())


class TestUnion:
    """The vote is the union over sub-machines."""

    def test_agreement_yields_the_direction(self):
        """Two systems reading the same field agree."""
        ao = _ao(
            SR={"BPM": {"X": _field(["Monitor"])}},
            BR={"BPM": {"X": _field(["BPM", "Monitor"])}},
        )
        vote = vote_directions(ao)[("BPM", "X")]
        assert vote == Vote("read", {"BR": "read", "SR": "read"}, "memberof")

    def test_disagreement_yields_none_with_per_system(self):
        """Systems voting opposite directions leave the field undecided."""
        ao = _ao(
            SR={"PS": {"Current": _field(["Monitor"])}},
            BR={"PS": {"Current": _field(["Setpoint"])}},
        )
        vote = vote_directions(ao)[("PS", "Current")]
        assert vote == Vote(None, {"BR": "write", "SR": "read"}, "undecided")

    def test_undecided_system_does_not_veto(self):
        """A system with no evidence adds nothing to the union."""
        ao = _ao(
            SR={"PS": {"Current": _field(["Monitor"])}},
            BR={"PS": {"Current": _field()}},
        )
        vote = vote_directions(ao)[("PS", "Current")]
        assert vote == Vote("read", {"BR": None, "SR": "read"}, "memberof")

    def test_mixed_sources_report_memberof(self):
        """A direction reached by tags in any system reports ``memberof``."""
        ao = _ao(
            SR={"PS": {"CurrentRB": _field()}},
            BR={"PS": {"CurrentRB": _field(["Monitor"])}},
        )
        vote = vote_directions(ao)[("PS", "CurrentRB")]
        assert (vote.direction, vote.source) == ("read", "memberof")

    def test_field_only_in_one_system(self):
        """``per_system`` names only the systems that carry the field."""
        ao = _ao(
            SR={"PS": {"Current": _field(["Monitor"])}},
            BR={"PS": {"Other": _field(["Setpoint"])}},
        )
        votes = vote_directions(ao)
        assert votes[("PS", "Current")].per_system == {"SR": "read"}
        assert votes[("PS", "Other")].per_system == {"BR": "write"}


class TestShape:
    """Only channel-bearing fields of real families are voted on."""

    def test_underscore_keys_are_skipped(self):
        """``_exports``, ``_import_order`` and a system ``_description`` are not voted."""
        ao = {
            "_exports": {"SR": {"exporter": "mml_export 1.0.0"}},
            "_import_order": ["SR"],
            "SR": {"_description": "Storage ring", "BPM": {"X": _field(["Monitor"])}},
        }
        assert list(vote_directions(ao)) == [("BPM", "X")]

    def test_fields_without_channels_and_setup_are_skipped(self):
        """A field with no channel key and the ``setup`` block get no vote."""
        ao = _ao(
            SR={
                "BEND": {
                    "setup": {"DeviceList": [[1, 1]]},
                    "Monitor": {"MemberOf": ["Monitor"], "HWUnits": "Amps"},
                }
            }
        )
        assert vote_directions(ao) == {}

    def test_tango_names_count_as_channels(self):
        """A ``TangoNames``-only field is voted on."""
        ao = _ao(RING={"PS": {"Current": {"TangoNames": ["a/b/c"], "MemberOf": ["Monitor"]}}})
        assert vote_directions(ao)[("PS", "Current")].direction == "read"

    def test_keys_are_sorted_and_input_untouched(self):
        """Output keys are sorted, and the input dict is not modified."""
        ao = _ao(SR={"Z": {"B": _field()}, "A": {"Setpoint": _field(), "Monitor": _field()}})
        before = repr(ao)
        votes = vote_directions(ao)
        assert list(votes) == [("A", "Monitor"), ("A", "Setpoint"), ("Z", "B")]
        assert repr(ao) == before

    def test_empty_ao(self):
        """An empty export yields no votes."""
        assert vote_directions({}) == {}
