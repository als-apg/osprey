"""Tests for `WriteOutcome`, the single owned verdict of a channel write.

The enum is the vocabulary the whole write path speaks: the connector sets one
member, every consumer reads it, and nobody re-derives a verdict. These tests
pin the closed set of six words and their wire values, because a seventh word —
or a renamed value — silently changes what every consumer sees.
"""

import json
from enum import StrEnum

import osprey_connectors.control_system as control_system_package
from osprey.connectors.control_system.base import WriteOutcome

EXPECTED_OUTCOMES = {
    "REFUSED": "refused",
    "FAILED": "failed",
    "CONFIRMED": "confirmed",
    "MISMATCH": "mismatch",
    "UNCONFIRMED": "unconfirmed",
    "UNREQUESTED": "unrequested",
}


class TestWriteOutcomeMembers:
    def test_is_a_str_enum(self):
        assert issubclass(WriteOutcome, StrEnum)

    def test_exactly_six_members_with_the_documented_values(self):
        assert {member.name: member.value for member in WriteOutcome} == EXPECTED_OUTCOMES

    def test_values_are_lowercase(self):
        for member in WriteOutcome:
            assert member.value == member.value.lower()

    def test_members_compare_equal_to_their_word(self):
        # Consumers compare against the word (`outcome == "confirmed"`), and the
        # IPC codec puts the word on the wire, so str equality must hold.
        assert WriteOutcome.CONFIRMED == "confirmed"
        assert WriteOutcome.MISMATCH != "confirmed"

    def test_serialises_as_its_word(self):
        assert json.dumps({"outcome": WriteOutcome.UNREQUESTED}) == '{"outcome": "unrequested"}'

    def test_a_word_round_trips_back_to_its_member(self):
        for word in EXPECTED_OUTCOMES.values():
            assert WriteOutcome(word) is getattr(WriteOutcome, word.upper())

    def test_docstring_explains_every_member(self):
        doc = WriteOutcome.__doc__ or ""
        for name in EXPECTED_OUTCOMES:
            assert name in doc


class TestWriteOutcomeExport:
    def test_exported_from_the_control_system_package(self):
        assert control_system_package.WriteOutcome is WriteOutcome

    def test_listed_in_package_all(self):
        assert "WriteOutcome" in control_system_package.__all__
