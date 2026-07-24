"""Tests for compact feedback-record formatting helpers."""

from __future__ import annotations

from osprey.services.channel_finder.feedback.formatters import (
    format_failure,
    format_selections,
    format_success,
)


class TestFormatSelections:
    def test_key_value_chain(self):
        assert format_selections({"system": "SR", "family": "BPM"}) == "system=SR, family=BPM"

    def test_empty_selections(self):
        assert format_selections({}) == ""


class TestFormatSuccess:
    def test_full_record(self):
        rec = {"selections": {"system": "SR"}, "channel_count": 4}
        assert format_success(rec) == "- GOOD: system=SR → 4 channels"

    def test_defaults_when_fields_missing(self):
        # No selections and no count -> empty selection chain, zero channels.
        assert format_success({}) == "- GOOD:  → 0 channels"


class TestFormatFailure:
    def test_with_partial_selections_and_reason(self):
        rec = {"partial_selections": {"system": "SR"}, "reason": "wrong family"}
        assert format_failure(rec) == "- BAD: system=SR — wrong family"

    def test_empty_partial_selections_renders_none_placeholder(self):
        rec = {"partial_selections": {}, "reason": "no match"}
        assert format_failure(rec) == "- BAD: (none) — no match"

    def test_default_reason_when_missing(self):
        assert format_failure({}) == "- BAD: (none) — rejected by operator"
