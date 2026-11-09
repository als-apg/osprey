"""Unit tests for the stored subject (C-SUBJ) and its readers."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from osprey.services.ariel_search.subject import SUBJECT_MAX_CHARS, derive_subject


class TestDeriveSubject:
    def test_als_metadata_subject(self):
        entry = {"raw_text": "Beam dump\n\ndetails", "metadata": {"subject": "Beam dump"}}
        assert derive_subject(entry) == "Beam dump"

    @pytest.mark.parametrize("facility", ["jlab", "ornl", "generic"])
    def test_title_facilities_use_metadata_title(self, facility):
        entry = {"raw_text": "Body first line", "metadata": {"title": f"{facility} title"}}
        assert derive_subject(entry) == f"{facility} title"

    def test_subject_wins_over_title(self):
        entry = {"raw_text": "x", "metadata": {"subject": "S", "title": "T"}}
        assert derive_subject(entry) == "S"

    def test_blank_subject_falls_through_to_title(self):
        entry = {"raw_text": "x", "metadata": {"subject": "  ", "title": "T"}}
        assert derive_subject(entry) == "T"

    def test_untitled_uses_first_non_blank_line_stripped(self):
        entry = {"raw_text": "\n   \n  Real first line  \nsecond", "metadata": {}}
        assert derive_subject(entry) == "Real first line"

    def test_first_line_capped(self):
        entry = {"raw_text": "x" * (SUBJECT_MAX_CHARS + 50), "metadata": {}}
        assert derive_subject(entry) == "x" * SUBJECT_MAX_CHARS
        assert SUBJECT_MAX_CHARS == 200

    @pytest.mark.parametrize("raw_text", ["", "   \n\t\n", None])
    def test_blank_text_gives_empty(self, raw_text):
        assert derive_subject({"raw_text": raw_text, "metadata": {}}) == ""

    def test_missing_keys_and_non_string_metadata(self):
        assert derive_subject({}) == ""
        assert derive_subject({"metadata": None, "raw_text": "a"}) == "a"
        assert derive_subject({"metadata": {"subject": 42}, "raw_text": "a"}) == "a"


def _row(**extra):
    now = datetime.now(UTC)
    return {
        "entry_id": "e1",
        "source_system": "Example eLog",
        "timestamp": now,
        "author": "a",
        "raw_text": "Body",
        "attachments": [],
        "metadata": {"subject": "Stored"},
        "created_at": now,
        "updated_at": now,
        **extra,
    }


class TestEntryFromRow:
    def test_carries_subject_and_has_images(self):
        from osprey.services.ariel_search.models import enhanced_entry_from_row

        entry = enhanced_entry_from_row(_row(subject="Stored", has_images=True))
        assert entry["subject"] == "Stored"
        assert entry["has_images"] is True

    def test_has_images_false_is_carried(self):
        from osprey.services.ariel_search.models import enhanced_entry_from_row

        assert enhanced_entry_from_row(_row(has_images=False))["has_images"] is False

    def test_absent_when_row_lacks_them(self):
        from osprey.services.ariel_search.models import enhanced_entry_from_row

        entry = enhanced_entry_from_row(_row(subject=None))
        assert "subject" not in entry
        assert "has_images" not in entry


class TestMcpOutputUnchanged:
    def test_format_entry_base_has_title_and_no_subject(self):
        from osprey.services.ariel_search.models import (
            _format_entry_base,
            enhanced_entry_from_row,
        )

        entry = enhanced_entry_from_row(
            _row(subject="Stored", has_images=True, metadata={"title": "Meta title"})
        )
        out = _format_entry_base(entry)
        assert "subject" not in out
        assert "has_images" not in out
        assert out["title"] == "Meta title"


class TestEntrySummaryReadsStoredSubject:
    def test_stored_subject_wins_over_text(self):
        from osprey.services.ariel_search.cli_operations import _entry_summary

        out = _entry_summary({"subject": "Stored", "raw_text": "Other line"})
        assert out["title"] == "Stored"

    def test_stored_subject_cut_at_100(self):
        from osprey.services.ariel_search.cli_operations import _entry_summary

        assert _entry_summary({"subject": "s" * 150})["title"] == "s" * 100

    def test_falls_back_to_derived_subject(self):
        from osprey.services.ariel_search.cli_operations import _entry_summary

        out = _entry_summary({"raw_text": "  \n First \nrest", "metadata": {}})
        assert out["title"] == "First"
        assert _entry_summary({"metadata": {"title": "T"}, "raw_text": "x"})["title"] == "T"
