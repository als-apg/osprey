"""Tests for OKFDocument parse, serialize, and validate."""

from __future__ import annotations

import pytest

from osprey.services.facility_knowledge.okf.document import OKFDocument, OKFDocumentError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FULL_DOC = """\
---
type: facility_overview
title: ALS Accelerator Complex
description: Overview of the Advanced Light Source accelerator systems.
---

# ALS Accelerator Complex

The Advanced Light Source (ALS) is a synchrotron light source.
"""

TYPE_ONLY_DOC = """\
---
type: facility_overview
---

Just a body line.
"""

NO_FRONTMATTER_DOC = "# Just a heading\n\nSome body text.\n"


# ---------------------------------------------------------------------------
# OKFDocument.parse
# ---------------------------------------------------------------------------


class TestOKFDocumentParse:
    """Parse behaviour: round-trips, edge cases, and error paths."""

    def test_parse_full_document(self):
        doc = OKFDocument.parse(FULL_DOC)

        assert doc.frontmatter["type"] == "facility_overview"
        assert doc.frontmatter["title"] == "ALS Accelerator Complex"
        assert "description" in doc.frontmatter
        assert "ALS" in doc.body

    def test_parse_no_frontmatter(self):
        doc = OKFDocument.parse(NO_FRONTMATTER_DOC)

        assert doc.frontmatter == {}
        assert "Just a heading" in doc.body

    def test_parse_type_only_frontmatter(self):
        doc = OKFDocument.parse(TYPE_ONLY_DOC)

        assert doc.frontmatter == {"type": "facility_overview"}
        assert "Just a body line." in doc.body

    def test_parse_empty_string(self):
        doc = OKFDocument.parse("")

        assert doc.frontmatter == {}
        assert doc.body == ""

    def test_parse_unterminated_frontmatter_raises(self):
        bad = "---\ntype: facility_overview\ntitle: Missing close delimiter\n"

        with pytest.raises(OKFDocumentError, match="Unterminated"):
            OKFDocument.parse(bad)

    def test_parse_non_mapping_frontmatter_raises(self):
        bad = "---\n- item1\n- item2\n---\n\nBody.\n"

        with pytest.raises(OKFDocumentError, match="mapping"):
            OKFDocument.parse(bad)

    def test_parse_invalid_yaml_raises(self):
        bad = "---\nkey: [\nunclosed bracket\n---\n\nBody.\n"

        with pytest.raises(OKFDocumentError, match="Invalid YAML"):
            OKFDocument.parse(bad)


# ---------------------------------------------------------------------------
# OKFDocument.serialize
# ---------------------------------------------------------------------------


class TestOKFDocumentSerialize:
    """Serialize produces syntactically correct OKF documents."""

    def test_serialize_produces_delimiter(self):
        doc = OKFDocument(frontmatter={"type": "note"}, body="Hello.\n")
        result = doc.serialize()

        assert result.startswith("---\n")
        assert "\n---\n" in result

    def test_serialize_includes_body(self):
        doc = OKFDocument(frontmatter={"type": "note"}, body="Body content.\n")
        result = doc.serialize()

        assert "Body content." in result

    def test_serialize_body_without_trailing_newline(self):
        doc = OKFDocument(frontmatter={"type": "note"}, body="No newline")
        result = doc.serialize()

        assert result.endswith("\n")


# ---------------------------------------------------------------------------
# Round-trip: parse → serialize → parse
# ---------------------------------------------------------------------------


class TestOKFDocumentRoundTrip:
    """Parsed documents survive a serialize → parse round-trip."""

    def test_round_trip_full_document(self):
        original = OKFDocument.parse(FULL_DOC)
        serialized = original.serialize()
        recovered = OKFDocument.parse(serialized)

        assert recovered.frontmatter == original.frontmatter
        assert recovered.body.strip() == original.body.strip()

    def test_round_trip_type_only(self):
        original = OKFDocument.parse(TYPE_ONLY_DOC)
        serialized = original.serialize()
        recovered = OKFDocument.parse(serialized)

        assert recovered.frontmatter == original.frontmatter
        assert recovered.body.strip() == original.body.strip()


# ---------------------------------------------------------------------------
# OKFDocument.validate
# ---------------------------------------------------------------------------


class TestOKFDocumentValidate:
    """Validate enforces the correct required keys per level."""

    # --- consuming level ---

    def test_consuming_passes_type_only(self):
        doc = OKFDocument.parse(TYPE_ONLY_DOC)
        doc.validate("consuming")  # must not raise

    def test_consuming_passes_full_document(self):
        doc = OKFDocument.parse(FULL_DOC)
        doc.validate("consuming")  # must not raise

    def test_consuming_fails_missing_type(self):
        doc = OKFDocument(frontmatter={"title": "A title"})

        with pytest.raises(OKFDocumentError, match="type"):
            doc.validate("consuming")

    def test_consuming_fails_empty_type(self):
        doc = OKFDocument(frontmatter={"type": ""})

        with pytest.raises(OKFDocumentError, match="type"):
            doc.validate("consuming")

    # --- authoring level ---

    def test_authoring_fails_type_only(self):
        """A type-only doc must fail authoring (missing title + description)."""
        doc = OKFDocument.parse(TYPE_ONLY_DOC)

        with pytest.raises(OKFDocumentError):
            doc.validate("authoring")

    def test_authoring_passes_type_title_description_no_timestamp(self):
        """Three-key doc without timestamp must pass authoring (OKF §9)."""
        doc = OKFDocument(
            frontmatter={
                "type": "facility_overview",
                "title": "ALS",
                "description": "A synchrotron.",
            }
        )
        doc.validate("authoring")  # must not raise

    def test_authoring_fails_missing_title(self):
        doc = OKFDocument(frontmatter={"type": "facility_overview", "description": "Desc."})

        with pytest.raises(OKFDocumentError, match="title"):
            doc.validate("authoring")

    def test_authoring_fails_missing_description(self):
        doc = OKFDocument(frontmatter={"type": "facility_overview", "title": "Title"})

        with pytest.raises(OKFDocumentError, match="description"):
            doc.validate("authoring")

    # --- unknown level ---

    def test_unknown_level_raises_value_error(self):
        doc = OKFDocument(frontmatter={"type": "x", "title": "t", "description": "d"})

        with pytest.raises(ValueError, match="Unknown validation level"):
            doc.validate("strict")  # type: ignore[arg-type]

    # --- default level ---

    def test_default_level_is_authoring(self):
        """Calling validate() with no args defaults to authoring strictness."""
        doc = OKFDocument.parse(TYPE_ONLY_DOC)

        with pytest.raises(OKFDocumentError):
            doc.validate()
