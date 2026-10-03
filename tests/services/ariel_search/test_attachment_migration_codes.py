"""The skip-reason codes the ``attachment_files_copy_state`` CHECK admits.

The migration writes its code list as a literal tuple, so an applied migration
never changes meaning when the registry grows. These tests pin that literal to
the registry sets as they stand: a new code in ``osprey.imaging.formats`` fails
here, which is the signal to write a new migration that replaces the
constraint rather than to edit this one.
"""

from __future__ import annotations

import re

from osprey.imaging.formats import (
    CONFIG_SKIP_REASONS,
    CONTENT_SKIP_REASONS,
    ROW_SKIP_REASONS,
    SOURCE_SKIP_REASONS,
    SUMMARY_ONLY_REASONS,
)
from osprey.services.ariel_search.database import attachment_migration, migrations
from osprey.services.ariel_search.database.attachment_migration import (
    COPY_STATE_SKIP_REASONS,
    AttachmentFilesCopyStateMigration,
)


class TestTheCheckCodeListIsALiteral:
    """``COPY_STATE_SKIP_REASONS`` matches the registry today and is a fixed tuple."""

    def test_equals_the_three_registry_sets(self):
        """The literal is exactly content ∪ config ∪ source codes, nothing more."""
        assert set(COPY_STATE_SKIP_REASONS) == (
            CONTENT_SKIP_REASONS | CONFIG_SKIP_REASONS | SOURCE_SKIP_REASONS
        )
        assert set(COPY_STATE_SKIP_REASONS) == ROW_SKIP_REASONS

    def test_is_a_tuple_without_duplicates(self):
        """A plain tuple of distinct strings -- not a reference to the live sets."""
        assert isinstance(COPY_STATE_SKIP_REASONS, tuple)
        assert len(COPY_STATE_SKIP_REASONS) == len(set(COPY_STATE_SKIP_REASONS))

    def test_summary_only_codes_are_not_admitted(self):
        """``no_source_url`` lives in summaries only, never on a stored row."""
        assert not set(COPY_STATE_SKIP_REASONS) & SUMMARY_ONLY_REASONS

    def test_written_literally_in_the_module_source(self):
        """Every code appears as a quoted literal in the migration's source.

        Guards against the tuple being rebuilt from the registry sets, which
        would let a registry edit silently change an applied migration.
        """
        from pathlib import Path

        source = Path(attachment_migration.__file__).read_text(encoding="utf-8")
        assert "osprey.imaging" not in source
        for code in COPY_STATE_SKIP_REASONS:
            assert f'"{code}"' in source

    def test_codes_are_plain_identifiers(self):
        """Codes are inlined into SQL, so each is a lowercase identifier."""
        for code in COPY_STATE_SKIP_REASONS:
            assert re.fullmatch(r"[a-z][a-z_]*", code), code


class TestCopyStateMigrationRegistration:
    """The migration is always on and ordered after both prerequisites."""

    def test_name_and_dependencies(self):
        """It depends on the table and on the entry text columns."""
        migration = AttachmentFilesCopyStateMigration()
        assert migration.name == "attachment_files_copy_state"
        assert migration.depends_on == ["attachment_files", "attachment_text_columns"]

    def test_registered_always_on(self):
        """The registry row needs no enhancement module."""
        rows = [r for r in migrations.KNOWN_MIGRATIONS if r[0] == "attachment_files_copy_state"]
        assert rows == [
            (
                "attachment_files_copy_state",
                "osprey.services.ariel_search.database.attachment_migration",
                "AttachmentFilesCopyStateMigration",
                None,
            )
        ]
