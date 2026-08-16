"""Tests for the qmd markdown mirror writer.

Covers the three load-bearing properties: invertible filenames for hostile
entry ids, sanitized and capped bodies for pathological content, and the
byte-compare that keeps unchanged entries from being rewritten.
"""

import os
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from osprey.services.ariel_search.enhancement.qmd_export.writer import (
    BODY_CAP_BYTES,
    MAX_ENCODED_ID_CHARS,
    TRUNCATION_MARKER,
    decode_entry_id,
    encode_entry_id,
    entry_id_from_path,
    mirror_path,
    render_entry,
    sanitize_text,
    write_entry,
)

# Ids that must survive a round trip through the filesystem.
HOSTILE_IDS = [
    "a/b",
    "..",
    ".",
    "../../etc/passwd",
    "/absolute",
    "\\windows\\path",
    "測定 2024",
    "emoji \U0001f600",
    "with space",
    "%2F",
    "%",
    "with\x00nul",
    "with\nnewline",
    "AbC-vs-abc",
    "ABC-VS-abc",
    "~tilde~",
    "a" * 200,
    "é" * 33,
]

# Ids that cannot be mirrored under an invertible name and must be refused.
OVERLONG_IDS = ["a" * 201, "b" * 300, "é" * 34, "é" * 300]


def make_entry(**overrides):
    """Build an ``enhanced_entries`` row shaped like real ARIEL data.

    Args:
        **overrides: Field values replacing the defaults.

    Returns:
        A row dict suitable for the writer's public functions.
    """
    entry = {
        "entry_id": "12345",
        "source_system": "ALS eLog",
        "timestamp": datetime(2024, 5, 17, 13, 45, 9, tzinfo=UTC),
        "author": "jdoe",
        "raw_text": "Beam lost at 13:45. Recovered after RF trip reset.",
        "attachments": [],
        "metadata": {"category": "operations", "level": "normal"},
        "created_at": datetime(2024, 5, 17, 13, 46, tzinfo=UTC),
        "updated_at": datetime(2024, 5, 17, 13, 46, tzinfo=UTC),
    }
    entry.update(overrides)
    return entry


class TestEncodeDecode:
    """Filename encoding must be lossless and invertible."""

    @pytest.mark.parametrize("entry_id", HOSTILE_IDS)
    def test_round_trip(self, entry_id):
        """Every hostile id decodes back to itself."""
        assert decode_entry_id(encode_entry_id(entry_id)) == entry_id

    @pytest.mark.parametrize("entry_id", HOSTILE_IDS)
    def test_encoding_has_no_path_syntax(self, entry_id):
        """Encoded ids carry no separator, dot, or leading-dot dotfile risk."""
        encoded = encode_entry_id(entry_id)
        assert "/" not in encoded
        assert "\\" not in encoded
        assert "." not in encoded
        assert not encoded.startswith("-")

    def test_empty_id_rejected(self):
        """An empty id has no representable filename."""
        with pytest.raises(ValueError, match="must not be empty"):
            encode_entry_id("")

    @pytest.mark.parametrize("entry_id", OVERLONG_IDS)
    def test_overlong_id_rejected(self, entry_id):
        """Ids too long to name are refused, not truncated into ambiguity."""
        with pytest.raises(ValueError, match="too long to mirror"):
            encode_entry_id(entry_id)

    def test_length_limit_boundary(self):
        """The limit counts encoded characters, and the boundary is inclusive."""
        assert len(encode_entry_id("a" * MAX_ENCODED_ID_CHARS)) == MAX_ENCODED_ID_CHARS
        with pytest.raises(ValueError):
            encode_entry_id("a" * (MAX_ENCODED_ID_CHARS + 1))

    def test_case_differences_do_not_collide(self):
        """Ids differing only by case get distinct names on any filesystem."""
        lower = encode_entry_id("abc")
        upper = encode_entry_id("ABC")
        assert lower != upper
        assert lower.lower() != upper.lower()

    def test_percent_literal_is_encoded(self):
        """A literal percent in an id does not become an escape on decode."""
        assert encode_entry_id("%41") == "%2541"
        assert decode_entry_id("%2541") == "%41"


class TestMirrorPath:
    """Path layout, sharding, and containment."""

    def test_shard_layout(self, tmp_path):
        """Entries land under YYYY/MM named by the encoded id."""
        path = mirror_path(tmp_path, make_entry())
        assert path.parent == tmp_path.resolve() / "2024" / "05"
        assert path.name == "12345.md"

    def test_shard_uses_utc(self, tmp_path):
        """A non-UTC timestamp shards by its UTC month, not its local one."""
        stamp = datetime(2024, 6, 1, 0, 30, tzinfo=timezone(timedelta(hours=2)))
        path = mirror_path(tmp_path, make_entry(timestamp=stamp))
        assert path.parent == tmp_path.resolve() / "2024" / "05"

    def test_missing_timestamp_uses_unknown_shard(self, tmp_path):
        """A row without a usable timestamp still gets a stable home."""
        path = mirror_path(tmp_path, make_entry(timestamp=None))
        assert path.parent == tmp_path.resolve() / "unknown" / "unknown"

    def test_iso_string_timestamp(self, tmp_path):
        """Timestamps arriving as ISO strings shard identically."""
        path = mirror_path(tmp_path, make_entry(timestamp="2024-05-17T13:45:09+00:00"))
        assert path.parent == tmp_path.resolve() / "2024" / "05"

    @pytest.mark.parametrize("entry_id", HOSTILE_IDS)
    def test_stays_inside_mirror_root(self, tmp_path, entry_id):
        """No id escapes the mirror root."""
        path = mirror_path(tmp_path, make_entry(entry_id=entry_id))
        assert path.is_relative_to(tmp_path.resolve())

    @pytest.mark.parametrize("entry_id", HOSTILE_IDS)
    def test_components_fit_the_filesystem(self, tmp_path, entry_id):
        """Every emitted path component fits NAME_MAX."""
        path = mirror_path(tmp_path, make_entry(entry_id=entry_id))
        relative = path.relative_to(tmp_path.resolve())
        for part in relative.parts:
            assert len(part.encode("utf-8")) <= 255

    @pytest.mark.parametrize("entry_id", HOSTILE_IDS)
    def test_path_round_trips_to_entry_id(self, tmp_path, entry_id):
        """Hydration recovers the original id from the mirrored path."""
        path = mirror_path(tmp_path, make_entry(entry_id=entry_id))
        assert entry_id_from_path(path, tmp_path.resolve()) == entry_id

    def test_distinct_ids_get_distinct_paths(self, tmp_path):
        """Ids that share a prefix still occupy distinct files."""
        long_path = mirror_path(tmp_path, make_entry(entry_id="a" * 200))
        prefix_path = mirror_path(tmp_path, make_entry(entry_id="a" * 190))
        assert long_path != prefix_path

    def test_empty_entry_id_rejected(self, tmp_path):
        """A row with no primary key is refused, not silently mirrored."""
        with pytest.raises(ValueError):
            mirror_path(tmp_path, make_entry(entry_id=""))

    def test_overlong_entry_id_rejected(self, tmp_path):
        """An unnameable id fails loudly so the caller can log and skip it."""
        with pytest.raises(ValueError, match="too long to mirror"):
            mirror_path(tmp_path, make_entry(entry_id="a" * 300))

    def test_container_path_decodes_without_root(self, tmp_path):
        """A sidecar-namespace path hydrates without a matching host root."""
        path = mirror_path(tmp_path, make_entry(entry_id="a/b"))
        container_path = Path("/corpus/ariel") / path.relative_to(tmp_path.resolve())
        assert entry_id_from_path(container_path) == "a/b"

    def test_foreign_path_rejected(self, tmp_path):
        """Hydration refuses a path outside a mirror root it was given."""
        with pytest.raises(ValueError, match="outside the mirror root"):
            entry_id_from_path(tmp_path / "elsewhere.md", tmp_path / "mirror")

    def test_non_markdown_path_rejected(self, tmp_path):
        """Hydration refuses a file that is not a mirrored document."""
        with pytest.raises(ValueError, match="not a mirrored markdown file"):
            entry_id_from_path(tmp_path / "2024" / "05" / "12345.txt", tmp_path)


class TestSanitize:
    """Sanitization of pathological content."""

    def test_strips_nul_and_control_chars(self):
        """NUL and other control characters never reach the file."""
        assert sanitize_text("a\x00b\x01c\x1fd\x7fe") == "abcde"

    def test_keeps_structural_whitespace(self):
        """Tab, newline and carriage return survive."""
        assert sanitize_text("a\tb\nc\rd") == "a\tb\nc\rd"

    def test_replaces_lone_surrogate(self):
        """Invalid UTF-8 is replaced rather than raising."""
        cleaned = sanitize_text("before\ud800after")
        cleaned.encode("utf-8")
        assert cleaned.startswith("before")
        assert cleaned.endswith("after")

    def test_none_becomes_empty(self):
        """A NULL column renders as empty text."""
        assert sanitize_text(None) == ""

    def test_preserves_unicode(self):
        """Legitimate unicode is untouched."""
        assert sanitize_text("β beam 測定 \U0001f600") == "β beam 測定 \U0001f600"


class TestRender:
    """Document rendering: body prose, determinism, cap."""

    def test_metadata_is_body_text_not_frontmatter(self):
        """Author, timestamp and source are prose; there is no frontmatter."""
        document = render_entry(make_entry())
        assert not document.startswith("---")
        assert "jdoe" in document
        assert "2024-05-17 13:45:09 UTC" in document
        assert "ALS eLog" in document
        assert "Beam lost at 13:45." in document

    def test_scalar_metadata_rendered_sorted(self):
        """Facility metadata is searchable and order-independent."""
        forward = render_entry(make_entry(metadata={"category": "ops", "level": "hi"}))
        reverse = render_entry(make_entry(metadata={"level": "hi", "category": "ops"}))
        assert "category: ops." in forward
        assert "level: hi." in forward
        assert forward == reverse

    def test_nested_metadata_skipped(self):
        """Nested JSON is left to Postgres, keeping rendering stable."""
        document = render_entry(make_entry(metadata={"tags": ["a", "b"], "n": 7}))
        assert "n: 7." in document
        assert "tags" not in document

    def test_missing_fields_degrade(self):
        """A sparse row still renders a usable document."""
        document = render_entry({"entry_id": "x"})
        assert "Logged by unknown on unknown time." in document
        assert "Source: unknown." in document

    def test_deterministic(self):
        """Rendering is a pure function of the row."""
        assert render_entry(make_entry()) == render_entry(make_entry())

    def test_ends_with_newline(self):
        """Documents are newline-terminated."""
        assert render_entry(make_entry()).endswith("\n")

    def test_control_characters_stripped_from_body(self):
        """Pathological raw_text is cleaned before it is written."""
        document = render_entry(make_entry(raw_text="bad\x00text\x07here"))
        assert "\x00" not in document
        assert "badtexthere" in document

    def test_cap_truncates_and_marks(self):
        """An oversized entry is capped and says so."""
        document = render_entry(make_entry(raw_text="x" * (BODY_CAP_BYTES * 2)))
        assert document.endswith(TRUNCATION_MARKER)
        assert len(document.encode("utf-8")) <= BODY_CAP_BYTES

    def test_cap_keeps_valid_utf8(self):
        """A multi-byte character split by the cut is dropped, not mangled."""
        document = render_entry(make_entry(raw_text="é" * BODY_CAP_BYTES))
        document.encode("utf-8")
        assert document.endswith(TRUNCATION_MARKER)
        assert len(document.encode("utf-8")) <= BODY_CAP_BYTES

    def test_realistic_entry_is_not_capped(self):
        """Real ALS entries are ~184 bytes; the cap is a tail guard only."""
        document = render_entry(make_entry())
        assert TRUNCATION_MARKER not in document
        assert len(document.encode("utf-8")) < 1024


class TestWriteEntry:
    """The byte-compare property and write mechanics."""

    def test_creates_file_and_parents(self, tmp_path):
        """The first write materializes the shard directories."""
        assert write_entry(tmp_path, make_entry()) is True
        path = tmp_path / "2024" / "05" / "12345.md"
        assert path.read_text(encoding="utf-8") == render_entry(make_entry())

    def test_unchanged_entry_is_not_rewritten(self, tmp_path):
        """Re-exporting an unchanged entry leaves the file's mtime alone."""
        write_entry(tmp_path, make_entry())
        path = tmp_path / "2024" / "05" / "12345.md"
        os.utime(path, (1_000_000_000, 1_000_000_000))
        before = path.stat()

        assert write_entry(tmp_path, make_entry()) is False

        after = path.stat()
        assert after.st_mtime_ns == before.st_mtime_ns
        assert after.st_ino == before.st_ino

    def test_changed_entry_is_rewritten(self, tmp_path):
        """A real content change does produce a write."""
        write_entry(tmp_path, make_entry())
        path = tmp_path / "2024" / "05" / "12345.md"
        os.utime(path, (1_000_000_000, 1_000_000_000))
        before = path.stat().st_mtime_ns

        assert write_entry(tmp_path, make_entry(raw_text="corrected text")) is True

        assert path.stat().st_mtime_ns != before
        assert "corrected text" in path.read_text(encoding="utf-8")

    def test_metadata_only_change_is_detected(self, tmp_path):
        """Changes outside raw_text still count as changes."""
        write_entry(tmp_path, make_entry())
        assert write_entry(tmp_path, make_entry(author="asmith")) is True

    def test_touched_timestamp_alone_is_not_a_change(self, tmp_path):
        """A bumped updated_at does not rewrite an otherwise identical entry."""
        write_entry(tmp_path, make_entry())
        later = datetime(2025, 1, 1, tzinfo=UTC)
        assert write_entry(tmp_path, make_entry(updated_at=later)) is False

    def test_no_temp_files_left_behind(self, tmp_path):
        """The atomic write cleans up after itself."""
        write_entry(tmp_path, make_entry())
        write_entry(tmp_path, make_entry(raw_text="second version"))
        assert [p.name for p in (tmp_path / "2024" / "05").iterdir()] == ["12345.md"]

    @pytest.mark.parametrize("entry_id", HOSTILE_IDS)
    def test_hostile_ids_write_and_reread(self, tmp_path, entry_id):
        """Every hostile id writes once, then compares equal."""
        entry = make_entry(entry_id=entry_id)
        assert write_entry(tmp_path, entry) is True
        assert write_entry(tmp_path, entry) is False

        path = mirror_path(tmp_path, entry)
        assert path.is_file()
        assert entry_id_from_path(path, tmp_path.resolve()) == entry_id

    def test_hostile_ids_do_not_collide(self, tmp_path):
        """Distinct ids occupy distinct files."""
        for index, entry_id in enumerate(HOSTILE_IDS):
            write_entry(tmp_path, make_entry(entry_id=entry_id, raw_text=str(index)))

        written = [p for p in tmp_path.rglob("*.md") if p.is_file()]
        assert len(written) == len(HOSTILE_IDS)

    def test_pathological_entry_written_once(self, tmp_path):
        """An oversized, control-char-laden entry is stable across passes."""
        entry = make_entry(
            entry_id="../../\x00 hostile",
            raw_text="\x00\x01" + "π" * (BODY_CAP_BYTES // 2),
            author="a\x07uthor",
        )
        assert write_entry(tmp_path, entry) is True
        assert write_entry(tmp_path, entry) is False

        path = mirror_path(tmp_path, entry)
        data = path.read_bytes()
        assert len(data) <= BODY_CAP_BYTES
        assert b"\x00" not in data
        assert data.decode("utf-8").endswith(TRUNCATION_MARKER)

    def test_existing_foreign_content_is_replaced(self, tmp_path):
        """A stale or corrupt mirror file is corrected on the next pass."""
        path = tmp_path / "2024" / "05" / "12345.md"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"stale content")

        assert write_entry(tmp_path, make_entry()) is True
        assert write_entry(tmp_path, make_entry()) is False

    def test_accepts_string_mirror_root(self, tmp_path):
        """The root may arrive as a configured string path."""
        assert write_entry(str(tmp_path), make_entry()) is True
        assert (Path(tmp_path) / "2024" / "05" / "12345.md").is_file()
