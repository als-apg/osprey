"""Tests for the changelog-fragment gate and fold.

The fragment file's *name* carries meaning — its leading digits become the
issue reference in the shipped changelog — so most of what can go wrong here
goes wrong quietly: a fragment nobody can parse, a reference printed twice, a
date read as an issue number. Every rule therefore has a negative control, and
the sharp edges (``2026-cleanup.changed.md`` → ``(#2026)``; a slug fragment
allowed to write its own ``(#735, #737)``) are pinned as behaviour rather than
left to be rediscovered.

No test here creates a git repository.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "changelog_fragments.py"
_spec = importlib.util.spec_from_file_location("changelog_fragments", _MODULE_PATH)
assert _spec and _spec.loader
cf = importlib.util.module_from_spec(_spec)
# import-time required because scripts/ is not a package: changelog_fragments.py is
# loaded by path and registered in sys.modules before exec so @dataclass can resolve
# annotations through cls.__module__.
sys.modules[_spec.name] = cf
_spec.loader.exec_module(cf)


def write(directory: Path, name: str, text: str) -> Path:
    """Drop a fragment into *directory* and hand back its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def refuse_io(self, *args, **kwargs):
    """A `Path` method stand-in that fails the way a read-only file does."""
    raise PermissionError(13, "Permission denied")


class TestParseFragmentName:
    def test_an_issue_number_name_carries_the_ref(self):
        assert cf.parse_fragment_name("745.fixed.md") == ("745", "fixed", "745")

    def test_an_issue_number_with_a_slug_suffix_still_carries_the_ref(self):
        """`745-gate` names the issue and says which of its fixes this is."""
        assert cf.parse_fragment_name("745-gate.fixed.md") == ("745-gate", "fixed", "745")

    def test_a_slug_name_has_no_ref(self):
        assert cf.parse_fragment_name("gate.fixed.md") == ("gate", "fixed", None)

    def test_a_leading_date_is_read_as_an_issue_number(self):
        """Documented sharp edge: the leading digits ARE the issue reference.

        `2026-cleanup.changed.md` renders `(#2026)`. The rule cannot tell a
        year from an issue, so the README tells contributors not to start a
        name with a date.
        """
        assert cf.parse_fragment_name("2026-cleanup.changed.md") == (
            "2026-cleanup",
            "changed",
            "2026",
        )

    def test_underscores_and_digits_inside_a_slug_are_fine(self):
        assert cf.parse_fragment_name("web_terminal2.added.md") == (
            "web_terminal2",
            "added",
            None,
        )

    @pytest.mark.parametrize(
        "filename",
        [
            "745.Fixed.md",  # the type is lower-case only
            "issue.745.md",  # digits are not a type
            "745.fixed.txt",  # not markdown
            "745.fixed.md.bak",
            "-745.fixed.md",  # a name starts with a letter or a digit
            ".fixed.md",  # no name at all
            "745.md",  # no type segment
            "README.md",  # skipped by name, never parsed
        ],
    )
    def test_names_outside_the_grammar_are_rejected(self, filename):
        with pytest.raises(cf.FragmentError) as excinfo:
            cf.parse_fragment_name(filename)
        message = str(excinfo.value)
        assert message.startswith(f"{filename}: ")
        assert "<name>.<type>.md" in message

    def test_an_unknown_type_names_every_type(self):
        with pytest.raises(cf.FragmentError) as excinfo:
            cf.parse_fragment_name("745.tweaked.md")
        message = str(excinfo.value)
        assert 'unknown type "tweaked"' in message
        for type_ in cf.TYPES:
            assert type_ in message
        assert "renders nothing" in message

    @pytest.mark.parametrize("type_", cf.TYPES)
    def test_every_type_in_the_vocabulary_parses(self, type_):
        assert cf.parse_fragment_name(f"745.{type_}.md") == ("745", type_, "745")


class TestHeadingVocabulary:
    """`TYPE_HEADING_RE` spells the headings out by hand — it is part of the
    module contract — so it is the one place the rendered vocabulary can still
    drift from `TYPES`. Pin it to the derived tables."""

    def test_the_heading_regex_matches_exactly_the_rendered_headings(self):
        assert cf.HEADING_ORDER == (
            "Added",
            "Changed",
            "Deprecated",
            "Removed",
            "Fixed",
            "Security",
        )
        for heading in cf.HEADING_ORDER:
            assert cf.TYPE_HEADING_RE.match(f"### {heading}")
        assert not cf.TYPE_HEADING_RE.match("### Internal")
        assert cf.TYPE_HEADING_RE.pattern.count("|") == len(cf.HEADING_ORDER) - 1


class TestNormalizeLines:
    def test_crlf_becomes_lf(self):
        assert cf.normalize_lines("first\r\nsecond\r\n") == ["first", "second"]

    def test_a_lone_cr_becomes_lf(self):
        assert cf.normalize_lines("first\rsecond\r") == ["first", "second"]

    def test_leading_and_trailing_blank_lines_are_stripped(self):
        assert cf.normalize_lines("\n\n  \nbody\n\n   \n") == ["body"]

    def test_interior_blank_lines_survive(self):
        assert cf.normalize_lines("lead-in\n\n- sub\n") == ["lead-in", "", "- sub"]

    def test_interior_indentation_survives(self):
        assert cf.normalize_lines("lead-in\n    indented\n") == ["lead-in", "    indented"]

    def test_a_blank_file_is_no_lines(self):
        assert cf.normalize_lines("\n  \n\t\n") == []
        assert cf.normalize_lines("") == []


class TestOpeningParagraphEnd:
    def test_a_single_line_ends_at_itself(self):
        assert cf.opening_paragraph_end(["one sentence."]) == 0

    def test_a_hand_wrapped_paragraph_ends_at_its_last_line(self):
        assert cf.opening_paragraph_end(["wrapped at about", "seventy-eight columns."]) == 1

    def test_a_blank_line_ends_the_paragraph(self):
        lines = ["lead-in:", "", "- one", "- two"]
        assert cf.opening_paragraph_end(lines) == 0

    def test_a_column_zero_sub_bullet_ends_the_paragraph_without_a_blank_line(self):
        lines = ["lead-in:", "- one", "- two"]
        assert cf.opening_paragraph_end(lines) == 0

    @pytest.mark.parametrize("marker", ["- item", "* item", "+ item", "```python"])
    def test_every_column_zero_marker_ends_the_paragraph(self, marker):
        assert cf.opening_paragraph_end(["lead-in:", marker]) == 0

    def test_a_continuation_line_opening_with_bold_does_not_end_the_paragraph(self):
        """`**` is not a list marker — `[-*+]\\s` needs whitespace after it."""
        lines = ["A wrapped sentence that continues with", "**emphasis** and keeps going."]
        assert cf.opening_paragraph_end(lines) == 1

    def test_an_indented_sub_bullet_does_not_end_the_paragraph(self):
        """The patterns are anchored at column 0; an indented line is prose."""
        assert cf.opening_paragraph_end(["lead-in:", "  - indented"]) == 1

    def test_an_empty_body_yields_zero(self):
        assert cf.opening_paragraph_end([]) == 0


class TestLoadFragment:
    def test_a_plain_fragment_loads(self, tmp_path):
        path = write(tmp_path, "745.changed.md", "Changelog entries now live in fragments.\n")
        fragment = cf.load_fragment(path)
        assert (fragment.name, fragment.type, fragment.ref) == ("745", "changed", "745")
        assert fragment.lines == ("Changelog entries now live in fragments.",)
        assert fragment.path == path

    def test_the_body_is_normalized_on_the_way_in(self, tmp_path):
        path = write(tmp_path, "gate.fixed.md", "\r\n\r\nWindows wrote this.\r\n\r\n")
        assert cf.load_fragment(path).lines == ("Windows wrote this.",)

    def test_a_bold_opener_is_prose(self, tmp_path):
        path = write(tmp_path, "745.changed.md", "**Breaking change:** the flag is gone.\n")
        assert cf.load_fragment(path).lines[0] == "**Breaking change:** the flag is gone."

    def test_sub_bullets_and_fences_survive(self, tmp_path):
        body = "The gate now checks two things:\n\n- a fragment exists\n- nobody hand-edited\n"
        fragment = cf.load_fragment(write(tmp_path, "745.added.md", body))
        assert fragment.lines == (
            "The gate now checks two things:",
            "",
            "- a fragment exists",
            "- nobody hand-edited",
        )
        assert cf.opening_paragraph_end(fragment.lines) == 0

    @pytest.mark.parametrize(
        "first_line",
        ["- a bullet", "* a bullet", "+ a bullet", "# a heading", "###### a heading", "```", "~~~"],
    )
    def test_a_marker_on_the_first_line_is_rejected(self, tmp_path, first_line):
        path = write(tmp_path, "745.fixed.md", f"{first_line}\nrest\n")
        with pytest.raises(cf.FragmentError) as excinfo:
            cf.load_fragment(path)
        assert "must open with prose" in str(excinfo.value)

    @pytest.mark.parametrize("text", ["", "\n\n", "   \n\t\n"])
    def test_an_empty_body_is_rejected(self, tmp_path, text):
        path = write(tmp_path, "745.fixed.md", text)
        with pytest.raises(cf.FragmentError) as excinfo:
            cf.load_fragment(path)
        assert "write one user-facing sentence" in str(excinfo.value)

    def test_bytes_that_are_not_utf8_are_rejected(self, tmp_path):
        tmp_path.mkdir(parents=True, exist_ok=True)
        path = tmp_path / "745.fixed.md"
        path.write_bytes(b"caf\xe9 au lait\n")
        with pytest.raises(cf.FragmentError) as excinfo:
            cf.load_fragment(path)
        assert "not valid UTF-8" in str(excinfo.value)
        assert "745.fixed.md" in str(excinfo.value)

    def test_a_file_that_cannot_be_read_is_a_fragment_error(self, tmp_path, monkeypatch):
        """Contributor-fixable like every other malformed fragment, so
        `validate_dir` lists it instead of the gate dying on a traceback."""
        path = write(tmp_path, "745.fixed.md", "A bug.\n")
        monkeypatch.setattr(Path, "read_text", refuse_io)
        with pytest.raises(cf.FragmentError) as excinfo:
            cf.load_fragment(path)
        assert str(excinfo.value).startswith("745.fixed.md: cannot read")
        assert "Permission denied" in str(excinfo.value)

    def test_a_ref_bearing_name_may_not_repeat_the_ref(self, tmp_path):
        path = write(tmp_path, "745.fixed.md", "The gate no longer double-counts. (#745)\n")
        with pytest.raises(cf.FragmentError) as excinfo:
            cf.load_fragment(path)
        assert "comes from the filename" in str(excinfo.value)

    def test_the_ref_check_reads_the_opening_paragraph_not_the_last_line(self, tmp_path):
        body = "A wrapped sentence that runs on\nand ends with the reference. (#745)\n\n- detail\n"
        with pytest.raises(cf.FragmentError):
            cf.load_fragment(write(tmp_path, "745.fixed.md", body))

    def test_a_ref_further_down_the_body_is_left_alone(self, tmp_path):
        """Only the line the fold would append to is checked."""
        body = "The gate landed.\n\n- it supersedes the union driver (#729)\n"
        assert cf.load_fragment(write(tmp_path, "745.fixed.md", body)).ref == "745"

    def test_a_slug_fragment_may_write_its_own_refs(self, tmp_path):
        """No ref comes from `gate`, so the text has to supply one itself."""
        path = write(tmp_path, "gate.fixed.md", "Seven issues from the wave. (#735, #737)\n")
        fragment = cf.load_fragment(path)
        assert fragment.ref is None
        assert fragment.lines[-1].endswith("(#735, #737)")

    def test_a_slug_fragment_may_even_write_a_single_ref(self, tmp_path):
        path = write(tmp_path, "gate.fixed.md", "One issue, named by hand. (#735)\n")
        assert cf.load_fragment(path).ref is None

    def test_a_bad_name_is_rejected_before_the_file_is_read(self, tmp_path):
        path = write(tmp_path, "745.tweaked.md", "body\n")
        with pytest.raises(cf.FragmentError) as excinfo:
            cf.load_fragment(path)
        assert 'unknown type "tweaked"' in str(excinfo.value)

    def test_a_byte_order_mark_is_stripped(self, tmp_path):
        """Windows editors write one by default; it is invisible in the changelog."""
        path = tmp_path / "745.fixed.md"
        path.write_bytes("\ufeffThe gate no longer eats the last line.\n".encode("utf-8"))
        fragment = cf.load_fragment(path)
        assert fragment.lines[0] == "The gate no longer eats the last line."
        assert "\ufeff" not in "".join(cf.render_bullet(fragment))


class TestValidateDir:
    def test_a_missing_directory_is_empty_not_an_error(self, tmp_path):
        assert cf.validate_dir(tmp_path / "nope") == ([], [])

    def test_an_empty_directory_is_empty(self, tmp_path):
        (tmp_path / "changelog.d").mkdir()
        assert cf.validate_dir(tmp_path / "changelog.d") == ([], [])

    def test_the_readme_is_never_validated(self, tmp_path):
        """It opens with a heading, which would fail every body rule."""
        write(tmp_path, cf.KEEP, "# Changelog fragments\n\nHow to write one.\n")
        assert cf.validate_dir(tmp_path) == ([], [])

    def test_fragments_come_back_in_filename_order(self, tmp_path):
        write(tmp_path, "745.fixed.md", "Second alphabetically.\n")
        write(tmp_path, "100.added.md", "First alphabetically.\n")
        write(tmp_path, "gate.changed.md", "Last alphabetically.\n")
        fragments, errors = cf.validate_dir(tmp_path)
        assert errors == []
        assert [f.path.name for f in fragments] == [
            "100.added.md",
            "745.fixed.md",
            "gate.changed.md",
        ]

    def test_every_offender_is_reported_not_just_the_first(self, tmp_path):
        write(tmp_path, "745.tweaked.md", "bad type\n")
        write(tmp_path, "746.fixed.md", "- bad body\n")
        write(tmp_path, "747.fixed.md", "")
        write(tmp_path, "good.added.md", "This one is fine.\n")
        fragments, errors = cf.validate_dir(tmp_path)
        assert [f.path.name for f in fragments] == ["good.added.md"]
        assert len(errors) == 3
        assert [e.split(":")[0] for e in errors] == [
            "745.tweaked.md",
            "746.fixed.md",
            "747.fixed.md",
        ]

    def test_a_name_outside_the_grammar_is_an_error(self, tmp_path):
        write(tmp_path, "notes.md", "This is not a fragment.\n")
        fragments, errors = cf.validate_dir(tmp_path)
        assert fragments == []
        assert len(errors) == 1
        assert "<name>.<type>.md" in errors[0]
        for type_ in cf.TYPES:
            assert type_ in errors[0]

    def test_a_subdirectory_is_an_error(self, tmp_path):
        (tmp_path / "drafts").mkdir()
        write(tmp_path / "drafts", "745.fixed.md", "Hidden away.\n")
        fragments, errors = cf.validate_dir(tmp_path)
        assert fragments == []
        assert errors == [f"drafts/: {cf.FRAGMENT_DIR}/ is flat — no subdirectories"]

    def test_files_that_are_not_markdown_are_ignored(self, tmp_path):
        (tmp_path / ".DS_Store").write_bytes(b"\x00\x01")
        write(tmp_path, "notes.txt", "scratch\n")
        write(tmp_path, "745.fixed.md", "The one real fragment.\n")
        fragments, errors = cf.validate_dir(tmp_path)
        assert [f.path.name for f in fragments] == ["745.fixed.md"]
        assert errors == []

    def test_an_internal_fragment_is_valid(self, tmp_path):
        write(tmp_path, "745.internal.md", "Refactored the loader; nobody sees this.\n")
        fragments, errors = cf.validate_dir(tmp_path)
        assert errors == []
        assert fragments[0].type == "internal"


class TestUnreleasedSpan:
    """`## [Unreleased]` is located the same way by both subcommands."""

    CHANGELOG = [
        "# Changelog",
        "",
        "All notable changes are recorded here.",
        "",
        "## [Unreleased]",
        "",
        "### Fixed",
        "",
        "- the gate no longer double-counts (#745)",
        "",
        "## [2026.8.0] - 2026-08-20",
        "",
        "### Added",
        "",
        "- the first release",
    ]

    def test_the_span_brackets_the_block(self):
        heading, end = cf.unreleased_span(self.CHANGELOG)
        assert (heading, end) == (4, 10)
        assert self.CHANGELOG[heading] == "## [Unreleased]"
        assert self.CHANGELOG[heading + 1 : end] == [
            "",
            "### Fixed",
            "",
            "- the gate no longer double-counts (#745)",
            "",
        ]

    def test_a_heading_on_the_first_line_is_found(self):
        lines = ["## [Unreleased]", "", "- a bullet", "", "## [2026.8.0] - 2026-08-20"]
        assert cf.unreleased_span(lines) == (0, 4)

    def test_a_bracket_less_heading_does_not_end_the_block(self):
        """`## Notes` is prose inside the section — the terminator needs `[`."""
        lines = ["## [Unreleased]", "", "## Notes", "", "- still unreleased"]
        assert cf.unreleased_span(lines) == (0, 5)

    def test_a_release_heading_without_a_date_ends_the_block(self):
        """A release in preparation has no date yet; it still closes the section."""
        lines = ["## [Unreleased]", "", "## [2026.8.0]", "", "- shipped"]
        assert cf.unreleased_span(lines) == (0, 2)

    def test_the_last_section_runs_to_the_end_of_the_file(self):
        lines = ["# Changelog", "", "## [Unreleased]", "", "### Fixed", "", "- one"]
        assert cf.unreleased_span(lines) == (2, len(lines))

    def test_an_empty_block_still_has_a_span(self):
        lines = ["## [Unreleased]", "", "## [2026.8.0] - 2026-08-20"]
        heading, end = cf.unreleased_span(lines)
        assert (heading, end) == (0, 2)
        assert lines[heading + 1 : end] == [""]

    def test_a_heading_with_trailing_spaces_matches(self):
        lines = ["## [Unreleased]   ", "", "- a bullet"]
        assert cf.unreleased_span(lines) == (0, 3)

    def test_the_first_heading_wins(self):
        lines = ["## [Unreleased]", "", "## [2026.8.0]", "", "## [Unreleased]"]
        assert cf.unreleased_span(lines) == (0, 2)

    @pytest.mark.parametrize(
        "lines",
        [
            [],
            ["# Changelog", "", "## [2026.8.0] - 2026-08-20"],
            ["## [Unreleased] - 2026-08-20"],  # a date makes it a release heading
            ["### [Unreleased]"],  # wrong level
            ["## Unreleased"],  # no brackets
            ["  ## [Unreleased]"],  # not at column 0
        ],
    )
    def test_a_missing_heading_is_none(self, lines):
        assert cf.unreleased_span(lines) is None


class TestIsEmptyBlock:
    @pytest.mark.parametrize("block", [[], [""], ["", "  "], ["\t", "", "   "]])
    def test_blank_lines_only_is_empty(self, block):
        assert cf.is_empty_block(block) is True

    @pytest.mark.parametrize("block", [["### Fixed"], ["", "- a bullet", ""], ["  indented"]])
    def test_any_content_at_all_is_not_empty(self, block):
        assert cf.is_empty_block(block) is False

    def test_the_post_rotation_state_is_empty(self):
        """A release rotation leaves the section as exactly one blank line."""
        lines = ["## [Unreleased]", "", "## [2026.8.0] - 2026-08-20", "", "### Fixed", "", "- one"]
        heading, end = cf.unreleased_span(lines)
        assert cf.is_empty_block(lines[heading + 1 : end]) is True

    def test_a_populated_block_is_not_empty(self):
        lines = ["## [Unreleased]", "", "### Fixed", "", "- one", "", "## [2026.8.0] - 2026-08-20"]
        heading, end = cf.unreleased_span(lines)
        assert cf.is_empty_block(lines[heading + 1 : end]) is False


class TestRenderBullet:
    """The bullet is what ships, so its shape is pinned line by line.

    The shapes below are the ones `CHANGELOG.md` already contains: a one-line
    bullet, a hand-wrapped one, a lead-in with sub-bullets, and a bullet with a
    fenced block inside it (modelled on the `osprey profile new` entry under
    `## [2026.8.0]`). Each is checked as a whole list rather than by substring —
    a stray blank line or a lost indent is exactly the failure that would
    otherwise reach the changelog unnoticed.
    """

    @staticmethod
    def make(tmp_path, name: str, body: str) -> cf.Fragment:
        """Load a real fragment, so the ref comes from the name the fold sees."""
        return cf.load_fragment(write(tmp_path, name, body))

    def test_a_single_line_bullet_carries_its_ref(self, tmp_path):
        frag = self.make(tmp_path, "745.fixed.md", "The gate no longer double-counts.\n")
        assert cf.render_bullet(frag) == ["- The gate no longer double-counts. (#745)"]

    def test_a_hand_wrapped_bullet_indents_the_continuation_and_ends_with_the_ref(self, tmp_path):
        body = "Changelog entries now live in one small file per change instead of\nthe shared section.\n"
        frag = self.make(tmp_path, "745.changed.md", body)
        assert cf.render_bullet(frag) == [
            "- Changelog entries now live in one small file per change instead of",
            "  the shared section. (#745)",
        ]

    def test_a_lead_in_with_sub_bullets_puts_the_ref_on_the_lead_in(self, tmp_path):
        body = (
            "The gate now checks two things before a pull request can\n"
            "merge.\n"
            "\n"
            "- a fragment exists for the change\n"
            "- nobody wrote a bullet into the block by hand\n"
        )
        frag = self.make(tmp_path, "745.added.md", body)
        assert cf.render_bullet(frag) == [
            "- The gate now checks two things before a pull request can",
            "  merge. (#745)",
            "",
            "  - a fragment exists for the change",
            "  - nobody wrote a bullet into the block by hand",
        ]

    def test_a_blank_line_renders_as_an_empty_line(self, tmp_path):
        """Not two spaces — trailing whitespace is somebody's diff noise later."""
        frag = self.make(tmp_path, "745.added.md", "Lead-in.\n\n- one\n")
        assert cf.render_bullet(frag)[1] == ""

    def test_a_fenced_block_survives_apart_from_the_indent(self, tmp_path):
        body = (
            "`osprey build --emit-profile` is gone, with no alias — the build command\n"
            "builds projects, and profile authoring now has its own verb. Materialize a\n"
            "profile directory with:\n"
            "\n"
            "```\n"
            "osprey profile new DIR --preset X\n"
            "```\n"
            "\n"
            "It writes everything the flag wrote, plus the preset's `data/` tree.\n"
        )
        frag = self.make(tmp_path, "745.removed.md", body)
        assert cf.render_bullet(frag) == [
            "- `osprey build --emit-profile` is gone, with no alias — the build command",
            "  builds projects, and profile authoring now has its own verb. Materialize a",
            "  profile directory with: (#745)",
            "",
            "  ```",
            "  osprey profile new DIR --preset X",
            "  ```",
            "",
            "  It writes everything the flag wrote, plus the preset's `data/` tree.",
        ]

    def test_the_fence_body_is_copied_byte_for_byte(self, tmp_path):
        """Whatever is inside the fence is content, not prose to be reflowed."""
        body = "Run it with:\n\n```\n  osprey up --detach   # two leading spaces stay\n```\n"
        frag = self.make(tmp_path, "gate.added.md", body)
        assert cf.render_bullet(frag)[3] == "    osprey up --detach   # two leading spaces stay"

    def test_a_bold_continuation_line_does_not_end_the_paragraph(self, tmp_path):
        """`**` is not a list marker, so the ref still lands on the wrapped line."""
        body = "A wrapped sentence that continues with\n**emphasis** and keeps going.\n"
        frag = self.make(tmp_path, "745.changed.md", body)
        assert cf.render_bullet(frag) == [
            "- A wrapped sentence that continues with",
            "  **emphasis** and keeps going. (#745)",
        ]

    def test_a_bold_opener_is_rendered_as_written(self, tmp_path):
        frag = self.make(tmp_path, "745.changed.md", "**Breaking change:** the flag is gone.\n")
        assert cf.render_bullet(frag) == ["- **Breaking change:** the flag is gone. (#745)"]

    def test_a_slug_name_appends_nothing(self, tmp_path):
        frag = self.make(tmp_path, "gate.fixed.md", "A change with no issue behind it.\n")
        assert frag.ref is None
        assert cf.render_bullet(frag) == ["- A change with no issue behind it."]

    def test_a_slug_fragment_keeps_its_hand_written_refs(self, tmp_path):
        """The text supplies what the name cannot — verbatim, and only once."""
        body = "Seven issues from the wave landed together. (#735, #737)\n"
        frag = self.make(tmp_path, "gate.fixed.md", body)
        assert cf.render_bullet(frag) == [
            "- Seven issues from the wave landed together. (#735, #737)"
        ]

    def test_a_name_with_an_issue_and_a_slug_still_appends_the_issue(self, tmp_path):
        frag = self.make(tmp_path, "745-gate.fixed.md", "One of several fixes for the wave.\n")
        assert cf.render_bullet(frag) == ["- One of several fixes for the wave. (#745)"]

    def test_an_internal_fragment_is_never_rendered(self, tmp_path):
        frag = self.make(tmp_path, "745.internal.md", "Refactored the loader; nobody sees this.\n")
        with pytest.raises(cf.FragmentError) as excinfo:
            cf.render_bullet(frag)
        message = str(excinfo.value)
        assert message.startswith("745.internal.md: ")
        assert "renders nothing" in message

    def test_rendering_does_not_touch_the_fragment(self, tmp_path):
        """The ref is appended to a copy — `apply` may not mutate what it loaded."""
        frag = self.make(tmp_path, "745.fixed.md", "The gate no longer double-counts.\n")
        cf.render_bullet(frag)
        assert frag.lines == ("The gate no longer double-counts.",)


class TestApplyFragments:
    """The fold rewrites one section of a file nobody wants to re-read by hand.

    `CHANGELOG.md` is 6000 lines old and every release below `## [Unreleased]`
    is history, so the whole result is asserted rather than sampled: the
    interesting failures are a blank line gained or lost, a bullet landing
    under a released heading, and an existing heading being quietly moved.
    """

    CHANGELOG = (
        "# Changelog\n"
        "\n"
        "## [Unreleased]\n"
        "\n"
        "### Changed\n"
        "\n"
        "- an entry that was already here\n"
        "\n"
        "## [2026.8.0] - 2026-08-20\n"
        "\n"
        "### Fixed\n"
        "\n"
        "- something in the last release\n"
    )

    @staticmethod
    def make(tmp_path, name: str, body: str) -> cf.Fragment:
        """Load a real fragment, so the ref and body come from the file itself."""
        return cf.load_fragment(write(tmp_path, name, body))

    def test_a_bullet_goes_under_an_existing_heading(self, tmp_path):
        frag = self.make(tmp_path, "745.changed.md", "The fold writes the bullet.\n")
        new_text, report = cf.apply_fragments(self.CHANGELOG, [frag])
        assert new_text == (
            "# Changelog\n"
            "\n"
            "## [Unreleased]\n"
            "\n"
            "### Changed\n"
            "\n"
            "- The fold writes the bullet. (#745)\n"
            "- an entry that was already here\n"
            "\n"
            "## [2026.8.0] - 2026-08-20\n"
            "\n"
            "### Fixed\n"
            "\n"
            "- something in the last release\n"
        )
        assert report == ["changelog.d/745.changed.md -> ### Changed (#745)"]

    def test_a_missing_heading_is_created_at_the_top_of_the_block(self, tmp_path):
        frag = self.make(tmp_path, "745.added.md", "A new capability.\n")
        new_text, report = cf.apply_fragments(self.CHANGELOG, [frag])
        assert new_text == (
            "# Changelog\n"
            "\n"
            "## [Unreleased]\n"
            "\n"
            "### Added\n"
            "\n"
            "- A new capability. (#745)\n"
            "\n"
            "### Changed\n"
            "\n"
            "- an entry that was already here\n"
            "\n"
            "## [2026.8.0] - 2026-08-20\n"
            "\n"
            "### Fixed\n"
            "\n"
            "- something in the last release\n"
        )
        assert report == ["changelog.d/745.added.md -> ### Added (#745)"]

    def test_created_headings_go_above_an_existing_one_that_is_never_moved(self, tmp_path):
        """`### Changed` stays where it is, so the block ends up Added, Fixed, Changed.

        Keep-a-Changelog order applies to headings this run creates. Reordering
        what is already in the file would turn one release's fold into a diff
        across the whole section.
        """
        frags = [
            self.make(tmp_path, "745.added.md", "A new capability.\n"),
            self.make(tmp_path, "746.changed.md", "Different behaviour.\n"),
            self.make(tmp_path, "747.fixed.md", "A bug.\n"),
        ]
        new_text, report = cf.apply_fragments(self.CHANGELOG, frags)
        assert new_text == (
            "# Changelog\n"
            "\n"
            "## [Unreleased]\n"
            "\n"
            "### Added\n"
            "\n"
            "- A new capability. (#745)\n"
            "\n"
            "### Fixed\n"
            "\n"
            "- A bug. (#747)\n"
            "\n"
            "### Changed\n"
            "\n"
            "- Different behaviour. (#746)\n"
            "- an entry that was already here\n"
            "\n"
            "## [2026.8.0] - 2026-08-20\n"
            "\n"
            "### Fixed\n"
            "\n"
            "- something in the last release\n"
        )
        assert report == [
            "changelog.d/745.added.md -> ### Added (#745)",
            "changelog.d/746.changed.md -> ### Changed (#746)",
            "changelog.d/747.fixed.md -> ### Fixed (#747)",
        ]

    def test_a_heading_far_down_the_block_still_receives_its_own_bullet(self, tmp_path):
        text = (
            "## [Unreleased]\n"
            "\n"
            "### Added\n"
            "\n"
            "- one\n"
            "\n"
            "### Changed\n"
            "\n"
            "- two\n"
            "\n"
            "### Security\n"
            "\n"
            "- three\n"
            "\n"
            "## [2026.8.0] - 2026-08-20\n"
        )
        frag = self.make(tmp_path, "745.security.md", "A hardening change.\n")
        new_text, _ = cf.apply_fragments(text, [frag])
        assert new_text == (
            "## [Unreleased]\n"
            "\n"
            "### Added\n"
            "\n"
            "- one\n"
            "\n"
            "### Changed\n"
            "\n"
            "- two\n"
            "\n"
            "### Security\n"
            "\n"
            "- A hardening change. (#745)\n"
            "- three\n"
            "\n"
            "## [2026.8.0] - 2026-08-20\n"
        )

    def test_a_released_section_with_the_same_heading_is_byte_identical(self, tmp_path):
        """The search is confined to the block — history is not editable."""
        frag = self.make(tmp_path, "745.fixed.md", "A bug.\n")
        new_text, _ = cf.apply_fragments(self.CHANGELOG, [frag])
        tail = "## [2026.8.0] - 2026-08-20\n\n### Fixed\n\n- something in the last release\n"
        assert new_text.endswith(tail)
        assert new_text.count("### Fixed") == 2
        assert "- A bug. (#745)" in new_text.split("## [2026.8.0]")[0]

    def test_the_first_of_several_duplicate_headings_wins(self, tmp_path):
        text = "## [Unreleased]\n"
        for ordinal in ("first", "second", "third", "fourth", "fifth"):
            text += f"\n### Fixed\n\n- the {ordinal} block\n"
        text += "\n## [2026.8.0] - 2026-08-20\n"
        frag = self.make(tmp_path, "745.fixed.md", "A bug.\n")
        new_text, _ = cf.apply_fragments(text, [frag])
        assert new_text.count("### Fixed") == 5
        assert "### Fixed\n\n- A bug. (#745)\n- the first block\n" in new_text
        assert "- the second block" in new_text

    def test_an_empty_unreleased_section_gains_headings_without_double_blanks(self, tmp_path):
        """The post-rotation state is exactly one blank line; the fold reuses it."""
        text = (
            "# Changelog\n\n## [Unreleased]\n\n## [2026.8.0] - 2026-08-20\n\n### Fixed\n\n- old\n"
        )
        frags = [
            self.make(tmp_path, "745.added.md", "A new capability.\n"),
            self.make(tmp_path, "746.fixed.md", "A bug.\n"),
        ]
        new_text, _ = cf.apply_fragments(text, frags)
        assert new_text == (
            "# Changelog\n"
            "\n"
            "## [Unreleased]\n"
            "\n"
            "### Added\n"
            "\n"
            "- A new capability. (#745)\n"
            "\n"
            "### Fixed\n"
            "\n"
            "- A bug. (#746)\n"
            "\n"
            "## [2026.8.0] - 2026-08-20\n"
            "\n"
            "### Fixed\n"
            "\n"
            "- old\n"
        )
        assert "\n\n\n" not in new_text

    def test_an_unreleased_section_at_the_end_of_the_file_is_folded(self, tmp_path):
        """Nothing follows the block, so the fold supplies the blank line itself."""
        frag = self.make(tmp_path, "745.fixed.md", "A bug.\n")
        new_text, _ = cf.apply_fragments("# Changelog\n\n## [Unreleased]\n", [frag])
        assert new_text == "# Changelog\n\n## [Unreleased]\n\n### Fixed\n\n- A bug. (#745)\n"

    def test_a_heading_with_no_blank_line_after_it_gets_one(self, tmp_path):
        text = "## [Unreleased]\n### Fixed\n- an entry that was already here\n"
        frag = self.make(tmp_path, "745.fixed.md", "A bug.\n")
        new_text, _ = cf.apply_fragments(text, [frag])
        assert new_text == (
            "## [Unreleased]\n### Fixed\n\n- A bug. (#745)\n- an entry that was already here\n"
        )

    def test_an_internal_fragment_renders_nothing_and_creates_no_heading(self, tmp_path):
        frag = self.make(tmp_path, "745.internal.md", "Refactored the loader.\n")
        new_text, report = cf.apply_fragments(self.CHANGELOG, [frag])
        assert new_text == self.CHANGELOG
        assert report == ["changelog.d/745.internal.md -> internal, not rendered"]

    def test_an_internal_fragment_is_reported_alongside_the_rendered_ones(self, tmp_path):
        frags = [
            self.make(tmp_path, "745.internal.md", "Refactored the loader.\n"),
            self.make(tmp_path, "746.changed.md", "Different behaviour.\n"),
        ]
        new_text, report = cf.apply_fragments(self.CHANGELOG, frags)
        assert "Refactored the loader" not in new_text
        assert report == [
            "changelog.d/746.changed.md -> ### Changed (#746)",
            "changelog.d/745.internal.md -> internal, not rendered",
        ]

    def test_a_slug_fragment_is_reported_without_a_ref(self, tmp_path):
        frag = self.make(tmp_path, "gate.changed.md", "Something with no issue behind it.\n")
        _, report = cf.apply_fragments(self.CHANGELOG, [frag])
        assert report == ["changelog.d/gate.changed.md -> ### Changed"]

    def test_an_empty_fragment_list_changes_nothing(self):
        assert cf.apply_fragments(self.CHANGELOG, []) == (self.CHANGELOG, [])

    def test_a_changelog_without_the_unreleased_heading_raises(self, tmp_path):
        frag = self.make(tmp_path, "745.fixed.md", "A bug.\n")
        with pytest.raises(cf.FragmentError) as excinfo:
            cf.apply_fragments("# Changelog\n\n## [2026.8.0] - 2026-08-20\n", [frag])
        assert "no ## [Unreleased] heading" in str(excinfo.value)

    def test_bullets_within_a_heading_are_ordered_by_filename(self, tmp_path):
        """A plain string sort — `115` before `745`, digits before letters."""
        frags = [
            self.make(tmp_path, "802.fixed.md", "Third.\n"),
            self.make(tmp_path, "745.fixed.md", "Second.\n"),
            self.make(tmp_path, "gate.fixed.md", "Fourth.\n"),
            self.make(tmp_path, "115.fixed.md", "First.\n"),
        ]
        new_text, report = cf.apply_fragments(self.CHANGELOG, frags)
        assert "### Fixed\n\n- First. (#115)\n- Second. (#745)\n- Third. (#802)\n- Fourth.\n" in (
            new_text
        )
        assert report == [
            "changelog.d/115.fixed.md -> ### Fixed (#115)",
            "changelog.d/745.fixed.md -> ### Fixed (#745)",
            "changelog.d/802.fixed.md -> ### Fixed (#802)",
            "changelog.d/gate.fixed.md -> ### Fixed",
        ]

    def test_a_trailing_newline_does_not_separate_two_bullets(self, tmp_path):
        """`normalize_lines` already trimmed it; the fold must not put it back."""
        frags = [
            self.make(tmp_path, "745.changed.md", "First.\n\n\n"),
            self.make(tmp_path, "746.changed.md", "Second.\n"),
        ]
        new_text, _ = cf.apply_fragments(self.CHANGELOG, frags)
        assert "- First. (#745)\n- Second. (#746)\n- an entry that was already here\n" in new_text

    def test_a_multi_line_body_keeps_its_shape_inside_the_block(self, tmp_path):
        body = "A lead-in that wraps onto\na second line.\n\n- a sub-bullet\n"
        frag = self.make(tmp_path, "745.changed.md", body)
        new_text, _ = cf.apply_fragments(self.CHANGELOG, [frag])
        assert new_text == (
            "# Changelog\n"
            "\n"
            "## [Unreleased]\n"
            "\n"
            "### Changed\n"
            "\n"
            "- A lead-in that wraps onto\n"
            "  a second line. (#745)\n"
            "\n"
            "  - a sub-bullet\n"
            "- an entry that was already here\n"
            "\n"
            "## [2026.8.0] - 2026-08-20\n"
            "\n"
            "### Fixed\n"
            "\n"
            "- something in the last release\n"
        )

    def test_crlf_input_comes_back_as_lf(self, tmp_path):
        frag = self.make(tmp_path, "745.changed.md", "The fold writes the bullet.\n")
        new_text, _ = cf.apply_fragments(self.CHANGELOG.replace("\n", "\r\n"), [frag])
        assert "\r" not in new_text
        assert new_text.endswith("- something in the last release\n")

    EMPTY_HEADING = (
        "# Changelog\n"
        "\n"
        "## [Unreleased]\n"
        "\n"
        "### Fixed\n"
        "\n"
        "## [2026.8.0] - 2026-08-20\n"
        "\n"
        "### Fixed\n"
        "\n"
        "- something in the last release\n"
    )

    def test_a_heading_with_no_bullets_keeps_its_blank_line_below(self, tmp_path):
        """A shape the fold never creates but a hand-edit can leave behind."""
        fragment = self.make(tmp_path, "745.fixed.md", "A bug.\n")
        new_text, _ = cf.apply_fragments(self.EMPTY_HEADING, [fragment])
        assert new_text == (
            "# Changelog\n"
            "\n"
            "## [Unreleased]\n"
            "\n"
            "### Fixed\n"
            "\n"
            "- A bug. (#745)\n"
            "\n"
            "## [2026.8.0] - 2026-08-20\n"
            "\n"
            "### Fixed\n"
            "\n"
            "- something in the last release\n"
        )


class TestNeedsFragment:
    @pytest.mark.parametrize(
        "path",
        [
            "src/osprey/cli/build.py",
            "packages/osprey-edu/pyproject.toml",
            "src/",
            "packages/deep/nested/file.py",
        ],
    )
    def test_a_path_under_a_gated_directory_needs_one(self, path):
        assert cf.needs_fragment([path]) is True

    @pytest.mark.parametrize(
        "path",
        [
            "srcs/x.py",  # the prefix is a directory, not a string
            "docs/source/x.rst",  # `source` is not `src/`
            "src",  # the directory itself, unslashed
            "packages",
            "tests/src/x.py",  # gated only at the repo root
            "CHANGELOG.md",
        ],
    )
    def test_a_path_outside_them_does_not(self, path):
        assert cf.needs_fragment([path]) is False

    def test_one_gated_path_among_many_is_enough(self):
        assert cf.needs_fragment(["docs/a.rst", "tests/b.py", "src/c.py"]) is True

    def test_an_empty_diff_needs_nothing(self):
        assert cf.needs_fragment([]) is False


class TestGateFailures:
    """The three PR-shape rules, exercised without a git repository.

    Every case is a diff shape the gate will actually meet: a normal PR, a
    correction PR, the release PR's own rotation. The messages are pinned by
    substring rather than in full — they are prose meant for a human who is
    already stuck, and the parts that matter are the ones that tell them what
    to do next.
    """

    EXISTING = "\n### Fixed\n\n- an entry that was already here\n\n"
    EMPTY = "\n"

    @staticmethod
    def changelog(block: str) -> str:
        """A changelog whose ``[Unreleased]`` block is exactly *block*."""
        return (
            "# Changelog\n"
            "\n"
            "## [Unreleased]\n"
            f"{block}"
            "## [2026.8.0] - 2026-08-20\n"
            "\n"
            "### Fixed\n"
            "\n"
            "- something in the last release\n"
        )

    @staticmethod
    def rotated() -> str:
        """The head of a release PR: `[Unreleased]` emptied INTO a new section.

        This is the shape the deletion exemption keys on. An empty block by
        itself is the steady state of every PR once fragments carry the
        changelog, so what marks the release is the dated heading it cuts.
        """
        return (
            "# Changelog\n"
            "\n"
            "## [Unreleased]\n"
            "\n"
            "## [2026.9.0] - 2026-09-01\n"
            "\n"
            "### Fixed\n"
            "\n"
            "- an entry that was already here\n"
            "\n"
            "## [2026.8.0] - 2026-08-20\n"
            "\n"
            "### Fixed\n"
            "\n"
            "- something in the last release\n"
        )

    # ---- rule 3: a gated change needs a fragment -------------------------

    def test_a_gated_change_with_a_fragment_passes(self):
        text = self.changelog(self.EXISTING)
        failures, ok_lines = cf.gate_failures(
            ["src/osprey/cli/build.py", "changelog.d/745.fixed.md"],
            ["changelog.d/745.fixed.md"],
            [],
            text,
            text,
        )
        assert failures == []
        assert ok_lines == [
            "✓ changelog gate: changelog.d/745.fixed.md added for 1 changed file(s) "
            "under src/, packages/",
            "✓ [Unreleased]: untouched",
        ]

    def test_a_gated_change_without_a_fragment_fails(self):
        text = self.changelog(self.EXISTING)
        failures, ok_lines = cf.gate_failures(["src/osprey/cli/build.py"], [], [], text, text)
        assert len(failures) == 1
        message = failures[0]
        assert "src/osprey/cli/build.py" in message
        assert "changelog.d/<name>.<type>.md" in message
        for type_ in cf.TYPES:
            assert type_ in message
        assert "the gate reads committed history — an unstaged fragment does not count" in message
        assert ok_lines == ["✓ [Unreleased]: untouched"]

    def test_the_failure_lists_five_paths_and_counts_the_rest(self):
        text = self.changelog(self.EXISTING)
        changed = [f"src/osprey/mod{index}.py" for index in range(8)]
        failures, _ = cf.gate_failures(changed, [], [], text, text)
        message = failures[0]
        assert "8 changed file(s)" in message
        for path in changed[:5]:
            assert path in message
        assert changed[5] not in message
        assert "+3 more" in message

    def test_a_pre_existing_fragment_does_not_satisfy_the_gate(self):
        """The fragment has to be *added* by this PR, not merely present."""
        text = self.changelog(self.EXISTING)
        failures, _ = cf.gate_failures(["src/osprey/a.py"], [], [], text, text)
        assert len(failures) == 1
        assert "no changelog fragment" in failures[0]

    def test_a_modified_fragment_does_not_satisfy_the_gate(self):
        text = self.changelog(self.EXISTING)
        failures, _ = cf.gate_failures(
            ["src/osprey/a.py", "changelog.d/745.fixed.md"], [], [], text, text
        )
        assert len(failures) == 1

    @pytest.mark.parametrize(
        "added",
        [
            ["changelog.d/745.Fixed.md"],  # the type is lower-case only
            ["changelog.d/README.md"],  # never a fragment
            ["changelog.d/sub/745.fixed.md"],  # the directory is flat
            ["changelog.doc/745.fixed.md"],  # a different directory
            ["745.fixed.md"],  # not in changelog.d/ at all
        ],
    )
    def test_an_added_file_that_is_not_a_fragment_does_not_satisfy_the_gate(self, added):
        text = self.changelog(self.EXISTING)
        failures, _ = cf.gate_failures(["src/osprey/a.py"], added, [], text, text)
        assert len(failures) == 1
        assert "no changelog fragment" in failures[0]

    def test_an_unknown_type_is_left_to_the_validator(self):
        """Rule 3 asks only that the *name* be a fragment name.

        `745.tweaked.md` matches the grammar, so the gate stops complaining
        about a missing fragment — and `validate_dir` fails the same run with
        the vocabulary, which is the message the contributor needs. Checking
        the type here as well would print two failures for one typo.
        """
        text = self.changelog(self.EXISTING)
        failures, ok_lines = cf.gate_failures(
            ["src/osprey/a.py"], ["changelog.d/745.tweaked.md"], [], text, text
        )
        assert failures == []
        assert ok_lines[0].startswith("✓ changelog gate: changelog.d/745.tweaked.md added")

    def test_an_internal_fragment_satisfies_the_gate(self):
        """A refactor passes honestly instead of inventing a user-facing sentence."""
        text = self.changelog(self.EXISTING)
        failures, ok_lines = cf.gate_failures(
            ["src/osprey/a.py"], ["changelog.d/745.internal.md"], [], text, text
        )
        assert failures == []
        assert ok_lines[0].startswith("✓ changelog gate: changelog.d/745.internal.md added")

    def test_the_ok_line_names_the_first_fragment_in_filename_order(self):
        text = self.changelog(self.EXISTING)
        _, ok_lines = cf.gate_failures(
            ["src/a.py", "packages/b.py"],
            ["changelog.d/746.added.md", "changelog.d/115.fixed.md"],
            [],
            text,
            text,
        )
        assert ok_lines[0] == (
            "✓ changelog gate: changelog.d/115.fixed.md added for 2 changed file(s) "
            "under src/, packages/"
        )

    def test_no_gated_change_requires_no_fragment(self):
        text = self.changelog(self.EXISTING)
        failures, ok_lines = cf.gate_failures(
            ["docs/source/index.rst", "tests/a.py"], [], [], text, text
        )
        assert failures == []
        assert ok_lines == [
            "✓ changelog gate: no src/ or packages/ changes — no fragment required",
            "✓ [Unreleased]: untouched",
        ]

    # ---- rule 4: nobody writes [Unreleased] by hand -----------------------

    def test_a_hand_written_bullet_fails(self):
        base = self.changelog(self.EXISTING)
        head = self.changelog(
            "\n### Fixed\n\n- a bullet added by hand\n- an entry that was already here\n\n"
        )
        failures, _ = cf.gate_failures(
            ["src/osprey/a.py", "CHANGELOG.md"],
            ["changelog.d/745.fixed.md"],
            [],
            head,
            base,
        )
        assert len(failures) == 1
        assert "adds to ## [Unreleased] by hand" in failures[0]
        assert "changelog.d/<name>.<type>.md" in failures[0]
        assert "CHANGELOG-only PR" in failures[0]

    def test_an_edit_outside_the_block_passes(self):
        """The preamble sentence this feature adds is outside the block."""
        base = self.changelog(self.EXISTING)
        head = base.replace("# Changelog\n", "# Changelog\n\nFragments live in changelog.d/.\n")
        failures, ok_lines = cf.gate_failures(
            ["src/osprey/a.py", "CHANGELOG.md"],
            ["changelog.d/745.fixed.md"],
            [],
            head,
            base,
        )
        assert failures == []
        assert ok_lines[1] == "✓ [Unreleased]: untouched"

    def test_a_head_without_the_heading_fails(self):
        base = self.changelog(self.EXISTING)
        head = "# Changelog\n\n## [2026.8.0] - 2026-08-20\n\n- something\n"
        failures, ok_lines = cf.gate_failures(["docs/a.rst"], [], [], head, base)
        assert len(failures) == 1
        assert "## [Unreleased]" in failures[0]
        assert not any("[Unreleased]:" in line for line in ok_lines)

    def test_an_emptied_block_is_the_release_rotation(self):
        base = self.changelog(self.EXISTING)
        head = self.rotated()
        failures, ok_lines = cf.gate_failures(["CHANGELOG.md"], [], [], head, base)
        assert failures == []
        assert ok_lines[1] == "✓ [Unreleased]: empty (rotation)"

    def test_emptying_the_block_without_cutting_a_release_is_not_a_rotation(self):
        """An empty block alone proves nothing — every PR has one between releases."""
        base = self.changelog(self.EXISTING)
        head = self.changelog(self.EMPTY)
        failures, _ = cf.gate_failures(
            ["CHANGELOG.md", "src/osprey/a.py"],
            ["changelog.d/745.fixed.md"],
            [],
            head,
            base,
        )
        assert len(failures) == 1
        assert "adds to ## [Unreleased] by hand" in failures[0]

    def test_a_changelog_only_correction_passes(self):
        base = self.changelog(self.EXISTING)
        head = self.changelog("\n### Fixed\n\n- an entry that was already here, reworded\n\n")
        failures, ok_lines = cf.gate_failures(["CHANGELOG.md"], [], [], head, base)
        assert failures == []
        assert ok_lines[1] == "✓ [Unreleased]: changelog-only correction"

    def test_a_changelog_only_removal_passes(self):
        base = self.changelog(self.EXISTING)
        head = self.changelog("\n### Fixed\n\n")
        failures, ok_lines = cf.gate_failures(["CHANGELOG.md"], [], [], head, base)
        assert failures == []
        assert ok_lines[1] == "✓ [Unreleased]: changelog-only correction"

    def test_a_changelog_only_addition_fails(self):
        base = self.changelog(self.EXISTING)
        head = self.changelog("\n### Fixed\n\n- an entry that was already here\n- a new one\n\n")
        failures, _ = cf.gate_failures(["CHANGELOG.md"], [], [], head, base)
        assert len(failures) == 1
        assert "adds to ## [Unreleased] by hand" in failures[0]

    def test_delete_one_add_one_in_a_changelog_only_pr_passes(self):
        """Named residual: a correction-shaped edit the bullet count cannot tell apart."""
        base = self.changelog(self.EXISTING)
        head = self.changelog("\n### Fixed\n\n- an entirely different entry\n\n")
        failures, ok_lines = cf.gate_failures(["CHANGELOG.md"], [], [], head, base)
        assert failures == []
        assert ok_lines[1] == "✓ [Unreleased]: changelog-only correction"

    def test_a_correction_alongside_another_file_fails(self):
        """The exemption is for a CHANGELOG-only PR, and nothing else."""
        base = self.changelog(self.EXISTING)
        head = self.changelog("\n### Fixed\n\n- an entry that was already here, reworded\n\n")
        failures, _ = cf.gate_failures(
            ["CHANGELOG.md", "src/osprey/a.py"],
            ["changelog.d/745.fixed.md"],
            [],
            head,
            base,
        )
        assert len(failures) == 1
        assert "adds to ## [Unreleased] by hand" in failures[0]

    def test_an_absent_base_changelog_compares_against_an_empty_block(self):
        head = self.changelog(self.EXISTING)
        failures, _ = cf.gate_failures(
            ["src/osprey/a.py"], ["changelog.d/745.fixed.md"], [], head, ""
        )
        assert len(failures) == 1
        assert "adds to ## [Unreleased] by hand" in failures[0]

    def test_an_absent_base_changelog_still_recognizes_a_rotation(self):
        head = self.changelog(self.EMPTY)
        failures, ok_lines = cf.gate_failures(["CHANGELOG.md"], [], [], head, "")
        assert failures == []
        assert ok_lines[1] == "✓ [Unreleased]: empty (rotation)"

    # ---- rule 5: only the release fold deletes fragments -------------------

    def test_deleting_a_fragment_fails(self):
        text = self.changelog(self.EXISTING)
        failures, _ = cf.gate_failures(
            ["changelog.d/745.fixed.md"], [], ["changelog.d/745.fixed.md"], text, text
        )
        assert len(failures) == 1
        assert "only the release PR's apply removes fragments" in failures[0]
        assert "changelog.d/745.fixed.md" in failures[0]

    def test_deleting_the_readme_is_not_a_fragment_deletion(self):
        text = self.changelog(self.EXISTING)
        failures, _ = cf.gate_failures(
            ["changelog.d/README.md"], [], ["changelog.d/README.md"], text, text
        )
        assert failures == []

    def test_the_release_pr_may_delete_fragments(self):
        base = self.changelog(self.EXISTING)
        head = self.rotated()
        changed = [
            "CHANGELOG.md",
            "RELEASE_NOTES.md",
            "README.md",
            "changelog.d/745.fixed.md",
            "changelog.d/gate.security.md",
        ]
        deleted = ["changelog.d/745.fixed.md", "changelog.d/gate.security.md"]
        failures, ok_lines = cf.gate_failures(changed, [], deleted, head, base)
        assert failures == []
        assert ok_lines == [
            "✓ changelog gate: no src/ or packages/ changes — no fragment required",
            "✓ [Unreleased]: empty (rotation)",
        ]

    def test_an_empty_unreleased_does_not_exempt_a_deletion(self):
        """The steady state, and the one the old head-only predicate let through.

        Once fragments carry the changelog, `[Unreleased]` is empty on every
        PR. A feature PR that also deletes somebody else's fragment must not
        be read as the release.
        """
        text = self.changelog(self.EMPTY)
        failures, ok_lines = cf.gate_failures(
            ["src/osprey/a.py", "changelog.d/745.fixed.md", "changelog.d/other.fixed.md"],
            ["changelog.d/745.fixed.md"],
            ["changelog.d/other.fixed.md"],
            text,
            text,
        )
        assert len(failures) == 1
        assert "only the release PR's apply removes fragments" in failures[0]
        assert "changelog.d/other.fixed.md" in failures[0]
        assert ok_lines[1] == "✓ [Unreleased]: untouched"

    def test_the_release_pr_passes_when_the_base_block_is_already_empty(self):
        """The release cut from the steady state: both blocks empty, one new section."""
        base = self.changelog(self.EMPTY)
        head = self.rotated()
        deleted = ["changelog.d/745.fixed.md", "changelog.d/gate.security.md"]
        failures, ok_lines = cf.gate_failures(["CHANGELOG.md", *deleted], [], deleted, head, base)
        assert failures == []
        # The blocks are byte-identical, so rule 4 says "untouched" — but the
        # new release heading is what lets rule 5 through.
        assert ok_lines[1] == "✓ [Unreleased]: untouched"

    def test_all_three_rules_can_fail_at_once(self):
        base = self.changelog(self.EXISTING)
        head = self.changelog(
            "\n### Fixed\n\n- a bullet added by hand\n- an entry that was already here\n\n"
        )
        failures, _ = cf.gate_failures(
            ["src/osprey/a.py", "CHANGELOG.md", "changelog.d/745.fixed.md"],
            [],
            ["changelog.d/745.fixed.md"],
            head,
            base,
        )
        assert len(failures) == 3
        assert "no changelog fragment" in failures[0]
        assert "adds to ## [Unreleased] by hand" in failures[1]
        assert "only the release PR's apply removes fragments" in failures[2]


# ---------------------------------------------------------------------------
# The CLI. Git is answered from a table, so nothing below builds a repository.
# ---------------------------------------------------------------------------

BASE_SHA = "a" * 40
MERGE_BASE = "b" * 40

CHANGELOG = (
    "# Changelog\n"
    "\n"
    "## [Unreleased]\n"
    "\n"
    "### Fixed\n"
    "\n"
    "- an entry that was already here\n"
    "\n"
    "## [2026.8.0] - 2026-08-20\n"
    "\n"
    "### Fixed\n"
    "\n"
    "- something in the last release\n"
)


def completed(
    stdout: str = "", returncode: int = 0, stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    """One finished git call."""
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


class FakeGit:
    """git as a lookup table, recording what it was asked.

    Anything the table does not answer comes back the way git does for a
    revision it cannot resolve, so a test that forgets an entry fails on the
    exit code rather than on a KeyError.
    """

    def __init__(self, table: dict[tuple[str, ...], subprocess.CompletedProcess[str]]):
        self.table = table
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str]) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(argv))
        return self.table.get(tuple(argv), completed(returncode=128, stderr="fatal: bad revision"))


def git_table(
    *,
    diff: str = "",
    base_text: str = CHANGELOG,
    ref: str = "origin/main",
) -> dict[tuple[str, ...], subprocess.CompletedProcess[str]]:
    """A table for one green ``check``: base resolves, merge base, diff, show."""
    return {
        ("git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"): completed(
            f"{BASE_SHA}\n"
        ),
        ("git", "merge-base", BASE_SHA, "HEAD"): completed(f"{MERGE_BASE}\n"),
        ("git", "diff", "--name-status", "-z", "--no-renames", MERGE_BASE, "HEAD"): completed(diff),
        ("git", "show", f"{MERGE_BASE}:CHANGELOG.md"): completed(base_text),
    }


def diff_z(*pairs: tuple[str, str]) -> str:
    """``git diff --name-status -z`` output for *pairs* of ``(status, path)``."""
    return "".join(f"{status}\0{path}\0" for status, path in pairs)


class TestResolveBase:
    """Which spelling of the base ref is tried, and in which order.

    In a fresh CI checkout the only copy of `main` is `origin/main`, and on a
    developer's machine `HEAD~1` must not be looked for under `origin/`. One
    rule covers both, and its edges are what these tests pin.
    """

    @staticmethod
    def rev_parse(ref: str) -> tuple[str, ...]:
        return ("git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")

    def test_a_plain_name_is_looked_for_under_origin_first(self):
        git = FakeGit({self.rev_parse("origin/main"): completed(f"{BASE_SHA}\n")})
        assert cf.resolve_base("main", git) == BASE_SHA
        assert git.calls == [list(self.rev_parse("origin/main"))]

    def test_a_plain_name_falls_back_to_the_local_branch(self):
        git = FakeGit({self.rev_parse("main"): completed(f"{BASE_SHA}\n")})
        assert cf.resolve_base("main", git) == BASE_SHA
        assert [call[-1] for call in git.calls] == ["origin/main^{commit}", "main^{commit}"]

    def test_an_already_qualified_ref_is_tried_as_given(self):
        git = FakeGit({self.rev_parse("origin/main"): completed(f"{BASE_SHA}\n")})
        assert cf.resolve_base("origin/main", git) == BASE_SHA
        assert git.calls == [list(self.rev_parse("origin/main"))]

    def test_a_revision_expression_is_tried_as_given(self):
        """`HEAD~1` is not a branch name; `origin/HEAD~1` must not come first."""
        git = FakeGit({self.rev_parse("HEAD~1"): completed(f"{BASE_SHA}\n")})
        assert cf.resolve_base("HEAD~1", git) == BASE_SHA
        assert git.calls == [list(self.rev_parse("HEAD~1"))]

    def test_head_itself_is_not_treated_as_a_branch_name(self):
        git = FakeGit({self.rev_parse("HEAD"): completed(f"{BASE_SHA}\n")})
        assert cf.resolve_base("HEAD", git) == BASE_SHA
        assert git.calls == [list(self.rev_parse("HEAD"))]

    def test_an_unresolvable_ref_is_none_after_both_spellings(self):
        git = FakeGit({})
        assert cf.resolve_base("nope", git) is None
        assert [call[-1] for call in git.calls] == ["origin/nope^{commit}", "nope^{commit}"]

    def test_a_success_with_no_output_is_not_a_resolution(self):
        """`--quiet` says nothing when it finds nothing; an empty sha is not one."""
        git = FakeGit({self.rev_parse("origin/main"): completed("\n")})
        assert cf.resolve_base("main", git) is None


class TestParseNameStatus:
    def test_pairs_are_read_two_fields_at_a_time(self):
        assert cf.parse_name_status(diff_z(("M", "a.py"), ("A", "b.py"))) == [
            ("M", "a.py"),
            ("A", "b.py"),
        ]

    def test_a_path_with_a_space_survives(self):
        """The whole reason the diff is asked for with `-z`."""
        assert cf.parse_name_status(diff_z(("M", "src/osprey/a b.py"))) == [
            ("M", "src/osprey/a b.py")
        ]

    def test_an_empty_diff_is_no_pairs(self):
        assert cf.parse_name_status("") == []


class TestMain:
    """The two subcommands end to end, with git faked and files in tmp_path."""

    @staticmethod
    def fixture(tmp_path: Path, head_text: str = CHANGELOG) -> tuple[Path, Path]:
        """A fragment directory and a changelog file to point the CLI at."""
        directory = tmp_path / "changelog.d"
        directory.mkdir()
        (directory / "README.md").write_text("how to write one\n", encoding="utf-8")
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text(head_text, encoding="utf-8")
        return directory, changelog

    @staticmethod
    def check_argv(directory: Path, changelog: Path, *extra: str) -> list[str]:
        return ["check", "--dir", str(directory), "--changelog", str(changelog), *extra]

    # ---- check: the environment exit code --------------------------------

    def test_an_unresolvable_base_is_an_environment_error(self, tmp_path, capsys):
        directory, changelog = self.fixture(tmp_path)
        code = cf.main(self.check_argv(directory, changelog), FakeGit({}))
        assert code == 2
        err = capsys.readouterr().err
        assert "base ref origin/main not found" in err
        assert "git fetch origin <branch>" in err
        assert "fetch-depth: 0" in err

    def test_an_empty_merge_base_is_an_environment_error(self, tmp_path, capsys):
        """A shallow clone answers `merge-base` with success and nothing else."""
        directory, changelog = self.fixture(tmp_path)
        table = git_table()
        table[("git", "merge-base", BASE_SHA, "HEAD")] = completed("\n")
        code = cf.main(self.check_argv(directory, changelog), FakeGit(table))
        assert code == 2
        assert (
            "no common ancestor — shallow clone? git fetch --unshallow" in capsys.readouterr().err
        )

    def test_a_failing_merge_base_is_an_environment_error(self, tmp_path, capsys):
        directory, changelog = self.fixture(tmp_path)
        table = git_table()
        table[("git", "merge-base", BASE_SHA, "HEAD")] = completed(
            returncode=1, stderr="fatal: refusing"
        )
        code = cf.main(self.check_argv(directory, changelog), FakeGit(table))
        assert code == 2
        assert "no common ancestor" in capsys.readouterr().err

    def test_a_failing_diff_names_the_command(self, tmp_path, capsys):
        directory, changelog = self.fixture(tmp_path)
        table = git_table()
        table[("git", "diff", "--name-status", "-z", "--no-renames", MERGE_BASE, "HEAD")] = (
            completed(returncode=128, stderr="fatal: bad object")
        )
        code = cf.main(self.check_argv(directory, changelog), FakeGit(table))
        assert code == 2
        err = capsys.readouterr().err
        assert "git diff --name-status -z --no-renames" in err
        assert "fatal: bad object" in err

    def test_a_failing_show_names_the_command(self, tmp_path, capsys):
        directory, changelog = self.fixture(tmp_path)
        table = git_table()
        table[("git", "show", f"{MERGE_BASE}:CHANGELOG.md")] = completed(
            returncode=128, stderr="fatal: path does not exist"
        )
        code = cf.main(self.check_argv(directory, changelog), FakeGit(table))
        assert code == 2
        err = capsys.readouterr().err
        assert f"git show {MERGE_BASE}:CHANGELOG.md" in err
        assert "fatal: path does not exist" in err

    # ---- check: the report -----------------------------------------------

    def test_a_green_run_prints_three_lines_and_exits_zero(self, tmp_path, capsys):
        directory, changelog = self.fixture(tmp_path)
        write(directory, "745.fixed.md", "the gate no longer eats the last line.\n")
        git = FakeGit(
            git_table(
                diff=diff_z(("M", "src/osprey/cli/build.py"), ("A", "changelog.d/745.fixed.md"))
            )
        )
        code = cf.main(self.check_argv(directory, changelog), git)
        assert code == 0
        assert capsys.readouterr().out.splitlines() == [
            "✓ changelog.d: 1 fragment(s) valid",
            "✓ changelog gate: changelog.d/745.fixed.md added for 1 changed file(s) "
            "under src/, packages/",
            "✓ [Unreleased]: untouched",
        ]

    def test_a_documentation_only_pull_request_needs_no_fragment(self, tmp_path, capsys):
        directory, changelog = self.fixture(tmp_path)
        git = FakeGit(git_table(diff=diff_z(("M", "docs/source/index.rst"))))
        code = cf.main(self.check_argv(directory, changelog), git)
        assert code == 0
        assert capsys.readouterr().out.splitlines() == [
            "✓ changelog.d: 0 fragment(s) valid",
            "✓ changelog gate: no src/ or packages/ changes — no fragment required",
            "✓ [Unreleased]: untouched",
        ]

    def test_a_red_run_reports_every_offender_at_once(self, tmp_path, capsys):
        """Four problems, one run: a contributor should not fix them one per push."""
        head_text = CHANGELOG.replace(
            "- an entry that was already here\n",
            "- an entry that was already here\n- one written by hand\n",
        )
        directory, changelog = self.fixture(tmp_path, head_text)
        write(directory, "745.tweaked.md", "an unknown type.\n")
        git = FakeGit(
            git_table(
                diff=diff_z(("M", "src/osprey/a b.py"), ("D", "changelog.d/old.fixed.md")),
            )
        )
        code = cf.main(self.check_argv(directory, changelog), git)
        assert code == 1
        out = capsys.readouterr().out
        assert '✗ 745.tweaked.md: unknown type "tweaked"' in out
        assert "✗ no changelog fragment for 1 changed file(s)" in out
        assert "✗ this PR adds to ## [Unreleased] by hand" in out
        assert "✗ 1 fragment(s) deleted" in out
        # The gated path came through `-z`, and detail lines are indented by two.
        assert "  src/osprey/a b.py" in out.splitlines()
        assert "  changelog.d/old.fixed.md" in out.splitlines()

    def test_a_malformed_fragment_fails_a_run_whose_gate_is_green(self, tmp_path, capsys):
        directory, changelog = self.fixture(tmp_path)
        write(directory, "745.fixed.md", "- already a bullet\n")
        git = FakeGit(git_table(diff=diff_z(("M", "docs/source/index.rst"))))
        code = cf.main(self.check_argv(directory, changelog), git)
        assert code == 1
        out = capsys.readouterr().out
        assert "✗ 745.fixed.md: a fragment must open with prose" in out
        assert "fragment(s) valid" not in out
        assert "✓ [Unreleased]: untouched" in out

    def test_the_base_ref_can_be_a_revision_expression(self, tmp_path):
        directory, changelog = self.fixture(tmp_path)
        git = FakeGit(git_table(ref="HEAD~1"))
        code = cf.main(self.check_argv(directory, changelog, "--base", "HEAD~1"), git)
        assert code == 0
        assert git.calls[0][-1] == "HEAD~1^{commit}"

    def test_a_missing_fragment_directory_is_not_an_error(self, tmp_path, capsys):
        """A branch cut before this landed still has to be able to pass."""
        _, changelog = self.fixture(tmp_path)
        git = FakeGit(git_table(diff=diff_z(("M", "README.md"))))
        code = cf.main(self.check_argv(tmp_path / "absent", changelog), git)
        assert code == 0
        assert "✓ changelog.d: 0 fragment(s) valid" in capsys.readouterr().out

    # ---- apply ------------------------------------------------------------

    def test_apply_folds_deletes_and_says_what_to_stage(self, tmp_path, capsys):
        directory, changelog = self.fixture(tmp_path)
        write(directory, "745.fixed.md", "the gate no longer eats the last line.\n")
        write(directory, "banner.added.md", "a banner naming the control target.\n")
        write(directory, "cleanup.internal.md", "moved a helper.\n")

        assert cf.main(["apply", "--changelog", str(changelog), "--dir", str(directory)]) == 0

        text = changelog.read_text(encoding="utf-8")
        assert "- the gate no longer eats the last line. (#745)" in text
        assert "### Added\n\n- a banner naming the control target." in text
        assert "moved a helper" not in text
        assert sorted(path.name for path in directory.iterdir()) == ["README.md"]
        assert capsys.readouterr().out.splitlines() == [
            "changelog.d/banner.added.md -> ### Added",
            "changelog.d/745.fixed.md -> ### Fixed (#745)",
            "changelog.d/cleanup.internal.md -> internal, not rendered",
            "stage with: git add -A changelog.d/ CHANGELOG.md",
        ]

    def test_a_second_apply_has_nothing_to_do(self, tmp_path, capsys):
        directory, changelog = self.fixture(tmp_path)
        write(directory, "745.fixed.md", "the gate no longer eats the last line.\n")
        argv = ["apply", "--changelog", str(changelog), "--dir", str(directory)]
        assert cf.main(argv) == 0
        before = changelog.read_text(encoding="utf-8")
        capsys.readouterr()

        assert cf.main(argv) == 0
        assert capsys.readouterr().out == "changelog.d: no fragments; nothing to do\n"
        assert changelog.read_text(encoding="utf-8") == before

    def test_apply_refuses_a_directory_that_does_not_validate(self, tmp_path, capsys):
        directory, changelog = self.fixture(tmp_path)
        bad = write(directory, "745.tweaked.md", "an unknown type.\n")
        code = cf.main(["apply", "--changelog", str(changelog), "--dir", str(directory)])
        assert code == 1
        assert '✗ 745.tweaked.md: unknown type "tweaked"' in capsys.readouterr().out
        assert bad.exists()
        assert changelog.read_text(encoding="utf-8") == CHANGELOG

    def test_apply_refuses_a_changelog_with_no_unreleased_heading(self, tmp_path, capsys):
        """After the rotation there is nowhere to fold into — say so, keep the files."""
        rotated = CHANGELOG.replace("## [Unreleased]\n", "## [2026.9.0] - 2026-09-01\n")
        directory, changelog = self.fixture(tmp_path, rotated)
        fragment = write(directory, "745.fixed.md", "the gate no longer eats the last line.\n")
        code = cf.main(["apply", "--changelog", str(changelog), "--dir", str(directory)])
        assert code == 1
        assert "✗ no ## [Unreleased] heading" in capsys.readouterr().out
        assert fragment.exists()
        assert changelog.read_text(encoding="utf-8") == rotated

    # ---- the runner, and failures that belong to the environment ----------

    def test_the_default_runner_pins_utf8_and_the_repository_root(self, monkeypatch):
        """git output carries em dashes and ✓; the ambient locale must not decide."""
        captured: dict[str, object] = {}

        def fake_run(argv, **kwargs):
            captured.update(kwargs)
            return subprocess.CompletedProcess(argv, 0, "", "")

        monkeypatch.setattr(cf.subprocess, "run", fake_run)
        cf._run(["git", "status"])
        assert captured["encoding"] == "utf-8"
        assert captured["text"] is True
        assert captured["capture_output"] is True
        assert captured["cwd"] == cf.REPO_ROOT

    def test_an_environment_error_does_not_hide_the_fragment_errors(self, tmp_path, capsys):
        """Exit 2 answers a different question; say the exit-1 problems are still there."""
        directory, changelog = self.fixture(tmp_path)
        write(directory, "745.tweaked.md", "an unknown type.\n")
        code = cf.main(self.check_argv(directory, changelog), FakeGit({}))
        assert code == 2
        captured = capsys.readouterr()
        assert '✗ 745.tweaked.md: unknown type "tweaked"' in captured.out
        assert "1 fragment error(s) reported above are separate" in captured.err

    def test_check_reports_a_changelog_it_cannot_decode(self, tmp_path, capsys):
        """A changelog that is not UTF-8 is the environment's problem, not a
        traceback — the same exit the unreadable-file case gets."""
        directory, changelog = self.fixture(tmp_path)
        changelog.write_bytes(b"# Changelog\n\n## [Unreleased]\n\ncaf\xe9\n")
        code = cf.main(self.check_argv(directory, changelog), FakeGit(git_table()))
        assert code == 2
        assert f"cannot read {changelog}" in capsys.readouterr().err

    def test_apply_reports_a_changelog_it_cannot_decode(self, tmp_path, capsys):
        directory, changelog = self.fixture(tmp_path)
        fragment = write(directory, "745.fixed.md", "A bug.\n")
        changelog.write_bytes(b"# Changelog\n\n## [Unreleased]\n\ncaf\xe9\n")
        code = cf.main(["apply", "--changelog", str(changelog), "--dir", str(directory)])
        assert code == 2
        assert f"cannot read {changelog}" in capsys.readouterr().err
        assert fragment.exists()

    def test_a_changelog_that_cannot_be_written_is_an_environment_error(
        self, tmp_path, capsys, monkeypatch
    ):
        directory, changelog = self.fixture(tmp_path)
        fragment = write(directory, "745.fixed.md", "A bug.\n")
        monkeypatch.setattr(Path, "write_text", refuse_io)
        code = cf.main(["apply", "--changelog", str(changelog), "--dir", str(directory)])
        assert code == 2
        assert f"cannot write {changelog}" in capsys.readouterr().err
        assert fragment.exists()

    def test_a_fragment_that_cannot_be_deleted_is_named(self, tmp_path, capsys, monkeypatch):
        """The changelog is already written, so a survivor would be folded in twice."""
        directory, changelog = self.fixture(tmp_path)
        fragment = write(directory, "745.fixed.md", "A bug.\n")
        monkeypatch.setattr(Path, "unlink", refuse_io)
        code = cf.main(["apply", "--changelog", str(changelog), "--dir", str(directory)])
        assert code == 2
        err = capsys.readouterr().err
        assert "1 fragment(s) could not be deleted" in err
        assert str(fragment) in err
        assert "- A bug. (#745)" in changelog.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The live tree. Everything below reads the real repository and writes nothing.
# ---------------------------------------------------------------------------

#: Surfaces that spell out the whole type vocabulary. A type added to `TYPES`
#: without being documented in both of these is a type nobody knows exists.
VOCABULARY_SURFACES = (
    "changelog.d/README.md",
    "src/osprey/templates/skills/osprey-contribute/SKILL.md",
)

#: Surfaces that have to send a contributor to the directory. Each one is a
#: place where someone reads "how do I record this change?" and must not be
#: told to edit `## [Unreleased]` by hand.
POINTER_SURFACES = (
    "docs/source/contributing/workflow.rst",
    "CONTRIBUTING.md",
    "scripts/README.md",
    "src/osprey/templates/skills/osprey-design-philosophy/SKILL.md",
    "src/osprey/templates/skills/osprey-release/SKILL.md",
)


def _surface_has(text: str, tokens: list[str]) -> bool:
    """True when *text* contains every token in *tokens*."""
    return all(token in text for token in tokens)


class TestLiveTree:
    """The documentation and the vocabulary move in lockstep, or this fails.

    The gate's messages name all seven types; so must the two surfaces a
    contributor actually reads. These assertions derive their expectations
    from `cf.TYPES` rather than repeating it, so adding a type turns into a
    failing test in the same commit instead of a stale README a year later.
    Each one has a negative control, because an assertion over a 200-line
    document passes by accident easily.
    """

    @staticmethod
    def read(relative: str) -> str:
        """Text of a repository file, failing by name when it has moved."""
        path = cf.REPO_ROOT / relative
        assert path.is_file(), f"{relative} is missing"
        return path.read_text(encoding="utf-8")

    @staticmethod
    def vocabulary() -> list[str]:
        """The filename token every type is documented by: `.<type>.md`."""
        return [f".{type_}.md" for type_ in cf.TYPES]

    def test_the_repositorys_own_fragments_validate(self):
        """The directory this branch ships passes its own validator."""
        _, errors = cf.validate_dir(cf.REPO_ROOT / cf.FRAGMENT_DIR)
        assert errors == []
        assert (cf.REPO_ROOT / cf.FRAGMENT_DIR / cf.KEEP).is_file()

    @pytest.mark.parametrize("relative", VOCABULARY_SURFACES)
    def test_a_vocabulary_surface_spells_out_every_type(self, relative):
        assert _surface_has(self.read(relative), self.vocabulary())

    @pytest.mark.parametrize("relative", VOCABULARY_SURFACES)
    def test_dropping_one_type_from_a_vocabulary_surface_would_fail(self, relative):
        text = self.read(relative)
        for token in self.vocabulary():
            assert _surface_has(text.replace(token, ""), self.vocabulary()) is False

    @pytest.mark.parametrize("relative", POINTER_SURFACES)
    def test_a_pointer_surface_sends_the_reader_to_the_directory(self, relative):
        assert _surface_has(self.read(relative), [f"{cf.FRAGMENT_DIR}/"])

    @pytest.mark.parametrize("relative", POINTER_SURFACES)
    def test_dropping_the_directory_from_a_pointer_surface_would_fail(self, relative):
        text = self.read(relative).replace(f"{cf.FRAGMENT_DIR}/", "")
        assert _surface_has(text, [f"{cf.FRAGMENT_DIR}/"]) is False

    def test_the_release_skill_folds_and_counts_what_is_on_disk(self):
        """Step 3 has to run `apply`, and the release has to see the count."""
        tokens = ["changelog_fragments.py apply", "fragment(s) on disk"]
        assert _surface_has(
            self.read("src/osprey/templates/skills/osprey-release/SKILL.md"), tokens
        )

    def test_dropping_either_release_token_would_fail(self):
        text = self.read("src/osprey/templates/skills/osprey-release/SKILL.md")
        tokens = ["changelog_fragments.py apply", "fragment(s) on disk"]
        for token in tokens:
            assert _surface_has(text.replace(token, ""), tokens) is False
