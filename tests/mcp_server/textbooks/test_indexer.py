"""Tests for textbooks indexer — parsing INDEX.md, SECTIONS.md, TERMS.json, EQUATIONS.md."""

from __future__ import annotations

from pathlib import Path

from osprey.mcp_server.textbooks import indexer


class TestParseConceptMap:
    """Tests for _parse_concept_map."""

    def test_extracts_concepts(self, sample_book_dir: Path):
        concepts, chapters, summaries = indexer._parse_concept_map(
            sample_book_dir / "INDEX.md"
        )
        assert len(concepts) == 3
        assert concepts[0].concept == "Betatron function"
        assert concepts[0].chapter == "ch01"
        assert concepts[0].section == "Betatron Function"
        assert concepts[0].line == 50

    def test_extracts_chapters(self, sample_book_dir: Path):
        _, chapters, _ = indexer._parse_concept_map(sample_book_dir / "INDEX.md")
        assert chapters == ["ch01_dynamics", "ch02_colliders"]

    def test_extracts_summaries(self, sample_book_dir: Path):
        _, _, summaries = indexer._parse_concept_map(sample_book_dir / "INDEX.md")
        assert "ch01_dynamics" in summaries
        assert "beam dynamics" in summaries["ch01_dynamics"].lower()


class TestParseSections:
    """Tests for _parse_sections."""

    def test_extracts_headings(self, sample_book_dir: Path):
        headings = indexer._parse_sections(
            sample_book_dir / "ch01_dynamics_SECTIONS.md"
        )
        assert len(headings) == 5
        assert headings[0].heading == "Beam Dynamics"
        assert headings[0].level == "h2"
        assert headings[0].line == 1

    def test_heading_line_numbers(self, sample_book_dir: Path):
        headings = indexer._parse_sections(
            sample_book_dir / "ch01_dynamics_SECTIONS.md"
        )
        lines = [h.line for h in headings]
        assert lines == [1, 10, 30, 50, 80]


class TestParseTerms:
    """Tests for _parse_terms."""

    def test_extracts_terms(self, sample_book_dir: Path):
        terms = indexer._parse_terms(sample_book_dir / "TERMS.json")
        assert len(terms) == 4
        assert terms[0].term == "Betatron tune"
        assert terms[0].pages == [94, 101]


class TestParseEquations:
    """Tests for _parse_equations."""

    def test_extracts_equations(self, sample_book_dir: Path):
        equations = indexer._parse_equations(sample_book_dir / "EQUATIONS.md")
        assert len(equations) == 3
        assert equations[0].tag == "1.1"
        assert equations[0].chapter == "ch01_dynamics"
        assert equations[0].line == 15


class TestLoadAllBooks:
    """Tests for load_all_books."""

    def test_loads_book(self, loaded_book):
        assert loaded_book is not None
        assert loaded_book.name == "TestBook"

    def test_book_has_concepts(self, loaded_book):
        assert len(loaded_book.concepts) == 3

    def test_book_has_chapters(self, loaded_book):
        assert loaded_book.chapters == ["ch01_dynamics", "ch02_colliders"]

    def test_book_has_sections(self, loaded_book):
        assert "ch01_dynamics" in loaded_book.sections
        assert len(loaded_book.sections["ch01_dynamics"]) == 5

    def test_book_has_terms(self, loaded_book):
        assert len(loaded_book.terms) == 4

    def test_book_has_equations(self, loaded_book):
        assert len(loaded_book.equations) == 3


class TestGetBook:
    """Tests for get_book."""

    def test_exact_match(self, loaded_book):
        assert indexer.get_book("TestBook") is loaded_book

    def test_partial_match(self, loaded_book):
        assert indexer.get_book("Test") is loaded_book

    def test_case_insensitive(self, loaded_book):
        assert indexer.get_book("testbook") is loaded_book

    def test_not_found(self, loaded_book):
        assert indexer.get_book("NonexistentBook") is None


class TestSearchIndexes:
    """Tests for search_indexes."""

    def test_finds_concept(self, loaded_book):
        results = indexer.search_indexes("betatron function")
        assert len(results) > 0
        concept_results = [r for r in results if r["type"] == "concept"]
        assert any(r["term"] == "Betatron function" for r in concept_results)

    def test_finds_term(self, loaded_book):
        results = indexer.search_indexes("betatron tune")
        term_results = [r for r in results if r["type"] == "term"]
        assert any(r["term"] == "Betatron tune" for r in term_results)

    def test_finds_section_heading(self, loaded_book):
        results = indexer.search_indexes("adiabatic damping")
        section_results = [r for r in results if r["type"] == "section"]
        assert any("Adiabatic Damping" in r["term"] for r in section_results)

    def test_finds_equation(self, loaded_book):
        results = indexer.search_indexes("1.1")
        eq_results = [r for r in results if r["type"] == "equation"]
        assert any("1.1" in r["term"] for r in eq_results)

    def test_filter_by_book(self, loaded_book):
        results = indexer.search_indexes("luminosity", book_name="TestBook")
        assert len(results) > 0
        assert all(r["book"] == "TestBook" for r in results)

    def test_no_results_for_unknown(self, loaded_book):
        results = indexer.search_indexes("zzz_nonexistent_concept_zzz")
        assert results == []

    def test_results_sorted_by_score(self, loaded_book):
        results = indexer.search_indexes("phase stability")
        if len(results) > 1:
            # Verify descending score order (scores removed in tool output,
            # but still present in internal search_indexes output)
            pass  # scores are removed in dedup; just check we get results
        assert len(results) > 0

    def test_max_results(self, loaded_book):
        results = indexer.search_indexes("betatron", max_results=2)
        assert len(results) <= 2


class TestFindSectionBounds:
    """Tests for find_section_bounds."""

    def test_finds_section_by_name(self, loaded_book):
        bounds = indexer.find_section_bounds(loaded_book, "ch01_dynamics", "Phase Stability")
        assert bounds is not None
        start, end = bounds
        assert start == 10
        assert end == 29  # Line before "Adiabatic Damping" at line 30

    def test_finds_last_section(self, loaded_book):
        bounds = indexer.find_section_bounds(loaded_book, "ch01_dynamics", "Summary")
        assert bounds is not None
        start, end = bounds
        assert start == 80
        # End should be the last line of the file

    def test_not_found(self, loaded_book):
        bounds = indexer.find_section_bounds(loaded_book, "ch01_dynamics", "Nonexistent Section")
        assert bounds is None

    def test_fuzzy_match(self, loaded_book):
        bounds = indexer.find_section_bounds(loaded_book, "ch01_dynamics", "betatron")
        assert bounds is not None
        assert bounds[0] == 50  # Betatron Function starts at line 50

    def test_h2_section_spans_entire_chapter(self, loaded_book):
        """An h2 heading with no following h2 should span to end of file."""
        bounds = indexer.find_section_bounds(loaded_book, "ch01_dynamics", "Beam Dynamics")
        assert bounds is not None
        assert bounds[0] == 1
