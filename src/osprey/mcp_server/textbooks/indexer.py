"""Textbook index loader and query engine.

Parses and caches structured indexes at startup:
- Concept map from INDEX.md
- Section headings from *_SECTIONS.md
- Term index from TERMS.json
- Equation index from EQUATIONS.md

Auto-discovers books in the ``processed_textbooks/`` directory.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

logger = logging.getLogger("osprey.mcp_server.textbooks.indexer")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ConceptEntry:
    """A single concept from the INDEX.md concept map."""

    concept: str
    chapter: str
    section: str
    line: int


@dataclass
class TermEntry:
    """A term from the alphabetical index (TERMS.json)."""

    term: str
    pages: list[int] = field(default_factory=list)


@dataclass
class SectionHeading:
    """A heading from a *_SECTIONS.md file."""

    line: int
    heading: str
    level: str  # h2, h3, h4, h5


@dataclass
class EquationEntry:
    """An equation from EQUATIONS.md."""

    tag: str
    chapter: str
    line: int


@dataclass
class BookIndex:
    """All indexes for a single textbook."""

    name: str
    path: Path
    chapters: list[str] = field(default_factory=list)
    chapter_summaries: dict[str, str] = field(default_factory=dict)
    concepts: list[ConceptEntry] = field(default_factory=list)
    terms: list[TermEntry] = field(default_factory=list)
    sections: dict[str, list[SectionHeading]] = field(default_factory=dict)
    equations: list[EquationEntry] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Module-level index store
# ---------------------------------------------------------------------------

_books: dict[str, BookIndex] = {}


def get_books() -> dict[str, BookIndex]:
    """Return the loaded book indexes."""
    return _books


def get_book(name: str) -> BookIndex | None:
    """Return a specific book index by name (case-insensitive partial match)."""
    name_lower = name.lower()
    for book_name, book in _books.items():
        if book_name.lower() == name_lower or name_lower in book_name.lower():
            return book
    return None


# ---------------------------------------------------------------------------
# Textbooks root discovery
# ---------------------------------------------------------------------------

_DEFAULT_TEXTBOOKS_ROOT = Path(
    os.environ.get(
        "TEXTBOOKS_ROOT",
        "/Users/thellert/LBL/ML/accellpapers/data/downloads/manual/processed_textbooks",
    )
)


def get_textbooks_root() -> Path:
    """Return the root directory containing processed textbooks."""
    return Path(os.environ.get("TEXTBOOKS_ROOT", str(_DEFAULT_TEXTBOOKS_ROOT)))


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def _parse_concept_map(index_md: Path) -> tuple[list[ConceptEntry], list[str], dict[str, str]]:
    """Parse INDEX.md to extract concept map, chapter list, and chapter summaries."""
    concepts: list[ConceptEntry] = []
    chapters: list[str] = []
    summaries: dict[str, str] = {}

    text = index_md.read_text(encoding="utf-8")
    lines = text.splitlines()

    in_concept_table = False
    in_chapters = False
    current_chapter = ""

    for line in lines:
        stripped = line.strip()

        # Detect concept map table
        if stripped.startswith("| Concept"):
            in_concept_table = True
            continue
        if in_concept_table and stripped.startswith("|---"):
            continue
        if in_concept_table and stripped.startswith("|"):
            parts = [p.strip() for p in stripped.split("|")]
            # parts: ['', concept, chapter, section_info, '']
            if len(parts) >= 4:
                concept = parts[1]
                chapter = parts[2]
                section_info = parts[3]
                # Parse "Section Name (L123)" format
                m = re.match(r"(.+?)\s*\(L(\d+)\)", section_info)
                if m:
                    section = m.group(1).strip()
                    line_num = int(m.group(2))
                    concepts.append(ConceptEntry(concept, chapter, section, line_num))
            continue
        if in_concept_table and not stripped.startswith("|"):
            in_concept_table = False

        # Detect chapters section
        if stripped == "## Chapters":
            in_chapters = True
            continue

        if in_chapters:
            # Chapter heading: ### ch01_special_relativity.md (lines 1-436 of source)
            ch_match = re.match(r"###\s+(ch\d+_\w+)\.md", stripped)
            if ch_match:
                current_chapter = ch_match.group(1)
                chapters.append(current_chapter)
                continue
            # Chapter summary is the non-empty line after the heading
            if current_chapter and stripped and not stripped.startswith("**Sections"):
                summaries[current_chapter] = stripped
                current_chapter = ""

    return concepts, chapters, summaries


def _parse_sections(sections_md: Path) -> list[SectionHeading]:
    """Parse a *_SECTIONS.md file to extract headings with line numbers."""
    headings: list[SectionHeading] = []
    text = sections_md.read_text(encoding="utf-8")

    in_table = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("| Line"):
            in_table = True
            continue
        if in_table and stripped.startswith("|---"):
            continue
        if in_table and stripped.startswith("|"):
            parts = [p.strip() for p in stripped.split("|")]
            if len(parts) >= 4:
                try:
                    line_num = int(parts[1])
                except ValueError:
                    continue
                heading = parts[2]
                level = parts[3]
                headings.append(SectionHeading(line_num, heading, level))
            continue
        if in_table and not stripped.startswith("|"):
            break

    return headings


def _parse_terms(terms_json: Path) -> list[TermEntry]:
    """Parse TERMS.json to extract term entries."""
    data = json.loads(terms_json.read_text(encoding="utf-8"))
    return [TermEntry(t["term"], t.get("refs", [])) for t in data.get("terms", [])]


def _parse_equations(equations_md: Path) -> list[EquationEntry]:
    """Parse EQUATIONS.md to extract equation entries."""
    equations: list[EquationEntry] = []
    text = equations_md.read_text(encoding="utf-8")

    in_table = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("| Tag"):
            in_table = True
            continue
        if in_table and stripped.startswith("|---"):
            continue
        if in_table and stripped.startswith("|"):
            parts = [p.strip() for p in stripped.split("|")]
            if len(parts) >= 4:
                tag = parts[1]
                chapter = parts[2]
                try:
                    line_num = int(parts[3])
                except ValueError:
                    continue
                equations.append(EquationEntry(tag, chapter, line_num))
            continue
        if in_table and not stripped.startswith("|"):
            break

    return equations


# ---------------------------------------------------------------------------
# Book loading
# ---------------------------------------------------------------------------


def _load_book(book_dir: Path) -> BookIndex:
    """Load all indexes for a single book."""
    book_name = book_dir.name
    book = BookIndex(name=book_name, path=book_dir)

    index_md = book_dir / "INDEX.md"
    if index_md.exists():
        book.concepts, book.chapters, book.chapter_summaries = _parse_concept_map(index_md)
        logger.info("  %s: %d concepts, %d chapters", book_name, len(book.concepts), len(book.chapters))

    # Load section headings for each chapter
    for chapter in book.chapters:
        sections_file = book_dir / f"{chapter}_SECTIONS.md"
        if sections_file.exists():
            book.sections[chapter] = _parse_sections(sections_file)

    # Also load back_matter sections if available
    back_sections = book_dir / "back_matter_SECTIONS.md"
    if back_sections.exists():
        book.sections["back_matter"] = _parse_sections(back_sections)

    terms_json = book_dir / "TERMS.json"
    if terms_json.exists():
        book.terms = _parse_terms(terms_json)
        logger.info("  %s: %d index terms", book_name, len(book.terms))

    equations_md = book_dir / "EQUATIONS.md"
    if equations_md.exists():
        book.equations = _parse_equations(equations_md)
        logger.info("  %s: %d equations", book_name, len(book.equations))

    return book


def load_all_books(root: Path | None = None) -> dict[str, BookIndex]:
    """Discover and load all processed textbooks.

    Args:
        root: Root directory containing book subdirectories.
              Defaults to TEXTBOOKS_ROOT env var.

    Returns:
        Dict mapping book name to BookIndex.
    """
    global _books

    root = root or get_textbooks_root()
    if not root.is_dir():
        logger.warning("Textbooks root not found: %s", root)
        return _books

    for entry in sorted(root.iterdir()):
        if entry.is_dir() and (entry / "INDEX.md").exists():
            book = _load_book(entry)
            _books[book.name] = book
            logger.info("Loaded textbook: %s", book.name)

    logger.info("Total textbooks loaded: %d", len(_books))
    return _books


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------


def _similarity(a: str, b: str) -> float:
    """Compute string similarity ratio (0-1)."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def search_indexes(
    query: str,
    book_name: str | None = None,
    max_results: int = 15,
) -> list[dict]:
    """Search across all indexes for a query string.

    Searches concept map, term index, section headings, and equation tags.
    Returns matches sorted by relevance score.
    """
    query_lower = query.lower().strip()
    if not query_lower:
        return []

    results: list[dict] = []
    books_to_search = [get_book(book_name)] if book_name else list(_books.values())
    books_to_search = [b for b in books_to_search if b is not None]

    for book in books_to_search:
        # Search concepts
        for c in book.concepts:
            concept_lower = c.concept.lower()
            if query_lower in concept_lower:
                score = 1.0 if query_lower == concept_lower else 0.9
            elif concept_lower in query_lower:
                score = 0.85
            else:
                sim = _similarity(query_lower, concept_lower)
                score = sim * 0.8 if sim > 0.5 else 0.0

            if score > 0:
                results.append({
                    "type": "concept",
                    "term": c.concept,
                    "book": book.name,
                    "chapter": c.chapter,
                    "section": c.section,
                    "line": c.line,
                    "score": score,
                })

        # Search terms
        for t in book.terms:
            term_lower = t.term.lower()
            if query_lower in term_lower:
                score = 0.8 if query_lower == term_lower else 0.7
            elif term_lower in query_lower:
                score = 0.65
            else:
                sim = _similarity(query_lower, term_lower)
                score = sim * 0.6 if sim > 0.5 else 0.0

            if score > 0:
                results.append({
                    "type": "term",
                    "term": t.term,
                    "book": book.name,
                    "pages": t.pages,
                    "score": score,
                })

        # Search section headings
        for chapter, headings in book.sections.items():
            for h in headings:
                heading_lower = h.heading.lower()
                if query_lower in heading_lower:
                    score = 0.75 if query_lower == heading_lower else 0.65
                elif heading_lower in query_lower:
                    score = 0.6
                else:
                    sim = _similarity(query_lower, heading_lower)
                    score = sim * 0.55 if sim > 0.5 else 0.0

                if score > 0:
                    results.append({
                        "type": "section",
                        "term": h.heading,
                        "book": book.name,
                        "chapter": chapter,
                        "line": h.line,
                        "level": h.level,
                        "score": score,
                    })

        # Search equations by tag
        for eq in book.equations:
            if query_lower in eq.tag.lower() or eq.tag.lower() in query_lower:
                results.append({
                    "type": "equation",
                    "term": f"Eq. {eq.tag}",
                    "book": book.name,
                    "chapter": eq.chapter,
                    "line": eq.line,
                    "score": 0.85,
                })

    # Sort by score descending, deduplicate by (type, term, chapter)
    results.sort(key=lambda r: r["score"], reverse=True)

    seen: set[tuple] = set()
    deduped: list[dict] = []
    for r in results:
        key = (r["type"], r["term"], r.get("chapter", ""))
        if key not in seen:
            seen.add(key)
            deduped.append(r)
            if len(deduped) >= max_results:
                break

    return deduped


def find_section_bounds(
    book: BookIndex,
    chapter: str,
    section_name: str,
) -> tuple[int, int] | None:
    """Find the start and end line of a section in a chapter.

    Returns (start_line, end_line) where end_line is the line before
    the next section of equal or higher level.  Returns None if not found.
    """
    headings = book.sections.get(chapter, [])
    if not headings:
        return None

    section_lower = section_name.lower().strip()

    # Find the matching heading
    match_idx = None
    for i, h in enumerate(headings):
        if section_lower in h.heading.lower() or h.heading.lower() in section_lower:
            match_idx = i
            break

    # Try fuzzy match if exact substring fails
    if match_idx is None:
        best_sim = 0.0
        for i, h in enumerate(headings):
            sim = _similarity(section_lower, h.heading.lower())
            if sim > best_sim and sim > 0.5:
                best_sim = sim
                match_idx = i

    if match_idx is None:
        return None

    start_heading = headings[match_idx]
    start_line = start_heading.line

    # Find end: next heading of equal or higher level
    level_rank = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
    start_rank = level_rank.get(start_heading.level, 4)

    end_line = None
    for h in headings[match_idx + 1 :]:
        h_rank = level_rank.get(h.level, 4)
        if h_rank <= start_rank:
            end_line = h.line - 1
            break

    # If no next heading found, read to the end of the chapter file
    if end_line is None:
        chapter_file = book.path / f"{chapter}.md"
        if chapter_file.exists():
            end_line = sum(1 for _ in chapter_file.open(encoding="utf-8"))

    return (start_line, end_line) if end_line else None
