"""Fixtures for textbooks MCP server tests."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from osprey.mcp_server.textbooks import indexer


@pytest.fixture()
def sample_book_dir(tmp_path: Path) -> Path:
    """Create a minimal textbook directory with INDEX.md, SECTIONS.md, etc."""
    book_dir = tmp_path / "TestBook"
    book_dir.mkdir()

    # INDEX.md
    (book_dir / "INDEX.md").write_text(textwrap.dedent("""\
        # Test Book — Reference Index

        **Source**: `TestBook.mmd`
        **Total chapters**: 2

        ## Quick Concept Map

        | Concept | Chapter | Section (line) |
        |---------|---------|----------------|
        | Betatron function | ch01 | Betatron Function (L50) |
        | Phase stability | ch01 | Phase Stability (L10) |
        | Luminosity | ch02 | Luminosity Definition (L5) |

        ## Chapters

        ### ch01_dynamics.md (lines 1-100 of source)
        Covers beam dynamics, phase stability, and betatron functions.
        **Sections index**: ch01_dynamics_SECTIONS.md

        ### ch02_colliders.md (lines 101-150 of source)
        Covers collider luminosity and beam-beam effects.
        **Sections index**: ch02_colliders_SECTIONS.md
    """))

    # ch01_dynamics_SECTIONS.md
    (book_dir / "ch01_dynamics_SECTIONS.md").write_text(textwrap.dedent("""\
        # ch01_dynamics.md — Section Index

        | Line | Heading | Level |
        |------|---------|-------|
        | 1 | Beam Dynamics | h2 |
        | 10 | Phase Stability | h3 |
        | 30 | Adiabatic Damping | h3 |
        | 50 | Betatron Function | h3 |
        | 80 | Summary | h3 |
    """))

    # ch02_colliders_SECTIONS.md
    (book_dir / "ch02_colliders_SECTIONS.md").write_text(textwrap.dedent("""\
        # ch02_colliders.md — Section Index

        | Line | Heading | Level |
        |------|---------|-------|
        | 1 | Colliders | h2 |
        | 5 | Luminosity Definition | h3 |
        | 25 | Beam-Beam Effects | h3 |
    """))

    # TERMS.json
    (book_dir / "TERMS.json").write_text(json.dumps({
        "book": "TestBook",
        "terms": [
            {"term": "Betatron tune", "refs": [94, 101]},
            {"term": "Luminosity", "refs": [251]},
            {"term": "Phase stability", "refs": [60]},
            {"term": "Adiabatic damping", "refs": [60]},
        ],
    }))

    # EQUATIONS.md
    (book_dir / "EQUATIONS.md").write_text(textwrap.dedent("""\
        # Equation Index

        | Tag | Chapter | Line |
        |-----|---------|------|
        | 1.1 | ch01_dynamics | 15 |
        | 1.2 | ch01_dynamics | 35 |
        | 2.1 | ch02_colliders | 12 |
    """))

    # ch01_dynamics.md - chapter content
    ch01_lines = []
    for i in range(1, 101):
        if i == 1:
            ch01_lines.append("## Beam Dynamics")
        elif i == 10:
            ch01_lines.append("### Phase Stability")
        elif i == 11:
            ch01_lines.append("Phase stability ensures bounded motion in longitudinal phase space.")
        elif i == 15:
            ch01_lines.append(r"\[E = mc^2 \tag{1.1}\]")
        elif i == 30:
            ch01_lines.append("### Adiabatic Damping")
        elif i == 31:
            ch01_lines.append("As particles accelerate, the oscillation amplitude decreases.")
        elif i == 35:
            ch01_lines.append(r"\[\Omega_l = \sqrt{\frac{qE_0\omega\sin\psi_s}{m_0v^3}} \tag{1.2}\]")
        elif i == 50:
            ch01_lines.append("### Betatron Function")
        elif i == 51:
            ch01_lines.append("The betatron function describes the envelope of particle oscillations.")
        elif i == 80:
            ch01_lines.append("### Summary")
        elif i == 81:
            ch01_lines.append("This chapter covered the fundamentals of beam dynamics.")
        else:
            ch01_lines.append(f"Line {i} of chapter 1 content.")
    (book_dir / "ch01_dynamics.md").write_text("\n".join(ch01_lines))

    # ch02_colliders.md
    ch02_lines = []
    for i in range(1, 51):
        if i == 1:
            ch02_lines.append("## Colliders")
        elif i == 5:
            ch02_lines.append("### Luminosity Definition")
        elif i == 6:
            ch02_lines.append("Luminosity measures the collision rate per unit cross section.")
        elif i == 12:
            ch02_lines.append(r"\[L = \frac{N_1 N_2 f}{4\pi\sigma_x\sigma_y} \tag{2.1}\]")
        elif i == 25:
            ch02_lines.append("### Beam-Beam Effects")
        elif i == 26:
            ch02_lines.append("The electromagnetic field of one beam affects the other.")
        else:
            ch02_lines.append(f"Line {i} of chapter 2 content.")
    (book_dir / "ch02_colliders.md").write_text("\n".join(ch02_lines))

    return book_dir


@pytest.fixture()
def loaded_book(sample_book_dir: Path) -> indexer.BookIndex:
    """Load the sample book and return the BookIndex."""
    # Clear existing books
    indexer._books.clear()
    indexer.load_all_books(sample_book_dir.parent)
    return indexer.get_book("TestBook")
