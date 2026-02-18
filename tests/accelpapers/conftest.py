"""Shared fixtures for AccelPapers MCP server tests.

Provides in-memory SQLite fixtures and sample paper data for testing
tools and indexer in isolation.
"""

import json
import sqlite3
from pathlib import Path

import pytest


def get_tool_fn(tool_or_fn):
    """Extract the raw async function from a FastMCP FunctionTool."""
    if hasattr(tool_or_fn, "fn"):
        return tool_or_fn.fn
    return tool_or_fn


SAMPLE_PAPERS = [
    {
        "texkey": "Smith:2020abc",
        "first_author_full_name": "Smith, John",
        "first_author_affiliations": "CERN",
        "other_authors_full_names": ["Doe, Jane", "Wang, Li"],
        "title": "Beam position monitor calibration for synchrotron light sources",
        "year": "2020",
        "keywords": ["beam position monitor", "calibration", "synchrotron"],
        "earliest_date": "2020-03-15",
        "number_of_pages": "12",
        "pdf_url": "https://example.com/smith2020.pdf",
        "citation_count": 45,
        "abstract_value": "We present a novel calibration method for beam position monitors (BPMs) at synchrotron light sources. The method achieves sub-micron accuracy.",
        "doi": "10.1234/test.2020.001",
        "c_num": "",
        "paper_id": "001",
        "arxiv_eprints": "2003.12345",
        "inspireHEP_url": "https://inspirehep.net/literature/123456",
        "conf_acronym": "IPAC2020",
        "journal_title": "Phys.Rev.Accel.Beams",
        "document_type": "conference paper",
        "publisher": "",
        "content": {
            "tables": [],
            "figures": [],
            "footnotes": [],
            "sections": {
                "1": {
                    "title": "Abstract",
                    "full_text": ["We present a novel calibration method for BPMs."],
                },
                "2": {
                    "title": "Introduction",
                    "full_text": [
                        "Beam position monitors are essential instruments.",
                        "Accurate calibration is critical for orbit correction.",
                    ],
                },
            },
        },
    },
    {
        "texkey": "Chen:2019xyz",
        "first_author_full_name": "Chen, Wei",
        "first_author_affiliations": "SLAC",
        "other_authors_full_names": [],
        "title": "RF cavity design optimization using machine learning",
        "year": "2019",
        "keywords": ["RF cavity", "machine learning", "optimization"],
        "earliest_date": "2019-06-01",
        "number_of_pages": "8",
        "pdf_url": "https://example.com/chen2019.pdf",
        "citation_count": 120,
        "abstract_value": "Machine learning techniques are applied to optimize RF cavity geometry for particle accelerators.",
        "doi": "10.1234/test.2019.002",
        "c_num": "",
        "paper_id": "002",
        "arxiv_eprints": "1906.54321",
        "inspireHEP_url": "https://inspirehep.net/literature/789012",
        "conf_acronym": "",
        "journal_title": "Nucl.Instrum.Meth.A",
        "document_type": "article",
        "publisher": "Elsevier",
        "content": {
            "tables": [],
            "figures": [],
            "footnotes": [],
            "sections": {
                "1": {
                    "title": "Abstract",
                    "full_text": ["ML techniques for RF cavity optimization."],
                },
            },
        },
    },
    {
        "texkey": "Mueller:2021def",
        "first_author_full_name": "Mueller, Hans",
        "first_author_affiliations": "DESY",
        "other_authors_full_names": ["Smith, John"],
        "title": "Undulator radiation characterization at PETRA IV",
        "year": "2021",
        "keywords": ["undulator", "radiation", "PETRA IV", "synchrotron"],
        "earliest_date": "2021-09-20",
        "number_of_pages": "6",
        "pdf_url": "",
        "citation_count": 15,
        "abstract_value": "Undulator radiation properties at the upcoming PETRA IV facility are characterized through simulation and measurement.",
        "doi": "",
        "c_num": "",
        "paper_id": "003",
        "arxiv_eprints": "",
        "inspireHEP_url": "https://inspirehep.net/literature/345678",
        "conf_acronym": "IPAC2021",
        "journal_title": "",
        "document_type": "conference paper",
        "publisher": "",
        "content": None,
    },
    {
        "texkey": "Tanaka:2018ghi",
        "first_author_full_name": "Tanaka, Kenji",
        "first_author_affiliations": "KEK, Tsukuba",
        "other_authors_full_names": ["Yamamoto, Hitoshi"],
        "title": "Bunch length measurement using streak camera at SuperKEKB",
        "year": "2018",
        "keywords": ["bunch length", "streak camera", "SuperKEKB"],
        "earliest_date": "2018-01-10",
        "number_of_pages": "4",
        "pdf_url": "https://example.com/tanaka2018.pdf",
        "citation_count": 8,
        "abstract_value": "We report bunch length measurements at the SuperKEKB electron-positron collider using a streak camera system.",
        "doi": "10.1234/test.2018.004",
        "c_num": "",
        "paper_id": "004",
        "arxiv_eprints": "1801.98765",
        "inspireHEP_url": "https://inspirehep.net/literature/567890",
        "conf_acronym": "NAPAC2018",
        "journal_title": "",
        "document_type": "conference paper",
        "publisher": "",
        "content": {
            "tables": [],
            "figures": [],
            "footnotes": [],
            "sections": {
                "1": {
                    "title": "Introduction",
                    "full_text": ["SuperKEKB requires precise bunch length control."],
                },
            },
        },
    },
    {
        "texkey": "Jones:2022jkl",
        "first_author_full_name": "Jones, Alice",
        "first_author_affiliations": "Argonne",
        "other_authors_full_names": [],
        "title": "Beam position monitor electronics upgrade for APS-U",
        "year": "2022",
        "keywords": ["beam position monitor", "APS-U", "electronics"],
        "earliest_date": "2022-05-05",
        "number_of_pages": "10",
        "pdf_url": "https://example.com/jones2022.pdf",
        "citation_count": 3,
        "abstract_value": "The APS Upgrade requires new BPM electronics with improved resolution and faster data rates.",
        "doi": "10.1234/test.2022.005",
        "c_num": "",
        "paper_id": "005",
        "arxiv_eprints": "",
        "inspireHEP_url": "https://inspirehep.net/literature/901234",
        "conf_acronym": "IPAC2022",
        "journal_title": "",
        "document_type": "conference paper",
        "publisher": "",
        "content": {
            "tables": [],
            "figures": [],
            "footnotes": [],
            "sections": {
                "1": {
                    "title": "Abstract",
                    "full_text": ["New BPM electronics for APS-U."],
                },
            },
        },
    },
]


@pytest.fixture
def sample_json_dir(tmp_path):
    """Create a temporary directory with sample paper JSON files."""
    # Create subdirectories mimicking INSPIRE structure
    ipac_dir = tmp_path / "IPAC2020"
    ipac_dir.mkdir()
    arxiv_dir = tmp_path / "ARXIV-acc"
    arxiv_dir.mkdir()
    ipac21_dir = tmp_path / "IPAC2021"
    ipac21_dir.mkdir()
    napac_dir = tmp_path / "NAPAC2018"
    napac_dir.mkdir()
    ipac22_dir = tmp_path / "IPAC2022"
    ipac22_dir.mkdir()

    # Write papers to appropriate subdirs
    target_dirs = [ipac_dir, arxiv_dir, ipac21_dir, napac_dir, ipac22_dir]
    for paper, target in zip(SAMPLE_PAPERS, target_dirs):
        fpath = target / f"{paper['texkey'].replace(':', '-')}.json"
        fpath.write_text(json.dumps(paper))

    return tmp_path


@pytest.fixture
def indexed_db(tmp_path, sample_json_dir):
    """Build an indexed database from sample papers and return its path."""
    from osprey.mcp_server.accelpapers.indexer import build_index

    db_path = tmp_path / "test_papers.db"
    build_index(data_dir=sample_json_dir, db_path=db_path)
    return db_path


@pytest.fixture
def db_connection(indexed_db):
    """Return a connection to the indexed test database."""
    conn = sqlite3.connect(str(indexed_db))
    conn.row_factory = sqlite3.Row
    return conn


@pytest.fixture
def patch_db(indexed_db, monkeypatch):
    """Patch get_db_path to return the test database path."""
    monkeypatch.setattr(
        "osprey.mcp_server.accelpapers.db.get_db_path",
        lambda: indexed_db,
    )
    return indexed_db
