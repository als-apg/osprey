"""Tests for the AccelPapers indexer — helpers and batch indexing."""

import json
import sqlite3

import pytest

from osprey.mcp_server.accelpapers.indexer import (
    build_all_authors,
    build_index,
    extract_full_text,
    get_conference,
    normalize_num_pages,
    normalize_year,
)


# --- Helper function tests ---------------------------------------------------


class TestNormalizeYear:
    def test_string_year(self):
        assert normalize_year("2020") == 2020

    def test_int_year(self):
        assert normalize_year(2021) == 2021

    def test_date_string(self):
        assert normalize_year("2019-06-01") == 2019

    def test_empty(self):
        assert normalize_year("") is None

    def test_none(self):
        assert normalize_year(None) is None

    def test_invalid(self):
        assert normalize_year("abc") is None

    def test_far_future(self):
        assert normalize_year("2200") is None


class TestNormalizeNumPages:
    def test_string(self):
        assert normalize_num_pages("12") == 12

    def test_int(self):
        assert normalize_num_pages(8) == 8

    def test_none(self):
        assert normalize_num_pages(None) is None

    def test_empty(self):
        assert normalize_num_pages("") is None

    def test_mixed(self):
        assert normalize_num_pages("5 pages") == 5


class TestExtractFullText:
    def test_basic_sections(self):
        content = {
            "sections": {
                "1": {"title": "Abstract", "full_text": ["First paragraph."]},
                "2": {"title": "Intro", "full_text": ["Second paragraph."]},
            }
        }
        text = extract_full_text(content)
        assert "Abstract" in text
        assert "First paragraph." in text
        assert "Intro" in text
        assert "Second paragraph." in text

    def test_none_content(self):
        assert extract_full_text(None) == ""

    def test_no_sections(self):
        assert extract_full_text({"tables": []}) == ""

    def test_empty_sections(self):
        assert extract_full_text({"sections": {}}) == ""

    def test_string_full_text(self):
        content = {"sections": {"1": {"title": "T", "full_text": "plain string"}}}
        text = extract_full_text(content)
        assert "plain string" in text


class TestGetConference:
    def test_from_conf_acronym(self):
        data = {"conf_acronym": "IPAC2023"}
        assert get_conference(data, "some-dir") == "IPAC2023"

    def test_fallback_to_subdir(self):
        data = {"conf_acronym": ""}
        assert get_conference(data, "NAPAC2016") == "NAPAC2016"

    def test_no_fallback_for_arxiv(self):
        data = {"conf_acronym": ""}
        assert get_conference(data, "ARXIV-acc") == ""

    def test_no_fallback_for_book(self):
        data = {"conf_acronym": ""}
        assert get_conference(data, "book-chapter") == ""

    def test_empty_subdir(self):
        data = {"conf_acronym": ""}
        assert get_conference(data, "") == ""


class TestBuildAllAuthors:
    def test_single_author(self):
        data = {"first_author_full_name": "Smith, John", "other_authors_full_names": []}
        assert build_all_authors(data) == "Smith, John"

    def test_multiple_authors(self):
        data = {
            "first_author_full_name": "Smith, John",
            "other_authors_full_names": ["Doe, Jane", "Wang, Li"],
        }
        result = build_all_authors(data)
        assert "Smith, John" in result
        assert "Doe, Jane" in result
        assert "Wang, Li" in result

    def test_no_author(self):
        data = {"first_author_full_name": "", "other_authors_full_names": []}
        assert build_all_authors(data) == ""


# --- Indexer integration tests ------------------------------------------------


class TestBuildIndex:
    def test_builds_db(self, tmp_path, sample_json_dir):
        db_path = tmp_path / "test.db"
        result = build_index(data_dir=sample_json_dir, db_path=db_path)
        assert result == db_path
        assert db_path.exists()

    def test_correct_row_count(self, indexed_db):
        conn = sqlite3.connect(str(indexed_db))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT COUNT(*) as c FROM papers").fetchone()
        assert row["c"] == 5
        conn.close()

    def test_fts_populated(self, indexed_db):
        conn = sqlite3.connect(str(indexed_db))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM papers_fts WHERE papers_fts MATCH 'beam position monitor'"
        ).fetchall()
        assert len(rows) >= 2  # Smith and Jones papers
        conn.close()

    def test_stats_materialized(self, indexed_db):
        conn = sqlite3.connect(str(indexed_db))
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT value FROM stats WHERE key='total_papers'").fetchone()
        assert row is not None
        assert int(row["value"]) == 5
        conn.close()

    def test_conference_indexed(self, indexed_db):
        conn = sqlite3.connect(str(indexed_db))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT texkey FROM papers WHERE conference = 'IPAC2020'"
        ).fetchall()
        assert len(rows) >= 1
        conn.close()

    def test_missing_data_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_index(data_dir=tmp_path / "nonexistent", db_path=tmp_path / "test.db")

    def test_invalid_json_skipped(self, tmp_path):
        """Files that aren't valid JSON should be skipped without crashing."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "bad.json").write_text("NOT JSON")
        (data_dir / "good.json").write_text(json.dumps({
            "texkey": "Test:2020abc",
            "title": "A valid paper",
            "year": "2020",
            "content": None,
        }))

        db_path = tmp_path / "test.db"
        build_index(data_dir=data_dir, db_path=db_path)

        conn = sqlite3.connect(str(db_path))
        row = conn.execute("SELECT COUNT(*) as c FROM papers").fetchone()
        assert row[0] == 1
        conn.close()
