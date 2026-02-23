"""Tests for the AccelPapers indexer — helpers and batch indexing."""

import json
from unittest.mock import MagicMock, patch

import pytest

from osprey.mcp_server.accelpapers.indexer import (
    PAPERS_SCHEMA,
    _parse_paper,
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


# --- _parse_paper tests -------------------------------------------------------


class TestParsePaper:
    def test_maps_texkey_to_id(self):
        data = {"texkey": "Test:2020abc", "title": "A paper", "content": None}
        doc = _parse_paper(data, "/tmp/test.json", "")
        assert doc is not None
        assert doc["id"] == "Test:2020abc"
        assert "texkey" not in doc

    def test_missing_texkey_returns_none(self):
        data = {"title": "No texkey", "content": None}
        assert _parse_paper(data, "/tmp/test.json", "") is None

    def test_missing_title_returns_none(self):
        data = {"texkey": "Test:2020abc", "content": None}
        assert _parse_paper(data, "/tmp/test.json", "") is None


# --- Indexer integration tests ------------------------------------------------


class TestBuildIndex:
    def test_builds_collection(self, tmp_path, sample_json_dir):
        """Test that build_index creates a collection and imports documents."""
        mock_docs = MagicMock()
        mock_docs.import_ = MagicMock(
            return_value=[{"success": True} for _ in range(5)]
        )
        mock_collection = MagicMock()
        mock_collection.documents = mock_docs

        mock_collections = MagicMock()
        mock_collections.__getitem__ = MagicMock(return_value=mock_collection)
        mock_collections.create = MagicMock(return_value={})

        mock_client = MagicMock()
        mock_client.collections = mock_collections

        with (
            patch(
                "osprey.mcp_server.accelpapers.db.get_client",
                return_value=mock_client,
            ),
            patch(
                "osprey.mcp_server.accelpapers.db.get_collection_name",
                return_value="test_papers",
            ),
        ):
            result = build_index(data_dir=sample_json_dir)

        assert result == "test_papers"
        mock_collections.create.assert_called_once()
        # At least one import_ call should have been made
        assert mock_docs.import_.call_count >= 1

    def test_missing_data_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_index(data_dir=tmp_path / "nonexistent")

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

        mock_docs = MagicMock()
        mock_docs.import_ = MagicMock(return_value=[{"success": True}])
        mock_collection = MagicMock()
        mock_collection.documents = mock_docs

        mock_collections = MagicMock()
        mock_collections.__getitem__ = MagicMock(return_value=mock_collection)
        mock_collections.create = MagicMock(return_value={})

        mock_client = MagicMock()
        mock_client.collections = mock_collections

        with (
            patch(
                "osprey.mcp_server.accelpapers.db.get_client",
                return_value=mock_client,
            ),
            patch(
                "osprey.mcp_server.accelpapers.db.get_collection_name",
                return_value="test_papers",
            ),
        ):
            result = build_index(data_dir=data_dir)

        assert result == "test_papers"
        # Should have imported 1 valid document
        assert mock_docs.import_.call_count == 1

    def test_schema_has_embedding_field(self):
        """Verify the schema includes auto-embedding config."""
        embedding_field = None
        for field in PAPERS_SCHEMA["fields"]:
            if field["name"] == "embedding":
                embedding_field = field
                break
        assert embedding_field is not None
        assert embedding_field["type"] == "float[]"
        assert embedding_field["num_dim"] == 768
        assert "embed" in embedding_field
        assert embedding_field["embed"]["from"] == ["title", "abstract", "keywords"]
