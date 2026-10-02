"""Tests for the entry-text budget and cut marker in ``serialize_entry``."""

import inspect
from datetime import UTC, datetime
from unittest.mock import patch

from osprey.mcp_server.ariel import server
from osprey.mcp_server.ariel.server import serialize_entry


def _no_template(path, default=None, _config_path=None):
    """Answer the entry_url template with None so no entry_url is emitted."""
    if path == "ariel.entry_url_template":
        return None
    return default


def _entry(raw_text: str) -> dict:
    return {
        "entry_id": "e1",
        "timestamp": datetime(2026, 6, 1, 0, 0, 0, tzinfo=UTC),
        "author": "op",
        "source_system": "elog",
        "raw_text": raw_text,
        "summary": None,
    }


def test_an_uncut_entry_keeps_its_shape():
    text = "beam loss on the north arc"
    with patch("osprey.utils.config.get_config_value", _no_template):
        out = serialize_entry(_entry(text), text_limit=len(text) + 10)
    assert set(out) == {"entry_id", "timestamp", "author", "source_system", "raw_text", "summary"}
    assert out["raw_text"] == text


def test_a_cut_entry_says_so_and_gives_its_length():
    with patch("osprey.utils.config.get_config_value", _no_template):
        out = serialize_entry(_entry("abcdefghi"), text_limit=4)
    assert out["raw_text"] == "abcd"
    assert out["raw_text_truncated"] is True
    assert out["raw_text_length"] == 9


def test_the_budget_is_a_required_keyword():
    param = inspect.signature(serialize_entry).parameters["text_limit"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is inspect.Parameter.empty


def test_server_instructions_explain_the_cut_marker():
    instructions = server.mcp.instructions
    assert "raw_text_truncated" in instructions
    assert "raw_text_length" in instructions
    assert "entry_get" in instructions
