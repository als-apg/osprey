"""Unit tests for the shared ARIEL search-tool envelope helpers."""

import json

import pytest

from osprey.mcp_server.ariel.server_context import initialize_ariel_context
from osprey.mcp_server.ariel.tools.search_envelope import ResultWindow, advanced_params
from osprey.services.ariel_search.models import DEFAULT_LISTING_TEXT_CHARS
from tests.mcp_server.ariel.conftest import make_mock_entry


def _setup_registry(tmp_path, monkeypatch, entry_text=None):
    monkeypatch.chdir(tmp_path)
    ariel: dict = {"database": {"uri": "postgresql://localhost/test"}}
    if entry_text is not None:
        ariel["entry_text"] = entry_text
    (tmp_path / "config.yml").write_text(json.dumps({"ariel": ariel}))
    initialize_ariel_context()


@pytest.mark.parametrize("value", [True, False])
def test_advanced_params_carries_an_explicit_rerank(value):
    """An explicit choice is an override and has to reach the service."""
    assert advanced_params(rerank=value)["rerank"] is value


def test_advanced_params_omits_an_unset_rerank():
    """An absent key is how the service hears "no preference"."""
    assert "rerank" not in advanced_params()


def test_advanced_params_keeps_the_other_filters_independent():
    """Adding rerank must not disturb what the other arguments emit."""
    params = advanced_params(author="chen", expand_query=False, rerank=True)

    assert params == {"author": "chen", "expand_query": False, "rerank": True}


def test_select_cuts_at_the_configured_listing_budget(tmp_path, monkeypatch):
    _setup_registry(tmp_path, monkeypatch, entry_text={"listing_chars": 10})

    [out] = ResultWindow.build(5, None).select([make_mock_entry(raw_text="x" * 25)])

    assert out["raw_text"] == "x" * 10
    assert out["raw_text_truncated"] is True
    assert out["raw_text_length"] == 25


def test_select_keeps_a_short_entry_whole_at_the_default_budget(tmp_path, monkeypatch):
    _setup_registry(tmp_path, monkeypatch)
    text = "x" * DEFAULT_LISTING_TEXT_CHARS

    [out] = ResultWindow.build(5, None).select([make_mock_entry(raw_text=text)])

    assert out["raw_text"] == text
    assert "raw_text_truncated" not in out
    assert "raw_text_length" not in out
