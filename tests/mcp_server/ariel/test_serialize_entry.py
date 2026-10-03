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
        out = serialize_entry(
            _entry(text),
            text_limit=len(text) + 10,
            attachment_limit=0,
            attachment_rows=None,
            model_id=None,
            file_source=False,
        )
    assert set(out) == {"entry_id", "timestamp", "author", "source_system", "raw_text", "summary"}
    assert out["raw_text"] == text


def test_a_cut_entry_says_so_and_gives_its_length():
    with patch("osprey.utils.config.get_config_value", _no_template):
        out = serialize_entry(
            _entry("abcdefghi"),
            text_limit=4,
            attachment_limit=0,
            attachment_rows=None,
            model_id=None,
            file_source=False,
        )
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


# ---------------------------------------------------------------------------
# Attachment summaries and match evidence
# ---------------------------------------------------------------------------

_PNG = {"url": "https://elog.example/f/plot.png", "type": "image/png", "filename": "plot.png"}
_PDF = {"url": "https://elog.example/f/report.pdf", "type": "application/pdf", "filename": "r.pdf"}


def _with_attachments(*items: dict, **extra) -> dict:
    entry = _entry("text")
    entry["attachments"] = list(items)
    entry.update(extra)
    return entry


def _serialize(entry: dict, **overrides):
    kwargs = {
        "text_limit": 100,
        "attachment_limit": 5,
        "attachment_rows": None,
        "model_id": None,
        "file_source": False,
    }
    kwargs.update(overrides)
    with patch("osprey.utils.config.get_config_value", _no_template):
        return serialize_entry(entry, **kwargs)


def _copied_row(entry_id: str, item: dict) -> dict:
    from osprey.services.ariel_search.attachments import attachment_id_for

    return {
        "attachment_id": attachment_id_for(entry_id, item),
        "entry_id": entry_id,
        "filename": item["filename"],
        "mime_type": item["type"],
        "copy_status": "copied",
        "skip_reason": None,
        "rendition_sha256": "a" * 64,
    }


def test_summaries_replace_the_stored_items_and_are_counted():
    out = _serialize(_with_attachments(_PDF, _PNG))
    assert out["attachment_count"] == 2
    # Image first; with no copy state every item is a pending, non-viewable fallback.
    assert [s["filename"] for s in out["attachments"]] == ["plot.png", "r.pdf"]
    assert all(s["copy_status"] == "pending" and s["viewable"] is False for s in out["attachments"])
    assert all("attachment_id" not in s for s in out["attachments"])


def test_the_limit_cuts_the_summaries_but_not_the_count():
    out = _serialize(_with_attachments(_PNG, _PDF), attachment_limit=1)
    assert out["attachment_count"] == 2
    assert len(out["attachments"]) == 1


def test_a_zero_limit_omits_the_summaries_and_keeps_the_count():
    out = _serialize(_with_attachments(_PNG, _PDF), attachment_limit=0)
    assert out["attachment_count"] == 2
    assert "attachments" not in out


def test_an_entry_without_attachments_carries_neither_key():
    out = _serialize(_with_attachments())
    assert "attachment_count" not in out
    assert "attachments" not in out


def test_rows_join_the_summary():
    rows = [_copied_row("e1", _PNG)]
    [summary] = _serialize(_with_attachments(_PNG), attachment_rows=rows)["attachments"]
    assert summary["attachment_id"] == rows[0]["attachment_id"]
    assert summary["viewable"] is True
    assert summary["copy_status"] == "copied"


def test_a_migrated_store_without_a_row_marks_a_url_less_item_no_source():
    [summary] = _serialize(_with_attachments({"filename": "lost.png"}), attachment_rows=[])[
        "attachments"
    ]
    assert summary["copy_status"] == "skipped"
    assert summary["skip_reason"] == "no_source_url"


def test_an_unmigrated_store_gives_a_url_less_item_the_fallback():
    [summary] = _serialize(_with_attachments({"filename": "lost.png"}), attachment_rows=None)[
        "attachments"
    ]
    assert summary["copy_status"] == "pending"
    assert "skip_reason" not in summary


def test_match_evidence_comes_from_the_underscore_keys_like_score():
    rows = [_copied_row("e1", _PNG), _copied_row("e1", _PDF)]
    pdf_id = rows[1]["attachment_id"]
    entry = _with_attachments(
        _PNG,
        _PDF,
        _score=0.5,
        _matched_via=["image", "text"],
        _matched_attachment_ids=[pdf_id],
    )
    out = _serialize(entry, attachment_rows=rows)
    assert out["score"] == 0.5
    assert out["matched_via"] == ["image", "text"]
    assert out["matched_attachment_ids"] == [pdf_id]
    # A matched attachment sorts ahead of the image.
    assert out["attachments"][0]["attachment_id"] == pdf_id


def test_no_match_keys_without_the_underscore_keys():
    out = _serialize(_with_attachments(_PNG))
    assert "matched_via" not in out
    assert "matched_attachment_ids" not in out
    assert "score" not in out


def test_full_captions_is_passed_through():
    item = {**_PNG, "caption": "c" * 300}
    cut = _serialize(_with_attachments(item))["attachments"][0]
    assert cut["caption"] == "c" * 200
    assert cut["caption_truncated"] is True
    whole = _serialize(_with_attachments(item), full_captions=True)["attachments"][0]
    assert whole["caption"] == "c" * 300
    assert "caption_truncated" not in whole


def test_the_caption_model_id_selects_the_stored_caption():
    rows = [_copied_row("e1", _PNG)]
    captions = {rows[0]["attachment_id"]: {"vision-1": {"caption": "orbit", "visible_text": ""}}}
    entry = _with_attachments(_PNG, attachment_captions=captions)
    [summary] = _serialize(entry, attachment_rows=rows, model_id="vision-1")["attachments"]
    assert summary["caption"] == "orbit"
    assert summary["caption_source"] == "model:vision-1"


def test_the_attachment_arguments_are_required_keywords():
    params = inspect.signature(serialize_entry).parameters
    for name in ("attachment_limit", "attachment_rows", "model_id", "file_source"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert params[name].default is inspect.Parameter.empty
    assert params["full_captions"].default is False
