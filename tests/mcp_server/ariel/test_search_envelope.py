"""Unit tests for the shared ARIEL search-tool envelope helpers."""

from unittest.mock import AsyncMock, patch

import pytest

from osprey.mcp_server.ariel.tools.search_envelope import (
    ResultWindow,
    advanced_params,
    serialize_entries,
)
from osprey.services.ariel_search.models import DEFAULT_LISTING_TEXT_CHARS
from tests.mcp_server.ariel.conftest import make_mock_entry

_PNG = {"url": "https://elog.example/f/plot.png", "type": "image/png", "filename": "plot.png"}
_PDF = {"url": "https://elog.example/f/report.pdf", "type": "application/pdf", "filename": "r.pdf"}


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


def test_select_returns_the_kept_raw_entries_without_reading_the_context():
    entries = [make_mock_entry(entry_id=f"e{i}", raw_text="x" * 900) for i in range(4)]

    with patch(
        "osprey.mcp_server.ariel.server_context.get_ariel_context",
        side_effect=AssertionError("select must not read the context"),
    ):
        kept = ResultWindow.build(2, ["e0"]).select(entries)

    assert kept == [entries[1], entries[2]]
    assert kept[0] is entries[1]


def _repository(rows=None, *, error=None):
    repository = AsyncMock()
    if error is not None:
        repository.get_attachment_rows = AsyncMock(side_effect=error)
    else:
        repository.get_attachment_rows = AsyncMock(return_value=rows)
    return repository


async def _serialize(entries, repository, **overrides):
    kwargs = {
        "text_limit": DEFAULT_LISTING_TEXT_CHARS,
        "attachment_limit": 5,
        "repository": repository,
        "model_id": None,
        "file_source": False,
    }
    kwargs.update(overrides)
    return await serialize_entries(entries, **kwargs)


async def test_serialize_entries_cuts_at_the_given_text_budget():
    [out] = await _serialize([make_mock_entry(raw_text="x" * 25)], _repository({}), text_limit=10)

    assert out["raw_text"] == "x" * 10
    assert out["raw_text_truncated"] is True
    assert out["raw_text_length"] == 25


async def test_serialize_entries_keeps_a_short_entry_whole_at_the_default_budget():
    text = "x" * DEFAULT_LISTING_TEXT_CHARS

    [out] = await _serialize([make_mock_entry(raw_text=text)], _repository({}))

    assert out["raw_text"] == text
    assert "raw_text_truncated" not in out
    assert "raw_text_length" not in out


async def test_a_page_of_ten_reads_the_attachment_rows_once():
    entries = [make_mock_entry(entry_id=f"e{i}", attachments=[_PNG]) for i in range(10)]
    repository = _repository({})

    out = await _serialize(entries, repository)

    repository.get_attachment_rows.assert_awaited_once_with([f"e{i}" for i in range(10)])
    assert [e["entry_id"] for e in out] == [f"e{i}" for i in range(10)]


async def test_an_empty_page_reads_no_attachment_rows():
    repository = _repository({})

    assert await _serialize([], repository) == []

    repository.get_attachment_rows.assert_not_awaited()


async def test_each_entry_gets_its_own_rows_and_an_absent_entry_gets_none():
    from osprey.services.ariel_search.attachments import attachment_id_for

    attachment_id = attachment_id_for("e1", _PNG)
    row = {
        "attachment_id": attachment_id,
        "entry_id": "e1",
        "filename": "plot.png",
        "mime_type": "image/png",
        "copy_status": "copied",
        "skip_reason": None,
        "rendition_sha256": "a" * 64,
    }
    entries = [
        make_mock_entry(entry_id="e1", attachments=[_PNG]),
        make_mock_entry(entry_id="e2", attachments=[{"filename": "lost.png"}]),
    ]

    first, second = await _serialize(entries, _repository({"e1": [row]}))

    assert first["attachments"][0]["attachment_id"] == attachment_id
    assert first["attachments"][0]["viewable"] is True
    # A migrated store with no row for a url-less item: never fetchable, not pending.
    assert second["attachments"][0]["copy_status"] == "skipped"
    assert second["attachments"][0]["skip_reason"] == "no_source_url"


async def test_an_unmigrated_store_gives_every_entry_the_fallback():
    entries = [make_mock_entry(entry_id="e1", attachments=[_PNG, {"filename": "lost.png"}])]

    [out] = await _serialize(entries, _repository(None))

    assert [s["copy_status"] for s in out["attachments"]] == ["pending", "pending"]
    assert all("attachment_id" not in s for s in out["attachments"])


async def test_a_failing_reader_is_treated_as_no_copy_state():
    from osprey.services.ariel_search.exceptions import DatabaseQueryError

    entries = [make_mock_entry(entry_id="e1", attachments=[_PNG])]
    repository = _repository(error=DatabaseQueryError("boom", query="SELECT attachment_files"))
    unmigrated = await _serialize(entries, _repository(None))

    with patch(
        "osprey.services.ariel_search.database.repository.warn_attachment_schema_gap_once"
    ) as warn:
        out = await _serialize(entries, repository)

    assert out == unmigrated
    warn.assert_called_once_with()


async def test_a_reader_answering_neither_none_nor_a_dict_is_refused():
    repository = AsyncMock()  # bare: the reader answers a MagicMock

    with pytest.raises(TypeError, match="get_attachment_rows"):
        await _serialize([make_mock_entry(entry_id="e1")], repository)


async def test_a_zero_limit_omits_the_summaries_and_keeps_the_count():
    entries = [make_mock_entry(entry_id="e1", attachments=[_PNG, _PDF])]

    [out] = await _serialize(entries, _repository({}), attachment_limit=0)

    assert out["attachment_count"] == 2
    assert "attachments" not in out


async def test_the_model_id_and_file_source_reach_the_summaries():
    item = {"url": "files/plot.png", "type": "image/png", "filename": "plot.png"}
    entries = [make_mock_entry(entry_id="e1", attachments=[item])]

    [remote] = await _serialize(entries, _repository({}), file_source=False)
    [local] = await _serialize(entries, _repository({}), file_source=True)

    # A relative path is fetchable only from a file source.
    assert remote["attachments"][0]["skip_reason"] == "no_source_url"
    assert local["attachments"][0]["copy_status"] == "pending"


# ---------------------------------------------------------------------------
# Image-only entries in the window
# ---------------------------------------------------------------------------


def _fused(entry_id, via):
    entry = make_mock_entry(entry_id=entry_id)
    entry["_matched_via"] = via
    return entry


def _fused_page(text_ids, image_ids):
    """Text hits ranked first, then image-only hits, as a fused result can be."""
    return [_fused(e, ["text"]) for e in text_ids] + [_fused(e, ["image"]) for e in image_ids]


def test_select_keeps_max_results_when_image_only_entries_overflow():
    """10 text hits and 5 admissible image-only hits at max_results=10 give exactly 10."""
    entries = _fused_page([f"T{i}" for i in range(1, 11)], [f"I{i}" for i in range(1, 6)])

    kept = ResultWindow.build(10, None).select(entries)

    assert [e["entry_id"] for e in kept] == [f"T{i}" for i in range(1, 11)]


def test_select_caps_image_only_entries_at_a_third_of_the_page():
    """Image-only entries beyond ceil(max_results / 3) never reach the page."""
    entries = [_fused(f"I{i}", ["image"]) for i in range(1, 7)] + [
        _fused(f"T{i}", ["text"]) for i in range(1, 11)
    ]

    kept = ResultWindow.build(10, None).select(entries)

    assert len(kept) == 10
    assert [e["entry_id"] for e in kept if e["_matched_via"] == ["image"]] == [
        "I1",
        "I2",
        "I3",
        "I4",
    ]
    assert [e["entry_id"] for e in kept if e["_matched_via"] == ["text"]] == [
        f"T{i}" for i in range(1, 7)
    ]


def test_select_does_not_count_entries_matched_by_both_lanes_as_image_only():
    entries = [_fused(f"B{i}", ["image", "text"]) for i in range(6)]

    assert len(ResultWindow.build(10, None).select(entries)) == 6


def test_select_refills_an_exclusion_from_text_hits():
    """Excluded entries are refilled from the text hits fetched for the window."""
    entries = _fused_page([f"T{i}" for i in range(1, 16)], [f"I{i}" for i in range(1, 6)])
    window = ResultWindow.build(10, [f"T{i}" for i in range(1, 6)])

    kept = window.select(entries)

    assert [e["entry_id"] for e in kept] == [f"T{i}" for i in range(6, 16)]


def test_select_with_ties_holds_exactly_ten_and_reaches_t11():
    """Excludes T1..T5, 15 text hits, 5 image-only hits each tying its text hit.

    Fusion ranks a text-lane entry before an image-only entry it ties, so the
    image-only hits interleave behind their text partners; the page still
    holds exactly ten entries and T11 is on it.
    """
    entries = []
    for i in range(1, 16):
        entries.append(_fused(f"T{i}", ["text"]))
        if i <= 5:
            entries.append(_fused(f"I{i}", ["image"]))
    window = ResultWindow.build(10, [f"T{i}" for i in range(1, 6)])

    kept = window.select(entries)
    ids = [e["entry_id"] for e in kept]

    assert len(kept) == 10
    assert "T11" in ids
    assert not {f"T{i}" for i in range(1, 6)} & set(ids)
    assert sum(1 for e in kept if e["_matched_via"] == ["image"]) == 4


def test_select_without_matched_via_is_the_plain_window():
    entries = [make_mock_entry(entry_id=f"e{i}") for i in range(12)]

    kept = ResultWindow.build(10, ["e0"]).select(entries)

    assert [e["entry_id"] for e in kept] == [f"e{i}" for i in range(1, 11)]


# ---------------------------------------------------------------------------
# The MCP hybrid tool's sources with a fused result
# ---------------------------------------------------------------------------


def _search_result(entries, sources):
    from unittest.mock import MagicMock

    result = MagicMock()
    result.entries = tuple(entries)
    result.answer = None
    result.reasoning = f"Hybrid search: {len(entries)} results"
    result.sources = tuple(sources)
    result.search_modes_used = ("hybrid",)
    result.diagnostics = ()
    result.expanded_terms = ()
    result.pipeline_details = None
    return result


async def _run_hybrid_tool(tmp_path, monkeypatch, result, **kwargs):
    from osprey.mcp_server.ariel.server_context import initialize_ariel_context
    from osprey.mcp_server.ariel.tools.hybrid_search import hybrid_search
    from tests.mcp_server.ariel.conftest import attach_fake_attachment_reader, get_tool_fn
    from tests.mcp_server.conftest import extract_response_dict

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.yml").write_text(
        '{"ariel": {"database": {"uri": "postgresql://localhost/test"}}}'
    )
    initialize_ariel_context()
    service = AsyncMock()
    attach_fake_attachment_reader(service)
    service.search.return_value = result
    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=service),
    ):
        response = await get_tool_fn(hybrid_search)(**kwargs)
    return extract_response_dict(response)


async def test_hybrid_tool_sources_name_the_windowed_entries(tmp_path, monkeypatch):
    """A lane admitting 4 image-only hits at max_results=10: sources match the page."""
    entries = _fused_page([f"T{i}" for i in range(1, 11)], [f"I{i}" for i in range(1, 5)])
    result = _search_result(entries, [e["entry_id"] for e in entries])

    data = await _run_hybrid_tool(tmp_path, monkeypatch, result, query="beam", max_results=10)

    assert len(data["sources"]) == data["results_found"] == 10
    assert data["sources"] == [e["entry_id"] for e in data["entries"]]


async def test_hybrid_tool_keeps_the_service_sources_without_matched_via(tmp_path, monkeypatch):
    """With the lane off the envelope's sources are the service's, as before."""
    entries = [make_mock_entry(entry_id=f"e{i}") for i in range(3)]
    result = _search_result(entries, ["e0", "e1", "e2", "extra"])

    data = await _run_hybrid_tool(tmp_path, monkeypatch, result, query="beam", max_results=10)

    assert data["sources"] == ["e0", "e1", "e2", "extra"]
