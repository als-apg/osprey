"""Tests for the semantic_search MCP tool."""

from functools import partial
from unittest.mock import AsyncMock

import pytest

from tests.mcp_server.ariel.conftest import (
    attach_fake_attachment_reader,
    get_tool_fn,
    make_mock_entry,
    make_search_result,
    patch_ariel_service,
    service_returning,
)
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict

_make_search_result = partial(make_search_result, mode="semantic")


def _get_semantic_search():
    from osprey.mcp_server.ariel.tools.semantic_search import semantic_search

    return get_tool_fn(semantic_search)


async def test_semantic_search_basic(ariel_registry):
    """Basic semantic search returns matching entries."""
    ariel_registry()

    entries = [make_mock_entry(entry_id="e1", raw_text="Beam loss event")]
    mock_result = _make_search_result(entries, reasoning="Semantic: 1 result")

    mock_service = AsyncMock()
    attach_fake_attachment_reader(mock_service)
    mock_service.search.return_value = mock_result

    with patch_ariel_service(mock_service):
        fn = _get_semantic_search()
        result = await fn(query="beam loss problems")

    data = extract_response_dict(result)
    assert not data.get("error", False)
    assert data["results_found"] == 1
    assert data["entries"][0]["entry_id"] == "e1"
    assert data["mode"] == "semantic"


async def test_semantic_search_similarity_threshold(ariel_registry):
    """similarity_threshold is passed through via advanced_params."""
    ariel_registry()

    mock_result = _make_search_result([])
    mock_service = AsyncMock()
    attach_fake_attachment_reader(mock_service)
    mock_service.search.return_value = mock_result

    with patch_ariel_service(mock_service):
        fn = _get_semantic_search()
        await fn(query="test", similarity_threshold=0.8)

    call_kwargs = mock_service.search.call_args.kwargs
    assert call_kwargs["advanced_params"]["similarity_threshold"] == 0.8


async def test_semantic_search_exclude_entry_ids(ariel_registry):
    """exclude_entry_ids filters out entries from results."""
    ariel_registry()

    entries = [
        make_mock_entry(entry_id="e1", raw_text="First entry"),
        make_mock_entry(entry_id="e2", raw_text="Second entry"),
    ]
    mock_result = _make_search_result(entries)

    mock_service = AsyncMock()
    attach_fake_attachment_reader(mock_service)
    mock_service.search.return_value = mock_result

    with patch_ariel_service(mock_service):
        fn = _get_semantic_search()
        result = await fn(query="entry", exclude_entry_ids=["e1"])

    data = extract_response_dict(result)
    assert data["results_found"] == 1
    assert data["entries"][0]["entry_id"] == "e2"


async def test_semantic_search_empty_query():
    """Empty query returns validation error."""
    fn = _get_semantic_search()
    with assert_raises_error(error_type="validation_error") as _exc_ctx:
        await fn(query="")

    _exc_ctx["envelope"]


async def test_semantic_search_degraded_returns_non_error(ariel_registry):
    """When semantic is unavailable, the tool returns a non-error result (#276).

    The service degrades gracefully (empty entries + reasoning steering to
    keyword search). The MCP tool must surface that as a normal response, not
    an error envelope, so agents don't hit a hard tool failure.
    """
    ariel_registry()

    mock_result = _make_search_result(
        [],
        reasoning="Semantic search is unavailable (embeddings not configured). "
        "Use keyword search instead.",
    )

    mock_service = AsyncMock()
    attach_fake_attachment_reader(mock_service)
    mock_service.search.return_value = mock_result

    with patch_ariel_service(mock_service):
        fn = _get_semantic_search()
        result = await fn(query="beam loss problems")

    data = extract_response_dict(result)
    assert not data.get("error", False)
    assert data["results_found"] == 0
    assert "keyword" in data["reasoning"].lower()


async def test_semantic_search_service_error(ariel_registry):
    """Service failure returns standard error format."""
    ariel_registry()

    mock_service = AsyncMock()
    attach_fake_attachment_reader(mock_service)
    mock_service.search.side_effect = RuntimeError("Embedding service down")

    with patch_ariel_service(mock_service):
        fn = _get_semantic_search()
        with assert_raises_error(error_type="internal_error") as _exc_ctx:
            await fn(query="test")

    data = _exc_ctx["envelope"]
    assert "Embedding service down" in data["error_message"]


async def test_vocabulary_error_names_config_key_and_remedy(ariel_registry):
    """A broken facility vocabulary is a fixable config problem, not an internal error."""
    from osprey.services.ariel_search.exceptions import VocabularyError

    ariel_registry()

    mock_service = AsyncMock()
    attach_fake_attachment_reader(mock_service)
    mock_service.search.side_effect = VocabularyError(
        ["vocabulary.yml: duplicate term 'ts'"],
    )

    with patch_ariel_service(mock_service):
        fn = _get_semantic_search()
        with assert_raises_error(error_type="service_unavailable") as _exc_ctx:
            await fn(query="test")

    data = _exc_ctx["envelope"]
    assert "ariel.vocabulary.path" in data["error_message"]
    assert "vocabulary.yml: duplicate term 'ts'" in data["error_message"]
    assert any(
        "disable ariel.vocabulary.enabled or repoint ariel.vocabulary.path" in s
        for s in data["suggestions"]
    )


# --- vocabulary expansion ---------------------------------------------------

TS_GROUP = {"original": "ts", "alternatives": ["troubleshoot", "timing system"]}


@pytest.mark.parametrize("value", [True, False])
async def test_expand_query_is_forwarded_in_advanced_params(value, ariel_registry):
    """An explicit expand_query reaches the service as an advanced parameter."""
    ariel_registry()

    mock_service = service_returning(_make_search_result([]))

    with patch_ariel_service(mock_service):
        fn = _get_semantic_search()
        await fn(query="ts on the bpm", expand_query=value)

    assert mock_service.search.call_args.kwargs["advanced_params"]["expand_query"] is value


async def test_expand_query_omitted_when_unset(ariel_registry):
    """Silence leaves the deployment's ``expand_by_default`` in charge."""
    ariel_registry()

    mock_service = service_returning(_make_search_result([]))

    with patch_ariel_service(mock_service):
        fn = _get_semantic_search()
        await fn(query="ts on the bpm")

    assert "expand_query" not in mock_service.search.call_args.kwargs["advanced_params"]


async def test_envelope_reports_the_applied_expansion(ariel_registry):
    """Semantic mode expands over the whole query and says what it embedded."""
    ariel_registry()

    mock_result = _make_search_result(
        [make_mock_entry(entry_id="e1", raw_text="troubleshooting the timing system")],
        expanded_terms=[TS_GROUP],
    )
    mock_service = service_returning(mock_result)

    with patch_ariel_service(mock_service):
        fn = _get_semantic_search()
        result = await fn(query="ts fault", expand_query=True)

    data = extract_response_dict(result)
    assert data["expanded_terms"] == [TS_GROUP]
    assert data["expanded_terms"][0]["alternatives"] == ["troubleshoot", "timing system"]


async def test_envelope_reports_diagnostics(ariel_registry):
    """A search that expanded nothing says why, in the same envelope."""
    from osprey.services.ariel_search.models import DiagnosticLevel, SearchDiagnostic

    ariel_registry()

    mock_result = _make_search_result(
        [],
        diagnostics=[
            SearchDiagnostic(
                level=DiagnosticLevel.INFO,
                source="service.semantic",
                message="expansion skipped: semantic is not in ariel.vocabulary.expand_modes",
                category="expansion",
            )
        ],
    )
    mock_service = service_returning(mock_result)

    with patch_ariel_service(mock_service):
        fn = _get_semantic_search()
        result = await fn(query="ts fault")

    data = extract_response_dict(result)
    assert data["expanded_terms"] == []
    assert data["diagnostics"][0]["level"] == "info"
    assert data["diagnostics"][0]["category"] == "expansion"
    assert "expand_modes" in data["diagnostics"][0]["message"]


async def test_timeout_diagnostic_becomes_an_error_envelope(ariel_registry):
    """A cancelled statement is an error, not an empty result set."""
    from osprey.services.ariel_search.models import DiagnosticLevel, SearchDiagnostic

    ariel_registry()

    mock_result = _make_search_result(
        [],
        diagnostics=[
            SearchDiagnostic(
                level=DiagnosticLevel.ERROR,
                source="service.timeout",
                message="Search timed out: vector scan exceeded 10.0s limit",
                category="timeout",
            )
        ],
        expanded_terms=[TS_GROUP],
    )
    mock_service = service_returning(mock_result)

    with patch_ariel_service(mock_service):
        fn = _get_semantic_search()
        with assert_raises_error(error_type="search_timeout") as _exc_ctx:
            await fn(query="ts fault")

    data = _exc_ctx["envelope"]
    assert "exceeded 10.0s" in data["error_message"]
    assert data["details"]["expanded_terms"] == [TS_GROUP]
    assert data["details"]["diagnostics"][0]["source"] == "service.timeout"


async def test_degraded_result_is_still_not_an_error(ariel_registry):
    """Only ERROR diagnostics of a statement-fault category raise (#276).

    The graceful-degradation path returns an empty result with a WARNING, and
    that must keep reading as a normal answer.
    """
    from osprey.services.ariel_search.models import DiagnosticLevel, SearchDiagnostic

    ariel_registry()

    mock_result = _make_search_result(
        [],
        reasoning="Semantic search is unavailable. Use keyword search instead.",
        diagnostics=[
            SearchDiagnostic(
                level=DiagnosticLevel.WARNING,
                source="service.semantic",
                message="embeddings are not configured",
                category="degraded",
            )
        ],
    )
    mock_service = service_returning(mock_result)

    with patch_ariel_service(mock_service):
        fn = _get_semantic_search()
        result = await fn(query="ts fault")

    data = extract_response_dict(result)
    assert not data.get("error", False)
    assert data["diagnostics"][0]["level"] == "warning"
