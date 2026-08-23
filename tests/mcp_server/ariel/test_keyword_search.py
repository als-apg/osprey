"""Tests for the keyword_search MCP tool."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.mcp_server.ariel.server_context import initialize_ariel_context
from tests.mcp_server.ariel.conftest import get_tool_fn, make_mock_entry
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict


def _make_search_result(entries, reasoning="", sources=(), diagnostics=(), expanded_terms=()):
    """Build a mock ARIELSearchResult.

    ``diagnostics`` and ``expanded_terms`` are set explicitly on every result:
    a MagicMock auto-attribute is neither iterable nor JSON-serializable, so
    leaving them unset would fail the envelope build rather than the assertion
    under test.
    """
    result = MagicMock()
    result.entries = tuple(entries)
    result.answer = None
    result.reasoning = reasoning
    result.sources = tuple(sources)
    result.search_modes_used = ("keyword",)
    result.diagnostics = tuple(diagnostics)
    result.expanded_terms = tuple(expanded_terms)
    result.pipeline_details = None
    return result


def _get_keyword_search():
    from osprey.mcp_server.ariel.tools.keyword_search import keyword_search

    return get_tool_fn(keyword_search)


def _setup_registry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = '{"ariel": {"database": {"uri": "postgresql://localhost/test"}}}'
    (tmp_path / "config.yml").write_text(config)
    initialize_ariel_context()


@pytest.mark.unit
async def test_keyword_search_basic(tmp_path, monkeypatch):
    """Basic keyword search returns matching entries."""
    _setup_registry(tmp_path, monkeypatch)

    entries = [make_mock_entry(entry_id="e1", raw_text="Beam loss event")]
    mock_result = _make_search_result(entries, reasoning="Keyword: 1 result")

    mock_service = AsyncMock()
    mock_service.search.return_value = mock_result

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_keyword_search()
        result = await fn(query="beam loss")

    data = json.loads(result)
    assert not data.get("error", False)
    assert data["results_found"] == 1
    assert data["entries"][0]["entry_id"] == "e1"
    assert data["mode"] == "keyword"


@pytest.mark.unit
async def test_keyword_search_date_filtering(tmp_path, monkeypatch):
    """Date strings are parsed and passed to service.search()."""
    _setup_registry(tmp_path, monkeypatch)

    mock_result = _make_search_result([])
    mock_service = AsyncMock()
    mock_service.search.return_value = mock_result

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_keyword_search()
        await fn(
            query="test",
            start_date="2024-01-01",
            end_date="2024-01-31",
        )

    call_kwargs = mock_service.search.call_args.kwargs
    from datetime import datetime
    from zoneinfo import ZoneInfo

    start, end = call_kwargs["time_range"]
    assert start == datetime(2024, 1, 1, tzinfo=ZoneInfo("UTC"))
    assert end == datetime(2024, 1, 31, tzinfo=ZoneInfo("UTC"))


@pytest.mark.unit
async def test_keyword_search_author_filtering(tmp_path, monkeypatch):
    """Author filter is passed via advanced_params."""
    _setup_registry(tmp_path, monkeypatch)

    mock_result = _make_search_result([])
    mock_service = AsyncMock()
    mock_service.search.return_value = mock_result

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_keyword_search()
        await fn(query="test", author="Jane")

    call_kwargs = mock_service.search.call_args.kwargs
    assert call_kwargs["advanced_params"]["author"] == "Jane"


@pytest.mark.unit
async def test_keyword_search_exclude_entry_ids(tmp_path, monkeypatch):
    """exclude_entry_ids filters out entries from results."""
    _setup_registry(tmp_path, monkeypatch)

    entries = [
        make_mock_entry(entry_id="e1", raw_text="First entry"),
        make_mock_entry(entry_id="e2", raw_text="Second entry"),
        make_mock_entry(entry_id="e3", raw_text="Third entry"),
    ]
    mock_result = _make_search_result(entries)

    mock_service = AsyncMock()
    mock_service.search.return_value = mock_result

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_keyword_search()
        result = await fn(query="entry", exclude_entry_ids=["e1", "e3"])

    data = extract_response_dict(result)
    assert data["results_found"] == 1
    assert data["entries"][0]["entry_id"] == "e2"

    # Verify over-fetch: requested max_results + len(exclude_ids)
    call_kwargs = mock_service.search.call_args.kwargs
    assert call_kwargs["max_results"] == 12  # 10 default + 2 excluded


@pytest.mark.unit
async def test_keyword_search_empty_query():
    """Empty query returns validation error."""
    fn = _get_keyword_search()
    with assert_raises_error(error_type="validation_error") as _exc_ctx:
        await fn(query="")

    _exc_ctx["envelope"]


@pytest.mark.unit
async def test_keyword_search_service_error(tmp_path, monkeypatch):
    """Service failure returns standard error format."""
    _setup_registry(tmp_path, monkeypatch)

    mock_service = AsyncMock()
    mock_service.search.side_effect = RuntimeError("DB connection failed")

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_keyword_search()
        with assert_raises_error(error_type="internal_error") as _exc_ctx:
            await fn(query="test")

    data = _exc_ctx["envelope"]
    assert "DB connection failed" in data["error_message"]


@pytest.mark.unit
async def test_vocabulary_error_names_config_key_and_remedy(tmp_path, monkeypatch):
    """A broken facility vocabulary is a fixable config problem, not an internal error."""
    from osprey.services.ariel_search.exceptions import VocabularyError

    _setup_registry(tmp_path, monkeypatch)

    mock_service = AsyncMock()
    mock_service.search.side_effect = VocabularyError(
        ["vocabulary.yml: duplicate term 'ts'"],
    )

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_keyword_search()
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
TSLASH_GROUP = {"original": "t/s", "alternatives": ["troubleshoot"]}


def _service_returning(mock_result):
    """An ARIEL service mock whose search returns *mock_result*."""
    mock_service = AsyncMock()
    mock_service.search.return_value = mock_result
    return mock_service


def _keyword_descriptor(execute, query_parser):
    """Build a keyword descriptor around *execute* for the real service."""
    from pydantic import BaseModel

    from osprey.services.ariel_search.search.base import SearchToolDescriptor

    class _Input(BaseModel):
        query: str

    return SearchToolDescriptor(
        name="keyword_search",
        description="fake keyword",
        search_mode="keyword",
        args_schema=_Input,
        execute=execute,
        format_result=lambda *a, **k: {},
        accepts_expansion=True,
        query_parser=query_parser,
    )


@pytest.mark.unit
@pytest.mark.parametrize("value", [True, False])
async def test_expand_query_is_forwarded_in_advanced_params(tmp_path, monkeypatch, value):
    """An explicit expand_query reaches the service as an advanced parameter."""
    _setup_registry(tmp_path, monkeypatch)

    mock_service = _service_returning(_make_search_result([]))

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_keyword_search()
        await fn(query="ts bpm", expand_query=value)

    assert mock_service.search.call_args.kwargs["advanced_params"]["expand_query"] is value


@pytest.mark.unit
async def test_expand_query_omitted_when_unset(tmp_path, monkeypatch):
    """Leaving the argument alone must not override the configured default.

    Sending ``expand_query`` on every call would make the tool's silence look
    like an explicit choice, and the deployment's ``expand_by_default`` would
    never apply.
    """
    _setup_registry(tmp_path, monkeypatch)

    mock_service = _service_returning(_make_search_result([]))

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_keyword_search()
        await fn(query="ts bpm")

    assert "expand_query" not in mock_service.search.call_args.kwargs["advanced_params"]


@pytest.mark.unit
async def test_envelope_reports_the_applied_expansion(tmp_path, monkeypatch):
    """An ambiguous form is one group with both canonicals, not two groups."""
    _setup_registry(tmp_path, monkeypatch)

    mock_result = _make_search_result(
        [make_mock_entry(entry_id="e1", raw_text="troubleshoot the timing system")],
        expanded_terms=[TS_GROUP],
    )
    mock_service = _service_returning(mock_result)

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_keyword_search()
        result = await fn(query="ts fault", expand_query=True)

    data = json.loads(result)
    assert data["expanded_terms"] == [TS_GROUP]
    assert data["expanded_terms"][0]["alternatives"] == ["troubleshoot", "timing system"]


@pytest.mark.unit
async def test_envelope_reports_diagnostics(tmp_path, monkeypatch):
    """Why expansion did nothing is visible without a second call."""
    from osprey.services.ariel_search.models import DiagnosticLevel, SearchDiagnostic

    _setup_registry(tmp_path, monkeypatch)

    mock_result = _make_search_result(
        [],
        diagnostics=[
            SearchDiagnostic(
                level=DiagnosticLevel.INFO,
                source="service.keyword",
                message="expansion skipped: query carries a boolean operator",
                category="expansion",
            )
        ],
    )
    mock_service = _service_returning(mock_result)

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_keyword_search()
        result = await fn(query="ts AND bpm")

    data = json.loads(result)
    assert data["expanded_terms"] == []
    assert data["diagnostics"] == [
        {
            "level": "info",
            "source": "service.keyword",
            "message": "expansion skipped: query carries a boolean operator",
            "category": "expansion",
        }
    ]


@pytest.mark.unit
async def test_timeout_diagnostic_becomes_an_error_envelope(tmp_path, monkeypatch):
    """A cancelled statement must not read as "nothing matched"."""
    from osprey.services.ariel_search.models import DiagnosticLevel, SearchDiagnostic

    _setup_registry(tmp_path, monkeypatch)

    mock_result = _make_search_result(
        [],
        diagnostics=[
            SearchDiagnostic(
                level=DiagnosticLevel.ERROR,
                source="service.timeout",
                message="Search timed out: pattern match exceeded 10.0s limit",
                category="timeout",
            )
        ],
        expanded_terms=[TS_GROUP],
    )
    mock_service = _service_returning(mock_result)

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_keyword_search()
        with assert_raises_error(error_type="search_timeout") as _exc_ctx:
            await fn(query="ts SR01C*")

    data = _exc_ctx["envelope"]
    assert "exceeded 10.0s" in data["error_message"]
    assert data["details"]["expanded_terms"] == [TS_GROUP]
    assert data["details"]["diagnostics"][0]["category"] == "timeout"


@pytest.mark.unit
async def test_invalid_pattern_diagnostic_names_the_pattern(tmp_path, monkeypatch):
    """The envelope names the rejected pattern and keeps the expansion.

    One unusable token costs the caller that token, not the meaning of the rest
    of the query -- so the expansion the failed statement carried travels with
    the error.
    """
    from osprey.services.ariel_search.models import DiagnosticLevel, SearchDiagnostic

    _setup_registry(tmp_path, monkeypatch)

    mock_result = _make_search_result(
        [],
        diagnostics=[
            SearchDiagnostic(
                level=DiagnosticLevel.ERROR,
                source="service.keyword",
                message="invalid pattern 'SR0[1-4': brackets [] not balanced",
                category="pattern",
            )
        ],
        expanded_terms=[TSLASH_GROUP],
    )
    mock_service = _service_returning(mock_result)

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_keyword_search()
        with assert_raises_error(error_type="invalid_pattern") as _exc_ctx:
            await fn(query="t/s /SR0[1-4/")

    data = _exc_ctx["envelope"]
    assert "SR0[1-4" in data["error_message"]
    assert data["details"]["expanded_terms"] == [TSLASH_GROUP]
    assert any("/regex/" in suggestion for suggestion in data["suggestions"])


@pytest.mark.unit
async def test_pattern_error_from_the_real_service_reaches_the_envelope(tmp_path, monkeypatch):
    """End-to-end over the real service: exception in, error envelope out.

    Exercises ``ARIELSearchService.ainvoke``'s ``PatternError`` handler rather
    than a hand-built diagnostic, so the tool and the service cannot drift on
    the diagnostic's category, source or message.
    """
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.exceptions import PatternError
    from osprey.services.ariel_search.search.keyword import parse_keyword_query
    from osprey.services.ariel_search.service import ARIELSearchService

    _setup_registry(tmp_path, monkeypatch)

    vocabulary_path = tmp_path / "vocabulary.yml"
    vocabulary_path.write_text(
        "concepts:\n  - canonical: troubleshoot\n    kind: shorthand\n    forms:\n      - t/s\n"
    )
    config = ARIELConfig.from_dict(
        {
            "database": {"uri": "postgresql://localhost:5432/test"},
            "search_modules": {"keyword": {"enabled": True}},
            "vocabulary": {"enabled": True, "path": str(vocabulary_path)},
        }
    )
    assert not config.vocabulary_errors

    pool = MagicMock()
    pool.close = AsyncMock()
    repository = MagicMock()
    repository.health_check = AsyncMock(return_value=(True, "OK"))
    repository.validate_search_model_table = AsyncMock()
    service = ARIELSearchService(config=config, pool=pool, repository=repository)

    async def _raise(*_args, **_kwargs):
        raise PatternError("brackets [] not balanced", pattern="SR0[1-4")

    module = MagicMock()
    module.get_tool_descriptor.return_value = _keyword_descriptor(_raise, parse_keyword_query)
    registry = MagicMock()
    registry.list_ariel_search_modules.return_value = ["keyword"]
    registry.get_ariel_search_module.return_value = module

    with (
        patch("osprey.registry.get_registry", return_value=registry),
        patch(
            "osprey.mcp_server.ariel.server_context.ARIELContext.service",
            new=AsyncMock(return_value=service),
        ),
    ):
        fn = _get_keyword_search()
        with assert_raises_error(error_type="invalid_pattern") as _exc_ctx:
            await fn(query="t/s /SR0[1-4/")

    data = _exc_ctx["envelope"]
    assert "SR0[1-4" in data["error_message"]
    assert data["details"]["expanded_terms"] == [TSLASH_GROUP]
