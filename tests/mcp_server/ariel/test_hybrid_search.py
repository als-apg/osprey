"""Tests for the hybrid_search MCP tool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from osprey.mcp_server.ariel.server_context import initialize_ariel_context
from osprey.services.ariel_search.models import DiagnosticLevel, SearchDiagnostic
from tests.mcp_server.ariel.conftest import get_tool_fn, make_mock_entry
from tests.mcp_server.conftest import assert_raises_error, extract_response_dict


def _make_search_result(entries, reasoning="", sources=(), diagnostics=()):
    """Build a mock ARIELSearchResult."""
    result = MagicMock()
    result.entries = tuple(entries)
    result.answer = None
    result.reasoning = reasoning
    result.sources = tuple(sources)
    result.search_modes_used = ("hybrid",)
    result.diagnostics = tuple(diagnostics)
    result.pipeline_details = None
    return result


def _qmd_error(message):
    """Build the diagnostic the service returns when the qmd module raised."""
    return SearchDiagnostic(
        level=DiagnosticLevel.ERROR,
        source="service.hybrid",
        message=message,
        category="search",
    )


def _get_hybrid_search():
    from osprey.mcp_server.ariel.tools.hybrid_search import hybrid_search

    return get_tool_fn(hybrid_search)


def _setup_registry(tmp_path, monkeypatch, config=None):
    monkeypatch.chdir(tmp_path)
    default = '{"ariel": {"database": {"uri": "postgresql://localhost/test"}}}'
    (tmp_path / "config.yml").write_text(config if config is not None else default)
    initialize_ariel_context()


@pytest.mark.unit
async def test_hybrid_search_basic(tmp_path, monkeypatch):
    """Basic hybrid search returns matching entries."""
    _setup_registry(tmp_path, monkeypatch)

    entries = [make_mock_entry(entry_id="e1", raw_text="Beam loss event", score=0.9)]
    mock_result = _make_search_result(entries, reasoning="Hybrid search: 1 results")

    mock_service = AsyncMock()
    mock_service.search.return_value = mock_result

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_hybrid_search()
        result = await fn(query="beam loss problems")

    data = extract_response_dict(result)
    assert not data.get("error", False)
    assert data["results_found"] == 1
    assert data["entries"][0]["entry_id"] == "e1"
    assert data["entries"][0]["score"] == 0.9
    assert data["mode"] == "hybrid"


@pytest.mark.unit
async def test_hybrid_search_dispatches_to_the_hybrid_mode(tmp_path, monkeypatch):
    """The tool must route to its own search module, not a neighbour's."""
    _setup_registry(tmp_path, monkeypatch)

    mock_service = AsyncMock()
    mock_service.search.return_value = _make_search_result([])

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_hybrid_search()
        await fn(query="beam loss")

    assert mock_service.search.call_args.kwargs["mode"] == "hybrid"


@pytest.mark.unit
async def test_hybrid_search_forwards_author_and_source_filters(tmp_path, monkeypatch):
    """author and source_system travel as advanced params."""
    _setup_registry(tmp_path, monkeypatch)

    mock_service = AsyncMock()
    mock_service.search.return_value = _make_search_result([])

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_hybrid_search()
        await fn(query="test", author="chen", source_system="ALS eLog")

    adv = mock_service.search.call_args.kwargs["advanced_params"]
    assert adv["author"] == "chen"
    assert adv["source_system"] == "ALS eLog"


@pytest.mark.unit
async def test_hybrid_search_forwards_date_range(tmp_path, monkeypatch):
    """start_date and end_date become the request's time range."""
    _setup_registry(tmp_path, monkeypatch)

    mock_service = AsyncMock()
    mock_service.search.return_value = _make_search_result([])

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_hybrid_search()
        await fn(query="test", start_date="2024-01-15", end_date="2024-02-15")

    start, end = mock_service.search.call_args.kwargs["time_range"]
    assert start.year == 2024 and start.month == 1 and start.day == 15
    assert end.month == 2 and end.day == 15


@pytest.mark.unit
async def test_hybrid_search_exclude_entry_ids(tmp_path, monkeypatch):
    """exclude_entry_ids filters out entries and over-fetches to compensate."""
    _setup_registry(tmp_path, monkeypatch)

    entries = [
        make_mock_entry(entry_id="e1", raw_text="First entry"),
        make_mock_entry(entry_id="e2", raw_text="Second entry"),
    ]
    mock_service = AsyncMock()
    mock_service.search.return_value = _make_search_result(entries)

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_hybrid_search()
        result = await fn(query="entry", max_results=2, exclude_entry_ids=["e1"])

    data = extract_response_dict(result)
    assert data["results_found"] == 1
    assert data["entries"][0]["entry_id"] == "e2"
    assert mock_service.search.call_args.kwargs["max_results"] == 3


@pytest.mark.unit
async def test_hybrid_search_caps_at_max_results(tmp_path, monkeypatch):
    """The response never exceeds what the caller asked for."""
    _setup_registry(tmp_path, monkeypatch)

    entries = [make_mock_entry(entry_id=f"e{i}") for i in range(5)]
    mock_service = AsyncMock()
    mock_service.search.return_value = _make_search_result(entries)

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_hybrid_search()
        result = await fn(query="entry", max_results=2)

    assert extract_response_dict(result)["results_found"] == 2


@pytest.mark.unit
async def test_hybrid_search_empty_query():
    """Empty query returns validation error."""
    fn = _get_hybrid_search()
    with assert_raises_error(error_type="validation_error") as _exc_ctx:
        await fn(query="")

    envelope = _exc_ctx["envelope"]
    assert "Empty search query" in envelope["error_message"]
    assert "description or keywords" in envelope["suggestions"][0]


@pytest.mark.unit
async def test_hybrid_search_no_results_is_not_an_error(tmp_path, monkeypatch):
    """A query that matched nothing is a normal, empty response."""
    _setup_registry(tmp_path, monkeypatch)

    mock_service = AsyncMock()
    mock_service.search.return_value = _make_search_result([], reasoning="Hybrid search: 0 results")

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_hybrid_search()
        result = await fn(query="nothing matches this")

    data = extract_response_dict(result)
    assert not data.get("error", False)
    assert data["results_found"] == 0


# ---------------------------------------------------------------------------
# Sidecar faults
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_sidecar_down_is_an_error_not_an_empty_result(tmp_path, monkeypatch):
    """A down sidecar must not read as "the corpus holds no answer".

    The service swallows the module's exception and returns an empty result
    carrying one ERROR diagnostic, which is byte-identical to a zero-hit query
    apart from that diagnostic. The tool has to notice.
    """
    _setup_registry(tmp_path, monkeypatch)

    mock_service = AsyncMock()
    mock_service.search.return_value = _make_search_result(
        [],
        reasoning="Hybrid search failed: the qmd sidecar at http://127.0.0.1:8180 is not answering",
        diagnostics=[
            _qmd_error(
                "Hybrid search failed: the qmd sidecar at http://127.0.0.1:8180 is not answering"
            )
        ],
    )

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_hybrid_search()
        with assert_raises_error(error_type="service_unavailable") as _exc_ctx:
            await fn(query="beam loss")

    data = _exc_ctx["envelope"]
    assert "not answering" in data["error_message"]


@pytest.mark.unit
async def test_sidecar_down_names_the_health_endpoint(tmp_path, monkeypatch):
    """An operator reading the error gets the exact command to run."""
    _setup_registry(
        tmp_path,
        monkeypatch,
        config='{"ariel": {"database": {"uri": "postgresql://localhost/test"}},'
        ' "services": {"qmd": {"port": 8199}}}',
    )

    mock_service = AsyncMock()
    mock_service.search.return_value = _make_search_result(
        [], diagnostics=[_qmd_error("Hybrid search failed: sidecar unreachable")]
    )

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_hybrid_search()
        with assert_raises_error(error_type="service_unavailable") as _exc_ctx:
            await fn(query="beam loss")

    suggestions = _exc_ctx["envelope"]["suggestions"]
    assert any("curl http://127.0.0.1:8199/health" in s for s in suggestions)
    assert any("keyword_search" in s for s in suggestions)


@pytest.mark.unit
async def test_unconfigured_sidecar_points_at_the_config_block(tmp_path, monkeypatch):
    """With no services.qmd block there is no URL to curl, so say that instead."""
    _setup_registry(tmp_path, monkeypatch)

    mock_service = AsyncMock()
    mock_service.search.return_value = _make_search_result(
        [], diagnostics=[_qmd_error("Hybrid search failed: no qmd sidecar is configured")]
    )

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_hybrid_search()
        with assert_raises_error(error_type="service_unavailable") as _exc_ctx:
            await fn(query="beam loss")

    suggestions = _exc_ctx["envelope"]["suggestions"]
    assert any("services.qmd" in s for s in suggestions)
    assert not any("curl" in s for s in suggestions)


@pytest.mark.unit
async def test_non_qmd_diagnostics_do_not_trip_the_fault_path(tmp_path, monkeypatch):
    """Only this mode's own ERROR diagnostic means this mode failed."""
    _setup_registry(tmp_path, monkeypatch)

    unrelated = SearchDiagnostic(
        level=DiagnosticLevel.WARNING,
        source="service.hybrid",
        message="some entries were dropped",
        category="search",
    )
    other_mode = SearchDiagnostic(
        level=DiagnosticLevel.ERROR,
        source="service.semantic",
        message="Semantic search failed",
        category="search",
    )
    entries = [make_mock_entry(entry_id="e1")]

    mock_service = AsyncMock()
    mock_service.search.return_value = _make_search_result(
        entries, diagnostics=[unrelated, other_mode]
    )

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_hybrid_search()
        result = await fn(query="beam loss")

    data = extract_response_dict(result)
    assert not data.get("error", False)
    assert data["results_found"] == 1


@pytest.mark.unit
async def test_disabled_mode_names_the_enable_key(tmp_path, monkeypatch):
    """A registered-but-disabled mode is an operator state, not an internal error."""
    from osprey.services.ariel_search.exceptions import ConfigurationError

    _setup_registry(tmp_path, monkeypatch)

    mock_service = AsyncMock()
    mock_service.search.side_effect = ConfigurationError(
        "Search mode 'hybrid' is not enabled. Available modes: keyword, semantic",
        config_key="search_modules.hybrid.enabled",
    )

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_hybrid_search()
        with assert_raises_error(error_type="service_unavailable") as _exc_ctx:
            await fn(query="beam loss")

    data = _exc_ctx["envelope"]
    assert any("search_modules.hybrid.enabled" in s for s in data["suggestions"])


@pytest.mark.unit
async def test_unregistered_mode_does_not_advise_the_enable_key(tmp_path, monkeypatch):
    """The other ConfigurationError branch needs different advice.

    The service raises with config_key "modes" when the mode resolved against
    no registered module at all — the search module failed to import. The
    enable key is very likely already set there, so pointing at it would send
    the operator to a setting that is not the problem.
    """
    from osprey.services.ariel_search.exceptions import ConfigurationError

    _setup_registry(tmp_path, monkeypatch)

    mock_service = AsyncMock()
    mock_service.search.side_effect = ConfigurationError(
        "Unknown search mode 'hybrid'. Available modes: keyword, semantic",
        config_key="modes",
    )

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_hybrid_search()
        with assert_raises_error(error_type="service_unavailable") as _exc_ctx:
            await fn(query="beam loss")

    suggestions = _exc_ctx["envelope"]["suggestions"]
    assert not any("search_modules.hybrid.enabled" in s for s in suggestions)
    assert any("not registered" in s for s in suggestions)
    assert any("startup log" in s for s in suggestions)


@pytest.mark.unit
async def test_hybrid_search_service_error(tmp_path, monkeypatch):
    """Service failure returns standard error format."""
    _setup_registry(tmp_path, monkeypatch)

    mock_service = AsyncMock()
    mock_service.search.side_effect = RuntimeError("Pool exhausted")

    with patch(
        "osprey.mcp_server.ariel.server_context.ARIELContext.service",
        new=AsyncMock(return_value=mock_service),
    ):
        fn = _get_hybrid_search()
        with assert_raises_error(error_type="internal_error") as _exc_ctx:
            await fn(query="test")

    assert "Pool exhausted" in _exc_ctx["envelope"]["error_message"]


# ---------------------------------------------------------------------------
# Prompt surface and wiring
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_docstring_states_the_best_effort_filtering_caveat():
    """The caveat is the tool's substance: an agent that misses it stops early."""
    from osprey.mcp_server.ariel.tools.hybrid_search import hybrid_search

    doc = get_tool_fn(hybrid_search).__doc__ or ""

    assert "best-effort" in doc
    assert "fewer than max_results" in doc
    # It must steer to the backends that filter in the database.
    assert "keyword_search" in doc
    assert "sql_query" in doc


@pytest.mark.unit
def test_docstring_promises_no_field_the_response_omits():
    """A prompt surface must not name a field the tool never emits.

    The Returns line inherited a "workspace file path" claim that no ARIEL
    search tool has ever populated, which invites the agent to hunt for it or
    to report its absence as a failure.
    """
    from osprey.mcp_server.ariel.tools.hybrid_search import hybrid_search

    doc = get_tool_fn(hybrid_search).__doc__ or ""

    assert "workspace file path" not in doc
    assert "matching entries and relevance scores" in doc


@pytest.mark.unit
def test_tool_is_registered_on_the_ariel_server():
    """The server's tool-import tuple must actually pull this module in."""
    from pathlib import Path

    from osprey.mcp_server.ariel import server

    source = Path(server.__file__).read_text(encoding="utf-8")
    tool_import = source.split("from osprey.mcp_server.ariel.tools import")[1]
    assert "hybrid_search," in tool_import.split(")")[0]


@pytest.mark.unit
def test_tool_is_auto_allowed():
    """hybrid_search is a read tool, so it belongs in permissions_allow."""
    from osprey.registry.mcp import FRAMEWORK_SERVERS

    ariel = FRAMEWORK_SERVERS["ariel"]
    assert "hybrid_search" in ariel.permissions_allow
    assert "hybrid_search" not in ariel.permissions_ask
