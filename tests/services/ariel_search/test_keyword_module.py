"""Tests for the keyword search module's service-facing contract.

`tests/services/ariel_search/test_search.py` pins the direct-call behaviour of
`keyword_search` and must stay green with zero edits; this file covers what the
ARIEL search service added on top of it -- the pre-parsed query, vocabulary
expansion, literal pattern predicates, the timeout trigger, the fuzzy-fallback
ordering, and the `ModuleOutput` return shape.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from osprey.services.ariel_search.config import ARIELConfig
from osprey.services.ariel_search.exceptions import PatternError, SearchTimeoutError
from osprey.services.ariel_search.models import DiagnosticLevel
from osprey.services.ariel_search.search.base import (
    ExpansionGroup,
    ModuleOutput,
    QueryExpansion,
)
from osprey.services.ariel_search.search.keyword import (
    get_tool_descriptor,
    keyword_search,
    parse_keyword_query,
)

TS_BPM_EXPANSION = QueryExpansion(
    groups=(
        ExpansionGroup(original="ts", alternatives=("troubleshoot",)),
        ExpansionGroup(original="bpm", alternatives=("beam position monitor",)),
    ),
    flattened_text="ts bpm troubleshoot beam position monitor",
)


def make_config(**keyword_settings) -> ARIELConfig:
    """Build a minimal ARIEL config with the keyword module enabled.

    Args:
        **keyword_settings: Entries for ``search_modules.keyword.settings``.

    Returns:
        The parsed configuration.
    """
    keyword: dict = {"enabled": True}
    if keyword_settings:
        keyword["settings"] = dict(keyword_settings)
    return ARIELConfig.from_dict(
        {
            "database": {"uri": "postgresql://localhost/test"},
            "search_modules": {"keyword": keyword},
        }
    )


@pytest.fixture
def mock_config() -> ARIELConfig:
    """Default keyword-enabled config (patterns armed, 10 s envelope)."""
    return make_config()


@pytest.fixture
def mock_repository(mock_config: ARIELConfig) -> MagicMock:
    """Repository double whose searches return no rows by default."""
    repo = MagicMock()
    repo.config = mock_config
    repo.keyword_search = AsyncMock(return_value=[])
    repo.fuzzy_search = AsyncMock(return_value=[])
    return repo


def repo_call(repo: MagicMock) -> dict:
    """Return the kwargs of the single `keyword_search` call on `repo`."""
    assert repo.keyword_search.call_args is not None
    return repo.keyword_search.call_args.kwargs


def pattern_clauses(kwargs: dict) -> list[str]:
    """Return the ``raw_text ~* %s`` clauses of a repository call."""
    return [clause for clause in kwargs["where_clauses"] if "~*" in clause]


class TestDirectCallUnchanged:
    """A caller that supplies neither `parsed` nor `query_expansion`."""

    @pytest.mark.asyncio
    async def test_self_parses_and_returns_a_bare_list(self, mock_repository, mock_config):
        """No service arguments: the module parses, and answers with a list."""
        result = await keyword_search("beam current", mock_repository, mock_config)

        assert result == []
        assert not isinstance(result, ModuleOutput)
        kwargs = repo_call(mock_repository)
        assert kwargs["search_text"] == "beam current"
        assert kwargs["params"] == ["beam current"]

    @pytest.mark.asyncio
    async def test_repository_kwargs_are_todays_exact_set(self, mock_repository, mock_config):
        """A pattern-free, expansion-free search sends today's kwargs only."""
        await keyword_search("beam current", mock_repository, mock_config)

        assert set(repo_call(mock_repository)) == {
            "where_clauses",
            "params",
            "search_text",
            "max_results",
            "include_highlights",
        }

    @pytest.mark.asyncio
    async def test_empty_query_returns_a_bare_list(self, mock_repository, mock_config):
        """The blank-query short circuit keeps the direct caller's shape."""
        assert await keyword_search("   ", mock_repository, mock_config) == []


class TestServiceStyleCall:
    """Calls carrying the service's parse and resolved expansion."""

    @pytest.mark.asyncio
    async def test_parsed_alone_returns_module_output(self, mock_repository, mock_config):
        """`parsed=` marks a service call, so the answer is a ModuleOutput."""
        parsed = parse_keyword_query("beam current")

        result = await keyword_search("beam current", mock_repository, mock_config, parsed=parsed)

        assert isinstance(result, ModuleOutput)
        assert result.entries == []
        assert result.expansion == ()
        assert "tsquery_sql" not in repo_call(mock_repository)

    @pytest.mark.asyncio
    async def test_supplied_parse_is_not_redone(self, mock_repository, mock_config):
        """The module uses the parse it was handed, not the raw query."""
        parsed = parse_keyword_query("beam current")

        await keyword_search("something else entirely", mock_repository, mock_config, parsed=parsed)

        assert repo_call(mock_repository)["search_text"] == "beam current"

    @pytest.mark.asyncio
    async def test_expansion_drives_the_tsquery(self, mock_repository, mock_config):
        """An expansion produces the alternation SQL and its own parameters."""
        parsed = parse_keyword_query("ts bpm")

        result = await keyword_search(
            "ts bpm",
            mock_repository,
            mock_config,
            parsed=parsed,
            query_expansion=TS_BPM_EXPANSION,
        )

        assert isinstance(result, ModuleOutput)
        assert result.expansion == TS_BPM_EXPANSION.groups

        kwargs = repo_call(mock_repository)
        sql = kwargs["tsquery_sql"]
        assert sql.count("%s") == 4
        assert "||" in sql and "&&" in sql
        assert kwargs["tsquery_params"] == ["ts", "troubleshoot", "bpm", "beam position monitor"]
        # The WHERE clause binds the same values once, in placeholder order.
        assert kwargs["params"] == ["ts", "troubleshoot", "bpm", "beam position monitor"]
        assert any(sql in clause for clause in kwargs["where_clauses"])

    @pytest.mark.asyncio
    async def test_expansion_without_groups_takes_todays_path(self, mock_repository, mock_config):
        """An empty expansion is not an expansion: today's SQL, empty groups."""
        parsed = parse_keyword_query("beam current")

        result = await keyword_search(
            "beam current",
            mock_repository,
            mock_config,
            parsed=parsed,
            query_expansion=QueryExpansion(groups=(), flattened_text="beam current"),
        )

        assert isinstance(result, ModuleOutput)
        assert result.expansion == ()
        assert "tsquery_sql" not in repo_call(mock_repository)

    @pytest.mark.asyncio
    async def test_parse_diagnostics_are_carried_out(self, mock_repository, mock_config):
        """Truncation and pattern notices from the parse reach the caller."""
        long_query = "beam " + "a" * 1500
        parsed = parse_keyword_query(long_query)

        result = await keyword_search(long_query, mock_repository, mock_config, parsed=parsed)

        assert isinstance(result, ModuleOutput)
        assert [d.category for d in result.diagnostics] == ["truncation"]
        assert result.diagnostics[0].level is DiagnosticLevel.WARNING


class TestBooleanOperators:
    """Boolean queries cannot be grouped, so they are searched unexpanded."""

    @pytest.mark.asyncio
    async def test_boolean_query_skips_expansion_with_a_diagnostic(
        self, mock_repository, mock_config
    ):
        """`ts AND bpm` takes the websearch path and says why."""
        parsed = parse_keyword_query("ts AND bpm")

        result = await keyword_search(
            "ts AND bpm",
            mock_repository,
            mock_config,
            parsed=parsed,
            query_expansion=TS_BPM_EXPANSION,
        )

        assert isinstance(result, ModuleOutput)
        assert result.expansion == ()

        kwargs = repo_call(mock_repository)
        assert "tsquery_sql" not in kwargs
        assert any("websearch_to_tsquery" in clause for clause in kwargs["where_clauses"])
        assert kwargs["params"] == ["ts AND bpm"]

        skipped = [d for d in result.diagnostics if d.category == "expansion"]
        assert len(skipped) == 1
        assert skipped[0].level is DiagnosticLevel.INFO
        assert skipped[0].source == "keyword"
        assert skipped[0].message == "expansion skipped: boolean operators present"

    @pytest.mark.asyncio
    async def test_no_skip_diagnostic_without_an_expansion(self, mock_repository, mock_config):
        """A boolean query nobody offered an expansion for says nothing."""
        parsed = parse_keyword_query("ts AND bpm")

        result = await keyword_search("ts AND bpm", mock_repository, mock_config, parsed=parsed)

        assert isinstance(result, ModuleOutput)
        assert result.diagnostics == ()


class TestPatternPredicates:
    """Accepted pattern spans become ``raw_text ~* %s`` predicates."""

    @pytest.mark.asyncio
    async def test_glob_becomes_a_pattern_predicate(self, mock_repository, mock_config):
        """`SR01C___BPM*` binds its anchored ARE and arms the envelope."""
        await keyword_search("SR01C___BPM*", mock_repository, mock_config)

        kwargs = repo_call(mock_repository)
        assert pattern_clauses(kwargs) == ["raw_text ~* %s"]
        assert kwargs["params"] == [r"\mSR01C___BPM\S*"]
        assert kwargs["pattern_timeout_seconds"] == 10.0

    @pytest.mark.asyncio
    async def test_pattern_predicate_follows_the_fts_clause(self, mock_repository, mock_config):
        """Clause order and parameter order stay aligned."""
        await keyword_search("trip SR01C___BPM*", mock_repository, mock_config)

        kwargs = repo_call(mock_repository)
        assert kwargs["where_clauses"][1] == "raw_text ~* %s"
        assert kwargs["params"] == ["trip", r"\mSR01C___BPM\S*"]

    @pytest.mark.asyncio
    async def test_too_generic_pattern_is_searched_as_text(self, mock_repository, mock_config):
        """`/ab/` has no 3-character literal run: no predicate, but a FTS leg."""
        await keyword_search("/ab/", mock_repository, mock_config)

        kwargs = repo_call(mock_repository)
        assert pattern_clauses(kwargs) == []
        assert "pattern_timeout_seconds" not in kwargs
        assert any("plainto_tsquery" in clause for clause in kwargs["where_clauses"])
        assert "ab" in kwargs["params"][0]

    @pytest.mark.asyncio
    async def test_union_select_injection_sends_no_pattern_clause(
        self, mock_repository, mock_config
    ):
        """The `*` of a SQL-injection probe never becomes a pattern predicate.

        The companion of ``test_union_select_injection`` in ``test_search.py``:
        that one pins the query as literal text, this one pins that the pattern
        machinery declines it too -- ``*`` alone has no literal run at all.
        """
        await keyword_search("test UNION SELECT * FROM users --", mock_repository, mock_config)

        kwargs = repo_call(mock_repository)
        assert pattern_clauses(kwargs) == []
        assert "pattern_timeout_seconds" not in kwargs
        assert "*" in kwargs["params"][0]

    @pytest.mark.asyncio
    async def test_patterns_disabled_makes_a_glob_ordinary_text(self, mock_repository):
        """`patterns_enabled: false` searches `SR01C___BPM*` as text."""
        config = make_config(patterns_enabled=False)

        await keyword_search("SR01C___BPM*", mock_repository, config)

        kwargs = repo_call(mock_repository)
        assert pattern_clauses(kwargs) == []
        assert kwargs["params"] == ["SR01C___BPM*"]
        assert "pattern_timeout_seconds" not in kwargs

    @pytest.mark.asyncio
    async def test_patterns_disabled_ignores_a_supplied_parse_spans(self, mock_repository):
        """The service parses with patterns armed; the knob still wins here."""
        config = make_config(patterns_enabled=False)
        parsed = parse_keyword_query("SR01C___BPM*")
        assert parsed.pattern_spans  # the service's parse did find one

        await keyword_search("SR01C___BPM*", mock_repository, config, parsed=parsed)

        kwargs = repo_call(mock_repository)
        assert pattern_clauses(kwargs) == []
        assert kwargs["params"] == ["SR01C___BPM*"]
        assert "pattern_timeout_seconds" not in kwargs

    @pytest.mark.asyncio
    async def test_configured_timeout_is_passed(self, mock_repository):
        """The envelope comes from config, never a constant."""
        config = make_config(pattern_timeout_seconds=0.5)

        await keyword_search("SR01C___BPM*", mock_repository, config)

        assert repo_call(mock_repository)["pattern_timeout_seconds"] == 0.5

    @pytest.mark.asyncio
    async def test_pattern_only_query_sends_empty_search_text(self, mock_repository, mock_config):
        """No FTS text at all: the repository orders by timestamp instead."""
        await keyword_search(r"/SR01C___BPM\d+/", mock_repository, mock_config)

        kwargs = repo_call(mock_repository)
        assert kwargs["search_text"] == ""
        assert kwargs["where_clauses"] == ["raw_text ~* %s"]
        assert kwargs["params"] == [r"SR01C___BPM\d+"]
        mock_repository.fuzzy_search.assert_not_called()


class TestFuzzyFallback:
    """The fallback probes what the operator typed before anything else."""

    @pytest.mark.asyncio
    async def test_original_text_is_probed_first(self, mock_repository, mock_config):
        """Zero rows without expansion: one fuzzy probe, on the typed text."""
        parsed = parse_keyword_query("beaam")

        await keyword_search("beaam", mock_repository, mock_config, parsed=parsed)

        assert mock_repository.fuzzy_search.await_count == 1
        assert mock_repository.fuzzy_search.call_args.kwargs["search_text"] == "beaam"

    @pytest.mark.asyncio
    async def test_flattened_text_probed_only_after_the_original(
        self, mock_repository, mock_config
    ):
        """Still zero rows and expansion active: probe the flattened text."""
        parsed = parse_keyword_query("ts bpm")

        await keyword_search(
            "ts bpm",
            mock_repository,
            mock_config,
            parsed=parsed,
            query_expansion=TS_BPM_EXPANSION,
        )

        probes = [
            call.kwargs["search_text"] for call in mock_repository.fuzzy_search.call_args_list
        ]
        assert probes == ["ts bpm", TS_BPM_EXPANSION.flattened_text]

    @pytest.mark.asyncio
    async def test_no_second_probe_when_the_first_found_rows(self, mock_repository, mock_config):
        """Expansion can add hits, never remove one: the first answer stands."""
        mock_repository.fuzzy_search = AsyncMock(return_value=[("entry", 0.4, [])])
        parsed = parse_keyword_query("ts bpm")

        await keyword_search(
            "ts bpm",
            mock_repository,
            mock_config,
            parsed=parsed,
            query_expansion=TS_BPM_EXPANSION,
        )

        assert mock_repository.fuzzy_search.await_count == 1

    @pytest.mark.asyncio
    async def test_patterns_skip_the_fallback_entirely(self, mock_repository, mock_config):
        """A literal pattern that matched nothing means nothing matched."""
        await keyword_search("trip SR01C___BPM*", mock_repository, mock_config)

        mock_repository.fuzzy_search.assert_not_called()


class TestErrorPropagation:
    """Pattern and timeout failures reach the caller, never an empty list."""

    @pytest.mark.asyncio
    async def test_pattern_error_is_named_and_propagates(self, mock_repository, mock_config):
        """The module knows which query text produced the rejected pattern."""
        mock_repository.keyword_search = AsyncMock(
            side_effect=PatternError("invalid regular expression: brackets [] not balanced")
        )

        with pytest.raises(PatternError) as excinfo:
            await keyword_search("t/s /SR0[1-4/", mock_repository, mock_config)

        assert excinfo.value.pattern == "/SR0[1-4/"
        mock_repository.fuzzy_search.assert_not_called()

    @pytest.mark.asyncio
    async def test_pattern_error_keeps_an_already_named_pattern(self, mock_repository, mock_config):
        """A repository that named the pattern itself is not second-guessed."""
        mock_repository.keyword_search = AsyncMock(
            side_effect=PatternError("nope", pattern="from the repository")
        )

        with pytest.raises(PatternError) as excinfo:
            await keyword_search("t/s /SR0[1-4/", mock_repository, mock_config)

        assert excinfo.value.pattern == "from the repository"

    @pytest.mark.asyncio
    async def test_timeout_propagates(self, mock_repository, mock_config):
        """The service converts a timeout into a diagnostic; the module does not."""
        mock_repository.keyword_search = AsyncMock(
            side_effect=SearchTimeoutError("too slow", timeout_seconds=10.0, operation="keyword")
        )

        with pytest.raises(SearchTimeoutError):
            await keyword_search("SR01C___BPM*", mock_repository, mock_config)


class TestDescriptor:
    """The descriptor is how the service learns what this module accepts."""

    def test_declares_expansion_and_its_parser(self):
        """Both opt-in fields are set, and the parser is this module's."""
        descriptor = get_tool_descriptor()

        assert descriptor.accepts_expansion is True
        assert descriptor.query_parser is parse_keyword_query
        assert descriptor.search_mode == "keyword"
