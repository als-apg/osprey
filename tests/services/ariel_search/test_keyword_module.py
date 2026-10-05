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
from osprey.services.ariel_search.exceptions import (
    DatabaseQueryError,
    PatternError,
    SearchTimeoutError,
)
from osprey.services.ariel_search.models import DiagnosticLevel
from osprey.services.ariel_search.search.base import (
    ExpansionGroup,
    ModuleOutput,
    QueryExpansion,
)
from osprey.services.ariel_search.search.keyword import (
    ALLOWED_FIELD_PREFIXES,
    get_tool_descriptor,
    keyword_search,
    parse_keyword_query,
)
from tests.services.ariel_search.repo_fakes import attach_fake_fts

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
    attach_fake_fts(repo, has_v2=False, has_copy_state=False)
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
            "v2",
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


class TestFuzzyThreshold:
    """Both fuzzy-fallback probes use the configured similarity floor."""

    @staticmethod
    def thresholds(repo: MagicMock) -> list[float]:
        """Return the ``threshold`` kwarg of every ``fuzzy_search`` call on `repo`."""
        return [call.kwargs["threshold"] for call in repo.fuzzy_search.call_args_list]

    @pytest.mark.asyncio
    async def test_default_floor_reaches_both_probes(self, mock_repository, mock_config):
        """No key: both probes use the default floor."""
        await keyword_search(
            "ts bpm",
            mock_repository,
            mock_config,
            parsed=parse_keyword_query("ts bpm"),
            query_expansion=TS_BPM_EXPANSION,
        )

        assert self.thresholds(mock_repository) == [0.3, 0.3]

    @pytest.mark.asyncio
    async def test_configured_floor_reaches_both_probes(self, mock_repository):
        """A configured floor reaches both probes."""
        await keyword_search(
            "ts bpm",
            mock_repository,
            make_config(fuzzy_threshold=0.55),
            parsed=parse_keyword_query("ts bpm"),
            query_expansion=TS_BPM_EXPANSION,
        )

        assert self.thresholds(mock_repository) == [0.55, 0.55]

    @pytest.mark.asyncio
    async def test_malformed_floor_is_refused_before_any_query(self, mock_repository):
        """A bad floor refuses the search before any statement runs."""
        with pytest.raises(ValueError, match=r"fuzzy_threshold must be a number in \[0, 1\]"):
            await keyword_search("beaam", mock_repository, make_config(fuzzy_threshold=2))

        mock_repository.keyword_search.assert_not_called()
        mock_repository.fuzzy_search.assert_not_called()


class TestFieldFilters:
    """Every field prefix the parser lifts is applied as a predicate."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("prefix", sorted(ALLOWED_FIELD_PREFIXES))
    async def test_every_prefix_reaches_a_predicate(self, prefix, mock_repository, mock_config):
        """A prefix with no predicate would drop its token from the search unapplied."""
        await keyword_search(f"beam {prefix}2024-01-15", mock_repository, mock_config)

        kwargs = repo_call(mock_repository)
        assert len(kwargs["where_clauses"]) == 2
        assert kwargs["params"][0] == "beam"
        assert any("2024-01-15" in str(param) for param in kwargs["params"][1:])


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


class TestSchemaFactSelection:
    """``has_v2_fts`` decides whether pattern and fuzzy matching reach attachment text."""

    @pytest.mark.asyncio
    async def test_schema_fact_is_read_once_and_passed_on(self, mock_repository, mock_config):
        attach_fake_fts(mock_repository, has_v2=True, has_copy_state=True)

        await keyword_search("beam current", mock_repository, mock_config)

        assert mock_repository.schema_facts.await_count == 1
        assert repo_call(mock_repository)["v2"] is True
        assert mock_repository.fuzzy_search.call_args.kwargs["v2"] is True

    @pytest.mark.asyncio
    async def test_v2_pattern_span_also_matches_attachment_text(self, mock_repository, mock_config):
        attach_fake_fts(mock_repository, has_v2=True, has_copy_state=True)

        await keyword_search("trip SR01C___BPM*", mock_repository, mock_config)

        kwargs = repo_call(mock_repository)
        assert kwargs["where_clauses"][1] == (
            "(raw_text ~* %s OR COALESCE(attachment_text,'') ~* %s)"
        )
        body = r"\mSR01C___BPM\S*"
        assert kwargs["params"] == ["trip", body, body]
        assert kwargs["where_clauses"][1].count("%s") == 2

    @pytest.mark.asyncio
    async def test_pattern_second_arm_is_the_trigram_index_expression(
        self, mock_repository, mock_config
    ):
        """The planner can use idx_entries_attachment_text_trgm only on this exact text."""
        import inspect

        from osprey.services.ariel_search.database.attachment_text_migration import (
            RawTextFtsIndexV2Migration,
        )
        from osprey.services.ariel_search.database.search_fts import ATTACHMENT_TEXT_DOCUMENT

        attach_fake_fts(mock_repository, has_v2=True, has_copy_state=True)
        await keyword_search("SR01C___BPM*", mock_repository, mock_config)

        (clause,) = pattern_clauses(repo_call(mock_repository))
        assert f"OR {ATTACHMENT_TEXT_DOCUMENT} ~* %s)" in clause
        source = inspect.getsource(RawTextFtsIndexV2Migration.up)
        assert "idx_entries_attachment_text_trgm" in source
        assert "({ATTACHMENT_TEXT_DOCUMENT}) gin_trgm_ops" in source

    @pytest.mark.asyncio
    async def test_schema_behind_pattern_is_b1_exact(self, mock_repository, mock_config):
        attach_fake_fts(mock_repository, has_v2=False, has_copy_state=True)

        await keyword_search("trip SR01C___BPM*", mock_repository, mock_config)

        kwargs = repo_call(mock_repository)
        assert kwargs["where_clauses"][1] == "raw_text ~* %s"
        assert kwargs["v2"] is False


class TestSchemaBehindStatements:
    """A store without the V2 indexes is never sent an ``attachment_text`` token."""

    @staticmethod
    def _repo(pool):
        from osprey.services.ariel_search.database.repository import ARIELRepository

        config = make_config()
        repo = ARIELRepository(pool, config)
        attach_fake_fts(repo, has_v2=False, has_copy_state=False)
        return repo, config

    @pytest.mark.asyncio
    async def test_pattern_query_sql_has_no_attachment_text(self, fake_pool):
        repo, config = self._repo(fake_pool)

        await keyword_search("QX-77*", repo, config)

        statements = fake_pool.recorder.matching("~*")
        assert statements, "the pattern query reached the database"
        assert all("attachment_text" not in sql for sql, _ in fake_pool.calls)

    @pytest.mark.asyncio
    async def test_fuzzy_fallback_sql_has_no_attachment_text(self, fake_pool):
        repo, config = self._repo(fake_pool)

        await keyword_search("quensh", repo, config)

        assert fake_pool.recorder.matching("similarity(raw_text, %s) >= %s")
        assert all("attachment_text" not in sql for sql, _ in fake_pool.calls)


class TestSchemaBehindDiagnostic:
    """Searches report a behind schema; the keyword module logs it once."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("has_v2", "has_copy_state", "reported"),
        [(False, False, True), (False, True, True), (True, False, True), (True, True, False)],
    )
    async def test_warning_while_any_fact_is_false(
        self, mock_repository, has_v2, has_copy_state, reported
    ):
        from osprey.services.ariel_search.database.repository import (
            SCHEMA_BEHIND_SEARCH_MESSAGE,
            schema_behind_diagnostics,
        )

        attach_fake_fts(mock_repository, has_v2=has_v2, has_copy_state=has_copy_state)

        found = await schema_behind_diagnostics(mock_repository)

        if reported:
            (diagnostic,) = found
            assert diagnostic.level is DiagnosticLevel.WARNING
            assert diagnostic.message == "schema behind code: run osprey ariel migrate"
            assert diagnostic.message == SCHEMA_BEHIND_SEARCH_MESSAGE
        else:
            assert found == []

    @pytest.mark.asyncio
    async def test_a_repository_without_facts_reports_nothing(self):
        from osprey.services.ariel_search.database.repository import schema_behind_diagnostics

        assert await schema_behind_diagnostics(object()) == []
        assert await schema_behind_diagnostics(MagicMock()) == []

    @pytest.mark.asyncio
    async def test_keyword_module_logs_a_behind_schema_once(
        self, mock_repository, mock_config, monkeypatch, caplog
    ):
        from osprey.services.ariel_search.database import repository as repository_module

        monkeypatch.setattr(repository_module, "_schema_behind_search_warned", False)
        with caplog.at_level("WARNING", logger="ariel"):
            await keyword_search("beam", mock_repository, mock_config)
            await keyword_search("beam", mock_repository, mock_config)

        messages = [r.getMessage() for r in caplog.records]
        assert messages.count(repository_module.SCHEMA_BEHIND_SEARCH_MESSAGE) == 1


class TestCaptionMatchedIds:
    """Hits carry the ids of the attachments whose caption matched (requirement 3)."""

    @staticmethod
    def hits(*entry_ids: str) -> list[tuple[dict, float, list[str]]]:
        return [({"entry_id": eid, "raw_text": "x"}, 0.5, []) for eid in entry_ids]

    @pytest.mark.asyncio
    async def test_caption_only_mention_returns_its_attachment_id(
        self, mock_repository, mock_config
    ):
        """`SR:C07 BPM` hit through a caption is marked with that caption's id."""
        from tests.services.ariel_search.repo_fakes import attach_fake_caption_matches

        mock_repository.keyword_search = AsyncMock(return_value=self.hits("e1", "e2"))
        fake = attach_fake_caption_matches(mock_repository, {"e1": ["att-plot"]})

        results = await keyword_search("SR:C07 BPM", mock_repository, mock_config)

        assert fake.await_count == 1
        args, kwargs = fake.call_args.args, fake.call_args.kwargs
        assert args == (["e1", "e2"], None)
        assert kwargs["tsquery_sql"] == repo_call(mock_repository)["where_clauses"][0].split(
            " @@ (", 1
        )[1].removesuffix(")")
        assert kwargs["tsquery_params"] == ["SR:C07 BPM"]
        assert kwargs["pattern_bodies"] == []
        by_id = {entry["entry_id"]: entry for entry, _s, _h in results}
        assert by_id["e1"]["_matched_attachment_ids"] == ["att-plot"]
        assert "_matched_attachment_ids" not in by_id["e2"]

    @pytest.mark.asyncio
    async def test_glob_matches_caption_and_returns_its_id(self, mock_repository, mock_config):
        """`QX-77*` passes its pattern body, and the caption-only entry gets its id."""
        from tests.services.ariel_search.repo_fakes import attach_fake_caption_matches

        mock_repository.keyword_search = AsyncMock(return_value=self.hits("e1"))
        fake = attach_fake_caption_matches(mock_repository, {"e1": ["att-qx"]})

        results = await keyword_search("QX-77*", mock_repository, mock_config)

        kwargs = fake.call_args.kwargs
        assert kwargs["pattern_bodies"] == [pattern_clauses_params(repo_call(mock_repository))]
        assert kwargs["tsquery_sql"] is None
        assert results[0][0]["_matched_attachment_ids"] == ["att-qx"]

    @pytest.mark.asyncio
    async def test_expanded_tsquery_is_the_caption_form(self, mock_repository, mock_config):
        """With expansion applied, the caption check uses the expanded tsquery."""
        mock_repository.keyword_search = AsyncMock(return_value=self.hits("e1"))

        await keyword_search(
            "ts bpm",
            mock_repository,
            mock_config,
            parsed=parse_keyword_query("ts bpm"),
            query_expansion=TS_BPM_EXPANSION,
        )

        main = repo_call(mock_repository)
        kwargs = mock_repository.caption_matches.call_args.kwargs
        assert kwargs["tsquery_sql"] == main["tsquery_sql"]
        assert kwargs["tsquery_params"] == list(main["tsquery_params"])

    @pytest.mark.asyncio
    async def test_configured_caption_model_is_passed(self, mock_repository):
        """The configured `image_caption` model id selects the model captions."""
        config = ARIELConfig.from_dict(
            {
                "database": {"uri": "postgresql://localhost/test"},
                "search_modules": {"keyword": {"enabled": True}},
                "enhancement_modules": {
                    "image_caption": {"enabled": False, "model": {"model_id": "cap-model"}}
                },
            }
        )
        mock_repository.keyword_search = AsyncMock(return_value=self.hits("e1"))

        await keyword_search("beam", mock_repository, config)

        assert mock_repository.caption_matches.call_args.args[1] == "cap-model"

    @pytest.mark.asyncio
    async def test_no_hits_asks_nothing(self, mock_repository, mock_config):
        """Without hits there is nothing to mark, and no statement runs."""
        await keyword_search("beam", mock_repository, mock_config)

        mock_repository.caption_matches.assert_not_called()

    @pytest.mark.asyncio
    async def test_fuzzy_hits_carry_no_ids(self, mock_repository, mock_config):
        """Fuzzy fallback emits no caption ids."""
        mock_repository.fuzzy_search = AsyncMock(return_value=self.hits("e1"))

        results = await keyword_search("beaam", mock_repository, mock_config)

        mock_repository.caption_matches.assert_not_called()
        assert "_matched_attachment_ids" not in results[0][0]

    @pytest.mark.asyncio
    async def test_service_call_keeps_the_tuple_shape(self, mock_repository, mock_config):
        """The ids ride on the entry; the result tuples keep their shape."""
        from tests.services.ariel_search.repo_fakes import attach_fake_caption_matches

        mock_repository.keyword_search = AsyncMock(return_value=self.hits("e1"))
        attach_fake_caption_matches(mock_repository, {"e1": ["a", "b"]})

        out = await keyword_search(
            "beam", mock_repository, mock_config, parsed=parse_keyword_query("beam")
        )

        assert isinstance(out, ModuleOutput)
        ((entry, score, highlights),) = out.entries
        assert (score, highlights) == (0.5, [])
        assert entry["_matched_attachment_ids"] == ["a", "b"]

    @pytest.mark.parametrize(
        "error",
        [
            SearchTimeoutError("caption check timed out", timeout_seconds=10, operation="kw"),
            DatabaseQueryError("caption check failed"),
        ],
        ids=["timeout", "database"],
    )
    @pytest.mark.asyncio
    async def test_failure_logs_one_warning_and_keeps_the_hits(
        self, error, mock_repository, mock_config, caplog
    ):
        """Supplementary evidence never fails a search the main statement answered."""
        import logging

        from tests.services.ariel_search.repo_fakes import attach_fake_caption_matches

        mock_repository.keyword_search = AsyncMock(return_value=self.hits("e1"))
        attach_fake_caption_matches(mock_repository, error=error)

        with caplog.at_level(logging.WARNING):
            results = await keyword_search("beam", mock_repository, mock_config)

        assert [entry["entry_id"] for entry, _s, _h in results] == ["e1"]
        assert "_matched_attachment_ids" not in results[0][0]
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len([r for r in warnings if "caption" in r.getMessage()]) == 1


def pattern_clauses_params(kwargs: dict) -> str:
    """Return the single pattern body bound in a repository call."""
    (clause,) = pattern_clauses(kwargs)
    index = kwargs["where_clauses"].index(clause)
    offset = sum(clause_.count("%s") for clause_ in kwargs["where_clauses"][:index])
    return kwargs["params"][offset]
