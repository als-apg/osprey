r"""Real-PostgreSQL criteria for vocabulary expansion and pattern tokens.

Every assertion here needs a real database: ranking (``ts_rank`` > 0), glob and
regex semantics under PostgreSQL's ARE engine, the ``pg_trgm`` fuzzy fallback,
and the ``SET LOCAL statement_timeout`` envelope -- none of which a fake pool
can answer. Unit coverage of the same machinery lives beside the modules; this
file exists to prove the composed path behaves as the proposal says it does.

Most tests drive the full ``parse -> expand -> SQL`` path through
:class:`ARIELSearchService`. The handful that assert on *what SQL was issued*
(zero ``~*`` clauses, no transaction for a pattern-free search, the timeout
observed inside the transaction) put a recording proxy between the repository
and the real pool -- see ``RecordingPool`` in this package's ``conftest.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from tests.services.ariel_search.integration.conftest import (
    ACRONYM_ONLY_ID,
    GLOB_BARE_ID,
    GLOB_BEAM_LOSS_ID,
    GLOB_BPM3_ID,
    GLOB_PREFIXED_ID,
    GLOB_SR011_ID,
    GLOB_SR02_ID,
    LITERAL_AB_ID,
    SHORT_NEEDLE_ID,
    THREE_TERM_NEEDLE_ID,
    RecordingPool,
)

if TYPE_CHECKING:
    from osprey.services.ariel_search.config import ARIELConfig
    from osprey.services.ariel_search.models import ARIELSearchResult

# xdist_group("docker"): pins every container-starting test file onto one worker, so
# a run has a single testcontainers session and a single ryuk reaper -- concurrent
# reaper starts race the Docker daemon's port mapper. It also serializes the shared
# database: the session ``database_url`` fixture prefers a running dev Postgres with
# ONE shared ``ariel_test`` database over a per-worker container, so parallel workers
# would otherwise collide on migrations/seed/truncate.
pytestmark = [pytest.mark.integration, pytest.mark.asyncio, pytest.mark.xdist_group("docker")]

#: Wide enough that the small probe sets are never truncated by the LIMIT.
WIDE = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ids(result: ARIELSearchResult) -> list[str]:
    """Entry ids of a search result, in rank order."""
    return [entry["entry_id"] for entry in result.entries]


def score_of(result: ARIELSearchResult, entry_id: str) -> float:
    """The ``_score`` the service attached to one entry."""
    for entry in result.entries:
        if entry["entry_id"] == entry_id:
            return float(entry["_score"])
    raise AssertionError(f"{entry_id} not in results: {ids(result)}")


def groups(result: ARIELSearchResult) -> list[tuple[str, tuple[str, ...]]]:
    """``expanded_terms`` as ``(original, alternatives)`` pairs."""
    return [(g["original"], tuple(g["alternatives"])) for g in result.expanded_terms]


def categories(result: ARIELSearchResult) -> list[str]:
    """Diagnostic categories carried by a result."""
    return [d.category for d in result.diagnostics]


def build_service(config: ARIELConfig, pool: Any) -> Any:
    """Construct a service over an already-migrated pool."""
    from osprey.services.ariel_search import ARIELSearchService
    from osprey.services.ariel_search.database.repository import ARIELRepository

    return ARIELSearchService(config=config, pool=pool, repository=ARIELRepository(pool, config))


def build_recording_service(config: ARIELConfig, pool: Any) -> tuple[Any, RecordingPool]:
    """Construct a service whose repository issues statements through a recorder.

    The service keeps the real pool (nothing else uses it), while the
    repository -- the only component that talks SQL -- gets the proxy.

    Args:
        config: The configuration to run under.
        pool: The migrated connection pool.

    Returns:
        ``(service, recording_pool)``.
    """
    from osprey.services.ariel_search import ARIELSearchService
    from osprey.services.ariel_search.database.repository import ARIELRepository

    recorder = RecordingPool(pool)
    service = ARIELSearchService(
        config=config, pool=pool, repository=ARIELRepository(recorder, config)
    )
    return service, recorder


async def search(service: Any, query: str, **advanced: Any) -> ARIELSearchResult:
    """Run a keyword search through the service."""
    return await service.search(
        query, mode="keyword", max_results=WIDE, advanced_params=dict(advanced)
    )


# ---------------------------------------------------------------------------
# Expansion effect
# ---------------------------------------------------------------------------


class TestExpansionEffect:
    """Vocabulary expansion against seeded canonical-only prose."""

    async def test_shorthand_query_finds_canonical_entry_with_positive_rank(
        self, prose_repository, migrated_pool, vocabulary_config_factory, vocabulary_file
    ):
        """`ts bpm` reaches a canonical-only entry and it ranks above zero.

        The composed tsquery drives ``ts_rank`` as well as the WHERE clause, so
        an expanded hit is a ranked hit -- not a rank-0 row the UI would sort
        last.
        """
        config = vocabulary_config_factory(vocabulary_file)
        service = build_service(config, migrated_pool)

        result = await search(service, "ts bpm")

        assert SHORT_NEEDLE_ID in ids(result)
        assert score_of(result, SHORT_NEEDLE_ID) > 0
        assert groups(result) == [
            ("ts", ("troubleshoot",)),
            ("bpm", ("beam position monitor",)),
        ]

    async def test_third_term_discriminates_between_seeded_entries(
        self, prose_repository, migrated_pool, vocabulary_config_factory, vocabulary_file
    ):
        """`ts bpm undulator` keeps the AND semantics of the unexpanded query."""
        config = vocabulary_config_factory(vocabulary_file)
        service = build_service(config, migrated_pool)

        result = await search(service, "ts bpm undulator")

        assert THREE_TERM_NEEDLE_ID in ids(result)
        assert SHORT_NEEDLE_ID not in ids(result)

    async def test_expand_query_false_finds_nothing_and_reports_nothing(
        self, prose_repository, migrated_pool, vocabulary_config_factory, vocabulary_file
    ):
        """The per-request off switch reaches the SQL, not just the report."""
        config = vocabulary_config_factory(vocabulary_file)
        service = build_service(config, migrated_pool)

        result = await search(service, "ts bpm", expand_query=False)

        assert SHORT_NEEDLE_ID not in ids(result)
        assert THREE_TERM_NEEDLE_ID not in ids(result)
        assert result.expanded_terms == ()

    async def test_ambiguous_form_yields_one_group_with_both_canonicals(
        self, prose_repository, migrated_pool, vocabulary_config_factory, ambiguous_vocabulary_file
    ):
        """A form bound twice reports ONE group listing both bindings."""
        config = vocabulary_config_factory(ambiguous_vocabulary_file)
        service = build_service(config, migrated_pool)

        result = await search(service, "ts fault")

        assert groups(result) == [("ts", ("troubleshoot", "timing system"))]


class TestDirectionGates:
    """`canonical_to_acronym` decides whether prose reaches acronym-only rows."""

    async def test_gate_on_reaches_acronym_only_entry(
        self, prose_repository, migrated_pool, vocabulary_config_factory, vocabulary_file
    ):
        """With the gate on, `beam position monitor` also searches `bpm`."""
        config = vocabulary_config_factory(vocabulary_file, canonical_to_acronym=True)
        service = build_service(config, migrated_pool)

        result = await search(service, "beam position monitor")

        assert ACRONYM_ONLY_ID in ids(result)
        assert groups(result) == [("beam position monitor", ("bpm", "bpms"))]

    async def test_gate_off_leaves_acronym_only_entry_unreachable(
        self, prose_repository, migrated_pool, vocabulary_config_factory, vocabulary_file
    ):
        """With the gate off the query is searched exactly as typed."""
        config = vocabulary_config_factory(vocabulary_file, canonical_to_acronym=False)
        service = build_service(config, migrated_pool)

        result = await search(service, "beam position monitor")

        assert result.expanded_terms == ()
        assert ACRONYM_ONLY_ID not in ids(result)
        # The unexpanded query still works -- the canonical prose rows are hits.
        assert SHORT_NEEDLE_ID in ids(result)


class TestExpandByDefault:
    """`expand_by_default` resolves expansion when the request says nothing."""

    async def test_unset_expands(
        self, prose_repository, migrated_pool, vocabulary_config_factory, vocabulary_file
    ):
        """Omitting the key keeps the documented default (expansion on)."""
        from osprey.services.ariel_search.capabilities import get_capabilities

        config = vocabulary_config_factory(vocabulary_file)
        service = build_service(config, migrated_pool)

        result = await search(service, "ts bpm")

        assert result.expanded_terms != ()
        assert SHORT_NEEDLE_ID in ids(result)
        assert get_capabilities(config)["vocabulary"]["expand_by_default"] is True

    async def test_false_runs_the_unexpanded_sql(
        self, prose_repository, migrated_pool, vocabulary_config_factory, vocabulary_file
    ):
        """`expand_by_default: false` with no request flag searches as typed."""
        from osprey.services.ariel_search.capabilities import get_capabilities

        config = vocabulary_config_factory(vocabulary_file, expand_by_default=False)
        service, recorder = build_recording_service(config, migrated_pool)

        result = await search(service, "ts bpm")

        assert result.expanded_terms == ()
        assert SHORT_NEEDLE_ID not in ids(result)
        # One plain leg, bound with the query as typed -- no alternation.
        assert "troubleshoot" not in recorder.bound_params
        assert get_capabilities(config)["vocabulary"]["expand_by_default"] is False

    async def test_request_flag_overrides_expand_by_default_false(
        self, prose_repository, migrated_pool, vocabulary_config_factory, vocabulary_file
    ):
        """An explicit `expand_query=True` wins over the deployment default."""
        config = vocabulary_config_factory(vocabulary_file, expand_by_default=False)
        service = build_service(config, migrated_pool)

        result = await search(service, "ts bpm", expand_query=True)

        assert SHORT_NEEDLE_ID in ids(result)
        assert result.expanded_terms != ()


# ---------------------------------------------------------------------------
# Pattern semantics
# ---------------------------------------------------------------------------


class TestGlobSemantics:
    """Glob and regex tokens executed as real ``raw_text ~* %s`` predicates."""

    async def test_star_matches_suffix_and_bare_name_but_not_a_prefixed_one(
        self, glob_repository, migrated_pool, vocabulary_config_factory, vocabulary_file
    ):
        """`SR01C___BPM*` is word-anchored at the front, open at the back."""
        config = vocabulary_config_factory(vocabulary_file)
        service = build_service(config, migrated_pool)

        found = ids(await search(service, "SR01C___BPM*"))

        assert GLOB_BPM3_ID in found
        assert GLOB_BARE_ID in found
        assert GLOB_PREFIXED_ID not in found

    async def test_question_mark_inside_a_glob_is_one_character(
        self, glob_repository, migrated_pool, vocabulary_config_factory, vocabulary_file
    ):
        """`SR0?C___BPM*` matches one sector digit, never two."""
        config = vocabulary_config_factory(vocabulary_file)
        service = build_service(config, migrated_pool)

        found = ids(await search(service, "SR0?C___BPM*"))

        assert GLOB_BPM3_ID in found
        assert GLOB_SR02_ID in found
        assert GLOB_SR011_ID not in found

    async def test_trailing_question_token_is_searched_as_text(
        self, glob_repository, migrated_pool, vocabulary_config_factory, vocabulary_file
    ):
        """`SR01C___BPM?` carries no `*`, so it never becomes a predicate."""
        config = vocabulary_config_factory(vocabulary_file)
        service, recorder = build_recording_service(config, migrated_pool)

        result = await search(service, "SR01C___BPM?")

        assert recorder.pattern_clause_count == 0
        assert GLOB_BARE_ID in ids(result)

    async def test_regex_spans_match_including_one_containing_a_space(
        self, glob_repository, migrated_pool, vocabulary_config_factory, vocabulary_file
    ):
        r"""`/SR0[1-4]C___BPM\d+/` and `/beam loss/` both run as predicates."""
        config = vocabulary_config_factory(vocabulary_file)
        service = build_service(config, migrated_pool)

        assert GLOB_BPM3_ID in ids(await search(service, r"/SR0[1-4]C___BPM\d+/"))
        assert GLOB_BEAM_LOSS_ID in ids(await search(service, "/beam loss/"))

    @pytest.mark.parametrize("query", ["BPM (SR01)", "C++", "did the BPM trip?"])
    async def test_ordinary_punctuation_is_never_read_as_a_pattern(
        self, query, glob_repository, migrated_pool, vocabulary_config_factory, vocabulary_file
    ):
        """Parentheses, `+` and a trailing `?` produce no predicate and no notice."""
        config = vocabulary_config_factory(vocabulary_file)
        service, recorder = build_recording_service(config, migrated_pool)

        result = await search(service, query)

        assert recorder.pattern_clause_count == 0
        assert "pattern" not in categories(result)

    async def test_empty_pattern_is_dropped_with_an_info_diagnostic(
        self, glob_repository, migrated_pool, vocabulary_config_factory, vocabulary_file
    ):
        """`//` is the one form whose text disappears -- with a notice."""
        from osprey.services.ariel_search.models import DiagnosticLevel
        from osprey.services.ariel_search.search.patterns import EMPTY_PATTERN_NOTICE

        config = vocabulary_config_factory(vocabulary_file)
        service, recorder = build_recording_service(config, migrated_pool)

        result = await search(service, "beam loss //")

        assert recorder.pattern_clause_count == 0
        notices = [
            d
            for d in result.diagnostics
            if d.category == "pattern" and d.level is DiagnosticLevel.INFO
        ]
        assert [d.message for d in notices] == [EMPTY_PATTERN_NOTICE]

    async def test_trailing_question_after_a_regex_span_keeps_the_body(
        self, glob_repository, migrated_pool, vocabulary_config_factory, vocabulary_file
    ):
        r"""`/…/?` yields exactly one predicate bound to the body without the `?`."""
        config = vocabulary_config_factory(vocabulary_file)
        service, recorder = build_recording_service(config, migrated_pool)

        result = await search(service, r"/SR0[1-4]C___BPM\d+/?")

        assert recorder.pattern_clause_count == 1
        assert r"SR0[1-4]C___BPM\d+" in recorder.bound_params
        assert GLOB_BPM3_ID in ids(result)

    async def test_internal_question_mark_survives_in_a_regex_body(
        self, glob_repository, migrated_pool, vocabulary_config_factory, vocabulary_file
    ):
        """`/BPMs?/` keeps its optional-`s` quantifier."""
        config = vocabulary_config_factory(vocabulary_file)
        service, recorder = build_recording_service(config, migrated_pool)

        await search(service, "/BPMs?/")

        assert recorder.pattern_clause_count == 1
        assert "BPMs?" in recorder.bound_params


class TestSlashFormCollision:
    """Facility shorthand containing `/` must not open a pattern span."""

    async def test_two_slash_forms_expand_and_open_no_span(
        self, prose_repository, migrated_pool, vocabulary_config_factory, vocabulary_file
    ):
        """`t/s i/o` is two vocabulary forms, not a regex."""
        config = vocabulary_config_factory(vocabulary_file)
        service, recorder = build_recording_service(config, migrated_pool)

        result = await search(service, "t/s i/o")

        assert recorder.pattern_clause_count == 0
        assert groups(result) == [
            ("t/s", ("troubleshoot",)),
            ("i/o", ("input output",)),
        ]

    async def test_slash_form_beside_a_real_regex_span(
        self, glob_repository, migrated_pool, vocabulary_config_factory, vocabulary_file
    ):
        r"""`t/s /…/` expands the form AND runs the span."""
        config = vocabulary_config_factory(vocabulary_file)
        service, recorder = build_recording_service(config, migrated_pool)

        result = await search(service, r"t/s /SR0[1-4]C___BPM\d+/")

        assert recorder.pattern_clause_count == 1
        assert r"SR0[1-4]C___BPM\d+" in recorder.bound_params
        assert groups(result) == [("t/s", ("troubleshoot",))]


class TestLiteralRunRule:
    """A pattern too generic for the trigram index is re-inserted as text."""

    async def test_short_body_is_searched_as_text(
        self, prose_repository, migrated_pool, vocabulary_config_factory, vocabulary_file
    ):
        """`/ab/` produces the INFO notice, no predicate, and an FTS leg for `ab`."""
        from osprey.services.ariel_search.models import DiagnosticLevel
        from osprey.services.ariel_search.search.patterns import TOO_GENERIC_REASON

        config = vocabulary_config_factory(vocabulary_file)
        service, recorder = build_recording_service(config, migrated_pool)

        result = await search(service, "/ab/")

        assert recorder.pattern_clause_count == 0
        # The criterion is the too-generic *INFO*: filtering on the level keeps
        # a promotion to WARNING/ERROR from passing silently.
        notices = [
            d
            for d in result.diagnostics
            if d.category == "pattern" and d.level is DiagnosticLevel.INFO
        ]
        assert len(notices) == 1
        assert TOO_GENERIC_REASON in notices[0].message
        assert "ab" in recorder.bound_params
        assert LITERAL_AB_ID in ids(result)

    @pytest.mark.parametrize(
        "query",
        ["test UNION SELECT * FROM users --", "5 * 3"],
    )
    async def test_bare_star_never_becomes_a_predicate(
        self, query, prose_repository, migrated_pool, vocabulary_config_factory, vocabulary_file
    ):
        """A lone `*` has no literal run at all, so it stays ordinary text."""
        config = vocabulary_config_factory(vocabulary_file)
        service, recorder = build_recording_service(config, migrated_pool)

        result = await search(service, query)

        assert recorder.pattern_clause_count == 0
        assert isinstance(result.entries, tuple)


# ---------------------------------------------------------------------------
# Fuzzy fallback
# ---------------------------------------------------------------------------


class TestFuzzyFallback:
    """The `pg_trgm` fallback, and the rule that expansion may only add hits."""

    async def test_short_entry_is_reached_only_with_expansion(
        self, prose_repository, migrated_pool, vocabulary_config_factory, vocabulary_file
    ):
        """A zero-FTS-row query reaches the short entry via the flattened probe.

        ``ts bpm offsett`` matches nothing in full text (the typo is not in any
        row) and its trigram similarity to the short entry is well under the
        threshold; the flattened text -- which carries both canonicals -- is
        over it.
        """
        config = vocabulary_config_factory(vocabulary_file)
        service = build_service(config, migrated_pool)

        with_expansion = await search(service, "ts bpm offsett")
        without = await search(service, "ts bpm offsett", expand_query=False)

        assert SHORT_NEEDLE_ID in ids(with_expansion)
        assert SHORT_NEEDLE_ID not in ids(without)

    async def test_expansion_never_removes_a_fuzzy_hit(
        self, prose_repository, migrated_pool, vocabulary_config_factory, vocabulary_file
    ):
        """The original probe runs first, so expansion is strictly additive."""
        config = vocabulary_config_factory(vocabulary_file)
        service = build_service(config, migrated_pool)

        query = "troubleshoott beam position monitor offsett"
        without = await search(service, query, expand_query=False)
        with_expansion = await search(service, query)

        assert set(ids(without))  # the unexpanded fuzzy path really does hit
        assert set(ids(without)) <= set(ids(with_expansion))


# ---------------------------------------------------------------------------
# Timeout envelope
# ---------------------------------------------------------------------------


class TestTimeoutMechanism:
    """`SET LOCAL statement_timeout` engages inside the transaction only."""

    async def test_timeout_is_set_inside_and_reverted_after(
        self, glob_repository, migrated_pool, integration_ariel_config
    ):
        """The configured value is live inside the block and gone afterwards.

        The pool is autocommit, so ``SET LOCAL`` is only meaningful because the
        repository opens an explicit transaction around the pattern statement.
        The recorder issues ``SHOW statement_timeout`` on the SAME connection
        right after ``set_config`` -- the only point at which the local value
        is observable.

        The "reverted after" half is a fresh checkout from the pool, and the
        backend PIDs are asserted equal: a ``SET LOCAL`` is only invisible from
        a *different* backend for trivial reasons, so without that pin a future
        pool change (a larger ``min_size``, a warm second connection) would turn
        the negative control into a vacuous pass.
        """
        from osprey.services.ariel_search.database.repository import ARIELRepository

        async with migrated_pool.connection() as conn:
            before_pid = conn.info.backend_pid
            before = str((await (await conn.execute("SHOW statement_timeout")).fetchone())[0])

        recorder = RecordingPool(migrated_pool)
        repository = ARIELRepository(recorder, integration_ariel_config)

        await repository.keyword_search(
            where_clauses=["raw_text ~* %s"],
            params=[r"\mSR01C___BPM\S*"],
            search_text="",
            max_results=5,
            pattern_timeout_seconds=2.5,
        )

        assert recorder.opened_transaction is True
        assert recorder.timeouts_inside == ["2500ms"]

        async with migrated_pool.connection() as conn:
            after_pid = conn.info.backend_pid
            after = str((await (await conn.execute("SHOW statement_timeout")).fetchone())[0])

        assert set(recorder.backend_pids) == {before_pid}
        assert after_pid == before_pid
        assert after == before
        assert after != "2500ms"

    async def test_pattern_free_search_opens_no_transaction(
        self, prose_repository, migrated_pool, vocabulary_config_factory, vocabulary_file
    ):
        """An ordinary keyword search runs exactly as it always has."""
        config = vocabulary_config_factory(vocabulary_file)
        service, recorder = build_recording_service(config, migrated_pool)

        await search(service, "ts bpm")

        assert recorder.opened_transaction is False
        assert recorder.timeouts_inside == []
        assert not any("set_config" in text for text in recorder.statements)


class TestTimeoutEffect:
    """Both directions of the envelope, on one 2000 x ~2 KB fixture."""

    #: The pattern the effect test runs. Its literal run (``SR01C___BPM``) is
    #: long enough to be accepted, and the bounded ``[a-z]{4,}`` tail makes the
    #: match work proportional to the row text.
    PATTERN = r"/SR01C___BPM[0-9]+ trip [a-z]{4,}/"

    async def test_default_timeout_completes(
        self, bulky_pattern_rows, migrated_pool, vocabulary_config_factory, vocabulary_file
    ):
        """With the shipped 10 s budget the pattern search returns rows."""
        config = vocabulary_config_factory(vocabulary_file)
        service = build_service(config, migrated_pool)

        result = await search(service, f"ts {self.PATTERN}")

        assert len(result.entries) >= 1
        assert "timeout" not in categories(result)
        assert groups(result) == [("ts", ("troubleshoot",))]

    async def test_one_millisecond_budget_reports_a_timeout_with_its_expansion(
        self, bulky_pattern_rows, migrated_pool, vocabulary_config_factory, vocabulary_file
    ):
        """The same search under a 1 ms budget degrades to a timeout diagnostic."""
        from osprey.services.ariel_search.models import DiagnosticLevel

        config = vocabulary_config_factory(
            vocabulary_file, keyword_settings={"pattern_timeout_seconds": 0.001}
        )
        service = build_service(config, migrated_pool)

        result = await search(service, f"ts {self.PATTERN}")

        assert result.entries == ()
        timeouts = [d for d in result.diagnostics if d.category == "timeout"]
        assert len(timeouts) == 1, [
            (d.category, d.level, d.message, d.source) for d in result.diagnostics
        ]
        assert timeouts[0].level is DiagnosticLevel.ERROR
        # The failed statement still reports what it was searching for, and the
        # fuzzy fallback never ran (it is skipped for pattern queries).
        assert groups(result) == [("ts", ("troubleshoot",))]

    async def test_invalid_regex_reports_the_pattern_and_the_expansion(
        self, bulky_pattern_rows, migrated_pool, vocabulary_config_factory, vocabulary_file
    ):
        """A body PostgreSQL refuses to compile names itself in the diagnostic.

        The service turns the repository's ``PatternError`` into an ERROR
        ``pattern`` diagnostic and returns no entries. Asserting that path
        directly -- rather than also tolerating a propagated exception -- is
        what makes a future flip back to propagation fail loudly instead of
        quietly unpinning the "names the pattern AND carries the expansion"
        contract.
        """
        from osprey.services.ariel_search.models import DiagnosticLevel

        config = vocabulary_config_factory(vocabulary_file)
        service = build_service(config, migrated_pool)

        result = await search(service, "t/s /SR0[1-4/")

        pattern_diagnostics = [d for d in result.diagnostics if d.category == "pattern"]
        assert len(pattern_diagnostics) == 1
        assert pattern_diagnostics[0].level is DiagnosticLevel.ERROR
        assert "/SR0[1-4/" in pattern_diagnostics[0].message
        assert result.entries == ()
        assert groups(result) == [("t/s", ("troubleshoot",))]

    async def test_bulky_fixture_seeded_every_row(self, bulky_pattern_rows, migrated_pool):
        """Every row the timeout tests rely on is really in the database.

        The fixture is memoized per package, so re-requesting it here is a
        ledger hit; the row count is what the two tests above depend on.
        """
        from tests.services.ariel_search.integration.conftest import BULK_PREFIX, BULK_ROW_COUNT

        async with migrated_pool.connection() as conn:
            cursor = await conn.execute(
                "SELECT count(*) FROM enhanced_entries WHERE entry_id LIKE %s",
                (f"{BULK_PREFIX}%",),
            )
            count = (await cursor.fetchone())[0]
        assert count == BULK_ROW_COUNT
        assert bulky_pattern_rows == BULK_ROW_COUNT


# ---------------------------------------------------------------------------
# Config-directory rule (crosses into the web layer)
# ---------------------------------------------------------------------------


_PROJECT_CONFIG = """
ariel:
  database:
    uri: {uri}
  search_modules:
    keyword:
      enabled: true
    semantic:
      enabled: false
  enhancement_modules:
    semantic_processor:
      enabled: false
    text_embedding:
      enabled: false
  vocabulary:
    enabled: true
    path: vocabulary.yml
"""


async def test_relative_vocabulary_path_resolves_against_the_config_file(
    prose_repository, database_url, tmp_path, monkeypatch
):
    """`CONFIG_FILE` outside the CWD still finds the vocabulary beside it.

    Goes through the real web lifespan -- the component that decides which
    ``config.yml`` was loaded and therefore which directory a relative
    ``ariel.vocabulary.path`` hangs off. The ASGI transport is driven directly
    (rather than through ``TestClient``) so the app runs on this test's event
    loop and can share the seeded database.
    """
    import httpx

    from osprey.interfaces.ariel.app import create_app
    from tests.services.ariel_search.integration.conftest import _BASE_VOCABULARY

    project = tmp_path / "project"
    project.mkdir()
    (project / "config.yml").write_text(_PROJECT_CONFIG.format(uri=database_url), encoding="utf-8")
    (project / "vocabulary.yml").write_text(_BASE_VOCABULARY, encoding="utf-8")

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    monkeypatch.setenv("CONFIG_FILE", str(project / "config.yml"))

    app = create_app()
    async with app.router.lifespan_context(app):
        assert app.state.config_path == project / "config.yml"
        assert app.state.config_errors == []
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://ariel.test") as client:
            capabilities = await client.get("/api/capabilities")
            response = await client.post(
                "/api/search",
                json={"query": "ts bpm", "mode": "keyword", "max_results": WIDE},
            )

    assert capabilities.status_code == 200
    assert capabilities.json()["vocabulary"]["concepts"] == 3

    assert response.status_code == 200
    payload = response.json()
    assert SHORT_NEEDLE_ID in [entry["entry_id"] for entry in payload["entries"]]
    assert [term["original"] for term in payload["expanded_terms"]] == ["ts", "bpm"]
