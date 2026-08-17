"""Tests for the ARIEL qmd search module.

Every test here runs against a stubbed client, so the suite needs no sidecar and
no network. What the stub cannot fake — the wire protocol — is covered by the
client's own tests; what it exists for is the mapping this module owns: qmd's
ranking and filenames onto ARIEL's rows.
"""

from __future__ import annotations

import asyncio
import re
import threading
import time
from datetime import UTC, datetime
from typing import Any

import pytest

from osprey.services.ariel_search.config import ARIELConfig, DatabaseConfig, SearchModuleConfig
from osprey.services.ariel_search.enhancement.qmd_export.writer import encode_entry_id
from osprey.services.ariel_search.search.qmd import (
    ARIEL_COLLECTION,
    DEFAULT_CANDIDATE_LIMIT,
    DEFAULT_RERANK,
    MAX_FETCH_LIMIT,
    OVERFETCH_FACTOR,
    HybridSearchSettings,
    format_qmd_result,
    get_parameter_descriptors,
    get_tool_descriptor,
    hybrid_search,
)
from osprey.services.qmd import QMDSearchResult, QMDUnavailableError

# --------------------------------------------------------------------------
# Doubles
# --------------------------------------------------------------------------


class StubClient:
    """A qmd client that answers from a canned hit list and records its calls."""

    def __init__(
        self,
        hits: list[QMDSearchResult] | None = None,
        *,
        available: bool = True,
        configured: bool = True,
        error: Exception | None = None,
    ) -> None:
        self._hits = hits or []
        self._available = available
        self.is_configured = configured
        self.base_url = "http://127.0.0.1:8180" if configured else None
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def is_available(self) -> bool:
        return self._available

    def query(self, collection: str | None, text: str, **kwargs: Any) -> list[QMDSearchResult]:
        self.calls.append({"collection": collection, "text": text, **kwargs})
        if self._error is not None:
            raise self._error
        return list(self._hits[: kwargs.get("limit", len(self._hits))])


class StubRepository:
    """A repository that hydrates from an in-memory row table."""

    def __init__(self, entries: list[dict[str, Any]]) -> None:
        self._by_id = {entry["entry_id"]: entry for entry in entries}
        self.requested: list[list[str]] = []

    async def get_entries_by_ids(self, entry_ids: list[str]) -> list[dict[str, Any]]:
        self.requested.append(list(entry_ids))
        # Deliberately returned in table order, not request order: the real
        # repository's ``= ANY(...)`` makes no ordering promise either.
        return [self._by_id[eid] for eid in sorted(self._by_id) if eid in set(entry_ids)]


def make_hit(
    entry_id: str,
    *,
    file: str | None = None,
    title: str | None = None,
    score: float = 1.0,
    snippet: str = "1: beam down",
    docid: str = "#abc123",
) -> QMDSearchResult:
    """Build one qmd hit for *entry_id*, as the daemon would report it.

    Both channels default to what the real pipeline produces: ``file`` to the
    mirror path the writer lays down (collection prefix already stripped by the
    client) and ``title`` to the ``# Entry <id>`` heading ``render_entry``
    writes. Override either to model a daemon that renamed the file — which is
    the defect these tests exist for — or a document that declares nothing.
    """
    return QMDSearchResult(
        docid=docid,
        file=mirror_file(entry_id) if file is None else file,
        collection=ARIEL_COLLECTION,
        title=f"Entry {entry_id}" if title is None else title,
        score=score,
        line=1,
        snippet=snippet,
    )


def make_entry(
    entry_id: str,
    *,
    author: str = "operator",
    source_system: str = "ALS eLog",
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Build one hydrated ``enhanced_entries`` row."""
    return {
        "entry_id": entry_id,
        "source_system": source_system,
        "timestamp": timestamp or datetime(2024, 6, 1, 12, 0, tzinfo=UTC),
        "author": author,
        "raw_text": f"text for {entry_id}",
        "attachments": [],
        "metadata": {},
        "created_at": datetime(2024, 6, 1, 12, 0, tzinfo=UTC),
        "updated_at": datetime(2024, 6, 1, 12, 0, tzinfo=UTC),
    }


def mirror_file(entry_id: str, *, shard: str = "2024/06") -> str:
    """Render the collection-relative path the mirror writer would produce."""
    return f"{shard}/{encode_entry_id(entry_id)}.md"


def make_config(settings: dict[str, Any] | None = None) -> ARIELConfig:
    """Build an ARIELConfig whose ``hybrid`` module carries *settings*."""
    config = ARIELConfig(database=DatabaseConfig(uri="postgresql://localhost/ariel"))
    config.search_modules["hybrid"] = SearchModuleConfig(enabled=True, settings=settings or {})
    return config


# --------------------------------------------------------------------------
# Descriptor
# --------------------------------------------------------------------------


class TestDescriptor:
    """The descriptor the agent executor auto-discovers."""

    def test_search_mode_is_the_plain_string_hybrid(self):
        assert get_tool_descriptor().search_mode == "hybrid"

    def test_descriptor_fields(self):
        descriptor = get_tool_descriptor()
        assert descriptor.name == "hybrid_search"
        assert descriptor.execute is hybrid_search
        assert descriptor.format_result is format_qmd_result
        assert descriptor.needs_embedder is False
        assert "query" in descriptor.args_schema.model_fields

    def test_parameter_descriptors_expose_both_knobs(self):
        by_name = {p.name: p for p in get_parameter_descriptors()}
        assert by_name["rerank"].default is DEFAULT_RERANK
        assert by_name["candidate_limit"].default == DEFAULT_CANDIDATE_LIMIT

    def test_registered_in_builtins(self):
        from osprey.registry.builtins import FrameworkRegistryProvider

        config = FrameworkRegistryProvider().get_registry_config()
        registrations = {r.name: r for r in config.ariel_search_modules}
        assert registrations["hybrid"].module_path == "osprey.services.ariel_search.search.qmd"

    def test_format_result_carries_score_and_highlights(self):
        formatted = format_qmd_result(make_entry("42"), 0.5, ["1: beam down"])
        assert formatted["score"] == 0.5
        assert formatted["highlights"] == ["1: beam down"]
        assert formatted["entry_id"] == "42"


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


class TestSettings:
    """``search_modules.hybrid.settings`` resolution."""

    def test_defaults_when_unconfigured(self):
        settings = HybridSearchSettings.from_ariel_config(make_config())
        assert settings.rerank is DEFAULT_RERANK is True
        assert settings.candidate_limit == DEFAULT_CANDIDATE_LIMIT
        assert settings.collection == ARIEL_COLLECTION

    def test_defaults_when_module_absent_entirely(self):
        bare = ARIELConfig(database=DatabaseConfig(uri="postgresql://localhost/ariel"))
        assert HybridSearchSettings.from_ariel_config(bare).rerank is True

    def test_config_turns_reranking_off(self):
        settings = HybridSearchSettings.from_ariel_config(make_config({"rerank": False}))
        assert settings.rerank is False

    def test_config_sets_candidate_limit(self):
        settings = HybridSearchSettings.from_ariel_config(make_config({"candidate_limit": 12}))
        assert settings.candidate_limit == 12

    def test_candidate_limit_none_defers_to_qmd(self):
        settings = HybridSearchSettings.from_ariel_config(make_config({"candidate_limit": None}))
        assert settings.candidate_limit is None

    @pytest.mark.parametrize("bad", ["false", 0, None])
    def test_malformed_rerank_is_refused(self, bad):
        with pytest.raises(ValueError, match="search_modules.hybrid.settings.rerank"):
            HybridSearchSettings.from_ariel_config(make_config({"rerank": bad}))

    @pytest.mark.parametrize("bad", [0, -1, True, "40"])
    def test_malformed_candidate_limit_is_refused(self, bad):
        with pytest.raises(ValueError, match="search_modules.hybrid.settings.candidate_limit"):
            HybridSearchSettings.from_ariel_config(make_config({"candidate_limit": bad}))


# --------------------------------------------------------------------------
# Query knobs reaching the client
# --------------------------------------------------------------------------


class TestQueryKnobs:
    """What this module hands the client."""

    @pytest.mark.asyncio
    async def test_config_rerank_is_forwarded(self):
        """The assertion documents the mode it measures: config says False."""
        client = StubClient([])
        await hybrid_search(
            "beam",
            StubRepository([]),
            make_config({"rerank": False, "candidate_limit": 12}),
            client=client,
        )
        assert client.calls[0]["rerank"] is False
        assert client.calls[0]["candidate_limit"] == 12

    @pytest.mark.asyncio
    async def test_default_is_reranked(self):
        """Nothing configured: the agent-facing tool keeps qmd's quality path."""
        client = StubClient([])
        await hybrid_search("beam", StubRepository([]), make_config(), client=client)
        assert client.calls[0]["rerank"] is True
        assert client.calls[0]["candidate_limit"] == DEFAULT_CANDIDATE_LIMIT

    @pytest.mark.asyncio
    async def test_per_query_override_beats_config(self):
        client = StubClient([])
        await hybrid_search(
            "beam",
            StubRepository([]),
            make_config({"rerank": True}),
            client=client,
            rerank=False,
            candidate_limit=5,
        )
        assert client.calls[0]["rerank"] is False
        assert client.calls[0]["candidate_limit"] == 5

    @pytest.mark.asyncio
    async def test_query_is_scoped_to_the_ariel_collection(self):
        client = StubClient([])
        await hybrid_search("beam", StubRepository([]), make_config(), client=client)
        assert client.calls[0]["collection"] == ARIEL_COLLECTION

    @pytest.mark.asyncio
    async def test_blank_query_never_reaches_the_sidecar(self):
        client = StubClient([])
        assert await hybrid_search("   ", StubRepository([]), make_config(), client=client) == []
        assert client.calls == []


# --------------------------------------------------------------------------
# Identity recovery
# --------------------------------------------------------------------------


def slugify(filename: str) -> str:
    """Reproduce qmd's rewriting of a filename into the name it reports.

    Measured against a real daemon by the container lane, not documented by
    qmd: ``%`` and ``_`` both become ``-``, runs of ``-`` collapse to one, and a
    leading ``-`` is dropped. Reproduced here so the unit suite can model the
    daemon's actual behaviour instead of an idealised one.
    """
    return re.sub(r"-+", "-", re.sub(r"[%_]", "-", filename)).lstrip("-")


def mangled_hit(entry_id: str, **kwargs: Any) -> QMDSearchResult:
    """A hit as a *real* daemon reports it: title intact, path slugified."""
    return make_hit(entry_id, file=slugify(mirror_file(entry_id)), **kwargs)


class TestIdentityRecovery:
    """The document names itself; the reported path does not.

    qmd slugifies the filenames it reports, so inverting a path is lossy — and
    for an ``_``/``-`` pair it is worse than lossy, resolving to a different
    real entry. Hydration therefore reads the identifier out of the hit's
    title, which is the heading the mirror writer put in the document.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "entry_id",
        [
            "12345",
            "a/b",
            "..",
            "ALS-2024",
            "entry with spaces",
            "übung",
            ".hidden",
            "%2F",
            "beam_current_setpoint",
            "ALS-2003-0001",
        ],
    )
    async def test_hostile_ids_survive_a_slugifying_daemon(self, entry_id):
        """Every one of these has a mangled path; all must still hydrate."""
        client = StubClient([mangled_hit(entry_id)])
        repo = StubRepository([make_entry(entry_id)])

        results = await hybrid_search("beam", repo, make_config(), client=client)

        assert repo.requested == [[entry_id]]
        assert [entry["entry_id"] for entry, _, _ in results] == [entry_id]

    @pytest.mark.asyncio
    async def test_underscore_and_hyphen_ids_do_not_cross_hydrate(self):
        """The collision case: two real entries whose paths slugify alike.

        ``beam_current_setpoint`` and ``beam-current-setpoint`` are reported
        under one path, so inverting it would serve one of them the other's
        row. Each must hydrate to its own.
        """
        underscore, hyphen = "beam_current_setpoint", "beam-current-setpoint"
        assert slugify(mirror_file(underscore)) == slugify(mirror_file(hyphen)), (
            "the fixture no longer models a collision"
        )

        client = StubClient([mangled_hit(underscore), mangled_hit(hyphen, score=0.5)])
        repo = StubRepository([make_entry(underscore), make_entry(hyphen)])

        results = await hybrid_search("beam", repo, make_config(), client=client)

        assert [entry["entry_id"] for entry, _, _ in results] == [underscore, hyphen]

    @pytest.mark.asyncio
    async def test_encoded_looking_id_is_not_double_decoded(self):
        """The title carries the raw id, so it must never be unquoted.

        ``a%2Fb`` is a legal identifier whose *encoding* is ``a%252Fb``. Reading
        the title as if it were encoded would turn it into ``a/b`` — a different
        entry, silently.
        """
        client = StubClient([mangled_hit("a%2Fb")])
        repo = StubRepository([make_entry("a%2Fb"), make_entry("a/b")])

        results = await hybrid_search("beam", repo, make_config(), client=client)

        assert [entry["entry_id"] for entry, _, _ in results] == ["a%2Fb"]

    @pytest.mark.asyncio
    async def test_path_is_ignored_however_the_daemon_spells_it(self):
        """Prefix, no prefix, bare name, wrong suffix — the title decides."""
        entry_id = "77"
        name = encode_entry_id(entry_id) + ".md"
        client = StubClient(
            [
                make_hit(entry_id, file=f"{ARIEL_COLLECTION}/2024/06/{name}", docid="#1"),
                make_hit(entry_id, file=name, score=0.5, docid="#2"),
                make_hit(entry_id, file="2024/06/notes.txt", score=0.33, docid="#3"),
                make_hit(entry_id, file="", score=0.25, docid="#4"),
            ]
        )
        repo = StubRepository([make_entry(entry_id)])

        results = await hybrid_search("beam", repo, make_config(), client=client)

        # All four name the same entry, so it is requested and returned once.
        assert repo.requested == [[entry_id]]
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_a_hit_declaring_no_entry_is_dropped_not_guessed(self):
        """Never fall back to the path — that is the channel known to be wrong."""
        good = "88"
        client = StubClient(
            [
                make_hit("ignored", title="Some Other Document", file=mirror_file("99")),
                make_hit("ignored", title="", file=mirror_file("99")),
                make_hit("ignored", title="Entry   ", file=mirror_file("99")),
                make_hit(good, score=0.5),
            ]
        )
        repo = StubRepository([make_entry(good), make_entry("99")])

        results = await hybrid_search("beam", repo, make_config(), client=client)

        # 99 has a row and is what every dropped hit's path pointed at, so it
        # would have appeared had the path been used as a fallback.
        assert repo.requested == [[good]]
        assert [entry["entry_id"] for entry, _, _ in results] == [good]

    @pytest.mark.asyncio
    async def test_path_disagreement_is_logged(self, caplog):
        """A silent corruption becomes a discoverable one."""
        with caplog.at_level("WARNING", logger="ariel"):
            await hybrid_search(
                "beam",
                StubRepository([make_entry("beam_current_setpoint")]),
                make_config(),
                client=StubClient([mangled_hit("beam_current_setpoint")]),
            )

        assert any("disagrees with the document's own title" in r.message for r in caplog.records)

    def test_title_prefix_matches_what_the_writer_actually_emits(self):
        """Pin the coupling to ``render_entry``'s heading.

        Hydration depends on a string the export pipeline chooses. The two live
        in different modules and cannot share a constant without the search
        module reaching into the writer's vocabulary, so the drift is caught
        here instead: render a row and read the heading back.
        """
        from osprey.services.ariel_search.enhancement.qmd_export.writer import render_entry
        from osprey.services.ariel_search.search.qmd import TITLE_ENTRY_PREFIX

        heading = render_entry(make_entry("ALS-2003-0001")).splitlines()[0]

        assert heading == f"# {TITLE_ENTRY_PREFIX}ALS-2003-0001"

    @pytest.mark.asyncio
    async def test_hit_without_a_row_is_dropped(self):
        """The mirror is a copy of the database and may lag a deletion."""
        client = StubClient([make_hit("gone"), make_hit("here", score=0.5)])
        repo = StubRepository([make_entry("here")])

        results = await hybrid_search("beam", repo, make_config(), client=client)

        assert [entry["entry_id"] for entry, _, _ in results] == ["here"]


# --------------------------------------------------------------------------
# Ranking, hydration and payload
# --------------------------------------------------------------------------


class TestRankingAndHydration:
    """qmd ranks, Postgres answers."""

    @pytest.mark.asyncio
    async def test_results_follow_qmd_rank_not_row_order(self):
        """The repository answers in its own order; the ranking must survive."""
        client = StubClient(
            [
                make_hit("c", score=1.0),
                make_hit("a", score=0.5),
                make_hit("b", score=0.33),
            ]
        )
        repo = StubRepository([make_entry("a"), make_entry("b"), make_entry("c")])

        results = await hybrid_search("beam", repo, make_config(), client=client)

        assert [entry["entry_id"] for entry, _, _ in results] == ["c", "a", "b"]
        assert [score for _, score, _ in results] == [1.0, 0.5, 0.33]

    @pytest.mark.asyncio
    async def test_snippet_is_carried_as_the_highlight(self):
        client = StubClient([make_hit("1", snippet="7: water leak")])
        repo = StubRepository([make_entry("1")])

        _, _, highlights = (await hybrid_search("beam", repo, make_config(), client=client))[0]

        assert highlights == ["7: water leak"]

    @pytest.mark.asyncio
    async def test_missing_snippet_yields_no_highlights(self):
        client = StubClient([make_hit("1", snippet="")])
        repo = StubRepository([make_entry("1")])

        _, _, highlights = (await hybrid_search("beam", repo, make_config(), client=client))[0]

        assert highlights == []

    @pytest.mark.asyncio
    async def test_entry_body_comes_from_postgres_not_the_index(self):
        client = StubClient([make_hit("1")])
        repo = StubRepository([make_entry("1", author="Chen")])

        entry, _, _ = (await hybrid_search("beam", repo, make_config(), client=client))[0]

        assert entry["author"] == "Chen"
        assert entry["raw_text"] == "text for 1"

    @pytest.mark.asyncio
    async def test_max_results_caps_the_returned_set(self):
        hits = [make_hit(str(i), score=1.0 / (i + 1)) for i in range(6)]
        repo = StubRepository([make_entry(str(i)) for i in range(6)])

        results = await hybrid_search(
            "beam", repo, make_config(), client=StubClient(hits), max_results=2
        )

        assert len(results) == 2


# --------------------------------------------------------------------------
# Post-filters and over-fetch
# --------------------------------------------------------------------------


class TestFiltersAndOverfetch:
    """Filters run after hydration, so the fetch must leave room for them."""

    @pytest.mark.asyncio
    async def test_unfiltered_query_fetches_exactly_max_results(self):
        client = StubClient([])
        await hybrid_search(
            "beam", StubRepository([]), make_config(), client=client, max_results=10
        )
        assert client.calls[0]["limit"] == 10

    @pytest.mark.asyncio
    async def test_filtered_query_overfetches(self):
        client = StubClient([])
        await hybrid_search(
            "beam",
            StubRepository([]),
            make_config(),
            client=client,
            max_results=10,
            author="chen",
        )
        assert client.calls[0]["limit"] == 10 * OVERFETCH_FACTOR

    @pytest.mark.asyncio
    async def test_overfetch_is_capped(self):
        client = StubClient([])
        await hybrid_search(
            "beam",
            StubRepository([]),
            make_config(),
            client=client,
            max_results=50,
            source_system="ALS eLog",
        )
        assert client.calls[0]["limit"] == MAX_FETCH_LIMIT

    @pytest.mark.asyncio
    async def test_overfetch_survives_filtering_to_a_full_page(self):
        """Two of every three hits are rejected; the page must still fill."""
        hits = [make_hit(str(i), score=1.0 / (i + 1)) for i in range(12)]
        entries = [make_entry(str(i), author="Chen" if i % 3 == 0 else "Other") for i in range(12)]

        results = await hybrid_search(
            "beam",
            StubRepository(entries),
            make_config(),
            client=StubClient(hits),
            max_results=3,
            author="chen",
        )

        assert [entry["entry_id"] for entry, _, _ in results] == ["0", "3", "6"]

    @pytest.mark.asyncio
    async def test_author_filter_is_case_insensitive_substring(self):
        hits = [make_hit("1"), make_hit("2", score=0.5)]
        entries = [make_entry("1", author="Wei Chen"), make_entry("2", author="Ada Lovelace")]

        results = await hybrid_search(
            "beam", StubRepository(entries), make_config(), client=StubClient(hits), author="chen"
        )

        assert [entry["entry_id"] for entry, _, _ in results] == ["1"]

    @pytest.mark.asyncio
    async def test_source_system_filter_is_exact(self):
        hits = [make_hit("1"), make_hit("2", score=0.5)]
        entries = [
            make_entry("1", source_system="ALS eLog"),
            make_entry("2", source_system="JLab Logbook"),
        ]

        results = await hybrid_search(
            "beam",
            StubRepository(entries),
            make_config(),
            client=StubClient(hits),
            source_system="JLab Logbook",
        )

        assert [entry["entry_id"] for entry, _, _ in results] == ["2"]

    @pytest.mark.asyncio
    async def test_date_range_filters_inclusively(self):
        hits = [make_hit(str(i), score=1.0 / (i + 1)) for i in range(3)]
        entries = [
            make_entry("0", timestamp=datetime(2024, 1, 1, tzinfo=UTC)),
            make_entry("1", timestamp=datetime(2024, 6, 1, tzinfo=UTC)),
            make_entry("2", timestamp=datetime(2024, 12, 1, tzinfo=UTC)),
        ]

        results = await hybrid_search(
            "beam",
            StubRepository(entries),
            make_config(),
            client=StubClient(hits),
            start_date=datetime(2024, 6, 1, tzinfo=UTC),
            end_date=datetime(2024, 12, 1, tzinfo=UTC),
        )

        assert [entry["entry_id"] for entry, _, _ in results] == ["1", "2"]

    @pytest.mark.asyncio
    async def test_naive_bound_compares_against_aware_timestamp(self):
        """A request carrying a naive datetime must not raise on comparison."""
        hits = [make_hit("1")]
        entries = [make_entry("1", timestamp=datetime(2024, 6, 1, tzinfo=UTC))]

        results = await hybrid_search(
            "beam",
            StubRepository(entries),
            make_config(),
            client=StubClient(hits),
            start_date=datetime(2024, 1, 1),
        )

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_entry_without_a_timestamp_fails_a_date_filter(self):
        hits = [make_hit("1")]
        entries = [make_entry("1")]
        entries[0]["timestamp"] = None

        results = await hybrid_search(
            "beam",
            StubRepository(entries),
            make_config(),
            client=StubClient(hits),
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
        )

        assert results == []


# --------------------------------------------------------------------------
# Event-loop discipline
# --------------------------------------------------------------------------


class BlockingClient:
    """A client whose every call blocks the thread it runs on.

    Stands in for a sidecar that accepts the TCP connection and then does not
    answer while it finishes its startup index pass — the state the real
    client pays its full 30 s timeout for.
    """

    is_configured = True
    base_url = "http://127.0.0.1:8180"

    def __init__(self, probe_seconds: float = 0.2, query_seconds: float = 0.2) -> None:
        self._probe = probe_seconds
        self._query = query_seconds
        self.threads: dict[str, str] = {}

    def is_available(self) -> bool:
        self.threads["is_available"] = threading.current_thread().name
        time.sleep(self._probe)
        return True

    def query(self, collection: str | None, text: str, **kwargs: Any) -> list[QMDSearchResult]:
        self.threads["query"] = threading.current_thread().name
        time.sleep(self._query)
        return []


class TestEventLoopIsNotBlocked:
    """Every blocking call this module makes belongs off the event loop.

    The health probe is as much I/O as the query: it is a synchronous
    ``GET /health`` on the client's 30 s timeout, cached for only 5 s, and a
    sidecar that accepts a connection without answering is an ordinary state
    rather than an exotic one. A probe left on the loop stalls every other
    request in the process — including the keyword and semantic searches this
    module's own unavailable-message tells the caller to fall back to.
    """

    @pytest.mark.asyncio
    async def test_no_blocking_call_runs_on_the_loop_thread(self):
        client = BlockingClient()
        loop_thread = threading.current_thread().name

        await hybrid_search("beam", StubRepository([]), make_config(), client=client)

        assert client.threads["is_available"] != loop_thread
        assert client.threads["query"] != loop_thread

    @pytest.mark.asyncio
    async def test_other_coroutines_keep_running_during_a_slow_probe(self):
        """The thread assertion says where the work went; this says it mattered.

        Only the *probe* blocks here — the query returns at once — so the
        counter is measuring the probe alone. Ticking every 10 ms across a
        200 ms probe it reaches the tens when the loop is free and 0-1 when it
        is not, which separates the two outcomes by a wide margin without
        asserting on a wall-clock duration. Letting the query block too would
        hide the regression: its 200 ms off-loop would feed the counter even
        with the probe back on the loop.
        """
        ticks = 0

        async def counter() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        client = BlockingClient(probe_seconds=0.2, query_seconds=0.0)
        ticker = asyncio.create_task(counter())
        try:
            await hybrid_search("beam", StubRepository([]), make_config(), client=client)
        finally:
            ticker.cancel()

        assert ticks >= 5, f"event loop was starved during the health probe (only {ticks} ticks)"


# --------------------------------------------------------------------------
# Sidecar faults
# --------------------------------------------------------------------------


class TestSidecarFaults:
    """ "Search is down" and "nothing matched" must not look alike."""

    @pytest.mark.asyncio
    async def test_unconfigured_sidecar_raises_rather_than_returning_empty(self):
        client = StubClient(available=False, configured=False)

        with pytest.raises(QMDUnavailableError, match="no qmd sidecar is configured"):
            await hybrid_search("beam", StubRepository([]), make_config(), client=client)

    @pytest.mark.asyncio
    async def test_configured_but_down_sidecar_names_its_endpoint(self):
        client = StubClient(available=False, configured=True)

        with pytest.raises(QMDUnavailableError, match="127.0.0.1:8180"):
            await hybrid_search("beam", StubRepository([]), make_config(), client=client)

    @pytest.mark.asyncio
    async def test_a_down_sidecar_is_never_queried(self):
        client = StubClient(available=False)

        with pytest.raises(QMDUnavailableError):
            await hybrid_search("beam", StubRepository([]), make_config(), client=client)
        assert client.calls == []

    @pytest.mark.asyncio
    async def test_query_failure_propagates(self):
        from osprey.services.qmd import QMDClientError

        client = StubClient(error=QMDClientError("qmd tool 'query' failed: index missing"))

        with pytest.raises(QMDClientError, match="index missing"):
            await hybrid_search("beam", StubRepository([]), make_config(), client=client)

    @pytest.mark.asyncio
    async def test_empty_result_set_is_not_an_error(self):
        repo = StubRepository([make_entry("1")])

        results = await hybrid_search("beam", repo, make_config(), client=StubClient([]))

        assert results == []
        # Nothing to hydrate, so Postgres is not touched at all.
        assert repo.requested == []


# --------------------------------------------------------------------------
# Service integration
# --------------------------------------------------------------------------


class TestServiceDispatch:
    """The shape the service's ``_run_module`` unpacks."""

    @pytest.mark.asyncio
    async def test_results_unpack_as_entry_score_extra(self):
        client = StubClient([make_hit("1")])
        repo = StubRepository([make_entry("1")])

        results = await hybrid_search("beam", repo, make_config(), client=client)

        for entry, score, *extra in results:
            assert entry["entry_id"] == "1"
            assert isinstance(score, float)
            assert extra and isinstance(extra[0], list)

    @pytest.mark.asyncio
    async def test_unknown_kwargs_are_tolerated(self):
        """The service forwards a request's advanced params verbatim."""
        client = StubClient([])

        await hybrid_search(
            "beam",
            StubRepository([]),
            make_config(),
            client=client,
            similarity_threshold=0.5,
            include_highlights=True,
        )

        assert client.calls
